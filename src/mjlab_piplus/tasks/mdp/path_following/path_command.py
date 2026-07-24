from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.entity.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer

from mjlab_piplus.tasks.mdp.path_following.path_gen import generate_parametric_paths
from mjlab_piplus.tasks.mdp.path_following.path_gen_astar import (
  OBSTACLE_KIND_CYLINDER,
  ObstacleWorldCfg,
  generate_astar_paths,
)
from mjlab_piplus.tasks.mdp.path_following.utils import (
  angle_distance,
  heading_from_quat,
  inv_mat,
  xy_yaw_to_mat,
)

# Dark orange, darkened a little, at half opacity: the obstacles are virtual, so
# they should read as markers rather than as solid scenery.
_OBSTACLE_COLOR = (0.65, 0.35, 0.02, 0.5)


def _add_box(
  visualizer: DebugVisualizer,
  x: float,
  y: float,
  yaw: float,
  half_x: float,
  half_y: float,
  height: float,
) -> None:
  """Draw a solid box standing on the ground, rotated about z by ``yaw``.

  ``add_box`` takes the box center and its half extents, so the z terms carry
  the extra factor of a half.
  """
  cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
  visualizer.add_box(
    np.array([x, y, 0.5 * height]),
    np.array([half_x, half_y, 0.5 * height]),
    np.array([[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]),
    _OBSTACLE_COLOR,
  )


class PathCommand(CommandTerm):
  # Eval-mode log filenames. Subclasses evaluating a *different* policy on the
  # same paths (e.g. ``VelocityPathCommand``) override these so their metrics
  # land in separate files; the path library and iteration order are shared, so
  # both policies are compared on identical paths.
  _commands_csv = "commands_path.csv"
  _log_prefix = "path_logs"

  def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.align = False
    self.global_command_counter = 0
    if cfg.use_pathlib:
      (
        self.all_paths,
        self.all_waypoints,
        self.all_obstacles,
      ) = self._load_or_generate_pathlib(cfg)
      # randomly select a path for each environment
      indices = torch.randint(0, cfg.pathlib_size, (self.num_envs,), device=self.device)
      self.paths = self.all_paths[indices]
      self.waypoints = self.all_waypoints[indices]
      self.obstacles = (
        self.all_obstacles[indices] if self.all_obstacles is not None else None
      )
    else:
      self.paths, self.waypoints, self.obstacles = self._generate(self.num_envs)
    # Command obs = N future path points [x, y] in the base frame, followed by
    # the goal orientation encoded as (cos(yaw_err), sin(yaw_err)).
    self.command_b = torch.zeros(
      self.num_envs, cfg.num_observations * 2 + 2, device=self.device
    )
    # Goal orientation (world yaw), sampled per path, plus derived per-step state.
    self.goal_yaw_w = torch.zeros(self.num_envs, device=self.device)
    self.goal_yaw_w.uniform_(*cfg.goal_yaw_range)
    self.dist_to_goal = torch.zeros(self.num_envs, device=self.device)
    self.yaw_err = torch.zeros(self.num_envs, device=self.device)
    self.near_goal = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.at_goal = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    # Seconds the robot has continuously held the full goal pose (position +
    # orientation); drives the delayed stand trigger (see ``stand_delay``).
    self.at_goal_time = torch.zeros(self.num_envs, device=self.device)
    # Twist command whose stand flag we drive near the goal (resolved lazily).
    self._twist_term = None
    # Resolve the foot sites once so the eval "goal reached" check can measure
    # goal distance at the base footprint (feet midpoint), exactly as the
    # training goal-reward does (see ``path_reward.base_footprint_xy``).
    foot_cfg = SceneEntityCfg(cfg.entity_name, site_names=cfg.foot_site_names)
    foot_cfg.resolve(env.scene)
    self._foot_site_ids = foot_cfg.site_ids
    self._update_robot_pose()

  def _generate(
    self, num_paths: int, progress: bool = False
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Generate ``num_paths`` paths with the configured generator.

    Returns ``(paths, waypoints, obstacles)``. ``obstacles`` is None for the
    parametric generator, which draws its paths through empty space.
    """
    cfg = self.cfg
    if cfg.generator == "astar":
      return generate_astar_paths(
        num_paths=num_paths,
        cfg=cfg.obstacle_world,
        point_spacing=cfg.point_spacing,
        num_waypoints=cfg.num_waypoints,
        max_points=cfg.max_points,
        seed=cfg.generation_seed,
        device=self.device,
        progress=progress,
        num_workers=cfg.num_workers,
      )
    paths, waypoints = generate_parametric_paths(
      num_paths=num_paths,
      max_point_dxy=cfg.max_point_dxy,
      num_waypoints=cfg.num_waypoints,
      curvature_limit=cfg.curvature_limit,
      point_spacing=cfg.point_spacing,
      max_points=cfg.max_points,
      device=self.device,
      progress=progress,
    )
    return paths, waypoints, None

  def _load_or_generate_pathlib(
    self, cfg: PathCommandCfg
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Load the pre-generated path library from disk, or generate and cache it.

    Both generators are deterministically seeded, so the library is fully
    determined by the generation parameters. We key the cache file on those
    params; a cache hit skips regenerating all ``pathlib_size`` paths on every
    training run.
    """
    cache_dir = Path(cfg.cache_dir or (Path(__file__).parent / "path_cache"))
    if cfg.generator == "astar":
      # The obstacle world has far too many knobs to spell out in a filename;
      # hash them instead.
      key = hashlib.sha1(
        repr(
          (
            cfg.obstacle_world.cache_key(),
            cfg.point_spacing,
            cfg.num_waypoints,
            cfg.max_points,
            cfg.generation_seed,
          )
        ).encode()
      ).hexdigest()[:12]
      fname = f"astar_pathlib_n{cfg.pathlib_size}_{key}.pt"
    else:
      fname = (
        f"pathlib_n{cfg.pathlib_size}_wp{cfg.num_waypoints}"
        f"_dxy{cfg.max_point_dxy}_curv{cfg.curvature_limit}"
        f"_sp{cfg.point_spacing}_mp{cfg.max_points}.pt"
      )
    cache_path = cache_dir / fname

    if cache_path.exists():
      print(f"Loading cached path library from {cache_path}")
      data = torch.load(cache_path, map_location=self.device)
      return data["paths"], data["waypoints"], data.get("obstacles")

    all_paths, all_waypoints, all_obstacles = self._generate(
      cfg.pathlib_size, progress=True
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {"paths": all_paths.cpu(), "waypoints": all_waypoints.cpu()}
    if all_obstacles is not None:
      payload["obstacles"] = all_obstacles.cpu()
    torch.save(payload, cache_path)
    print(f"Cached path library to {cache_path}")
    return all_paths, all_waypoints, all_obstacles

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  def _resample_command(self, env_ids):
    # In eval mode, tell apart a resample caused by the command *timeout* from one
    # caused by an env reset: CommandTerm.reset() zeroes command_counter *before*
    # resampling, whereas the periodic (timeout) path in compute() does not. So a
    # nonzero command_counter here means the 30 s command timeout fired without the
    # robot reaching the goal — reset the env so it restarts the next path from the
    # origin instead of continuing from wherever it stalled. That env reset
    # re-enters this method with command_counter zeroed and does the path selection
    # + global_command_counter increment, so we return early here to avoid
    # advancing the counter twice (which would skip a path). Goal-reached /
    # fell-over / startup resets all arrive here with command_counter already 0 and
    # fall through to the normal body.
    if self.cfg.eval_mode and bool((self.command_counter[env_ids] != 0).any()):
      self._env.reset()
      return
    self.starting_step = self._env.common_step_counter
    self._update_robot_pose()
    # New goal: reset the dwell timer so accumulated time doesn't carry over.
    self.at_goal_time[env_ids] = 0.0
    # Goal orientation (world yaw) for the new paths. In eval mode make it a
    # deterministic function of the path index (which is iterated deterministically
    # too), so a path-policy run and a velocity-policy run see the *same* goal
    # pose for each path and the comparison stays fair. Otherwise sample randomly.
    if self.cfg.eval_mode:
      idx = self.global_command_counter % self.cfg.pathlib_size
      lo, hi = self.cfg.goal_yaw_range
      goal_yaw = float(
        np.random.default_rng(self.cfg.goal_yaw_seed + idx).uniform(lo, hi)
      )
      self.goal_yaw_w[env_ids] = goal_yaw
    else:
      self.goal_yaw_w[env_ids] = torch.empty(
        len(env_ids), device=self.device
      ).uniform_(*self.cfg.goal_yaw_range)
    if self.cfg.use_pathlib:
      if self.cfg.eval_mode:
        indices = torch.tensor(
          [self.global_command_counter % self.cfg.pathlib_size], device=self.device
        ).repeat(len(env_ids))
      else:
        indices = torch.randint(
          0, self.cfg.pathlib_size, (len(env_ids),), device=self.device
        )
      paths = self.all_paths[indices]
      waypoints = self.all_waypoints[indices]
      obstacles = (
        self.all_obstacles[indices] if self.all_obstacles is not None else None
      )
    else:
      paths, waypoints, obstacles = self._generate(len(env_ids))

    self.paths[env_ids] = paths
    self.waypoints[env_ids] = waypoints
    if obstacles is not None:
      assert self.obstacles is not None
      self.obstacles[env_ids] = obstacles
    if self.align:
      # Paths are generated in their start frame; shift them — and the obstacle
      # world around them — onto the robot's current position.
      offset = self.current_pose[env_ids, :2].unsqueeze(1)
      self.paths[env_ids, :, :2] += offset
      self.waypoints[env_ids] += offset
      if obstacles is not None:
        assert self.obstacles is not None
        # Columns 1:3 are the obstacle center. Padding slots are shifted too,
        # which is harmless: the kind check skips them at draw time.
        self.obstacles[env_ids, :, 1:3] += offset

    if self.cfg.eval_mode:
      # write waypoints to csv
      with open(self._commands_csv, "a") as f:
        for i in range(self.waypoints.shape[0]):
          waypoint_str = ",".join(
            [f"{x:.2f}" for x in self.waypoints[i].cpu().numpy().flatten()]
          )
          f.write(f"{waypoint_str},")
        f.write(f"{self.global_command_counter}\n")
    self.global_command_counter += 1

  def _set_twist_stand(self, stand: torch.Tensor) -> None:
    """Drive the twist command's stand flag from a boolean mask (near-goal).

    The twist command runs before this one each step, so overwriting its
    stand_flag here takes effect for the current step's observation and rewards.
    """
    if self._twist_term is None:
      self._twist_term = self._env.command_manager.get_term(
        self.cfg.twist_command_name
      )
    twist = self._twist_term
    if twist is not None and hasattr(twist, "stand_flag"):
      twist.stand_flag[:, 0] = stand.float()

  def _update_metrics(self):
    pass

  def _update_robot_pose(self) -> None:
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    heading = heading_from_quat(root_quat).unsqueeze(-1)
    self.current_pose = torch.cat([root_pos[..., :2], heading], dim=-1)

  def _update_command(self) -> None:
    # get closest point on path to robot
    self._update_robot_pose()
    root_pos = self.current_pose[..., :2]  # (n_envs, 2)
    dists = torch.norm(
      self.paths[..., :2] - root_pos.unsqueeze(1), dim=-1
    )  # (num_envs, resolution)
    closest_indices = torch.argmin(dists, dim=-1)  # (num_envs,)

    obs_offsets = (
      torch.arange(self.cfg.num_observations, device=self.device)
      * self.cfg.observation_spacing
    )
    obs_indices = closest_indices.unsqueeze(1) + obs_offsets.unsqueeze(
      0
    )  # (num_envs, num_observations)
    # Clamp indices to be within path bounds
    obs_indices = torch.clamp(obs_indices, 0, self.paths.shape[1] - 1)

    future_points_in_world = torch.gather(
      self.paths[..., :2], 1, obs_indices.unsqueeze(-1).expand(-1, -1, 2)
    )  # (num_envs, num_observations, 2)
    # make into homogenous coordinates
    future_points_in_world_hom = torch.cat(
      [
        future_points_in_world,
        torch.zeros(
          (self.num_envs, self.cfg.num_observations, 1), device=self.device
        ),
        torch.ones(
          (self.num_envs, self.cfg.num_observations, 1), device=self.device
        ),
      ],
      dim=-1,
    ).unsqueeze(-1)  # (num_envs, num_observations, 4, 1)

    world_to_robot = xy_yaw_to_mat(self.current_pose)  # (num_envs, 4, 4)
    robot_to_world = inv_mat(world_to_robot)  # (num_envs, 4, 4)
    future_points_relative = torch.matmul(
      robot_to_world.unsqueeze(1), future_points_in_world_hom
    ).squeeze(-1)  # (num_envs, num_observations, 4)
    path_obs = future_points_relative[..., :2].reshape(
      self.num_envs, -1
    )  # (num_envs, num_observations*2)

    # Goal state: distance to the final waypoint and orientation error to the
    # sampled goal yaw.
    heading = self.current_pose[:, 2]
    self.dist_to_goal = torch.norm(self.waypoints[:, -1, :] - root_pos, dim=-1)
    self.yaw_err = angle_distance(heading, self.goal_yaw_w)  # wrap(goal - heading)
    # near_goal: position only. Used by the path rewards (progress/alignment
    # saturate to max) so the robot is not penalized while settling at the goal.
    self.near_goal = self.dist_to_goal < self.cfg.near_goal_threshold
    # at_goal: position AND orientation within threshold. Used only to trigger
    # the stand. Requiring the orientation too means the gait stays active while
    # the robot turns onto the goal heading, and only freezes once it is fully
    # on the goal pose.
    self.at_goal = self.near_goal & (
      self.yaw_err.abs() < self.cfg.near_goal_yaw_threshold
    )

    # Append the goal orientation as (cos(yaw_err), sin(yaw_err)): continuous,
    # egocentric, and mirrors as [1, -1] like the path-point pairs.
    orient = torch.stack(
      [torch.cos(self.yaw_err), torch.sin(self.yaw_err)], dim=-1
    )  # (num_envs, 2)
    self.command_b = torch.cat([path_obs, orient], dim=-1)

    # At the goal pose (position + orientation), ask the (hidden) twist command
    # to stand: this freezes the gait phase and gates off the phase-based rewards
    # so the robot stands still. Delayed by a dwell timer: the robot must hold
    # the goal pose continuously for ``stand_delay`` seconds first, so it settles
    # onto the goal instead of freezing the instant it arrives.
    dt = self._env.step_dt
    self.at_goal_time = torch.where(
      self.at_goal,
      self.at_goal_time + dt,
      torch.zeros_like(self.at_goal_time),
    )
    stand = self.at_goal & (self.at_goal_time >= self.cfg.stand_delay)
    #self._set_twist_stand(stand)

    if self.cfg.eval_mode:
      self._log_eval_metrics(dists, closest_indices, root_pos)

  def _footprint_xy(self) -> torch.Tensor:
    """World-frame xy of the base footprint (midpoint of the two foot sites).

    Matches ``path_reward.base_footprint_xy`` — the position the training
    goal-reward is measured at — so the eval "goal reached" check uses the same
    reference. Returns a ``(num_envs, 2)`` tensor.
    """
    foot_xy = self.robot.data.site_pos_w[:, self._foot_site_ids, :2]
    return foot_xy.mean(dim=1)

  def _log_eval_metrics(
    self,
    dists: torch.Tensor,
    closest_indices: torch.Tensor,
    root_pos: torch.Tensor,
  ) -> None:
    """Append per-step path-following metrics and reset once the goal pose is hit.

    Logs ``starting_step, step, dist_to_path, |heading_err_vs_tangent|,
    dist_to_goal, yaw_err_to_goal`` to ``{_log_prefix}{counter}.csv`` (one row
    per step). ``dist_to_goal`` is the base-footprint distance to the final
    waypoint and ``yaw_err_to_goal`` is ``|wrap(goal_yaw - heading)|`` — the same
    two quantities the training goal-reward thresholds on.

    Shared by ``PathCommand`` (path policy) and ``VelocityPathCommand`` (velocity
    policy) so both are scored with the exact same metric on the exact same
    paths. Assumes a single env (``.item()`` calls) — this only runs at
    eval/play time.
    """
    del root_pos  # Goal distance is measured at the footprint (matches training).
    env_ar = torch.arange(self.num_envs, device=self.device)
    closest_dists = torch.min(dists, dim=-1).values  # (num_envs,)
    next_indices = torch.clamp(closest_indices + 1, max=self.paths.shape[1] - 1)
    closest_waypoints = self.paths[env_ar, closest_indices][..., :2]  # (num_envs, 2)
    next_waypoints = self.paths[env_ar, next_indices][..., :2]  # (num_envs, 2)
    path_tangents = next_waypoints - closest_waypoints  # (num_envs, 2)
    path_headings = torch.atan2(
      path_tangents[..., 1], path_tangents[..., 0]
    )  # (num_envs,)
    robot_headings = self.current_pose[:, 2]
    abs_angle_diff = torch.abs(angle_distance(robot_headings, path_headings))
    # Goal-reached criterion, matching training's goal-reward success metric:
    # base-footprint distance to the goal AND yaw error to the sampled goal yaw.
    dist_to_goal = torch.norm(self.waypoints[:, -1, :] - self._footprint_xy(), dim=-1)
    yaw_err = self.yaw_err.abs()
    with open(f"{self._log_prefix}{self.global_command_counter:03d}.csv", "a") as f:
      f.write(
        f"{self.starting_step},{self._env.common_step_counter},"
        f"{closest_dists.item()},{abs_angle_diff.item()},"
        f"{dist_to_goal.item()},{yaw_err.item()}\n"
      )

    goal_reached = (dist_to_goal < self.cfg.goal_pos_threshold) & (
      yaw_err < self.cfg.goal_yaw_threshold
    )
    if goal_reached.item():
      print("Reached goal pose, resetting env")
      print(f"{self.global_command_counter}/{self.cfg.pathlib_size} paths completed")
      self._env.reset()

  def _draw_obstacles(self, visualizer: DebugVisualizer, batch: int) -> None:
    """Draw the (virtual) obstacle world the path was planned through.

    The obstacles only ever exist in the planner — no MuJoCo geometry is created
    for them and the robot cannot collide with them — so this is the only place
    they become visible.
    """
    if self.obstacles is None:
      return
    for kind, x, y, yaw, a, b, half_height in self.obstacles[batch].cpu().numpy():
      if kind < 0:  # Empty slot.
        continue
      if kind == OBSTACLE_KIND_CYLINDER:
        visualizer.add_cylinder(
          np.array([x, y, 0.0]),
          np.array([x, y, 2.0 * half_height]),
          radius=float(a),
          color=_OBSTACLE_COLOR,
        )
      else:
        _add_box(visualizer, x, y, yaw, a, b, 2.0 * half_height)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    self._update_robot_pose()
    env_indices = visualizer.get_env_indices(self.num_envs)
    for batch in env_indices:
      self._draw_obstacles(visualizer, batch)
      for i in range(self.paths.shape[1]):
        path_point_with_z = np.concatenate(
          [self.paths[batch, i, :2].cpu().numpy(), [0.0]]
        )
        visualizer.add_sphere(path_point_with_z, radius=0.02, color=(0, 0, 1, 0.5))
      for i in range(self.waypoints[batch].shape[0]):
        waypoint_with_z = np.concatenate(
          [self.waypoints[batch, i].cpu().numpy(), [0.0]]
        )
        visualizer.add_sphere(waypoint_with_z, radius=0.05, color=(0, 1, 0, 1))

      # Goal orientation arrow at the final waypoint (cyan).
      goal_xy = self.waypoints[batch, -1].cpu().numpy()
      gyaw = float(self.goal_yaw_w[batch].cpu())
      goal_from = np.array([goal_xy[0], goal_xy[1], 0.05])
      goal_to = goal_from + np.array(
        [0.3 * math.cos(gyaw), 0.3 * math.sin(gyaw), 0.0]
      )
      visualizer.add_arrow(goal_from, goal_to, color=(0.0, 1.0, 1.0, 0.9), width=0.02)

      world_to_robot = (
        xy_yaw_to_mat(self.current_pose[batch : batch + 1]).cpu().numpy()
      )  # (1, 4, 4)
      for i in range(self.cfg.num_observations):
        command_point_hom = np.concatenate(
          [self.command_b[batch, i * 2 : i * 2 + 2].cpu().numpy(), [0.0, 1.0]]
        )
        command_point_in_world_hom = world_to_robot @ command_point_hom
        command_point_with_z = command_point_in_world_hom[0, :3]
        visualizer.add_sphere(command_point_with_z, radius=0.05, color=(1, 0, 0, 1))


@dataclass(kw_only=True)
class PathCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  num_waypoints: int = 4  # Number of waypoints in the generated path.
  curvature_limit: float = 10.0  # Max curvature; higher allows sharper turns.
  max_point_dxy: float = 3.0  # Max change in x/y between consecutive waypoints.
  point_spacing: float = 0.01  # Euclidean (arc-length) spacing between path samples, meters.
  num_observations: int = 20  # Number of future path points in the command obs.
  observation_spacing: int = 15  # Spacing between observed path points.
  goal_yaw_range: tuple[float, float] = (-math.pi, math.pi)
  """Range (radians) from which the goal orientation (world yaw) is sampled."""
  goal_yaw_seed: int = 0
  """Base seed for the deterministic per-path goal yaw used in ``eval_mode`` (so a
  path-policy run and a velocity-policy run get the same goal pose for each
  path). Ignored outside eval mode, where the goal yaw is sampled randomly."""
  foot_site_names: tuple[str, ...] = ("l_foot", "r_foot")
  """Foot sites whose midpoint is the base footprint — the position the eval
  "goal reached" check measures, matching the training goal-reward."""
  goal_pos_threshold: float = 0.05
  """Eval "goal reached" position limit (m): base-footprint distance to the final
  waypoint under which the goal position counts as reached. Defaults to the
  training goal-reward success threshold (``goal_reached_linear``)."""
  goal_yaw_threshold: float = 0.2
  """Eval "goal reached" orientation limit (rad): ``|wrap(goal_yaw - heading)|``
  under which the goal orientation counts as reached. Both this and
  ``goal_pos_threshold`` must hold to trigger the reset. Defaults to the training
  goal-reward success threshold (``goal_reached_angular``)."""
  near_goal_threshold: float = 0.3
  """Distance (m) to the final waypoint under which the robot is "near the goal":
  the progress/alignment rewards saturate to their max so the robot is not
  penalized while settling onto the goal."""
  near_goal_yaw_threshold: float = 0.3
  """Additional orientation gate (radians) for the *stand* trigger. The twist
  command is set to stand (freezing the gait) only once the robot is both within
  ``near_goal_threshold`` and within this yaw error of the goal orientation, so
  the gait stays active while it turns onto the goal heading."""
  stand_delay: float = 1.0
  """Seconds the robot must remain within both ``near_goal_threshold`` and
  ``near_goal_yaw_threshold`` *continuously* before the twist command is set to
  stand. 0 stands the instant the goal pose is reached; larger values make the
  robot settle onto the goal before the gait freezes. The timer resets whenever
  the robot leaves the goal pose or a new goal is sampled."""
  twist_command_name: str = "twist"
  """Name of the velocity command whose stand flag is driven at the goal pose."""
  max_points: int = 1500  # Samples stored per path (also the padded length).
  generator: Literal["spline", "astar"] = "spline"
  """How paths are generated.

  - ``spline``: a random b-spline through empty space (``path_gen.py``).
  - ``astar``: A* through a randomly generated field of cubes and cylinders
    (``path_gen_astar.py``), so the path corners around obstacles. The obstacles
    are *virtual*: they are drawn in the viewer but never simulated, so the robot
    can still walk straight through them — only the reference path knows about
    them.
  """
  obstacle_world: ObstacleWorldCfg = field(default_factory=ObstacleWorldCfg)
  """Obstacle sampling and planning parameters for the ``astar`` generator."""
  generation_seed: int = 0
  """Master seed for the ``astar`` generator; also part of its cache key."""
  num_workers: int | None = None
  """Processes used to generate the ``astar`` library. None picks a default;
  results are identical regardless of the worker count."""
  use_pathlib: bool = True
  pathlib_size: int = 50_000
  cache_dir: str | None = None  # Directory for the cached path library
  eval_mode: bool = False  # Use a fixed set of paths for evaluation if True.

  @dataclass
  class VizCfg:
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> PathCommand:
    return PathCommand(self, env)

  def __post_init__(self):
    if self.generator == "astar" and not self.use_pathlib:
      raise ValueError(
        "generator='astar' requires use_pathlib=True: planning a fresh A* path "
        "for every env on every reset is far too slow. Generate the library "
        "once (it is cached to disk) and sample from it instead."
      )
