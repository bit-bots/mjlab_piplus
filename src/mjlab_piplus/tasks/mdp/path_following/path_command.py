from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from mjlab.entity.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  import viser
  from mjlab.viewer.debug_visualizer import DebugVisualizer

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


# A path library is expensive twice over: minutes to plan and hundreds of
# megabytes on the GPU (50k paths x 1500 points). It is also fully determined by
# its generation parameters, so a library that has already been loaded is handed
# straight back to the next term that asks for the same one -- notably the
# footstep command, which places its footsteps along these same paths and would
# otherwise double the memory bill.
_PATHLIB_MEMO: dict[
  tuple[str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def load_or_generate_pathlib(
  *,
  pathlib_size: int,
  obstacle_world: ObstacleWorldCfg,
  point_spacing: float,
  num_waypoints: int,
  max_points: int,
  generation_seed: int,
  num_workers: int | None = None,
  cache_dir: str | None = None,
  device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Load the pre-generated path library from disk, or generate and cache it.

  The generator is deterministically seeded, so the library is fully determined
  by the generation parameters. We key the cache file on those params; a cache
  hit skips regenerating all ``pathlib_size`` paths on every training run, and
  an in-process hit skips the load (and the second copy in device memory).

  Returns ``(paths, waypoints, obstacles)``.
  """
  directory = Path(cache_dir or (Path(__file__).parent / "path_cache"))
  # The obstacle world has far too many knobs to spell out in a filename; hash
  # them instead.
  key = hashlib.sha1(
    repr(
      (
        obstacle_world.cache_key(),
        point_spacing,
        num_waypoints,
        max_points,
        generation_seed,
      )
    ).encode()
  ).hexdigest()[:12]
  cache_path = directory / f"astar_pathlib_n{pathlib_size}_{key}.pt"

  memo_key = (str(cache_path), str(device))
  memoized = _PATHLIB_MEMO.get(memo_key)
  if memoized is not None:
    return memoized

  if cache_path.exists():
    print(f"Loading cached path library from {cache_path}")
    data = torch.load(cache_path, map_location=device)
    result = (data["paths"], data["waypoints"], data["obstacles"])
    _PATHLIB_MEMO[memo_key] = result
    return result

  result = generate_astar_paths(
    num_paths=pathlib_size,
    cfg=obstacle_world,
    point_spacing=point_spacing,
    num_waypoints=num_waypoints,
    max_points=max_points,
    seed=generation_seed,
    device=device,
    progress=True,
    num_workers=num_workers,
  )
  directory.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      "paths": result[0].cpu(),
      "waypoints": result[1].cpu(),
      "obstacles": result[2].cpu(),
    },
    cache_path,
  )
  print(f"Cached path library to {cache_path}")
  _PATHLIB_MEMO[memo_key] = result
  return result


class PathCommand(CommandTerm):
  # Narrow the base class's ``cfg: CommandTermCfg`` so the config fields read
  # throughout this class resolve to PathCommandCfg (same pattern as
  # GoalPoseCommand / UniformVelocityCommandWithStop in ``mdp/commands.py``).
  cfg: PathCommandCfg

  def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.align = cfg.align
    self.global_command_counter = 0
    (
      self.all_paths,
      self.all_waypoints,
      self.all_obstacles,
    ) = self._load_or_generate_pathlib(cfg)
    # randomly select a path for each environment
    indices = torch.randint(0, cfg.pathlib_size, (self.num_envs,), device=self.device)
    self.paths = self.all_paths[indices]
    self.waypoints = self.all_waypoints[indices]
    self.obstacles = self.all_obstacles[indices]
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
    self.near_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.at_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    # Seconds the robot has continuously held the full goal pose (position +
    # orientation). Reported only; standing is the stand command's job.
    self.at_goal_time = torch.zeros(self.num_envs, device=self.device)
    # Envs whose freshly sampled path still has to be shifted onto the robot.
    # The shift is applied in _update_command, not here: at an episode reset the
    # command manager runs *before* sim.forward(), so the robot pose visible
    # during _resample_command is still the pre-reset one, and the path would be
    # anchored on wherever the robot used to be.
    self._pending_align = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    # Set by create_gui() when the viewer is active.
    self._gui_show_obstacles: "viser.GuiCheckboxHandle | None" = None
    self._gui_show_path: "viser.GuiCheckboxHandle | None" = None
    self._update_robot_pose()

  def _load_or_generate_pathlib(
    self, cfg: PathCommandCfg
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return load_or_generate_pathlib(
      pathlib_size=cfg.pathlib_size,
      obstacle_world=cfg.obstacle_world,
      point_spacing=cfg.point_spacing,
      num_waypoints=cfg.num_waypoints,
      max_points=cfg.max_points,
      generation_seed=cfg.generation_seed,
      num_workers=cfg.num_workers,
      cache_dir=cfg.cache_dir,
      device=self.device,
    )

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  @property
  def goal_pos_w(self) -> torch.Tensor:
    """World xy of the goal, i.e. the final waypoint. ``(num_envs, 2)``.

    Named to match ``GoalPoseCommand`` so the shared goal-reached rewards can
    read either term.
    """
    return self.waypoints[:, -1, :]

  @property
  def goal_command_b(self) -> torch.Tensor:
    """Goal in the base frame, ``(num_envs, 3)`` = ``[dx_b, dy_b, yaw_err]``.

    Completes the ``GoalPoseTerm`` contract, so the goal-pose rewards can read
    this command too. ``current_pose`` is refreshed every ``_update_command``.
    """
    delta_w = self.goal_pos_w - self.current_pose[:, :2]
    heading = self.current_pose[:, 2]
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    return torch.stack(
      [
        cos_h * delta_w[:, 0] + sin_h * delta_w[:, 1],
        -sin_h * delta_w[:, 0] + cos_h * delta_w[:, 1],
        self.yaw_err,
      ],
      dim=1,
    )

  def _resample_command(self, env_ids):
    self.starting_step = self._env.common_step_counter
    self._update_robot_pose()
    # New goal: reset the dwell timer so accumulated time doesn't carry over.
    self.at_goal_time[env_ids] = 0.0
    # Sample a fresh random goal orientation (world yaw) for the new paths.
    self.goal_yaw_w[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.goal_yaw_range
    )
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
    obstacles = self.all_obstacles[indices]

    self.paths[env_ids] = paths
    self.waypoints[env_ids] = waypoints
    self.obstacles[env_ids] = obstacles
    if self.align:
      self._pending_align[env_ids] = True

    if self.cfg.eval_mode:
      # write waypoints to csv
      with open("commands_path.csv", "a") as f:
        for i in range(self.waypoints.shape[0]):
          waypoint_str = ",".join(
            [f"{x:.2f}" for x in self.waypoints[i].cpu().numpy().flatten()]
          )
          f.write(f"{waypoint_str},")
        f.write(f"{self.global_command_counter}\n")
    self.global_command_counter += 1

  def _update_metrics(self):
    pass

  def _update_robot_pose(self) -> None:
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    heading = heading_from_quat(root_quat).unsqueeze(-1)
    self.current_pose = torch.cat([root_pos[..., :2], heading], dim=-1)

  def _align_pending(self) -> None:
    """Shift freshly sampled paths onto the robot's current position.

    Paths are generated in their start frame; this moves them -- and the
    obstacle world around them -- onto the robot. Only a translation: both
    generators start a path at the local origin but with a uniformly random
    initial direction, so the path already leaves the robot on a random bearing
    and there is nothing for a rotation to fix.
    """
    pending = self._pending_align.nonzero().flatten()
    if len(pending) == 0:
      return
    offset = self.current_pose[pending, :2].unsqueeze(1)
    self.paths[pending, :, :2] += offset
    self.waypoints[pending] += offset
    # Columns 1:3 are the obstacle center. Padding slots are shifted too, which
    # is harmless: the kind check skips them at draw time.
    self.obstacles[pending, :, 1:3] += offset
    self._pending_align[pending] = False

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    # get closest point on path to robot
    self._update_robot_pose()
    self._align_pending()
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
        torch.zeros((self.num_envs, self.cfg.num_observations, 1), device=self.device),
        torch.ones((self.num_envs, self.cfg.num_observations, 1), device=self.device),
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

    # Dwell timer on the full goal pose. It used to trigger a stand at the goal;
    # standing is the stand command's own type now, so this only records how long
    # the robot has held the goal.
    dt = self._env.step_dt
    self.at_goal_time = torch.where(
      self.at_goal,
      self.at_goal_time + dt,
      torch.zeros_like(self.at_goal_time),
    )

    if self.cfg.eval_mode:
      # hacky calculate metrics
      closest_dists = torch.min(dists, dim=-1).values  # (num_envs,)
      closest_waypoints = self.paths[:, closest_indices]  # (num_envs, 2)
      next_indices = torch.clamp(closest_indices + 1, max=self.paths.shape[1] - 1)
      next_waypoints = self.paths[:, next_indices]  # (num_envs, 2)
      path_tangents = next_waypoints - closest_waypoints  # (num_envs, 2)
      path_headings = torch.atan2(
        path_tangents[..., 1], path_tangents[..., 0]
      )  # (num_envs,)
      robot_headings = self.current_pose[:, 2]
      abs_angle_diff = torch.abs(angle_distance(robot_headings, path_headings))
      distances_to_goal = torch.norm(self.waypoints[:, -1, :] - root_pos, dim=-1)
      with open(f"path_logs{self.global_command_counter:03d}.csv", "a") as f:
        f.write(
          f"{self.starting_step},{self._env.common_step_counter},"
          f"{closest_dists.item()},{abs_angle_diff.item()},"
          f"{distances_to_goal.item()}\n"
        )

      if distances_to_goal.item() < 0.15:
        print("Reached goal, resetting env")
        self._env.reset()

  def _draw_obstacles(self, visualizer: DebugVisualizer, batch: int) -> None:
    """Draw the (virtual) obstacle world the path was planned through.

    The obstacles only ever exist in the planner — no MuJoCo geometry is created
    for them and the robot cannot collide with them — so this is the only place
    they become visible.
    """
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

  # Visualization toggles. Handles are created by create_gui when the viewer is
  # up; before that (and headless) the config values stand.

  def _show_obstacles(self) -> bool:
    if self._gui_show_obstacles is not None:
      return bool(self._gui_show_obstacles.value)
    return self.cfg.viz.show_obstacles

  def _show_path(self) -> bool:
    if self._gui_show_path is not None:
      return bool(self._gui_show_path.value)
    return self.cfg.viz.show_path

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    del get_env_idx, on_change, request_action
    with server.gui.add_folder(f"{name.capitalize()} viz"):
      self._gui_show_obstacles = server.gui.add_checkbox(
        "Obstacles", initial_value=self.cfg.viz.show_obstacles
      )
      self._gui_show_path = server.gui.add_checkbox(
        "Path points", initial_value=self.cfg.viz.show_path
      )

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    self._update_robot_pose()
    env_indices = visualizer.get_env_indices(self.num_envs)
    show_obstacles, show_path = self._show_obstacles(), self._show_path()
    stride = max(int(self.cfg.viz.path_point_stride), 1)
    for batch in env_indices:
      if show_obstacles:
        self._draw_obstacles(visualizer, batch)

      if show_path:
        # One host transfer for the whole path. Fetching each point separately
        # cost ~29 ms a frame at max_points=1500 -- more than the physics step.
        xy = self.paths[batch, ::stride, :2].cpu().numpy()
        for point in np.column_stack([xy, np.zeros(len(xy))]):
          visualizer.add_sphere(point, radius=0.02, color=(0, 0, 1, 0.5))

      waypoints = self.waypoints[batch].cpu().numpy()
      for point in np.column_stack([waypoints, np.zeros(len(waypoints))]):
        visualizer.add_sphere(point, radius=0.05, color=(0, 1, 0, 1))

      # Goal orientation arrow at the final waypoint (cyan).
      gyaw = float(self.goal_yaw_w[batch].cpu())
      goal_from = np.array([waypoints[-1, 0], waypoints[-1, 1], 0.05])
      goal_to = goal_from + np.array([0.3 * math.cos(gyaw), 0.3 * math.sin(gyaw), 0.0])
      visualizer.add_arrow(goal_from, goal_to, color=(0.0, 1.0, 1.0, 0.9), width=0.02)

      # The observed path points, mapped back into the world (red).
      world_to_robot = xy_yaw_to_mat(self.current_pose[batch : batch + 1]).cpu().numpy()
      n_obs = self.cfg.num_observations
      obs_xy = self.command_b[batch, : 2 * n_obs].cpu().numpy().reshape(n_obs, 2)
      hom = np.column_stack([obs_xy, np.zeros(n_obs), np.ones(n_obs)])
      for point in (world_to_robot[0] @ hom.T).T:
        visualizer.add_sphere(point[:3], radius=0.05, color=(1, 0, 0, 1))


@dataclass(kw_only=True)
class PathCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  num_waypoints: int = 4  # Number of waypoints in the generated path.
  point_spacing: float = (
    0.01  # Euclidean (arc-length) spacing between path samples, meters.
  )
  num_observations: int = 20  # Number of future path points in the command obs.
  observation_spacing: int = 15  # Spacing between observed path points.
  goal_yaw_range: tuple[float, float] = (-math.pi, math.pi)
  """Range (radians) from which the goal orientation (world yaw) is sampled."""
  near_goal_threshold: float = 0.3
  """Distance (m) to the final waypoint under which the robot is "near the goal":
  the progress/alignment rewards saturate to their max so the robot is not
  penalized while settling onto the goal."""
  near_goal_yaw_threshold: float = 0.3
  """Additional orientation gate (radians) for the *stand* trigger. The twist
  command is set to stand (freezing the gait) only once the robot is both within
  ``near_goal_threshold`` and within this yaw error of the goal orientation, so
  the gait stays active while it turns onto the goal heading."""
  align: bool = False
  """Shift each newly sampled path (and its obstacle world) so it starts at the
  robot's current xy. Required when the path can be handed to a robot that is
  already somewhere other than its spawn -- i.e. when switching to a path
  mid-episode under a ``MultiCommand``. Note this only translates: the path is
  not rotated onto the robot's heading, so it may start behind the robot."""
  max_points: int = 1500  # Samples stored per path (also the padded length).
  obstacle_world: ObstacleWorldCfg = field(default_factory=ObstacleWorldCfg)
  """Obstacle sampling and planning parameters for the A* generator
  (``path_gen_astar.py``). The obstacles are *virtual*: they are drawn in the
  viewer but never simulated, so the robot can still walk straight through
  them -- only the reference path knows about them."""
  generation_seed: int = 0
  """Master seed for the generator; also part of the library's cache key."""
  num_workers: int | None = None
  """Processes used to generate the path library. None picks a default; results
  are identical regardless of the worker count."""
  pathlib_size: int = 50_000
  """Paths pre-planned once and sampled from thereafter. Planning a fresh A*
  path for every env on every reset is far too slow, so the library is always
  used; it is cached to disk keyed on the generation parameters."""
  cache_dir: str | None = None  # Directory for the cached path library
  eval_mode: bool = False  # Use a fixed set of paths for evaluation if True.

  @dataclass
  class VizCfg:
    scale: float = 0.5
    show_obstacles: bool = True
    """Draw the virtual obstacle world. Also toggleable live in the viewer."""
    show_path: bool = True
    """Draw the reference path. Also toggleable live in the viewer. This is by
    far the most expensive thing the term draws -- ``max_points`` spheres a
    frame -- so it is the first thing to turn off if the viewer is sluggish."""
    path_point_stride: int = 1
    """Draw every Nth path point. Raising this thins the path out rather than
    hiding it, trading fidelity for frame time."""

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> PathCommand:
    return PathCommand(self, env)
