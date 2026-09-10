"""Footstep command: a stream of 6D footstep targets for the next N steps.

The command hands the policy a *footstep plan*: where each of the next few
footfalls should land and how the foot should be oriented there. One footstep is
a 6-vector

    [x, y, z, roll, pitch, yaw]

expressed **relative to the resting pose of that foot in the robot's default
standing pose**, so an all-zero plan means "put every foot back where it stands
by default", i.e. walk in place. Only x, y and yaw are exercised for now; z,
roll and pitch are carried through the whole pipeline (sampling, observation,
reward) but sampled at zero until terrain arrives.

Frames
------
Three frames are involved:

- **Foot rest offset** ``rest_j``: where foot ``j``'s site sits in a
  ground-projected base frame when the robot stands in its default pose (``+-``
  half the stance width laterally). Computed once from the robot's
  ``init_state`` keyframe.
- **Support frame** ``F̂``: the stance frame implied by the foot that most
  recently touched down, recovered from that foot's *measured* pose as
  ``F̂ = M ∘ rest_j^-1``.
- **Stance frame** ``F_k``: the pose of footstep ``k``, one delta ``D_k`` on
  from footstep ``k-1``.

The world target of footstep ``k`` for its foot ``j`` is
``F̂ ∘ D_{s+1} ∘ ... ∘ D_k ∘ rest_j``, where ``s`` is the footstep the support
foot is standing on. Because the rest offset is rotated by the stance yaw, a
plan that only turns yaw makes the robot pivot on the spot with the feet keeping
their stance width -- which is what "relative to the resting pose" has to mean
for a turn.

Re-grounding
------------
``F̂`` is re-measured **every step**, so the plan is not a fixed trajectory laid
out in the world that the robot can fall arbitrarily behind. It is a chain of
relative steps rooted on wherever the robot actually has its weight: undershoot
one footstep and the next is asked for relative to where the foot really landed,
not where it was supposed to. Tracking error therefore stays per-step and
bounded instead of accumulating over a plan.

Height and tilt are the exception: a target height is a property of the terrain,
not of the robot's footing, so ``z``/``roll``/``pitch`` stay ground-referenced
and are not re-grounded on the support foot.

Because the live targets move with the support foot, the reward cannot read them
back after a touchdown -- by then the target has slid onto the very foot that
hit it and every landing would score perfect. So the target of each footstep is
**latched** at touchdown into ``stance_target_w``, and that frozen pose is what
the planted foot is scored against.

Timing
------
Which footstep is next is read straight off the gait clock the rest of this task
already runs on (``phase``/``feet_phase``): with two feet at gait offsets
``0.0`` and ``0.5``, one foot lands every half period. Footstep ``k`` is the
``k``-th touchdown after the episode start, landing at ``t = (k + 1) * T / 2``,
and the feet strictly alternate. So the plan is a *timed* footstep schedule --
the policy has to hit the placement and the timing, exactly as the gait rewards
already demand.

Sampling
--------
Two scenario families, mixed by ``path_fraction``:

- **Path**: take an A* path from the same library the path command walks and
  read footsteps off it at a fixed arc-length spacing, one stance frame per
  step, yaw following the path tangent. Gives coherent, curved, obstacle-shaped
  walking.
- **Gaussian**: sample a per-env nominal step ``mu`` (this is the env's cruising
  stride, and ``mu = 0`` is walking in place), then per footstep a delta
  ``mu + N(0, sigma)`` from the *previous* footstep, in that footstep's own
  frame.

Both families end up in the same representation -- a chain of per-step deltas --
and both are clipped to ``max_step_delta``, so no footstep can ever ask for more
than the configured reach in any axis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

from mjlab_piplus.tasks.mdp.path_following.path_command import load_or_generate_pathlib
from mjlab_piplus.tasks.mdp.path_following.path_gen_astar import ObstacleWorldCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# One footstep command: x, y, z, roll, pitch, yaw.
FOOTSTEP_DIM = 6
# Per footstep in the observation: which foot (+1 left / -1 right), the target
# position in the robot's ground-projected base frame, the target roll/pitch,
# and the target yaw as a continuous (cos, sin) pair.
OBS_PER_FOOTSTEP = 8

# Marker colors: left foot blue-ish, right foot red-ish, dimmed for the steps
# further into the future.
_LEFT_COLOR = (0.15, 0.45, 0.95)
_RIGHT_COLOR = (0.95, 0.35, 0.15)


@dataclass
class FootCfg:
  """One foot of the gait: where it is measured and when it swings."""

  site_name: str
  """Site whose pose is compared against the footstep target."""
  gait_offset: float
  """Phase offset of this foot in the gait cycle, matching the ``offset`` list
  of the ``feet_phase`` / ``feet_gait`` rewards. The foot touches down when its
  leg phase wraps, i.e. at global phase ``(-gait_offset) % 1``."""
  side: Literal["left", "right"]
  """Which side of the body. Only used for the sign the observation carries and
  for the debug colors."""
  rest_offset: tuple[float, float, float] | None = None
  """Position of this foot's site in a ground-projected base frame when the
  robot stands in its default pose, ``(x, y, z_above_ground)``. ``None``
  (default) computes it from the robot's ``init_state`` keyframe."""


def _rest_offsets_from_keyframe(
  env: ManagerBasedRlEnv, robot: Entity, site_ids: Sequence[int]
) -> list[tuple[float, float, float]]:
  """Foot rest offsets read off the robot's default standing keyframe.

  Runs forward kinematics on the scene's host-side template model, put in the
  ``init_state`` keyframe that :class:`Entity` adds for its default pose, and
  measures each foot site in a ground-projected base frame. ``z`` is kept as the
  site's absolute height, since footstep heights are ground-referenced.

  The scene's model is used rather than the robot's own spec because an entity
  spec attached to a scene can no longer be compiled on its own, and a detached
  copy of it loses the keyframe.

  ``site_ids`` are entity-local foot site indices, in the order the offsets are
  wanted.
  """
  model = env.sim.mj_model
  if model.nkey == 0:
    raise RuntimeError(
      "The scene model has no keyframe to read the default standing pose from; "
      "set FootCfg.rest_offset explicitly."
    )
  key_id = next((i for i in range(model.nkey) if "init_state" in model.key(i).name), 0)
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, key_id)
  mujoco.mj_kinematics(model, data)

  root_pos = data.xpos[int(robot.indexing.root_body_id)]
  w, x, y, z = data.xquat[int(robot.indexing.root_body_id)]
  yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

  offsets = []
  for local_id in site_ids:
    site_pos = data.site_xpos[int(robot.indexing.site_ids[local_id])]
    dx, dy = site_pos[0] - root_pos[0], site_pos[1] - root_pos[1]
    offsets.append(
      (
        float(cos_yaw * dx + sin_yaw * dy),
        float(-sin_yaw * dx + cos_yaw * dy),
        float(site_pos[2]),
      )
    )
  return offsets


class FootstepCommand(CommandTerm):
  cfg: FootstepCommandCfg

  def __init__(self, cfg: FootstepCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.num_feet = len(cfg.feet)
    if self.num_feet != 2:
      raise ValueError(
        "FootstepCommand assumes a biped: exactly two feet alternating every "
        f"half gait period, got {self.num_feet}."
      )

    # Foot sites, resolved one at a time so the order follows ``cfg.feet``
    # rather than the model's site order.
    site_ids = []
    for foot in cfg.feet:
      entity_cfg = SceneEntityCfg(cfg.entity_name, site_names=(foot.site_name,))
      entity_cfg.resolve(env.scene)
      site_ids.append(int(entity_cfg.site_ids[0]))
    self.foot_site_ids = torch.tensor(site_ids, dtype=torch.long, device=self.device)

    auto_offsets = None
    if any(foot.rest_offset is None for foot in cfg.feet):
      auto_offsets = _rest_offsets_from_keyframe(env, self.robot, site_ids)
    rest = [
      foot.rest_offset if foot.rest_offset is not None else auto_offsets[i]
      for i, foot in enumerate(cfg.feet)
    ]
    # (num_feet, 3): foot site in the ground-projected default stance frame.
    self.rest_offset = torch.tensor(rest, dtype=torch.float32, device=self.device)

    # Which foot lands on an even footstep index. Footstep ``k`` lands at
    # ``t/T = (k + 1) / 2``, i.e. at global phase 0.5 for even ``k`` and 0.0 for
    # odd ``k``; a foot touches down at global phase ``(-gait_offset) % 1``.
    touchdown_phase = [(-foot.gait_offset) % 1.0 for foot in cfg.feet]
    self._even_foot = min(
      range(self.num_feet), key=lambda j: abs(touchdown_phase[j] - 0.5)
    )
    self._odd_foot = min(
      range(self.num_feet),
      key=lambda j: min(touchdown_phase[j], 1.0 - touchdown_phase[j]),
    )
    if self._even_foot == self._odd_foot:
      raise ValueError(
        "The two feet must touch down half a gait period apart (gait offsets "
        f"0.0 and 0.5), got {[f.gait_offset for f in cfg.feet]}."
      )
    self._foot_sign = torch.tensor(
      [1.0 if foot.side == "left" else -1.0 for foot in cfg.feet],
      device=self.device,
    )

    # The plan, as the chain of per-step deltas: footstep ``plan_base[e] + i``
    # is stored at ``deltas[e, i]``. The deltas are the stored form on purpose --
    # the horizontal part of the plan is re-grounded on the measured support
    # foot every step, so there is no fixed frame to hold absolute poses in.
    self.deltas = torch.zeros(
      self.num_envs, cfg.max_steps, FOOTSTEP_DIM, device=self.device
    )
    # Absolute (ground-referenced) z, roll and pitch per footstep. These are the
    # one part of the plan that is *not* re-grounded: a target height is a
    # property of the terrain, not of where the robot happens to be standing.
    self.plan_height = torch.zeros(self.num_envs, cfg.max_steps, 3, device=self.device)
    self.plan_base = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # Start-of-plan bookkeeping is deferred to _update_command: at an episode
    # reset the command manager runs before ``episode_length_buf`` is zeroed
    # *and* before the sim forward, so neither the gait clock nor the robot pose
    # is trustworthy here.
    self._pending_start = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._prev_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    self.command_b = torch.zeros(
      self.num_envs, cfg.num_observations * OBS_PER_FOOTSTEP, device=self.device
    )
    # Stance frame implied by the foot currently carrying the robot; the live
    # root of the plan, recomputed every step. Kept for the debug viewer.
    self.support_frame = torch.zeros(self.num_envs, 3, device=self.device)
    # Where the next touchdown is currently asked to land, ``(num_envs, 6)``.
    # Latched into ``stance_target_w`` the moment that footstep lands.
    self.next_target_w = torch.zeros(self.num_envs, FOOTSTEP_DIM, device=self.device)
    # Target each foot was given for its most recent landing, frozen at
    # touchdown, and which feet are in stance now. Read by the footstep rewards.
    self.stance_target_w = torch.zeros(
      self.num_envs, self.num_feet, FOOTSTEP_DIM, device=self.device
    )
    self.foot_in_stance = torch.zeros(
      self.num_envs, self.num_feet, dtype=torch.bool, device=self.device
    )

    # ``ActiveAwareTerm``: set by MultiCommand to the envs this command is
    # currently driving, and left None when the term runs standalone.
    self.active_env_mask: torch.Tensor | None = None

    self._all_paths: torch.Tensor | None = None
    if cfg.path_fraction > 0.0:
      self._all_paths = load_or_generate_pathlib(
        pathlib_size=cfg.pathlib.pathlib_size,
        obstacle_world=cfg.pathlib.obstacle_world,
        point_spacing=cfg.pathlib.point_spacing,
        num_waypoints=cfg.pathlib.num_waypoints,
        max_points=cfg.pathlib.max_points,
        generation_seed=cfg.pathlib.generation_seed,
        num_workers=cfg.pathlib.num_workers,
        cache_dir=cfg.pathlib.cache_dir,
        device=self.device,
      )[0]

    self.metrics["error_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_z"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

  ##
  # Properties.
  ##

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  @property
  def gait_period(self) -> float:
    return 1.0 / self.cfg.frequency

  def step_index(self) -> torch.Tensor:
    """``(num_envs,)`` index of the *next* touchdown.

    Footstep ``k`` lands at ``t = (k + 1) * T / 2``, so the footstep being
    stepped toward at time ``t`` is ``floor(2t / T)``. Derived from the same
    ``episode_length_buf`` clock as the ``phase`` observation, so the plan and
    the gait can never drift apart.
    """
    t = self._env.episode_length_buf * self._env.step_dt
    return torch.floor(2.0 * t / self.gait_period).long()

  def foot_of_step(self, step: torch.Tensor) -> torch.Tensor:
    """Which foot lands footstep ``step`` (any shape, may be negative)."""
    return torch.where(
      step % 2 == 0,
      torch.full_like(step, self._even_foot),
      torch.full_like(step, self._odd_foot),
    )

  ##
  # Plan generation.
  ##

  def _integrate(
    self, deltas: torch.Tensor, start: torch.Tensor | None = None
  ) -> torch.Tensor:
    """Chain per-step deltas into a run of stance frames ``(x, y, yaw)``.

    ``deltas[e, i]`` is a footstep's pose relative to the one before it, with
    the translation expressed in that previous footstep's own frame (so a
    constant delta walks a circle rather than a straight line once yaw
    accumulates).

    ``start`` is the ``(num_envs, 3)`` frame the chain hangs off -- in practice
    the live support frame; ``None`` means the identity, which is what the
    delta round-trip in :meth:`_to_deltas` is defined against.

    The chain closes in the form ``p_i = p_start + sum_{j<=i} R(yaw_{j-1}) d_j``,
    which is a cumulative sum -- no sequential loop over the horizon.
    """
    yaw_start = torch.zeros_like(deltas[:, :1, 5]) if start is None else start[:, 2:3]
    yaw = yaw_start + torch.cumsum(deltas[..., 5], dim=1)
    yaw_prev = torch.cat([yaw_start, yaw[:, :-1]], dim=1)
    cos_prev, sin_prev = torch.cos(yaw_prev), torch.sin(yaw_prev)
    dx = cos_prev * deltas[..., 0] - sin_prev * deltas[..., 1]
    dy = sin_prev * deltas[..., 0] + cos_prev * deltas[..., 1]
    pos = torch.cumsum(torch.stack([dx, dy], dim=-1), dim=1)
    if start is not None:
      pos = pos + start[:, None, :2]
    return torch.cat([pos, yaw.unsqueeze(-1)], dim=-1)

  def _to_deltas(self, poses: torch.Tensor) -> torch.Tensor:
    """Absolute stance frames to per-step deltas (all six axes).

    Inverse of :meth:`_integrate` on the horizontal axes, and a plain
    first difference on the vertical/tilt ones.
    """
    zero = torch.zeros_like(poses[:, :1])
    prev = torch.cat([zero, poses[:, :-1]], dim=1)
    yaw_prev = prev[..., 5]
    cos_prev, sin_prev = torch.cos(yaw_prev), torch.sin(yaw_prev)
    gx = poses[..., 0] - prev[..., 0]
    gy = poses[..., 1] - prev[..., 1]
    return torch.stack(
      [
        cos_prev * gx + sin_prev * gy,
        -sin_prev * gx + cos_prev * gy,
        poses[..., 2] - prev[..., 2],
        wrap_to_pi(poses[..., 3] - prev[..., 3]),
        wrap_to_pi(poses[..., 4] - prev[..., 4]),
        wrap_to_pi(poses[..., 5] - prev[..., 5]),
      ],
      dim=-1,
    )

  def _sample_gaussian_deltas(self, num: int) -> torch.Tensor:
    """Per-env cruising stride plus per-footstep Gaussian jitter.

    The nominal ``mu`` is what makes this useful as training data: pure i.i.d.
    noise around zero would only ever produce a robot shuffling on the spot,
    whereas a per-env stride sampled once gives coherent travel (and ``mu = 0``
    still covers walking in place).
    """
    cfg = self.cfg
    low = torch.tensor([r[0] for r in cfg.nominal_delta_range], device=self.device)
    high = torch.tensor([r[1] for r in cfg.nominal_delta_range], device=self.device)
    mu = low + (high - low) * torch.rand(num, FOOTSTEP_DIM, device=self.device)
    std = torch.tensor(cfg.delta_std, device=self.device)
    noise = torch.randn(num, cfg.max_steps, FOOTSTEP_DIM, device=self.device)
    return mu.unsqueeze(1) + noise * std

  def _sample_path_poses(self, num: int) -> torch.Tensor:
    """``_path_poses`` for ``num`` randomly drawn paths and step lengths."""
    assert self._all_paths is not None
    idx = torch.randint(0, self._all_paths.shape[0], (num,), device=self.device)
    step_length = torch.empty(num, 1, device=self.device).uniform_(
      *self.cfg.path_step_length_range
    )
    return self._path_poses(idx, step_length)

  def _path_poses(
    self, idx: torch.Tensor, step_length: torch.Tensor
  ) -> torch.Tensor:
    """Stance frames read off named A* paths at a fixed arc-length spacing.

    ``idx`` is ``(num,)`` into the path library and ``step_length`` ``(num, 1)``
    in meters. Split out from the sampler so an evaluation can ask for one
    specific path at one specific cadence (see ``eval/eval_commands.py``).

    The path is rotated so its initial tangent points along the anchor's
    forward axis (a path is generated on a random bearing, and the robot must
    not be asked to jump sideways onto it) and translated to start at the
    anchor. Footstep ``k`` sits at arc length ``(k + 1) * step_length``, with
    the stance yaw following the path tangent there. Past the end of the path
    the stored points repeat the goal, so the plan flattens into walking in
    place at the goal.
    """
    assert self._all_paths is not None
    cfg = self.cfg
    arc = (
      torch.arange(1, cfg.max_steps + 1, device=self.device, dtype=torch.float32)
      * step_length
    )
    point_idx = torch.round(arc / cfg.pathlib.point_spacing).long()
    point_idx = point_idx.clamp(0, self._all_paths.shape[1] - 1)
    # Index the library directly: materializing whole paths first would copy
    # 1500 points per env to keep max_steps of them.
    points = self._all_paths[idx.unsqueeze(1), point_idx]  # (num, max_steps, 6)
    start = self._all_paths[idx, 0]  # (num, 6): x, y, dx, dy, ddx, ddy

    theta0 = torch.atan2(start[:, 3], start[:, 2]).unsqueeze(1)
    cos0, sin0 = torch.cos(-theta0), torch.sin(-theta0)
    rel_x = points[..., 0] - start[:, 0].unsqueeze(1)
    rel_y = points[..., 1] - start[:, 1].unsqueeze(1)
    x = cos0 * rel_x - sin0 * rel_y
    y = sin0 * rel_x + cos0 * rel_y
    yaw = wrap_to_pi(torch.atan2(points[..., 3], points[..., 2]) - theta0)
    zeros = torch.zeros_like(x)
    return torch.stack([x, y, zeros, zeros, zeros, yaw], dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    num = len(env_ids)
    deltas = self._sample_gaussian_deltas(num)

    if self._all_paths is not None:
      from_path = (
        (torch.rand(num, device=self.device) < self.cfg.path_fraction)
        .unsqueeze(-1)
        .unsqueeze(-1)
      )
      path_deltas = self._to_deltas(self._sample_path_poses(num))
      deltas = torch.where(from_path, path_deltas, deltas)

    # The reach limit is enforced on the deltas, so it holds for both scenario
    # families and for every axis independently. A path corner sharper than
    # ``max_step_delta``'s yaw allowance is therefore cut rather than followed:
    # the plan stays reachable, it just no longer traces that corner exactly.
    limit = torch.tensor(self.cfg.max_step_delta, device=self.device)
    deltas = deltas.clamp(-limit, limit)

    self.deltas[env_ids] = deltas
    # Height and tilt are ground-referenced rather than chained off the robot,
    # so their absolute values are accumulated once here.
    self.plan_height[env_ids] = torch.cumsum(deltas[..., 2:5], dim=1)
    # ``plan_base`` is set in _update_command, where the gait clock is current.
    self._pending_start[env_ids] = True

  ##
  # Command update.
  ##

  def _base_pose(self) -> torch.Tensor:
    """``(num_envs, 3)`` ground-projected robot base pose ``(x, y, yaw)``."""
    return torch.cat(
      [
        self.robot.data.root_link_pos_w[:, :2],
        self.robot.data.heading_w.unsqueeze(-1),
      ],
      dim=-1,
    )

  def _support_frame(self, step: torch.Tensor) -> torch.Tensor:
    """``(num_envs, 3)`` stance frame implied by the foot carrying the robot.

    The support foot is the one that took footstep ``step - 1``, i.e. the most
    recent touchdown on the gait clock. Its *measured* pose is inverted through
    that foot's rest offset to recover the stance frame it represents:
    ``F = M ∘ rest^-1``. This is the root the whole plan hangs off, and it is
    re-measured every step -- so where the robot actually put its foot, not
    where it was told to, is what the next footsteps are laid out from.
    """
    foot = self.foot_of_step(step - 1)
    pos, rpy = self.foot_pose_w()
    gather_idx = foot.view(-1, 1, 1).expand(-1, 1, 3)
    measured_pos = torch.gather(pos, 1, gather_idx).squeeze(1)
    measured_yaw = torch.gather(rpy, 1, gather_idx).squeeze(1)[:, 2]
    rest = self.rest_offset[foot]
    cos_m, sin_m = torch.cos(measured_yaw), torch.sin(measured_yaw)
    return torch.stack(
      [
        measured_pos[:, 0] - (cos_m * rest[:, 0] - sin_m * rest[:, 1]),
        measured_pos[:, 1] - (sin_m * rest[:, 0] + cos_m * rest[:, 1]),
        measured_yaw,
      ],
      dim=-1,
    )

  def _targets_world(self, step: torch.Tensor, count: int) -> torch.Tensor:
    """World poses of footsteps ``step .. step + count - 1``.

    Returns ``(num_envs, count, 6)`` holding ``[x, y, z, roll, pitch, yaw]``.
    The horizontal axes are chained off the live support frame, so the whole
    run is re-derived from the robot's current footing on every call; height and
    tilt come from the ground-referenced part of the plan.

    Slots outside the plan get a zero delta, which holds the stance where it is
    -- before the plan starts and after it runs out, the command reads as
    walking in place.
    """
    horizon = torch.arange(count, device=self.device)
    steps = step.unsqueeze(1) + horizon.unsqueeze(0)
    slot = steps - self.plan_base.unsqueeze(1)
    inside = (slot >= 0) & (slot < self.cfg.max_steps)
    slot = slot.clamp(0, self.cfg.max_steps - 1)

    deltas = torch.gather(
      self.deltas, 1, slot.unsqueeze(-1).expand(-1, -1, FOOTSTEP_DIM)
    )
    deltas = torch.where(inside.unsqueeze(-1), deltas, torch.zeros_like(deltas))
    self.support_frame = self._support_frame(step)
    frames = self._integrate(deltas, start=self.support_frame)  # x, y, yaw

    height = torch.gather(self.plan_height, 1, slot.unsqueeze(-1).expand(-1, -1, 3))
    height = torch.where(
      inside.unsqueeze(-1), height, torch.zeros_like(height)
    )  # z, roll, pitch

    rest = self.rest_offset[self.foot_of_step(steps)]  # (num_envs, count, 3)
    cos_k, sin_k = torch.cos(frames[..., 2]), torch.sin(frames[..., 2])
    return torch.stack(
      [
        frames[..., 0] + cos_k * rest[..., 0] - sin_k * rest[..., 1],
        frames[..., 1] + sin_k * rest[..., 0] + cos_k * rest[..., 1],
        height[..., 0] + rest[..., 2],
        height[..., 1],
        height[..., 2],
        wrap_to_pi(frames[..., 2]),
      ],
      dim=-1,
    )

  def _update_command(self) -> None:
    step = self.step_index()  # index of the next touchdown
    # Under a MultiCommand only the *selected* sub-command is resampled at an
    # episode reset, so an inactive footstep term keeps a plan whose base index
    # belongs to the previous episode while the gait clock restarts at 0.
    # Restart whenever the clock has rewound, so the term always holds a sane
    # (if unused) plan for the episode it is in.
    self._pending_start |= step < self.plan_base
    pending = self._pending_start.nonzero().flatten()
    if len(pending) > 0:
      self.plan_base[pending] = step[pending]
      self._prev_step[pending] = step[pending]
      # No commanded footstep has landed yet, so hold each foot to where it
      # already is: a fresh plan must not open with a phantom tracking error.
      pos, rpy = self.foot_pose_w()
      self.stance_target_w[pending, :, :3] = pos[pending]
      self.stance_target_w[pending, :, 3:] = rpy[pending]
      self._pending_start[pending] = False

    # A footstep has just landed: freeze the target it was given as the standard
    # that foot is now held to. Latching is what keeps the reward honest -- the
    # live targets are re-grounded on the support foot every step, so reading a
    # target back after touchdown would read one that has already moved onto the
    # foot that hit it, and every landing would score perfect.
    landed = (step > self._prev_step).nonzero().flatten()
    if len(landed) > 0:
      foot = self.foot_of_step(self._prev_step)[landed]
      self.stance_target_w[landed, foot] = self.next_target_w[landed]
    self._prev_step = step

    # Observation: the next ``num_observations`` footsteps, each in the robot's
    # ground-projected base frame.
    steps = step.unsqueeze(1) + torch.arange(
      self.cfg.num_observations, device=self.device
    ).unsqueeze(0)
    targets = self._targets_world(step, self.cfg.num_observations)
    self.next_target_w = targets[:, 0]
    base = self._base_pose()
    heading = base[:, 2].unsqueeze(1)
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    dx = targets[..., 0] - base[:, 0].unsqueeze(1)
    dy = targets[..., 1] - base[:, 1].unsqueeze(1)
    delta_yaw = wrap_to_pi(targets[..., 5] - heading)
    obs = torch.stack(
      [
        self._foot_sign[self.foot_of_step(steps)],
        cos_h * dx + sin_h * dy,
        -sin_h * dx + cos_h * dy,
        # Absolute target height: zero on flat ground, the terrain height under
        # the footstep once terrain lands.
        targets[..., 2],
        targets[..., 3],
        targets[..., 4],
        torch.cos(delta_yaw),
        torch.sin(delta_yaw),
      ],
      dim=-1,
    )
    self.command_b = obs.reshape(self.num_envs, -1)

    # Stance/swing split, using the same leg phase as the gait rewards.
    global_phase = (
      self._env.episode_length_buf * self._env.step_dt % self.gait_period
    ) / self.gait_period
    offsets = torch.tensor(
      [foot.gait_offset for foot in self.cfg.feet], device=self.device
    )
    leg_phase = (global_phase.unsqueeze(1) + offsets.unsqueeze(0)) % 1.0
    self.foot_in_stance = leg_phase < self.cfg.stance_threshold

  def _update_metrics(self) -> None:
    """Accumulate the mean stance-foot error, per axis group.

    Only counted while this command is the active type: under a
    ``MultiCommand`` the plan keeps ticking for every env, and averaging in the
    error of a command nobody is following would drown the signal. The flip side
    is that the logged value scales with how much of the episode the footstep
    command actually drove.
    """
    max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    pos, rpy = self.foot_pose_w()
    error = pos - self.stance_target_w[..., :3]
    yaw_error = wrap_to_pi(rpy[..., 2] - self.stance_target_w[..., 5]).abs()
    in_stance = self.foot_in_stance.float()
    if self.active_env_mask is not None:
      in_stance = in_stance * self.active_env_mask.unsqueeze(1).float()
    count = in_stance.sum(dim=1).clamp(min=1.0)
    for name, value in (
      ("error_xy", torch.norm(error[..., :2], dim=-1)),
      ("error_z", error[..., 2].abs()),
      ("error_yaw", yaw_error),
    ):
      self.metrics[name] += (value * in_stance).sum(dim=1) / count / max_command_step

  def foot_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Measured foot poses, both ``(num_envs, num_feet, 3)``.

    Returns ``(position, roll_pitch_yaw)`` of the foot sites in the world frame,
    laid out to line up axis-for-axis with ``stance_target_w``.
    """
    pos = self.robot.data.site_pos_w[:, self.foot_site_ids]
    quat = self.robot.data.site_quat_w[:, self.foot_site_ids]
    rpy = torch.stack(euler_xyz_from_quat(quat.reshape(-1, 4)), dim=-1)
    return pos, rpy.reshape(self.num_envs, self.num_feet, 3)

  ##
  # Visualization.
  ##

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    step = self.step_index()
    steps = step.unsqueeze(1) + torch.arange(
      self.cfg.num_observations, device=self.device
    ).unsqueeze(0)
    # Recomputed here rather than reusing the cached command, so the markers
    # follow the support foot live even while the viewer is paused.
    targets = self._targets_world(step, self.cfg.num_observations).cpu().numpy()
    feet = self.foot_of_step(steps).cpu().numpy()
    latched = self.stance_target_w.cpu().numpy()
    half = self.cfg.viz.marker_size

    def draw(pose, rgb, alpha: float, filled: bool) -> None:
      x, y, z, _, _, yaw = pose
      cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
      mat = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]
      )
      thickness = 0.005 if filled else 0.001
      visualizer.add_box(
        np.array([x, y, z]),
        np.array([half[0], half[1], thickness]),
        mat,
        (*rgb, alpha),
      )
      visualizer.add_arrow(
        np.array([x, y, z + 0.01]),
        np.array([x + 0.12 * cos_yaw, y + 0.12 * sin_yaw, z + 0.01]),
        color=(*rgb, alpha),
        width=0.008,
      )

    for batch in env_indices:
      for i in range(self.cfg.num_observations):
        side = self.cfg.feet[feet[batch, i]].side
        rgb = _LEFT_COLOR if side == "left" else _RIGHT_COLOR
        # The nearest footstep is drawn solid, later ones fade out.
        alpha = 0.9 * (1.0 - 0.7 * i / max(self.cfg.num_observations - 1, 1))
        draw(targets[batch, i], rgb, alpha, filled=True)
      # The latched targets the planted feet are actually being scored against,
      # drawn flat so they read as "already judged" next to the live plan.
      for foot_id, foot in enumerate(self.cfg.feet):
        rgb = _LEFT_COLOR if foot.side == "left" else _RIGHT_COLOR
        draw(latched[batch, foot_id], rgb, 0.35, filled=False)


@dataclass
class PathLibCfg:
  """Which A* path library the path-derived footsteps are read off.

  Defaults match ``PathCommandCfg`` so the two share one cached library (and one
  copy in device memory) when both commands are configured.
  """

  pathlib_size: int = 50_000
  obstacle_world: ObstacleWorldCfg = field(default_factory=ObstacleWorldCfg)
  point_spacing: float = 0.01
  num_waypoints: int = 4
  max_points: int = 1500
  generation_seed: int = 0
  num_workers: int | None = None
  cache_dir: str | None = None


@dataclass(kw_only=True)
class FootstepCommandCfg(CommandTermCfg):
  entity_name: str = "robot"

  feet: tuple[FootCfg, ...] = (
    FootCfg(site_name="r_foot", gait_offset=0.0, side="right"),
    FootCfg(site_name="l_foot", gait_offset=0.5, side="left"),
  )
  """The two feet, in the same order (and with the same gait offsets) as the
  ``feet_phase`` / ``feet_gait`` rewards."""

  frequency: float = 1.75
  """Gait frequency (Hz). Must match the ``phase`` observation and the gait
  rewards: the footstep schedule is read off the same clock."""

  stance_threshold: float = 0.56
  """Leg phase below which a foot counts as planted, matching ``feet_phase``.
  The footstep rewards only score a foot while it is in stance."""

  num_observations: int = 4
  """How many upcoming footsteps the policy sees. Each contributes
  ``OBS_PER_FOOTSTEP`` (8) dimensions."""

  max_steps: int = 96
  """Footsteps planned per resample. Must cover the longest resampling interval
  at two footsteps per gait period; past the end the last footstep repeats,
  which reads as standing on it."""

  max_step_delta: tuple[float, float, float, float, float, float] = (
    0.30,
    0.14,
    0.0,
    0.0,
    0.0,
    0.45,
  )
  """Maximum reach of one footstep relative to the previous one, per axis:
  ``(x, y, z in meters, roll, pitch, yaw in radians)``. Enforced on both
  scenario families, so no sampled plan can ask for a step the robot has no
  chance of making. ``z``/``roll``/``pitch`` are held at 0 until terrain lands."""

  path_fraction: float = 0.5
  """Fraction of resamples whose footsteps are read off an A* path; the rest are
  sampled as Gaussian deltas. 0 disables the path library entirely (and skips
  loading it)."""

  path_step_length_range: tuple[float, float] = (0.10, 0.25)
  """Arc length between consecutive footsteps along a path, in meters. Sampled
  once per resample -- this is the env's walking cadence along that path."""

  nominal_delta_range: tuple[tuple[float, float], ...] = (
    (-0.10, 0.25),
    (-0.06, 0.06),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (-0.20, 0.20),
  )
  """Range the per-env nominal step ``mu`` is drawn from, per axis. ``mu`` near
  zero is walking in place; the ranges are what makes a Gaussian-mode env
  actually travel."""

  delta_std: tuple[float, float, float, float, float, float] = (
    0.05,
    0.03,
    0.0,
    0.0,
    0.0,
    0.10,
  )
  """Standard deviation of the per-footstep Gaussian jitter around ``mu``."""

  pathlib: PathLibCfg = field(default_factory=PathLibCfg)

  @dataclass
  class VizCfg:
    marker_size: tuple[float, float] = (0.06, 0.035)
    """Half extents of the footstep marker (roughly a foot's footprint)."""

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> FootstepCommand:
    return FootstepCommand(self, env)

  def __post_init__(self) -> None:
    if not 0.0 <= self.path_fraction <= 1.0:
      raise ValueError(f"path_fraction must be in [0, 1], got {self.path_fraction}.")
    for name in ("max_step_delta", "delta_std"):
      if len(getattr(self, name)) != FOOTSTEP_DIM:
        raise ValueError(f"{name} must have {FOOTSTEP_DIM} entries (x y z r p y).")
    if len(self.nominal_delta_range) != FOOTSTEP_DIM:
      raise ValueError(
        f"nominal_delta_range must have {FOOTSTEP_DIM} (low, high) pairs."
      )
    # Two footsteps per gait period: the plan has to outlast the resampling
    # interval or the robot spends the tail of it standing on the last footstep.
    needed = math.ceil(2.0 * self.frequency * self.resampling_time_range[1])
    if self.max_steps < needed:
      raise ValueError(
        f"max_steps={self.max_steps} is too short for a resampling interval of "
        f"{self.resampling_time_range[1]}s at {self.frequency} Hz: needs at "
        f"least {needed}."
      )
