"""Deterministic replay of the command book, plus the per-episode measurements.

Each class here subclasses the *training* command term and changes exactly two
things:

1. **Where the command comes from.** Every sampled quantity is replaced by the
   corresponding field of the command book row selected by
   :attr:`EvalMultiCommand.command_index`. Nothing else about the term changes,
   so the policy sees the identical observation it saw in training.
2. **What is measured.** ``eval_update`` accumulates per-episode statistics and
   ``eval_record`` reports them; :class:`EvalMultiCommand` latches the record
   the moment the episode ends and hands it to the runner.

The measurement lives on the command term rather than in a separate observer
because the term is what already holds the quantities that matter -- the goal
it placed, the path it chose, the footstep target it latched, the kick it
scored. An observer would have to re-derive all of it.

Index assignment
----------------
Env ``e`` runs command indices ``e, e + N, e + 2N, ...`` for ``N`` envs, so
which episode gets which command does not depend on the policy, on how quickly
an env falls over, or on the order envs happen to finish in. Two policies
evaluated at the same ``num_envs`` therefore see identical scenarios in
identical order; at a *different* ``num_envs`` they see the same *set* of
commands (indices 0..runs-1), just distributed over the envs differently.

One asymmetry is worth knowing about when reading a CSV: the first episode of
each pass accumulates one sample fewer than the rest (999 against 1000 over a
20 s episode). ``env.step`` resets a finished env *before* recomputing the
commands, so every episode but the first inherits its ``t=0`` sample from the
step that ended its predecessor. It is a 0.1 % weighting difference, it lands
on the same command indices for every policy, and it therefore does not affect
a comparison -- but it is why round 1 and round 2 report different
``tracked_steps`` for the same episode length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import wrap_to_pi

from mjlab_piplus.tasks.eval.command_book import CommandBook, footstep_deltas
from mjlab_piplus.tasks.mdp.commands import (
  GoalPoseCommand,
  GoalPoseCommandCfg,
  UniformVelocityCommandWithStop,
  UniformVelocityCommandWithStopCfg,
)
from mjlab_piplus.tasks.mdp.footstep.footstep_command import (
  FootstepCommand,
  FootstepCommandCfg,
)
from mjlab_piplus.tasks.mdp.kick import KickCommand, KickCommandCfg
from mjlab_piplus.tasks.mdp.multi_command import (
  MultiCommand,
  MultiCommandCfg,
  StandCommand,
  StandCommandCfg,
)
from mjlab_piplus.tasks.mdp.path_following.path_command import (
  PathCommand,
  PathCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

__all__ = [
  "EvalFootstepCommandCfg",
  "EvalGoalPoseCommandCfg",
  "EvalKickCommandCfg",
  "EvalMultiCommand",
  "EvalMultiCommandCfg",
  "EvalPathCommandCfg",
  "EvalStandCommandCfg",
  "EvalVelocityCommandCfg",
  "eval_time_out",
]


class _EvalMixin:
  """Book access and the per-episode step clock, shared by every eval term.

  ``command_index`` is the *same tensor object* the owning
  :class:`EvalMultiCommand` writes into, so a term always reads the current
  episode's row without any hand-off.
  """

  # Supplied by the CommandTerm this is always mixed into; declared so the
  # mixin's own type checks resolve.
  num_envs: int
  device: str
  _env: ManagerBasedRlEnv

  def attach_eval(self, book: CommandBook, command_index: torch.Tensor) -> None:
    self._book = book
    self._command_index = command_index
    self._book_cache: dict[str, torch.Tensor] = {}
    self._eval_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._eval_setup()

  def _eval_setup(self) -> None:
    """Per-term one-off setup, once the book is available."""

  @property
  def eval_type_name(self) -> str:
    raise NotImplementedError

  def _column(self, key: str) -> torch.Tensor:
    """One field of this term's book rows, as a device tensor."""
    cached = self._book_cache.get(key)
    if cached is None:
      values = self._book.column(self.eval_type_name, key)
      cached = torch.as_tensor(values, dtype=torch.float32, device=self.device)
      self._book_cache[key] = cached
    return cached

  def _row_index(self, env_ids: torch.Tensor) -> torch.Tensor:
    return self._command_index[env_ids] % self._book.num_commands

  def _value(self, env_ids: torch.Tensor, key: str) -> torch.Tensor:
    return self._column(key)[self._row_index(env_ids)]

  # The three hooks EvalMultiCommand drives.

  def eval_reset(self, env_ids: torch.Tensor) -> None:
    self._eval_steps[env_ids] = 0

  def eval_update(self) -> None:
    self._eval_steps += 1

  def eval_record(self) -> dict[str, torch.Tensor]:
    return {}

  @property
  def eval_elapsed_s(self) -> torch.Tensor:
    return self._eval_steps.float() * self._env.step_dt


def _mean(total: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
  """Mean that reports NaN rather than 0 where nothing was measured.

  A zero would read as a perfect score for an episode that never got to make
  the measurement at all (no footstep landed, no kick happened), which is the
  opposite of the truth.
  """
  return torch.where(count > 0, total / count.clamp(min=1), torch.nan)


##
# Velocity.
##


class EvalVelocityCommand(UniformVelocityCommandWithStop, _EvalMixin):
  cfg: EvalVelocityCommandCfg
  eval_type_name = "velocity"

  def _eval_setup(self) -> None:
    self._err = torch.zeros(self.num_envs, 3, device=self.device)
    self._achieved = torch.zeros(self.num_envs, 3, device=self.device)
    self._samples = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Replay the booked twist, with every behavioral draw switched off.

    The training term also produces heading-driven, world-frame, forward-only
    and walk-in-place envs. All of them are pinned off here: a tracking number
    is only readable when every episode asks the same thing, "hold this twist".
    """
    self.vel_command_b[env_ids, 0] = self._value(env_ids, "lin_vel_x")
    self.vel_command_b[env_ids, 1] = self._value(env_ids, "lin_vel_y")
    self.vel_command_b[env_ids, 2] = self._value(env_ids, "ang_vel_z")
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
    for flag in (
      self.is_heading_env,
      self.is_standing_env,
      self.is_walk_in_place_env,
      self.is_world_env,
      self.is_forward_env,
    ):
      flag[env_ids] = False
    self.stand_flag[env_ids] = 0.0

  def eval_reset(self, env_ids: torch.Tensor) -> None:
    super().eval_reset(env_ids)
    self._err[env_ids] = 0.0
    self._achieved[env_ids] = 0.0
    self._samples[env_ids] = 0.0

  def eval_update(self) -> None:
    lin = self.robot.data.root_link_lin_vel_b[:, :2]
    yaw = self.robot.data.root_link_ang_vel_b[:, 2]
    measured = torch.cat([lin, yaw.unsqueeze(1)], dim=1)
    # The first ``warmup_s`` are the robot accelerating into a command it was
    # handed at a standstill; scoring them measures the step response, not the
    # tracking accuracy the command asks about.
    counted = (self.eval_elapsed_s >= self.cfg.warmup_s).float().unsqueeze(1)
    # Absolute per step, then averaged: the signed error averages toward zero
    # over a gait cycle, and a robot that overshoots half the time and
    # undershoots the other half would score perfect.
    self._err += (measured - self.vel_command_b).abs() * counted
    # The signed velocity itself, which the error deliberately throws away.
    self._achieved += measured * counted
    self._samples += counted.squeeze(1)
    super().eval_update()

  def eval_record(self) -> dict[str, torch.Tensor]:
    """Both readings of the same episode, per axis.

    ``err_*`` is the accuracy: absolute per step, so it cannot cancel. But
    taking the absolute value throws away the *sign*, which is what separates a
    policy that consistently undershoots from one that jitters around the right
    value -- so the signed ``mean_*`` is reported beside it. Read ``err_*`` for
    how well the command was followed and ``mean_*`` against the command for
    whether the miss has a direction.
    """
    n = self._samples
    err = _mean(self._err, n.unsqueeze(1))
    achieved = _mean(self._achieved, n.unsqueeze(1))
    cmd = self.vel_command_b
    return {
      "cmd_vel_x": cmd[:, 0],
      "cmd_vel_y": cmd[:, 1],
      "cmd_yaw_rate": cmd[:, 2],
      "err_vel_x": err[:, 0],
      "err_vel_y": err[:, 1],
      "err_vel_xy": torch.linalg.norm(err[:, :2], dim=1),
      "err_yaw_rate": err[:, 2],
      "mean_vel_x": achieved[:, 0],
      "mean_vel_y": achieved[:, 1],
      "mean_yaw_rate": achieved[:, 2],
      "tracked_steps": n,
    }


@dataclass(kw_only=True)
class EvalVelocityCommandCfg(UniformVelocityCommandWithStopCfg):
  warmup_s: float = 1.0
  """Seconds at the start of the episode excluded from the tracking error: the
  robot is accelerating from a standstill into a command it was just handed."""

  def build(self, env: ManagerBasedRlEnv) -> EvalVelocityCommand:
    return EvalVelocityCommand(self, env)


##
# Goal pose.
##


class EvalGoalPoseCommand(GoalPoseCommand, _EvalMixin):
  cfg: EvalGoalPoseCommandCfg
  eval_type_name = "goal_pose"

  def _eval_setup(self) -> None:
    self._initial_dist = torch.zeros(self.num_envs, device=self.device)
    self._min_dist = torch.zeros(self.num_envs, device=self.device)
    self._reach_step = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self._pos_step = torch.full_like(self._reach_step, -1)
    self._bearing = torch.zeros(self.num_envs, device=self.device)
    self._yaw_offset = torch.zeros(self.num_envs, device=self.device)
    self._distance = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Book the goal as a polar offset and a yaw, both relative to the heading.

    Relative rather than absolute so the scenario is independent of the spawn
    yaw: an absolute goal yaw would make the episode's difficulty a function of
    which way the robot happened to be facing when it was reset.
    """
    self._distance[env_ids] = self._value(env_ids, "distance")
    self._bearing[env_ids] = self._value(env_ids, "bearing")
    self._yaw_offset[env_ids] = self._value(env_ids, "goal_yaw")
    self.is_standing_env[env_ids] = False
    self._pending_place[env_ids] = True

  def _update_command(self) -> None:
    # The parent anchors ``_pending_offset`` on the robot in world coordinates,
    # and only here is the robot pose current (see GoalPoseCommand._resample_
    # command's note on the reset ordering), so the polar offset is rotated
    # into the world by the live heading just before the parent consumes it.
    pending = self._pending_place.nonzero().flatten()
    if len(pending) > 0:
      heading = self.robot.data.heading_w[pending]
      angle = heading + self._bearing[pending]
      distance = self._distance[pending]
      self._pending_offset[pending] = torch.stack(
        [distance * torch.cos(angle), distance * torch.sin(angle)], dim=1
      )
      self.goal_yaw_w[pending] = wrap_to_pi(heading + self._yaw_offset[pending])
    super()._update_command()

  def eval_reset(self, env_ids: torch.Tensor) -> None:
    super().eval_reset(env_ids)
    self._initial_dist[env_ids] = 0.0
    self._min_dist[env_ids] = math.inf
    self._reach_step[env_ids] = -1
    self._pos_step[env_ids] = -1

  def eval_update(self) -> None:
    dist = torch.linalg.norm(self.goal_command_b[:, :2], dim=1)
    yaw_err = self.yaw_err.abs()
    first = self._eval_steps == 0
    self._initial_dist = torch.where(first, dist, self._initial_dist)
    self._min_dist = torch.minimum(self._min_dist, dist)
    at_pos = dist < self.cfg.goal_pos_threshold
    at_pose = at_pos & (yaw_err < self.cfg.goal_yaw_threshold)
    # First arrival only: a robot that reaches the goal, drifts off and comes
    # back should report the time it first got there.
    self._pos_step = torch.where(
      at_pos & (self._pos_step < 0), self._eval_steps, self._pos_step
    )
    self._reach_step = torch.where(
      at_pose & (self._reach_step < 0), self._eval_steps, self._reach_step
    )
    super().eval_update()

  def eval_record(self) -> dict[str, torch.Tensor]:
    dt = self._env.step_dt
    nan = torch.tensor(torch.nan, device=self.device)
    reached = self._reach_step >= 0
    at_pos = self._pos_step >= 0
    return {
      "initial_dist": self._initial_dist,
      "final_dist": torch.linalg.norm(self.goal_command_b[:, :2], dim=1),
      "final_yaw_err": self.yaw_err.abs(),
      "min_dist": self._min_dist,
      "time_to_goal": torch.where(reached, self._reach_step.float() * dt, nan),
      "time_to_position": torch.where(at_pos, self._pos_step.float() * dt, nan),
      "reached_goal": reached.float(),
      "reached_position": at_pos.float(),
    }


@dataclass(kw_only=True)
class EvalGoalPoseCommandCfg(GoalPoseCommandCfg):
  goal_pos_threshold: float = 0.3
  """Distance (m) to the goal that counts as arrived."""
  goal_yaw_threshold: float = 0.3
  """Additional yaw error (rad) required for the full goal *pose*."""

  def build(self, env: ManagerBasedRlEnv) -> EvalGoalPoseCommand:
    return EvalGoalPoseCommand(self, env)


##
# Path.
##


class EvalPathCommand(PathCommand, _EvalMixin):
  cfg: EvalPathCommandCfg
  eval_type_name = "path"

  def _eval_setup(self) -> None:
    if self._book.meta.get("pathlib_size") != self.cfg.pathlib_size:
      raise ValueError(
        "The command book's path indices were drawn against a library of "
        f"{self._book.meta.get('pathlib_size')} paths, but this env loads "
        f"{self.cfg.pathlib_size}. The i-th path of one library is not the "
        "i-th path of the other; regenerate the book or match the size."
      )
    self._dist_sum = torch.zeros(self.num_envs, device=self.device)
    self._dist_max = torch.zeros(self.num_envs, device=self.device)
    self._dist_now = torch.zeros(self.num_envs, device=self.device)
    self._goal_step = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self._initial_dist = torch.zeros(self.num_envs, device=self.device)
    self._min_goal_dist = torch.zeros(self.num_envs, device=self.device)
    self._yaw_offset = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Replay one booked path. Deliberately does not call ``super()``.

    The parent's version samples a random path index and a random goal yaw, and
    carries the older ``eval_mode`` CSV logging; none of that applies here.
    """
    indices = self._value(env_ids, "path_index").long()
    self.paths[env_ids] = self.all_paths[indices]
    self.waypoints[env_ids] = self.all_waypoints[indices]
    self.obstacles[env_ids] = self.all_obstacles[indices]
    self.at_goal_time[env_ids] = 0.0
    self._yaw_offset[env_ids] = self._value(env_ids, "goal_yaw")
    if self.align:
      self._pending_align[env_ids] = True

  def _update_command(self) -> None:
    # Relative to the live heading, for the same reason as the goal-pose
    # command: the spawn yaw must not decide the difficulty.
    pending = self._pending_align.nonzero().flatten()
    if len(pending) > 0:
      self.goal_yaw_w[pending] = wrap_to_pi(
        self.robot.data.heading_w[pending] + self._yaw_offset[pending]
      )
    super()._update_command()
    # Cross-track distance: the closest point of the reference path. The tail of
    # a stored path repeats its goal point, so the padding cannot pull this
    # below the true distance.
    root_xy = self.current_pose[:, :2]
    self._dist_now = torch.min(
      torch.linalg.norm(self.paths[..., :2] - root_xy.unsqueeze(1), dim=-1), dim=-1
    ).values

  def eval_reset(self, env_ids: torch.Tensor) -> None:
    super().eval_reset(env_ids)
    self._dist_sum[env_ids] = 0.0
    self._dist_max[env_ids] = 0.0
    self._goal_step[env_ids] = -1
    self._initial_dist[env_ids] = 0.0
    self._min_goal_dist[env_ids] = math.inf

  def eval_update(self) -> None:
    self._dist_sum += self._dist_now
    self._dist_max = torch.maximum(self._dist_max, self._dist_now)
    first = self._eval_steps == 0
    self._initial_dist = torch.where(first, self.dist_to_goal, self._initial_dist)
    self._min_goal_dist = torch.minimum(self._min_goal_dist, self.dist_to_goal)
    at_goal = self.dist_to_goal < self.cfg.goal_pos_threshold
    self._goal_step = torch.where(
      at_goal & (self._goal_step < 0), self._eval_steps, self._goal_step
    )
    super().eval_update()

  def eval_record(self) -> dict[str, torch.Tensor]:
    dt = self._env.step_dt
    nan = torch.tensor(torch.nan, device=self.device)
    reached = self._goal_step >= 0
    steps = self._eval_steps.clamp(min=1).float()
    return {
      "mean_path_dist": self._dist_sum / steps,
      "max_path_dist": self._dist_max,
      "initial_dist_to_goal": self._initial_dist,
      "final_dist_to_goal": self.dist_to_goal,
      "min_dist_to_goal": self._min_goal_dist,
      "final_yaw_err": self.yaw_err.abs(),
      "time_to_goal": torch.where(reached, self._goal_step.float() * dt, nan),
      "reached_goal": reached.float(),
    }


@dataclass(kw_only=True)
class EvalPathCommandCfg(PathCommandCfg):
  goal_pos_threshold: float = 0.3
  """Distance (m) to the final waypoint that counts as arrived."""

  def build(self, env: ManagerBasedRlEnv) -> EvalPathCommand:
    return EvalPathCommand(self, env)


##
# Footstep.
##


class EvalFootstepCommand(FootstepCommand, _EvalMixin):
  cfg: EvalFootstepCommandCfg
  eval_type_name = "footstep"

  def _eval_setup(self) -> None:
    rows = self._book.rows("footstep")
    max_steps = self.cfg.max_steps
    deltas = np.zeros((len(rows), max_steps, 6), dtype=np.float32)
    path_rows = [i for i, row in enumerate(rows) if row["mode"] == "path"]
    for i, row in enumerate(rows):
      gaussian = footstep_deltas(row, max_steps=max_steps, delta_std=self.cfg.delta_std)
      if gaussian is not None:
        deltas[i] = gaussian
    book_deltas = torch.as_tensor(deltas, device=self.device)

    if path_rows:
      if self._book.meta.get("pathlib_size") != self.cfg.pathlib.pathlib_size:
        raise ValueError(
          "The command book's footstep path indices were drawn against a "
          f"library of {self._book.meta.get('pathlib_size')} paths, but this "
          f"env loads {self.cfg.pathlib.pathlib_size}."
        )
      idx = torch.tensor(
        [rows[i]["path_index"] for i in path_rows],
        dtype=torch.long,
        device=self.device,
      )
      step_length = torch.tensor(
        [[rows[i]["step_length"]] for i in path_rows], device=self.device
      )
      poses = self._path_poses(idx, step_length)
      book_deltas[path_rows] = self._to_deltas(poses)

    # The same clamp the training sampler applies, so no booked plan can ask
    # for a step outside the configured reach either.
    limit = torch.tensor(self.cfg.max_step_delta, device=self.device)
    self._book_deltas = book_deltas.clamp(-limit, limit)
    self._book_height = torch.cumsum(self._book_deltas[..., 2:5], dim=1)

    self._err_xy = torch.zeros(self.num_envs, device=self.device)
    self._err_z = torch.zeros(self.num_envs, device=self.device)
    self._err_yaw = torch.zeros(self.num_envs, device=self.device)
    self._err_xy_max = torch.zeros(self.num_envs, device=self.device)
    self._landings = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    rows = self._row_index(env_ids)
    self.deltas[env_ids] = self._book_deltas[rows]
    self.plan_height[env_ids] = self._book_height[rows]
    self._pending_start[env_ids] = True

  def _update_command(self) -> None:
    # ``_prev_step`` is the footstep the support foot is standing on; the parent
    # advances it and latches that footstep's target the step it lands. Reading
    # it either side of the call is what turns "a footstep landed" into an
    # event, and the latched target is what the landing is scored against --
    # the live targets are re-grounded on the support foot every step, so
    # scoring against those would call every landing perfect.
    prev_step = self._prev_step.clone()
    super()._update_command()
    landed = self._prev_step > prev_step
    if not bool(landed.any()):
      return
    foot = self.foot_of_step(prev_step)
    pos, rpy = self.foot_pose_w()
    rows = torch.arange(self.num_envs, device=self.device)
    target = self.stance_target_w[rows, foot]
    err = pos[rows, foot] - target[:, :3]
    err_xy = torch.linalg.norm(err[:, :2], dim=1)
    err_yaw = wrap_to_pi(rpy[rows, foot, 2] - target[:, 5]).abs()
    counted = landed.float()
    self._err_xy += err_xy * counted
    self._err_z += err[:, 2].abs() * counted
    self._err_yaw += err_yaw * counted
    self._err_xy_max = torch.maximum(
      self._err_xy_max, torch.where(landed, err_xy, torch.zeros_like(err_xy))
    )
    self._landings += counted

  def eval_reset(self, env_ids: torch.Tensor) -> None:
    super().eval_reset(env_ids)
    self._err_xy[env_ids] = 0.0
    self._err_z[env_ids] = 0.0
    self._err_yaw[env_ids] = 0.0
    self._err_xy_max[env_ids] = 0.0
    self._landings[env_ids] = 0.0

  def eval_record(self) -> dict[str, torch.Tensor]:
    return {
      "step_err_xy": _mean(self._err_xy, self._landings),
      "step_err_z": _mean(self._err_z, self._landings),
      "step_err_yaw": _mean(self._err_yaw, self._landings),
      "step_err_xy_max": torch.where(self._landings > 0, self._err_xy_max, torch.nan),
      "footsteps": self._landings,
    }


@dataclass(kw_only=True)
class EvalFootstepCommandCfg(FootstepCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> EvalFootstepCommand:
    return EvalFootstepCommand(self, env)


##
# Kick.
##


class EvalKickCommand(KickCommand, _EvalMixin):
  cfg: EvalKickCommandCfg
  eval_type_name = "kick"

  def _eval_setup(self) -> None:
    self._touch_step = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self._kicks = torch.zeros(self.num_envs, device=self.device)
    self._first = {
      name: torch.full((self.num_envs,), torch.nan, device=self.device)
      for name in ("speed", "speed_error", "aim_error", "style", "right_foot")
    }

  def _begin_ball_life(self, env_ids: torch.Tensor) -> None:
    """Place the ball where the book says, in the robot's own frame.

    Every ball life in an episode replays the same row, so a second kick is the
    same task as the first rather than a fresh random one.
    """
    super()._begin_ball_life(env_ids)
    if len(env_ids) == 0:
      return
    distance = self._value(env_ids, "distance")
    bearing = self._value(env_ids, "bearing")
    self._pending_forward[env_ids] = distance * torch.cos(bearing)
    self._pending_side[env_ids] = distance * torch.sin(bearing)

  def _apply_pending_placement(self) -> None:
    # The parent draws direction and strength here (not at sample time) because
    # the direction is a cone around the heading *at placement*; the booked
    # values are relative to that same heading, so they are written straight
    # after, once the parent has read the pose.
    env_ids = self._pending_place.nonzero().flatten()
    super()._apply_pending_placement()
    if len(env_ids) == 0:
      return
    heading = self.robot.data.heading_w[env_ids]
    self.kick_dir_w[env_ids] = wrap_to_pi(heading + self._value(env_ids, "direction"))
    self.strength[env_ids] = self._value(env_ids, "strength")

  def _score_finished_kicks(self, done: torch.Tensor) -> None:
    super()._score_finished_kicks(done)
    ball_dir = torch.atan2(self.peak_vel_w[:, 1], self.peak_vel_w[:, 0])
    values = {
      "speed": self.peak_speed,
      "speed_error": self.peak_speed - self.strength,
      "aim_error": wrap_to_pi(ball_dir - self.kick_dir_w).abs(),
      "style": self.style,
      "right_foot": self.foot_correct.float(),
    }
    # Only the first kick of the episode is reported: the ball comes straight
    # back after a kick, so a later one is scored from a different starting
    # pose than the booked one.
    first = done & (self._kicks == 0)
    for name, value in values.items():
      self._first[name] = torch.where(first, value, self._first[name])
    self._kicks += done.float()

  def eval_reset(self, env_ids: torch.Tensor) -> None:
    super().eval_reset(env_ids)
    self._touch_step[env_ids] = -1
    self._kicks[env_ids] = 0.0
    for value in self._first.values():
      value[env_ids] = torch.nan

  def eval_update(self) -> None:
    self._touch_step = torch.where(
      self.touched & (self._touch_step < 0), self._eval_steps, self._touch_step
    )
    super().eval_update()

  def eval_record(self) -> dict[str, torch.Tensor]:
    dt = self._env.step_dt
    touched = self._touch_step >= 0
    kicked = self._kicks > 0
    return {
      "cmd_strength": self.strength,
      "kick_speed": self._first["speed"],
      "kick_speed_error": self._first["speed_error"],
      "kick_aim_error": self._first["aim_error"],
      "kick_style": self._first["style"],
      "kick_right_foot": self._first["right_foot"],
      "time_to_contact": torch.where(touched, self._touch_step.float() * dt, torch.nan),
      "kicked": kicked.float(),
      "kicks": self._kicks,
      "final_ball_dist": torch.linalg.norm(self.ball_pos_b, dim=1),
    }


@dataclass(kw_only=True)
class EvalKickCommandCfg(KickCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> EvalKickCommand:
    return EvalKickCommand(self, env)


##
# Stand.
##


class EvalStandCommand(StandCommand, _EvalMixin):
  cfg: EvalStandCommandCfg
  eval_type_name = "stand"

  def __init__(self, cfg: EvalStandCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

  def _eval_setup(self) -> None:
    self._start_xy = torch.zeros(self.num_envs, 2, device=self.device)
    self._start_yaw = torch.zeros(self.num_envs, device=self.device)
    self._drift_max = torch.zeros(self.num_envs, device=self.device)
    self._posture_sum = torch.zeros(self.num_envs, device=self.device)

  def eval_reset(self, env_ids: torch.Tensor) -> None:
    super().eval_reset(env_ids)
    self._drift_max[env_ids] = 0.0
    self._posture_sum[env_ids] = 0.0

  def eval_update(self) -> None:
    first = self._eval_steps == 0
    self._start_xy = torch.where(
      first.unsqueeze(1), self.robot.data.root_link_pos_w[:, :2], self._start_xy
    )
    self._start_yaw = torch.where(first, self.robot.data.heading_w, self._start_yaw)
    drift = torch.linalg.norm(
      self.robot.data.root_link_pos_w[:, :2] - self._start_xy, dim=1
    )
    self._drift_max = torch.maximum(self._drift_max, drift)
    deviation = self.robot.data.joint_pos - self.robot.data.default_joint_pos
    self._posture_sum += deviation.pow(2).mean(dim=1).sqrt()
    super().eval_update()

  def eval_record(self) -> dict[str, torch.Tensor]:
    steps = self._eval_steps.clamp(min=1).float()
    return {
      "drift": torch.linalg.norm(
        self.robot.data.root_link_pos_w[:, :2] - self._start_xy, dim=1
      ),
      "max_drift": self._drift_max,
      "yaw_drift": wrap_to_pi(self.robot.data.heading_w - self._start_yaw).abs(),
      "posture_rms": self._posture_sum / steps,
    }


@dataclass(kw_only=True)
class EvalStandCommandCfg(StandCommandCfg):
  entity_name: str = "robot"
  """The robot whose drift and posture the stand episode is scored on."""

  def build(self, env: ManagerBasedRlEnv) -> EvalStandCommand:
    return EvalStandCommand(self, env)


##
# The composite.
##


class EvalMultiCommand(MultiCommand):
  """A :class:`MultiCommand` that replays the book and never switches type.

  One command type is active for a whole episode (``forced_type``, or the
  book's round-robin schedule when it is None), the sub-command is resampled
  only at an episode reset, and every episode's parameters come from the book
  row its ``command_index`` names. What the policy sees is unchanged -- the
  one-hot and the zeroed inactive blocks are exactly as in training -- so a
  policy cannot tell it is being evaluated from its observation alone.
  """

  cfg: EvalMultiCommandCfg

  def __init__(self, cfg: EvalMultiCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.book = CommandBook.load(cfg.book_path)

    self.schedule = tuple(cfg.type_schedule or self.term_names)
    unknown = [name for name in self.schedule if name not in self.terms]
    if unknown:
      raise KeyError(
        f"type_schedule names {unknown}, which are not sub-commands. "
        f"Configured: {list(self.term_names)}."
      )
    self._forced_type: str | None = cfg.forced_type
    if self._forced_type is not None and self._forced_type not in self.terms:
      raise KeyError(
        f"forced_type='{self._forced_type}' is not a sub-command. "
        f"Configured: {list(self.term_names)}."
      )

    # Env ``e`` runs commands e, e+N, e+2N, ...; see the module docstring.
    self._env_offset = torch.arange(self.num_envs, device=self.device)
    self.command_index = self._env_offset.clone()
    self.episode_index = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self.episode_limit = torch.full(
      (self.num_envs,),
      self._limit_steps(self.term_names[0]),
      dtype=torch.long,
      device=self.device,
    )

    missing = [name for name in self.terms if name not in self.book.commands]
    if missing:
      raise ValueError(
        f"The command book at {cfg.book_path} has no rows for {missing}. "
        "Regenerate it after adding a command type."
      )

    for name, term in self.terms.items():
      if not isinstance(term, _EvalMixin):
        raise TypeError(
          f"Sub-command '{name}' is a {type(term).__name__}, which does not "
          "replay the command book. Every sub-command of an EvalMultiCommand "
          "must be one of the Eval* terms in eval_commands.py."
        )
      term.attach_eval(self.book, self.command_index)

    # Records of episodes that have ended, drained by the eval runner. Kept as
    # already-transferred CPU tensors: one transfer per batch of finished envs
    # rather than one per value.
    self.finished: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
    self._suppress_latch = False
    # Off until a runner turns it on, so playing this env in the viewer does
    # not accumulate a record per episode with nothing ever draining them.
    self.collect_records = False

  ##
  # Type selection.
  ##

  @property
  def forced_type(self) -> str | None:
    return self._forced_type

  def set_forced_type(self, name: str | None) -> None:
    """Pin every env to one command type (None restores the book schedule).

    Takes effect at the next episode reset, so the runner resets all envs after
    calling it. Changing the type in place is what lets one env instance
    evaluate every type in turn: rebuilding the sim per type costs far more
    than the evaluation itself.
    """
    if name is not None and name not in self.terms:
      raise KeyError(
        f"'{name}' is not a sub-command. Configured: {list(self.term_names)}."
      )
    self._forced_type = name

  def _limit_steps(self, type_name: str) -> int:
    seconds = self.cfg.episode_length_s.get(
      type_name, self.cfg.default_episode_length_s
    )
    return max(1, round(seconds / self._env.step_dt))

  def _sample_type(self, env_ids: torch.Tensor) -> None:
    """Deterministic type: the forced one, or the book's round-robin."""
    if self._forced_type is not None:
      chosen = torch.full_like(env_ids, self._type_index[self._forced_type])
    else:
      order = torch.tensor(
        [self._type_index[name] for name in self.schedule], device=self.device
      )
      chosen = order[self.command_index[env_ids] % len(self.schedule)]
    self.active_type[env_ids] = chosen
    self._redraw_from_fractions[env_ids] = False

  ##
  # Episode lifecycle.
  ##

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    self._latch(env_ids)
    self.episode_index[env_ids] += 1
    # In place: the sub-terms hold a reference to this very tensor.
    self.command_index[env_ids] = (
      self._env_offset[env_ids] + self.episode_index[env_ids] * self.num_envs
    )
    extras = super().reset(env_ids)
    for i, name in enumerate(self.term_names):
      selected = env_ids[self.active_type[env_ids] == i]
      if len(selected) > 0:
        self.episode_limit[selected] = self._limit_steps(name)
        term = self.terms[name]
        assert isinstance(term, _EvalMixin)
        term.eval_reset(selected)
    return extras

  def restart(self) -> None:
    """Rewind the command counter so the next episodes replay rows 0, 1, 2, ...

    Called by the runner between command types, so each type is evaluated on
    the same slice of its own book. The reset that follows cuts every running
    episode off mid-way; those are not results, so the next latch is skipped.
    """
    self.episode_index[:] = -1
    self.finished.clear()
    self._suppress_latch = True

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if dt <= 0.0:
      # ``env.reset()`` computes the commands once with dt=0 so the post-reset
      # observation is current. That is not a step of the episode: counting it
      # would give every env a one-step "episode" that the next reset latches
      # as a result.
      return
    for i, name in enumerate(self.term_names):
      term = self.terms[name]
      assert isinstance(term, _EvalMixin)
      if bool((self.active_type == i).any()):
        term.eval_update()

  def _latch(self, env_ids: torch.Tensor) -> None:
    """Freeze the finished episodes' measurements before the reset wipes them.

    Runs from ``reset``, which the env calls *after* the termination manager
    and *before* the sub-commands are resampled, so the statistics still
    describe the episode that just ended.
    """
    if self._suppress_latch:
      self._suppress_latch = False
      return
    if not self.collect_records:
      return
    for i, name in enumerate(self.term_names):
      selected = env_ids[self.active_type[env_ids] == i]
      if len(selected) == 0:
        continue
      term = self.terms[name]
      assert isinstance(term, _EvalMixin)
      # An env that has not stepped yet has nothing to report: this is the
      # initial reset, not the end of an episode.
      selected = selected[term._eval_steps[selected] > 0]
      if len(selected) == 0:
        continue
      record = {key: value[selected] for key, value in term.eval_record().items()}
      steps = term._eval_steps[selected]
      terminated = self._env.termination_manager.terminated[selected]
      record["command_index"] = self.command_index[selected].float()
      record["steps"] = steps.float()
      record["duration"] = steps.float() * self._env.step_dt
      record["fell"] = terminated.float()
      record["timed_out"] = (~terminated).float()
      self.finished.append(
        (
          torch.full((len(selected),), i, dtype=torch.long).cpu(),
          {key: value.detach().cpu() for key, value in record.items()},
        )
      )

  def drain(self) -> list[tuple[str, dict[str, torch.Tensor]]]:
    """Take the finished-episode records accumulated since the last call."""
    batches = [
      (self.term_names[int(type_index[0])], record)
      for type_index, record in self.finished
    ]
    self.finished.clear()
    return batches


@dataclass(kw_only=True)
class EvalMultiCommandCfg(MultiCommandCfg):
  book_path: str = "eval_commands.json"
  """The command book to replay."""

  forced_type: str | None = None
  """Pin every env to this command type. None runs the ``type_schedule``
  round-robin by command index, which is what makes the registered eval task
  show all six types in the viewer."""

  type_schedule: tuple[str, ...] | None = None
  """Type order used when ``forced_type`` is None. Defaults to the sub-command
  order."""

  episode_length_s: dict[str, float] = field(default_factory=dict)
  """Per-type episode length. A goal 8 m away and a velocity command need very
  different amounts of time, and padding every type out to the longest one
  would waste most of the evaluation."""

  default_episode_length_s: float = 20.0
  """Episode length for a type with no ``episode_length_s`` entry."""

  fractions: dict[str, float] = field(default_factory=dict)
  """Unused: the type comes from ``forced_type`` / ``type_schedule``. Kept
  non-empty only to satisfy ``MultiCommandCfg`` validation."""

  def build(self, env: ManagerBasedRlEnv) -> EvalMultiCommand:
    return EvalMultiCommand(self, env)

  def __post_init__(self) -> None:
    if not self.fractions:
      self.fractions = {name: 1.0 for name in self.terms}
    super().__post_init__()
    unknown = set(self.episode_length_s) - set(self.terms)
    if unknown:
      raise KeyError(
        f"episode_length_s names {sorted(unknown)}, which are not sub-commands."
      )


def eval_time_out(env: ManagerBasedRlEnv, command_name: str = "command"):
  """Time out at the *active command type's* own episode length.

  ``mdp.time_out`` compares against one env-wide maximum, which would force
  every type onto the longest type's clock.
  """
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, EvalMultiCommand), (
    f"eval_time_out expects '{command_name}' to be an EvalMultiCommand, got "
    f"{type(term).__name__}."
  )
  return env.episode_length_buf >= term.episode_limit
