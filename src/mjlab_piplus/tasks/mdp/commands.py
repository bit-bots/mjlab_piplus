"""Goal-pose command for the PiPlus go-to-pose task.

Samples a 2D target position (relative to the robot's current base position,
at a random distance/bearing) plus a random target yaw. The command exposed to
the policy is the goal expressed in the robot base frame:

    [dx_b, dy_b, yaw_err]

where ``(dx_b, dy_b)`` is the goal position in the base frame and ``yaw_err`` is
the wrapped yaw difference to the goal orientation. This 3-vector mirrors with
signs ``[1, -1, -1]`` under a left-right reflection, matching the velocity
command layout used by ``symmetry.py`` (so the actor obs layout is unchanged).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class GoalPoseCommand(CommandTerm):
  cfg: GoalPoseCommandCfg

  def __init__(self, cfg: GoalPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]

    # Goal expressed in the world frame.
    self.goal_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.goal_yaw_w = torch.zeros(self.num_envs, device=self.device)

    # Goal expressed in the robot base frame: [dx_b, dy_b, yaw_err].
    self.goal_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    self.is_standing_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.goal_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_pos"] += (
      torch.norm(self.goal_command_b[:, :2], dim=1) / max_command_step
    )
    self.metrics["error_yaw"] += torch.abs(self.goal_command_b[:, 2]) / max_command_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)

    base_pos_w = self.robot.data.root_link_pos_w[env_ids, :2]

    radius = r.uniform_(*self.cfg.radius_range)
    bearing = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
    offset = torch.stack(
      [radius * torch.cos(bearing), radius * torch.sin(bearing)], dim=1
    )
    self.goal_pos_w[env_ids] = base_pos_w + offset
    self.goal_yaw_w[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.yaw_range
    )

    # Standing envs: goal == current pose so the robot learns to hold still.
    is_standing = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    self.is_standing_env[env_ids] = is_standing
    stand_ids = env_ids[is_standing]
    if len(stand_ids) > 0:
      self.goal_pos_w[stand_ids] = self.robot.data.root_link_pos_w[stand_ids, :2]
      self.goal_yaw_w[stand_ids] = self.robot.data.heading_w[stand_ids]

  def _update_command(self) -> None:
    base_pos_w = self.robot.data.root_link_pos_w[:, :2]
    heading_w = self.robot.data.heading_w

    delta_w = self.goal_pos_w - base_pos_w  # [N, 2]
    cos_h = torch.cos(heading_w)
    sin_h = torch.sin(heading_w)
    # Rotate world-frame delta into the base frame (rotation by -heading).
    self.goal_command_b[:, 0] = cos_h * delta_w[:, 0] + sin_h * delta_w[:, 1]
    self.goal_command_b[:, 1] = -sin_h * delta_w[:, 0] + cos_h * delta_w[:, 1]
    self.goal_command_b[:, 2] = wrap_to_pi(self.goal_yaw_w - heading_w)

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    goal_pos_ws = self.goal_pos_w.cpu().numpy()
    goal_yaw_ws = self.goal_yaw_w.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()

    arrow_len = self.cfg.viz.arrow_length
    z = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      gx, gy = goal_pos_ws[batch]
      gyaw = goal_yaw_ws[batch]
      goal_xyz = np.array([gx, gy, z])

      # Line from robot to goal (yellow).
      robot_xyz = np.array([base_pos_w[0], base_pos_w[1], z])
      visualizer.add_arrow(robot_xyz, goal_xyz, color=(0.9, 0.9, 0.1, 0.5), width=0.01)

      # Goal orientation arrow (red).
      heading_to = goal_xyz + np.array(
        [arrow_len * math.cos(gyaw), arrow_len * math.sin(gyaw), 0.0]
      )
      visualizer.add_arrow(goal_xyz, heading_to, color=(0.9, 0.1, 0.1, 0.9), width=0.02)


@dataclass(kw_only=True)
class GoalPoseCommandCfg(CommandTermCfg):
  entity_name: str
  radius_range: tuple[float, float] = (0.5, 3.0)
  """Distance range (meters) from the robot's current base position at which the
  goal position is sampled."""
  yaw_range: tuple[float, float] = (-math.pi, math.pi)
  """Goal yaw sampling range (radians)."""
  rel_standing_envs: float = 0.0
  """Fraction of environments whose goal is set to the current pose (stand)."""

  @dataclass
  class VizCfg:
    z_offset: float = 0.05
    arrow_length: float = 0.3

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> GoalPoseCommand:
    return GoalPoseCommand(self, env)


def _rotate_xy_w_to_b(xy_w: torch.Tensor, heading_w: torch.Tensor) -> torch.Tensor:
  cos_h = torch.cos(heading_w)
  sin_h = torch.sin(heading_w)
  xy_b = torch.empty_like(xy_w)
  xy_b[:, 0] = cos_h * xy_w[:, 0] + sin_h * xy_w[:, 1]
  xy_b[:, 1] = -sin_h * xy_w[:, 0] + cos_h * xy_w[:, 1]
  return xy_b


def _rotate_xy_b_to_w(xy_b: torch.Tensor, heading_w: torch.Tensor) -> torch.Tensor:
  cos_h = torch.cos(heading_w)
  sin_h = torch.sin(heading_w)
  xy_w = torch.empty_like(xy_b)
  xy_w[:, 0] = cos_h * xy_b[:, 0] - sin_h * xy_b[:, 1]
  xy_w[:, 1] = sin_h * xy_b[:, 0] + cos_h * xy_b[:, 1]
  return xy_w


class PassKickCommand(CommandTerm):
  cfg: PassKickCommandCfg

  def __init__(self, cfg: PassKickCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.robot_name]
    self.ball: Entity = env.scene[cfg.ball_name]

    self.target_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.target_radius = torch.zeros(self.num_envs, device=self.device)
    self.target_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    self.metrics["ball_to_target_dist"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["target_radius"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_command_b

  def _update_metrics(self) -> None:
    ball_pos_w = self.ball.data.root_link_pos_w[:, :2]
    self.metrics["ball_to_target_dist"] += torch.linalg.norm(
      self.target_pos_w - ball_pos_w, dim=1
    )
    self.metrics["target_radius"] += self.target_radius

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    target_distance = torch.empty(n, device=self.device).uniform_(
      *self.cfg.target_distance_range
    )
    target_bearing = torch.empty(n, device=self.device).uniform_(
      *self.cfg.target_bearing_range
    )
    target_offset_b = torch.stack(
      [
        target_distance * torch.cos(target_bearing),
        target_distance * torch.sin(target_bearing),
      ],
      dim=1,
    )
    target_offset_w = _rotate_xy_b_to_w(
      target_offset_b, self.robot.data.heading_w[env_ids]
    )
    self.target_pos_w[env_ids] = (
      self.ball.data.root_link_pos_w[env_ids, :2] + target_offset_w
    )
    self.target_radius[env_ids] = torch.empty(n, device=self.device).uniform_(
      *self.cfg.target_radius_range
    )

  def _update_command(self) -> None:
    target_from_ball_w = self.target_pos_w - self.ball.data.root_link_pos_w[:, :2]
    target_from_ball_b = _rotate_xy_w_to_b(
      target_from_ball_w, self.robot.data.heading_w
    )
    self.target_command_b[:, :2] = target_from_ball_b
    self.target_command_b[:, 2] = self.target_radius

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    target_pos_ws = self.target_pos_w.cpu().numpy()
    ball_pos_ws = self.ball.data.root_link_pos_w.cpu().numpy()
    z = self.cfg.viz.z_offset

    for batch in env_indices:
      ball_pos_w = ball_pos_ws[batch]
      if np.linalg.norm(ball_pos_w) < 1e-6:
        continue

      ball_xyz = np.array([ball_pos_w[0], ball_pos_w[1], z])
      target_xyz = np.array([target_pos_ws[batch, 0], target_pos_ws[batch, 1], z])
      visualizer.add_arrow(
        ball_xyz, target_xyz, color=(0.1, 0.8, 0.2, 0.8), width=0.015
      )


@dataclass(kw_only=True)
class PassKickCommandCfg(CommandTermCfg):
  robot_name: str = "robot"
  ball_name: str = "ball"
  target_distance_range: tuple[float, float] = (0.5, 3.0)
  """Distance from the ball to the sampled target point."""
  target_bearing_range: tuple[float, float] = (-math.pi / 4, math.pi / 4)
  """Target bearing in the robot base frame."""
  target_radius_range: tuple[float, float] = (0.15, 0.35)
  """Acceptable target radius for pass success."""

  @dataclass
  class VizCfg:
    z_offset: float = 0.08

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> PassKickCommand:
    return PassKickCommand(self, env)
