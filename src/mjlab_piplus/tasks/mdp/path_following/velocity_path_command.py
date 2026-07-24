"""Evaluate a velocity policy on the path-following benchmark.

``VelocityPathCommand`` is a drop-in replacement for the velocity task's
``twist`` command (used only at play/eval time). Instead of sampling a random
velocity, it runs a follow-the-carrot ``Controller`` along a reference path and
emits the resulting body-frame velocity as the command. This lets a policy
trained on ``Mjlab-Velocity-PiPlus`` be scored on the *same* path-following
metric as the dedicated path policy (``Mjlab-Path-PiPlus``).

It subclasses ``PathCommand`` so the two share the exact same cached path
library, deterministic eval iteration and metric computation — the only
difference is what the robot is *told*: a velocity from the carrot controller
here vs. the path points directly for the path policy. The metrics land in
``velo_logs*.csv`` / ``commands_velo.csv`` (vs. ``path_logs*`` / ``commands_path``)
so the two runs can be compared side by side.

The emitted command is 4-dim ``[vx, vy, wz, stand_flag]`` to match the piplus
velocity policy's command layout (the ``phase`` observation reads ``stand_flag``
at index 3). ``stand_flag`` is held at 0: the robot keeps walking toward the
goal and the episode resets once it arrives (``_log_eval_metrics``), mirroring
the path policy, which likewise does not freeze at the goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from mjlab_piplus.tasks.mdp.path_following.controller import (
  Controller,
  ControllerConfig,
)
from mjlab_piplus.tasks.mdp.path_following.path_command import (
  PathCommand,
  PathCommandCfg,
)
from mjlab_piplus.tasks.mdp.path_following.utils import angle_distance

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class VelocityPathCommand(PathCommand):
  # Separate eval logs so a velocity-policy run and a path-policy run over the
  # same path library can be compared without clobbering each other's files.
  _commands_csv = "commands_velo.csv"
  _log_prefix = "velo_logs"

  cfg: VelocityPathCommandCfg

  def __init__(self, cfg: VelocityPathCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.local_planner = Controller(cfg.controller_cfg)
    self.carrot_point = np.zeros(3)
    # Override the parent's path-point command with a body-frame velocity command
    # [vx, vy, wz, stand_flag]; stand_flag stays 0 (keep walking to the goal).
    self.command_b = torch.zeros(self.num_envs, 4, device=self.device)
    self.stand_flag = torch.zeros(self.num_envs, 1, device=self.device)

  def _resample_command(self, env_ids) -> None:
    # Reuse the parent's path selection / library / csv logging, then reset the
    # carrot controller so its smoothing and integral state start fresh on the
    # new path.
    super()._resample_command(env_ids)
    self.local_planner = Controller(self.cfg.controller_cfg)

  def _update_command(self) -> None:
    # Closest point on the path to the robot (start of the remaining plan).
    self._update_robot_pose()
    root_pos = self.current_pose[..., :2]  # (n_envs, 2)
    dists = torch.norm(
      self.paths[..., :2] - root_pos.unsqueeze(1), dim=-1
    )  # (num_envs, resolution)
    closest_indices = torch.argmin(dists, dim=-1)  # (num_envs,)

    # Run the (per-env, numpy) carrot controller to get a velocity command.
    velocities = torch.zeros(self.num_envs, 3, device=self.device)
    # step_dt already includes decimation, so this is seconds since the last
    # resample — the controller uses it only for its smoothing filter.
    current_time = (
      self._env.common_step_counter - self.starting_step
    ) * self._env.step_dt

    for env_idx in range(self.num_envs):
      closest_idx = int(closest_indices[env_idx].item())
      # All path points from the closest point to the (padded) end of the path.
      future_path_points = self.paths[env_idx, closest_idx:, :2]  # (rem, 2)
      n = future_path_points.shape[0]
      if n >= 2:
        # Per-segment tangent heading; repeat the last so lengths match.
        path_orientation = torch.atan2(
          future_path_points[1:, 1] - future_path_points[:-1, 1],
          future_path_points[1:, 0] - future_path_points[:-1, 0],
        )
        path_orientation = torch.cat(
          [path_orientation, path_orientation[-1:]], dim=0
        )  # (rem,)
      else:
        path_orientation = torch.zeros(n, device=self.device)

      # Steer the robot onto the *goal* orientation (not the path end-tangent) at
      # the end of the plan: the carrot controller aligns to the final point's
      # yaw once close, so the velocity policy is asked for the same goal pose the
      # path policy is trained on (position AND goal_yaw_w). Both use the same
      # deterministic goal_yaw_w per path (see PathCommand._resample_command).
      if n >= 1:
        path_orientation[-1] = self.goal_yaw_w[env_idx]

      path_np = (
        torch.cat([future_path_points, path_orientation.unsqueeze(-1)], dim=-1)
        .cpu()
        .numpy()
      )  # (rem, 3): [x, y, yaw]
      current_pose_np = self.current_pose[env_idx].cpu().numpy()  # (3,)

      if len(path_np) > 0:
        cmd_vel, self.carrot_point = self.local_planner.step(
          path_np, current_pose_np, current_time
        )
        velocities[env_idx] = torch.tensor(cmd_vel, device=self.device)

    # 4-dim command [vx, vy, wz, stand_flag] to match the velocity policy's obs.
    self.command_b = torch.cat([velocities, self.stand_flag], dim=1)

    # Orientation error to the goal yaw, needed by the shared goal-reached check
    # (the parent computes this in its own _update_command, which we override).
    self.yaw_err = angle_distance(self.current_pose[:, 2], self.goal_yaw_w)

    if self.cfg.eval_mode:
      self._log_eval_metrics(dists, closest_indices, root_pos)

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

      # Carrot point the controller is steering toward (green), plus a short
      # arrow for its orientation.
      carrot_xyz = np.concatenate([self.carrot_point[:2], [0.0]])
      visualizer.add_sphere(carrot_xyz, radius=0.07, color=(0, 0.5, 0, 1))
      arrow_len = 0.2
      arrow_dir = (
        np.array(
          [np.cos(self.carrot_point[2]), np.sin(self.carrot_point[2]), 0.0]
        )
        * arrow_len
      )
      visualizer.add_arrow(
        carrot_xyz, carrot_xyz + arrow_dir, width=0.03, color=(0, 0.5, 0, 1)
      )


@dataclass(kw_only=True)
class VelocityPathCommandCfg(PathCommandCfg):

  controller_cfg = ControllerConfig()

  def build(self, env: ManagerBasedRlEnv) -> VelocityPathCommand:
    return VelocityPathCommand(self, env)
