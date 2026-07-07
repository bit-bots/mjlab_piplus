from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _rotate_xy_b_to_w(xy_b: torch.Tensor, heading_w: torch.Tensor) -> torch.Tensor:
  cos_h = torch.cos(heading_w)
  sin_h = torch.sin(heading_w)
  xy_w = torch.empty_like(xy_b)
  xy_w[:, 0] = cos_h * xy_b[:, 0] - sin_h * xy_b[:, 1]
  xy_w[:, 1] = sin_h * xy_b[:, 0] + cos_h * xy_b[:, 1]
  return xy_w


def reset_ball_around_robot(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  distance_range: tuple[float, float],
  bearing_range: tuple[float, float],
  velocity_range: tuple[float, float] = (0.0, 0.0),
  robot_cfg: SceneEntityCfg | None = None,
  ball_cfg: SceneEntityCfg | None = None,
) -> None:
  """Reset the ball at a sampled base-frame offset around the robot."""
  env_ids = resolve_env_ids(env, env_ids)
  robot_cfg = robot_cfg or SceneEntityCfg("robot")
  ball_cfg = ball_cfg or SceneEntityCfg("ball")
  robot = env.scene[robot_cfg.name]
  ball = env.scene[ball_cfg.name]

  n = len(env_ids)
  distance = torch.empty(n, device=env.device).uniform_(*distance_range)
  bearing = torch.empty(n, device=env.device).uniform_(*bearing_range)
  offset_b = torch.stack(
    [distance * torch.cos(bearing), distance * torch.sin(bearing)], dim=1
  )
  offset_w = _rotate_xy_b_to_w(offset_b, robot.data.heading_w[env_ids])

  root_state = ball.data.default_root_state[env_ids].clone()
  root_state[:, 0:2] = robot.data.root_link_pos_w[env_ids, :2] + offset_w

  speed = torch.empty(n, device=env.device).uniform_(*velocity_range)
  vel_heading = torch.empty(n, device=env.device).uniform_(-torch.pi, torch.pi)
  vel_b = torch.stack(
    [speed * torch.cos(vel_heading), speed * torch.sin(vel_heading)], dim=1
  )
  root_state[:, 7:9] = _rotate_xy_b_to_w(vel_b, robot.data.heading_w[env_ids])
  root_state[:, 9:13] = 0.0

  ball.write_root_state_to_sim(root_state, env_ids=env_ids)
