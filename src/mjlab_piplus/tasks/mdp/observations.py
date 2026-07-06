import torch
from mjlab.envs import ManagerBasedRlEnv


def _rotate_xy_w_to_b(
  xy_w: torch.Tensor,
  heading_w: torch.Tensor,
) -> torch.Tensor:
  cos_h = torch.cos(heading_w)
  sin_h = torch.sin(heading_w)
  xy_b = torch.empty_like(xy_w)
  xy_b[:, 0] = cos_h * xy_w[:, 0] + sin_h * xy_w[:, 1]
  xy_b[:, 1] = -sin_h * xy_w[:, 0] + cos_h * xy_w[:, 1]
  return xy_b


def phase(
  env: ManagerBasedRlEnv, frequency: float, command_name: str, cmd_threshold: float
) -> torch.Tensor:
  period = 1 / frequency
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  phase = torch.zeros(env.num_envs, 2, device=env.device)
  phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
  stand_mask = (
    torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    < cmd_threshold
  )
  phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
  return phase


def goal_pose_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Network-friendly goal-pose observation.

  The goal command term stores ``[dx_b, dy_b, yaw_err]`` (goal in the base
  frame). ``yaw_err`` is a wrapping angle, which is discontinuous at +/-pi and
  hard to learn an equivariant policy for. Here we encode the orientation as
  ``(cos(yaw_err), sin(yaw_err))`` instead, giving a continuous 4-dim obs
  ``[dx_b, dy_b, cos(yaw_err), sin(yaw_err)]``. The command tensor itself stays
  3-dim, so rewards/metrics/gating are unaffected.
  """
  command = env.command_manager.get_command(command_name)
  out = torch.empty(env.num_envs, 4, device=env.device)
  out[:, :2] = command[:, :2]
  out[:, 2] = torch.cos(command[:, 2])
  out[:, 3] = torch.sin(command[:, 2])
  return out


def ball_pos_b(
  env: ManagerBasedRlEnv,
  robot_name: str = "robot",
  ball_name: str = "ball",
) -> torch.Tensor:
  """Ball xy position relative to the robot base, expressed in base frame."""
  robot = env.scene[robot_name]
  ball = env.scene[ball_name]
  rel_pos_w = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
  return _rotate_xy_w_to_b(rel_pos_w, robot.data.heading_w)


def ball_vel_b(
  env: ManagerBasedRlEnv,
  robot_name: str = "robot",
  ball_name: str = "ball",
) -> torch.Tensor:
  """Ball xy velocity relative to the robot root, expressed in base frame."""
  robot = env.scene[robot_name]
  ball = env.scene[ball_name]
  rel_vel_w = (
    ball.data.root_link_lin_vel_w[:, :2] - robot.data.root_link_lin_vel_w[:, :2]
  )
  return _rotate_xy_w_to_b(rel_vel_w, robot.data.heading_w)


def zeros(env: ManagerBasedRlEnv, shape) -> torch.Tensor:
  return torch.zeros((env.num_envs, *shape), device=env.device)
