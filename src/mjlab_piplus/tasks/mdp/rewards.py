from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from mjlab_piplus.tasks.mdp.commands import PassKickCommand


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class track_angular_velocity_filtered:
  """Reward yaw-rate tracking against a low-pass filtered base angular velocity.

  The commanded yaw rate is compared against an EMA of the measured yaw rate so
  that short oscillations (e.g. torso swing) do not yield reward. XY angular
  velocity is still penalized instantaneously.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    tau: float = cfg.params["tau"]
    dt = env.step_dt
    self._alpha = dt / (tau + dt)
    self._yaw_ema = torch.zeros(env.num_envs, device=env.device)
    self._last_yaw_raw = torch.zeros(env.num_envs, device=env.device)
    self._last_yaw_cmd = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    tau: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del tau
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    self._yaw_ema = (1.0 - self._alpha) * self._yaw_ema + self._alpha * actual[:, 2]
    self._last_yaw_raw = actual[:, 2]
    self._last_yaw_cmd = command[:, 2]
    z_error = torch.square(command[:, 2] - self._yaw_ema)
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    return torch.exp(-(z_error + xy_error) / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    self._yaw_ema[env_ids] = 0.0


class upright_relaxed:
  """Like mjlab's ``upright`` but with a cubic exponent for a flatter plateau.

  Uses ``exp(-|tilt|^3 / std^2)`` instead of ``exp(-|tilt|^2 / std^2)`` so small
  tilts are barely penalized and large ones fall off sharply. Useful when
  natural gait-induced torso sway shouldn't be punished.

  Assumes flat ground (no terrain sensor support).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg, env

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    else:
      body_quat_w = asset.data.root_link_quat_w
    gravity_w = asset.data.gravity_vec_w
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    tilt_cubed = xy_squared * torch.sqrt(xy_squared.clamp(min=1e-12))
    return torch.exp(-tilt_cubed / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids


def alive(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return ~env.termination_manager.terminated


def feet_air_time_zero_vel(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)  # number of feet in the air
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
      command_zero_mask = total_command == 0.0
      reward[command_zero_mask] = (
        1.0  # maybe reward max value, if we reach zero support
      )
  return reward


def feet_gait(
  env: ManagerBasedRlEnv,
  frequency: float,
  offset: list[float],
  threshold: float,
  command_threshold: float,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  period = 1 / frequency
  sensor: ContactSensor = env.scene[sensor_name]
  is_contact = sensor.data.current_contact_time > 0
  global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
  offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(
    1, -1
  )
  leg_phase = (global_phase + offsets) % 1.0
  is_stance = leg_phase < threshold
  reward = (is_stance == is_contact).float().mean(dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )

    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  diff_angle = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  reward = torch.sum(torch.square(diff_angle), dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command <= command_threshold).float()
      reward *= scale
  return reward


def feet_clearance_positive(*args, **kwargs):
  r = feet_clearance(*args, **kwargs)
  return 1 - r


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def _rotate_xy_w_to_b(xy_w: torch.Tensor, heading_w: torch.Tensor) -> torch.Tensor:
  cos_h = torch.cos(heading_w)
  sin_h = torch.sin(heading_w)
  xy_b = torch.empty_like(xy_w)
  xy_b[:, 0] = cos_h * xy_w[:, 0] + sin_h * xy_w[:, 1]
  xy_b[:, 1] = -sin_h * xy_w[:, 0] + cos_h * xy_w[:, 1]
  return xy_b


def _pass_kick_state(
  env: ManagerBasedRlEnv,
  command_name: str,
  robot_name: str = "robot",
  ball_name: str = "ball",
):
  term = cast("PassKickCommand", env.command_manager.get_term(command_name))
  robot: Entity = env.scene[robot_name]
  ball: Entity = env.scene[ball_name]
  target_pos_w = term.target_pos_w
  target_radius = term.target_radius
  ball_pos_w = ball.data.root_link_pos_w[:, :2]
  ball_vel_w = ball.data.root_link_vel_w[:, :2]
  robot_pos_w = robot.data.root_link_pos_w[:, :2]
  robot_vel_w = robot.data.root_link_vel_w[:, :2]
  target_delta_w = target_pos_w - ball_pos_w
  target_dist = torch.linalg.norm(target_delta_w, dim=1)
  ball_delta_w = ball_pos_w - robot_pos_w
  ball_dist = torch.linalg.norm(ball_delta_w, dim=1)
  ball_pos_b = _rotate_xy_w_to_b(ball_delta_w, robot.data.heading_w)
  target_from_ball_b = term.command[:, :2]
  return (
    term,
    target_radius,
    target_delta_w,
    target_dist,
    ball_vel_w,
    ball_dist,
    ball_pos_b,
    target_from_ball_b,
    robot_vel_w,
  )


class ball_target_progress:
  """Reward per-step reduction in ball-to-target distance."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    term = cast(
      "PassKickCommand", env.command_manager.get_term(cfg.params["command_name"])
    )
    ball: Entity = env.scene[cfg.params.get("ball_name", "ball")]
    self.last_dist = torch.linalg.norm(
      term.target_pos_w - ball.data.root_link_pos_w[:, :2], dim=1
    )
    self.last_counter = term.command_counter.clone()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    max_speed: float = 2.0,
    success_hold_radius_scale: float = 1.0,
    robot_name: str = "robot",
    ball_name: str = "ball",
  ) -> torch.Tensor:
    del robot_name
    term = cast("PassKickCommand", env.command_manager.get_term(command_name))
    ball: Entity = env.scene[ball_name]
    target_radius = term.target_radius
    dist = torch.linalg.norm(
      term.target_pos_w - ball.data.root_link_pos_w[:, :2], dim=1
    )
    progress = self.last_dist - dist
    command_changed = term.command_counter != self.last_counter
    progress = torch.where(command_changed, torch.zeros_like(progress), progress)
    reward = torch.clamp(progress / (max_speed * env.step_dt), max=1.0)
    success_zone = dist <= target_radius * success_hold_radius_scale
    reward = torch.where(success_zone, torch.ones_like(reward), reward)
    self.last_dist = dist
    self.last_counter = term.command_counter.clone()
    return reward

  def reset(self, env_ids: torch.Tensor) -> None:
    self.last_dist[env_ids] = 0.0


class approach_ball_progress:
  """Reward per-step reduction in robot-to-ball distance before the kick."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    robot: Entity = env.scene[cfg.params.get("robot_name", "robot")]
    ball: Entity = env.scene[cfg.params.get("ball_name", "ball")]
    self.last_dist = torch.linalg.norm(
      ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=1
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    max_speed: float = 0.6,
    active_until_ball_speed: float = 0.15,
    robot_name: str = "robot",
    ball_name: str = "ball",
  ) -> torch.Tensor:
    (
      _term,
      _target_radius,
      _target_delta_w,
      _target_dist,
      ball_vel_w,
      ball_dist,
      _ball_pos_b,
      _target_from_ball_b,
      _robot_vel_w,
    ) = _pass_kick_state(env, command_name, robot_name, ball_name)
    progress = self.last_dist - ball_dist
    reward = torch.clamp(progress / (max_speed * env.step_dt), max=1.0)
    ball_speed = torch.linalg.norm(ball_vel_w, dim=1)
    reward *= (ball_speed < active_until_ball_speed).float()
    self.last_dist = ball_dist
    return reward

  def reset(self, env_ids: torch.Tensor) -> None:
    self.last_dist[env_ids] = 0.0


def ball_close_to_target(
  env: ManagerBasedRlEnv,
  command_name: str,
  epsilon: float = 3.0,
  normalize_by_radius: bool = True,
  robot_name: str = "robot",
  ball_name: str = "ball",
) -> torch.Tensor:
  (
    _term,
    target_radius,
    _target_delta_w,
    target_dist,
    _ball_vel_w,
    _ball_dist,
    _ball_pos_b,
    _target_from_ball_b,
    _robot_vel_w,
  ) = _pass_kick_state(env, command_name, robot_name, ball_name)
  if normalize_by_radius:
    target_dist = target_dist / target_radius.clamp(min=1e-3)
  return torch.exp(-epsilon * target_dist)


def ball_velocity_to_target(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_speed: float = 2.0,
  robot_name: str = "robot",
  ball_name: str = "ball",
) -> torch.Tensor:
  (
    _term,
    _target_radius,
    target_delta_w,
    target_dist,
    ball_vel_w,
    _ball_dist,
    _ball_pos_b,
    _target_from_ball_b,
    _robot_vel_w,
  ) = _pass_kick_state(env, command_name, robot_name, ball_name)
  target_dir_w = target_delta_w / target_dist.clamp(min=1e-6).unsqueeze(1)
  projected_speed = torch.sum(ball_vel_w * target_dir_w, dim=1)
  return torch.clamp(projected_speed / max_speed, min=0.0, max=1.0)


def ball_success(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_ball_speed: float = 0.1,
  robot_name: str = "robot",
  ball_name: str = "ball",
) -> torch.Tensor:
  (
    _term,
    target_radius,
    _target_delta_w,
    target_dist,
    ball_vel_w,
    _ball_dist,
    _ball_pos_b,
    _target_from_ball_b,
    _robot_vel_w,
  ) = _pass_kick_state(env, command_name, robot_name, ball_name)
  ball_speed = torch.linalg.norm(ball_vel_w, dim=1)
  return ((target_dist <= target_radius) & (ball_speed <= max_ball_speed)).float()


def ball_speed_near_target(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_ball_speed: float = 0.1,
  near_radius_scale: float = 2.0,
  robot_name: str = "robot",
  ball_name: str = "ball",
) -> torch.Tensor:
  (
    _term,
    target_radius,
    _target_delta_w,
    target_dist,
    ball_vel_w,
    _ball_dist,
    _ball_pos_b,
    _target_from_ball_b,
    _robot_vel_w,
  ) = _pass_kick_state(env, command_name, robot_name, ball_name)
  ball_speed = torch.linalg.norm(ball_vel_w, dim=1)
  near_target = target_dist <= target_radius * near_radius_scale
  return torch.clamp(ball_speed - max_ball_speed, min=0.0) * near_target.float()


def close_to_ball(
  env: ManagerBasedRlEnv,
  command_name: str,
  epsilon: float = 2.0,
  active_until_ball_speed: float = 0.15,
  robot_name: str = "robot",
  ball_name: str = "ball",
) -> torch.Tensor:
  (
    _term,
    _target_radius,
    _target_delta_w,
    _target_dist,
    ball_vel_w,
    ball_dist,
    _ball_pos_b,
    _target_from_ball_b,
    _robot_vel_w,
  ) = _pass_kick_state(env, command_name, robot_name, ball_name)
  ball_speed = torch.linalg.norm(ball_vel_w, dim=1)
  return (
    torch.exp(-epsilon * ball_dist) * (ball_speed < active_until_ball_speed).float()
  )


def ball_in_kick_zone(
  env: ManagerBasedRlEnv,
  command_name: str,
  target_x: float = 0.22,
  target_abs_y: float = 0.08,
  std_x: float = 0.25,
  std_y: float = 0.18,
  active_until_ball_speed: float = 0.15,
  robot_name: str = "robot",
  ball_name: str = "ball",
) -> torch.Tensor:
  (
    _term,
    _target_radius,
    _target_delta_w,
    _target_dist,
    ball_vel_w,
    _ball_dist,
    ball_pos_b,
    _target_from_ball_b,
    _robot_vel_w,
  ) = _pass_kick_state(env, command_name, robot_name, ball_name)
  x_error = (ball_pos_b[:, 0] - target_x) / std_x
  y_error = (torch.abs(ball_pos_b[:, 1]) - target_abs_y) / std_y
  ball_speed = torch.linalg.norm(ball_vel_w, dim=1)
  return (
    torch.exp(-(x_error**2 + y_error**2))
    * (ball_speed < active_until_ball_speed).float()
  )


def face_ball_or_target(
  env: ManagerBasedRlEnv,
  command_name: str,
  coeff: float = 1.0,
  switch_distance: float = 0.4,
  robot_name: str = "robot",
  ball_name: str = "ball",
) -> torch.Tensor:
  (
    _term,
    _target_radius,
    _target_delta_w,
    _target_dist,
    _ball_vel_w,
    ball_dist,
    ball_pos_b,
    target_from_ball_b,
    _robot_vel_w,
  ) = _pass_kick_state(env, command_name, robot_name, ball_name)
  ball_bearing = torch.atan2(ball_pos_b[:, 1], ball_pos_b[:, 0])
  target_bearing = torch.atan2(target_from_ball_b[:, 1], target_from_ball_b[:, 0])
  bearing = torch.where(ball_dist > switch_distance, ball_bearing, target_bearing)
  return torch.exp(-coeff * torch.abs(bearing))


def standstill_before_kick(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_base_speed: float = 0.08,
  max_ball_distance: float = 1.5,
  active_until_ball_speed: float = 0.15,
  distance_std: float = 0.75,
  robot_name: str = "robot",
  ball_name: str = "ball",
) -> torch.Tensor:
  (
    _term,
    _target_radius,
    _target_delta_w,
    _target_dist,
    ball_vel_w,
    ball_dist,
    _ball_pos_b,
    _target_from_ball_b,
    robot_vel_w,
  ) = _pass_kick_state(env, command_name, robot_name, ball_name)
  ball_speed = torch.linalg.norm(ball_vel_w, dim=1)
  robot_speed = torch.linalg.norm(robot_vel_w, dim=1)
  active = (ball_dist <= max_ball_distance) & (ball_speed < active_until_ball_speed)
  speed_cost = torch.clamp(min_base_speed - robot_speed, min=0.0) / min_base_speed
  distance_scale = torch.exp(-ball_dist / distance_std)
  return speed_cost * distance_scale * active.float()


class goal_progress:
  """Reward progress toward the goal: the per-step reduction in Euclidean
  distance to the goal position.

  Normalized so that approaching the goal at ``max_vel`` yields a reward of 1.
  Moving away yields a proportional negative reward (unbounded below). The
  reward is capped at 1 so the policy is not encouraged to exceed ``max_vel``,
  which can be unstable.

  Within ``distance_threshold`` of the goal the reward is always 1, giving the
  policy a reason to hold position at the goal instead of oscillating once no
  more progress can be made.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    term = env.command_manager.get_term(cfg.params["command_name"])
    self.last_dist = torch.linalg.norm(term.command[:, :2], dim=1)
    self.last_counter = term.command_counter.clone()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    max_vel: float = 1.0,
    distance_threshold: float = 0.5,
  ) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    assert term is not None, f"Command '{command_name}' not found."
    dist = torch.linalg.norm(term.command[:, :2], dim=1)  # [B]
    # Positive when the robot got closer to the goal since the previous step.
    progress = self.last_dist - dist
    # On a goal change (periodic resample or episode reset) the distance delta
    # is meaningless, so zero those steps out.
    goal_changed = term.command_counter != self.last_counter
    progress = torch.where(goal_changed, torch.zeros_like(progress), progress)
    # step_dt already includes decimation (physics_dt * decimation), so the max
    # distance reachable in one policy step is max_vel * step_dt.
    reward = torch.clamp(progress / (max_vel * env.step_dt), max=1.0)
    # Hold reward at the goal so the policy does not oscillate near it.
    reward = torch.where(dist <= distance_threshold, torch.ones_like(reward), reward)
    self.last_dist = dist
    self.last_counter = term.command_counter.clone()
    return reward


def close_to_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  epsilon: float = 0.5,
) -> torch.Tensor:
  """Reward proximity to the goal position: ``exp(-epsilon * d)``.

  ``d`` is the Euclidean distance to the goal position, read from the base-frame
  goal command ``[dx_b, dy_b, yaw_err]``.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  d = torch.norm(command[:, :2], dim=1)
  return torch.exp(-epsilon * d)


def align_with_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  pos_coeff: float = 5.0,
  ori_coeff: float = 1.0,
) -> torch.Tensor:
  """Reward reaching the goal pose: ``exp(-pos_coeff * d) * exp(-ori_coeff * |alpha|)``.

  ``d`` is the Euclidean distance to the goal position and ``alpha`` is the
  angular distance to the goal orientation (wrapped yaw error).
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  d = torch.norm(command[:, :2], dim=1)
  alpha = torch.abs(command[:, 2])
  return torch.exp(-pos_coeff * d) * torch.exp(-ori_coeff * alpha)


def orient_towards_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  coeff: float = 1.0,
  distance_threshold: float = 0.3,
) -> torch.Tensor:
  """Reward facing the goal position while far away.

  When farther than ``distance_threshold`` from the goal, returns
  ``exp(-coeff * |beta|)`` where ``beta`` is the angle between the robot's
  forward (base x) axis and the vector from the base to the goal position
  (i.e. the goal bearing in the base frame). When within
  ``distance_threshold``, returns 1.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  d = torch.norm(command[:, :2], dim=1)
  beta = torch.atan2(command[:, 1], command[:, 0])
  reward = torch.exp(-coeff * torch.abs(beta))
  return torch.where(d <= distance_threshold, torch.ones_like(reward), reward)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def flat_foot_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-flat feet via each foot site's projected gravity.

  Replacement for ``mdp.flat_orientation_l2``, which only reads the *root link*
  quaternion (ignoring the passed selection) and so penalizes base tilt, not
  foot tilt. This uses the foot sites in ``asset_cfg.site_names`` and sums the
  squared xy projected-gravity components over the feet — matching the
  ``foot_tilt`` metrics. Yaw-invariant.
  """
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.site_quat_w[:, asset_cfg.site_ids]  # (E, n_feet, 4)
  gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(
    -1, quat.shape[1], -1
  )  # (E, n_feet, 3)
  proj_g = quat_apply_inverse(quat, gravity_w)  # (E, n_feet, 3)
  return torch.sum(torch.square(proj_g[..., :2]), dim=(-2, -1))
