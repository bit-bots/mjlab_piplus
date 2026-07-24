"""Follow-the-carrot path controller.

Turns a reference path into a body-frame velocity command ``[vx, vy, wz]`` so a
velocity-conditioned policy can be evaluated on the path-following benchmark
(see ``velocity_path_command.py``). This is a plain-numpy, per-environment
controller — it is only used at eval/play time (a handful of envs), never in the
training loop.

Ported from mjlab_x02 with the same control law; only the default velocity
limits are left to the caller (``VelocityPathCommand`` sets piplus-appropriate
values). The algorithm is the Bit-Bots "walk to a carrot" controller:
https://github.com/bit-bots/bitbots_main/issues/366#issuecomment-2161460917
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ControllerConfig:
  """Parameters for the follow-the-carrot path controller."""

  carrot_distance: int = 100
  """Index offset from the closest path point to the carrot the robot walks
  toward. Paths are sampled at a fixed arc-length spacing, so this is a fixed
  look-ahead distance (e.g. 50 points x 1 cm spacing = 0.5 m ahead)."""
  translation_slow_down_factor: float = 2.0
  orient_to_goal_distance: float = 0.3
  rotation_slow_down_factor: float = 3.0
  rotation_i_factor: float = 0.005
  max_rotation_vel: float = 1.0
  max_vel_x: float = 1.0
  min_vel_x: float = -0.9
  max_vel_y: float = 0.3
  smoothing_tau: float = 0.04


class Controller:
  """Follow-the-carrot controller that keeps a robot on a given path."""

  def __init__(self, config: ControllerConfig) -> None:
    self.config = config

    # Last command velocity [linear_x, linear_y, angular_z].
    self.last_cmd_vel = np.zeros(3)

    # Last update time (seconds).
    self.last_update_time: Optional[float] = 0.0

    # Accumulator for the angular error (integral term).
    self.angular_error_accumulator = 0.0

  def step(
    self,
    path: np.ndarray,
    current_pose: np.ndarray,
    current_time: Optional[float] = None,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Compute a body-frame velocity command that follows ``path``.

    Args:
      path: (num_points, 3) array of upcoming path points ``[x, y, yaw]`` in the
        world frame, starting at the point closest to the robot.
      current_pose: robot pose ``[x, y, yaw]`` in the world frame.
      current_time: time in seconds; used for the smoothing filter's timestep.
        Falls back to wall-clock time if omitted.

    Returns:
      ``(cmd_vel, carrot_point)`` where ``cmd_vel`` is ``[vx, vy, wz]`` in the
      body frame and ``carrot_point`` is the ``[x, y, yaw]`` target (for viz).
    """
    # Starting point for our goal velocity calculation.
    cmd_vel = np.zeros(3)  # [linear_x, linear_y, angular_z]

    if current_time is None:
      current_time = time.time()

    # Select the carrot (goal for this controller) and the end of the plan.
    # Handle plans shorter than the carrot distance.
    if len(path) > 0:
      end_pose = path[-1]  # [x, y, yaw]
      goal_pose = end_pose
      if len(path) > self.config.carrot_distance:
        goal_pose = path[self.config.carrot_distance]
    else:
      carrot_point = np.zeros(3)
      return cmd_vel, carrot_point

    # Intended walk angle from the carrot and our position.
    walk_angle = math.atan2(
      goal_pose[1] - current_pose[1], goal_pose[0] - current_pose[0]
    )

    # Heading from our current position to the final position of the plan.
    final_walk_angle = math.atan2(
      goal_pose[1] - current_pose[1], goal_pose[0] - current_pose[0]
    )

    if len(path) < 3:
      # Distance from our position to the final position of the plan.
      distance = math.hypot(
        end_pose[0] - current_pose[0], end_pose[1] - current_pose[1]
      )
    else:
      # Path length from the robot along the remaining plan.
      distance = 0
      a_position = current_pose[:2]  # [x, y]
      for point in path:
        b_position = point[:2]  # [x, y]
        distance += math.hypot(
          b_position[0] - a_position[0], b_position[1] - a_position[1]
        )
        a_position = b_position
      distance += math.hypot(
        end_pose[0] - a_position[0], end_pose[1] - a_position[1]
      )

    # Translational walk velocity: proportional to remaining distance, so the
    # robot brakes as it approaches the final position of the plan.
    walk_vel = distance * self.config.translation_slow_down_factor

    # Close to the goal, align with the goal orientation instead of the heading
    # toward the goal position.
    diff = 0.0
    if distance > self.config.orient_to_goal_distance:
      diff = final_walk_angle - current_pose[2]  # yaw is at index 2
    else:
      if len(end_pose) > 2:
        diff = end_pose[2] - current_pose[2]  # yaw difference
      else:
        diff = final_walk_angle - current_pose[2]

    # Wrap the angular difference to [-pi, pi].
    min_angle = math.remainder(diff, math.tau)

    # Accumulate the angular error (integral).
    self.angular_error_accumulator += min_angle

    # Target rotational velocity (P + I on the angular error).
    target_rot_vel = (
      self.config.rotation_slow_down_factor * min_angle
      + self.config.rotation_i_factor * self.angular_error_accumulator
    )

    # Clamp rotational velocity to max (both directions).
    rot_goal_vel = min(
      max(target_rot_vel, -self.config.max_rotation_vel),
      self.config.max_rotation_vel,
    )

    # Heading of our linear velocity vector in the robot's local frame.
    local_heading = walk_angle - current_pose[2]  # yaw is at index 2
    local_heading_vector_x = math.cos(local_heading)
    local_heading_vector_y = math.sin(local_heading)

    def interpolate_max_vel(x_limit, y_limit, target_x, target_y):
      """Interpolate the x/y max velocity linearly based on the heading."""
      n = (x_limit * abs(target_y)) + y_limit * target_x
      intersection_x = (x_limit * y_limit * target_x) / n
      intersection_y = (x_limit * y_limit * abs(target_y)) / n
      return math.hypot(intersection_x, intersection_y)

    # Limit x/y velocity with the heading-specific maximum velocity.
    walk_vel = min(
      walk_vel,
      interpolate_max_vel(
        # Different max for forward vs. backward x movement.
        self.config.max_vel_x
        if local_heading_vector_x > 0
        else self.config.min_vel_x,
        self.config.max_vel_y,
        local_heading_vector_x,
        local_heading_vector_y,
      ),
    )

    # x/y components of the linear velocity from the desired heading and speed.
    cmd_vel[0] = local_heading_vector_x * walk_vel  # linear.x
    cmd_vel[1] = local_heading_vector_y * walk_vel  # linear.y
    cmd_vel[2] = rot_goal_vel  # angular.z

    # Smooth the command velocity to avoid sudden changes. Skip on the first
    # update (no previous time).
    if self.last_update_time is not None:
      delta_time = current_time - self.last_update_time
      # Exponential-decay smoothing factor alpha derived from tau, so it is
      # timestep-independent.
      exponent_in_s = -delta_time / self.config.smoothing_tau
      alpha = 1 - math.exp(exponent_in_s)
      cmd_vel[0] = alpha * cmd_vel[0] + (1 - alpha) * self.last_cmd_vel[0]
      cmd_vel[1] = alpha * cmd_vel[1] + (1 - alpha) * self.last_cmd_vel[1]
      cmd_vel[2] = alpha * cmd_vel[2] + (1 - alpha) * self.last_cmd_vel[2]

    self.last_update_time = current_time
    self.last_cmd_vel = cmd_vel.copy()

    # Carrot point for visualization.
    carrot_point = np.array(
      [goal_pose[0], goal_pose[1], goal_pose[2] if len(goal_pose) > 2 else 0.0]
    )

    return cmd_vel, carrot_point
