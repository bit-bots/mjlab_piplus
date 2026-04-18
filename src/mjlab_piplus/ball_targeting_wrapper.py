"""
Wrapper that overrides velocity commands with ball-relative targets.

Train velocity policy separately, then use this wrapper at inference to
inject ball-seeking commands without retraining.
"""

import torch
from mjlab.envs import ManagerBasedRlEnv


class BallTargetingWrapper:
    """
    Wraps velocity-tracking env to inject ball-relative velocity commands.

    Policy trained on velocity task expects (vx, vy, omega_z) commands.
    This wrapper computes commands based on ball position, policy sees
    them as normal velocity targets.

    Args:
        env: ManagerBasedRlEnv with velocity commands and ball entity
        k_forward: Forward velocity gain (m/s per m distance)
        k_lateral: Lateral velocity gain (m/s per m offset)
        k_turn: Angular velocity gain (rad/s per rad heading error)
        max_forward_vel: Max forward velocity command (m/s)
        max_lateral_vel: Max lateral velocity command (m/s)
        max_angular_vel: Max angular velocity command (rad/s)
        min_distance_threshold: Stop commanding forward vel when this close (m)
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        k_forward: float = 0.5,
        k_lateral: float = 0.3,
        k_turn: float = 1.0,
        max_forward_vel: float = 1.0,
        max_lateral_vel: float = 0.4,
        max_angular_vel: float = 1.5,
        min_distance_threshold: float = 0.3,
    ):
        self.env = env
        self.k_forward = k_forward
        self.k_lateral = k_lateral
        self.k_turn = k_turn
        self.max_forward_vel = max_forward_vel
        self.max_lateral_vel = max_lateral_vel
        self.max_angular_vel = max_angular_vel
        self.min_distance_threshold = min_distance_threshold

        # Verify env has ball entity and twist command
        assert "ball" in env.scene.entities, "Env must have 'ball' entity"
        assert "twist" in env.command_manager._terms, "Env must have 'twist' command"

        self.robot = env.scene["robot"]
        self.ball = env.scene["ball"]
        self.twist_cmd = env.command_manager._terms["twist"]

    def _compute_ball_relative_position(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ball position relative to robot in robot body frame.

        Returns:
            rel_x: Forward offset (m), shape (num_envs,)
            rel_y: Lateral offset (m), shape (num_envs,)
            distance: Euclidean distance (m), shape (num_envs,)
        """
        ball_pos = self.ball.data.root_link_pos_w[:, :2]  # (N, 2)
        robot_pos = self.robot.data.root_link_pos_w[:, :2]  # (N, 2)
        rel_pos_w = ball_pos - robot_pos  # World frame

        # Transform to robot body frame
        heading = self.robot.data.heading_w  # (N,)
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        rel_x = cos_h * rel_pos_w[:, 0] + sin_h * rel_pos_w[:, 1]
        rel_y = -sin_h * rel_pos_w[:, 0] + cos_h * rel_pos_w[:, 1]

        distance = torch.sqrt(rel_x**2 + rel_y**2)
        return rel_x, rel_y, distance

    def _compute_velocity_command(self) -> torch.Tensor:
        """
        Compute velocity command (vx, vy, omega_z) based on ball position.

        Returns:
            command: Velocity command tensor, shape (num_envs, 3)
                     [vx (m/s), vy (m/s), omega_z (rad/s)]
        """
        rel_x, rel_y, distance = self._compute_ball_relative_position()

        # Forward velocity: proportional to distance, saturate when close
        vx = torch.where(
            distance > self.min_distance_threshold,
            torch.clamp(
                self.k_forward * (distance - self.min_distance_threshold),
                0.0,
                self.max_forward_vel,
            ),
            torch.zeros_like(distance),
        )

        # Lateral velocity: proportional to lateral offset
        vy = torch.clamp(
            self.k_lateral * rel_y, -self.max_lateral_vel, self.max_lateral_vel
        )

        # Angular velocity: proportional to heading error
        heading_error = torch.atan2(rel_y, rel_x)  # Angle to ball in body frame
        omega_z = torch.clamp(
            self.k_turn * heading_error, -self.max_angular_vel, self.max_angular_vel
        )

        # Stack into command tensor
        command = torch.stack([vx, vy, omega_z], dim=1)  # (N, 3)
        return command

    def reset(self, env_ids=None):
        """Reset env and return observations."""
        return self.env.reset(env_ids=env_ids)

    def step(self, actions):
        """
        Step env with ball-targeting commands injected.

        Before policy inference, override twist command based on ball position.
        Policy receives observation with overridden command, produces actions,
        env steps normally.

        Args:
            actions: Policy actions

        Returns:
            obs, rewards, terminated, truncated, info (standard gym API)
        """
        # Compute and inject ball-relative velocity command
        ball_command = self._compute_velocity_command()
        self.twist_cmd.command = ball_command

        # Step env (observation will include overridden command)
        return self.env.step(actions)

    def __getattr__(self, name):
        """Forward attribute access to wrapped env."""
        return getattr(self.env, name)
