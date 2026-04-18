#!/usr/bin/env python3
"""
Play trained velocity policy with ball-targeting wrapper.

Usage:
    uv run python -m mjlab_piplus.scripts.play_with_ball --checkpoint path/to/model.pt

This script loads a trained velocity-tracking policy and wraps it with
BallTargetingWrapper to inject ball-seeking commands at inference time.
No retraining needed - velocity policy sees overridden commands as normal.
"""

import argparse
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlModelCfg

from mjlab_piplus.tasks.piplus_velocity_env_cfg import piplus_velocity_env_cfg
from mjlab_piplus.ball_targeting_wrapper import BallTargetingWrapper


def load_policy(
    checkpoint_path: str,
    model_cfg: RslRlModelCfg,
    obs_dim: int,
    act_dim: int,
    device: str,
):
    """Load trained policy from checkpoint."""
    from rsl_rl.modules import ActorCritic

    # Create actor-critic module
    policy = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,  # Will not use critic
        num_actions=act_dim,
        actor_hidden_dims=model_cfg.hidden_dims,
        critic_hidden_dims=model_cfg.hidden_dims,
        activation=model_cfg.activation,
        init_noise_std=model_cfg.distribution_cfg.get("init_std", 1.0),
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    return policy


def play_with_ball_targeting(
    checkpoint_path: str,
    num_envs: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    k_forward: float = 0.5,
    k_lateral: float = 0.3,
    k_turn: float = 1.0,
    max_forward_vel: float = 1.0,
    max_lateral_vel: float = 0.4,
    max_angular_vel: float = 1.5,
    min_distance_threshold: float = 0.3,
):
    """
    Run velocity policy with ball-targeting wrapper.

    Args:
        checkpoint_path: Path to trained policy checkpoint
        num_envs: Number of parallel environments
        device: Device to run on ("cuda" or "cpu")
        k_forward: Forward velocity gain
        k_lateral: Lateral velocity gain
        k_turn: Angular velocity gain
        max_forward_vel: Max forward velocity (m/s)
        max_lateral_vel: Max lateral velocity (m/s)
        max_angular_vel: Max angular velocity (rad/s)
        min_distance_threshold: Min distance to ball before stopping (m)
    """
    print(f"Loading velocity env with ball...")
    env_cfg = piplus_velocity_env_cfg(play=True, with_ball=True)
    env = ManagerBasedRlEnv(
        env_cfg, num_envs=num_envs, device=device, enable_viewport=True
    )

    # Wrap env with ball-targeting controller
    print(f"Wrapping env with BallTargetingWrapper...")
    print(f"  Gains: k_forward={k_forward}, k_lateral={k_lateral}, k_turn={k_turn}")
    print(
        f"  Limits: vx_max={max_forward_vel}, vy_max={max_lateral_vel}, omega_max={max_angular_vel}"
    )
    wrapped_env = BallTargetingWrapper(
        env,
        k_forward=k_forward,
        k_lateral=k_lateral,
        k_turn=k_turn,
        max_forward_vel=max_forward_vel,
        max_lateral_vel=max_lateral_vel,
        max_angular_vel=max_angular_vel,
        min_distance_threshold=min_distance_threshold,
    )

    # Load policy
    print(f"Loading policy from {checkpoint_path}...")
    from mjlab_piplus.tasks.piplus_velocity_rl_cfg import get_piplus_velocity_rl_cfg

    rl_cfg = get_piplus_velocity_rl_cfg()

    obs_dim = env.observation_manager.group_obs_dim["actor"][0]
    act_dim = env.action_manager.total_action_dim

    policy = load_policy(checkpoint_path, rl_cfg.actor, obs_dim, act_dim, device)

    # Run play loop
    print(f"Starting play loop...")
    obs, _ = wrapped_env.reset()
    actor_obs = obs["actor"]

    step_count = 0
    episode_count = 0

    try:
        while True:
            with torch.no_grad():
                actions = policy.act_inference(actor_obs)

            obs, rewards, terminated, truncated, info = wrapped_env.step(actions)
            actor_obs = obs["actor"]

            step_count += 1

            # Reset terminated/truncated envs
            if terminated.any() or truncated.any():
                reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                episode_count += len(reset_ids)
                print(f"Episode {episode_count} finished. Total steps: {step_count}")

    except KeyboardInterrupt:
        print("\nStopping play...")

    print(f"Completed {episode_count} episodes, {step_count} steps")


def main():
    parser = argparse.ArgumentParser(
        description="Play velocity policy with ball targeting"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to policy checkpoint"
    )
    parser.add_argument(
        "--num-envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Controller gains
    parser.add_argument(
        "--k-forward", type=float, default=0.5, help="Forward velocity gain"
    )
    parser.add_argument(
        "--k-lateral", type=float, default=0.3, help="Lateral velocity gain"
    )
    parser.add_argument(
        "--k-turn", type=float, default=1.0, help="Angular velocity gain"
    )

    # Velocity limits
    parser.add_argument(
        "--max-forward-vel", type=float, default=1.0, help="Max forward velocity (m/s)"
    )
    parser.add_argument(
        "--max-lateral-vel", type=float, default=0.4, help="Max lateral velocity (m/s)"
    )
    parser.add_argument(
        "--max-angular-vel",
        type=float,
        default=1.5,
        help="Max angular velocity (rad/s)",
    )

    parser.add_argument(
        "--min-distance", type=float, default=0.3, help="Min distance to ball (m)"
    )

    args = parser.parse_args()

    play_with_ball_targeting(
        checkpoint_path=args.checkpoint,
        num_envs=args.num_envs,
        device=args.device,
        k_forward=args.k_forward,
        k_lateral=args.k_lateral,
        k_turn=args.k_turn,
        max_forward_vel=args.max_forward_vel,
        max_lateral_vel=args.max_lateral_vel,
        max_angular_vel=args.max_angular_vel,
        min_distance_threshold=args.min_distance,
    )


if __name__ == "__main__":
    main()
