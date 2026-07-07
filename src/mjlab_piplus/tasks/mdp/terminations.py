"""Custom termination terms for PiPlus."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg

  from mjlab_piplus.tasks.mdp.commands import PassKickCommand

__all__ = ["ball_out_of_bounds", "pass_kick_success", "time_out_jittered"]


def ball_out_of_bounds(
  env: ManagerBasedRlEnv,
  ball_name: str = "ball",
  max_distance_from_origin: float = 6.0,
) -> torch.Tensor:
  ball = env.scene[ball_name]
  pos_xy = ball.data.root_link_pos_w[:, :2]
  return (
    torch.linalg.norm(pos_xy - env.scene.env_origins[:, :2], dim=1)
    > max_distance_from_origin
  )


def pass_kick_success(
  env: ManagerBasedRlEnv,
  command_name: str,
  ball_name: str = "ball",
  max_ball_speed: float = 0.1,
) -> torch.Tensor:
  term = cast("PassKickCommand", env.command_manager.get_term(command_name))
  ball = env.scene[ball_name]
  target_dist = torch.linalg.norm(
    term.target_pos_w - ball.data.root_link_pos_w[:, :2], dim=1
  )
  ball_speed = torch.linalg.norm(ball.data.root_link_vel_w[:, :2], dim=1)
  return (target_dist <= term.target_radius) & (ball_speed <= max_ball_speed)


class time_out_jittered:
  """Episode time-out at a per-env random horizon (mark with ``time_out=True``).

  Drop-in replacement for ``mdp.time_out``. The plain version fires exactly at
  ``max_episode_length`` for every env, so once the policy can survive a full
  episode all envs time out in lockstep and reset together. That synchronized
  reset shows up as a periodic dip in the posture/tracking/alive rewards (and a
  spike in ``is_terminated``) every ~episode-length iterations as the whole
  cohort recovers from the reset pose at once.

  This term instead times out when ``episode_length_buf`` reaches a per-env
  horizon drawn uniformly from ``[min_fraction * max, max]``, re-rolled on every
  reset, so resets stay decorrelated for the entire run — unlike the one-shot
  ``init_at_random_ep_len`` startup randomization, which a synchronized reset
  could undo. The horizon is additionally seeded across the *full* ``[1, max]``
  range at construction so episodes start staggered without relying on the
  runner's ``init_at_random_ep_len`` flag.
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv) -> None:
    self._env = env
    self._min_fraction: float = cfg.params.get("min_fraction", 0.8)
    self._horizon = torch.empty(env.num_envs, dtype=torch.long, device=env.device)
    # Seed across the full range so the first episodes are already staggered.
    self._horizon[:] = torch.randint(
      1, int(env.max_episode_length) + 1, (env.num_envs,), device=env.device
    )

  def _resample(self, env_ids: torch.Tensor) -> None:
    lo = max(1, int(self._min_fraction * self._env.max_episode_length))
    hi = int(self._env.max_episode_length)
    self._horizon[env_ids] = torch.randint(
      lo, hi + 1, (len(env_ids),), device=self._env.device
    )

  def __call__(self, env: ManagerBasedRlEnv, min_fraction: float = 0.8) -> torch.Tensor:
    del min_fraction  # Consumed in __init__.
    return env.episode_length_buf >= self._horizon

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self._env.num_envs, device=self._env.device)
    self._resample(env_ids)
