"""Custom termination terms for PiPlus."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg

__all__ = ["time_out_jittered"]


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
