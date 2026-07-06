"""Left-right bilateral symmetry augmentation for the PiPlus robot."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
  from rsl_rl.env import VecEnv


def _switch_left_right(joint_data: torch.Tensor) -> torch.Tensor:
  joint_data_switched = joint_data.clone()
  joint_data_switched[..., [0, 1, 2, 3, 4, 5]] = joint_data[..., [6, 7, 8, 9, 10, 11]]
  joint_data_switched[..., [6, 7, 8, 9, 10, 11]] = joint_data[..., [0, 1, 2, 3, 4, 5]]
  return joint_data_switched


def _transform_actions(actions: torch.Tensor) -> torch.Tensor:
  actions = actions.clone()
  actions[:] = _switch_left_right(actions[:]) * -1
  return actions


def _transform_obs_common(obs: torch.Tensor) -> torch.Tensor:
  """Mirror everything except the command tail (indices >= 44).

  Shared prefix layout: 3 ang_vel + 3 proj_grav + 12 jpos + 12 jvel +
  12 actions + 2 phase = 44 dims.
  """
  device = obs.device
  obs[:, :3] = obs[:, :3] * torch.tensor([-1.0, 1.0, -1.0], device=device)  # ang vel
  obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1.0, -1.0, 1.0], device=device)  # gravity
  obs[:, 6:18] = _switch_left_right(obs[:, 6:18]) * -1  # joint pos
  obs[:, 18:30] = _switch_left_right(obs[:, 18:30]) * -1  # joint vel
  obs[:, 30:42] = _switch_left_right(obs[:, 30:42]) * -1  # actions
  # phase (sin, cos): half-period shift.
  obs[:, 42:44] = obs[:, 42:44] * torch.tensor([-1.0, -1.0], device=device)
  return obs


def _transform_obs(obs: torch.Tensor) -> torch.Tensor:
  """Velocity task: 3-dim command [vx, vy, wz] at 44:47 (OBS_DIM 47)."""
  obs = obs.clone()
  obs = _transform_obs_common(obs)
  obs[:, 44:47] = obs[:, 44:47] * torch.tensor([1.0, -1.0, -1.0], device=obs.device)
  return obs


def _transform_obs_goal_pose(obs: torch.Tensor) -> torch.Tensor:
  """Goal-pose task: 4-dim command [dx_b, dy_b, cos(yaw_err), sin(yaw_err)] at
  44:48 (OBS_DIM 48).

  Under a left-right reflection: dx_b unchanged, dy_b flips, cos(yaw_err)
  unchanged (cos is even), sin(yaw_err) flips (sin is odd).
  """
  obs = obs.clone()
  obs = _transform_obs_common(obs)
  obs[:, 44:48] = obs[:, 44:48] * torch.tensor(
    [1.0, -1.0, 1.0, -1.0], device=obs.device
  )
  return obs


def _transform_obs_pass_kick(obs: torch.Tensor) -> torch.Tensor:
  """Pass/kick task: 7-dim ball and target tail at 44:51 (OBS_DIM 51)."""
  obs = obs.clone()
  obs = _transform_obs_common(obs)
  obs[:, 44:51] = obs[:, 44:51] * torch.tensor(
    [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], device=obs.device
  )
  return obs


class PiPlusSymmetry:
  """Callable left-right symmetry function for the PiPlus robot.

  Passed as ``data_augmentation_func`` to rsl_rl's ``Symmetry`` extension.
  Returns (obs, actions) with the original batch prepended to the mirrored
  batch along dim 0 (num_aug = 2).

  Only the actor observation is mirrored. The critic observation is left
  unmirrored on purpose: under data augmentation the mirrored sample reuses the
  original's value/return (``V(mirror(s)) == V(s)`` by symmetry), so the value
  function is only ever evaluated on the original obs.
  """

  # Per-task actor-obs mirror transform. Subclasses override for a different
  # command layout.
  obs_transform: Callable[[torch.Tensor], torch.Tensor] = staticmethod(_transform_obs)

  @torch.no_grad()
  def __call__(
    self,
    env: VecEnv,
    obs: TensorDict | torch.Tensor | None,
    actions: torch.Tensor | None,
  ) -> tuple[TensorDict | torch.Tensor | None, torch.Tensor | None]:
    if obs is not None:
      batch_size = obs.batch_size[0]
      obs_aug = obs.repeat(2)
      obs_aug["actor"][:batch_size] = obs["actor"][:]
      obs_aug["actor"][batch_size:] = type(self).obs_transform(obs["actor"][:])
    else:
      obs_aug = None

    if actions is not None:
      batch_size = actions.shape[0]
      action_aug = torch.zeros(
        (batch_size * 2, actions.shape[1]), device=actions.device, dtype=actions.dtype
      )
      action_aug[:batch_size] = actions[:]
      action_aug[batch_size:] = _transform_actions(actions[:])
    else:
      action_aug = None

    return obs_aug, action_aug


class PiPlusGoalPoseSymmetry(PiPlusSymmetry):
  """Symmetry for the goal-pose task (4-dim cos/sin command, OBS_DIM 48)."""

  obs_transform: Callable[[torch.Tensor], torch.Tensor] = staticmethod(
    _transform_obs_goal_pose
  )


class PiPlusPassKickSymmetry(PiPlusSymmetry):
  """Symmetry for the pass/kick task layout documented with OBS_DIM 51."""

  obs_transform: Callable[[torch.Tensor], torch.Tensor] = staticmethod(
    _transform_obs_pass_kick
  )
