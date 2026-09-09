"""The training-time push: one disturbance, either a shove or a spin.

Like mjlab's ``push_by_setting_velocity`` this is an *instantaneous,
mass-independent* kick -- a delta written straight onto the root velocity,
ignoring inertia and contact. What it changes is the shape of the sample. The
mjlab term draws all six components of the twist independently from their own
box, so a typical push is a little of everything at once: a diagonal shove with
some roll, some pitch and some yaw mixed in, and its magnitude concentrates
near the box's RMS rather than covering the range.

Here a push picks *one kind* -- translational or rotational -- and gets a
direction and a magnitude of its own:

- **translational**: a horizontal direction, uniform on the circle, times a
  speed. No vertical component: a push is something that arrives sideways.
- **rotational**: the same horizontal direction used as a *spin axis*, times an
  angular speed about it (right-hand rule). A horizontal axis is what tips the
  robot; spinning it about z instead only turns it on the spot, so yaw is not
  part of this term.

Splitting the two matters because they ask for different recoveries -- a shove
is caught by stepping, a spin by the trunk and arms -- and mixing them in one
sample means neither ever arrives clean. Magnitude ranges are ramped by the
``event_param_linear`` curriculum; see the ``PUSH_*`` constants in the multi
env cfg.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["push_by_velocity_delta"]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def push_by_velocity_delta(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  linear_speed_range: tuple[float, float],
  angular_speed_range: tuple[float, float],
  rotational_probability: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Add one sampled velocity delta to the root. Use with ``mode="interval"``.

  Each pushed env draws its kind first (rotational with
  ``rotational_probability``, translational otherwise), then a horizontal
  direction and a magnitude for that kind. The other kind contributes nothing,
  so ``linear_speed_range`` is the speed of a shove and
  ``angular_speed_range`` the speed of a spin -- neither is diluted by the
  other, and both are read directly off the range they are sampled from.

  The direction covers the full circle, so the magnitudes need no sign: a spin
  about ``-axis`` is the same as a negative spin about ``+axis``.
  """
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  n = len(env_ids)
  device = env.device

  # One horizontal direction per push: the shove direction, or the spin axis.
  angle = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
  direction = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)

  rotational = torch.rand(n, device=device) < rotational_probability
  zero = torch.zeros(n, device=device)
  linear_speed = torch.where(
    rotational, zero, torch.empty(n, device=device).uniform_(*linear_speed_range)
  )
  angular_speed = torch.where(
    rotational, torch.empty(n, device=device).uniform_(*angular_speed_range), zero
  )

  # root_link_vel_w is [lin_xyz | ang_xyz] in the world frame. Advanced indexing
  # copies, so this edits a temporary and writes it back, as mjlab's push does.
  vel_w = asset.data.root_link_vel_w[env_ids]
  vel_w[:, :2] += direction * linear_speed.unsqueeze(1)
  vel_w[:, 3:5] += direction * angular_speed.unsqueeze(1)
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)
