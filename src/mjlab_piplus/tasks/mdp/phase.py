
from __future__ import annotations


from dataclasses import dataclass
from typing import TYPE_CHECKING

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mujoco
import torch
import torch.nn.functional as F

# ── mjlab imports ──────────────────────────────────────────────────────────────
from mjlab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, RewardTermCfg  # type: ignore

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.entity import Entity

class Phase(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        p = cfg.params

        self._heavy_tail_shift: float = float( p.get("heavy_tail_shift", 0.5))  # Between 0 and 1.
        self._target_height: float = float( p.get("target_height", 0.05))
        self._rest_height: float = float( p.get("rest_height", 0.0))

        self.asset: Entity = env.scene[cfg.params["asset_cfg"].name]

        self.site_name_id_map = {name: i for i, name in enumerate(self.asset.site_names)}


    def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
        phase = env.command_manager.get_command("phase").squeeze(-1)

        # Phase reward
        target_height_left = (1 - (torch.cos(math.tau * torch.pow(phase, self._heavy_tail_shift)) + 1) / 2) * self._target_height + self._rest_height
        target_height_right = (1 - (torch.cos(math.tau * torch.pow((phase + 0.5) % 1.0, self._heavy_tail_shift)) + 1) / 2) * self._target_height + self._rest_height

        # Get current foot heights
        left_foot_height = self.asset.data.site_pos_w[:, self.site_name_id_map["l_foot"], 2]
        right_foot_height = self.asset.data.site_pos_w[:, self.site_name_id_map["r_foot"], 2]

        #print(f"Phase: {phase[0]}, Target Left: {target_height_left[0]}, Target Right: {target_height_right[0]}, Left Foot Height: {left_foot_height[0]}, Right Foot Height: {right_foot_height[0]}")

        # Compute reward
        reward_left = torch.exp(-2000 * (left_foot_height - target_height_left) ** 2)
        reward_right = torch.exp(-2000 * (right_foot_height - target_height_right) ** 2)
        reward = (reward_left + reward_right) / 2

        return reward


@dataclass(kw_only=True)
class PhaseCommandTermCfg(CommandTermCfg):
  frequency: float = 1.3
  resampling_time_range: tuple[float, float] = (99999999.0, 99999999.0) # No resampling by default.

  debug_vis: bool = False

  def build(self, env: ManagerBasedRlEnv) -> PhaseCommand:
    return PhaseCommand(self, env)


class PhaseCommand(CommandTerm):
  def __init__(self, cfg: PhaseCommandTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.env = env
    self._phase: torch.Tensor = torch.zeros(
        env.num_envs,
        dtype=torch.float32, device=self.device,
    ) + 0.5
    self.metrics["phase"] = self._phase.clone()

  @property
  def command(self):
    return self._phase.unsqueeze(-1)

  def compute(self, dt):
    super().compute(dt)
    self._phase = (self._phase + 1 / self.cfg.frequency * self.env.step_dt) % 1.0

  def _update_metrics(self) -> None:
    self.metrics["phase"] = self._phase.clone()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._phase[env_ids] = 0.5

  def _update_command(self) -> None:
    pass

