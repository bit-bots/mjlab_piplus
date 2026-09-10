"""The evaluation env: the multi task, made repeatable.

Built by taking ``piplus_multi_env_cfg()`` and changing only what stops a run
from being a measurement:

* **The commands are replayed, not sampled.** Every sub-command is swapped for
  its ``Eval*`` counterpart, which reads its parameters from the command book.
  The sub-command configs are *copied field by field* from the training ones,
  so the ranges, geometry and observation widths cannot drift apart from what
  the policy was trained on.
* **One command type per episode.** The resampling intervals are pushed past
  the episode length, so a command is drawn at the reset and then held: the
  first experiment asks how well a single command is executed, without the
  hand-over between types in the way.
* **The episode length is per type.** A goal up to 8 m away and a 3-second
  velocity command need very different clocks; ``eval_time_out`` reads the
  active type's own limit.
* **The scenario is the only variable.** Domain randomization, pushes and
  observation noise are all off by default (each behind its own flag), and the
  reset is deterministic -- fixed spawn yaw, no joint jitter -- so two policies
  meet the same episode, not the same distribution of episodes.

Everything else -- the robot, the observations, the action term, the sensors,
the sim -- is the training config untouched. Note the consequence of the
defaults: these numbers are *not* a robustness measurement. Turn the flags on
for that, and say which run had them on.
"""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from mjlab_piplus.tasks.eval.command_book import DEFAULT_BOOK_PATH
from mjlab_piplus.tasks.eval.eval_commands import (
  EvalFootstepCommandCfg,
  EvalGoalPoseCommandCfg,
  EvalKickCommandCfg,
  EvalMultiCommandCfg,
  EvalPathCommandCfg,
  EvalStandCommandCfg,
  EvalVelocityCommandCfg,
  eval_time_out,
)
from mjlab_piplus.tasks.mdp.footstep.footstep_command import (
  FootstepCommandCfg,
  PathLibCfg,
)
from mjlab_piplus.tasks.mdp.multi_command import MultiCommandCfg
from mjlab_piplus.tasks.piplus_multi_env_cfg import piplus_multi_env_cfg

__all__ = [
  "EVAL_EPISODE_LENGTH_S",
  "EVAL_PATHLIB_SIZE",
  "piplus_multi_eval_env_cfg",
]

# The A* library the booked ``path_index`` values index into. Matches the multi
# task's path command, so the evaluation walks paths from the same distribution
# the policy trained on. The footstep command's library is pinned to the same
# size here, which also means the two share one file, one cache entry and one
# copy in device memory (the training default of 50 000 is a 1.9 GB file).
EVAL_PATHLIB_SIZE = 2000

# How long each command type gets. Sized from what the command can ask for: a
# path or a goal pose is up to 8 m away at a top speed of 0.8 m/s, plus the
# turning and the settling at the end; a velocity command and a footstep plan
# are steady-state behaviors that say everything they have to say in 20 s; a
# kick is placed at most 1.2 m away and its scoring window is 1 s; standing has
# nothing to wait for.
EVAL_EPISODE_LENGTH_S: dict[str, float] = {
  "velocity": 20.0,
  "path": 45.0,
  "goal_pose": 30.0,
  "footstep": 20.0,
  "kick": 20.0,
  "stand": 10.0,
}

# Where the sub-command types map onto their eval subclass.
_EVAL_CFG_CLASSES: dict[str, type[CommandTermCfg]] = {
  "velocity": EvalVelocityCommandCfg,
  "path": EvalPathCommandCfg,
  "goal_pose": EvalGoalPoseCommandCfg,
  "footstep": EvalFootstepCommandCfg,
  "kick": EvalKickCommandCfg,
  "stand": EvalStandCommandCfg,
}


def _as_eval_cfg(
  name: str, source: CommandTermCfg, overrides: dict[str, Any]
) -> CommandTermCfg:
  """Rebuild a training sub-command config as its ``Eval*`` subclass.

  Copied field by field rather than restated: a range added to (or changed in)
  the training config is then automatically part of the evaluation, and the two
  cannot silently disagree about what the command means.
  """
  eval_cls = _EVAL_CFG_CLASSES.get(name)
  if eval_cls is None:
    raise KeyError(
      f"Sub-command '{name}' has no eval counterpart in _EVAL_CFG_CLASSES. "
      "Add one (and a generator in command_book.py) before evaluating it."
    )
  kwargs = {f.name: getattr(source, f.name) for f in fields(source)}
  kwargs.update(overrides)
  return eval_cls(**kwargs)  # type: ignore[arg-type]


def piplus_multi_eval_env_cfg(
  *,
  book_path: str = str(DEFAULT_BOOK_PATH),
  forced_type: str | None = None,
  num_envs: int = 64,
  episode_length_s: dict[str, float] | None = None,
  domain_rand: bool = False,
  push: bool = False,
  obs_noise: bool = False,
  velocity_warmup_s: float = 1.0,
  goal_pos_threshold: float = 0.3,
  goal_yaw_threshold: float = 0.3,
  pathlib_size: int = EVAL_PATHLIB_SIZE,
) -> ManagerBasedRlEnvCfg:
  cfg = piplus_multi_env_cfg()
  lengths = dict(
    EVAL_EPISODE_LENGTH_S if episode_length_s is None else episode_length_s
  )

  multi_cfg = cfg.commands["command"]
  assert isinstance(multi_cfg, MultiCommandCfg), (
    "The eval env is built from the multi task's MultiCommand, got "
    f"{type(multi_cfg).__name__}."
  )
  terms: dict[str, CommandTermCfg] = {}
  for name, sub_cfg in multi_cfg.terms.items():
    # Past the episode length, so the command is drawn once at the reset and
    # then held: this experiment measures one command, not the hand-over.
    hold_s = 2.0 * lengths.get(name, 20.0) + 10.0
    overrides: dict[str, Any] = {"resampling_time_range": (hold_s, hold_s)}
    if name == "velocity":
      overrides["warmup_s"] = velocity_warmup_s
    elif name == "goal_pose":
      overrides["goal_pos_threshold"] = goal_pos_threshold
      overrides["goal_yaw_threshold"] = goal_yaw_threshold
    elif name == "path":
      overrides["goal_pos_threshold"] = goal_pos_threshold
      overrides["pathlib_size"] = pathlib_size
    elif name == "footstep":
      assert isinstance(sub_cfg, FootstepCommandCfg)
      # The plan has to outlast the hold interval at two footsteps per gait
      # period -- FootstepCommandCfg validates exactly this.
      overrides["max_steps"] = math.ceil(2.0 * sub_cfg.frequency * hold_s) + 2
      # The booked row decides whether a plan is path- or Gaussian-shaped;
      # path_fraction only still decides whether the library is loaded at all.
      overrides["path_fraction"] = 1.0
      pathlib_cfg = PathLibCfg(
        **{f.name: getattr(sub_cfg.pathlib, f.name) for f in fields(sub_cfg.pathlib)}
      )
      pathlib_cfg.pathlib_size = pathlib_size
      overrides["pathlib"] = pathlib_cfg
    terms[name] = _as_eval_cfg(name, sub_cfg, overrides)

  cfg.commands = {
    "command": EvalMultiCommandCfg(
      debug_vis=multi_cfg.debug_vis,
      terms=terms,
      book_path=book_path,
      forced_type=forced_type,
      episode_length_s=lengths,
      velocity_command_name=multi_cfg.velocity_command_name,
      stand_command_name=multi_cfg.stand_command_name,
    )
  }

  ##
  # Events: a deterministic reset, and nothing else unless asked for.
  ##
  events = {
    "reset_base": cfg.events["reset_base"],
    "reset_robot_joints": cfg.events["reset_robot_joints"],
  }
  # A random spawn yaw would make every command's difficulty a function of
  # which way the robot happened to face; the booked goals, paths and kick
  # directions are all relative to the heading, so pinning it costs nothing.
  events["reset_base"].params["pose_range"] = {
    **events["reset_base"].params["pose_range"],
    "yaw": (0.0, 0.0),
  }
  events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
  events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
  if domain_rand:
    events.update(
      {
        name: term
        for name, term in cfg.events.items()
        if term.mode == "startup"  # The DR set; see piplus_multi_env_cfg.
      }
    )
  if push:
    events["push_robot"] = cfg.events["push_robot"]
  cfg.events = events

  if not obs_noise:
    # Noise is resampled every step, so leaving it on makes the measurement a
    # function of the RNG as well as of the policy. The one-step sensor delay
    # stays: it is fixed at 1 here (DELAY_MIN == DELAY_MAX) and is part of the
    # plant, not of the noise.
    for group in cfg.observations.values():
      for term in group.terms.values():
        term.noise = None

  ##
  # Terminations: a fall ends the episode and is reported as a fall.
  ##
  cfg.terminations["time_out"] = TerminationTermCfg(
    func=eval_time_out, time_out=True, params={"command_name": "command"}
  )

  cfg.curriculum = {}
  cfg.scene.num_envs = num_envs
  # The env-wide cap only has to cover the longest per-type clock; the actual
  # limit is enforced per type by ``eval_time_out``.
  cfg.episode_length_s = max(lengths.values())
  return cfg
