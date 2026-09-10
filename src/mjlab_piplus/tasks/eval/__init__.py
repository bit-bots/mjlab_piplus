"""Deterministic evaluation of the multi-command policy.

``command_book`` defines and generates the commands, ``eval_commands`` replays
them and measures what happened, ``eval_env_cfg`` assembles the env, and
``run_eval`` drives it. See ``eval_env_cfg`` for what the evaluation env
changes about the training env, and ``eval_commands`` for how an episode
counter selects a command.
"""

from mjlab_piplus.tasks.eval.command_book import (
  COMMAND_TYPES,
  DEFAULT_BOOK_PATH,
  CommandBook,
  generate_book,
)
from mjlab_piplus.tasks.eval.eval_commands import EvalMultiCommand, EvalMultiCommandCfg
from mjlab_piplus.tasks.eval.eval_env_cfg import (
  EVAL_EPISODE_LENGTH_S,
  EVAL_PATHLIB_SIZE,
  piplus_multi_eval_env_cfg,
)

__all__ = [
  "COMMAND_TYPES",
  "DEFAULT_BOOK_PATH",
  "EVAL_EPISODE_LENGTH_S",
  "EVAL_PATHLIB_SIZE",
  "CommandBook",
  "EvalMultiCommand",
  "EvalMultiCommandCfg",
  "generate_book",
  "piplus_multi_eval_env_cfg",
]
