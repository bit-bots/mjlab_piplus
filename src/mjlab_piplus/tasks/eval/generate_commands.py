"""Generate the evaluation command book.

    uv run python -m mjlab_piplus.tasks.eval.generate_commands

Writes ``eval_commands.json`` (1000 commands per type by default) from the
*training* env cfg's own sampling ranges. Regenerate it after changing a
command's ranges -- and note that regenerating invalidates comparisons with
results measured against the old book, so the book is a checked-in artifact,
not a per-run temp file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from mjlab_piplus.tasks.eval.command_book import (
  COMMAND_TYPES,
  DEFAULT_BOOK_PATH,
  generate_book,
)
from mjlab_piplus.tasks.eval.eval_env_cfg import EVAL_PATHLIB_SIZE
from mjlab_piplus.tasks.piplus_multi_env_cfg import piplus_multi_env_cfg


@dataclass(frozen=True)
class GenerateConfig:
  out: Path = DEFAULT_BOOK_PATH
  """Where to write the book."""
  count: int = 1000
  """Commands generated per command type."""
  seed: int = 0
  """Master seed. Each type's stream is derived from it and the type's name, so
  the types are independent of each other and of the order they are listed."""
  pathlib_size: int = EVAL_PATHLIB_SIZE
  """Size of the A* path library the ``path`` / ``footstep`` indices are drawn
  against. Must match what the evaluation env loads."""
  force: bool = False
  """Overwrite an existing book. Off by default: an overwritten book silently
  invalidates every result measured against it."""


def main(cfg: GenerateConfig) -> None:
  if cfg.out.exists() and not cfg.force:
    raise SystemExit(
      f"{cfg.out} already exists. Results measured against it are only "
      "comparable to results from the same book -- pass --force to overwrite."
    )
  book = generate_book(
    piplus_multi_env_cfg(),
    num_commands=cfg.count,
    seed=cfg.seed,
    pathlib_size=cfg.pathlib_size,
  )
  book.save(cfg.out)
  print(f"Wrote {cfg.count} commands per type to {cfg.out}")
  for type_name in COMMAND_TYPES:
    rows = book.rows(type_name)
    print(f"  {type_name:<10} {len(rows):>5} e.g. {rows[0]}")


if __name__ == "__main__":
  main(tyro.cli(GenerateConfig))
