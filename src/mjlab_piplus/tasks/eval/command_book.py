"""The pre-generated command book the evaluation replays.

An evaluation is only comparable between two policies if both are handed the
*same* commands, so nothing here is sampled at run time. A book holds
``num_commands`` fully specified commands per command type, generated once by
``generate_commands.py`` and written to a JSON file; during evaluation an
episode counter picks the row (see ``EvalMultiCommand.command_index``), so
episode ``k`` of a type always replays command ``k`` no matter which policy is
being evaluated, how many envs are running, or in which order they finish.

What a "command" is differs per type, and the row is the *sampled parameters*
of the corresponding training command term, not its observation:

``velocity``   ``lin_vel_x``/``lin_vel_y``/``ang_vel_z``: the commanded twist,
               held for the whole episode.
``path``       ``path_index`` into the A* path library plus a goal yaw. The
               library itself is deterministic given its generation
               parameters, so the index names one specific path -- ``meta``
               records the library size the indices were drawn against, and
               the eval term refuses a mismatch.
``goal_pose``  Polar offset from the robot at reset (``distance``,
               ``bearing``) plus a goal yaw, all *relative to the robot's
               heading*, so the scenario does not depend on the spawn yaw.
``footstep``   The plan, as either a Gaussian scenario (a per-episode nominal
               stride ``mu`` plus jitter regenerated from ``jitter_seed``) or
               a path scenario (``path_index`` + ``step_length``). The jitter
               is regenerated rather than stored so the file stays readable:
               ``footstep_deltas`` is the single definition of that mapping,
               and it runs on CPU numpy so it is identical on every device.
``kick``       Where the ball is placed relative to the robot (``distance``,
               ``bearing``) and what is asked of the kick (``direction``,
               ``strength``). Every ball life in the episode reuses the row,
               so a second kick is the same task as the first.
``stand``      Nothing to sample; the rows are empty and exist only so the
               type has a consistent length.

Ranges are read off the training env cfg (see :func:`generate_book`) rather
than restated here, so the evaluation covers what the policy was trained on.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
  "BOOK_VERSION",
  "COMMAND_TYPES",
  "CommandBook",
  "DEFAULT_BOOK_PATH",
  "footstep_deltas",
  "generate_book",
]

# Every command type of the multi task, in the order an "all commands after one
# another" evaluation runs them.
COMMAND_TYPES = ("velocity", "path", "goal_pose", "footstep", "kick", "stand")

BOOK_VERSION = 1

DEFAULT_BOOK_PATH = Path("eval_commands.json")


def _rng(seed: int, type_name: str) -> np.random.Generator:
  """A generator seeded per type, so the types are independent.

  Deriving each type's stream from the type *name* rather than from a running
  counter means adding or reordering a type leaves the other types' commands
  bit-for-bit unchanged -- an evaluation run before the change stays comparable
  with one run after it.
  """
  return np.random.default_rng([seed, zlib.crc32(type_name.encode())])


@dataclass(frozen=True)
class CommandBook:
  version: int
  seed: int
  num_commands: int
  meta: dict[str, Any]
  commands: dict[str, list[dict[str, Any]]]

  def rows(self, type_name: str) -> list[dict[str, Any]]:
    if type_name not in self.commands:
      raise KeyError(
        f"Command book has no rows for '{type_name}'. It holds: "
        f"{sorted(self.commands)}."
      )
    return self.commands[type_name]

  def column(self, type_name: str, key: str) -> np.ndarray:
    """One field of one type as a ``(num_commands,)`` array."""
    return np.array([row[key] for row in self.rows(type_name)])

  def save(self, path: str | Path) -> None:
    """Write the book as JSON, one command per line.

    ``json.dump(indent=...)`` would put every scalar on its own line and turn a
    1000-command book into ~40k lines; one line per command keeps it diffable
    and greppable.
    """
    path = Path(path)
    parts = [
      '{\n  "version": %d,\n  "seed": %d,\n  "num_commands": %d,'
      % (self.version, self.seed, self.num_commands),
      '\n  "meta": %s,' % json.dumps(self.meta, sort_keys=True),
      '\n  "commands": {\n',
    ]
    type_blocks = []
    for type_name in self.commands:
      rows = ",\n".join(
        f"      {json.dumps(row, sort_keys=True)}" for row in self.commands[type_name]
      )
      type_blocks.append(f'    "{type_name}": [\n{rows}\n    ]')
    parts.append(",\n".join(type_blocks))
    parts.append("\n  }\n}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts))

  @classmethod
  def load(cls, path: str | Path) -> CommandBook:
    path = Path(path)
    if not path.exists():
      raise FileNotFoundError(
        f"No command book at {path}. Generate one first:\n"
        f"  uv run python -m mjlab_piplus.tasks.eval.generate_commands "
        f"--out {path}"
      )
    data = json.loads(path.read_text())
    if data.get("version") != BOOK_VERSION:
      raise ValueError(
        f"Command book {path} has version {data.get('version')}, this code "
        f"reads version {BOOK_VERSION}. Regenerate it."
      )
    book = cls(
      version=data["version"],
      seed=data["seed"],
      num_commands=data["num_commands"],
      meta=data.get("meta", {}),
      commands=data["commands"],
    )
    for type_name, rows in book.commands.items():
      if len(rows) != book.num_commands:
        raise ValueError(
          f"Command book {path}: type '{type_name}' has {len(rows)} rows, "
          f"expected num_commands={book.num_commands}."
        )
    return book


##
# Footstep plans.
##


def footstep_deltas(
  row: dict[str, Any],
  *,
  max_steps: int,
  delta_std: tuple[float, ...],
) -> np.ndarray | None:
  """The per-footstep deltas of a Gaussian-scenario row, ``(max_steps, 6)``.

  Returns ``None`` for a path scenario, whose footsteps are read off the path
  library on device instead (see ``EvalFootstepCommand``).

  Run on CPU numpy from the row's own seed, so the plan is identical on every
  device and in every process -- the same reason the jitter is regenerated here
  rather than stored: 96 x 6 numbers per command would be most of the file.
  """
  if row["mode"] != "gaussian":
    return None
  rng = np.random.default_rng(int(row["jitter_seed"]))
  mu = np.asarray(row["mu"], dtype=np.float64)
  noise = rng.standard_normal((max_steps, len(mu)))
  return mu[None, :] + noise * np.asarray(delta_std, dtype=np.float64)[None, :]


##
# Generation.
##


def _gen_velocity(cfg, rng: np.random.Generator, n: int) -> list[dict[str, Any]]:
  """A twist held for the whole episode, uniform over the training box.

  The training term also draws walk-in-place, heading-driven, world-frame and
  forward-only envs; none of those are replayed here. They change *what the
  command means* rather than what it is, and a tracking number is only readable
  when every episode is the same kind of ask: hold this twist.
  """
  r = cfg.ranges
  return [
    {
      "lin_vel_x": round(float(rng.uniform(*r.lin_vel_x)), 4),
      "lin_vel_y": round(float(rng.uniform(*r.lin_vel_y)), 4),
      "ang_vel_z": round(float(rng.uniform(*r.ang_vel_z)), 4),
    }
    for _ in range(n)
  ]


def _gen_path(
  cfg, rng: np.random.Generator, n: int, pathlib_size: int
) -> list[dict[str, Any]]:
  return [
    {
      "path_index": int(rng.integers(0, pathlib_size)),
      "goal_yaw": round(float(rng.uniform(*cfg.goal_yaw_range)), 4),
    }
    for _ in range(n)
  ]


def _gen_goal_pose(cfg, rng: np.random.Generator, n: int) -> list[dict[str, Any]]:
  return [
    {
      "distance": round(float(rng.uniform(*cfg.radius_range)), 4),
      "bearing": round(float(rng.uniform(-math.pi, math.pi)), 4),
      "goal_yaw": round(float(rng.uniform(*cfg.yaw_range)), 4),
    }
    for _ in range(n)
  ]


def _gen_footstep(
  cfg, rng: np.random.Generator, n: int, pathlib_size: int
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for _ in range(n):
    if rng.random() < cfg.path_fraction:
      rows.append(
        {
          "mode": "path",
          "path_index": int(rng.integers(0, pathlib_size)),
          "step_length": round(float(rng.uniform(*cfg.path_step_length_range)), 4),
        }
      )
    else:
      mu = [round(float(rng.uniform(lo, hi)), 4) for lo, hi in cfg.nominal_delta_range]
      rows.append(
        {
          "mode": "gaussian",
          "mu": mu,
          # The jitter itself is regenerated from this seed; see
          # ``footstep_deltas``.
          "jitter_seed": int(rng.integers(0, 2**31 - 1)),
        }
      )
  return rows


def _gen_kick(cfg, rng: np.random.Generator, n: int) -> list[dict[str, Any]]:
  cone = math.radians(cfg.placement_cone_deg)
  dir_cone = math.radians(cfg.direction_cone_deg)
  return [
    {
      "distance": round(float(rng.uniform(*cfg.distance_range)), 4),
      "bearing": round(float(rng.uniform(-cone, cone)), 4),
      "direction": round(float(rng.uniform(-dir_cone, dir_cone)), 4),
      "strength": round(float(rng.uniform(*cfg.strength_range)), 4),
    }
    for _ in range(n)
  ]


def generate_book(
  env_cfg,
  *,
  num_commands: int,
  seed: int,
  pathlib_size: int,
) -> CommandBook:
  """Generate a book from the *training* env cfg's sampling ranges.

  Reading the ranges off ``env_cfg`` rather than restating them keeps the
  evaluation inside the distribution the policy was trained on: widening a
  training range and regenerating the book widens the evaluation with it.

  ``pathlib_size`` bounds the path indices, and has to be the size the
  evaluation env actually loads -- the i-th path of a 2000-path library is not
  the i-th path of a 50000-path one.
  """
  multi = env_cfg.commands["command"]
  terms = multi.terms
  book_commands: dict[str, list[dict[str, Any]]] = {
    "velocity": _gen_velocity(terms["velocity"], _rng(seed, "velocity"), num_commands),
    "path": _gen_path(terms["path"], _rng(seed, "path"), num_commands, pathlib_size),
    "goal_pose": _gen_goal_pose(
      terms["goal_pose"], _rng(seed, "goal_pose"), num_commands
    ),
    "footstep": _gen_footstep(
      terms["footstep"], _rng(seed, "footstep"), num_commands, pathlib_size
    ),
    "kick": _gen_kick(terms["kick"], _rng(seed, "kick"), num_commands),
    "stand": [{} for _ in range(num_commands)],
  }
  missing = set(multi.terms) - set(book_commands)
  if missing:
    raise ValueError(
      f"The multi command has type(s) {sorted(missing)} with no generator in "
      "command_book.py. Add one before evaluating them."
    )
  footstep_cfg = terms["footstep"]
  meta = {
    "pathlib_size": pathlib_size,
    "footstep_delta_std": list(footstep_cfg.delta_std),
    "footstep_max_step_delta": list(footstep_cfg.max_step_delta),
    "velocity_ranges": {
      "lin_vel_x": list(terms["velocity"].ranges.lin_vel_x),
      "lin_vel_y": list(terms["velocity"].ranges.lin_vel_y),
      "ang_vel_z": list(terms["velocity"].ranges.ang_vel_z),
    },
    "goal_pose_radius_range": list(terms["goal_pose"].radius_range),
    "kick_distance_range": list(terms["kick"].distance_range),
    "kick_strength_range": list(terms["kick"].strength_range),
  }
  return CommandBook(
    version=BOOK_VERSION,
    seed=seed,
    num_commands=num_commands,
    meta=meta,
    commands=book_commands,
  )
