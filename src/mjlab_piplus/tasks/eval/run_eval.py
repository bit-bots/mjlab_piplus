"""Run the multi-command evaluation and write per-episode results.

    CUDA_VISIBLE_DEVICES=2 uv run python -m mjlab_piplus.tasks.eval.run_eval \\
      --wandb-run-path jgueldenstein/mjlab_piplus/<id>

By default every command type is evaluated in turn, ``--runs-per-command``
episodes each, replaying rows 0..runs-1 of that type's command book. Pass
``--types velocity kick`` for a subset.

One env instance serves every type: the type is re-pinned between passes and
the envs reset, which is far cheaper than rebuilding the simulation each time.
Env ``e`` runs command indices ``e, e + N, e + 2N, ...`` (``N`` envs), so which
episode gets which command never depends on the policy -- two policies see the
same scenarios in the same order.

Outputs, under ``--out``:

* ``episodes_<type>.csv``  one row per episode, every measured quantity.
* ``summary.json``         aggregates per type, plus the full run configuration.

and a summary table on stdout.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

from mjlab_piplus.tasks.eval.command_book import COMMAND_TYPES, DEFAULT_BOOK_PATH
from mjlab_piplus.tasks.eval.eval_commands import EvalMultiCommand
from mjlab_piplus.tasks.eval.eval_env_cfg import piplus_multi_eval_env_cfg

MULTI_TASK_ID = "Mjlab-Multi-PiPlus"

# The few numbers worth putting on stdout per type. Everything else is in the
# CSV; this is the "did it work" line, not the analysis.
HEADLINE: dict[str, tuple[str, ...]] = {
  "velocity": ("err_vel_xy", "err_yaw_rate", "fell"),
  "path": ("mean_path_dist", "max_path_dist", "time_to_goal", "reached_goal", "fell"),
  "goal_pose": ("time_to_goal", "reached_goal", "final_dist", "fell"),
  "footstep": ("step_err_xy", "step_err_yaw", "footsteps", "fell"),
  "kick": ("kick_speed_error", "kick_aim_error", "kicked", "kick_right_foot", "fell"),
  "stand": ("drift", "yaw_drift", "posture_rms", "fell"),
}


@dataclass(frozen=True)
class EvalConfig:
  wandb_run_path: str | None = None
  """W&B run holding the checkpoint, e.g. jgueldenstein/mjlab_piplus/<id>."""
  wandb_checkpoint_name: str | None = None
  """Checkpoint within that run (e.g. model_4000.pt). Default: the latest."""
  checkpoint_file: str | None = None
  """A local checkpoint, instead of a W&B run."""
  agent: Literal["trained", "zero", "random"] = "trained"
  """``zero`` / ``random`` need no checkpoint and are the floor to read
  against: they say what each metric looks like for a policy doing nothing."""

  commands: Path = DEFAULT_BOOK_PATH
  """The command book to replay."""
  types: tuple[str, ...] = COMMAND_TYPES
  """Command types to evaluate, in order. Defaults to all of them."""
  runs_per_command: int = 100
  """Episodes per command type, replaying book rows 0..runs-1."""
  num_envs: int = 64
  """Parallel envs. Changes the throughput, and which env runs which command,
  but not the set of commands evaluated."""

  domain_rand: bool = False
  """Enable the startup domain randomization."""
  push: bool = False
  """Enable the push disturbance."""
  obs_noise: bool = False
  """Enable observation noise. Off by default so the number is a property of
  the policy rather than of the RNG."""

  episode_length_s: dict[str, float] = field(default_factory=dict)
  """Per-type episode length overrides (seconds)."""

  device: str | None = None
  seed: int = 0
  """Seeds torch. Only matters when a randomizing flag above is on."""
  out: Path | None = None
  """Output directory. Default: logs/eval/<checkpoint tag>_<timestamp>."""
  log_root: str = "logs/rsl_rl"
  """Where W&B checkpoints are cached, matching the play script."""


def _resolve_checkpoint(cfg: EvalConfig, experiment_name: str) -> Path | None:
  if cfg.agent != "trained":
    return None
  if cfg.checkpoint_file is not None:
    path = Path(cfg.checkpoint_file)
    if not path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return path
  if cfg.wandb_run_path is None:
    raise ValueError(
      "Pass --wandb-run-path or --checkpoint-file (or --agent zero/random)."
    )
  log_root = (Path(cfg.log_root) / experiment_name).resolve()
  path, cached = get_wandb_checkpoint_path(
    log_root, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
  )
  print(f"[INFO] Checkpoint {path.name} ({'cached' if cached else 'downloaded'})")
  return path


def _build_policy(
  cfg: EvalConfig, env, agent_cfg, device: str, checkpoint: Path | None
):
  if cfg.agent == "zero":
    shape = env.unwrapped.action_space.shape
    return lambda obs: torch.zeros(shape, device=device)
  if cfg.agent == "random":
    shape = env.unwrapped.action_space.shape
    return lambda obs: 2 * torch.rand(shape, device=device) - 1
  assert checkpoint is not None
  runner_cls = load_runner_cls(MULTI_TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  return runner.get_inference_policy(device=device)


def _run_type(
  env: ManagerBasedRlEnv,
  wrapped,
  policy,
  multi: EvalMultiCommand,
  type_name: str,
  runs: int,
) -> list[dict[str, float]]:
  """Evaluate one command type; returns one record per booked command."""
  multi.set_forced_type(type_name)
  multi.restart()
  wrapped.reset()

  num_envs = env.num_envs
  # Env e runs commands e, e+N, ...; every env must finish this many episodes
  # for rows 0..runs-1 to all have been played.
  rounds = math.ceil(runs / num_envs)
  records: dict[int, dict[str, float]] = {}
  started = time.time()
  last_print = 0.0

  while int(multi.episode_index.min()) < rounds:
    with torch.no_grad():
      obs = wrapped.get_observations()
      wrapped.step(policy(obs))
    for batch_type, batch in multi.drain():
      # A record is only this pass's if it was measured under this pass's
      # command type; see EvalMultiCommand.restart.
      if batch_type != type_name:
        continue
      indices = batch["command_index"].numpy().astype(int)
      for row, index in enumerate(indices):
        if index >= runs or index in records:
          continue
        records[index] = {key: float(value[row].item()) for key, value in batch.items()}
    elapsed = time.time() - started
    if elapsed - last_print > 5.0:
      last_print = elapsed
      done = min(int(multi.episode_index.min()), rounds)
      print(
        f"  {type_name}: {len(records)}/{runs} episodes "
        f"({done}/{rounds} rounds, {elapsed:.0f}s)",
        flush=True,
      )
  print(
    f"  {type_name}: {len(records)}/{runs} episodes in {time.time() - started:.0f}s"
  )
  missing = [i for i in range(runs) if i not in records]
  if missing:
    raise RuntimeError(
      f"{type_name}: no record for command(s) {missing[:5]}"
      f"{'...' if len(missing) > 5 else ''}. This is a bug in the latching."
    )
  return [records[i] for i in range(runs)]


def _aggregate(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
  """Mean / std / median / p90 per field, ignoring the NaNs.

  A NaN is "this episode never made the measurement" -- no footstep landed, the
  goal was never reached, the ball was never touched -- so it must not count as
  a zero. ``n`` says how many episodes did make it, which is itself a result.
  """
  summary: dict[str, dict[str, float]] = {}
  for key in records[0]:
    values = np.array([record[key] for record in records], dtype=float)
    valid = values[~np.isnan(values)]
    if valid.size == 0:
      summary[key] = {"n": 0}
      continue
    summary[key] = {
      "n": int(valid.size),
      "mean": float(valid.mean()),
      "std": float(valid.std()),
      "median": float(np.median(valid)),
      "p90": float(np.percentile(valid, 90)),
      "min": float(valid.min()),
      "max": float(valid.max()),
    }
  return summary


def _print_summary(summaries: dict[str, dict[str, dict[str, float]]]) -> None:
  print("\n" + "=" * 78)
  for type_name, summary in summaries.items():
    fields_ = [key for key in HEADLINE.get(type_name, ()) if key in summary]
    print(f"\n{type_name}  ({summary['steps']['n']} episodes)")
    for key in fields_:
      stat = summary[key]
      if stat["n"] == 0:
        print(f"  {key:<20} never measured")
        continue
      print(
        f"  {key:<20} mean {stat['mean']:8.3f}   median {stat['median']:8.3f}"
        f"   p90 {stat['p90']:8.3f}   n {stat['n']:>4d}"
      )
  print("\n" + "=" * 78)


def main(cfg: EvalConfig) -> None:
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  unknown = [name for name in cfg.types if name not in COMMAND_TYPES]
  if unknown:
    raise SystemExit(f"Unknown command type(s) {unknown}. Known: {COMMAND_TYPES}.")

  agent_cfg = load_rl_cfg(MULTI_TASK_ID)
  checkpoint = _resolve_checkpoint(cfg, agent_cfg.experiment_name)

  tag = cfg.agent
  if checkpoint is not None:
    tag = f"{checkpoint.parent.name}_{checkpoint.stem}"
  out = cfg.out or Path("logs/eval") / f"{tag}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
  out.mkdir(parents=True, exist_ok=True)

  env_cfg = piplus_multi_eval_env_cfg(
    book_path=str(cfg.commands),
    forced_type=cfg.types[0],
    num_envs=cfg.num_envs,
    episode_length_s=cfg.episode_length_s or None,
    domain_rand=cfg.domain_rand,
    push=cfg.push,
    obs_noise=cfg.obs_noise,
  )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  policy = _build_policy(cfg, wrapped, agent_cfg, device, checkpoint)

  multi = env.command_manager.get_term("command")
  assert isinstance(multi, EvalMultiCommand)
  multi.collect_records = True

  summaries: dict[str, dict[str, dict[str, float]]] = {}
  for type_name in cfg.types:
    records = _run_type(env, wrapped, policy, multi, type_name, cfg.runs_per_command)
    summaries[type_name] = _aggregate(records)
    csv_path = out / f"episodes_{type_name}.csv"
    with csv_path.open("w", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(records[0]))
      writer.writeheader()
      writer.writerows(records)
    print(f"  -> {csv_path}")

  payload: dict[str, Any] = {
    "config": {
      **{
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in asdict(cfg).items()
      },
      "device": device,
      "checkpoint": str(checkpoint) if checkpoint else None,
      "book": str(cfg.commands),
    },
    "summary": summaries,
  }
  (out / "summary.json").write_text(json.dumps(payload, indent=2))
  _print_summary(summaries)
  print(f"Results in {out}")
  env.close()


if __name__ == "__main__":
  main(tyro.cli(EvalConfig))
