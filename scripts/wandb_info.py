#!/usr/bin/env -S uv run python
"""Fetch info about mjlab_piplus wandb runs.

Usage:
  uv run scripts/wandb_info.py list [N]        # list N most recent runs
  uv run scripts/wandb_info.py info <id>...    # summary + key config per run
  uv run scripts/wandb_info.py diff <id> <id>  # flat config diff between two runs
  uv run scripts/wandb_info.py metrics <id>... # final summary metrics
  uv run scripts/wandb_info.py config <id>     # full flattened config

All commands take wandb run IDs (8-char). Entity/project is hardcoded.
"""

from __future__ import annotations

import json
import sys

import wandb

ENTITY_PROJECT = "jgueldenstein/mjlab_piplus"


def _api():
  return wandb.Api()


def _flatten(d, prefix=""):
  out = {}
  if isinstance(d, dict):
    for k, v in d.items():
      out.update(_flatten(v, f"{prefix}.{k}" if prefix else k))
  elif isinstance(d, list):
    for i, v in enumerate(d):
      out.update(_flatten(v, f"{prefix}[{i}]"))
  else:
    out[prefix] = d
  return out


def _key_fields(cfg):
  """Extract fields useful for quick identification."""
  env = cfg.get("env_cfg", {})
  train = cfg.get("train_cfg", {})
  rewards = env.get("rewards", {})
  push = (
    env.get("events", {})
    .get("push_robot", {})
    .get("params", {})
    .get("velocity_range", {})
    or {}
  )
  cmd = env.get("commands", {}).get("twist", {})
  return {
    "max_iters": train.get("max_iterations"),
    "seed": train.get("seed") or env.get("seed"),
    "num_envs": env.get("scene", {}).get("num_envs"),
    "fine_w": rewards.get("track_linear_velocity_fine", {}).get("weight"),
    "track_lin_w": rewards.get("track_linear_velocity", {}).get("weight"),
    "track_ang_w": rewards.get("track_angular_velocity", {}).get("weight"),
    "track_ang_params": rewards.get("track_angular_velocity", {}).get("params"),
    "upright_w": rewards.get("upright", {}).get("weight"),
    "upright_params": rewards.get("upright", {}).get("params"),
    "push_x": push.get("x"),
    "push_y": push.get("y"),
    "push_z": push.get("z"),
    "push_roll": push.get("roll"),
    "push_pitch": push.get("pitch"),
    "push_yaw": push.get("yaw"),
    "cmd_heading": cmd.get("heading_command"),
    "cmd_rel_forward": cmd.get("rel_forward_envs"),
    "cmd_rel_heading": cmd.get("rel_heading_envs"),
    "cmd_rel_standing": cmd.get("rel_standing_envs"),
  }


def cmd_list(n=10):
  runs = _api().runs(ENTITY_PROJECT, order="-created_at", per_page=int(n))
  for r in list(runs)[: int(n)]:
    rt = r.summary.get("_runtime")
    rt_s = f"{rt / 60:.0f}min" if isinstance(rt, (int, float)) else "?"
    print(
      f"{r.id}  {r.name:25s}  {r.state:10s}  {rt_s:>7s}  iters={r.summary.get('_step', '?')}"
    )


def cmd_info(ids):
  for rid in ids:
    r = _api().run(f"{ENTITY_PROJECT}/{rid}")
    k = _key_fields(r.config)
    print(f"=== {rid}  {r.name}  {r.state} ===")
    for key, val in k.items():
      print(f"  {key}: {val}")
    print()


def cmd_metrics(ids):
  for rid in ids:
    r = _api().run(f"{ENTITY_PROJECT}/{rid}")
    s = r.summary._json_dict
    print(f"=== {rid}  {r.name} ===")
    for k in sorted(k for k in s if not k.startswith("_")):
      v = s[k]
      if isinstance(v, (int, float)):
        print(f"  {k}: {v:.4f}")
    print()


def cmd_diff(a, b):
  ca = _flatten(_api().run(f"{ENTITY_PROJECT}/{a}").config)
  cb = _flatten(_api().run(f"{ENTITY_PROJECT}/{b}").config)
  keys = set(ca) | set(cb)
  any_diff = False
  for k in sorted(keys):
    if ca.get(k) != cb.get(k):
      if k == "log_dir":
        continue
      print(f"  {k}:  {ca.get(k)!r}  ->  {cb.get(k)!r}")
      any_diff = True
  if not any_diff:
    print("(configs identical, ignoring log_dir)")


def cmd_config(rid):
  cfg = _api().run(f"{ENTITY_PROJECT}/{rid}").config
  print(json.dumps(cfg, indent=2, default=str))


def main():
  argv = sys.argv[1:]
  if not argv:
    print(__doc__)
    sys.exit(1)
  sub, *rest = argv
  if sub == "list":
    cmd_list(rest[0] if rest else 10)
  elif sub == "info":
    cmd_info(rest)
  elif sub == "metrics":
    cmd_metrics(rest)
  elif sub == "diff":
    cmd_diff(*rest)
  elif sub == "config":
    cmd_config(rest[0])
  else:
    print(__doc__)
    sys.exit(1)


if __name__ == "__main__":
  main()
