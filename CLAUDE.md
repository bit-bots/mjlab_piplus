# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a [mjlab](https://github.com/mujocolab/mjlab) project for training RL policies for the PiPlus humanoid robot. It uses MuJoCo Warp (GPU-accelerated MuJoCo) with a manager-based API similar to Isaac Lab, and RSL-RL for PPO training.

## Commands

Always use `uv run`, not `python` directly.

```bash
# Setup
uv sync

# Train — launch long runs in tmux, pick an explicit free GPU via nvidia-smi
# After launch, poll wandb/ dir for new run-YYYYMMDD_HHMMSS-* directories
# (verify timestamp matches current time, as runs take time to initialize).
# Inside the dir, find the run-<id>.wandb file — the ID is in the filename.
# Read the first few lines of that file to find a path like
# logs/rsl_rl/piplus_velocity/2026-04-12_23-56-38 — the last segment is the
# run name. Then prepend a line to runs.log (newest first):
# <name>: <id>: <short summary>
tmux new-session -d -s train "uv run train Mjlab-Velocity-PiPlus --env.scene.num-envs 4096 --gpu-ids '[GPU]'"

# Play/evaluate a trained policy — launch in tmux, pick an explicit free GPU via nvidia-smi
# Export CUDA_VISIBLE_DEVICES to a single GPU before launching.
# Prefer unused GPUs; if all busy, pick one with low utilization and free memory.
# When playing multiple runs, always run them SEQUENTIALLY — wait for each to
# finish before starting the next, not in parallel.
tmux new-session -d -s play "CUDA_VISIBLE_DEVICES=GPU uv run play Mjlab-Velocity-PiPlus --wandb-run-path jgueldenstein/mjlab_piplus/<id>"

# Sanity-check with dummy agents (no training needed)
uv run play Mjlab-Velocity-PiPlus --agent zero
uv run play Mjlab-Velocity-PiPlus --agent random

# Format and lint
uv run ruff format
uv run ruff check --fix

# Type check
uv run ty check    # fast
uv run pyright     # thorough
```

## Architecture

### Package layout

- `src/mjlab_piplus/robot/` — Robot definition: MuJoCo XML spec, actuator configs (PD gains, effort limits, delay), collision configs, and home keyframe pose.
- `src/mjlab_piplus/tasks/` — Task registration and configs: environment config (observations, rewards, terminations, curriculum) and RL runner config (PPO hyperparameters).
- Upstream `mjlab` is a uv-managed dependency from `pyproject.toml` (`tool.uv.sources`, rev `v1.5.0`); `rsl_rl` is used through mjlab/RSL-RL configs and is not vendored in this checkout.


### Task registration

Tasks are registered in `src/mjlab_piplus/tasks/__init__.py` via `register_mjlab_task()`. The package entry point `mjlab.tasks` (declared in `pyproject.toml`) causes mjlab's CLI to discover tasks automatically. The registered task IDs in this checkout are `"Mjlab-Velocity-PiPlus"` and `"Mjlab-GoalPose-PiPlus"`.

Pass/kick work is currently documented/prototyped (`docs/pass_kick_policy.md`, pass/kick symmetry and RL config helpers), but no pass/kick task is registered yet.

### Robot definition (`piplus_constants.py`)

Three motor series are used:
- **3536** — head joints (currently disabled in `PIPLUS_ARTICULATION`)
- **4438** — arm joints
- **5036** — leg joints (hip pitch/roll, thigh, calf, ankle pitch/roll), with per-joint tuned PD gains

Each actuator series uses `XmlActuatorCfg` with action delay controlled by `ACTUATOR_LAG_MIN` and `ACTUATOR_LAG_MAX` constants. `PIPLUS_ARTICULATION` currently enables only the leg actuator group (`ACTUATOR_5036`); head and arm actuator groups are defined/commented for future use. The `get_piplus_robot_cfg()` function assembles the full `EntityCfg` with the MuJoCo spec, collision model, and actuator list.

Collision configs (`FULL_COLLISION`, `LEG_ONLY_COLLISION`, `FEET_ONLY_COLLISION`) use regex matching on geom names ending in `_collision[0-9]`.

### Environment config (`piplus_velocity_env_cfg.py`)

The function `piplus_velocity_env_cfg(play=False)` builds a `ManagerBasedRlEnvCfg`. Key design choices:

- **Actor obs**: base angular velocity, projected gravity, joint pos/vel relative to default, last action, gait phase (sin/cos), velocity command. The sensor-derived terms (ang vel, gravity, joint pos/vel) are noised; actions, phase, and command are not.
- **Critic obs**: actor obs + base linear velocity, foot height (from `TerrainHeightSensorCfg`), foot air time, foot contact, foot contact forces. No noise.

> ⚠️ **Symmetry depends on the actor obs layout.** The left-right mirror in
> `symmetry.py` (`_transform_obs` / `_transform_obs_common`) is hard-coded by
> index, and each term has its own mirror rule (swap left/right joints, plus a
> sign flip for axial vs. polar quantities — e.g. ang-vel `[-1,1,-1]`, gravity
> `[1,-1,1]`, command `[1,-1,-1]`, phase half-period shift `[-1,-1]`). **Any
> change to the actor obs — term order, dimensions, or what a term measures —
> requires updating `_transform_obs` and `check_symmetry.py` to match.** The
> **critic obs does NOT need a mirror transform**: under data augmentation the
> mirrored sample reuses the original's value/return (`V(mirror(s)) == V(s)`),
> so the value function is only evaluated on the original obs.

- **Sensors**: `ContactSensorCfg` for feet ground contact (ankle roll link subtrees vs. terrain), self-collision sensor, terrain height scan anchored at foot sites.
- **Curriculum**: velocity command ranges ramp up in two stages, from step 0 to step 800 × 24 (`command_vel`).
- **Terminations**: time out (20 s episodes), orientation > 45° (`bad_orientation`). The root-height termination is currently commented out.
- **Simulation**: 5 ms timestep, decimation 4 (→ 50 Hz policy), MuJoCo Warp with 10 solver iterations.
- In `play` mode: curriculum disabled, episode length unbounded, extra air-time metrics added. Push events stay active.

### RL config (`piplus_velocity_rl_cfg.py`)

PPO with asymmetric actor-critic (actor/critic each 512-256-128 MLP, ELU activation). Left-right symmetry is enabled via task-specific symmetry configs (data augmentation + mirror loss, coeff 0.5). Logs to W&B project `mjlab_piplus`. 24 steps per env per rollout, up to 3 200 iterations, checkpoint every 200 iterations.

## Editing conventions

- All custom MDP terms (rewards, metrics, curriculums, etc.) go in
  `src/mjlab_piplus/tasks/mdp/` and are wired into the env cfg via
  `pi_mdp.<name>`. Do **not** edit `mjlab/` sources unless the user
  explicitly says to.

## Code style

- Line length: 88 columns.
- Avoid local imports unless necessary for circular import resolution.
- Joint name patterns use regex strings (e.g., `".*_hip_pitch_joint"`) — these are matched by MuJoCo name filters throughout mjlab.
