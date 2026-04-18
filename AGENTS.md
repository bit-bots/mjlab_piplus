# AGENTS.md

## Branch Focus
- Primary development target on this branch: `Mjlab-WalkToBall-PiPlus`.
- Use `Mjlab-Velocity-PiPlus` only for explicit velocity work or comparisons.
- `README.md` and `CLAUDE.md` are velocity-first; default to WalkToBall on this branch unless told otherwise.

## Commands You Should Not Guess
- Setup: `uv sync`.
- Use `uv run ...` for project commands (do not run `python` directly).
- Default training command: `uv run train Mjlab-WalkToBall-PiPlus`.
- Long train/play runs should be launched in `tmux` with explicit GPU choice (see templates in `CLAUDE.md`).
- Verification order after edits:
  1. `uv run ruff format`
  2. `uv run ruff check --fix`
  3. `uv run ty check`
  4. `uv run pyright` (slower, thorough)

## Real Wiring Points
- Task discovery entry point: `pyproject.toml` under `[project.entry-points."mjlab.tasks"]`.
- Task registration and IDs: `src/mjlab_piplus/tasks/__init__.py`.
- Both registered tasks currently share PPO config from `src/mjlab_piplus/tasks/piplus_velocity_rl_cfg.py`.

## Edit Map (Where Changes Actually Go)
- WalkToBall behavior (obs/rewards/terminations/curriculum/events): `src/mjlab_piplus/tasks/piplus_walk_to_ball_env_cfg.py`.
- PPO hyperparameters/logging/checkpoint cadence: `src/mjlab_piplus/tasks/piplus_velocity_rl_cfg.py`.
- Robot actuators/home pose/collision presets: `src/mjlab_piplus/robot/piplus_constants.py`.
- Robot MJCF and naming basis for regex selectors: `src/mjlab_piplus/robot/piplus/piplus.xml`.

## Easy-To-Miss Gotchas
- `piplus_velocity_rl_cfg.py` name is misleading on this branch; it is also used by WalkToBall.
- WalkToBall includes stateful in-file components (`BallProgressReward`, `SuccessFacingBallTermination`); preserve reset/state behavior when editing.
- Joint/body/geom selection is regex-heavy; MJCF renames require synchronized regex updates in env/robot configs.
- `logs/` and `wandb/` are git-ignored (`.gitignore`), so run artifacts will not appear in git status.
