# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a [mjlab](https://github.com/mujocolab/mjlab) project for training RL policies for the PiPlus humanoid robot. It uses MuJoCo Warp (GPU-accelerated MuJoCo) with a manager-based API similar to Isaac Lab, and RSL-RL for PPO training.

## Commands

Always use `uv run`, not `python` directly.

```sh
# <task-id> is any registered task (see "Task registration"), e.g.
# Mjlab-Velocity-PiPlus or Mjlab-Multi-PiPlus. <experiment> is that task's
# experiment_name in its RL cfg, e.g. piplus_velocity or piplus_multi.
#
# GPU selection (train and play alike): pin each process to exactly ONE GPU
# with CUDA_VISIBLE_DEVICES=<n> — one GPU per run is the most efficient setup.
# Pick it via nvidia-smi: prefer an unused GPU; if all are busy, one with low
# utilization and free memory.

# Train — launch as a background process via Bash run_in_background=true (NOT tmux).
CUDA_VISIBLE_DEVICES=<n> uv run train <task-id> --env.scene.num-envs 4096
# After launch, find the new run's W&B ID and run name:
# - Preferred: use the W&B Claude Code skill (e.g. wandb-primary) if it is
#   available, to list the newest runs in jgueldenstein/mjlab_piplus and pick
#   the one created just now.
# - Fallback: poll wandb/ for a new run-YYYYMMDD_HHMMSS-* directory (verify
#   the timestamp matches the current time, as runs take time to initialize).
#   The ID is in the filename of the run-<id>.wandb file inside it; its first
#   few lines contain a path like logs/rsl_rl/<experiment>/2026-04-12_23-56-38.
# The run name is the log directory's last segment (the timestamp). Then
# prepend a line to runs.log at the repo root (newest first; create the file
# if it does not exist yet):
# <name>: <id>: <short summary>

# Play/evaluate a trained policy — launch as a background process via Bash run_in_background=true (NOT tmux).
# When playing multiple runs, always run them SEQUENTIALLY — wait for each to
# finish before starting the next, not in parallel.
CUDA_VISIBLE_DEVICES=<n> uv run play <task-id> --wandb-run-path jgueldenstein/mjlab_piplus/<id>

# Sanity-check with dummy agents (no training needed)
CUDA_VISIBLE_DEVICES=<n> uv run play <task-id> --agent zero
CUDA_VISIBLE_DEVICES=<n> uv run play <task-id> --agent random

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
- `src/mjlab_piplus/objects/` — Non-robot entities; `ball.py` is the kick ball (`get_ball_cfg`).
- `src/mjlab_piplus/tasks/` — Task registration and one `*_env_cfg.py` / `*_rl_cfg.py` pair per task.
- `src/mjlab_piplus/tasks/mdp/` — Custom MDP terms. Sub-packages `path_following/` (A* path library, path command and rewards) and `footstep/` (footstep command and rewards); kick, goal-pose, stand and the `MultiCommand` composite live in `kick.py`, `commands.py` and `multi_command.py`, the per-command reward weighting in `multi_rewards.py`.


### Task registration

Tasks are registered in `src/mjlab_piplus/tasks/__init__.py` via `register_mjlab_task()`. The package entry point `mjlab.tasks` (declared in `pyproject.toml`) causes mjlab's CLI to discover them automatically. Registered task IDs:

| task ID | env cfg | what it is |
|---|---|---|
| `Mjlab-Velocity-PiPlus` | `piplus_velocity_env_cfg.py` | 12-DoF velocity tracking |
| `Mjlab-Path-PiPlus` | `piplus_path_env_cfg.py` | 12-DoF path following |
| `Mjlab-Multi-PiPlus` | `piplus_multi_env_cfg.py` | 20-DoF, six command types incl. kick — see below |
| `Mjlab-StandupImitation-PiPlus` | `piplus_standup_imitation_env_cfg.py` | validation env for the getup teacher bridge |

### Robot definition (`piplus_constants.py`)

Three motor series are used:
- **3536** — head joints (still commented out in the MJCF and in the actuator list)
- **4438** — arm joints, enabled via `get_piplus_robot_cfg(with_arms=True)`
- **5036** — leg joints (hip pitch/roll, thigh, calf, ankle pitch/roll), with per-joint tuned PD gains

Each actuator series uses `BuiltinPositionActuatorCfg` with built-in action lag controlled by `ACTUATOR_LAG_MIN` and `ACTUATOR_LAG_MAX` constants (same lag applied to all actuators). The `get_piplus_robot_cfg()` function assembles the full `EntityCfg` with the MuJoCo spec, collision model, and actuator list.

> ⚠️ **`with_arms` changes the DoF count (12 → 20)** and therefore the action
> dim and the joint blocks of the observation. Only the **multi** task uses it.
> `mdp/symmetry.py` carries both layouts (`JOINT_ORDER_12` / `JOINT_ORDER_20`),
> so the mirror follows either — but a task's symmetry class has to name the
> right one. `get_spec(with_arms=False)` deletes the arm joints and their
> actuators from the spec, so the arms stay welded to the torso exactly as
> before.

The `*_selfpad` geoms are declared in `piplus.xml` under the
`self_collision_pad` class, the way the reference declares its own — they are
always in the spec. `get_piplus_robot_cfg(self_pads=True)` is what makes them
live, by adding `SELF_PAD_COLLISION` to put them on contype/conaffinity **bit
2**, so they can only ever touch each other: never the terrain, never the ball,
never the real (bit 1) collision geoms. They are what the multi task's
`self_collision` sensor scores. Because a pad cannot reach the ground, its size
is free — which is the point. Without `self_pads`, `FULL_COLLISION`'s
`disable_other_geoms` leaves them at contype 0 and they cost nothing, which is
how velocity and path see them.

> ⚠️ **A pad the same size as its collision geom is not a bug.** The pad set's
> first job is to *select* which link pairs count as a self-collision; only where
> near-misses matter is the copy enlarged — torso 1.2×, hands 2×, feet 1.3×, the
> rest 1:1, and the upper arms not padded at all. The MJCF says the same next to
> the class definition.

Collision configs (`FULL_COLLISION`, `LEG_ONLY_COLLISION`, `FEET_ONLY_COLLISION`) use regex matching on geom names ending in `_collision[0-9]`. `SELF_PAD_COLLISION` must be applied **after** `FULL_COLLISION`, whose `disable_other_geoms` default would otherwise zero the pads out again.

### Environment config (`piplus_velocity_env_cfg.py`)

The function `piplus_velocity_env_cfg(play=False)` builds a `ManagerBasedRlEnvCfg`. Key design choices:

- **Actor obs**: base angular velocity, projected gravity, joint pos/vel relative to default, last action, gait phase (sin/cos), velocity command. The sensor-derived terms (ang vel, gravity, joint pos/vel) are noised; actions, phase, and command are not.
- **Critic obs**: actor obs + base linear velocity, foot air time, foot contact, foot contact forces. No noise. (`foot_height` from the terrain height scan is commented out here; the multi task's critic does use it.)

> ⚠️ **Symmetry depends on the actor obs layout.** The left-right mirror in
> `symmetry.py` is a list of `Block`s, one per actor obs term (and one per
> sub-command inside a `MultiCommand`), each with its own rule: swap left/right
> joints and negate, plus a sign flip for axial vs. polar quantities — ang-vel
> `[-1,1,-1]`, gravity `[1,-1,1]`, velocity command `[1,-1,-1]`, phase
> half-period shift `[-1,-1]`. **Any change to the actor obs — term order,
> dimensions, or what a term measures — requires editing the block list for
> that task.** Widths are no longer implicit: the mirror is resolved against the
> real observation width on first use and raises with the full block breakdown
> if they disagree, so a drift fails at startup instead of silently mirroring
> the wrong columns. The **critic obs does NOT need a mirror transform**: under
> data augmentation the mirrored sample reuses the original's value/return
> (`V(mirror(s)) == V(s)`), so the value function is only evaluated on the
> original obs.
>
> Run `uv run python -m mjlab_piplus.tasks.mdp.check_symmetry` after touching
> the observations, the commands, or the robot: it checks the joint order and
> the left/right axis mirroring against the compiled MJCF, the block widths
> against each env cfg, and that every mirror is an involution.

- **Sensors**: `ContactSensorCfg` for feet ground contact (ankle roll link subtrees vs. terrain), self-collision sensor, terrain height scan anchored at foot sites.
- **Curriculum**: velocity command ranges ramp up in two stages, from step 0 to step 800 × 24 (`command_vel`).
- **Terminations**: time out (20 s episodes), orientation > 70° (`fell_over`, via `bad_orientation`). The root-height termination is currently commented out.
- **Simulation**: 5 ms timestep, decimation 4 (→ 50 Hz policy), MuJoCo Warp with 10 solver iterations, `nconmax=45`.
- In `play` mode: 1 env, curriculum and all DR disabled (only the two reset events remain — `push_robot` too is off), episode length unbounded, extra air-time / foot-tilt metrics added.

### Multi-command env (`piplus_multi_env_cfg.py`)

One 20-DoF policy (`with_arms=True`, `self_pads=True`) for six command types.
`MultiCommand` (`mdp/multi_command.py`) assigns each env one type, drawn by
`fractions` — velocity 0.20, path 0.15, goal_pose 0.15, footstep 0.15, kick
0.25, stand 0.10 — and redraws on each resample (`change_command_probability`
0.3), arming the next resample from the *selected* sub-command's own
`resampling_time_range`. The policy therefore learns the hand-over between
modes, not six separate behaviours.

| type | cfg | what it asks for |
|---|---|---|
| `velocity` | `UniformVelocityCommandWithStopCfg` | twist tracking, `lin_vel_x` up to `MAX_COMMANDED_SPEED` (0.8); stand flag off, so zero means walk-in-place |
| `path` | `PathCommandCfg` | follow an A* path from the library, re-anchored at the robot (`align=True`) |
| `footstep` | `FootstepCommandCfg` | hit a timed footstep plan on the gait clock; half from the path library, half Gaussian |
| `goal_pose` | `GoalPoseCommandCfg` | reach a pose 0.5–8 m away, no reference path |
| `kick` | `KickCommandCfg` | approach the ball, kick it in a direction at a strength; ball respawns after each kick |
| `stand` | `StandCommandCfg` | hold the default pose |

- **Actor obs** = the velocity task's proprioception (noised, 1-step delay)
  + phase + `command` + `ball`. The command block is
  `[one_hot(6) | velocity(3) | path(2N+2) | footstep(8M) | goal_pose(4) | kick(3)]`
  with inactive blocks zeroed (N = `NUM_PATH_OBSERVATIONS` = 8,
  M = `NUM_FOOTSTEP_OBSERVATIONS` = 4). The ball's position/velocity is *not* in
  the command: it is its own term (`pi_mdp.ball_state`) with a camera-like
  2–5-step delay, zeroed outside the kick type.
- **Critic obs** additionally has unbiased joint pos, base lin vel, foot height
  scan, foot air time / contact / contact forces.
- **Per-command rewards.** Task terms are `MultiRewardTermCfg`
  (`mdp/multi_rewards.py`), whose `command_weights` scale the term per active
  type (missing = 1.0, `None` = everywhere). Presets at the top of the file:
  `VELOCITY_AND_STAND`, `PATH_ONLY`, `FOOTSTEP_ONLY`, `GOAL_POSE_ONLY`,
  `KICK_ONLY`, `STAND_BOOSTED` (×3 under stand). Safety/gait terms
  (`upright`, `foot_gait`, `feet_phase`, `self_collisions`, ...) are plain
  `RewardTermCfg` and apply everywhere.
- **Curriculum** (kick only): `drop_kick_contact` zeroes the flat per-contact
  bootstrap at iteration 1500; `narrow_kick_aim` narrows the aim sigma 60° → 20°
  linearly from iteration 1500 to 6400. The aim terms pay *negative* outside the
  tolerance, so narrowing too fast makes the robot stop kicking.
- **Play mode**: 1 env, only the reset events (no DR, no push), unbounded
  episodes, small path libraries (200); switch type with the viewer dropdown.


### Falls in the multi task

There is **no getup**. A fall ends the episode — `fell_over` (`bad_orientation`
past **60°**) or `root_height_below_minimum` (0.25 m, the backstop for a squat,
which is vertical and so invisible to `fell_over`) — and `is_terminated`
charges -200 once. The reset is the walking task's `reset_base` +
`reset_robot_joints` pair. The getup rewards, deadline, metrics and gated
reward wrappers were all removed; the getup machinery lives on only in
`Mjlab-StandupImitation-PiPlus` (below).

> ⚠️ **`nconmax` is 45**, down from the 120 getup needed (a robot on its back
> makes ~43 contacts). With falls terminal it never reaches those poses: the
> measured peak is **19** in any single world across 512 envs x 400 steps of
> standing, of thrashing under random actions, and of being force-tipped to 65
> and 89 degrees. Contacts past `nconmax` are dropped *silently* and look like
> the robot sinking through the floor, so re-measure before trimming, and raise
> it for anything that puts the robot on the ground on purpose or for terrain
> that is not a plane. Measured cost at 2048 envs: 120 -> 3384 MB,
> 45 -> 2264 MB, 30 -> 2038 MB of process GPU memory.

- **The arms stay actuated** (`with_arms=True`, 20 DoF). They were enabled for
  getup, but they earn their place in walking as a balance aid and in the kick.
  The mirror follows: `PiPlusMultiSymmetry` uses `JOINT_ORDER_20` and mirrors
  the composite command block by block.
- **Self-collision is scored on the pads, with no `secondary`.** A secondary
  *pattern* silently collapses to one geom — `ContactSensorCfg.secondary_policy`
  defaults to `"first"` — so matching `.*_selfpad` on both sides scores every pad
  against the base pad alone and reads ~0 however folded the robot is. Dropping
  the secondary is correct here, not a shortcut: bit-2 isolation already means a
  pad's only possible contact partner is another pad.

### Standing in the multi task

`stand_mask` is **exactly "the stand command type is active"** — nothing else.
Sub-commands used to be able to declare themselves standing through a
`wants_stand` property (the path command at its goal, the velocity command's
stand flag, the goal-pose command's standing envs); that opt-in is gone, along
with the path command's `stand_at_goal` / `stand_delay` / `twist_command_name`
plumbing that fed it. Standing is its own command type here, so a path or
goal-pose env that has arrived is still under a *moving* command and keeps the
walking regime. `MultiCommandCfg.stand_command_name` (default `"stand"`) names
the type the mask reads.

> ⚠️ **`std_standing` is 0.1 now (all joints); the history below is why it
> must not go back to 0.05.** At 0.05 the stand pose was lost entirely and
> permanently. `variable_posture` is one exponent over all 20 joints, so the
> standing regime's tolerance is a cliff, not a slope. Measured on
> `bp9zuz30` (std 0.05), forcing every env onto the stand command, no DR:
>
> | iter | RMS Δq (rad) | \|shoulder_roll\| | posture | total/step |
> |---|---|---|---|---|
> | 0 | 0.025 | 0.03 | 2.32 | 6.30 |
> | 200 | 0.029 | 0.01 | 2.13 | 6.13 |
> | 400 | 0.082 | 0.04 | 0.20 | 4.19 |
> | 800 | 0.340 | 0.29 | **0.0000** | 4.00 |
> | 3000 | 0.408 | 0.85 | 0.0000 | 4.00 |
> | 6399 | 0.716 | 2.03 | 0.0000 | 4.00 |
>
> A ~0.08 rad wobble at iteration 400 already costs 90 % of the term; by 800 it
> is *identically zero in float32* and so is its gradient. From there the pose
> is a free direction — the total is flat at 4.00 for 5600 iterations while the
> shoulders travel another 1.7 rad — and the policy random-walks into an
> arms-out, legs-splayed stance. Nothing else prices a static offset:
> `action_rate_l2` charges changes, not magnitude, and there is no `action_l2`.
> Holding the default pose instead is worth **6.27/step against 3.99** (posture
> 2.30 at weight 3.0), so the policy is not trading the pose for anything — it
> simply cannot see it any more.
>
> Widening the std keeps the pose: at 0.1 and 0.15 (`cct7hpwm`, `ww5l7ll3`) the
> stand pose is still RMS 0.012–0.031 rad at iteration 1000, where 0.05 had
> already escaped by 800. The cliff moves, it does not disappear — at std 0.1 an
> RMS of 0.3 is still a dead 0.0004 — so treat any *transient* excursion during
> early training as permanent.

> ⚠️ **With std 0.05, standing was scored more leniently than walking.**
> Same policy, same checkpoint, per-step posture payout: **0.85 under a velocity
> command** (std_walking, weight 1.0, RMS 0.13), **0.11 under kick**, **0.0000
> under stand** (std_standing 0.05, ×3 from `STAND_BOOSTED`). The boost was
> multiplying a term the tight std had already zeroed — the arms sat within
> 0.02 rad of default while walking and 2.03 rad from it while standing.
> Re-check this payout split if you touch `std_standing`.

### Action scale

The multi task uses mjlab's plain `JointPositionActionCfg` at a uniform
`scale=0.1`, `use_default_offset=True` — `target = default + 0.1 * a`.

Two alternatives were built and measured while getup was in scope, and both are
still available if the action range ever becomes a constraint again:

- **Per-joint `scale` dict.** `BaseActionCfg.scale` accepts `dict[str, float]`
  matched by regex over joint names (`actions.py:64-73`).
- **`QuadraticJointPositionActionCfg`** (`mdp/actions.py`), a signed-quadratic
  map `target = default + sign(a) * scale * a^2`, subclassing
  `JointPositionAction` and overriding only `process_actions`.

> ⚠️ **Scale multiplies the policy's exploration noise as well as its mean.**
> `Policy/mean_std` sits near 1.3 and does not anneal away (`entropy_coef=0.01`
> is the floor), so a uniform 0.5 leaves ~0.65 rad of permanent per-joint jitter
> against ~0.13 rad at 0.1. Measured: a uniform 0.5 run collapsed `posture`
> 0.246 -> 0.001, `foot_gait` 0.186 -> 0.002 and `feet_phase` 0.198 -> 0.005
> against a 0.1 baseline, while running *further* into training.

> ⚠️ **`action_rate_l2` is charged on raw policy output** (mjlab's docstring:
> "Operates on raw policy output"), so the cost of a physical motion is
> `w*(dq/scale)^2`. Raising a joint's scale 5x makes the same motion **25x**
> cheaper to make abruptly — under a uniform 0.5 that silently removed the
> smoothness constraint from walking (`Episode_Reward/action_rate_l2` fell
> -1.74 -> -0.014). **Changing a scale also changes that joint's effective
> smoothness weight by 1/scale^2.**

If you do split the scale per joint, split it by *measured walking swing*, not
by joint range. Rolling out `ewxi1s0n` (uniform 0.1) under a velocity command
gives noise-to-signal per joint, at 0.1 -> at 0.5: `hip_pitch` 0.05 -> 0.23 and
`calf` 0.08 -> 0.38 (they carry the gait's large excursions, +-1.9 and +-1.4
rad, so 0.5 is cheap for them), against `thigh` 0.27 -> 1.35, `hip_roll`
0.35 -> 1.73, `ankle_pitch` 0.23 -> 1.17, `ankle_roll` 0.28 -> 1.37 — those do
small precise work and drown. **`thigh` included, despite its 2.77 rad range**,
which is exactly why range is the wrong criterion.

### Getup by imitation (`mdp/imitation.py`, `Mjlab-StandupImitation-PiPlus`)

Not used by the multi task any more; kept for the standalone validation task.
The fallen robot is paid `exp(-mean_j (Δq)^2 / std^2)` for matching the **joint
position target** a trained getup teacher (`getup_model.pt`, from
`mjlab_playground`; `GETUP_TEACHER_CHECKPOINT`) would command from the same
state. Comparing targets in radians bridges the teacher's
`RelativeJointPositionAction` (`joint_pos + 0.6a`) and this env's
default-relative action. `piplus_standup_imitation_env_cfg.py` makes that
reward (weight 4.0) plus the reference's three height/orientation shaping terms
the whole objective: every env spawns fallen, the actor obs is exactly the
teacher's 66-vector, and the multi task's DR and obs noise are carried over
(no push). A teacher-following oracle scores 0.997 on the term and stands the
robot from 0.09 m to ~0.31–0.35 m.

Gotchas, if you touch it:

- The teacher's obs is rebuilt from **true state** with mjlab's own obs
  functions (not sliced from this env's noised actor obs), relative to the
  **teacher's** default pose (`TEACHER_DEFAULT_JOINT_POS`, which differs from
  ours on 9 of 20 joints), with its own last-action buffer zeroed on reset.
  Joint order is resolved by name (`TEACHER_JOINT_ORDER`).
- `teacher_biased=False`: the teacher does not see the encoder bias; feeding it
  the biased position drops getup success from ~0.87 to 0.65.
- The teacher target is cached from the **pre-step** state the student acted
  on; recomputing at reward time scored a perfect oracle 0.618.
- `std` 0.3; tighter hits the exponential dead zone.
- The playground teacher env shares this DR set; after editing either side run

  ```sh
  cd ~/mjlab_piplus && PYTHONPATH=~/mjlab_playground/src uv run python \
    ~/mjlab_playground/src/mjlab_playground/getup/config/pi_plus/verify_dr_matches_piplus.py
  ```

### Viewer GUI (`mdp/push_gui.py`)

mjlab's viser viewer only hands *command terms* a GUI hook
(`CommandTerm.create_gui`), so anything the project wants in the panel has to
ride along on one. `PushGui` (a "Push" folder: six velocity-delta sliders,
±3 m/s linear / ±6 rad/s angular, a heading-frame toggle, and a *Push robot*
button) is built by whichever term owns the panel — `MultiCommand` in the multi
task, `UniformVelocityCommandWithStop` when it runs standalone (velocity and
path). A sub-term under a composite command must skip it (`if
self.joystick_gate is None:`), or the folder appears twice.

The button click arrives on viser's thread, so it only sets a flag; the write
happens in `PushGui.apply`, called from the owning term's `compute`. The push is
the same mass-independent root-velocity delta as the training-time
`push_by_setting_velocity` event. It shows up in `asset.data` only after the
next forward, which is what makes it look like a no-op if you assert on the same
step.

### RL config (`piplus_*_rl_cfg.py`)

PPO with asymmetric actor-critic (actor/critic each 512-256-128 MLP, ELU activation, `entropy_coef=0.01`). Logs to W&B project `mjlab_piplus`. 24 steps per env per rollout. Per task:

| task | experiment | max iterations | save interval |
|---|---|---|---|
| velocity | `piplus_velocity` | 3 200 | 200 |
| path | `piplus_path` | 3 200 | 200 |
| multi | `piplus_multi` | 6 400 | 200 |
| standup imitation | `piplus_standup_imitation` | 1 000 | 100 |

Each task's RL cfg carries its own symmetry class from `mdp/symmetry.py`, and
the two enforcement paths are configured independently — `use_data_augmentation`
appends the mirrored samples to every mini-batch, `use_mirror_loss` adds a soft
penalty at `mirror_loss_coeff`:

| task | class | augmentation | mirror loss |
|---|---|---|---|
| velocity | `PiPlusSymmetry` | off | off (coeff 0.5) |
| path | `PiPlusPathSymmetry` | on | on, 0.1 |
| multi | `PiPlusMultiSymmetry` | on | on, 0.1 |

## Editing conventions

- All custom MDP terms (rewards, metrics, curriculums, etc.) go in
  `src/mjlab_piplus/tasks/mdp/` and are wired into the env cfg via
  `pi_mdp.<name>`. Do **not** edit `mjlab/` sources unless the user
  explicitly says to.

## Code style

- Line length: 88 columns.
- Avoid local imports unless necessary for circular import resolution.
- Joint name patterns use regex strings (e.g., `".*_hip_pitch_joint"`) — these are matched by MuJoCo name filters throughout mjlab.
