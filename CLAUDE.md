# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a [mjlab](https://github.com/mujocolab/mjlab) project for training RL policies for the PiPlus humanoid robot. It uses MuJoCo Warp (GPU-accelerated MuJoCo) with a manager-based API similar to Isaac Lab, and RSL-RL for PPO training.

## Commands

Always use `uv run`, not `python` directly.

```sh
# Train — launch as a background process via Bash run_in_background=true (NOT tmux), auto-select a free GPU via nvidia-smi
# After launch, poll wandb/ dir for new run-YYYYMMDD_HHMMSS-* directories
# (verify timestamp matches current time, as runs take time to initialize).
# Inside the dir, find the run-<id>.wandb file — the ID is in the filename.
# Read the first few lines of that file to find a path like
# logs/rsl_rl/piplus_velocity/2026-04-12_23-56-38 — the last segment is the
# run name. Then prepend a line to runs.log (newest first):
# <name>: <id>: <short summary>
uv run train Mjlab-Velocity-PiPlus --env.scene.num-envs 4096 --gpu-ids '[GPU]'

# Play/evaluate a trained policy — launch as a background process via Bash run_in_background=true (NOT tmux), pick a free GPU via nvidia-smi
# Export CUDA_VISIBLE_DEVICES to a single GPU before launching.
# Prefer unused GPUs; if all busy, pick one with low utilization and free memory.
# When playing multiple runs, always run them SEQUENTIALLY — wait for each to
# finish before starting the next, not in parallel.
CUDA_VISIBLE_DEVICES=GPU uv run play Mjlab-Velocity-PiPlus --wandb-run-path jgueldenstein/mjlab_piplus/<id>

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


### Task registration

Tasks are registered in `src/mjlab_piplus/tasks/__init__.py` via `register_mjlab_task()`. The package entry point `mjlab.tasks` (declared in `pyproject.toml`) causes mjlab's CLI to discover the task automatically. The registered task ID is `"Mjlab-Velocity-PiPlus"`.

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
- **Critic obs**: actor obs + base linear velocity, foot height (from `TerrainHeightSensorCfg`), foot air time, foot contact, foot contact forces. No noise.

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
- **Terminations**: time out (20 s episodes), orientation > 45° (`bad_orientation`). The root-height termination is currently commented out.
- **Simulation**: 5 ms timestep, decimation 4 (→ 50 Hz policy), MuJoCo Warp with 10 solver iterations.
- In `play` mode: curriculum disabled, episode length unbounded, extra air-time metrics added. Push events stay active.

### Falls in the multi task

There is **no getup**. A fall ends the episode (`fell_over`, `bad_orientation`
past 70 deg, the same limit the velocity task uses) and `is_terminated` charges
-200 once. Everything that supported fall recovery was removed: the
`getup_imitation` and `on_ground` rewards, the `fallen_too_long` deadline, the
`fallen`/`getup_success` metrics, and the `GatedRewardTermCfg` wrappers on all
ten walking and safety terms — those gates existed only to fade walking rewards
out while the robot was down, and with falls terminal there is nothing to fade.
The reset is back to the walking task's `reset_base` (`reset_root_state_uniform`)
+ `reset_robot_joints` (`reset_joints_by_offset`) pair, replacing
`reset_fallen_or_standing`.

The getup machinery itself still exists and is still used — see
`mdp/getup.py`, `mdp/imitation.py` and `Mjlab-StandupImitation-PiPlus` below.
It is only the multi task that no longer uses it.

> ⚠️ **`nconmax` is 30**, down from the 120 getup needed. A robot lying on its
> back touches the floor with the whole trunk, both arms and both legs at once
> (~43 contacts). With falls terminal it never reaches those poses, and the
> count is bounded by how many collision geoms can touch a plane at once: the
> measured peak is **19** in any single world across 512 envs x 400 steps of
> standing, of thrashing under random actions, and of being force-tipped to 65
> and 89 degrees. Contacts past `nconmax` are dropped *silently* and look like
> the robot sinking through the floor, so re-measure before trimming further,
> and raise it for anything that puts the robot on the ground on purpose or for
> terrain that is not a plane. Measured cost at 2048 envs: 120 -> 3384 MB,
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

> ⚠️ **`std_standing` at 0.05 loses the stand pose entirely, and the loss is
> permanent.** `variable_posture` is one exponent over all 20 joints, so the
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

> ⚠️ **Standing is scored more leniently than walking, which is backwards.**
> Same policy, same checkpoint, per-step posture payout: **0.85 under a velocity
> command** (std_walking, weight 1.0, RMS 0.13), **0.11 under kick**, **0.0000
> under stand** (std_standing, weight 3.0). The 3x stand boost is multiplying a
> term the tight std has already zeroed. Under walking the term is alive and it
> shows — the arms sit within 0.02 rad of default while walking and 2.03 rad
> from it while standing.

### The push (`mdp/push.py`)

`push_by_velocity_delta` replaces mjlab's `push_by_setting_velocity` in the
multi task (the velocity and path tasks still use mjlab's). Same
instantaneous, mass-independent root-velocity delta; different sample. mjlab's
draws all six twist components from their own boxes, so every push is a mixture
— a diagonal shove with roll, pitch and yaw stirred in — and its magnitude
concentrates near the box RMS instead of covering the range. This one picks
**one kind per push**: translational (a horizontal direction, uniform on the
circle, times a speed) or rotational (that same direction as a *spin axis*,
times an angular speed about it), at `rotational_probability` 0.5.

> ⚠️ **There is no yaw push and no vertical shove any more.** The spin axis is
> horizontal by construction — that is what tips the robot; a spin about z only
> turns it on the spot. If yaw disturbance is wanted back it needs its own kind,
> not a third component.

Magnitudes are sampled from 0 to a ceiling that `ramp_push`
(`event_param_linear`) grows linearly, so raising it adds harder pushes without
removing the gentle ones. `PUSH_LINEAR_START` 0.25 m/s and `PUSH_ANGULAR_START`
0.52 rad/s are what the old box term topped out at per horizontal axis and in
roll/pitch; both ramp to 1.0 between iterations 1000 and 5000. The ramp is
linear rather than staged (`event_curriculum`, which the path task uses) because
a step change in push strength is a cliff the gait has to re-learn through.

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

### Getup by imitation (`mdp/imitation.py`)

Instead of discovering a stand-up from shaped rewards, the fallen robot is paid
for reproducing what an already-trained getup policy would do from the same
state. The teacher is a checkpoint from the reference env in `mjlab_playground`
(`getup_model.pt` at the repo root); `GETUP_TEACHER_CHECKPOINT` names it.

Two things are bridged, and they are the whole module:

- **Observation.** The teacher's actor is a 66 → 20 MLP taking
  `[base_ang_vel(3) | projected_gravity(3) | joint_pos_rel(20) |
  joint_vel_rel(20) | last_action(20)]`. The first four are the *same mjlab
  functions* this env already uses, so they are rebuilt by calling them
  directly — **not** by slicing this env's actor obs, which has extra terms
  (phase, command) in a different layout and is noised. The fifth is the
  **teacher's own** previous output, kept in a buffer here and zeroed on reset;
  carrying it across an episode boundary would feed it the tail of a different
  getup. The reference runner sets `obs_normalization=False` on its actor, so
  there is no normalizer to apply — the loader raises if a checkpoint has one.
- **Action.** The teacher is a `RelativeJointPositionAction`
  (`target = joint_pos + a*0.6`); this env is default-relative
  (`target = default + f(a)`). Comparing raw action vectors would be
  meaningless. Both are converted to a **joint position target in radians** and
  the reward is `exp(-mean_j (Δq)^2 / std^2)` over that. This also makes the
  term indifferent to this env's action scale or to swapping in a nonlinear map.
  `_processed_actions` is read *before* the encoder-bias subtraction, which is
  where the teacher's expression sits too, so the bias cancels.

Joint order is resolved **by name** (`TEACHER_JOINT_ORDER`), not assumed — it
happens to be the identity for all 20 actuated joints, but a reorder on either
side would otherwise scramble the mapping silently.

> ⚠️ **The teacher's `joint_pos` observation is relative to the TEACHER's
> default pose, not this env's.** The reference robot's home keyframe
> ("walkready") differs from ours on 9 of 20 joints — elbows by 0.7 rad, calves
> by 0.2, hips and ankles by 0.1, RMS 0.235 — so subtracting our default (what
> mjlab's `joint_pos_rel` does) offsets the teacher's input by that much.
> Measured: it changed the teacher's action by mean 0.30 (max 1.55), i.e. a
> 0.179 rad error in the joint target it asks for, and cost ~0.036 m of the
> height its getup reaches. `TEACHER_DEFAULT_JOINT_POS` holds the correct pose.
> Note the consequence: following the teacher drives the robot toward *its*
> standing pose, not ours — which in the multi task pulls against `posture`.

> ⚠️ **Rewards run after the physics step, but the action being judged was
> chosen from the pre-step state.** The teacher's target is therefore cached one
> call earlier, from the state the student actually acted on, rather than
> recomputed at reward time. During a getup the joints move ~0.06 rad per 20 ms
> step, and comparing across that gap scored a *perfect* teacher-following
> oracle **0.618** instead of 0.997. An env's first step after a reset has no
> cached target from its own episode and scores 0.

> ⚠️ `std` is 0.3, and should not go tighter. An untrained policy sits ~0.39 rad
> from the teacher; at `std=0.2` that is a reward of 0.018 with gradient 0.44,
> versus 0.17 and 1.60 at 0.3 — the same exponential-saturation dead zone that
> kills `posture` when its std is too tight.

The term is gated to the fallen half with `fallen_gate`: the teacher's env has
no velocity command and it was trained to stand up and *stop*, so ungated it
would fight the walking controller. `on_ground` is now the mirror image — a flat
`always_one` at weight -1.0 through the same `fallen_gate`, so the "is it
fallen" test is the gate rather than something the term re-derives.

### `Mjlab-StandupImitation-PiPlus` — the imitation validation task

`piplus_standup_imitation_env_cfg.py`. A stripped env whose only job is to
answer "does the teacher bridge work": no commands, no gait shaping, no
penalties, no ball, no curriculum, no terminations but the time-out. The
**imitation reward is the entire objective**, so the return is a direct readout
of how closely the student tracks the teacher, uncontaminated by anything else.

Deliberate differences from the multi task:

- **The imitation term is ungated.** Gating it would blind the reward exactly
  when the robot succeeds and stands up. (The multi task no longer uses this
  reward at all — see "Falls in the multi task".)
- **The three shaping terms come from the reference** at its own weights:
  `torso_height` 2.0, `head_height` 1.0, `orientation` 1.0, all ungated (gating
  them on "is fallen" would step the return *down* the instant the robot stands
  up). They sum to 4.0/step; `IMITATION_WEIGHT` matches that total at 4.0.
- **No `on_ground` and no terminations but the time-out.** Being on the floor is
  priced by the height terms falling toward zero there, which is a gradient
  rather than a flat rate.
- **Every env spawns fallen** (`fall_probability=1.0`).
- **The actor obs is exactly the teacher's 66-vector**, so the student can in
  principle represent the teacher and a poor score is evidence about the bridge
  rather than a missing input.

It carries the multi task's **domain randomization** (all 11 startup terms) and
**observation noise + 1-step delay**, verbatim, so a policy that works here is
answering the same sensing and dynamics question it will face there. No
`push_robot`: a shove is something to walk off, and this env's only job is
standing up from the floor.

> ⚠️ **The teacher does not see any of the noise.** `imitation.py` builds its
> observation from true state, so the target stays a clean function of where the
> robot actually is while the student reproduces it from noisy, delayed sensing.
> A teacher fed the same noise would emit a noisy target — variance in the
> reward with no added information.

That extends to the encoder bias, and it matters. `teacher_biased` defaults
**False**. Teacher getup success (fraction of envs above 0.29 m after 300 steps,
256 envs, teacher-following oracle):

| configuration | final base h | success |
|---|---|---|
| no DR, no obs noise | 0.329 | 1.00 |
| no DR, obs noise on | 0.329 | 1.00 |
| full DR minus `encoder_bias` | 0.327 | 0.98 |
| full DR, `encoder_bias` at the reference's ±0.0698 | 0.310 | 0.85 |
| full DR, teacher sees biased position | 0.292 | 0.65 |
| **full DR, teacher sees true position (default)** | **0.312** | **0.84–0.89** |

The teacher's own env in `mjlab_playground`
(`getup/config/pi_plus/env_cfgs.py`) now carries **this same DR set and sensor
noise**, so the next teacher is trained under the conditions it will advise
under. `getup/config/pi_plus/verify_dr_matches_piplus.py` diffs the two configs
term by term and exits non-zero on any mismatch — run it after touching either
side:

```sh
cd ~/mjlab_piplus && PYTHONPATH=~/mjlab_playground/src uv run python \
  ~/mjlab_playground/src/mjlab_playground/getup/config/pi_plus/verify_dr_matches_piplus.py
```

The numbers below predate that change and were measured with `getup_model.pt`
(iteration 200), trained under the playground's *old* randomization.

Two readings. **Observation noise costs the teacher nothing** — it reads true
state, so only the student is affected. **`encoder_bias` is the entire DR
effect**: it is the one randomization that reaches the teacher's input, this env
runs it at ±0.1 against the ±0.0698 the reference trained on, and a persistent
per-joint offset is not something a policy can average away. Feeding the teacher
the true position recovers most of the loss. The residual (0.84 vs 1.00) is the
physical DR — masses, gains, friction — which the teacher genuinely never saw,
and `getup_model.pt` is an early (iteration 200) checkpoint.

Measured, 128 envs x 300 steps, no DR (values are per-step, i.e. `reward_buf / dt`):

| policy | total/step | imitation value | base height start → end |
|---|---|---|---|
| zero action | 1.233 | 0.0003 | 0.398 → 0.092 |
| random action | 1.216 | 0.0003 | 0.398 → 0.089 |
| **teacher-following oracle** | **7.627** | **0.9967** | 0.397 → **0.348** |

The oracle scores 0.997 of a possible 1.0 on the imitation term, ~2500x above
chance, and the teacher driven through this env's default-relative action term
genuinely stands the robot up (0.09 collapsed → 0.31). The bridge is exact: the
student's commanded target matches the teacher's to 2e-8 rad.

Per-term payout, robot held in each pose (max 4.036/step of shaping):

| state | torso_height | head_height | orientation | shaping total |
|---|---|---|---|---|
| flat on the floor | 0.519 | 0.150 | 0.500 | 1.171 |
| standing upright | 2.000 | 1.000 | 1.000 | 4.036 |

So the shaping alone offers 2.87/step for standing up, against up to 2.0/step
for imitating well. A policy that stands *without* imitating out-scores one that
imitates on the floor — intended, but worth knowing before reading a run.

### Evaluation (`tasks/eval/`, `Mjlab-MultiEval-PiPlus`)

A deterministic benchmark for the multi-command policy: **one command type per
episode, no switching**, with every command replayed from a pre-generated file
so two policies meet the same scenarios rather than the same distribution.

```sh
# Once: 1000 commands per type, from the training env cfg's own ranges.
uv run python -m mjlab_piplus.tasks.eval.generate_commands   # -> eval_commands.json

# All six types in turn, 100 episodes each.
CUDA_VISIBLE_DEVICES=2 uv run python -m mjlab_piplus.tasks.eval.run_eval \
  --wandb-run-path jgueldenstein/mjlab_piplus/<id>
# A subset, and the do-nothing floor to read the numbers against:
  ... --types velocity kick --runs-per-command 200 --num-envs 128
  ... --agent zero
```

Results land in `logs/eval/<run>_<timestamp>/`: `episodes_<type>.csv` (one row
per episode, every measured quantity) and `summary.json` (mean/median/p90/n per
field, plus the run config). `uv run play Mjlab-MultiEval-PiPlus
--wandb-run-path ...` shows the same scenarios in the viewer.

| file | what it holds |
|---|---|
| `command_book.py` | the schema of a command per type, the generator, JSON load/save |
| `generate_commands.py` | the CLI that writes `eval_commands.json` |
| `eval_commands.py` | `Eval*` subclasses of every command term: replay + measurement |
| `eval_env_cfg.py` | `piplus_multi_eval_env_cfg()` — the multi env, made repeatable |
| `run_eval.py` | the headless runner, CSV/JSON output, summary table |

**What is measured.** Velocity: mean absolute tracking error in `vx`, `vy` and
`wz` after a 1 s warmup (the acceleration out of a standstill is a step
response, not tracking), plus the signed achieved mean beside it — the absolute
error is the accuracy, the signed mean is what separates a consistent
under-shoot from jitter around the right value. Goal pose: time to the goal, split into
position-only and full pose, plus final/min distance and yaw error. Path: mean
and max distance to the reference path, and time to goal. Footstep: xy, z and
yaw error of each landing against the target that was **latched** for it at
touchdown, averaged over the episode. Kick: the first completed kick's peak
speed against the commanded strength, the aim error of the ball's actual travel
direction, style and whether the intended foot struck. Stand: base drift, yaw
drift, posture RMS. Every type also reports `fell`, `steps` and `duration`.

**Plotting.** `plot_eval.py` turns those CSVs into figures — **one file per
plot**, named `<command type>_<plot>.pdf`:

```sh
uv run python -m mjlab_piplus.tasks.eval.plot_eval --runs logs/eval/<run>
uv run python -m mjlab_piplus.tasks.eval.plot_eval \
  --runs logs/eval/<a> logs/eval/<b> --labels baseline candidate
  ... --bare              # no title at all: the document's caption names it
  ... --legend inside     # legend in the plot's top-left corner; or 'off'
  ... --no-tex            # skip LaTeX; matplotlib draws the serif face itself
  ... --font-scale 1.0    # every point size; default 1.5, bands scale with it
  ... --combined          # also write the multi-panel sheet per type
  ... --format png|both   # default pdf (vector); dpi only affects the raster
  ... --theme dark        # its own steps for the dark surface, not an inversion
```

So `goal_pose` yields exactly `goal_pose_time_vs_distance.pdf`,
`goal_pose_final_distance.pdf` and `goal_pose_outcomes.pdf`. Files land in
`<first run>/plots` unless `--out` says otherwise. The contact sheet is off by
default — the individual files are the deliverable, the sheet is for reviewing
them at a glance. The form follows each type's question: for velocity, achieved
against commanded on the `y = x` line (a slope is a gain, an offset a bias) and
absolute error against the size of the ask; commanded-vs-achieved on `y = x`
for kick speed; time-to-goal against how far the goal *was*
for path and goal pose (a slow episode over 8 m is not the slow episode over
1 m), and ECDFs for every error distribution — several runs overlay without
occluding each other, and the median and the tail are both readable, which a
histogram gives up. Rates are horizontal bars, direct-labelled.

**Typography.** Text is set in Times through LaTeX (`text.usetex`), with
`newtxtext,newtxmath` so the symbols that must go through math mode (`≤`, the
`|…|` in the axis labels) come out in Times rather than Computer Modern beside
it. The PDF then carries embedded Type 1 outlines and no Type 3, which is what
a publisher's font check asks for; `--no-tex` falls back to matplotlib's own
serif and Type 42 TrueType (`pdf.fonttype: 42`). Point sizes stay the script's
own — `font.size` is deliberately left at its default, because `tight_layout`
measures its padding in font sizes and the title and legend bands were tuned
against that.

> ⚠️ **Every string is escaped for LaTeX** (`tex()`). Run labels are directory
> names (`det_a`) and the bar labels carry percent signs — an unescaped `_` or
> `%` ends a usetex render with "Missing $ inserted" instead of a figure, so
> any new label must go through it too.

> ⚠️ **At most three runs per plot.** The scatter panels put every pair of
> series on screen at once, and only the first three slots of the palette clear
> the colour-separation floors that way; a fourth would seat a pair nobody with
> a colour-vision deficiency can tell apart. The script refuses rather than
> cycling hues — plot in groups instead. Colours are the `dataviz` reference
> palette's categorical slots 1–3 plus its chart chrome.

> ⚠️ **A NaN is a result, not a zero.** "No footstep landed", "the goal was
> never reached", "the ball was never touched" are reported as NaN and dropped
> from the aggregate, with `n` saying how many episodes actually made the
> measurement. Averaging them in as zeros would score a policy that never
> attempts the task as perfect at it.

**How determinism is achieved.** Env `e` runs command indices
`e, e+N, e+2N, ...` for `N` envs, assigned at reset from an episode counter —
so which episode gets which command does not depend on the policy, on when a
robot falls, or on the order envs finish in. Two policies evaluated at the same
`--num-envs` see identical scenarios in identical order; at a different
`--num-envs` they see the same *set* (rows `0..runs-1`), distributed
differently. Each `Eval*` term overrides only its sampling: every drawn
quantity is replaced by its book row, and the sub-command configs are copied
field by field off the training cfg (`_as_eval_cfg`), so a range changed in
training is automatically part of the evaluation and the observation layout
cannot drift. The spawn yaw is pinned to 0 and the joint reset jitter to zero;
the booked goals, paths and kick directions are all **relative to the robot's
heading**, so nothing depends on where the robot happens to face.

> ⚠️ **DR, pushes and observation noise are OFF by default** (`--domain-rand`,
> `--push`, `--obs-noise` turn them on). The default numbers are a clean
> measurement of the controller, **not** a robustness measurement — say which
> flags were on when quoting them. The fixed one-step sensor delay stays on in
> both cases: it is part of the plant, not of the noise.

> ⚠️ **The book is a checked-in artifact, not a temp file.** Regenerating it
> (`--force`) silently invalidates every result measured against the old one.
> `path_index` names a path in the A* library, so the book records the library
> size it was drawn against (`EVAL_PATHLIB_SIZE`, 2000 — the multi task's own
> path library, also pinned onto the footstep command, whose training default
> of 50 000 is a 1.9 GB file) and the eval terms refuse a mismatch: the i-th
> path of one library is not the i-th path of another.

Per-type episode lengths (`EVAL_EPISODE_LENGTH_S`) are enforced by
`eval_time_out` reading the *active* type's limit — a path up to 8 m long and a
velocity command need very different clocks, and `mdp.time_out`'s single
env-wide maximum would put every type on the longest one. The sub-commands'
resampling intervals are pushed past the episode length, so a command is drawn
at the reset and held. One env instance serves every type: `set_forced_type`
re-pins it between passes, which is far cheaper than rebuilding the sim.

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

### RL config (`piplus_velocity_rl_cfg.py`)

PPO with asymmetric actor-critic (actor/critic each 512-256-128 MLP, ELU activation). Logs to W&B project `mjlab_piplus`. 24 steps per env per rollout, up to 6 400 iterations, checkpoint every 50 iterations.

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
