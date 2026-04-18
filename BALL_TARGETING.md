# Ball Targeting with Velocity Policy

Train stable walking policy, then add ad-hoc ball targeting at inference. No RL for ball-seeking behavior.

## Approach

1. **Train velocity-tracking policy** (omnidirectional walking)
   - Policy learns to track commanded (vx, vy, omega_z) velocities
   - Rewards: velocity tracking, stability, gait quality
   - No ball in training env

2. **Inject ball-targeting commands at play time**
   - `BallTargetingWrapper` computes velocity commands based on ball position
   - Override twist command before policy inference
   - Policy treats overridden commands as normal velocity targets

## Advantages

- **Decoupled learning**: Walking skill separated from ball-seeking logic
- **Zero-shot transfer**: No retraining when switching to ball targeting
- **Interpretable control**: Proportional gains easy to tune vs RL reward weights
- **Robust exploration**: Random velocity commands during training cover broader state space than ball-only rewards

## Usage

### Phase 1: Train Velocity Policy

```bash
# Train walking policy (track random velocity commands)
uv run train Mjlab-Velocity-PiPlus

# Monitor wandb: mjlab_piplus/piplus_walk_to_ball
# Train until stable gait + good velocity tracking (~1200 iters sufficient)
```

**Evaluation criteria**:
- Fall rate <10% across random commands
- Velocity tracking error <0.2 m/s (linear), <0.3 rad/s (angular)
- Smooth gait (no excessive jitter or self-collision)

### Phase 2: Play with Ball Targeting

```bash
# Play velocity policy with ball-targeting wrapper
uv run python -m mjlab_piplus.scripts.play_with_ball \
  --checkpoint logs/Mjlab-Velocity-PiPlus/YYYY-MM-DD_HH-MM-SS/model_XXXX.pt

# With custom gains
uv run python -m mjlab_piplus.scripts.play_with_ball \
  --checkpoint path/to/model.pt \
  --k-forward 0.6 \
  --k-lateral 0.4 \
  --k-turn 1.2 \
  --max-forward-vel 1.0 \
  --max-lateral-vel 0.5
```

### Tuning Controller Gains

Ball-targeting controller computes velocity commands:

```
vx = k_forward * (distance - min_distance_threshold)  [clamped to max_forward_vel]
vy = k_lateral * rel_y                                [clamped to max_lateral_vel]
omega_z = k_turn * atan2(rel_y, rel_x)                [clamped to max_angular_vel]
```

**Recommended starting gains**:
- `k_forward = 0.5` — Forward velocity gain (m/s per m distance)
- `k_lateral = 0.3` — Lateral velocity gain (m/s per m offset)
- `k_turn = 1.0` — Angular velocity gain (rad/s per rad heading error)

**Velocity limits** (must stay within training distribution):
- `max_forward_vel = 1.0` m/s (training range: -1.0 to 1.4)
- `max_lateral_vel = 0.4` m/s (training range: -0.5 to 0.5)
- `max_angular_vel = 1.5` rad/s (training range: -π/2 to π/2)

**Tuning guide**:
- **Robot overshoot/oscillate** → Reduce `k_forward`, `k_lateral`, `k_turn`
- **Robot too slow/doesn't turn** → Increase gains
- **Robot falls often** → Commanded velocities exceed training distribution → reduce max limits
- **Robot stops too far from ball** → Reduce `min_distance_threshold` (default 0.3m)

## Implementation Details

### Files

- **`ball_targeting_wrapper.py`** — Wrapper that overrides velocity commands based on ball position
- **`scripts/play_with_ball.py`** — Play script with ball-targeting controller
- **`tasks/piplus_velocity_env_cfg.py`** — Velocity task config (added `with_ball=True` param for play mode)

### How Wrapper Works

1. **Env setup**: Velocity env with ball entity spawned at reset (1-2m ahead)
2. **Each step**:
   - Wrapper computes ball position relative to robot (body frame)
   - Calculates velocity command (vx, vy, omega_z) via proportional controller
   - Overrides `env.command_manager._terms["twist"].command` tensor
3. **Policy inference**: Policy sees overridden command in observation, produces actions
4. **Env step**: Robot executes actions normally

### Ball Position Calculation

```python
# World frame ball position
ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2]
robot_pos = env.scene["robot"].data.root_link_pos_w[:, :2]
rel_pos_w = ball_pos - robot_pos

# Transform to robot body frame
heading = env.scene["robot"].data.heading_w
rel_x = cos(heading) * rel_pos_w[0] + sin(heading) * rel_pos_w[1]  # Forward
rel_y = -sin(heading) * rel_pos_w[0] + cos(heading) * rel_pos_w[1]  # Lateral
distance = sqrt(rel_x^2 + rel_y^2)
heading_error = atan2(rel_y, rel_x)
```

## Comparison to RL Ball-Seeking

### Ad-Hoc Controller (This Approach)
- ✅ Simple, interpretable, tunable
- ✅ No retraining needed
- ✅ Decoupled walking + ball-seeking logic
- ❌ May be suboptimal (not learned end-to-end)
- ❌ Fixed control law (no adaptation to ball dynamics)

### RL Ball-Seeking (WalkToBall Task)
- ✅ End-to-end optimization (potentially better performance)
- ✅ Can learn ball interception, not just approach
- ❌ Longer training (must learn walking + ball-seeking together)
- ❌ Reward shaping critical (progress, facing, success bonuses)
- ❌ Less interpretable

**When to use which**:
- **Ad-hoc**: Fast prototyping, simple ball approach, interpretable control
- **RL**: Need optimal ball interception, dynamic ball, complex strategies

## Training Tips

### Velocity Task Training
- Start curriculum at zero velocity (standing stability first)
- Gradually expand velocity ranges over 1200 iters
- Push events (random impulses) improve robustness
- Air time reward encourages dynamic gait (enable after 200 iters)

### Checkpoints
- Save every 50 iters (`save_interval=50` in rl_cfg)
- Best checkpoint: stable gait + low velocity tracking error
- Test multiple checkpoints if robot falls often with wrapper

### Troubleshooting
- **Robot falls during ball targeting but not during velocity play**: Commands outside training distribution → check wrapper velocity limits match training ranges
- **Robot ignores ball**: Wrapper not injecting commands → verify `with_ball=True` in env_cfg, check `twist_cmd.command` override
- **Robot walks but wrong direction**: Heading transform error → verify body frame calculation (cos/sin signs)

## Next Steps

After validating ad-hoc ball targeting:
1. Compare performance to RL-trained WalkToBall policy
2. Tune gains for specific ball distances/approach speeds
3. Extend to dynamic ball (moving target)
4. Add hierarchical controller (switch between "approach" and "position" modes)

## Quick Start Checklist

- [ ] Train velocity policy: `uv run train Mjlab-Velocity-PiPlus`
- [ ] Monitor training until stable gait (~1200 iters)
- [ ] Play with ball targeting: `uv run python -m mjlab_piplus.scripts.play_with_ball --checkpoint path/to/model.pt`
- [ ] Tune gains if robot overshoot/undershoot
- [ ] (Optional) Compare to RL WalkToBall: `uv run train Mjlab-WalkToBall-PiPlus`
