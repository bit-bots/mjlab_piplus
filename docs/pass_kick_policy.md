# Pass/Kick Policy

## Goal

The pass/kick policy should execute a targeted kick with the PiPlus robot. The
policy should be usable for both direct goal kicks and passes to teammates.

The intended abstraction is:

```text
Given the current robot and ball state, kick the ball toward a target point.
```

The low-level policy should not decide whether a target is tactically useful.
A higher-level behavior module should choose the target point. For example:

- For a goal kick, target a point inside or behind the opponent goal.
- For a pass, target a receive point in front of the teammate.
- For a moving teammate, target a predicted receive point rather than the
  teammate's current position.

This keeps the RL policy focused on kick execution instead of mixing execution
with tactical pass selection.

## Coordinate Frames

Follow the repository convention:

- `_w` means world frame.
- `_b` means robot base/body frame.

Examples:

```text
ball_pos_w              ball position in world coordinates
ball_pos_b              ball position relative to the robot base
target_pos_w            desired target position in world coordinates
target_from_ball_b      target vector from ball to target, expressed in base frame
```

Policy inputs should use base-frame quantities where possible. This makes the
same kick geometry look the same to the model independent of field position and
robot yaw.

## Command Semantics

The command should describe where the ball should go, not how hard to kick.

Recommended command meaning:

```text
Move the ball from its current position to the desired target position.
```

For passes, the target can be a position where the ball should come to rest near
the teammate. A desired ball speed is not required for this basic behavior. The
policy can learn the required kick strength from target distance, ball state,
robot state, and terrain/contact dynamics.

For goal kicks, the same command can be used by placing the target at or behind
the goal. This avoids needing a separate goal-kick mode in the low-level policy.

A desired ball speed or arrival time should only be added if the same target
position needs multiple different behaviors, such as a soft layoff versus a fast
through pass.

## Actor Inputs

Start with the existing proprioceptive observation prefix used by the current
PiPlus tasks:

```text
base_ang_vel             3
projected_gravity        3
joint_pos                12
joint_vel                12
actions                  12
phase                    2
```

This gives a shared 44-dimensional robot-state prefix.

Add a compact kick/pass task observation:

```text
ball_pos_b               2   # x/y ball position relative to robot base
ball_vel_b               2   # x/y ball velocity relative to robot base
target_from_ball_b       2   # x/y vector from ball to desired target
target_radius            1   # acceptable target tolerance
```

Recommended initial actor input layout:

```text
[
  base_ang_vel,
  projected_gravity,
  joint_pos,
  joint_vel,
  actions,
  phase,
  ball_pos_b,
  ball_vel_b,
  target_from_ball_b,
  target_radius,
]
```

Total size:

```text
44 + 7 = 51 dimensions
```

`target_radius` allows the same policy to represent different precision
requirements. A pass receive point can allow a larger radius than a precise shot
target.

## Optional Actor Inputs

Only add these if the task actually needs them:

```text
target_vel_b             2   # useful if the policy should learn lead passes
desired_arrival_time     1   # useful for timed passes
desired_ball_speed       1   # useful if pass pace matters independently of target
is_goal_kick             1   # only if goal kicks and passes need different semantics
```

The first version should avoid these. The higher-level module should handle
receive-point selection, teammate prediction, and goal target selection.

## Critic Inputs

The critic can use privileged simulation-only information to improve training.
These should not be required by the deployed actor.

Possible critic-only inputs:

```text
true_ball_pos_w
true_ball_vel_w
target_pos_w
ball_to_target_distance
ball_contact_state
last_ball_contact_impulse
foot_contact_states
foot_contact_forces
base_lin_vel
```

## Policy Outputs

Use the same action style as the existing PiPlus locomotion tasks unless there is
a strong reason to change it:

```text
joint_pos_action         12
```

The action should command joint position targets for the actuated joints. The
current robot configuration only actuates the leg joints, so the first kick/pass
policy should be designed around leg-only kicking.

If arm swing, head stabilization, or a more expressive kick motion becomes
important later, the robot actuator configuration and observation/action
symmetry need to be updated together.

## Higher-Level Responsibilities

The higher-level behavior module should choose the target point.

For a pass, it should avoid targeting the teammate directly if that would place
the ball behind them. Instead, it should choose a receive point in front of the
teammate:

```text
target_pos = teammate_pos + teammate_forward_dir * receive_offset
```

For moving teammates, prefer a predicted receive point:

```text
target_pos = predicted_teammate_pos_at_arrival + teammate_forward_dir * receive_offset
```

The low-level policy should only receive the resulting `target_from_ball_b`.

## Reward Considerations

For passes, reward the ball ending near the target and slowing down near that
target:

```text
reward = target_proximity_reward - ball_speed_near_target_penalty
```

For goal kicks, either use the same stop-near-target semantics with a point
behind the goal, or use a goal-crossing reward:

```text
reward = ball_crosses_goal_line_reward
```

The stop-near-target version keeps pass and goal-kick semantics unified. The
goal-crossing version may be more directly aligned with scoring.

## Symmetry Considerations

If symmetry augmentation is used, the added task inputs must be mirrored
consistently.

Under left/right reflection:

```text
ball_x_b                 unchanged
ball_y_b                 sign flips
ball_vx_b                unchanged
ball_vy_b                sign flips
target_dx_b              unchanged
target_dy_b              sign flips
target_radius            unchanged
```

So the kick/pass command tail should mirror like:

```text
[1, -1, 1, -1, 1, -1, 1]
```
