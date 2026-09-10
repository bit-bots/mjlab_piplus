"""Sanity check for the PiPlus left-right mirrors.

Run with ``uv run python -m mjlab_piplus.tasks.mdp.check_symmetry``. Three
things are checked, none of which needs a simulator:

1. **Joint layout.** ``JOINT_ORDER_12`` / ``JOINT_ORDER_20`` must equal the
   joint order of the compiled MJCF, and every left/right pair must have
   mirrored axes -- that is what makes "swap sides and negate" the correct
   mirror for *every* joint, pitch included.
2. **Observation width.** Each task's mirror must cover exactly the actor
   observation its env cfg produces, computed from the cfg's own constants.
3. **Involution.** ``PiPlusSymmetry.__call__`` prepends the original batch to
   the mirrored batch (num_aug = 2), so the batch doubles on every call:

       call 1 (N -> 2N):  [orig, M(orig)]
       call 2 (2N -> 4N): [orig, M(orig), M(orig), M(M(orig))]

   With quarters q0..q3 of size N: q0 == orig, q1 == q2 == M(orig), and
   q3 == M(M(orig)) == q0 -- the involution check. A mirror must also actually
   change the data, so a silent no-op cannot pass.
"""

from __future__ import annotations

from typing import Any, cast

import mujoco
import torch
from tensordict import TensorDict

from mjlab_piplus.robot.piplus_constants import get_spec
from mjlab_piplus.tasks.mdp.footstep.footstep_command import OBS_PER_FOOTSTEP
from mjlab_piplus.tasks.mdp.symmetry import (
  JOINT_ORDER_12,
  JOINT_ORDER_20,
  WITH_ARMS,
  PiPlusGoalPoseSymmetry,
  PiPlusMultiSymmetry,
  PiPlusPathSymmetry,
  PiPlusSymmetry,
)

N = 8

# The mirrors never touch the env argument, and building one would need a
# simulator -- which is the whole point of this check being cheap to run.
_NO_ENV: Any = None

torch.manual_seed(0)

_failures: list[str] = []


def _check(ok: bool, label: str) -> None:
  print(f"  {'PASS' if ok else 'FAIL'}  {label}")
  if not ok:
    _failures.append(label)


##
# 1. Joint layout, against the compiled spec.
##


def check_joint_layout() -> None:
  print("\n===== joint layout (compiled MJCF) =====")
  for with_arms, expected in ((False, JOINT_ORDER_12), (True, JOINT_ORDER_20)):
    model = get_spec(with_arms).compile()
    # Model ids are kept alongside the names: the free joint is filtered out
    # here, so a name's position in ``names`` is not its ``jnt_axis`` row.
    hinges = [
      (i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
      for i in range(model.njnt)
      if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    names = tuple(name for _, name in hinges)
    _check(names == expected, f"with_arms={with_arms}: order matches the model")
    if names != expected:
      print(f"        model: {names}")
      print(f"        symmetry.py: {expected}")
      continue
    jnt_id = {name: i for i, name in hinges}
    # A left/right pair mirrors with a sign flip iff the reflected right axis
    # (pseudovector: w -> (-x, y, -z)) is the *negative* of the left axis.
    for name in names:
      if not name.startswith("r_"):
        continue
      ar = model.jnt_axis[jnt_id[name]]
      al = model.jnt_axis[jnt_id["l_" + name[2:]]]
      reflected = (-ar[0], ar[1], -ar[2])
      flips = all(abs(reflected[k] + al[k]) < 1e-9 for k in range(3))
      _check(flips, f"    {name}: mirrors as a sign flip (axes {ar} / {al})")


##
# 2. Observation width, against the env cfgs.
##


def check_obs_widths() -> tuple[tuple[str, PiPlusSymmetry, int, int], ...]:
  from mjlab_piplus.tasks.mdp.path_following import PathCommandCfg
  from mjlab_piplus.tasks.piplus_multi_env_cfg import (
    NUM_FOOTSTEP_OBSERVATIONS,
    NUM_PATH_OBSERVATIONS,
  )
  from mjlab_piplus.tasks.piplus_path_env_cfg import piplus_path_env_cfg

  print("\n===== observation width =====")

  # Shared prefix: ang_vel(3) + gravity(3) + jpos + jvel + actions + phase(2).
  def prefix(num_joints: int) -> int:
    return 3 + 3 + 3 * num_joints + 2

  velocity_dim = prefix(12) + 4  # [vx, vy, wz, stand_flag]
  goal_pose_dim = prefix(12) + 4  # [dx, dy, cos(yaw_err), sin(yaw_err)]

  path_cfg = piplus_path_env_cfg()
  path_command = cast(PathCommandCfg, path_cfg.commands["path"])
  path_n = path_command.num_observations
  path_dim = prefix(12) + 2 * path_n + 2

  # The multi task follows ``WITH_ARMS``: 20 joints with the arms actuated, 12
  # without. Hard-coding 20 here would make this check pass against a layout the
  # env does not have.
  multi_joints = 20 if WITH_ARMS else 12
  multi_dim = (
    prefix(multi_joints)
    + 6  # one_hot over the six command types
    + 3  # velocity, no stand flag
    + 2 * NUM_PATH_OBSERVATIONS
    + 2
    + OBS_PER_FOOTSTEP * NUM_FOOTSTEP_OBSERVATIONS
    + 4  # goal_pose
    + 3  # kick: [cos(dir_err), sin(dir_err), strength]
    + 4  # ball: [x, y, vx, vy], its own observation term
  )

  cases = (
    ("velocity", PiPlusSymmetry(), velocity_dim, 12),
    ("goal_pose", PiPlusGoalPoseSymmetry(), goal_pose_dim, 12),
    ("path", PiPlusPathSymmetry(), path_dim, 12),
    (
      "multi",
      PiPlusMultiSymmetry(
        num_path_observations=NUM_PATH_OBSERVATIONS,
        num_footstep_observations=NUM_FOOTSTEP_OBSERVATIONS,
      ),
      multi_dim,
      multi_joints,
    ),
  )
  for label, sym, obs_dim, _act_dim in cases:
    try:
      sym.obs_mirror.build(obs_dim)
      ok = True
      detail = ""
    except ValueError as exc:
      ok = False
      detail = f" -- {exc}"
    _check(ok, f"{label}: mirror covers the {obs_dim}-dim actor obs{detail}")
    print(f"        layout: {sym.obs_mirror.layout()}")
  return cases


##
# 3. Involution.
##


def report(name: str, x: torch.Tensor) -> None:
  q = x.shape[0] // 4
  q0, q1, q2, q3 = x[:q], x[q : 2 * q], x[2 * q : 3 * q], x[3 * q :]

  invol = (q3 - q0).abs().max().item()
  mirror_consistent = (q1 - q2).abs().max().item()
  changed = (q0 - q1).abs().max().item()

  _check(invol < 1e-6, f"{name}: mirror(mirror(x)) == x (err {invol:.2e})")
  _check(
    mirror_consistent < 1e-6,
    f"{name}: both mirrored copies agree (err {mirror_consistent:.2e})",
  )
  _check(changed > 0.0, f"{name}: mirror is not a no-op (max delta {changed:.2e})")


def check_involution(
  cases: tuple[tuple[str, PiPlusSymmetry, int, int], ...],
) -> None:
  print("\n===== involution =====")
  for label, sym, obs_dim, act_dim in cases:
    print(f"\n  [{label}] OBS_DIM={obs_dim} ACT_DIM={act_dim}")
    obs = TensorDict({"actor": torch.randn(N, obs_dim)}, batch_size=[N])
    actions = torch.randn(N, act_dim)

    obs1, act1 = sym(_NO_ENV, obs, actions)
    obs1, act1 = cast(TensorDict, obs1), cast(torch.Tensor, act1)
    assert obs1.batch_size[0] == 2 * N and act1.shape[0] == 2 * N

    obs2, act2 = sym(_NO_ENV, obs1, act1)
    obs2, act2 = cast(TensorDict, obs2), cast(torch.Tensor, act2)
    assert obs2.batch_size[0] == 4 * N and act2.shape[0] == 4 * N

    report("obs.actor", cast(torch.Tensor, obs2["actor"]))
    report("actions", act2)


def main() -> int:
  check_joint_layout()
  cases = check_obs_widths()
  check_involution(cases)
  print()
  if _failures:
    print(f"FAILED ({len(_failures)}):")
    for failure in _failures:
      print(f"  - {failure}")
    return 1
  print("All symmetry checks passed.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
