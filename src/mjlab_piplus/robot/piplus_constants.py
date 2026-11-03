import os
import re
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

PIPLUS_XML: Path = Path(os.path.dirname(__file__)) / "piplus/piplus.xml"

VISUAL_RGBA = (0.2, 0.2, 0.2, 1.0)
"""Dark gray (80% gray) for the robot's visual meshes. The MJCF paints them
light gray (0.75); they are recolored here so the vendored file stays as
exported. Collision geoms are named ``*_collision[0-9]`` and are left alone."""


ARM_JOINT_EXPR = r"^[lr]_(shoulder_pitch|shoulder_roll|upper_arm|elbow)_joint$"
"""Arm joints, present in the MJCF but deleted from the spec unless a task asks
for them (see :func:`get_spec`)."""


def get_spec(with_arms: bool = False) -> mujoco.MjSpec:
  """Build the PiPlus spec.

  Args:
    with_arms: Keep the arm joints and their position actuators. When False they
      are deleted from the spec, leaving the arms welded to the torso -- the
      12-DoF layout the velocity and path tasks (and ``JOINT_ORDER_12`` in
      ``mdp/symmetry.py``) are built around.
  """
  spec = mujoco.MjSpec.from_file(str(PIPLUS_XML))
  for geom in spec.geoms:
    if geom.type == mujoco.mjtGeom.mjGEOM_MESH and not geom.name:
      geom.rgba = VISUAL_RGBA

  if not with_arms:
    # Actuators first: an actuator whose target joint is gone fails to compile.
    for actuator in list(spec.actuators):
      if re.match(ARM_JOINT_EXPR, actuator.target):
        spec.delete(actuator)
    for joint in list(spec.joints):
      if re.match(ARM_JOINT_EXPR, joint.name):
        spec.delete(joint)

  return spec


ACTUATOR_LAG_MIN = 1
ACTUATOR_LAG_MAX = 1

# ACTUATOR_3536 = XmlActuatorCfg(
#   target_names_expr=(
#     "head_yaw_joint",
#     "head_pitch_joint"
#   ),
#   delay_min_lag=ACTUATOR_LAG_MIN,
#   delay_max_lag=ACTUATOR_LAG_MAX,
# )

# Motors used in Arms
ACTUATOR_4438 = XmlActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_upper_arm_joint",
    ".*_elbow_joint",
  ),
  delay_min_lag=ACTUATOR_LAG_MIN,
  delay_max_lag=ACTUATOR_LAG_MAX,
)

# Motors used in Legs
ACTUATOR_5036 = XmlActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_thigh_joint",
    ".*_calf_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
  ),
  delay_min_lag=ACTUATOR_LAG_MIN,
  delay_max_lag=ACTUATOR_LAG_MAX,
)


LEG_HOME_JOINT_POS: dict[str, float] = {
  "r_hip_pitch_joint": 0.5,
  "l_hip_pitch_joint": -0.5,
  ".*_hip_roll_joint": 0.0,
  ".*_thigh_joint": 0.0,
  "r_calf_joint": 1.0,
  "l_calf_joint": -1.0,
  "r_ankle_pitch_joint": 0.5,
  "l_ankle_pitch_joint": -0.5,
  ".*_ankle_roll_joint": 0.0,
}

ARM_HOME_JOINT_POS: dict[str, float] = {
  "r_shoulder_pitch_joint": -1.5708,
  "l_shoulder_pitch_joint": 1.5708,
  "r_shoulder_roll_joint": -1.2,
  "l_shoulder_roll_joint": 1.2,
  ".*_upper_arm_joint": 0.0,
  "r_elbow_joint": -0.7,
  "l_elbow_joint": 0.7,

}


def get_home_keyframe(with_arms: bool = False) -> EntityCfg.InitialStateCfg:
  joint_pos = dict(LEG_HOME_JOINT_POS)
  if with_arms:
    joint_pos.update(ARM_HOME_JOINT_POS)
  return EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.39),
    joint_pos=joint_pos,
    joint_vel={".*": 0.0},
  )


FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision[0-9]$",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

SELF_PAD_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_selfpad$",),
  contype=2,
  conaffinity=2,
  condim=3,
  priority=0,
  disable_other_geoms=False,
)
"""Activate the MJCF's ``*_selfpad`` geoms by putting them on bit 2.

The pads are declared in ``piplus.xml`` under the ``self_collision_pad`` class;
this is what makes them live. On bit 2 a pad's only possible contact partner is
another pad -- never the terrain, never the ball, never the bit-1 collision
geoms -- so a pad can be any size without touching ground dynamics.

Order matters: apply *after* ``FULL_COLLISION``, whose ``disable_other_geoms``
default would otherwise zero the pads out again. That is also what keeps them
inert for tasks that do not ask for them.
"""

LEG_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"(r|l)_(thigh|calf|ankle_pitch|ankle_roll)_link_collision[0-9]$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"(r|l)_ankle_roll_link_collision[0-9]$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)


def get_piplus_articulation(with_arms: bool = False) -> EntityArticulationInfoCfg:
  actuators = (ACTUATOR_5036, ACTUATOR_4438) if with_arms else (ACTUATOR_5036,)
  return EntityArticulationInfoCfg(
    actuators=actuators,
    soft_joint_pos_limit_factor=0.95,
  )


def get_piplus_robot_cfg(
  with_arms: bool = False,
  self_pads: bool = False,
) -> EntityCfg:
  """Assemble the PiPlus entity config.

  Args:
    with_arms: Actuate the eight arm joints (20 DoF instead of 12). Off by
      default: the velocity and path tasks are built around the 12-joint
      layout (``JOINT_ORDER_12`` in ``mdp/symmetry.py``).
    self_pads: Activate the MJCF's ``*_selfpad`` geoms, for tasks that score
      self-collision. The geoms are always in the spec; without this they stay
      inert, because ``FULL_COLLISION``'s ``disable_other_geoms`` leaves them at
      contype/conaffinity 0. See :data:`SELF_PAD_COLLISION`.
  """
  collisions = (FULL_COLLISION, SELF_PAD_COLLISION) if self_pads else (FULL_COLLISION,)
  return EntityCfg(
    init_state=get_home_keyframe(with_arms),
    collisions=collisions,
    spec_fn=lambda: get_spec(with_arms),
    articulation=get_piplus_articulation(with_arms),
  )
