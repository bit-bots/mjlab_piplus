
from pathlib import Path
import os
import mujoco

from mjlab.actuator import DelayedActuatorCfg, BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
##
# MJCF and assets.
##

PIPLUS_XML: Path = (
  Path(os.path.dirname(__file__)) / "piplus/piplus.xml"
)

def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PIPLUS_XML.parent / "assets", meshdir)
  return assets

def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PIPLUS_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec

EFFORT_LIMIT_3536 = 2.5
EFFORT_LIMIT_4438 = 10.0
EFFORT_LIMIT_5036 = 20.0

ARMATURE_4438 = 0.008419
ARMATURE_5036 = 0.044277

STIFFNESS_3536 = 6
STIFFNESS_4438 = 6
STIFFNESS_5036_HIP_PITCH = 110
STIFFNESS_5036_HIP_ROLL = 110
STIFFNESS_5036_THIGH = 110
STIFFNESS_5036_CALF = 110
STIFFNESS_5036_ANKLE_PITCH = 210
STIFFNESS_5036_ANKLE_ROLL = 110

DAMPING_3536 = 0.6
DAMPING_4438 = 0.6
DAMPING_5036_HIP_PITCH = 1.1
DAMPING_5036_HIP_ROLL = 1.4
DAMPING_5036_THIGH = 1.1
DAMPING_5036_CALF = 1.1
DAMPING_5036_ANKLE_PITCH = 1.1
DAMPING_5036_ANKLE_ROLL = 1.1

FRICTIONLOSS = 0.4

# Motors used in Head
ACTUATOR_3536 = BuiltinPositionActuatorCfg(
  target_names_expr=(
      "head_yaw_joint",
      "head_pitch_joint",
  ),
  stiffness=STIFFNESS_3536,
  damping=DAMPING_3536,
  effort_limit=EFFORT_LIMIT_3536,
  frictionloss=FRICTIONLOSS,
)
DELAYED_ACTUATOR_3536 = DelayedActuatorCfg(
  base_cfg=ACTUATOR_3536,
  delay_target="position",
  delay_min_lag=2,
  delay_max_lag=3,
)

# Motors used in Arms
ACTUATOR_4438 = BuiltinPositionActuatorCfg(
  target_names_expr=(
      ".*_shoulder_pitch_joint",
      ".*_shoulder_roll_joint",
      ".*_upper_arm_joint",
      ".*_elbow_joint",
  ),
  stiffness=STIFFNESS_4438,
  damping=DAMPING_4438,
  armature=ARMATURE_4438,
  effort_limit=EFFORT_LIMIT_4438,
  frictionloss=FRICTIONLOSS,
)
DELAYED_ACTUATOR_4438 = DelayedActuatorCfg(
  base_cfg=ACTUATOR_4438,
  delay_target="position",
  delay_min_lag=2,
  delay_max_lag=3,
)

# Motors used in Legs
ACTUATOR_5036_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=STIFFNESS_5036_HIP_PITCH,
  damping=DAMPING_5036_HIP_PITCH,
  armature=ARMATURE_5036,
  effort_limit=EFFORT_LIMIT_5036,
  frictionloss=FRICTIONLOSS,
)
ACTUATOR_5036_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=STIFFNESS_5036_HIP_ROLL,
  damping=DAMPING_5036_HIP_ROLL,
  armature=ARMATURE_5036,
  effort_limit=EFFORT_LIMIT_5036,
  frictionloss=FRICTIONLOSS,
)
ACTUATOR_5036_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_thigh_joint",),
  stiffness=STIFFNESS_5036_THIGH,
  damping=DAMPING_5036_THIGH,
  armature=ARMATURE_5036,
  effort_limit=EFFORT_LIMIT_5036,
  frictionloss=FRICTIONLOSS,
)
ACTUATOR_5036_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=STIFFNESS_5036_CALF,
  damping=DAMPING_5036_CALF,
  armature=ARMATURE_5036,
  effort_limit=EFFORT_LIMIT_5036,
  frictionloss=FRICTIONLOSS,
)
ACTUATOR_5036_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=STIFFNESS_5036_ANKLE_PITCH,
  damping=DAMPING_5036_ANKLE_PITCH,
  armature=ARMATURE_5036,
  effort_limit=EFFORT_LIMIT_5036,
  frictionloss=FRICTIONLOSS,
)
ACTUATOR_5036_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=STIFFNESS_5036_ANKLE_ROLL,
  damping=DAMPING_5036_ANKLE_ROLL,
  armature=ARMATURE_5036,
  effort_limit=EFFORT_LIMIT_5036,
  frictionloss=FRICTIONLOSS,
)

DELAYED_ACTUATOR_5036_HIP_PITCH = DelayedActuatorCfg(
  base_cfg=ACTUATOR_5036_HIP_PITCH,
  delay_target="position",
  delay_min_lag=2,
  delay_max_lag=3,
)
DELAYED_ACTUATOR_5036_HIP_ROLL = DelayedActuatorCfg(
  base_cfg=ACTUATOR_5036_HIP_ROLL,
  delay_target="position",
  delay_min_lag=2,
  delay_max_lag=3,
)
DELAYED_ACTUATOR_5036_THIGH = DelayedActuatorCfg(
  base_cfg=ACTUATOR_5036_THIGH,
  delay_target="position",
  delay_min_lag=2,
  delay_max_lag=3,
)
DELAYED_ACTUATOR_5036_CALF = DelayedActuatorCfg(
  base_cfg=ACTUATOR_5036_CALF,
  delay_target="position",
  delay_min_lag=2,
  delay_max_lag=3,
)
DELAYED_ACTUATOR_5036_ANKLE_PITCH = DelayedActuatorCfg(
  base_cfg=ACTUATOR_5036_ANKLE_PITCH,
  delay_target="position",
  delay_min_lag=2,
  delay_max_lag=3,
)
DELAYED_ACTUATOR_5036_ANKLE_ROLL = DelayedActuatorCfg(
  base_cfg=ACTUATOR_5036_ANKLE_ROLL,
  delay_target="position",
  delay_min_lag=2,
  delay_max_lag=3,
)


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.39),
  joint_pos={
    ".*_hip_pitch_joint": -0.25,
    ".*_hip_roll_joint": 0.0,
    ".*_thigh_joint": 0.0,
    ".*_calf_joint": 0.65,
    ".*_ankle_pitch_joint": -0.4,
    ".*_ankle_roll_joint": 0.0,
    ".*_shoulder_roll_joint": 0.0,
    ".*_shoulder_pitch_joint": 0.0,
    ".*_upper_arm_joint": 0.0,
    ".*_elbow_joint": 0.0,
  },
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

PIPLUS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    #ACTUATOR_3536,
    ACTUATOR_4438,
    ACTUATOR_5036_HIP_PITCH,
    ACTUATOR_5036_HIP_ROLL,
    ACTUATOR_5036_THIGH,
    ACTUATOR_5036_CALF,
    ACTUATOR_5036_ANKLE_PITCH,
    ACTUATOR_5036_ANKLE_ROLL,
  ),
  soft_joint_pos_limit_factor=0.9,
)
def get_piplus_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=PIPLUS_ARTICULATION,
  )


BALL_XML: Path = (
  Path(os.path.dirname(__file__)) / "ball/ball.xml"
)

def get_ball_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(BALL_XML))

BALL_INIT = EntityCfg.InitialStateCfg(
  pos=(1.5, 0.0, 0.11),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

def get_ball_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=BALL_INIT,
    spec_fn=get_ball_spec,
  )