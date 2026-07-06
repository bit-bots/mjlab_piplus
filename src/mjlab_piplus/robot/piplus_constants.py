import os
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

PIPLUS_XML: Path = Path(os.path.dirname(__file__)) / "piplus/piplus.xml"
BALL_XML: Path = Path(os.path.dirname(__file__)) / "ball/ball.xml"


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PIPLUS_XML))
  return spec


def get_ball_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(BALL_XML))


ACTUATOR_LAG_MIN = 0
ACTUATOR_LAG_MAX = 3

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


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.39),
  joint_pos={
    "r_hip_pitch_joint": 0.45,
    "l_hip_pitch_joint": -0.45,
    ".*_hip_roll_joint": 0.0,
    ".*_thigh_joint": 0.0,
    "r_calf_joint": 0.9,
    "l_calf_joint": -0.9,
    "r_ankle_pitch_joint": 0.45,
    "l_ankle_pitch_joint": -0.45,
    ".*_ankle_roll_joint": 0.0,
    # "r_shoulder_roll_joint": 1.2,
    # "l_shoulder_roll_joint": -1.2,
    # "r_shoulder_pitch_joint": 1.5708,
    # "l_shoulder_pitch_joint": -1.5708,
    # ".*_upper_arm_joint": 0.0,
    # ".*_elbow_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

BALL_INIT = EntityCfg.InitialStateCfg(
  pos=(1.5, 0.0, 0.10),
  joint_pos={".*": 0.0},
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
    # ACTUATOR_3536,
    # ACTUATOR_4438,
    ACTUATOR_5036,
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


def get_ball_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=BALL_INIT,
    spec_fn=get_ball_spec,
  )
