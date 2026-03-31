from copy import deepcopy
import math
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.tasks.velocity import mdp
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab_piplus.robot.piplus_constants import get_piplus_robot_cfg
from mjlab_piplus.tasks.mdp.diffusion_style_reward import DiffusionStyleReward

from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

def piplus_velocity_env_cfg(play: bool = False, static: bool = False) -> ManagerBasedRlEnvCfg:
  foot_sites = ("l_foot", "r_foot")

  ##
  # Observations
  ##
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/local_linvel"},
    noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/gyro"},
    noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
    func=mdp.projected_gravity,
    noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
    func=mdp.joint_pos_rel,
    noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
    func=mdp.joint_vel_rel,
    noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "twist"},
    ),
  }
  critic_terms = {
    **actor_terms,
    "foot_height": ObservationTermCfg(
    func=mdp.foot_height,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=foot_sites)},
    ),
    "foot_air_time": ObservationTermCfg(
    func=mdp.foot_air_time,
    params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
    func=mdp.foot_contact,
    params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
    func=mdp.foot_contact_forces,
    params={"sensor_name": "feet_ground_contact"},
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=True,
    enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=0.5,
    use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rel_standing_envs=0.1,
    rel_heading_envs=0.3,
    heading_command=False,
    heading_control_stiffness=0.0,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-0.5, 0.5),
      heading=None,
    ),
    )
  }
  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.01, 0.05),
        "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.4, 0.4),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
        },
      },
    ),
  }

  ##
  # Rewards
  ##
  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "upright": RewardTermCfg(
      func=mdp.flat_orientation,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.0,  # Override per-robot
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.0,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=0.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {".*": 0.05},  # Set per-robot.
        "std_walking": {
          r".*_hip_pitch_joint": 0.3,
          r".*_hip_roll_joint": 0.15,
          r".*_thigh_joint": 0.15,
          r".*_calf_joint": 0.35,
          r".*_ankle_pitch_joint": 0.25,
          r".*_ankle_roll_joint": 0.1,
          r".*_shoulder_roll_joint": 0.15,
          r".*_shoulder_pitch_joint": 0.15,
          r".*_upper_arm_joint": 0.1,
          r".*_elbow_joint": 0.15,
        },
        "std_running": {
          r".*_hip_pitch_joint": 0.3,
          r".*_hip_roll_joint": 0.15,
          r".*_thigh_joint": 0.15,
          r".*_calf_joint": 0.35,
          r".*_ankle_pitch_joint": 0.25,
          r".*_ankle_roll_joint": 0.1,
          r".*_shoulder_roll_joint": 0.15,
          r".*_shoulder_pitch_joint": 0.15,
          r".*_upper_arm_joint": 0.1,
          r".*_elbow_joint": 0.15,
        },
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.4),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-0.0,
      params={
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=foot_sites),
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=-0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "target_height": 0.13,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=foot_sites),
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=foot_sites),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "diffusion_style": RewardTermCfg(
        func=DiffusionStyleReward,
        weight=1500.0,
        params=dict(
            asset_cfg=SceneEntityCfg("robot", site_names=["l_foot", "r_foot", "l_hand", "r_hand"], body_names=("base_link",)),
            checkpoint_path="/homes/17vahl/mjlab_piplus/epoch_0500.pt",
            robot_scale=0.25,               # your robot's size relative to training data
            stance_shift=0.1,               # lateral EE offset in model-space metres
            noise_levels=(0.1, 0.7, 0.95),
            level_weights=(0.3, 0.3, 0.4),
            # tune these to balance style components:
            scale_root_vel_trans=0.0,
            scale_root_vel_rot=0.5,
            scale_foot_trans=1.0,
            scale_foot_rot=0.5,
            scale_hand_trans=1.0,
            scale_hand_rot=0.5,
            reward_scale=300.0,
        ),
    )
  }
  ##
  # Terminations
  ##
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  ##
  # Curriculum
  ##
  curriculum = {
      "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
           "command_name": "twist",
           "velocity_stages": [
           {"step": 0, "lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
           {"step": 200 * 24, "lin_vel_x": (-0.25, 0.25), "lin_vel_y": (-0.1, 0.1), "ang_vel_z": (-0.5, 0.5)},
           {"step": 8000 * 24, "lin_vel_x": (-0.75, 0.75), "lin_vel_y": (-0.4, 0.4), "ang_vel_z": (-1.0, 1.0)},
           {"step": 12000 * 24, "lin_vel_x": (-1.5, 1.5), "lin_vel_y": (-0.75, 0.75), "ang_vel_z": (-1.5, 1.5)},
           ],
       },
   ),
    "upright_weight": CurriculumTermCfg(
    func=mdp.reward_curriculum,
    params={
        "reward_name": "upright",
        "stages": [
            {"step": 0, "weight": 1.0},
            {"step": 500 * 24, "weight": 0.5},
            {"step": 1400 * 24, "weight": 0.05},
        ],
    },
  ),

  }

  ##
  # Sensors
  ##

  feet_contact_sensor_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(r_ankle_roll_link|l_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  scene = SceneCfg(
    entities={"robot": get_piplus_robot_cfg()},
    sensors=(feet_contact_sensor_cfg, self_collision_cfg),
    terrain=TerrainEntityCfg(
      terrain_type="plane",
      terrain_generator=None,
      max_init_terrain_level=0,
    ),
  )

  ##
  # Viewer
  ##
  viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="base_link",
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
  )

  ##
  # Simulation
  ##
  sim = SimulationCfg(
    nconmax=45,
    njmax=1500,
    contact_sensor_maxmatch=500,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
      ccd_iterations=500,
    ),
  )
  if play:
    curriculum = {}
    del events["push_robot"]

  ##
  # Assemble the config
  ##
  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=viewer,
    sim=sim,
    decimation=4,
    episode_length_s=20.0,
  )
