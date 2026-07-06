import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, TerrainHeightSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_piplus.robot.piplus_constants import get_piplus_robot_cfg
from mjlab_piplus.tasks import mdp as pi_mdp

DELAY_MIN = 0
DELAY_MAX = 3
WALK_FREQUENCY = 1.5


def piplus_goal_pose_env_cfg(
  play: bool = False, static: bool = False
) -> ManagerBasedRlEnvCfg:
  foot_sites = ("l_foot", "r_foot")
  # foot_geoms = ("l_foot00", "r_foot00")
  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(
      ObjRef(type="site", name="l_foot", entity="robot"),
      ObjRef(type="site", name="r_foot", entity="robot"),
    ),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=False,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),  # Magenta rays.
      hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
  )
  ##
  # Observations
  ##
  actor_terms = {  # careful: when changing, also change symmetry config
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/gyro"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      delay_min_lag=DELAY_MIN,
      delay_max_lag=DELAY_MAX,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=NoiseModelWithAdditiveBiasCfg(
        noise_cfg=Unoise(
          n_min=-0.05,
          n_max=0.05,
        ),
        bias_noise_cfg=Unoise(n_min=-0.03, n_max=0.03, operation="add"),
        sample_bias_per_component=True,
      ),
      delay_min_lag=DELAY_MIN,
      delay_max_lag=DELAY_MAX,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      delay_min_lag=DELAY_MIN,
      delay_max_lag=DELAY_MAX,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      delay_min_lag=DELAY_MIN,
      delay_max_lag=DELAY_MAX,
    ),
    "actions": ObservationTermCfg(
      func=mdp.last_action,
      # noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "phase": ObservationTermCfg(
      func=pi_mdp.phase,
      params={
        "frequency": WALK_FREQUENCY,
        "command_name": "goal_pose",
        "cmd_threshold": 0.01,
      },
    ),
    "command": ObservationTermCfg(
      func=pi_mdp.goal_pose_command,
      params={"command_name": "goal_pose"},
    ),
  }
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/local_linvel"},
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"sensor_name": "foot_height_scan"},
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
    "goal_pose": pi_mdp.GoalPoseCommandCfg(
      entity_name="robot",
      resampling_time_range=(10.0, 15.0),
      debug_vis=True,
      radius_range=(0.1, 5.0),
      yaw_range=(-math.pi, math.pi),
      rel_standing_envs=0.1,
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="startup",
      params={
        "pose_range": {
          "x": (-0.0, 0.0),
          "y": (-0.0, 0.0),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="startup",
      params={
        "position_range": (-0.2, 0.2),
        "velocity_range": (-0.5, 0.5),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 6.0),
      params={
        "velocity_range": {
          "x": (-0.25, 0.25),
          "y": (-0.25, 0.25),
          "z": (-0.15, 0.15),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          geom_names=("l_ankle_roll_link_collision.*", "r_ankle_roll_link_collision.*"),
        ),
        "operation": "scale",
        "ranges": (0.4, 1.0),
        "shared_random": True,
      },
    ),
    "joint_frictionloss": EventTermCfg(
      mode="startup",
      func=dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "joint_armature": EventTermCfg(
      mode="startup",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (1.0, 1.05),
        "operation": "scale",
      },
    ),
    "joint_damping": EventTermCfg(
      mode="startup",
      func=dr.joint_damping,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (0.95, 1.05),
        "operation": "scale",
      },
    ),
    "actuator_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=(r".*",)),
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "link_masses": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "ranges": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "torso_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "ranges": (-0.25, 0.25),
        "operation": "add",
      },
    ),
    "link_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "operation": "add",
        "ranges": {
          0: (-0.01, 0.01),
          1: (-0.01, 0.01),
          2: (-0.01, 0.01),
        },
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "bias_range": (-0.1, 0.1),
      },
    ),
  }

  ##
  # Rewards
  ##
  rewards = {
    # goal-pose task rewards
    "close_to_goal": RewardTermCfg(
      func=pi_mdp.close_to_goal,
      weight=1.0,
      params={"command_name": "goal_pose", "epsilon": 5.0},
    ),
    "goal_progress": RewardTermCfg(
      func=pi_mdp.goal_progress,
      weight=1.0,
      params={
        "command_name": "goal_pose",
        "max_vel": 0.5,
        "distance_threshold": 0.5,
      },
    ),
    "align_with_goal": RewardTermCfg(
      func=pi_mdp.align_with_goal,
      weight=1.0,
      params={"command_name": "goal_pose", "pos_coeff": 5.0, "ori_coeff": 1.0},
    ),
    "orient_towards_goal": RewardTermCfg(
      func=pi_mdp.orient_towards_goal,
      weight=0.5,
      params={
        "command_name": "goal_pose",
        "coeff": 1.0,
        "distance_threshold": 0.3,
      },
    ),
    # stability rewards
    "upright": RewardTermCfg(
      func=pi_mdp.body_orientation_l2,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    # safety rewards
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    # "torque_penalty": RewardTermCfg(
    #   func=mdp.joint_torques_l2,
    #   weight=-2.5e-4,
    #   params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
    # ),
    # "joint_vel_l2": RewardTermCfg(
    #   func=mdp.joint_vel_l2,
    #   weight=-2.5e-3,
    #   params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    # ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={
        "sensor_name": "self_collision",
        "force_threshold": 2.0,
      },  # we dont want even soft collisions
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "goal_pose",
        "std_standing": {r".*": 0.05},
        "std_walking": {
          r".*_hip_pitch_joint": 0.5,
          r".*_hip_roll_joint": 0.15,
          r".*_thigh_joint": 0.15,
          r".*_calf_joint": 0.5,
          r".*_ankle_pitch_joint": 0.15,
          r".*_ankle_roll_joint": 0.1,
        },
        "std_running": {
          r".*_hip_pitch_joint": 0.5,
          r".*_hip_roll_joint": 0.25,
          r".*_thigh_joint": 0.25,
          r".*_calf_joint": 0.5,
          r".*_ankle_pitch_joint": 0.25,
          r".*_ankle_roll_joint": 0.1,
        },
        "walking_threshold": 0.01,
        "running_threshold": 0.25,
      },
    ),
    # shaping rewards
    # encourage feet to be at target height when the foot velocity is high (i.e. during swing)
    "foot_gait": RewardTermCfg(
      func=pi_mdp.feet_gait,
      weight=0.5,
      params={
        "frequency": WALK_FREQUENCY,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_threshold": 0.01,
        "command_name": "goal_pose",
        "sensor_name": "feet_ground_contact",
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=pi_mdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.02,
        "command_name": "goal_pose",
        "command_threshold": 0.01,
        "asset_cfg": SceneEntityCfg("robot", site_names=foot_sites),
      },
    ),
    "foot_slip": RewardTermCfg(
      func=pi_mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "goal_pose",
        "command_threshold": 0.01,
        "asset_cfg": SceneEntityCfg("robot", site_names=foot_sites),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=pi_mdp.soft_landing,
      weight=-1e-2,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "goal_pose",
        "command_threshold": 0.01,
      },
    ),
    "stand_still": RewardTermCfg(
      func=pi_mdp.stand_still,
      weight=-1.0,
      params={
        "command_name": "goal_pose",
        "command_threshold": 0.01,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
  }
  ##
  # Metrics
  ##
  metrics = {
    "base_ang_vel_xy_l2": MetricsTermCfg(
      func=pi_mdp.base_ang_vel_xy_l2,
    ),
    "step_frequency": MetricsTermCfg(
      func=pi_mdp.step_frequency,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_force": MetricsTermCfg(
      func=pi_mdp.foot_contact_force,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "phase_metric": MetricsTermCfg(
      func=pi_mdp.phase_metric,
      params={
        "frequency": WALK_FREQUENCY,
        "command_name": "goal_pose",
        "cmd_threshold": 0.01,
      },
    ),
  }

  ##
  # Terminations
  ##
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(45.0)},
    ),
    # "root_height_below_minimum": TerminationTermCfg(
    #  func=mdp.root_height_below_minimum,
    #  params={"minimum_height": 0.25},
    # ),
  }

  ##
  # Curriculum
  ##
  curriculum = {}

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
    history_length=4,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  scene = SceneCfg(
    entities={"robot": get_piplus_robot_cfg()},
    sensors=(feet_contact_sensor_cfg, self_collision_cfg, foot_height_scan),
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
  episode_length_s = 30.0
  if play:
    episode_length_s = 1e10
    curriculum = {}
    metrics["foot_air_time"] = MetricsTermCfg(
      func=pi_mdp.current_air_time,
      params={"sensor_name": "feet_ground_contact"},
    )
    metrics["last_air_time"] = MetricsTermCfg(
      func=pi_mdp.last_air_time,
      params={"sensor_name": "feet_ground_contact"},
    )

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
    metrics=metrics,
    terminations=terminations,
    curriculum=curriculum,
    viewer=viewer,
    sim=sim,
    decimation=4,
    episode_length_s=episode_length_s,
  )
