import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_piplus.robot.piplus_constants import get_ball_cfg, get_piplus_robot_cfg
from mjlab_piplus.tasks import mdp as pi_mdp
from mjlab_piplus.terrain.astroturf import AstroturfTerrainCfg

DELAY_MIN = 0
DELAY_MAX = 3
WALK_FREQUENCY = 1.5


def piplus_pass_kick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  foot_sites = ("l_foot", "r_foot")
  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(
      ObjRef(type="site", name="l_foot", entity="robot"),
      ObjRef(type="site", name="r_foot", entity="robot"),
    ),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),
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
        noise_cfg=Unoise(n_min=-0.05, n_max=0.05),
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
    "actions": ObservationTermCfg(func=mdp.last_action),
    "phase": ObservationTermCfg(
      func=pi_mdp.phase,
      params={
        "frequency": WALK_FREQUENCY,
        "command_name": "pass_kick",
        "cmd_threshold": 0.01,
      },
    ),
    "ball_pos": ObservationTermCfg(
      func=pi_mdp.ball_pos_b,
      params={"robot_name": "robot", "ball_name": "ball"},
    ),
    "ball_vel": ObservationTermCfg(
      func=pi_mdp.ball_vel_b,
      params={"robot_name": "robot", "ball_name": "ball"},
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "pass_kick"},
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
    "pass_kick": pi_mdp.PassKickCommandCfg(
      resampling_time_range=(8.0, 12.0),
      debug_vis=True,
      target_distance_range=(0.5, 2.0),
      target_bearing_range=(-math.pi / 4, math.pi / 4),
      target_radius_range=(0.15, 0.35),
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
          "x": (-0.0, 0.0),
          "y": (-0.0, 0.0),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {
          "x": (-0.15, 0.15),
          "y": (-0.10, 0.10),
          "yaw": (-0.4, 0.4),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.2, 0.2),
        "velocity_range": (-0.5, 0.5),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "reset_ball": EventTermCfg(
      func=pi_mdp.reset_ball_around_robot,
      mode="reset",
      params={
        "distance_range": (0.25, 0.8),
        "bearing_range": (-math.pi / 4, math.pi / 4),
        "velocity_range": (0.0, 0.05),
        "rolling": True,
        "ball_radius": 0.1,
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(4.0, 8.0),
      params={
        "velocity_range": {
          "x": (-0.12, 0.12),
          "y": (-0.12, 0.12),
          "z": (-0.08, 0.08),
          "roll": (-0.25, 0.25),
          "pitch": (-0.25, 0.25),
          "yaw": (-0.35, 0.35),
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
        "asset_cfg": SceneEntityCfg("robot", actuator_names=r".*"),
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
    "ball_target_progress": RewardTermCfg(
      func=pi_mdp.ball_target_progress,
      weight=2.0,
      params={"command_name": "pass_kick", "max_speed": 2.0},
    ),
    "ball_close_to_target": RewardTermCfg(
      func=pi_mdp.ball_close_to_target,
      weight=2.0,
      params={"command_name": "pass_kick", "epsilon": 3.0},
    ),
    "ball_velocity_to_target": RewardTermCfg(
      func=pi_mdp.ball_velocity_to_target,
      weight=1.0,
      params={"command_name": "pass_kick", "max_speed": 2.0},
    ),
    "ball_success": RewardTermCfg(
      func=pi_mdp.ball_success,
      weight=20.0,
      params={"command_name": "pass_kick", "max_ball_speed": 0.1},
    ),
    "ball_speed_near_target": RewardTermCfg(
      func=pi_mdp.ball_speed_near_target,
      weight=-2.0,
      params={"command_name": "pass_kick", "max_ball_speed": 0.1},
    ),
    "approach_ball_progress": RewardTermCfg(
      func=pi_mdp.approach_ball_progress,
      weight=0.5,
      params={"command_name": "pass_kick"},
    ),
    "close_to_ball": RewardTermCfg(
      func=pi_mdp.close_to_ball,
      weight=0.25,
      params={"command_name": "pass_kick", "epsilon": 2.0},
    ),
    "ball_in_kick_zone": RewardTermCfg(
      func=pi_mdp.ball_in_kick_zone,
      weight=0.5,
      params={"command_name": "pass_kick"},
    ),
    "face_ball_or_target": RewardTermCfg(
      func=pi_mdp.face_ball_or_target,
      weight=0.25,
      params={"command_name": "pass_kick"},
    ),
    "standstill_before_kick": RewardTermCfg(
      func=pi_mdp.standstill_before_kick,
      weight=-0.1,
      params={"command_name": "pass_kick"},
    ),
    "upright": RewardTermCfg(
      func=pi_mdp.body_orientation_l2,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
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
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.025),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 2.0},
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=0.5,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "pass_kick",
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
    "foot_gait": RewardTermCfg(
      func=pi_mdp.feet_gait,
      weight=0.25,
      params={
        "frequency": WALK_FREQUENCY,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_threshold": 0.01,
        "command_name": "pass_kick",
        "sensor_name": "feet_ground_contact",
      },
    ),
    "foot_slip": RewardTermCfg(
      func=pi_mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "pass_kick",
        "command_threshold": 0.01,
        "asset_cfg": SceneEntityCfg("robot", site_names=foot_sites),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=pi_mdp.soft_landing,
      weight=-1e-2,
      params={"sensor_name": "feet_ground_contact", "command_name": "pass_kick"},
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "base_ang_vel_xy_l2": MetricsTermCfg(func=pi_mdp.base_ang_vel_xy_l2),
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
        "command_name": "pass_kick",
        "cmd_threshold": 0.01,
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=pi_mdp.time_out_jittered, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(45.0)},
    ),
    "root_height_below_minimum": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.28},
    ),
    "ball_out_of_bounds": TerminationTermCfg(
      func=pi_mdp.ball_out_of_bounds,
      params={"ball_name": "ball", "max_distance_from_origin": 6.0},
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "ball_reset_range": CurriculumTermCfg(
      func=pi_mdp.event_curriculum,
      params={
        "event_name": "reset_ball",
        "stages": [
          {
            "step": 0,
            "params": {
              "distance_range": (0.25, 0.8),
              "bearing_range": (-math.pi / 4, math.pi / 4),
              "velocity_range": (0.0, 0.05),
            },
          },
          {
            "step": 800 * 24,
            "params": {
              "distance_range": (0.25, 1.2),
              "bearing_range": (-math.pi / 2, math.pi / 2),
              "velocity_range": (0.0, 0.15),
            },
          },
          {
            "step": 1600 * 24,
            "params": {
              "distance_range": (0.25, 1.5),
              "bearing_range": (-math.pi, math.pi),
              "velocity_range": (0.0, 0.25),
            },
          },
        ],
      },
    )
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
    num_envs=4096,
    entities={"robot": get_piplus_robot_cfg(), "ball": get_ball_cfg()},
    sensors=(feet_contact_sensor_cfg, self_collision_cfg, foot_height_scan),
    terrain=AstroturfTerrainCfg(
      terrain_type="astroturf",
      top_solref=(0.02, 1.0),
      turf_thickness=0.03,
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
    nconmax=60,
    njmax=1500,
    contact_sensor_maxmatch=500,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
      ccd_iterations=500,
    ),
  )
  episode_length_s = 15.0
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
