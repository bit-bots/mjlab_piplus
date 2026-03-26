
from copy import deepcopy
import math
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.tasks.velocity import mdp
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab_piplus.robot.piplus_constants import get_piplus_robot_cfg, get_ball_cfg

from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.sensor import ContactMatch, ContactSensorCfg


def ball_position_command(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    ball_pos = ball.data.root_link_pos_w
    robot_pos = robot.data.root_link_pos_w
    rel_pos = ball_pos - robot_pos
    return rel_pos[:, :2]


def distance_to_ball(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    ball_pos = ball.data.root_link_pos_w[:, :2]
    robot_pos = robot.data.root_link_pos_w[:, :2]
    distance = torch.norm(ball_pos - robot_pos, dim=1, keepdim=True)
    return distance


def piplus_walk_to_ball_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    foot_sites = ("l_foot", "r_foot")

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
        "ball_position": ObservationTermCfg(
            func=ball_position_command,
        ),
        "distance_to_ball": ObservationTermCfg(
            func=distance_to_ball,
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

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
            use_default_offset=True,
        )
    }

    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
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
        "reset_ball": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (1.0, 2.0),
                    "y": (-0.5, 0.5),
                    "z": (0.11, 0.11),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "asset_cfg": SceneEntityCfg("ball"),
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
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
        ),
        "angular_momentum": RewardTermCfg(
            func=mdp.angular_momentum_penalty,
            weight=-0.02,
            params={"sensor_name": "robot/root_angmom"},
        ),
        "pose": RewardTermCfg(
            func=mdp.variable_posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "command_name": "twist",
                "std_standing": {".*": 0.05},
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
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.25,
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
            weight=-2.0,
            params={
                "target_height": 0.05,
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=foot_sites),
            },
        ),
        "foot_swing_height": RewardTermCfg(
            func=mdp.feet_swing_height,
            weight=-0.5,
            params={
                "sensor_name": "feet_ground_contact",
                "target_height": 0.1,
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
        "track_ball_velocity": RewardTermCfg(
            func=track_ball_velocity,
            weight=1.5,
            params={},
        ),
        "reach_ball": RewardTermCfg(
            func=reach_ball_reward,
            weight=2.0,
            params={},
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }

    curriculum = {
        "command_vel": CurriculumTermCfg(
            func=mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {"step": 0, "lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
                    {"step": 100 * 24, "lin_vel_x": (-0.25, 0.25), "lin_vel_y": (-0.1, 0.1), "ang_vel_z": (-0.5, 0.5)},
                    {"step": 300 * 24, "lin_vel_x": (-0.5, 0.5), "lin_vel_y": (-0.2, 0.2), "ang_vel_z": (-1.0, 1.0)},
                    {"step": 500 * 24, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
                ],
            },
        ),
    }

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
        entities={
            "robot": get_piplus_robot_cfg(),
            "ball": get_ball_cfg(),
        },
        sensors=(feet_contact_sensor_cfg, self_collision_cfg),
        terrain=TerrainEntityCfg(
            terrain_type="plane",
            terrain_generator=None,
            max_init_terrain_level=0,
        ),
    )

    viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="base_link",
        distance=3.0,
        elevation=-5.0,
        azimuth=90.0,
    )

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


def track_ball_velocity(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    ball_pos = ball.data.root_link_pos_w[:, :2]
    robot_pos = robot.data.root_link_pos_w[:, :2]
    rel_pos = ball_pos - robot_pos

    rel_pos_norm = torch.norm(rel_pos, dim=1)
    rel_pos_normalized = rel_pos / (rel_pos_norm.unsqueeze(-1) + 1e-6)

    robot_lin_vel = robot.data.root_link_lin_vel_w[:, :2]

    desired_vel = rel_pos_normalized * 0.5

    vel_error = torch.norm(robot_lin_vel - desired_vel, dim=1)
    reward = torch.exp(-vel_error)
    return reward


def reach_ball_reward(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    ball_pos = ball.data.root_link_pos_w[:, :2]
    robot_pos = robot.data.root_link_pos_w[:, :2]
    distance = torch.norm(ball_pos - robot_pos, dim=1)

    reward = torch.where(distance < 0.3, torch.ones_like(distance), torch.zeros_like(distance))
    return reward
