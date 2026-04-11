
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
from mjlab.tasks.velocity.mdp.rewards import variable_posture as VelocityVariablePosture
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab_piplus.robot.piplus_constants import get_piplus_robot_cfg, get_ball_cfg

from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.sensor import ContactMatch, ContactSensorCfg


def ball_distance_curriculum(env, env_ids, distance_stages: list) -> dict:
    step = env.common_step_counter
    current_stage = distance_stages[0]
    for stage in distance_stages:
        if step >= stage["step"]:
            current_stage = stage

    event_cfg = env.event_manager.get_term_cfg("reset_ball")
    event_cfg.params["pose_range"]["x"] = current_stage["x"]
    event_cfg.params["pose_range"]["y"] = current_stage["y"]

    return {
        "min_distance": torch.tensor(current_stage["x"][0]),
        "max_distance": torch.tensor(current_stage["x"][1]),
    }


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


class variable_posture_actual_speed(VelocityVariablePosture):
    """Pose reward with regime masks from robot actual speed.

    Keeps reward form and per-joint std tensors from base class, but chooses
    standing/walking/running masks using measured root-link velocities instead
    of commanded velocities.
    """

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_standing,
        std_walking,
        std_running,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        walking_threshold: float = 0.05,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running, command_name

        asset = env.scene[asset_cfg.name]

        linear_speed = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
        angular_speed = torch.abs(asset.data.root_link_ang_vel_b[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = (
            (total_speed >= walking_threshold) & (total_speed < running_threshold)
        ).float()
        running_mask = (total_speed >= running_threshold).float()

        std = (
            self.std_standing * standing_mask.unsqueeze(1)
            + self.std_walking * walking_mask.unsqueeze(1)
            + self.std_running * running_mask.unsqueeze(1)
        )

        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)

        return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


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
            rel_heading_envs=0.5,
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
        "upright": RewardTermCfg(
            func=mdp.flat_orientation,
            weight=2.0,
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
            func=variable_posture_actual_speed,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "command_name": "twist",
                "std_standing": {
                    r".*_hip_pitch_joint": 0.08,
                    r".*_hip_roll_joint": 0.06,
                    r".*_thigh_joint": 0.05,
                    r".*_calf_joint": 0.10,
                    r".*_ankle_pitch_joint": 0.08,
                    r".*_ankle_roll_joint": 0.06,
                    r".*_shoulder_.*": 0.15,
                    r".*_upper_arm_joint": 0.12,
                    r".*_elbow_joint": 0.12,
                },
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
        "alive": RewardTermCfg(
            func=mdp.is_alive,
            weight=1.0,
        ),
        "termination_penalty": RewardTermCfg(
            func=mdp.is_terminated,
            weight=-50.0,
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
        "ball_distance": CurriculumTermCfg(
            func=ball_distance_curriculum,
            params={
                "distance_stages": [
                    {"step": 0,          "x": (0.3, 0.5), "y": (-0.2, 0.2)},
                    {"step": 500 * 24,   "x": (0.5, 1.0), "y": (-0.3, 0.3)},
                    {"step": 1500 * 24,  "x": (0.8, 1.5), "y": (-0.5, 0.5)},
                    {"step": 3000 * 24,  "x": (1.0, 2.0), "y": (-0.5, 0.5)},
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
    else:
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
