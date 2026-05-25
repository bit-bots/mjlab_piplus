import math
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.tasks.velocity import mdp
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab_piplus.robot.piplus_constants import get_piplus_robot_cfg, get_ball_cfg
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
    ball_pos = ball.data.root_link_pos_w[:, :2]
    robot_pos = robot.data.root_link_pos_w[:, :2]
    rel_pos_w = ball_pos - robot_pos

    heading = robot.data.heading_w
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    rel_x_b = cos_h * rel_pos_w[:, 0] + sin_h * rel_pos_w[:, 1]
    rel_y_b = -sin_h * rel_pos_w[:, 0] + cos_h * rel_pos_w[:, 1]

    return torch.stack((rel_x_b, rel_y_b), dim=1)


def distance_to_ball(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    ball_pos = ball.data.root_link_pos_w[:, :2]
    robot_pos = robot.data.root_link_pos_w[:, :2]
    distance = torch.norm(ball_pos - robot_pos, dim=1, keepdim=True)
    return distance


def _ball_relative_xy(env: ManagerBasedRlEnv):
    """Return robot-ball relative xy data in world frame."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    ball_pos = ball.data.root_link_pos_w[:, :2]
    robot_pos = robot.data.root_link_pos_w[:, :2]
    rel_pos = ball_pos - robot_pos
    distance = torch.norm(rel_pos, dim=1)
    direction = rel_pos / distance.unsqueeze(-1).clamp_min(1e-6)
    return robot, rel_pos, distance, direction


ARM_JOINT_PATTERNS = (
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_upper_arm_joint",
    ".*_elbow_joint",
)

LEG_JOINT_PATTERNS = (
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_thigh_joint",
    ".*_calf_joint",
)

ANKLE_JOINT_PATTERNS = (
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
)


def toward_ball_velocity_reward(
    env,
    asset_cfg: SceneEntityCfg = None,
    target_speed: float = 0.3,
    stop_distance: float = 0.26,
    fade_distance: float = 0.20,
    facing_cos_min: float = 0.5,
) -> torch.Tensor:
    """Reward signed velocity component toward ball."""
    del asset_cfg
    robot, _, distance, direction = _ball_relative_xy(env)
    vel_xy = robot.data.root_link_lin_vel_w[:, :2]
    toward_speed = torch.sum(vel_xy * direction, dim=1)
    toward = torch.clamp(toward_speed / max(target_speed, 1e-6), min=-1.0, max=1.0)
    heading = robot.data.heading_w
    heading_vec = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
    face_cos = torch.sum(heading_vec * direction, dim=1)
    face = torch.clamp(
        (face_cos - facing_cos_min) / max(1.0 - facing_cos_min, 1e-6),
        min=0.0,
        max=1.0,
    )
    move_gate = torch.clamp(
        (distance - stop_distance) / max(fade_distance, 1e-6), min=0.0, max=1.0
    )
    return toward * move_gate * face


def face_ball_near_reward(
    env,
    asset_cfg: SceneEntityCfg = None,
    near_distance: float = 0.6,
    facing_cos_min: float = 0.7,
) -> torch.Tensor:
    """Reward facing ball when already near it."""
    del asset_cfg
    robot, _, distance, direction = _ball_relative_xy(env)

    heading = robot.data.heading_w
    heading_vec = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
    face_cos = torch.sum(heading_vec * direction, dim=1)

    near = torch.clamp(distance.new_tensor(near_distance) - distance, min=0.0) / max(
        near_distance, 1e-6
    )
    face = torch.clamp(
        (face_cos - facing_cos_min) / max(1.0 - facing_cos_min, 1e-6),
        min=0.0,
        max=1.0,
    )
    return near * face


class SuccessFacingBallTermination:
    """Terminate when robot is near ball, facing it, and settled for N steps."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self.hold_counter = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.int32
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.hold_counter[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        min_distance_threshold: float = 0.23,
        max_distance_threshold: float = 0.35,
        facing_cos_threshold: float = 0.90,
        speed_threshold: float = 0.10,
        hold_steps: int = 6,
    ) -> torch.Tensor:
        robot, _, distance, direction = _ball_relative_xy(env)

        heading = robot.data.heading_w
        heading_vec = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
        face_cos = torch.sum(heading_vec * direction, dim=1)
        speed_xy = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)

        hold_steps = max(int(hold_steps), 1)
        success_step = (
            (distance > min_distance_threshold)
            & (distance < max_distance_threshold)
            & (face_cos > facing_cos_threshold)
            & (speed_xy < speed_threshold)
        )

        self.hold_counter = torch.where(
            success_step,
            self.hold_counter + 1,
            torch.zeros_like(self.hold_counter),
        )
        return self.hold_counter >= hold_steps


def fell_over_only_penalty(
    env, asset_cfg: SceneEntityCfg = None, term_name: str = "fell_over"
) -> torch.Tensor:
    del asset_cfg
    return env.termination_manager.get_term(term_name).float()


def success_bonus(
    env, asset_cfg: SceneEntityCfg = None, term_name: str = "success_facing_ball"
) -> torch.Tensor:
    del asset_cfg
    return env.termination_manager.get_term(term_name).float()


def approach_ball_reward(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """Potential-style distance reward for being close to ball."""
    del asset_cfg
    _, _, distance, _ = _ball_relative_xy(env)
    distance_for_reward = torch.clamp(distance, min=0.26)
    return torch.exp(-distance_for_reward)


def near_ball_speed_penalty(
    env,
    asset_cfg: SceneEntityCfg = None,
    min_distance: float = 0.23,
    max_distance: float = 0.35,
    speed_scale: float = 0.25,
) -> torch.Tensor:
    del asset_cfg
    robot, _, distance, _ = _ball_relative_xy(env)
    speed_xy = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
    in_band = ((distance > min_distance) & (distance < max_distance)).float()
    speed_term = torch.clamp(speed_xy / max(speed_scale, 1e-6), max=2.0)
    return in_band * speed_term


def piplus_walk_to_ball_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
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
        "ball_position": ObservationTermCfg(func=ball_position_command),
        "distance_to_ball": ObservationTermCfg(func=distance_to_ball),
    }
    critic_terms = {**actor_terms}
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
            actuator_names=LEG_JOINT_PATTERNS + ANKLE_JOINT_PATTERNS,
            scale=0.35,
            use_default_offset=True,
        )
    }

    commands = {}
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
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    geom_names=(
                        "r_ankle_roll_link_collision0",
                        "r_ankle_roll_link_collision1",
                        "r_ankle_roll_link_collision2",
                        "l_ankle_roll_link_collision0",
                        "l_ankle_roll_link_collision1",
                        "l_ankle_roll_link_collision2",
                    ),
                ),
                "operation": "abs",
                "ranges": (0.4, 1.1),
                "shared_random": True,
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.012, 0.012),
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
                "operation": "add",
                "ranges": {
                    0: (-0.015, 0.015),
                    1: (-0.015, 0.015),
                    2: (-0.02, 0.02),
                },
            },
        ),
        "actuator_pd_gains": EventTermCfg(
            mode="startup",
            func=dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "kp_range": (0.9, 1.1),
                "kd_range": (0.9, 1.1),
                "operation": "scale",
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.5, 4.0),
            params={
                "velocity_range": {
                    "x": (-0.15, 0.15),
                    "y": (-0.15, 0.15),
                    "z": (-0.1, 0.1),
                    "roll": (-0.2, 0.2),
                    "pitch": (-0.2, 0.2),
                    "yaw": (-0.35, 0.35),
                },
                "asset_cfg": SceneEntityCfg("robot"),
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
                    "z": (0.10, 0.10),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-0.5, 0.5),
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
                    "x": (-0.25, 0.25),
                    "y": (-0.25, 0.25),
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
        "alive": RewardTermCfg(
            func=mdp.is_alive,
            weight=0.2,
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-0.001,
            params={
                "sensor_name": "feet_ground_contact",
            },
        ),
        "upright": RewardTermCfg(
            func=mdp.upright,
            weight=2.0,
            params={
                "std": math.sqrt(0.2),
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
            },
        ),
        "leg_pose": RewardTermCfg(
            func=mdp.posture,
            weight=0.8,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERNS),
                "std": {".*": 0.45},
            },
        ),
        "ankle_pose": RewardTermCfg(
            func=mdp.posture,
            weight=0.6,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_JOINT_PATTERNS),
                "std": {".*": 0.08},
            },
        ),
        "arm_pose": RewardTermCfg(
            func=mdp.posture,
            weight=0.7,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_PATTERNS),
                "std": {".*": 0.12},
            },
        ),
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=-0.02,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.08),
        "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.005),
        "arm_joint_vel_l2": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-0.002,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ARM_JOINT_PATTERNS,
                )
            },
        ),
        "leg_joint_vel_l2": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-0.0005,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=LEG_JOINT_PATTERNS,
                )
            },
        ),
        "ankle_joint_vel_l2": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-0.001,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ANKLE_JOINT_PATTERNS,
                )
            },
        ),
        "arm_torque_l2": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-2e-4,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    actuator_names=ARM_JOINT_PATTERNS,
                )
            },
        ),
        "leg_torque_l2": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-8e-5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    actuator_names=LEG_JOINT_PATTERNS,
                )
            },
        ),
        "ankle_torque_l2": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-1.6e-4,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    actuator_names=ANKLE_JOINT_PATTERNS,
                )
            },
        ),
        "air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.15,
            params={
                "sensor_name": "feet_ground_contact",
                "threshold_min": 0.1,
                "threshold_max": 0.45,
            },
        ),
        "toward_ball_velocity": RewardTermCfg(
            func=toward_ball_velocity_reward,
            weight=0.8,
            params={
                "target_speed": 0.3,
                "stop_distance": 0.26,
                "fade_distance": 0.20,
                "facing_cos_min": 0.4,
            },
        ),
        "face_ball_near": RewardTermCfg(
            func=face_ball_near_reward,
            weight=1.3,
            params={"near_distance": 0.6, "facing_cos_min": 0.75},
        ),
        "approach_ball": RewardTermCfg(
            func=approach_ball_reward,
            weight=0.5,
            params={},
        ),
        "near_ball_speed": RewardTermCfg(
            func=near_ball_speed_penalty,
            weight=-2.0,
            params={
                "min_distance": 0.23,
                "max_distance": 0.35,
                "speed_scale": 0.25,
            },
        ),
        "success_bonus": RewardTermCfg(
            func=success_bonus,
            weight=300.0,
            params={"term_name": "success_facing_ball"},
        ),
        "termination_penalty": RewardTermCfg(
            func=fell_over_only_penalty,
            weight=-300.0,
            params={"term_name": "fell_over"},
        ),
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
        "success_facing_ball": TerminationTermCfg(
            func=SuccessFacingBallTermination,
            params={
                "min_distance_threshold": 0.23,
                "max_distance_threshold": 0.35,
                "facing_cos_threshold": 0.90,
                "speed_threshold": 0.10,
                "hold_steps": 6,
            },
        ),
    }

    ##
    # Curriculum
    ##

    curriculum = {
        "ball_distance": CurriculumTermCfg(
            func=ball_distance_curriculum,
            params={
                "distance_stages": [
                    {"step": 0, "x": (0.3, 0.6), "y": (-0.2, 0.2)},
                    {"step": 600 * 24, "x": (0.6, 1.0), "y": (-0.3, 0.3)},
                    {"step": 1600 * 24, "x": (0.9, 1.6), "y": (-0.5, 0.5)},
                    {"step": 3200 * 24, "x": (1.2, 2.0), "y": (-0.5, 0.5)},
                ],
            },
        ),
    }

    ##
    # Scene
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
    scene = SceneCfg(
        entities={
            "robot": get_piplus_robot_cfg(),
            "ball": get_ball_cfg(),
        },
        sensors=[
            feet_contact_sensor_cfg,
        ],
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
    # Assemble the confgi
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
