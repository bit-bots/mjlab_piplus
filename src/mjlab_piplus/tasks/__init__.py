from mjlab.tasks.registry import register_mjlab_task

from .piplus_velocity_rl_cfg import get_piplus_velocity_rl_cfg
from .piplus_velocity_env_cfg import piplus_velocity_env_cfg
from .piplus_walk_to_ball_env_cfg import piplus_walk_to_ball_env_cfg
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner


register_mjlab_task(
    task_id="Mjlab-Velocity-PiPlus",
    env_cfg=piplus_velocity_env_cfg(),
    play_env_cfg=piplus_velocity_env_cfg(play=True),
    rl_cfg=get_piplus_velocity_rl_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-WalkToBall-PiPlus",
    env_cfg=piplus_walk_to_ball_env_cfg(),
    play_env_cfg=piplus_walk_to_ball_env_cfg(play=True),
    rl_cfg=get_piplus_velocity_rl_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
