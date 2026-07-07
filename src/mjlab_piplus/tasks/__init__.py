from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner

from .piplus_goal_pose_env_cfg import piplus_goal_pose_env_cfg
from .piplus_pass_kick_env_cfg import piplus_pass_kick_env_cfg
from .piplus_velocity_env_cfg import piplus_velocity_env_cfg
from .piplus_velocity_rl_cfg import (
  get_piplus_goal_pose_rl_cfg,
  get_piplus_pass_kick_rl_cfg,
  get_piplus_velocity_rl_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-PiPlus",
  env_cfg=piplus_velocity_env_cfg(),
  play_env_cfg=piplus_velocity_env_cfg(play=True),
  rl_cfg=get_piplus_velocity_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-GoalPose-PiPlus",
  env_cfg=piplus_goal_pose_env_cfg(),
  play_env_cfg=piplus_goal_pose_env_cfg(play=True),
  rl_cfg=get_piplus_goal_pose_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-PassKick-PiPlus",
  env_cfg=piplus_pass_kick_env_cfg(),
  play_env_cfg=piplus_pass_kick_env_cfg(play=True),
  rl_cfg=get_piplus_pass_kick_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
