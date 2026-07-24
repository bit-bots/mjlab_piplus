from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner

from .piplus_path_comparison_env_cfg import piplus_path_comparison_env_cfg
from .piplus_path_env_cfg import piplus_path_env_cfg
from .piplus_path_rl_cfg import get_piplus_path_rl_cfg
from .piplus_velocity_env_cfg import piplus_velocity_env_cfg
from .piplus_velocity_rl_cfg import get_piplus_velocity_rl_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-PiPlus",
  env_cfg=piplus_velocity_env_cfg(),
  play_env_cfg=piplus_velocity_env_cfg(play=True),
  rl_cfg=get_piplus_velocity_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Path-PiPlus",
  env_cfg=piplus_path_env_cfg(),
  play_env_cfg=piplus_path_env_cfg(play=True),
  rl_cfg=get_piplus_path_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
# Same task, but the reference paths are planned with A* through a random
# obstacle world (virtual obstacles: drawn in the viewer, never simulated).
register_mjlab_task(
  task_id="Mjlab-Path-Obstacles-PiPlus",
  env_cfg=piplus_path_env_cfg(obstacles=True),
  play_env_cfg=piplus_path_env_cfg(play=True, obstacles=True),
  rl_cfg=get_piplus_path_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

##
# Path-following comparison: score the velocity policy and the path policy on the
# same path benchmark.
#
#  - Play the PATH policy on `Mjlab-Path-Eval-PiPlus` -> path_logs*.csv.
#  - Play the VELOCITY policy on `Mjlab-PathComparison-PiPlus` -> velo_logs*.csv
#    (the velocity command is driven along the same paths by a carrot follower).
#
# Both iterate the identical (deterministically ordered) path library, so the
# logs line up path-for-path. `*-Obstacles-*` variants use the A* obstacle paths.
##
register_mjlab_task(
  task_id="Mjlab-Path-Eval-PiPlus",
  env_cfg=piplus_path_env_cfg(eval=True),
  play_env_cfg=piplus_path_env_cfg(play=True, eval=True),
  rl_cfg=get_piplus_path_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-Path-Eval-Obstacles-PiPlus",
  env_cfg=piplus_path_env_cfg(obstacles=True, eval=True),
  play_env_cfg=piplus_path_env_cfg(play=True, obstacles=True, eval=True),
  rl_cfg=get_piplus_path_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-PathComparison-PiPlus",
  env_cfg=piplus_path_comparison_env_cfg(),
  play_env_cfg=piplus_path_comparison_env_cfg(play=True),
  rl_cfg=get_piplus_velocity_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Mjlab-PathComparison-Obstacles-PiPlus",
  env_cfg=piplus_path_comparison_env_cfg(obstacles=True),
  play_env_cfg=piplus_path_comparison_env_cfg(play=True, obstacles=True),
  rl_cfg=get_piplus_velocity_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
