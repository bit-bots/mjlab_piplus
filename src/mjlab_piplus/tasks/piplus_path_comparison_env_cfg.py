"""Evaluate a velocity policy on the path-following benchmark.

This is the plain velocity task with a single swap in ``play`` mode: the random
``twist`` velocity command is replaced by a :class:`VelocityPathCommand`, which
drives the robot along the *same* reference paths the path task uses via a
follow-the-carrot controller. Everything else (observations, rewards, robot,
sensors) is identical to ``Mjlab-Velocity-PiPlus``, so a velocity-policy
checkpoint loads and runs unchanged — it just now sees velocity commands that
trace out a path instead of random ones.

With ``eval_mode`` on, the swapped command logs per-step path-following metrics
to ``velo_logs*.csv`` / ``commands_velo.csv``. Running the *path* policy on
``Mjlab-Path-Eval-PiPlus`` logs the same metric to ``path_logs*`` /
``commands_path`` over the identical (deterministically iterated) path library,
so the two can be compared directly.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_piplus.tasks.mdp.path_following import (
  ObstacleWorldCfg,
  VelocityPathCommandCfg,
)
from mjlab_piplus.tasks.piplus_path_env_cfg import (
  GOAL_POS_SUCCESS,
  GOAL_YAW_SUCCESS,
)
from mjlab_piplus.tasks.piplus_velocity_env_cfg import piplus_velocity_env_cfg


def piplus_path_comparison_env_cfg(
  play: bool = False, static: bool = False, obstacles: bool = False
) -> ManagerBasedRlEnvCfg:
  """Velocity task set up to follow reference paths (velocity-policy eval).

  Args:
    obstacles: If True, follow A*-through-obstacles paths (matching
      ``Mjlab-Path-Obstacles-PiPlus``) instead of free-space b-splines. Only the
      reference paths change; the obstacles are virtual (never simulated).

  Notes:
    The non-play cfg is just the velocity task (a registration placeholder — this
    env is only meaningful in play). The swap happens in play, where the velocity
    env already sets ``num_envs = 1``, unbounded episodes and disables the domain
    randomization / pushes, giving a clean single-robot eval.
  """
  cfg = piplus_velocity_env_cfg(play=play, static=static)
  if play:
    # Match the path task's play library exactly (same generation params via the
    # inherited PathCommandCfg defaults, same pathlib_size) so both policies are
    # scored on identical paths. eval_mode iterates the library deterministically
    # and writes the metric CSVs.
    cfg.commands["twist"] = VelocityPathCommandCfg(
      entity_name="robot",
      debug_vis=True,
      resampling_time_range=(30.0, 30.0),
      eval_mode=True,
      pathlib_size=1000,
      # Full-pose "goal reached" limits, identical to the path task's (the same
      # training goal-reward success thresholds). The carrot controller steers the
      # robot onto goal_yaw_w near the goal so the velocity policy is scored on the
      # same pose target.
      goal_pos_threshold=GOAL_POS_SUCCESS,
      goal_yaw_threshold=GOAL_YAW_SUCCESS,
    )
    if obstacles:
      cfg.commands["twist"].generator = "astar"
      cfg.commands["twist"].obstacle_world = ObstacleWorldCfg()
      cfg.commands["twist"].pathlib_size = 100
  return cfg
