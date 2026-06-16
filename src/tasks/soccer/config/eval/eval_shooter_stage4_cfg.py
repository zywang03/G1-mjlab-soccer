"""Stage IV shooter evaluation config — local coords, goal-plane target.

Reuses ``unitree_g1_stage4_env_cfg`` (which has Stage IV rewards, relaxed
motion tracking, and only time_out + fell_over terminations) and adds a
goal entity so eval scripts can detect goal-plane crossing.

All ball position / goal-line checks in eval scripts MUST use
``ball_world - env_origins`` because parallel envs have different origins.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.soccer.config.g1.training_env_cfgs import unitree_g1_stage4_env_cfg
from src.tasks.soccer.config.soccer_settings import SETTINGS
from src.tasks.soccer.goal import get_goal_cfg


def eval_shooter_stage4_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Stage IV eval config — adds goal entity to Stage IV training env."""
  cfg = unitree_g1_stage4_env_cfg(play=play)

  cfg.scene.entities["goal"] = get_goal_cfg(
    pos=(0.0, -5.0, 0.0),
    rot=(0.7071068, 0.0, 0.0, 0.7071068),
  )

  if not play:
    cfg.episode_length_s = SETTINGS.episode_length_s

  return cfg
