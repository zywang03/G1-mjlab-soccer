"""Shooter evaluation config — reuses Stage II play config, adds goal entity.

Uses the EXACT same environment as play Stage 2: same observations,
ball placement (command-driven, randomized per motion), robot
positioning, sampling mode, and noise settings.  The only difference
is a goal entity added so the eval script can detect when the ball
crosses the goal line.

Goal is placed in motion-local coords matching the default
destination_center = (0, -5, 0.11): goal at (0, -5, 0), rotated 90°
about Z so the opening faces ±y (toward G1).
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.soccer.config.g1.training_env_cfgs import unitree_g1_stage2_env_cfg
from src.tasks.soccer.config.soccer_settings import SETTINGS
from src.tasks.soccer.goal import get_goal_cfg


def eval_shooter_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Shooter eval config — identical to play Stage 2 + goal entity.

  Reuses unitree_g1_stage2_env_cfg(play=True) so observations, ball
  placement, robot positioning, sampling, and noise settings are all
  identical to play mode.
  """
  cfg = unitree_g1_stage2_env_cfg(play=True)

  # Add goal entity in front of G1 (motion-local coords: G1 faces -y).
  # destination_center defaults to (0, -5, 0.11) — place goal there,
  # rotated 90° about Z so opening faces ±y.
  cfg.scene.entities["goal"] = get_goal_cfg(
    pos=(0.0, -5.0, 0.0),
    rot=(0.7071068, 0.0, 0.0, 0.7071068),
  )

  # Eval episode length (play uses infinite).
  if not play:
    cfg.episode_length_s = SETTINGS.episode_length_s

  return cfg
