"""Stage VI shooter evaluation config — compete-aligned coordinates.

Reuses ``unitree_g1_stage6_env_cfg`` which adds high-speed tuning on
top of Stage V's compete frame.  Adds a goal entity so eval scripts
can detect goal-plane crossing.

All ball position / goal-line checks in eval scripts MUST use
``ball_world - env_origins`` because parallel envs have different origins.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.soccer.config.g1.training_env_cfgs import unitree_g1_stage6_env_cfg
from src.tasks.soccer.config.soccer_settings import SETTINGS
from src.tasks.soccer.config.training.stage5_env_cfg import STAGE5_GOAL_POS_LOCAL
from src.tasks.soccer.goal import get_goal_cfg


def eval_shooter_stage6_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Stage VI eval config — adds goal entity to Stage VI training env."""
    cfg = unitree_g1_stage6_env_cfg(play=play)

    cfg.scene.entities["goal"] = get_goal_cfg(pos=STAGE5_GOAL_POS_LOCAL)

    if not play:
        cfg.episode_length_s = SETTINGS.episode_length_s

    return cfg
