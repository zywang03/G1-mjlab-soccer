"""Register evaluation tasks for shooter and goalkeeper.

The shooter eval reuses the Stage II play config directly (via
eval_shooter_env_cfg) so the environment is identical to play mode.
"""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.soccer.config.eval.eval_goalkeeper_cfg import eval_goalkeeper_env_cfg
from src.tasks.soccer.config.eval.eval_shooter_cfg import eval_shooter_env_cfg
from src.tasks.soccer.config.eval.eval_shooter_stage3_cfg import eval_shooter_stage3_env_cfg
from src.tasks.soccer.config.eval.eval_shooter_stage4_cfg import eval_shooter_stage4_env_cfg
from src.tasks.soccer.config.eval.eval_shooter_stage5_cfg import eval_shooter_stage5_env_cfg
from src.tasks.soccer.config.eval.eval_shooter_stage6_cfg import eval_shooter_stage6_env_cfg
from src.tasks.soccer.config.g1.rl_cfg import (
    SoccerRecurrentRunner,
    unitree_g1_soccer_ppo_runner_cfg,
    unitree_g1_soccer_recurrent_runner_cfg,
)


register_mjlab_task(
  task_id="Eval-Shooter",
  env_cfg=eval_shooter_env_cfg(play=False),
  play_env_cfg=eval_shooter_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="Eval-Shooter-Stage3",
  env_cfg=eval_shooter_stage3_env_cfg(play=False),
  play_env_cfg=eval_shooter_stage3_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="Eval-Shooter-Stage4",
  env_cfg=eval_shooter_stage4_env_cfg(play=False),
  play_env_cfg=eval_shooter_stage4_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="Eval-Shooter-Stage5",
  env_cfg=eval_shooter_stage5_env_cfg(play=False),
  play_env_cfg=eval_shooter_stage5_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_recurrent_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)

register_mjlab_task(
  task_id="Eval-Shooter-Stage6",
  env_cfg=eval_shooter_stage6_env_cfg(play=False),
  play_env_cfg=eval_shooter_stage6_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_recurrent_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)

register_mjlab_task(
  task_id="Eval-Goalkeeper",
  env_cfg=eval_goalkeeper_env_cfg(play=False),
  play_env_cfg=eval_goalkeeper_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)
