"""Unitree G1 soccer task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_g1_shooter_env_cfg,
  unitree_g1_goalkeeper_env_cfg,
)
from .rl_cfg import (
  SoccerRecurrentRunner,
  unitree_g1_soccer_ppo_runner_cfg,
  unitree_g1_soccer_recurrent_runner_cfg,
  unitree_g1_student_shooter_ppo_runner_cfg,
)
from .training_env_cfgs import (
  unitree_g1_stage1_env_cfg,
  unitree_g1_stage2_env_cfg,
  unitree_g1_stage3_env_cfg,
  unitree_g1_student_shooter_env_cfg,
)

# -- Playable tasks ------------------------------------------------------------

register_mjlab_task(
  task_id="Unitree-G1-Shooter",
  env_cfg=unitree_g1_shooter_env_cfg(),
  play_env_cfg=unitree_g1_shooter_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper",
  env_cfg=unitree_g1_goalkeeper_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)

# -- Training tasks ------------------------------------------------------------

# Stage I: motion tracking (LSTM, 128-64-32 + LSTM 2×128, matching PAiD paper)
register_mjlab_task(
  task_id="Unitree-G1-Shooter-Stage1",
  env_cfg=unitree_g1_stage1_env_cfg(),
  play_env_cfg=unitree_g1_stage1_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_recurrent_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)

# Stage II: perception-guided kicking (LSTM, 128-64-32 + LSTM 2×128)
register_mjlab_task(
  task_id="Unitree-G1-Shooter-Stage2",
  env_cfg=unitree_g1_stage2_env_cfg(),
  play_env_cfg=unitree_g1_stage2_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_recurrent_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)

# Stage III: goal-plane accuracy + speed curriculum (LSTM, same architecture)
register_mjlab_task(
  task_id="Unitree-G1-Shooter-Stage3",
  env_cfg=unitree_g1_stage3_env_cfg(),
  play_env_cfg=unitree_g1_stage3_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_recurrent_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)

# Student shooter: motion-free observation, LSTM policy (BC→PPO).
# SoccerRecurrentRunner injects rnn_type/layers/hidden_dim into the config dict.
register_mjlab_task(
  task_id="Unitree-G1-Shooter-Student",
  env_cfg=unitree_g1_student_shooter_env_cfg(),
  play_env_cfg=unitree_g1_student_shooter_env_cfg(play=True),
  rl_cfg=unitree_g1_student_shooter_ppo_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)
