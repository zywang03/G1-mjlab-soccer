"""Stage IV training config: high-speed goal-plane accuracy (target 10 m/s).

Builds on Stage III by increasing speed shaping range and strengthening
fall-over / ground-contact penalties for high-speed stability.  Stage III
is NOT modified — this factory creates a new config object from
``make_stage3_env_cfg()`` and only mutates its own copy.

Key changes from Stage III:
- ``ball_speed`` sigma widened to 5.0 (10 m/s target), weight 20.0
- ``is_terminated`` penalty increased to -500 (harder constraint at speed)
- ``undesired_contacts`` penalty increased to -1.0 (early warning for body drag)
- ``goals_miss`` penalty increased to -20 (from -5) to strongly discourage off-target kicks
- All goal-plane, adaptive-target, and motion-tracking settings inherited unchanged
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from src.tasks.soccer.config.training.stage3_env_cfg import make_stage3_env_cfg
from src.tasks.soccer.mdp.shooter_rewards import goal_miss_penalty


def make_stage4_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Stage IV (high-speed goal-plane accuracy) environment config."""

  cfg = make_stage3_env_cfg()

  # -- Rewards: widen speed shaping, strengthen fall penalties -----------------

  cfg.rewards["ball_speed"] = replace(
    cfg.rewards["ball_speed"],
    weight=20.0,
    params={**cfg.rewards["ball_speed"].params, "std": 5.0},
  )

  cfg.rewards["is_terminated"] = replace(
    cfg.rewards["is_terminated"],
    weight=-500.0,
  )

  cfg.rewards["undesired_contacts"] = replace(
    cfg.rewards["undesired_contacts"],
    weight=-1.0,
  )

  cfg.rewards["goal_miss"] = RewardTermCfg(
    func=goal_miss_penalty,
    weight=-20.0,
    params={"command_name": "motion"},
  )

  return cfg
