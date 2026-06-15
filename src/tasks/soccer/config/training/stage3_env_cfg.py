"""Stage III training config: goal-plane accuracy + speed curriculum on flat ground.

Builds on Stage II by relaxing motion tracking constraints and adding
goal-plane crossing rewards (accuracy + miss penalty).  Stage II is
NOT modified — this factory creates a new config object from
``make_stage2_env_cfg()`` and only mutates its own copy.

Key changes from Stage II:
- Goal-plane target: x in [-1.25, 1.25], y fixed at -5.0
- ``ball_vel_align`` sigma tightened to 0.3
- ``ball_speed`` sigma 3.0, weight 15.0 (curriculum overrides via CLI)
- ``z_speed`` penalty enabled (w=-2.0)
- New ``goal_accuracy`` (w=10) + ``goal_miss`` (w=-5)
- Motion tracking weights reduced to 0.5 (anchor_pos stays 0.0)
- Motion-reference terminations removed (only time_out + fell_over remain)
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from src.tasks.soccer.config.training.stage2_env_cfg import make_stage2_env_cfg
from src.tasks.soccer.mdp.shooter_rewards import goal_plane_accuracy, goal_miss_penalty


def make_stage3_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Stage III (goal-plane accuracy + speed) environment config."""

  cfg = make_stage2_env_cfg()

  # -- Command: widen lateral target, fix y on goal plane --------------------
  motion_cmd = cfg.commands["motion"]
  motion_cmd.destination_center = (0.0, -5.0, 0.11)
  motion_cmd.destination_length = 2.5
  motion_cmd.destination_width = 0.0

  # -- Rewards: tighten direction, boost speed, add goal-plane terms ---------

  cfg.rewards["ball_vel_align"] = replace(
    cfg.rewards["ball_vel_align"],
    params={**cfg.rewards["ball_vel_align"].params, "std": 0.3},
  )
  cfg.rewards["ball_speed"] = replace(
    cfg.rewards["ball_speed"],
    weight=15.0,
    params={**cfg.rewards["ball_speed"].params, "std": 3.0},
  )
  cfg.rewards["z_speed"] = replace(
    cfg.rewards["z_speed"],
    weight=-2.0,
  )

  cfg.rewards["goal_accuracy"] = RewardTermCfg(
    func=goal_plane_accuracy,
    weight=10.0,
    params={"command_name": "motion", "std": 0.3},
  )
  cfg.rewards["goal_miss"] = RewardTermCfg(
    func=goal_miss_penalty,
    weight=-5.0,
    params={"command_name": "motion"},
  )

  # -- Motion tracking: reduce weights to 0.5 (anchor_pos stays 0.0) ---------
  for key in (
    "track_anchor_ori", "track_body_pos", "track_body_ori",
    "track_body_lin_vel", "track_body_ang_vel", "track_foot_pos",
  ):
    cfg.rewards[key] = replace(cfg.rewards[key], weight=0.5)

  # -- Terminations: keep only time_out + fell_over --------------------------
  for key in ("anchor_pos_z", "anchor_ori", "ee_body_pos"):
    cfg.terminations.pop(key, None)

  return cfg
