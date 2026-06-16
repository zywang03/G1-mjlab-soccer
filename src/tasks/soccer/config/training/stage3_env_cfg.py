"""Stage III training config: goal-plane accuracy + speed curriculum on flat ground.

Builds on Stage II by relaxing motion tracking constraints and adding
goal-plane crossing rewards (accuracy + miss penalty).  Stage II is
NOT modified — this factory creates a new config object from
``make_stage2_env_cfg()`` and only mutates its own copy.

Key changes from Stage II:
- Goal-plane target: x in [-1.1, 1.1], y fixed at -5.0 (adaptive bin sampling)
- ``ball_vel_align`` sigma tightened to 0.3
- ``ball_speed`` sigma 3.0, weight 15.0 (curriculum overrides via CLI)
- ``z_speed`` penalty disabled (weight 0.0, same as Stage II)
- New ``goal_accuracy`` (w=10) + ``goal_miss`` (w=-5)
- Motion tracking weights reduced to 0.5 (anchor_pos stays 0.0)
- Motion-reference terminations removed (only time_out + fell_over remain)
- Uses ``Stage3SoccerCommandCfg`` for adaptive target point sampling
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from src.tasks.soccer.config.training.stage2_env_cfg import make_stage2_env_cfg
from src.tasks.soccer.mdp.shooter_commands import Stage3SoccerCommandCfg
from src.tasks.soccer.mdp.shooter_rewards import goal_plane_accuracy, goal_miss_penalty


def make_stage3_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Stage III (goal-plane accuracy + speed) environment config."""

  cfg = make_stage2_env_cfg()

  # -- Command: use Stage 3 command with adaptive target sampling -----------
  base_cmd = cfg.commands["motion"]
  cfg.commands["motion"] = Stage3SoccerCommandCfg(
    motion_dir="",
    motion_glob="soccer-standard-*.npz",
    anchor_body_name=base_cmd.anchor_body_name,
    body_names=base_cmd.body_names,
    entity_name="robot",
    ball_entity_name="ball",
    resampling_time_range=(1e9, 1e9),
    debug_vis=False,
    pose_range=base_cmd.pose_range,
    velocity_range=base_cmd.velocity_range,
    joint_position_range=base_cmd.joint_position_range,
    sampling_mode="uniform",
    curve_offset_range=base_cmd.curve_offset_range,
    destination_center=(0.0, -5.0, 0.11),
    destination_length=2.2,
    destination_width=0.0,
    enable_soccer_ball_init_vel=False,
    adaptive_target=True,
    target_bins=11,
    target_alpha=0.3,
  )

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
    weight=0.0,
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
