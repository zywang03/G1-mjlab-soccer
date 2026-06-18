"""Stage VI training config: high-speed compete-aligned kicking.

Inherits all Stage V settings (compete coordinates, zero perturbation,
zero obs noise, no corruption, no push_robot).  Only adjusts four
rewards to push kick speed from ~8 m/s toward 10-12 m/s.

Key changes from Stage V:

- ``ball_speed`` weight = 40.0, std = 10.0
- ``z_speed`` weight = -5.0 (stronger anti-lob at high speed)
- ``ball_vel_align`` weight = 20.0, std = 0.20 (relaxed direction)
- ``goal_accuracy`` weight = 20.0 (accuracy preserved, speed gets priority)
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.soccer.config.training.stage5_env_cfg import make_stage5_env_cfg


def make_stage6_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create Stage VI config — Stage V base + speed-oriented reward tuning."""

    cfg = make_stage5_env_cfg()

    cfg.rewards["ball_speed"] = replace(
        cfg.rewards["ball_speed"],
        weight=40.0,
        params={**cfg.rewards["ball_speed"].params, "std": 10.0},
    )

    cfg.rewards["z_speed"] = replace(
        cfg.rewards["z_speed"],
        weight=-5.0,
    )

    cfg.rewards["ball_vel_align"] = replace(
        cfg.rewards["ball_vel_align"],
        weight=20.0,
        params={**cfg.rewards["ball_vel_align"].params, "std": 0.20},
    )

    cfg.rewards["goal_accuracy"] = replace(
        cfg.rewards["goal_accuracy"],
        weight=20.0,
    )

    return cfg
