"""Stage V training config: compete-aligned coordinates + quality control.

Builds on Stage IV but replaces the command with
``Stage5CompeteSoccerCommandCfg`` so all motion positions, velocities,
and orientations are in compete frame.  No Stage 1-4 behaviour is affected
because Stage 3/4 continue to use ``Stage3SoccerCommandCfg``.

Key changes from Stage IV:

- Command: ``Stage5CompeteSoccerCommandCfg`` (compete coordinates, zero perturbation)
- ``destination_center = (-0.5, 0, 0.10)``, lateral axis = y
- ``fixed_ball_pos = (3.0, 0, 0.10)`` (compete ball position)
- ``goal_axis="x"``, ``goal_coord=-0.5``, ``lateral_axis="y"`` for all goal rewards
- ``goal_accuracy`` weight = 30.0
- ``goal_miss`` → ``goal_miss_scaled`` (weight -50.0)
- ``ball_vel_align`` std tightened to 0.12 (higher direction precision)
- ``z_speed`` penalty (weight -2.0, mild anti-lob)
- ``both_feet_air_time`` penalty (weight -5.0)
- ``ball_out_of_bounds`` penalty (weight -30.0)
- ``proximity`` weight = 3.0 (ball pursuit), tracking weights = 0.05 each (total 0.3)
- Zero observation noise + no corruption + no push_robot (compete-aligned)
- ``ball_speed`` unchanged (weight=20.0, std=5.0, XY-only)
"""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from src.tasks.soccer.config.training.stage4_env_cfg import make_stage4_env_cfg
from src.tasks.soccer.mdp.shooter_commands import Stage5CompeteSoccerCommandCfg
from src.tasks.soccer.mdp.shooter_rewards import (
    ball_out_of_bounds,
    both_feet_air_time,
    goal_miss_scaled,
)

# -- Stage 5 compete-aligned constants -----------------------------------------
STAGE5_BALL_POS_LOCAL = (3.0, 0.0, 0.10)
STAGE5_DESTINATION_CENTER = (-0.5, 0.0, 0.10)
STAGE5_GOAL_AXIS = "x"
STAGE5_GOAL_COORD = -0.5
STAGE5_LATERAL_AXIS = "y"
STAGE5_GOAL_HALF_WIDTH = 1.5
STAGE5_GOAL_HEIGHT = 1.8
STAGE5_COMPETE_YAW = -math.pi / 2       # -y facing → -x facing
STAGE5_COMPETE_ORIGIN = (4.0, 0.1, 0.0)
STAGE5_GOAL_POS_LOCAL = (-0.5, 0.0, 0.0)


def make_stage5_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create Stage V config with compete-aligned coordinates."""

    cfg = make_stage4_env_cfg()

    # -- Command: compete-aligned coord system, lateral axis = y -----------------

    base_cmd = cfg.commands["motion"]
    cfg.commands["motion"] = Stage5CompeteSoccerCommandCfg(
        motion_dir="",
        motion_glob="soccer-standard-*.npz",
        anchor_body_name=base_cmd.anchor_body_name,
        body_names=base_cmd.body_names,
        entity_name="robot",
        ball_entity_name="ball",
        resampling_time_range=(1e9, 1e9),
        debug_vis=False,
        pose_range={},
        velocity_range={},
        joint_position_range=(0.0, 0.0),
        sampling_mode="uniform",
        curve_offset_range=base_cmd.curve_offset_range,
        # Compete-specific
        destination_center=STAGE5_DESTINATION_CENTER,
        destination_length=3.0,
        destination_width=0.0,
        destination_lateral_axis="y",
        enable_soccer_ball_init_vel=False,
        fixed_ball_pos=STAGE5_BALL_POS_LOCAL,
        compete_yaw_offset=STAGE5_COMPETE_YAW,
        compete_origin_offset=STAGE5_COMPETE_ORIGIN,
        # target_bins=11, target_alpha=0.3 inherited from Stage3SoccerCommandCfg
    )

    # -- Rewards — compete goal plane geometry -----------------------------------

    goal_params = {
        "goal_axis": STAGE5_GOAL_AXIS,
        "goal_coord": STAGE5_GOAL_COORD,
        "lateral_axis": STAGE5_LATERAL_AXIS,
        "goal_half_width": STAGE5_GOAL_HALF_WIDTH,
        "goal_height": STAGE5_GOAL_HEIGHT,
    }

    cfg.rewards["goal_accuracy"] = replace(
        cfg.rewards["goal_accuracy"],
        weight=30.0,
        params={**cfg.rewards["goal_accuracy"].params, **goal_params},
    )

    cfg.rewards["goal_miss"] = RewardTermCfg(
        func=goal_miss_scaled,
        weight=-50.0,
        params={"command_name": "motion", **goal_params},
    )

    cfg.rewards["ball_oob"] = RewardTermCfg(
        func=ball_out_of_bounds,
        weight=-30.0,
        params={"command_name": "motion", "margin": 1.0, **goal_params},
    )

    cfg.rewards["ball_vel_align"] = replace(
        cfg.rewards["ball_vel_align"],
        params={**cfg.rewards["ball_vel_align"].params, "std": 0.12},
    )

    cfg.rewards["both_feet_air_time"] = RewardTermCfg(
        func=both_feet_air_time,
        weight=-5.0,
        params={"threshold": 0.15},
    )

    # ball_speed — unchanged (weight=20.0, std=5.0, already XY-only)
    cfg.rewards["z_speed"] = replace(
        cfg.rewards["z_speed"],
        weight=-2.0,
    )

    # -- Ball pursuit vs motion tracking balance (compete-aligned) --------------

    cfg.rewards["proximity"] = replace(
        cfg.rewards["proximity"],
        weight=3.0,
    )

    for key in (
        "track_anchor_ori", "track_body_pos", "track_body_ori",
        "track_body_lin_vel", "track_body_ang_vel", "track_foot_pos",
    ):
        cfg.rewards[key] = replace(cfg.rewards[key], weight=0.05)
    # track_anchor_pos remains 0.0 (disabled)

    # -- Compete-aligned: zero obs noise, no corruption, no push_robot ---------

    for term in cfg.observations["actor"].terms.values():
        term.noise = None
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    return cfg