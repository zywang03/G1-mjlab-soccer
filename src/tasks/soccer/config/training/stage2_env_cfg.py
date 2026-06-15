"""Stage II training config: perception-guided kicking on flat ground.

Builds on Stage I by overriding the motion command (uniform sampling,
ball position randomization) and adjusting reward weights to match
the original HumanoidSoccer code.  Both stages share the same 160D
observation space (soc terms are already in Stage I).
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.tasks.soccer.config.training.stage1_env_cfg import make_stage1_env_cfg, TRACKING_BODY_NAMES

# Original code Stage II excludes ankles from body tracking; feet are tracked
# separately via track_foot_pos (matching G1FlatProximityEnvCfg).
TRACKING_BODY_NAMES_NO_ANKLES = tuple(
    name for name in TRACKING_BODY_NAMES
    if name not in ("left_ankle_roll_link", "right_ankle_roll_link")
)

from src.tasks.soccer.mdp.shooter_rewards import (
  action_rate_l2_clip,
  ball_speed_reward,
  ball_velocity_direction_alignment,
  ball_z_speed_penalty,
  both_feet_ball_contact,
  foot_distance,
  foot_lift_penalty,
  foot_stomp_penalty,
  nonfoot_ball_contact,
  pelvis_orientation,
  sideways_kick,
  target_point_contact,
  target_point_proximity,
  waist_action_rate_l2_clip,
)

from src.tasks.soccer.mdp.shared_rewards import is_terminated

from src.tasks.soccer.mdp.shooter_commands import CurveOffsetCfg, MultiMotionSoccerCommandCfg


def make_stage2_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create Stage II (perception-guided kicking) soccer environment config.

  Key changes from Stage I:
  - Uniform motion sampling (always starts from frame 0)
  - Ball position randomization (arc offset from motion trajectory)
  - Soccer observations: ball + goal in robot pelvis frame
  - Anchor position tracking weight → 0.0 (allows ball pursuit)
  - Motion tracking weights reduced per original HumanoidSoccer code:
    anchor_ori stays at 1.0; body-pos/ori track 12 bodies (no ankles)
    at 1.0; lin-vel/ang-vel tracking unchanged from Stage I (1.0).
  - Feet tracked separately via track_foot_pos (weight=1.0).
  - Soccer kick rewards: proximity, contact, sideways kick, ball vel/speed
  - Ball initial velocity disabled (stationary penalty kick)
  """

  cfg = make_stage1_env_cfg()

  # -- Commands: override to uniform sampling + ball randomization -----------

  base_cmd = cfg.commands["motion"]
  cfg.commands["motion"] = MultiMotionSoccerCommandCfg(
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
    curve_offset_range=CurveOffsetCfg(
      radius=(-0.15, 0.15),
      arc_angle=math.pi / 18,
      height=0.11,
    ),
    destination_center=(0.0, -5.0, 0.11),
    destination_length=1.0,
    destination_width=0.5,
    enable_soccer_ball_init_vel=False,
  )

  # -- Terminations: relax ee_body_pos for Stage II ----------------------------
  # Stage II kicking requires larger foot displacement than Stage I tracking.
  # Relax from 0.25m to 0.35m so the policy can explore kicking without
  # prematurely terminating.
  cfg.terminations["ee_body_pos"] = TerminationTermCfg(
    func=cfg.terminations["ee_body_pos"].func,
    params={
      "command_name": "motion",
      "threshold": 0.35,
      "body_names": (
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_wrist_yaw_link", "right_wrist_yaw_link",
      ),
    },
  )

  # -- Rewards: match original HumanoidSoccer code (Stage II weights) --------

  # Disable global anchor position tracking (allows robot to pursue ball).
  cfg.rewards["track_anchor_pos"] = RewardTermCfg(
    func=cfg.rewards["track_anchor_pos"].func,
    weight=0.0,
    params={"command_name": "motion", "std": 0.3},
  )
  # Anchor orientation tracking kept at 1.0 (original code).
  cfg.rewards["track_anchor_ori"] = RewardTermCfg(
    func=cfg.rewards["track_anchor_ori"].func,
    weight=1.0,
    params={"command_name": "motion", "std": 0.4},
  )
  # Body tracking on 12 bodies (excluding ankles, which are tracked
  # separately via track_foot_pos).  Weight stays at 1.0.
  cfg.rewards["track_body_pos"] = RewardTermCfg(
    func=cfg.rewards["track_body_pos"].func,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 0.3,
      "body_names": TRACKING_BODY_NAMES_NO_ANKLES,
    },
  )
  cfg.rewards["track_body_ori"] = RewardTermCfg(
    func=cfg.rewards["track_body_ori"].func,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 0.4,
      "body_names": TRACKING_BODY_NAMES_NO_ANKLES,
    },
  )
  # lin-vel and ang-vel tracking: unchanged from Stage I (1.0), no override needed.

  # Soccer rewards.
  _foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
  _waist_joints = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")

  cfg.rewards["proximity"] = RewardTermCfg(
    func=target_point_proximity,
    weight=1.0,
    params={"std": 2.0, "command_name": "motion"},
  )
  cfg.rewards["contact"] = RewardTermCfg(
    func=target_point_contact,
    weight=50.0,
    params={
      "command_name": "motion",
      "ball_sensor_name": "ball_robot_contact",
      "horizontal_force_threshold": 0.0,
      "foot_body_names": _foot_names,
    },
  )
  cfg.rewards["sideways_kick"] = RewardTermCfg(
    func=sideways_kick,
    weight=50.0,
    params={
      "command_name": "motion",
      "ball_sensor_name": "ball_robot_contact",
      "horizontal_force_threshold": 0.0,
      "foot_body_names": _foot_names,
    },
  )
  cfg.rewards["ball_vel_align"] = RewardTermCfg(
    func=ball_velocity_direction_alignment,
    weight=30.0,
    params={
      "command_name": "motion",
      "std": 0.8,
      "velocity_threshold": 0.5,
      "ball_sensor_name": "ball_robot_contact",
      "horizontal_force_threshold": 0.0,
      "foot_body_names": _foot_names,
    },
  )
  cfg.rewards["ball_speed"] = RewardTermCfg(
    func=ball_speed_reward,
    weight=10.0,
    params={
      "command_name": "motion",
      "std": 1.2,
      "velocity_threshold": 0.5,
      "ball_sensor_name": "ball_robot_contact",
      "horizontal_force_threshold": 0.0,
      "foot_body_names": _foot_names,
    },
  )
  # z-speed penalty: original code has weight=0.0 (disabled).
  cfg.rewards["z_speed"] = RewardTermCfg(
    func=ball_z_speed_penalty,
    weight=0.0,
    params={
      "command_name": "motion",
      "ball_sensor_name": "ball_robot_contact",
      "horizontal_force_threshold": 10.0,
      "foot_body_names": _foot_names,
    },
  )

  # Stabilization rewards.
  cfg.rewards["foot_distance"] = RewardTermCfg(
    func=foot_distance,
    weight=0.2,
    params={
      "threshold": 0.24,
      "std": 0.5,
      "entity_name": "robot",
      "body_names": _foot_names,
    },
  )
  cfg.rewards["pelvis_orientation"] = RewardTermCfg(
    func=pelvis_orientation,
    weight=-1.0,
    params={"command_name": "motion"},
  )
  cfg.rewards["waist_action_rate"] = RewardTermCfg(
    func=waist_action_rate_l2_clip,
    weight=-0.25,
    params={
      "entity_name": "robot",
      "joint_names": _waist_joints,
    },
  )

  # Foot position tracking for the two ankle bodies, which are excluded
  # from track_body_pos/track_body_ori above (matching original code).
  cfg.rewards["track_foot_pos"] = RewardTermCfg(
    func=cfg.rewards["track_body_pos"].func,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 0.3,
      "body_names": _foot_names,
    },
  )

  # -- Anti-clamp penalties and termination penalty (Stage 2.5) ---------------

  # Termination penalty: aligns with compete.py (-200).
  cfg.rewards["is_terminated"] = RewardTermCfg(
    func=is_terminated,
    weight=-200.0,
  )

  # Both feet simultaneously contacting ball (prevents clamping).
  cfg.rewards["both_feet_ball"] = RewardTermCfg(
    func=both_feet_ball_contact,
    weight=-5.0,
    params={
      "command_name": "motion",
      "ball_sensor_name": "ball_robot_contact",
      "horizontal_force_threshold": 0.0,
    },
  )

  # Non-foot body contacting ball (prevents body-bumping).
  cfg.rewards["nonfoot_ball"] = RewardTermCfg(
    func=nonfoot_ball_contact,
    weight=-3.0,
    params={
      "command_name": "motion",
      "ball_sensor_name": "ball_robot_contact",
      "horizontal_force_threshold": 0.0,
    },
  )

  # Downward foot velocity at kick moment (prevents stomping on ball).
  # Only penalizes when foot is above ball AND vertical velocity dominates.
  cfg.rewards["foot_stomp"] = RewardTermCfg(
    func=foot_stomp_penalty,
    weight=-20.0,
    params={
      "command_name": "motion",
      "ball_sensor_name": "ball_robot_contact",
      "horizontal_force_threshold": 0.0,
      "foot_body_names": _foot_names,
      "foot_above_ball_margin": 0.02,
      "vz_ratio_threshold": 0.5,
    },
  )

  # Penalize kicking foot being too high before valid kick (prevents stomp setup).
  cfg.rewards["foot_lift"] = RewardTermCfg(
    func=foot_lift_penalty,
    weight=-2.0,
    params={
      "command_name": "motion",
      "ball_sensor_name": "ball_robot_contact",
      "horizontal_force_threshold": 0.0,
      "foot_body_names": _foot_names,
      "foot_above_threshold": 0.15,
    },
  )

  return cfg
