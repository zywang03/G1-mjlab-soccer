"""Motion-free shooter student PPO config."""

from __future__ import annotations

import math

from mjlab.envs.mdp.rewards import joint_pos_limits
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from src.tasks.soccer import mdp
from src.tasks.soccer.config.training.stage1_env_cfg import make_stage1_env_cfg
from src.tasks.soccer.mdp.shooter_rewards import (
  action_rate_l2_clip,
  foot_distance,
  undesired_contacts,
  waist_action_rate_l2_clip,
)
from src.tasks.soccer.mdp.shared_rewards import is_terminated
from src.tasks.soccer.mdp.student_shooter_commands import StudentShooterCommandCfg
from src.tasks.soccer.mdp.student_shooter_obs import student_shooter_obs
from src.tasks.soccer.mdp.student_shooter_rewards import (
  student_ball_speed_capped,
  student_ball_velocity_direction_alignment,
  student_both_feet_ball_contact,
  student_foot_stomp_penalty,
  student_nonfoot_ball_contact,
  student_pelvis_orientation,
  student_target_point_proximity,
  student_valid_kick_contact,
)


def make_student_shooter_env_cfg():
  """Create motion-free shooter student PPO environment config.

  The command term samples robot/ball/goal-plane target state but exposes no
  motion-reference trajectory to the policy or reward.
  """
  cfg = make_stage1_env_cfg()

  cfg.commands = {
    "motion": StudentShooterCommandCfg(
      resampling_time_range=(1e9, 1e9),
      debug_vis=False,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.05, 0.05),
        "pitch": (-0.05, 0.05),
        "yaw": (-0.15, 0.15),
      },
      velocity_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.02, 0.02),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
    ),
  }

  obs_terms = {
    "student": ObservationTermCfg(
      func=student_shooter_obs,
      params={"command_name": "motion"},
    ),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=obs_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=obs_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
  waist_joints = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
  cfg.rewards = {
    "proximity": RewardTermCfg(
      func=student_target_point_proximity,
      weight=1.0,
      params={"std": 2.0, "command_name": "motion"},
    ),
    "contact": RewardTermCfg(
      func=student_valid_kick_contact,
      weight=50.0,
      params={"command_name": "motion", "ball_sensor_name": "ball_robot_contact"},
    ),
    "ball_vel_align": RewardTermCfg(
      func=student_ball_velocity_direction_alignment,
      weight=30.0,
      params={"command_name": "motion", "std": 0.8, "velocity_threshold": 0.5},
    ),
    "ball_speed_capped": RewardTermCfg(
      func=student_ball_speed_capped,
      weight=2.0,
      params={"command_name": "motion", "min_speed": 1.0, "cap_speed": 2.5},
    ),
    "both_feet_ball": RewardTermCfg(
      func=student_both_feet_ball_contact,
      weight=-5.0,
      params={"command_name": "motion", "ball_sensor_name": "ball_robot_contact"},
    ),
    "nonfoot_ball": RewardTermCfg(
      func=student_nonfoot_ball_contact,
      weight=-3.0,
      params={"command_name": "motion", "ball_sensor_name": "ball_robot_contact"},
    ),
    "foot_stomp": RewardTermCfg(
      func=student_foot_stomp_penalty,
      weight=-20.0,
      params={"command_name": "motion", "ball_sensor_name": "ball_robot_contact", "foot_body_names": foot_names},
    ),
    "action_rate": RewardTermCfg(func=action_rate_l2_clip, weight=-0.1),
    "joint_limit": RewardTermCfg(
      func=joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "undesired_contacts": RewardTermCfg(
      func=undesired_contacts,
      weight=-0.1,
      params={
        "threshold": 1.0,
        "entity_name": "robot",
        "excluded_body_names": (
          "left_ankle_roll_link", "right_ankle_roll_link",
          "left_wrist_yaw_link", "right_wrist_yaw_link",
        ),
      },
    ),
    "foot_distance": RewardTermCfg(
      func=foot_distance,
      weight=0.2,
      params={"threshold": 0.24, "std": 0.5, "entity_name": "robot", "body_names": foot_names},
    ),
    "pelvis_orientation": RewardTermCfg(
      func=student_pelvis_orientation,
      weight=-1.0,
      params={"command_name": "motion"},
    ),
    "waist_action_rate": RewardTermCfg(
      func=waist_action_rate_l2_clip,
      weight=-0.25,
      params={"entity_name": "robot", "joint_names": waist_joints},
    ),
    "is_terminated": RewardTermCfg(func=is_terminated, weight=-200.0),
  }

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }
  cfg.events = {}
  cfg.episode_length_s = 4.0
  return cfg
