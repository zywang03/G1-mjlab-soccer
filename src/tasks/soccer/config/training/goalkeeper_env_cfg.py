"""Goalkeeper training config: single-task curriculum for ball interception.

Builds a task-reward goalkeeper environment matching Humanoid-Goalkeeper:
- 96D actor obs (ball_pos + ang_vel + gravity + joint_pos + joint_vel + actions)
  with 10-frame history stacking (960D input to MLP)
- 113D critic obs (actor terms + lin_vel + region + target + ball_vel + hands + dist)
- Official 6-region shots from the start with range/weight curriculum
- Ball launching via parabolic trajectory with staged landing regions
- Position-conditioned task rewards, AMP motion prior, and recovery stability
- Domain randomization: robot push + ball velocity perturbation
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.rewards import joint_pos_limits
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.tasks.soccer import mdp
from src.tasks.soccer.ball import get_ball_cfg
from src.tasks.soccer.goal import get_goal_cfg
from src.tasks.soccer.ground import get_ground_cfg
from src.tasks.soccer.config.soccer_settings import SETTINGS
from src.tasks.soccer.soccer_env_cfg import _add_soccer_scene_postproc

from src.tasks.soccer.mdp.goalkeeper_rewards import (
  _reset_gk_state,
  goalkeeper_ee_reach,
  goalkeeper_success,
  goalkeeper_stop_ball,
  goalkeeper_stay_on_line,
  goalkeeper_no_retreat,
  goalkeeper_feet_slippage,
  goalkeeper_feet_orientation,
  goalkeeper_success_land,
  goalkeeper_penalize_high_ball_foot_block,
  goalkeeper_penalize_sharp_contact,
  goalkeeper_penalize_knee_height,
  goalkeeper_post_orientation,
  goalkeeper_post_ang_vel,
  goalkeeper_post_lin_vel,
  goalkeeper_post_upper_dof_pos,
  goalkeeper_post_waist_dof_pos,
  goalkeeper_dof_acc,
  goalkeeper_torques,
  goalkeeper_dof_vel,
  goalkeeper_dof_vel_limits,
  goalkeeper_torque_limits,
  goalkeeper_deviation_waist_pitch_joint,
  goalkeeper_action_smoothness,
  goalkeeper_ang_vel_xy,
)
from src.tasks.soccer.mdp.goalkeeper_obs import (
  goalkeeper_ball_distance,
  goalkeeper_ee_positions,
  goalkeeper_end_region,
  goalkeeper_end_target_pos,
  gk_ang_vel,
  gk_ball_vel_local,
  gk_ball_pos_local,
  gk_joint_pos_rel,
  gk_joint_vel_rel,
  gk_last_action,
  gk_lin_vel,
)
from src.tasks.soccer.mdp.goalkeeper_ball_reset import RegionBallVelCfg, reset_ball_with_parabolic_trajectory
from src.tasks.soccer.mdp.goalkeeper_curriculum import goalkeeper_curriculum
from src.tasks.soccer.mdp.goalkeeper_actions import GoalkeeperJointPositionActionCfg
from src.tasks.soccer.mdp.goalkeeper_reset import (
  decrement_goalkeeper_catchstep,
  reset_goalkeeper_joints_official,
  reset_goalkeeper_root_official,
)


_PHYSICS_TIMESTEP = 0.005
_CONTROL_DECIMATION = 4
_CONTROL_DT = _PHYSICS_TIMESTEP * _CONTROL_DECIMATION


def _one_shot_reward_weight(episode_scale: float) -> float:
  """Convert one-shot reward scale to raw weight under dt-scaled rewards."""
  return episode_scale / _CONTROL_DT


def _region(
  height: tuple[float, float],
  width: tuple[float, float],
  motion_id: int,
) -> dict[str, tuple[float, float] | int]:
  return {"height": height, "width": width, "motion_id": motion_id}


def _goalkeeper_curriculum_stages(
  full_regions: list[dict[str, list[float]]],
  full_start_y_range: tuple[float, float],
  full_start_z_range: tuple[float, float],
) -> list[dict]:
  """Return official-style curriculum: all six regions from the start."""
  del full_start_y_range, full_start_z_range

  def official_weights(scale: float = 0.0) -> dict[str, float]:
    return {
      "ee_reach": 10.0 * (1.0 + 0.5 * scale),
      "success": 5.0 * (1.0 + 0.5 * scale),
      "stop_ball": 60.0 + 40.0 * scale,
      "high_ball_foot_block": -80.0,
      "stay_on_line": -2.0,
      "no_retreat": -2.0,
      "success_land": 4.0,
      "feet_orientation": 3.0,
      "penalize_sharp_contact": -100.0,
      "penalize_knee_height": -100.0,
      "feet_slippage": 3.0,
      "post_orientation": 3.0,
      "post_ang_vel": 3.0,
      "post_upper_dof_pos": 1.0,
      "post_waist_dof_pos": 1.0,
      "post_lin_vel": 1.0,
      "ang_vel_xy": -0.1,
      "dof_acc": -2.5e-7,
      "action_smoothness": -0.1,
      "torques": -1.0e-5,
      "dof_vel": -5.0e-4,
      "joint_limit": -3.0 * max(1.0, 2.0 * scale),
      "dof_vel_limits": -2.0,
      "torque_limits": -3.0 * max(1.0, 3.0 * scale),
      "deviation_waist_pitch_joint": -0.001,
    }

  return [
    {
      "step": 0,
      "name": "official_six_region_initial",
      "regions": full_regions,
      "ball_start_x_range": (3.0, 5.0),
      "ball_end_x_range": (0.1, 0.6),
      "t_flight_range": (0.4, 1.0),
      "ball_start_y_range": (-1.8, 1.8),
      "ball_start_z_range": (0.1, 1.5),
      "push_vel_xy_range": (0.0, 0.0),
      "push_vel_z_range": (0.0, 0.0),
      "push_ang_vel_range": (0.0, 0.0),
      "push_interval_range_s": (15.0, 15.0),
      "perturb_vel_range": (0.0, 0.0),
      "perturb_interval_range_s": (0.5, 0.5),
      "curriculum_update": 0,
      "curriculumsigma": SETTINGS.goalkeeper_training.ee_reach_sigmoid_scale,
      "reward_weights": official_weights(0.0),
    },
    {
      "step": 50_000,
      "name": "official_six_region_expanded",
      "regions": full_regions,
      "ball_start_x_range": (3.0, 5.0),
      "ball_end_x_range": (0.1, 0.6),
      "t_flight_range": (0.4, 1.0),
      "ball_start_y_range": (-1.8, 1.8),
      "ball_start_z_range": (0.1, 1.5),
      "push_vel_xy_range": (-0.5, 0.5),
      "push_vel_z_range": (0.0, 0.0),
      "push_ang_vel_range": (0.0, 0.0),
      "push_interval_range_s": (15.0, 15.0),
      "perturb_vel_range": (-0.25, 0.25),
      "perturb_interval_range_s": (0.5, 0.5),
      "curriculum_update": 1,
      "curriculumsigma": SETTINGS.goalkeeper_training.ee_reach_sigmoid_scale,
      "reward_weights": official_weights(1.0),
    },
  ]


def make_goalkeeper_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create goalkeeper training environment config (single-stage).

  Key design:
  - Parabolic trajectory ball launch with 6 landing regions
  - 96D actor obs with 10-frame history stacking -> 960D network input
  - 113D asymmetric critic matching the official observation order
  - Position-conditioned task rewards plus AMP motion prior
  """

  _BALL_CFG = SceneEntityCfg("ball")
  _ROBOT_CFG = SceneEntityCfg("robot")

  # -- Observations -----------------------------------------------------------

  # Actor: 96D = ball_pos(3) + ang_vel(3) + gravity(3)
  #              + joint_pos(29) + joint_vel(29) + actions(29).
  actor_terms = {
    "ball_pos_local": ObservationTermCfg(
      func=gk_ball_pos_local,
      params={
        "ball_cfg": _BALL_CFG,
        "robot_cfg": _ROBOT_CFG,
        "position_noise": 0.05,
        "dropout_after_s": 0.4,
        "dropout_prob": 0.1,
        "stop_speed_threshold": 0.1,
        "use_official_visibility": True,
      },
    ),
    "base_ang_vel": ObservationTermCfg(
      func=gk_ang_vel,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=gk_joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=gk_joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=gk_last_action),
  }

  # Critic: actor(96) + lin_vel(3) + region(1) + end_target(3)
  #        + ball_vel(3) + hand_pos(6) + reach_dist(1) = 113D.
  critic_actor_terms = {
    **actor_terms,
    "ball_pos_local": ObservationTermCfg(
      func=gk_ball_pos_local,
      params={
        "ball_cfg": _BALL_CFG,
        "robot_cfg": _ROBOT_CFG,
        "position_noise": 0.0,
        "dropout_prob": 0.0,
        "stop_speed_threshold": -1.0,
        "use_official_visibility": False,
      },
    ),
  }
  critic_terms = {
    **critic_actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=gk_lin_vel,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "end_region": ObservationTermCfg(
      func=goalkeeper_end_region,
    ),
    "end_target_pos": ObservationTermCfg(
      func=goalkeeper_end_target_pos,
      params={
        "approach_distance_threshold": SETTINGS.goalkeeper_training.approach_distance_threshold,
        "ball_cfg": _BALL_CFG,
        "robot_cfg": _ROBOT_CFG,
      },
    ),
    "ball_vel_local": ObservationTermCfg(
      func=gk_ball_vel_local,
      params={"ball_cfg": _BALL_CFG, "robot_cfg": _ROBOT_CFG},
    ),
    "ee_positions": ObservationTermCfg(
      func=goalkeeper_ee_positions,
      params={"robot_cfg": _ROBOT_CFG},
    ),
    "ball_distance": ObservationTermCfg(
      func=goalkeeper_ball_distance,
      params={
        "approach_distance_threshold": SETTINGS.goalkeeper_training.approach_distance_threshold,
        "ball_cfg": _BALL_CFG,
        "robot_cfg": _ROBOT_CFG,
      },
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=10,  # Paper-style history, 96x10=960D
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  # -- Actions ----------------------------------------------------------------

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": GoalkeeperJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Matches reference PD controller action_scale
      use_default_offset=True,
    )
  }

  # -- Events -----------------------------------------------------------------

  # Build 6-region configuration from settings.
  regions_raw = SETTINGS.goalkeeper_regions
  all_widths_lo = [r.width[0] for r in regions_raw]
  all_widths_hi = [r.width[1] for r in regions_raw]
  all_heights_lo = [r.height[0] for r in regions_raw]
  all_heights_hi = [r.height[1] for r in regions_raw]

  region_dicts = [
    {"height": r.height, "width": r.width, "motion_id": index}
    for index, r in enumerate(regions_raw)
  ]

  vel_cfg = RegionBallVelCfg(
    ball_start_x_range=(3.0, 5.0),
    ball_end_x_range=(0.1, 0.6),
    t_flight_range=(0.4, 1.0),
    train_t_flight_range=(0.4, 1.0),
    play_t_flight_range=(0.5, 1.0),
    regions=region_dicts,
    ball_start_y_range=(-1.8, 1.8),
    ball_start_z_range=(0.1, 1.5),
    balanced_regions=True,
  )

  events = {
    "decrement_catchstep": EventTermCfg(
      func=decrement_goalkeeper_catchstep,
      mode="step",
      params={},
    ),
    # Ball launch with 6-region parabolic trajectory.
    "reset_ball": EventTermCfg(
      func=reset_ball_with_parabolic_trajectory,
      mode="reset",
      params={
        "vel_cfg": vel_cfg,
        "ball_cfg": _BALL_CFG,
      },
    ),
    # DR: random push to robot base.
    "push_robot": EventTermCfg(
      func=mdp.push_robot_base,
      mode="interval",
      interval_range_s=(15.0, 15.0),
      params={
        "vel_xy_range": (0.0, 0.0),
        "vel_z_range": (0.0, 0.0),
        "ang_vel_range": (0.0, 0.0),
      },
    ),
    # DR: ball velocity perturbation.
    "perturb_ball_vel": EventTermCfg(
      func=mdp.perturb_ball_velocity,
      mode="interval",
      interval_range_s=(0.5, 0.5),
      params={
        "vel_range": (0.0, 0.0),
        "ball_cfg": _BALL_CFG,
      },
    ),
    "reset_robot_base": EventTermCfg(
      func=reset_goalkeeper_root_official,
      mode="reset",
      params={
        "pose_z_range": (-0.01, 0.01),
        "velocity_range": (-0.3, 0.3),
        "asset_cfg": _ROBOT_CFG,
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=reset_goalkeeper_joints_official,
      mode="reset",
      params={
        "scale_range": (0.5, 1.5),
        "offset_range": (-0.1, 0.1),
        "continue_keep": True,
        "continue_keep_prob": 0.8,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Clear per-env GK reward tracking state.
    "reset_gk_state": EventTermCfg(
      func=_reset_gk_state,
      mode="reset",
      params={},
    ),
  }

  # -- Rewards ----------------------------------------------------------------

  rewards = {
    "ee_reach": RewardTermCfg(
      func=goalkeeper_ee_reach,
      weight=10.0,
      params={
        "std": SETTINGS.goalkeeper_training.ee_reach_std,
        "catch_distance": SETTINGS.goalkeeper_training.ee_reach_catch_distance,
        "sigmoid_scale": SETTINGS.goalkeeper_training.ee_reach_sigmoid_scale,
        "approach_distance_threshold": SETTINGS.goalkeeper_training.approach_distance_threshold,
        "ee_body_names": (
          "left_wrist_yaw_link",
          "right_wrist_yaw_link",
        ),
        "high_ball_region_ids": (2, 3),
        "high_ball_target_z_threshold": 1.2,
        "high_ball_reward_scale": 1.5,
      },
    ),
    "success": RewardTermCfg(
      func=goalkeeper_success,
      weight=5.0,
      params={
        "strict_distance": 0.15,
        "approach_distance_threshold": SETTINGS.goalkeeper_training.approach_distance_threshold,
        "high_ball_region_ids": (2, 3),
        "high_ball_target_z_threshold": 1.2,
        "high_ball_success_scale": 2.0,
      },
    ),
    "stop_ball": RewardTermCfg(
      func=goalkeeper_stop_ball,
      weight=60.0,
      params={
        "velocity_drop_threshold": SETTINGS.goalkeeper_training.stop_ball_vel_drop,
        "behind_robot_x_threshold": SETTINGS.goalkeeper_training.behind_robot_x,
        "ee_body_names": (
          "left_wrist_yaw_link",
          "right_wrist_yaw_link",
        ),
        "high_ball_region_ids": (2, 3),
        "high_ball_target_z_threshold": 1.2,
        "high_ball_hand_gate_radius": 0.35,
      },
    ),
    "high_ball_foot_block": RewardTermCfg(
      func=goalkeeper_penalize_high_ball_foot_block,
      weight=-80.0,
    ),
    "stay_on_line": RewardTermCfg(func=goalkeeper_stay_on_line, weight=-2.0),
    "no_retreat": RewardTermCfg(func=goalkeeper_no_retreat, weight=-2.0),
    "success_land": RewardTermCfg(func=goalkeeper_success_land, weight=4.0),
    "feet_orientation": RewardTermCfg(func=goalkeeper_feet_orientation, weight=3.0),
    "penalize_sharp_contact": RewardTermCfg(
      func=goalkeeper_penalize_sharp_contact,
      weight=-100.0,
      params={
        "max_contact_force": 100.0,
      },
    ),
    "penalize_knee_height": RewardTermCfg(
      func=goalkeeper_penalize_knee_height,
      weight=-100.0,
      params={
        "min_height": 0.15,
      },
    ),
    "feet_slippage": RewardTermCfg(func=goalkeeper_feet_slippage, weight=3.0),
    "post_orientation": RewardTermCfg(
      func=goalkeeper_post_orientation,
      weight=3.0,
      params={
        "goal_x": SETTINGS.goalkeeper_training.behind_robot_x,
        "velocity_drop_threshold": SETTINGS.goalkeeper_training.stop_ball_vel_drop,
      },
    ),
    "post_ang_vel": RewardTermCfg(
      func=goalkeeper_post_ang_vel,
      weight=3.0,
      params={
        "goal_x": SETTINGS.goalkeeper_training.behind_robot_x,
        "velocity_drop_threshold": SETTINGS.goalkeeper_training.stop_ball_vel_drop,
      },
    ),
    "post_lin_vel": RewardTermCfg(
      func=goalkeeper_post_lin_vel,
      weight=1.0,
      params={
        "goal_x": SETTINGS.goalkeeper_training.behind_robot_x,
        "velocity_drop_threshold": SETTINGS.goalkeeper_training.stop_ball_vel_drop,
      },
    ),
    "post_upper_dof_pos": RewardTermCfg(
      func=goalkeeper_post_upper_dof_pos,
      weight=1.0,
      params={
        "goal_x": SETTINGS.goalkeeper_training.behind_robot_x,
        "velocity_drop_threshold": SETTINGS.goalkeeper_training.stop_ball_vel_drop,
      },
    ),
    "post_waist_dof_pos": RewardTermCfg(
      func=goalkeeper_post_waist_dof_pos,
      weight=1.0,
      params={
        "goal_x": SETTINGS.goalkeeper_training.behind_robot_x,
        "velocity_drop_threshold": SETTINGS.goalkeeper_training.stop_ball_vel_drop,
      },
    ),
    "ang_vel_xy": RewardTermCfg(func=goalkeeper_ang_vel_xy, weight=-0.1),
    "dof_acc": RewardTermCfg(
      func=goalkeeper_dof_acc,
      weight=-2.5e-7,
      params={"robot_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "action_smoothness": RewardTermCfg(func=goalkeeper_action_smoothness, weight=-0.1),
    "torques": RewardTermCfg(
      func=goalkeeper_torques,
      weight=-1.0e-5,
      params={"robot_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "dof_vel": RewardTermCfg(
      func=goalkeeper_dof_vel,
      weight=-5.0e-4,
      params={"robot_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_limit": RewardTermCfg(
      func=joint_pos_limits,
      weight=-3.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "dof_vel_limits": RewardTermCfg(
      func=goalkeeper_dof_vel_limits,
      weight=-2.0,
      params={"robot_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "torque_limits": RewardTermCfg(
      func=goalkeeper_torque_limits,
      weight=-3.0,
      params={"robot_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "deviation_waist_pitch_joint": RewardTermCfg(
      func=goalkeeper_deviation_waist_pitch_joint,
      weight=-0.001,
    ),
  }

  # -- Curriculum -------------------------------------------------------------

  curriculum = {
    "goalkeeper_difficulty": CurriculumTermCfg(
      func=goalkeeper_curriculum,
      params={
        "stages": _goalkeeper_curriculum_stages(
          region_dicts,
          (min(all_widths_lo), max(all_widths_hi)),
          (min(all_heights_lo), max(all_heights_hi)),
        ),
      },
    ),
  }

  # -- Terminations -----------------------------------------------------------

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  # -- Assemble ---------------------------------------------------------------

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      entities={
        "ground": get_ground_cfg(),
        "ball": get_ball_cfg(),
        # Goal at -x behind G1, default orientation (posts y, opening ±x).
        "goal": get_goal_cfg(pos=(-0.5, 0.0, 0.0)),
      },
      num_envs=64,
      env_spacing=2.5,
      spec_fn=_add_soccer_scene_postproc,
    ),
    observations=observations,
    actions=actions,
    commands={},  # No motion command — single-stage reactive policy
    events=events,
    rewards=rewards,
    curriculum=curriculum,
    terminations=terminations,
    scale_rewards_by_dt=True,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=6.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=128,  # Increased for goalkeeper self-contact (broadphase overflow fix)
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=_PHYSICS_TIMESTEP,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=_CONTROL_DECIMATION,
    episode_length_s=SETTINGS.goalkeeper_episode_length_s,  # 3s, matches paper
  )
