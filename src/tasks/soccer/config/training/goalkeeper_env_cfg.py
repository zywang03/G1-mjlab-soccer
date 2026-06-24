"""Goalkeeper training config: single-stage perception-based ball interception.

Builds a training environment matching the Humanoid-Goalkeeper paper's design:
- 96D actor obs (ball_pos + ang_vel + gravity + joint_pos + joint_vel + actions)
  with 10-frame history stacking (960D input to MLP)
- ~109D critic obs (actor terms + base_lin_vel + ball_vel + hand_pos + ball_dist)
- Ball launching via parabolic trajectory with 6 landing regions
- 10 prioritized reward terms
- Domain randomization: robot push + ball velocity perturbation
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.rewards import joint_pos_limits
from mjlab.managers.action_manager import ActionTermCfg
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
  goalkeeper_active_action_rate,
  goalkeeper_active_body_intercept,
  goalkeeper_active_condition_target_reach,
  goalkeeper_active_goal_conceded,
  goalkeeper_active_intercept_point,
  goalkeeper_active_stop_ball,
  goalkeeper_ball_started,
  goalkeeper_idle_action_rate,
  goalkeeper_idle_base_height_band,
  goalkeeper_body_intercept,
  goalkeeper_idle_timeout,
  goalkeeper_ee_reach,
  goalkeeper_goal_conceded,
  goalkeeper_idle_fall_penalty,
  goalkeeper_idle_alive,
  goalkeeper_idle_leg_ready_pose,
  goalkeeper_idle_low_base_height,
  goalkeeper_intercept_point,
  goalkeeper_stop_ball,
  goalkeeper_stay_on_line,
  goalkeeper_no_retreat,
  goalkeeper_feet_slippage,
  goalkeeper_posture_orientation,
  goalkeeper_ang_vel_xy,
)
from src.tasks.soccer.mdp.goalkeeper_obs import (
  goalkeeper_ball_distance,
  goalkeeper_ee_positions,
  goalkeeper_end_region,
  goalkeeper_end_target_pos,
  gk_ang_vel,
  gk_ball_vel_local,
  gk_joint_pos_rel,
  gk_joint_vel_rel,
  gk_last_action,
  gk_lin_vel,
)
from src.tasks.soccer.mdp.goalkeeper_student_obs import goalkeeper_student_obs
from src.tasks.soccer.mdp.shooter_rewards import action_rate_l2_clip
from src.tasks.soccer.mdp.goalkeeper_ball_reset import (
  RegionBallVelCfg,
  reset_ball_static_for_goalkeeper_idle,
  reset_ball_with_parabolic_trajectory,
)


def make_goalkeeper_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create goalkeeper training environment config (single-stage).

  Key design (matching Humanoid-Goalkeeper paper):
  - No motion tracking (unlike shooter's two-stage pipeline)
  - Parabolic trajectory ball launch with 6 landing regions
  - 96D actor obs with 10-frame history stacking → 960D network input
  - Asymmetric actor-critic (critic sees privileged lin_vel + ball_vel)
  - 10 prioritized reward terms from the paper's 24-term reward set
  """

  _BALL_CFG = SceneEntityCfg("ball")
  _ROBOT_CFG = SceneEntityCfg("robot")

  # -- Observations -----------------------------------------------------------

  # Actor: 96D = ball_pos(3) + ang_vel(3) + gravity(3) + joint_pos(29) + joint_vel(29) + actions(29)
  # Matches paper Table I exactly. Observations are scaled to match the reference
  # model's training distribution (ang_vel*0.25, dof_vel*0.05, dof_pos uses
  # goalkeeper-stance default instead of HOME_KEYFRAME).
  actor_terms = {
    "ball_pos_local": ObservationTermCfg(
      func=mdp.ball_pos_in_robot_frame,
      params={"ball_cfg": _BALL_CFG, "robot_cfg": _ROBOT_CFG},
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

  # Critic: actor(96) + lin_vel(3, scaled ×2.0) + ball_vel(3, scaled ×0.2)
  #        + hand_pos(6) + ball_dist(1) + end_target(3) + region(1)
  #        = 113D (matches reference exactly)
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=gk_lin_vel,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
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
      params={"ball_cfg": _BALL_CFG, "robot_cfg": _ROBOT_CFG},
    ),
    "end_target_pos": ObservationTermCfg(
      func=goalkeeper_end_target_pos,
      params={"ball_cfg": _BALL_CFG, "robot_cfg": _ROBOT_CFG},
    ),
    "end_region": ObservationTermCfg(
      func=goalkeeper_end_region,
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=10,  # Paper: num_actor_history=10, 96×10=960D
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
    "joint_pos": JointPositionActionCfg(
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

  region_dicts = [{"height": r.height, "width": r.width} for r in regions_raw]

  vel_cfg = RegionBallVelCfg(
    ball_start_x_range=tuple(SETTINGS.ball_trajectory.ball_start_distance),
    ball_end_x_range=tuple(SETTINGS.ball_trajectory.ball_end_distance),
    t_flight_range=tuple(SETTINGS.ball_trajectory.t_flight),
    regions=region_dicts,
    ball_start_y_range=(min(all_widths_lo), max(all_widths_hi)),
    ball_start_z_range=(min(all_heights_lo), max(all_heights_hi)),
  )

  events = {
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
      interval_range_s=(1.0, 3.0),
      params={
        "vel_xy_range": (-0.5, 0.5),
        "vel_z_range": (-0.2, 0.2),
        "ang_vel_range": (-0.52, 0.52),
      },
    ),
    # DR: ball velocity perturbation.
    "perturb_ball_vel": EventTermCfg(
      func=mdp.perturb_ball_velocity,
      mode="interval",
      interval_range_s=(0.3, 1.0),
      params={
        "vel_range": (-0.5, 0.5),
        "ball_cfg": _BALL_CFG,
      },
    ),
    # Robot state reset with small noise.
    "reset_robot_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"z": (-0.01, 0.01)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (-0.05, 0.05),
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
    # Primary task rewards.
    "ee_reach": RewardTermCfg(
      func=goalkeeper_ee_reach,
      weight=10.0,
      params={
        "std": SETTINGS.goalkeeper_training.ee_reach_std,
      },
    ),
    "stop_ball": RewardTermCfg(
      func=goalkeeper_stop_ball,
      weight=100.0,
      params={
        "velocity_drop_threshold": SETTINGS.goalkeeper_training.stop_ball_vel_drop,
        "behind_robot_x_threshold": SETTINGS.goalkeeper_training.behind_robot_x,
      },
    ),
    # Movement constraints.
    "stay_on_line": RewardTermCfg(func=goalkeeper_stay_on_line, weight=-2.0),
    "no_retreat": RewardTermCfg(func=goalkeeper_no_retreat, weight=-2.0),
    # Foot contact quality.
    "feet_slippage": RewardTermCfg(func=goalkeeper_feet_slippage, weight=-3.0),
    # Posture.
    "posture_orientation": RewardTermCfg(func=goalkeeper_posture_orientation, weight=3.0),
    "ang_vel_xy": RewardTermCfg(func=goalkeeper_ang_vel_xy, weight=-0.1),
    # Regularization (reused).
    "action_rate": RewardTermCfg(func=action_rate_l2_clip, weight=-0.1),
    "joint_limit": RewardTermCfg(
      func=joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
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
      num_envs=1,
      env_spacing=2.5,
      spec_fn=_add_soccer_scene_postproc,
    ),
    observations=observations,
    actions=actions,
    commands={},  # No motion command — single-stage reactive policy
    events=events,
    rewards=rewards,
    terminations=terminations,
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
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=SETTINGS.goalkeeper_episode_length_s,  # 3s, matches paper
  )


def configure_goalkeeper_polish_reward_env_cfg(
  cfg: ManagerBasedRlEnvCfg,
  *,
  idle_fall_penalty_weight: float | None = None,
  posture_weight: float = 0.0,
  action_rate_weight: float = -0.1,
  ang_vel_xy_weight: float | None = None,
  gate_active_rewards: bool = False,
  idle_alive_weight: float | None = None,
  idle_action_rate_weight: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """Use the train_polish-style save/concede reward set."""
  goal_conceded_func = goalkeeper_active_goal_conceded if gate_active_rewards else goalkeeper_goal_conceded
  intercept_func = goalkeeper_active_intercept_point if gate_active_rewards else goalkeeper_intercept_point
  body_func = goalkeeper_active_body_intercept if gate_active_rewards else goalkeeper_body_intercept
  stop_ball_func = goalkeeper_active_stop_ball if gate_active_rewards else goalkeeper_stop_ball
  action_rate_func = goalkeeper_active_action_rate if gate_active_rewards else action_rate_l2_clip
  cfg.rewards = {
    "goal_conceded": RewardTermCfg(
      func=goal_conceded_func,
      weight=-10.0,
      params={},
    ),
    "intercept": RewardTermCfg(
      func=intercept_func,
      weight=2.0,
      params={"std": 0.4},
    ),
    "condition_target_reach": RewardTermCfg(
      func=goalkeeper_active_condition_target_reach,
      weight=1.0,
      params={"std": 0.35, "ball_switch_x": 0.5},
    ),
    "body": RewardTermCfg(
      func=body_func,
      weight=2.0,
      params={"std": 0.35},
    ),
    "stop_ball": RewardTermCfg(
      func=stop_ball_func,
      weight=1.0,
      params={"velocity_drop_threshold": 2.0, "goal_x": -0.5},
    ),
    "posture": RewardTermCfg(
      func=goalkeeper_posture_orientation,
      weight=posture_weight,
    ),
    "action_rate": RewardTermCfg(func=action_rate_func, weight=action_rate_weight),
  }
  if idle_fall_penalty_weight is not None:
    cfg.rewards["idle_fall_penalty"] = RewardTermCfg(
      func=goalkeeper_idle_fall_penalty,
      weight=idle_fall_penalty_weight,
    )
  if idle_alive_weight is not None:
    cfg.rewards["idle_alive"] = RewardTermCfg(
      func=goalkeeper_idle_alive,
      weight=idle_alive_weight,
    )
  if idle_action_rate_weight is not None:
    cfg.rewards["idle_action_rate"] = RewardTermCfg(
      func=goalkeeper_idle_action_rate,
      weight=idle_action_rate_weight,
    )
  if ang_vel_xy_weight is not None:
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(
      func=goalkeeper_ang_vel_xy,
      weight=ang_vel_xy_weight,
    )
  return cfg


def configure_goalkeeper_student_ppo_env_cfg(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Train the 964D position-conditioned LSTM goalkeeper student with PPO."""
  cfg.observations["actor"] = ObservationGroupCfg(
    terms={"student": ObservationTermCfg(func=goalkeeper_student_obs)},
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )
  configure_goalkeeper_polish_reward_env_cfg(
    cfg,
    idle_fall_penalty_weight=-2000.0,
    posture_weight=2.0,
    gate_active_rewards=True,
    idle_alive_weight=0.1,
    idle_action_rate_weight=-0.05,
  )
  cfg.rewards["idle_low_base_height"] = RewardTermCfg(
    func=goalkeeper_idle_low_base_height,
    weight=-20.0,
    params={"target_z": 0.73},
  )
  cfg.rewards["idle_base_height_band"] = RewardTermCfg(
    func=goalkeeper_idle_base_height_band,
    weight=5.0,
    params={"target_z": 0.72, "tolerance": 0.05, "low_margin": 0.08},
  )
  cfg.rewards["idle_leg_ready_pose"] = RewardTermCfg(
    func=goalkeeper_idle_leg_ready_pose,
    weight=-5.0,
  )
  return cfg


def configure_goalkeeper_idle_env_cfg(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Turn a goalkeeper interception cfg into an adversarial idle-expert cfg."""
  _BALL_CFG = SceneEntityCfg("ball")

  cfg.events["reset_ball"] = EventTermCfg(
    func=reset_ball_static_for_goalkeeper_idle,
    mode="reset",
    params={
      "ball_pos": (3.0, 0.0, SETTINGS.ball.radius),
      "idle_wait_range_s": (8.0, 12.0),
      "ball_cfg": _BALL_CFG,
    },
  )
  cfg.events.pop("push_robot", None)
  cfg.events.pop("perturb_ball_vel", None)

  cfg.rewards = {
    "idle_fall_penalty": RewardTermCfg(func=goalkeeper_idle_fall_penalty, weight=-2000.0),
    "idle_low_base_height": RewardTermCfg(
      func=goalkeeper_idle_low_base_height,
      weight=-20.0,
      params={"target_z": 0.72},
    ),
    "idle_base_height_band": RewardTermCfg(
      func=goalkeeper_idle_base_height_band,
      weight=5.0,
      params={"target_z": 0.72, "tolerance": 0.05, "low_margin": 0.08},
    ),
  }

  cfg.terminations["time_out"] = TerminationTermCfg(func=goalkeeper_idle_timeout, time_out=True)
  cfg.terminations["ball_started"] = TerminationTermCfg(
    func=goalkeeper_ball_started,
    params={"speed_threshold": 0.5, "incoming_vx_threshold": -0.5},
  )
  cfg.episode_length_s = 12.0
  return cfg


def configure_goalkeeper_prepare_adversarial_env_cfg(
  cfg: ManagerBasedRlEnvCfg,
) -> ManagerBasedRlEnvCfg:
  """Prepare expert: polish reward plus a waiting-phase fall penalty."""
  _BALL_CFG = SceneEntityCfg("ball")

  cfg.events["reset_ball"] = EventTermCfg(
    func=reset_ball_static_for_goalkeeper_idle,
    mode="reset",
    params={
      "ball_pos": (3.0, 0.0, SETTINGS.ball.radius),
      "idle_wait_range_s": (8.0, 12.0),
      "ball_cfg": _BALL_CFG,
    },
  )
  cfg.events.pop("push_robot", None)
  cfg.events.pop("perturb_ball_vel", None)
  configure_goalkeeper_polish_reward_env_cfg(
    cfg,
    idle_fall_penalty_weight=-2000.0,
    posture_weight=2.0,
    action_rate_weight=-0.5,
    ang_vel_xy_weight=-0.5,
  )
  cfg.rewards["idle_low_base_height"] = RewardTermCfg(
    func=goalkeeper_idle_low_base_height,
    weight=-20.0,
    params={"target_z": 0.73},
  )
  cfg.rewards["idle_leg_ready_pose"] = RewardTermCfg(
    func=goalkeeper_idle_leg_ready_pose,
    weight=-5.0,
  )
  cfg.terminations["time_out"] = TerminationTermCfg(func=mdp.time_out, time_out=True)
  cfg.terminations.pop("ball_started", None)
  cfg.episode_length_s = 12.0
  return cfg
