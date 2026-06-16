"""Soccer-specific reward functions for shooter training.

Port of HumanoidSoccer's soccer/kick rewards to mjlab, with anti-clamp
improvements: all kick rewards gate on valid_kick events (correct single
foot, no other body contact, ball gains speed).

NOTE: Reward functions accept raw entity/body/joint names rather than
SceneEntityCfg objects to avoid mjlab's config-resolution re-evaluation bug.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_inv

from .shooter_commands import MultiMotionSoccerCommand
from .shooter_kick_detection import KickContactTracker

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.entity import Entity


def _get_kick_tracker(command: MultiMotionSoccerCommand) -> KickContactTracker:
  tracker = getattr(command, "kick_contact_tracker", None)
  if tracker is None:
    raise RuntimeError("MultiMotionSoccerCommand missing kick_contact_tracker")
  return tracker


def _make_foot_cfg(env: ManagerBasedRlEnv, foot_body_names: tuple[str, ...]) -> SceneEntityCfg:
  """Build a SceneEntityCfg for foot bodies (used at runtime, not config-time)."""
  return SceneEntityCfg("robot", body_names=foot_body_names, body_ids=None)


# -- Action regularization ------------------------------------------------------

def action_rate_l2_clip(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize action rate of change, clamped at 100."""
  delta = env.action_manager.action - env.action_manager.prev_action
  return torch.sum(torch.square(delta), dim=1).clamp(max=100.0)


def waist_action_rate_l2_clip(
  env: ManagerBasedRlEnv,
  entity_name: str = "robot",
  joint_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """Penalize waist joint action rate of change."""
  robot: Entity = env.scene[entity_name]
  idx = torch.as_tensor(
    robot.find_joints(joint_names, preserve_order=True)[0],
    device=env.device,
  )
  delta = env.action_manager.action[:, idx] - env.action_manager.prev_action[:, idx]
  return torch.sum(torch.square(delta), dim=1).clamp(max=100.0)


# -- Stabilization rewards ------------------------------------------------------

def foot_distance(
  env: ManagerBasedRlEnv,
  threshold: float,
  std: float,
  entity_name: str = "robot",
  body_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """Reward minimum separation between feet to avoid crossing."""
  robot: Entity = env.scene[entity_name]
  idx = torch.as_tensor(
    robot.find_bodies(body_names, preserve_order=True)[0],
    device=env.device,
  )
  # NOTE: assumes body_names[0] is left foot, body_names[1] is right foot.
  left_pos = robot.data.body_link_pos_w[:, idx[0]]
  right_pos = robot.data.body_link_pos_w[:, idx[1]]
  dist = torch.norm(left_pos - right_pos, dim=1)
  return torch.where(
    dist >= threshold,
    torch.ones_like(dist),
    torch.exp(-((dist / threshold - 1) ** 2) / (std**2)),
  )


def pelvis_orientation(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """Penalize pelvis pitch/roll tilt to keep the robot upright."""
  command = env.command_manager.get_term(command_name)
  gravity_vec_w = torch.tensor(
    [[0.0, 0.0, -1.0]], device=env.device
  ).expand(env.num_envs, -1)
  pelvis_proj = quat_apply_inverse(command.robot_pelvis_quat_w, gravity_vec_w)
  return torch.sum(torch.square(pelvis_proj[:, :2]), dim=1)


# -- Soccer / kick rewards ------------------------------------------------------

def _get_target_point_world(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  return command.target_point_pos + env.scene.env_origins


def target_point_proximity(
  env: ManagerBasedRlEnv, std: float, command_name: str = "motion",
) -> torch.Tensor:
  """Reward proximity to the ball; freezes value only after a valid kick.

  - Before valid kick: returns live distance-based proximity (always).
  - After valid kick: freezes proximity at the contact-moment value.

  Invalid contacts (wrong foot, two feet, nonfoot) are punished by the
  separate both_feet_ball and nonfoot_ball penalties, NOT by killing
  proximity. Proximity must remain as continuous guidance signal.
  """
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  tracker = _get_kick_tracker(command)

  base_xy = command.robot_anchor_pos_w[..., :2]
  target_w = _get_target_point_world(env, command_name)
  diff_xy = base_xy - target_w[..., :2]
  error = torch.sum(diff_xy * diff_xy, dim=-1)
  proximity = torch.exp(-error / std**2)

  valid_kick_awarded = tracker.get_valid_kick_awarded()
  frozen = tracker.get_frozen_proximity_reward()

  # Freeze on first valid kick (use explicit bool, not value-based sentinel).
  prox_frozen = tracker.get_proximity_frozen()
  new_valid = valid_kick_awarded & (~prox_frozen)
  if torch.any(new_valid):
    ids = torch.nonzero(new_valid, as_tuple=False).squeeze(-1)
    tracker.freeze_proximity_reward(ids, proximity[ids])
    frozen = tracker.get_frozen_proximity_reward()

  # Two cases:
  # 1. Valid kick occurred → return frozen value (protect from ball flying away).
  # 2. Otherwise → return live proximity (continuous guidance).
  return torch.where(valid_kick_awarded, frozen, proximity)


def target_point_contact(
  env: ManagerBasedRlEnv,
  horizontal_force_threshold: float = 0.0,
  command_name: str = "motion",
  ball_sensor_name: str = "ball_robot_contact",
  foot_body_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """One-shot reward for first valid ball contact with correct foot.

  Only fires on valid kick events (correct single foot, no clamp).
  """
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  tracker = _get_kick_tracker(command)
  event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if not torch.any(event.new_contact):
    return reward

  # new_contact is already gated on valid_kick in the tracker.
  reward[event.new_contact] = 1.0

  # Record success for metrics.
  tracker.record_expected_success(event.new_contact, event.new_contact)
  return reward


def sideways_kick(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  ball_sensor_name: str = "ball_robot_contact",
  horizontal_force_threshold: float = 0.0,
  foot_body_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """Reward foot swing along the expected lateral direction at valid kick.

  Only evaluates on valid kick events.
  """
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  tracker = _get_kick_tracker(command)
  event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if not torch.any(event.new_contact):
    return reward

  # Get foot info from the valid kick event.
  foot_cfg = _make_foot_cfg(env, foot_body_names)
  foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
  if foot_info.env_ids.numel() == 0:
    return reward

  robot = command.robot
  arange = torch.arange(len(foot_info.env_ids), device=env.device)
  foot_vel_w = robot.data.body_link_lin_vel_w[foot_info.env_ids][arange, foot_info.body_indices]
  foot_quat_w = robot.data.body_link_quat_w[foot_info.env_ids][arange, foot_info.body_indices]
  vel_local = quat_apply(quat_inv(foot_quat_w), foot_vel_w)
  vel_norm = torch.norm(vel_local, dim=-1)

  expected_leg = foot_info.expected.to(torch.int8)
  desired_sign = torch.where(expected_leg == 0, -1.0, 1.0)
  directional = vel_local[:, 1] * desired_sign
  axis = torch.clamp(directional, min=0.0)
  alignment = torch.where(vel_norm > 1e-6, axis / vel_norm, torch.zeros_like(vel_norm))
  reward[foot_info.env_ids] = alignment.to(reward.dtype)
  return reward


def foot_stomp_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  ball_sensor_name: str = "ball_robot_contact",
  horizontal_force_threshold: float = 0.0,
  foot_body_names: tuple[str, ...] = (),
  foot_above_ball_margin: float = 0.02,
  vz_ratio_threshold: float = 0.5,
) -> torch.Tensor:
  """Penalty for stomping on ball at valid kick (anti-stomp).

  Fires only when BOTH conditions are true at valid kick moment:
    1. Kicking foot is above ball center (foot_z > ball_z + margin)
    2. Vertical velocity dominates total velocity (|vz|/|v| > ratio_threshold)

  A proper side-kick has foot at or below ball center with mostly
  horizontal velocity — it will never trigger this penalty.

  Returns the vz_ratio as penalty magnitude when both conditions hold.
  """
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  tracker = _get_kick_tracker(command)
  event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

  penalty = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if not torch.any(event.new_contact):
    return penalty

  # Get foot info from the valid kick event.
  foot_cfg = _make_foot_cfg(env, foot_body_names)
  foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
  if foot_info.env_ids.numel() == 0:
    return penalty

  robot = command.robot
  arange = torch.arange(len(foot_info.env_ids), device=env.device)
  foot_vel_w = robot.data.body_link_lin_vel_w[foot_info.env_ids][arange, foot_info.body_indices]
  foot_pos_w = robot.data.body_link_pos_w[foot_info.env_ids][arange, foot_info.body_indices]

  # Ball world position.
  ball_pos = command.soccer_ball_pos[foot_info.env_ids] + env.scene.env_origins[foot_info.env_ids]

  # Condition 1: foot is above ball center (+ small margin).
  foot_above_ball = foot_pos_w[:, 2] > (ball_pos[:, 2] + foot_above_ball_margin)

  # Condition 2: vertical velocity dominates (|vz| / |v| > threshold).
  total_speed = torch.norm(foot_vel_w, dim=-1)
  vz_ratio = torch.abs(foot_vel_w[:, 2]) / (total_speed + 1e-6)
  stomp_like = vz_ratio > vz_ratio_threshold

  # Both conditions must hold.
  is_stomp = foot_above_ball & stomp_like

  # Penalty magnitude = vz_ratio (higher ratio = worse stomp).
  penalty[foot_info.env_ids[is_stomp]] = vz_ratio[is_stomp]
  return penalty


def foot_lift_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  ball_sensor_name: str = "ball_robot_contact",
  horizontal_force_threshold: float = 0.0,
  foot_body_names: tuple[str, ...] = (),
  foot_above_threshold: float = 0.15,
) -> torch.Tensor:
  """Penalize expected kicking foot being too high before valid kick.

  Active every step BEFORE a valid kick occurs. Once valid kick happens,
  stops penalizing (post-kick follow-through is free to lift the foot).

  Penalizes the excess of kicking foot Z above foot_above_threshold.
  Default threshold=0.15m means the foot can be up to 0.15m off ground
  without penalty; above that, penalty grows linearly.

  Ball center is at z=0.11m, so threshold=0.15 catches feet 4cm above
  ball center (likely setting up for a stomp).
  """
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  tracker = _get_kick_tracker(command)

  # Only penalize before valid kick.
  valid_kick_awarded = tracker.get_valid_kick_awarded()
  if torch.all(valid_kick_awarded):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  robot = command.robot
  kick_leg = command.kick_leg  # (num_envs,) int8: 0=left, 1=right, -1=unknown

  # Get foot body indices.
  foot_cfg = _make_foot_cfg(env, foot_body_names)
  body_indices = torch.as_tensor(
    robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0],
    dtype=torch.long, device=env.device,
  )
  # body_indices[0] = left ankle, body_indices[1] = right ankle
  left_idx = body_indices[0]
  right_idx = body_indices[1]

  # Select expected kicking foot Z.
  left_z = robot.data.body_link_pos_w[:, left_idx, 2]
  right_z = robot.data.body_link_pos_w[:, right_idx, 2]
  foot_z = torch.where(kick_leg == 0, left_z, right_z)

  # Excess above threshold.
  excess = torch.clamp(foot_z - foot_above_threshold, min=0.0)

  # Only apply before valid kick; zero out for envs that already kicked.
  penalty = excess * (~valid_kick_awarded).float()

  # Zero out for unknown kick legs.
  penalty = penalty * (kick_leg >= 0).float()

  return penalty


def _get_or_init_timer(env: ManagerBasedRlEnv, name: str, length: int) -> torch.Tensor:
  timer = getattr(env, name, None)
  if timer is None or timer.shape[0] != env.num_envs:
    timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    setattr(env, name, timer)
  return timer.to(device=env.device, dtype=torch.int32)


def _open_timer_window_valid_kick(
  env: ManagerBasedRlEnv,
  timer_name: str,
  command: MultiMotionSoccerCommand,
  ball_sensor_name: str,
  horizontal_force_threshold: float,
  window_size: int,
) -> torch.Tensor:
  """Open a reward window of `window_size` steps on valid kick event."""
  timer = _get_or_init_timer(env, timer_name, window_size)
  tracker = _get_kick_tracker(command)
  event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

  # Only open window on valid kick (new_contact is already valid_kick-gated).
  if torch.any(event.new_contact):
    timer[event.new_contact] = window_size
  return timer


def ball_velocity_direction_alignment(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  std: float = 0.8,
  velocity_threshold: float = 0.5,
  horizontal_force_threshold: float = 0.0,
  ball_sensor_name: str = "ball_robot_contact",
  foot_body_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """Reward ball velocity aligning with target destination after valid kick."""
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  ball = env.scene["ball"]
  vel = ball.data.root_link_lin_vel_w
  vel_norm = torch.norm(vel, dim=-1, keepdim=True)
  vel_xy = vel[:, :2]
  vel_xy_norm = torch.norm(vel_xy, dim=-1, keepdim=True)

  direction = command.target_destination_pos - command.initial_target_point_pos
  dir_xy = direction[:, :2]
  dir_norm = torch.norm(dir_xy, dim=-1, keepdim=True)

  timer_name = f"_{command_name}_dir_align_timer"
  timer = _open_timer_window_valid_kick(env, timer_name, command, ball_sensor_name,
                                        horizontal_force_threshold, 5)

  speed_valid = (vel_norm.squeeze(-1) > velocity_threshold) & (vel_xy_norm.squeeze(-1) > 1e-6) & (dir_norm.squeeze(-1) > 1e-6)
  active = (timer > 0) & speed_valid

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if torch.any(active):
    dir_unit = dir_xy[active] / dir_norm[active]
    vel_unit = vel_xy[active] / vel_xy_norm[active]
    cos_theta = torch.sum(vel_unit * dir_unit, dim=-1).clamp(-1.0, 1.0)
    error = torch.acos(cos_theta) ** 2
    reward[active] = torch.exp(-error / (std**2))

  timer = torch.where(timer > 0, timer - 1, timer)
  setattr(env, timer_name, timer)
  return reward


def ball_speed_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  std: float = 1.2,
  velocity_threshold: float = 0.5,
  horizontal_force_threshold: float = 0.0,
  ball_sensor_name: str = "ball_robot_contact",
  foot_body_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """Reward ball speed magnitude after valid kick."""
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  ball = env.scene["ball"]
  speed_xy = torch.norm(ball.data.root_link_lin_vel_w[:, :2], dim=-1)

  timer_name = f"_{command_name}_speed_timer"
  timer = _open_timer_window_valid_kick(env, timer_name, command, ball_sensor_name,
                                        horizontal_force_threshold, 5)

  speed_valid = speed_xy > 1e-6
  active = (timer > 0) & speed_valid

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if torch.any(active):
    reward[active] = 1.0 - torch.exp(-(speed_xy[active] ** 2) / (std**2))

  timer = torch.where(timer > 0, timer - 1, timer)
  setattr(env, timer_name, timer)
  return reward


def ball_z_speed_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  ball_sensor_name: str = "ball_robot_contact",
  horizontal_force_threshold: float = 0.0,
  foot_body_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """Penalize vertical ball speed after kick contact (paper z-speed, weight=-0.2)."""
  ball = env.scene["ball"]
  vel_z = ball.data.root_link_lin_vel_w[:, 2]

  timer_name = f"_{command_name}_z_speed_timer"
  timer = _open_timer_window_valid_kick(env, timer_name, env.command_manager.get_term(command_name),
                                        ball_sensor_name, horizontal_force_threshold, 5)

  active = timer > 0
  penalty = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if torch.any(active):
    penalty[active] = vel_z[active] ** 2

  timer = torch.where(timer > 0, timer - 1, timer)
  setattr(env, timer_name, timer)
  return penalty


# -- Anti-clamp penalties -------------------------------------------------------

def both_feet_ball_contact(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  ball_sensor_name: str = "ball_robot_contact",
  horizontal_force_threshold: float = 0.0,
) -> torch.Tensor:
  """Penalty: both feet simultaneously contacting the ball (anti-clamp).

  Returns 1.0 for environments where both left and right foot contact the ball.
  """
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  tracker = _get_kick_tracker(command)
  event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
  return (event.left_foot_contact & event.right_foot_contact).to(torch.float32)


def nonfoot_ball_contact(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  ball_sensor_name: str = "ball_robot_contact",
  horizontal_force_threshold: float = 0.0,
) -> torch.Tensor:
  """Penalty: non-foot body part contacting the ball (anti-clamp).

  Returns 1.0 for environments where ball contacts a non-foot body.
  """
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  tracker = _get_kick_tracker(command)
  event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
  return event.nonfoot_contact.to(torch.float32)


# -- Undesired contacts (ground) ------------------------------------------------

def undesired_contacts(
  env: ManagerBasedRlEnv,
  threshold: float = 1.0,
  entity_name: str = "robot",
  excluded_body_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """Penalize undesired body-ground contacts (original HumanoidSoccer, weight=-0.1).

  Counts the number of robot bodies (excluding feet and wrists) with contact
  force exceeding ``threshold`` Newtons against the ground.
  """
  sensor = env.scene.sensors.get("contact_forces")
  if sensor is None or sensor.data.force_history is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  force_hist = sensor.data.force_history  # [B, N_slots, H, 3]
  num_slots_per_body = 2  # matches ContactSensorCfg num_slots

  force_norm = torch.norm(force_hist, dim=-1)          # [B, N_slots, H]
  max_force = force_norm.amax(dim=-1)                   # [B, N_slots]
  max_force = max_force.view(env.num_envs, -1, num_slots_per_body).amax(dim=-1)  # [B, n_bodies]

  if excluded_body_names:
    robot: Entity = env.scene[entity_name]
    exc_idx = torch.as_tensor(
      robot.find_bodies(excluded_body_names, preserve_order=True)[0],
      device=env.device,
    )
    mask = torch.ones(max_force.shape[1], dtype=torch.bool, device=env.device)
    mask[exc_idx] = False
    max_force = max_force[:, mask]

  return (max_force > threshold).float().sum(dim=-1)


# -- Goal-plane crossing rewards (Stage 3) -------------------------------------


def _goal_plane_crossing(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  goal_y: float = -5.0,
  goal_half_width: float = 1.5,
  goal_height: float = 1.8,
):
  """Detect ball crossing goal plane — per-step cached so reward order doesn't matter.

  Returns (crossed, in_goal, cross_pos, target_error) all as (num_envs,) or (num_envs, 3) tensors.
  """
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  tracker = _get_kick_tracker(command)
  valid_kick_awarded = tracker.get_valid_kick_awarded()

  prefix = f"_{command_name}"

  # --- per-step cache: skip recomputation within same env step ---
  cache_step_attr = f"{prefix}_goal_cross_cache_step"
  cache_step = getattr(env, cache_step_attr, -1)
  if cache_step == env.common_step_counter:
    return (
      getattr(env, f"{prefix}_goal_cross_cache_crossed"),
      getattr(env, f"{prefix}_goal_cross_cache_in_goal"),
      getattr(env, f"{prefix}_goal_cross_cache_cross_pos"),
      getattr(env, f"{prefix}_goal_cross_cache_target_error"),
    )
  setattr(env, cache_step_attr, env.common_step_counter)

  # --- episode state ---
  _processed_name = f"{prefix}_goal_cross_processed"
  _prev_name = f"{prefix}_prev_ball_local"
  _prev_valid_name = f"{prefix}_prev_ball_local_valid"

  processed = getattr(env, _processed_name, None)
  if processed is None or processed.shape[0] != env.num_envs:
    processed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    setattr(env, _processed_name, processed)

  prev = getattr(env, _prev_name, None)
  if prev is None or prev.shape[0] != env.num_envs:
    prev = torch.zeros(env.num_envs, 3, dtype=torch.float32, device=env.device)
    setattr(env, _prev_name, prev)

  prev_valid = getattr(env, _prev_valid_name, None)
  if prev_valid is None or prev_valid.shape[0] != env.num_envs:
    prev_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    setattr(env, _prev_valid_name, prev_valid)

  # --- cache output tensors (init once, zero each step) ---
  _cache_crossed_name = f"{prefix}_goal_cross_cache_crossed"
  _cache_in_goal_name = f"{prefix}_goal_cross_cache_in_goal"
  _cache_cross_pos_name = f"{prefix}_goal_cross_cache_cross_pos"
  _cache_target_err_name = f"{prefix}_goal_cross_cache_target_error"

  out_crossed = getattr(env, _cache_crossed_name, None)
  if out_crossed is None or out_crossed.shape[0] != env.num_envs:
    out_crossed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    setattr(env, _cache_crossed_name, out_crossed)

  out_in_goal = getattr(env, _cache_in_goal_name, None)
  if out_in_goal is None or out_in_goal.shape[0] != env.num_envs:
    out_in_goal = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    setattr(env, _cache_in_goal_name, out_in_goal)

  out_cross_pos = getattr(env, _cache_cross_pos_name, None)
  if out_cross_pos is None or out_cross_pos.shape[0] != env.num_envs:
    out_cross_pos = torch.zeros(env.num_envs, 3, dtype=torch.float32, device=env.device)
    setattr(env, _cache_cross_pos_name, out_cross_pos)

  out_target_err = getattr(env, _cache_target_err_name, None)
  if out_target_err is None or out_target_err.shape[0] != env.num_envs:
    out_target_err = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    setattr(env, _cache_target_err_name, out_target_err)

  out_crossed.zero_()
  out_in_goal.zero_()
  out_cross_pos.zero_()
  out_target_err.zero_()

  # --- crossing detection ---
  ball = env.scene["ball"]
  ball_local = ball.data.root_link_pos_w - env.scene.env_origins

  active = valid_kick_awarded & (~processed)
  if not torch.any(active):
    return out_crossed, out_in_goal, out_cross_pos, out_target_err

  need_init = active & (~prev_valid)
  if torch.any(need_init):
    prev[need_init] = ball_local[need_init]
    prev_valid[need_init] = True

  can_check = active & prev_valid & (~need_init)
  if torch.any(can_check):
    crossed = can_check & (prev[:, 1] > goal_y) & (ball_local[:, 1] <= goal_y)
    if torch.any(crossed):
      dy = ball_local[:, 1] - prev[:, 1]
      safe = torch.where(dy >= 0, torch.tensor(1e-6, device=env.device),
                         torch.tensor(-1e-6, device=env.device))
      alpha = ((goal_y - prev[:, 1]) / (dy + safe)).clamp(0.0, 1.0).unsqueeze(-1)
      cross_pos = prev + alpha * (ball_local - prev)

      in_goal = (
        (cross_pos[:, 0].abs() <= goal_half_width)
        & (cross_pos[:, 2] >= 0.0)
        & (cross_pos[:, 2] <= goal_height)
      )
      target_err = (cross_pos[:, 0] - command.target_destination_pos[:, 0]).abs()

      out_crossed[crossed] = True
      out_in_goal[crossed] = in_goal[crossed]
      out_cross_pos[crossed] = cross_pos[crossed]
      out_target_err[crossed] = target_err[crossed]

      processed[crossed] = True

  prev[active] = ball_local[active]
  return out_crossed, out_in_goal, out_cross_pos, out_target_err


def goal_plane_accuracy(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  std: float = 0.3,
  goal_y: float = -5.0,
  goal_half_width: float = 1.5,
  goal_height: float = 1.8,
) -> torch.Tensor:
  """Reward for ball crossing goal plane inside the frame, measured against target x.

  Fires once per episode when the ball first crosses y=goal_y inside the goal frame.
  reward = exp(-target_error² / std²), where target_error = |cross_x - destination_x|.
  """
  crossed, in_goal, _cross_pos, target_err = _goal_plane_crossing(
    env, command_name, goal_y, goal_half_width, goal_height,
  )
  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  mask = crossed & in_goal
  if torch.any(mask):
    reward[mask] = torch.exp(-(target_err[mask] ** 2) / (std**2))
  return reward


def goal_miss_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  goal_y: float = -5.0,
  goal_half_width: float = 1.5,
  goal_height: float = 1.8,
) -> torch.Tensor:
  """One-shot penalty when ball crosses goal plane outside the frame.

  Returns 1.0 for miss; weight should be negative (e.g. -5.0).
  """
  crossed, in_goal, _cross_pos, _target_err = _goal_plane_crossing(
    env, command_name, goal_y, goal_half_width, goal_height,
  )
  return (crossed & (~in_goal)).to(torch.float32)
