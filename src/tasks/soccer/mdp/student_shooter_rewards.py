"""Rewards for motion-free shooter student PPO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from .student_shooter_commands import StudentShooterCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> StudentShooterCommand:
  return env.command_manager.get_term(command_name)


def _tracker(env: ManagerBasedRlEnv, command_name: str):
  return _command(env, command_name).kick_contact_tracker


def _get_or_init_timer(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  timer = getattr(env, name, None)
  if timer is None or timer.shape[0] != env.num_envs:
    timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    setattr(env, name, timer)
  return timer.to(device=env.device, dtype=torch.int32)


def _open_student_timer(
  env: ManagerBasedRlEnv,
  timer_name: str,
  command_name: str,
  ball_sensor_name: str,
  horizontal_force_threshold: float,
  window_size: int,
) -> torch.Tensor:
  timer = _get_or_init_timer(env, timer_name)
  event = _tracker(env, command_name).detect_any_single_foot(ball_sensor_name, horizontal_force_threshold)
  if torch.any(event.new_contact):
    timer[event.new_contact] = window_size
  return timer


def student_target_point_proximity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str = "motion",
) -> torch.Tensor:
  """Reward robot-anchor proximity to the ball, frozen after valid student kick."""
  command = _command(env, command_name)
  tracker = command.kick_contact_tracker
  base_xy = command.robot_anchor_pos_w[:, :2]
  ball_w = command.target_point_pos + env.scene.env_origins
  error = torch.sum((base_xy - ball_w[:, :2]) ** 2, dim=-1)
  proximity = torch.exp(-error / std**2)

  valid_kick_awarded = tracker.get_student_valid_kick_awarded()
  prox_frozen = tracker.get_student_proximity_frozen()
  new_valid = valid_kick_awarded & (~prox_frozen)
  if torch.any(new_valid):
    ids = torch.nonzero(new_valid, as_tuple=False).squeeze(-1)
    tracker.freeze_student_proximity_reward(ids, proximity[ids])

  frozen = tracker.get_student_frozen_proximity_reward()
  return torch.where(valid_kick_awarded, frozen, proximity)


def student_valid_kick_contact(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  ball_sensor_name: str = "ball_robot_contact",
  horizontal_force_threshold: float = 0.0,
) -> torch.Tensor:
  """One-shot reward for first valid single-foot kick by either foot."""
  event = _tracker(env, command_name).detect_any_single_foot(ball_sensor_name, horizontal_force_threshold)
  return event.new_contact.to(torch.float32)


def student_ball_velocity_direction_alignment(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  std: float = 0.8,
  velocity_threshold: float = 0.5,
  horizontal_force_threshold: float = 0.0,
  ball_sensor_name: str = "ball_robot_contact",
  window_size: int = 8,
) -> torch.Tensor:
  """Reward ball velocity direction toward the sampled goal-plane target."""
  command = _command(env, command_name)
  ball = env.scene[command.cfg.ball_entity_name]
  vel = ball.data.root_link_lin_vel_w
  vel_xy = vel[:, :2]
  vel_xy_norm = torch.linalg.vector_norm(vel_xy, dim=-1, keepdim=True)

  direction = command.target_destination_pos - command.initial_target_point_pos
  dir_xy = direction[:, :2]
  dir_norm = torch.linalg.vector_norm(dir_xy, dim=-1, keepdim=True)

  timer_name = command.kick_contact_tracker._tensor_name("student_dir_align_timer")
  timer = _open_student_timer(env, timer_name, command_name, ball_sensor_name, horizontal_force_threshold, window_size)
  active = (timer > 0) & (vel_xy_norm.squeeze(-1) > velocity_threshold) & (dir_norm.squeeze(-1) > 1e-6)

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if torch.any(active):
    dir_unit = dir_xy[active] / dir_norm[active]
    vel_unit = vel_xy[active] / vel_xy_norm[active]
    cos_theta = torch.sum(vel_unit * dir_unit, dim=-1).clamp(-1.0, 1.0)
    error = torch.acos(cos_theta) ** 2
    reward[active] = torch.exp(-error / std**2)

  timer = torch.where(timer > 0, timer - 1, timer)
  setattr(env, timer_name, timer)
  return reward


def student_ball_speed_capped(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  min_speed: float = 1.0,
  cap_speed: float = 2.5,
  horizontal_force_threshold: float = 0.0,
  ball_sensor_name: str = "ball_robot_contact",
  window_size: int = 8,
) -> torch.Tensor:
  """Capped post-kick speed reward; avoids early pressure for uncontrolled speed."""
  command = _command(env, command_name)
  ball = env.scene[command.cfg.ball_entity_name]
  speed_xy = torch.linalg.vector_norm(ball.data.root_link_lin_vel_w[:, :2], dim=-1)

  timer_name = command.kick_contact_tracker._tensor_name("student_speed_timer")
  timer = _open_student_timer(env, timer_name, command_name, ball_sensor_name, horizontal_force_threshold, window_size)
  active = timer > 0

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if torch.any(active):
    normalized = (speed_xy[active] - min_speed) / max(cap_speed - min_speed, 1e-6)
    reward[active] = torch.clamp(normalized, 0.0, 1.0)

  timer = torch.where(timer > 0, timer - 1, timer)
  setattr(env, timer_name, timer)
  return reward


def student_both_feet_ball_contact(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  ball_sensor_name: str = "ball_robot_contact",
  horizontal_force_threshold: float = 0.0,
) -> torch.Tensor:
  """Penalty for simultaneous left/right foot ball contact."""
  event = _tracker(env, command_name).detect_any_single_foot(ball_sensor_name, horizontal_force_threshold)
  return (event.left_foot_contact & event.right_foot_contact).to(torch.float32)


def student_nonfoot_ball_contact(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  ball_sensor_name: str = "ball_robot_contact",
  horizontal_force_threshold: float = 0.0,
) -> torch.Tensor:
  """Penalty for non-foot body contact with the ball."""
  event = _tracker(env, command_name).detect_any_single_foot(ball_sensor_name, horizontal_force_threshold)
  return event.nonfoot_contact.to(torch.float32)


def student_foot_stomp_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  ball_sensor_name: str = "ball_robot_contact",
  horizontal_force_threshold: float = 0.0,
  foot_body_names: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link"),
  foot_above_ball_margin: float = 0.02,
  vz_ratio_threshold: float = 0.5,
) -> torch.Tensor:
  """Penalty for a valid student kick that looks like a downward stomp."""
  command = _command(env, command_name)
  event = command.kick_contact_tracker.detect_any_single_foot(ball_sensor_name, horizontal_force_threshold)
  penalty = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if not torch.any(event.new_contact):
    return penalty

  env_ids = torch.nonzero(event.new_contact, as_tuple=False).squeeze(-1)
  robot = command.robot
  left_idx, right_idx = robot.find_bodies(foot_body_names, preserve_order=True)[0]
  body_indices = torch.where(
    event.right_foot_contact[env_ids],
    torch.full_like(env_ids, right_idx),
    torch.full_like(env_ids, left_idx),
  )
  arange = torch.arange(env_ids.numel(), device=env.device)
  foot_vel_w = robot.data.body_link_lin_vel_w[env_ids][arange, body_indices]
  foot_pos_w = robot.data.body_link_pos_w[env_ids][arange, body_indices]
  ball_pos_w = env.scene[command.cfg.ball_entity_name].data.root_link_pos_w[env_ids]

  foot_above_ball = foot_pos_w[:, 2] > (ball_pos_w[:, 2] + foot_above_ball_margin)
  total_speed = torch.linalg.vector_norm(foot_vel_w, dim=-1)
  vz_ratio = torch.abs(foot_vel_w[:, 2]) / (total_speed + 1e-6)
  is_stomp = foot_above_ball & (vz_ratio > vz_ratio_threshold)
  penalty[env_ids[is_stomp]] = vz_ratio[is_stomp]
  return penalty


def student_pelvis_orientation(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """Penalize pelvis pitch/roll tilt to keep the robot upright."""
  command = _command(env, command_name)
  gravity_vec_w = torch.tensor([[0.0, 0.0, -1.0]], device=env.device).expand(env.num_envs, -1)
  pelvis_proj = quat_apply(quat_inv(command.robot_pelvis_quat_w), gravity_vec_w)
  return torch.sum(torch.square(pelvis_proj[:, :2]), dim=1)
