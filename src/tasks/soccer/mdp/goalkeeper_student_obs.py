"""Position-conditioned goalkeeper student observation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.soccer.ball_prediction import predict_ball_plane_intersection
from src.tasks.soccer import mdp
from src.tasks.soccer.mdp.goalkeeper_obs import (
  gk_ang_vel,
  gk_joint_pos_rel,
  gk_joint_vel_rel,
  gk_last_action,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


GOALKEEPER_TEACHER_ACTOR_OBS_DIM = 960
GOALKEEPER_PREDICTION_CONDITION_DIM = 4
GOALKEEPER_TASK_CONTEXT_DIM = 19
GOALKEEPER_STUDENT_OBS_DIM = GOALKEEPER_TEACHER_ACTOR_OBS_DIM + GOALKEEPER_TASK_CONTEXT_DIM
GOALKEEPER_PHASE_ACTIVE_INDEX = GOALKEEPER_TEACHER_ACTOR_OBS_DIM
GOALKEEPER_PHASE_IDLE_INDEX = GOALKEEPER_TEACHER_ACTOR_OBS_DIM + 1
GOALKEEPER_CONDITION_Y_SCALE = 1.5
GOALKEEPER_CONDITION_Z_CENTER = 0.9
GOALKEEPER_CONDITION_Z_SCALE = 0.9
GOALKEEPER_CONDITION_TIME_SCALE = 1.5
GOALKEEPER_ACTOR_FRAME_DIM = 96
GOALKEEPER_ACTOR_HISTORY = GOALKEEPER_TEACHER_ACTOR_OBS_DIM // GOALKEEPER_ACTOR_FRAME_DIM
GOALKEEPER_ACTOR_TERM_DIMS = (3, 3, 3, 29, 29, 29)


def normalize_goalkeeper_prediction(
  yz: torch.Tensor,
  time: torch.Tensor,
  idle: torch.Tensor,
  *,
  y_scale: float = GOALKEEPER_CONDITION_Y_SCALE,
  z_center: float = GOALKEEPER_CONDITION_Z_CENTER,
  z_scale: float = GOALKEEPER_CONDITION_Z_SCALE,
  time_scale: float = GOALKEEPER_CONDITION_TIME_SCALE,
) -> torch.Tensor:
  """Normalize ball prediction outputs before condition encoding."""
  if yz.shape[-1] != 2:
    raise ValueError(f"yz must have final dimension 2, got {yz.shape[-1]}")
  time = time.to(device=yz.device, dtype=yz.dtype)
  idle_f = idle.to(device=yz.device, dtype=yz.dtype)
  y, z_raw, time, idle_f = torch.broadcast_tensors(yz[..., 0], yz[..., 1], time, idle_f)
  y = y / y_scale
  z = (z_raw - z_center) / z_scale
  time_norm = torch.clamp(time / time_scale, 0.0, 1.0)
  return torch.stack([y, z, time_norm, idle_f], dim=-1)


def build_goalkeeper_student_obs(
  teacher_actor_obs: torch.Tensor,
  prediction_condition: torch.Tensor,
) -> torch.Tensor:
  """Append a rich task context to the normal goalkeeper actor obs.

  The public prediction input remains the compact normalized
  ``[target_y, target_z, time, idle]`` tuple.  Internally we expand it into a
  stronger task context with phase, current ball kinematics from the 960D
  history, target crossing, and coarse route hints.
  """
  if teacher_actor_obs.shape[-1] != GOALKEEPER_TEACHER_ACTOR_OBS_DIM:
    raise ValueError(
      "teacher_actor_obs must have final dimension "
      f"{GOALKEEPER_TEACHER_ACTOR_OBS_DIM}, got {teacher_actor_obs.shape[-1]}"
    )
  if prediction_condition.shape[-1] != GOALKEEPER_PREDICTION_CONDITION_DIM:
    raise ValueError(
      "prediction_condition must have final dimension "
      f"{GOALKEEPER_PREDICTION_CONDITION_DIM}, got {prediction_condition.shape[-1]}"
    )
  prediction_condition = prediction_condition.to(
    device=teacher_actor_obs.device,
    dtype=teacher_actor_obs.dtype,
  )
  batch_shape = torch.broadcast_shapes(
    teacher_actor_obs.shape[:-1],
    prediction_condition.shape[:-1],
  )
  teacher_actor_obs = teacher_actor_obs.expand(*batch_shape, GOALKEEPER_TEACHER_ACTOR_OBS_DIM)
  prediction_condition = prediction_condition.expand(*batch_shape, GOALKEEPER_PREDICTION_CONDITION_DIM)
  task_context = build_goalkeeper_task_context(teacher_actor_obs, prediction_condition)
  return torch.cat([teacher_actor_obs, task_context], dim=-1)


def build_goalkeeper_task_context(
  teacher_actor_obs: torch.Tensor,
  prediction_condition: torch.Tensor,
) -> torch.Tensor:
  """Build 19D context: phase, launch progress, ball state, target, route hints."""
  idle = prediction_condition[..., 3:4].clamp(0.0, 1.0)
  active = 1.0 - idle
  launch_progress = active
  time_to_crossing = prediction_condition[..., 2:3]
  target_yz = prediction_condition[..., :2]
  ball_history = teacher_actor_obs[..., :30].reshape(*teacher_actor_obs.shape[:-1], 10, 3)
  ball_pos = ball_history[..., -1, :]
  prev_pos = ball_history[..., -2, :]
  ball_vel = torch.clamp((ball_pos - prev_pos) / 0.02, -10.0, 10.0) / 10.0
  ball_pos_norm = torch.stack([
    torch.clamp(ball_pos[..., 0] / 3.0, -1.5, 1.5),
    torch.clamp(ball_pos[..., 1] / GOALKEEPER_CONDITION_Y_SCALE, -1.5, 1.5),
    torch.clamp((ball_pos[..., 2] - GOALKEEPER_CONDITION_Z_CENTER) / GOALKEEPER_CONDITION_Z_SCALE, -1.5, 1.5),
  ], dim=-1)
  high = (target_yz[..., 1:2] > 0.35).to(dtype=teacher_actor_obs.dtype)
  low = (target_yz[..., 1:2] < -0.35).to(dtype=teacher_actor_obs.dtype)
  center_h = 1.0 - torch.clamp(high + low, 0.0, 1.0)
  left = (target_yz[..., 0:1] > 0.2).to(dtype=teacher_actor_obs.dtype)
  right = (target_yz[..., 0:1] < -0.2).to(dtype=teacher_actor_obs.dtype)
  center_y = 1.0 - torch.clamp(left + right, 0.0, 1.0)
  route_hint = torch.cat([left, center_y, right, high, center_h, low], dim=-1)
  target_radius = torch.linalg.vector_norm(target_yz, dim=-1, keepdim=True).clamp(max=2.0) / 2.0
  return torch.cat([
    active,
    idle,
    launch_progress,
    time_to_crossing,
    ball_pos_norm,
    ball_vel,
    target_yz,
    target_radius,
    route_hint,
  ], dim=-1)


def goalkeeper_prediction_condition(
  env: ManagerBasedRlEnv,
  plane_x: float = 0.0,
  idle_speed_threshold: float = 0.5,
  idle_incoming_vx_threshold: float = -0.5,
) -> torch.Tensor:
  """Return normalized ``[pred_y, pred_z, time, idle]`` condition.

  ``_gk_predicted_ball_condition`` is the integration point for an external
  predictor and should contain raw ``[pred_y, pred_z, time, idle]`` values in
  robot frame before normalization.  Without an external predictor we use the
  closed-form ball-plane predictor on the current ball state.
  """
  predicted = getattr(env, "_gk_predicted_ball_condition", None)
  if predicted is not None:
    predicted = predicted.to(device=env.device, dtype=torch.float32)
    launched = getattr(env, "_gk_delayed_ball_launched", None)
    if launched is not None and launched.shape[0] == predicted.shape[0]:
      predicted = predicted.clone()
      predicted[..., 3] = (~launched.to(device=predicted.device, dtype=torch.bool)).to(predicted.dtype)
    return normalize_goalkeeper_prediction(
      predicted[..., :2],
      predicted[..., 2],
      predicted[..., 3].bool(),
    )

  ball = env.scene["ball"]
  robot = env.scene["robot"]
  ball_pos_b = quat_apply_inverse(
    robot.data.root_link_quat_w,
    ball.data.root_link_pos_w - robot.data.root_link_pos_w,
  )
  ball_vel_b = quat_apply_inverse(
    robot.data.root_link_quat_w,
    ball.data.root_link_lin_vel_w,
  )
  pred = predict_ball_plane_intersection(ball_pos_b, ball_vel_b, plane_x=plane_x)
  speed = torch.linalg.vector_norm(ball_vel_b, dim=-1)
  launched = getattr(env, "_gk_delayed_ball_launched", None)
  if launched is not None and launched.shape[0] == env.scene["ball"].data.root_link_pos_w.shape[0]:
    idle = ~launched.to(device=env.device, dtype=torch.bool)
  else:
    prelaunch = (speed < idle_speed_threshold) | (ball_vel_b[..., 0] >= idle_incoming_vx_threshold)
    idle = pred.idle | prelaunch | (pred.time > GOALKEEPER_CONDITION_TIME_SCALE)
  yz = torch.where(idle.unsqueeze(-1), ball_pos_b[..., 1:3], pred.yz)
  time = torch.where(idle, torch.zeros_like(pred.time), pred.time)
  return normalize_goalkeeper_prediction(yz, time, idle)


def goalkeeper_student_obs(
  env: ManagerBasedRlEnv,
  teacher_obs_group: str | None = None,
) -> torch.Tensor:
  """Build student obs from normal MoE-teacher actor obs plus predicted landing."""
  if teacher_obs_group is not None and hasattr(env, "observation_manager"):
    obs = env.observation_manager.compute_group(teacher_obs_group, update_history=True)
  else:
    obs = _goalkeeper_actor_obs_history(env)
  condition = goalkeeper_prediction_condition(env)
  return build_goalkeeper_student_obs(obs, condition)


def _goalkeeper_actor_obs_frame(env: ManagerBasedRlEnv) -> torch.Tensor:
  ball_cfg = SceneEntityCfg("ball")
  robot_cfg = SceneEntityCfg("robot")
  return torch.cat([
    mdp.ball_pos_in_robot_frame(env, ball_cfg=ball_cfg, robot_cfg=robot_cfg),
    gk_ang_vel(env, sensor_name="robot/imu_ang_vel"),
    mdp.projected_gravity(env, asset_cfg=robot_cfg),
    gk_joint_pos_rel(env, asset_cfg=robot_cfg),
    gk_joint_vel_rel(env, asset_cfg=robot_cfg),
    gk_last_action(env),
  ], dim=-1)


def _flatten_goalkeeper_actor_history(history: torch.Tensor) -> torch.Tensor:
  """Flatten 10-frame actor history with the same term-major layout as obs manager."""
  if history.shape[-1] != GOALKEEPER_ACTOR_FRAME_DIM:
    raise ValueError(
      f"history frame dimension must be {GOALKEEPER_ACTOR_FRAME_DIM}, got {history.shape[-1]}"
    )
  chunks = []
  start = 0
  for dim in GOALKEEPER_ACTOR_TERM_DIMS:
    end = start + dim
    chunks.append(history[:, :, start:end].reshape(history.shape[0], -1))
    start = end
  if start != GOALKEEPER_ACTOR_FRAME_DIM:
    raise ValueError("goalkeeper actor term dimensions do not sum to frame dimension")
  return torch.cat(chunks, dim=-1)


def _goalkeeper_actor_obs_history(env: ManagerBasedRlEnv) -> torch.Tensor:
  frame = _goalkeeper_actor_obs_frame(env)
  history = getattr(env, "_gk_student_actor_history", None)
  if history is None or history.shape[0] != env.num_envs or history.shape[-1] != GOALKEEPER_ACTOR_FRAME_DIM:
    history = frame.unsqueeze(1).repeat(1, GOALKEEPER_ACTOR_HISTORY, 1)
  else:
    history = torch.roll(history, shifts=-1, dims=1)
    history[:, -1] = frame
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is not None and episode_length_buf.shape[0] == env.num_envs:
      reset_mask = episode_length_buf.to(device=frame.device) == 0
      if torch.any(reset_mask):
        history[reset_mask] = frame[reset_mask].unsqueeze(1).repeat(1, GOALKEEPER_ACTOR_HISTORY, 1)
  setattr(env, "_gk_student_actor_history", history)
  return _flatten_goalkeeper_actor_history(history)
