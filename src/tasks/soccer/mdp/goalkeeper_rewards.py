"""Goalkeeper-specific reward functions for training.

Matches the Humanoid-Goalkeeper paper's reward set (24 terms), prioritized
to 7 goalkeeper-specific functions. The remaining regularization terms
(action_rate, joint_pos_limits, etc.) are reused from existing modules.

State tracking follows the pattern in training_rewards.py: per-environment
tensors stored on the env object via _gk_get_or_init_state, reset via a
dedicated reset event.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.soccer.mdp.goalkeeper_obs import _REF_DEFAULT_DOF_POS
if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.entity import Entity


MILD_READY_JOINT_POS: dict[str, float] = {
  "left_hip_pitch_joint": -0.18,
  "left_knee_joint": 0.45,
  "left_ankle_pitch_joint": -0.27,
  "right_hip_pitch_joint": -0.18,
  "right_knee_joint": 0.45,
  "right_ankle_pitch_joint": -0.27,
}

GOALKEEPER_MOTION_PRIOR_FILES: tuple[str, ...] = (
  "lefthand.pt",
  "righthand.pt",
  "leftjump.pt",
  "rightjump.pt",
  "leftstep.pt",
  "rightstep.pt",
)

GOALKEEPER_REGION_MOTION_PRIOR_FILES: tuple[str, ...] = (
  "righthand.pt",  # Right-Mid
  "lefthand.pt",  # Left-Mid
  "rightjump.pt",  # Right-Up
  "leftjump.pt",  # Left-Up
  "rightstep.pt",  # Right-Low
  "leftstep.pt",  # Left-Low
)


def _gk_get_or_init_state(
  env: ManagerBasedRlEnv, name: str, default: float, dtype=torch.float32,
) -> torch.Tensor:
  t = getattr(env, name, None)
  if t is None or t.shape[0] != env.num_envs:
    t = torch.full((env.num_envs,), default, dtype=dtype, device=env.device)
    setattr(env, name, t)
  return t


def _resolve_ee_indices(
  robot: Entity, ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  """Look up body indices for named end-effector bodies at runtime."""
  return torch.as_tensor(
    robot.find_bodies(ee_body_names, preserve_order=True)[0],
    device=robot.data.body_link_pos_w.device,
  )


def _resolve_joint_indices(
  robot: Entity,
  joint_names: tuple[str, ...],
  device: torch.device,
) -> torch.Tensor:
  return torch.as_tensor(
    robot.find_joints(joint_names, preserve_order=True)[0],
    dtype=torch.long,
    device=device,
  )


def _load_goalkeeper_motion_prior(
  env: ManagerBasedRlEnv,
  motion_dir: str,
  robot: Entity,
  motion_names: tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  cache_key = "_gk_motion_prior_joint_pose_cache"
  cached = getattr(env, cache_key, None)
  selected_motion_names = tuple(motion_names or GOALKEEPER_MOTION_PRIOR_FILES)
  cache_id = (motion_dir, selected_motion_names, str(env.device))
  if cached is not None and cached[0] == cache_id:
    return cached[1], cached[2], cached[3]

  path = Path(motion_dir)
  joint_lines = (path / "joint_id.txt").read_text(encoding="utf-8").splitlines()
  joint_names: list[str] = []
  for line in joint_lines:
    parts = line.strip().split()
    if len(parts) >= 2:
      joint_names.append(parts[1])
  if not joint_names:
    raise ValueError(f"No joint names found in goalkeeper motion prior: {path / 'joint_id.txt'}")

  motion_tensors: list[torch.Tensor] = []
  lengths: list[int] = []
  max_len = 0
  for name in selected_motion_names:
    motion_path = path / name
    if not motion_path.exists():
      continue
    data = torch.load(motion_path, map_location=env.device)
    joint_pos = data["joint_position"].to(device=env.device, dtype=torch.float32)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(joint_names):
      raise ValueError(
        f"{motion_path} joint_position must have shape (T, {len(joint_names)}), got {tuple(joint_pos.shape)}"
      )
    motion_tensors.append(joint_pos)
    lengths.append(int(joint_pos.shape[0]))
    max_len = max(max_len, int(joint_pos.shape[0]))
  if not motion_tensors:
    raise FileNotFoundError(f"No goalkeeper motion prior .pt files found in {path}")

  padded = torch.zeros(len(motion_tensors), max_len, len(joint_names), device=env.device)
  for idx, motion in enumerate(motion_tensors):
    padded[idx, : motion.shape[0]] = motion
    padded[idx, motion.shape[0] :] = motion[-1]
  length_tensor = torch.tensor(lengths, dtype=torch.long, device=env.device)
  joint_ids = _resolve_joint_indices(robot, tuple(joint_names), env.device)
  setattr(env, cache_key, (cache_id, padded, length_tensor, joint_ids))
  return padded, length_tensor, joint_ids


def _idle_wait_mask(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  speed_threshold: float = 0.5,
  incoming_vx_threshold: float = -0.5,
) -> torch.Tensor:
  launched = getattr(env, "_gk_delayed_ball_launched", None)
  if launched is not None and launched.shape[0] == env.num_envs:
    return ~launched.to(device=env.device, dtype=torch.bool)

  ball: Entity = env.scene[ball_cfg.name]
  ball_vel = ball.data.root_link_lin_vel_w
  speed = torch.norm(ball_vel, dim=-1)
  return (speed < speed_threshold) | (ball_vel[:, 0] >= incoming_vx_threshold)


def _active_ball_mask(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  return ~_idle_wait_mask(env, ball_cfg)


def _apply_active_ball_mask(
  env: ManagerBasedRlEnv,
  reward: torch.Tensor,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  return reward * _active_ball_mask(env, ball_cfg).to(dtype=reward.dtype)


# -- Reset event ---------------------------------------------------------------


def _reset_gk_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
) -> None:
  """Reset per-environment GK tracking state on episode start."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  for key in ("_gk_max_ball_speed", "_gk_block_awarded", "_gk_conceded"):
    t = getattr(env, key, None)
    if t is not None and t.shape[0] == env.num_envs:
      t[env_ids] = 0.0


# -- Reward functions ----------------------------------------------------------


def goalkeeper_ee_reach(
  env: ManagerBasedRlEnv,
  std: float = 0.3,
  planar: bool = False,
  near_gate_x: float = 0.0,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ee_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ),
) -> torch.Tensor:
  """End-effector reach reward: exponential distance to ball.

  Paper's 'eereach' — encourages hands and feet to reach toward the ball,
  rewarding proximity of 4 end-effectors (both hands + both feet) via an
  exponential decay: sum_i exp(-||ee_i - ball||² / σ²).

  If ``planar`` is True, only the lateral/vertical (y, z) components of the
  offset are used (the forward x axis is ignored). This is important for the
  goalkeeper: the ball starts 3-5m in front, so a full 3D reach reward pulls the
  keeper into a forward lunge that topples it. Rewarding y-z alignment instead
  makes the keeper stay on its line (x≈0) and move a hand/foot to the ball's
  crossing point — the actual blocking behavior.

  If ``near_gate_x`` > 0, the reward is gated to fire only while the ball is
  within ``near_gate_x`` metres of the keeper plane (|ball_x_rel| < near_gate_x).
  Combined with full 3D distance (planar=False), this rewards an actual
  interception at the moment the ball arrives — it cannot be gamed by loose
  alignment, and being gated it does not pull the keeper into an early lunge at
  the far ball.

  Weight: 10.0 (matches paper).
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]

  ball_pos = ball.data.root_link_pos_w  # (B, 3)
  ee_indices = _resolve_ee_indices(robot, ee_body_names)
  ee_pos = robot.data.body_link_pos_w[:, ee_indices]  # (B, N, 3)

  delta = ee_pos - ball_pos.unsqueeze(1)  # (B, N, 3)
  if planar:
    delta = delta[..., 1:]  # keep (y, z) only — ignore forward x
  dist_sq = torch.sum(delta * delta, dim=-1)  # (B, N)
  reward_per_ee = torch.exp(-dist_sq / (std * std))  # (B, N)
  reward = reward_per_ee.sum(dim=-1)  # (B,)

  if near_gate_x > 0.0:
    ball_x_rel = ball_pos[:, 0] - env.scene.env_origins[:, 0]
    gate = (ball_x_rel.abs() < near_gate_x).to(reward.dtype)
    reward = reward * gate

  return reward


def goalkeeper_body_intercept(
  env: ManagerBasedRlEnv,
  std: float = 0.35,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Dense whole-body blocking reward: exp(-min_dist(ball, ANY body link)²/σ²).

  A block is ANY body part intersecting the ball, so rewarding the minimum
  distance from the ball to any of the robot's body links is directly aligned
  with blocking and dense whenever the ball is near the keeper.
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  ball_pos = ball.data.root_link_pos_w           # (B,3)
  body_pos = robot.data.body_link_pos_w          # (B,L,3)
  d2 = (body_pos - ball_pos.unsqueeze(1)).pow(2).sum(dim=-1)
  return torch.exp(-d2.min(dim=1).values / (std * std))


def goalkeeper_active_body_intercept(
  env: ManagerBasedRlEnv,
  std: float = 0.35,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  reward = goalkeeper_body_intercept(env, std=std, ball_cfg=ball_cfg, robot_cfg=robot_cfg)
  return _apply_active_ball_mask(env, reward, ball_cfg)


def goalkeeper_intercept_point(
  env: ManagerBasedRlEnv,
  std: float = 0.4,
  ee_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link", "right_wrist_yaw_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
  ),
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Dense reward: nearest end-effector close to the ball's INTERCEPTION POINT.

  We extrapolate the ball's trajectory (ballistic, with gravity) to the keeper's
  plane (x_rel = 0) to get the exact (y, z) where the ball will cross, then
  reward the nearest hand/foot for being at that 3D point. Unlike rewarding
  proximity to the ball's *current* position (which pulls a forward lunge) or
  loose planar alignment (which the keeper games without intercepting), this
  gives a STABLE, directional target over the whole flight, so the keeper learns
  to pre-position a limb exactly where the ball will arrive — the actual block.
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  bp = ball.data.root_link_pos_w           # (B,3) world
  bv = ball.data.root_link_lin_vel_w       # (B,3)
  org = env.scene.env_origins              # (B,3)

  ball_x_rel = bp[:, 0] - org[:, 0]
  vx = bv[:, 0]
  # time for the ball to reach the keeper plane (x_rel = 0); 0 if already past.
  t = torch.clamp(-ball_x_rel / (vx - 1e-3), min=0.0, max=2.0)
  g = 9.81
  cross = torch.empty_like(bp)
  cross[:, 0] = org[:, 0]                              # keeper plane (x_rel = 0)
  cross[:, 1] = bp[:, 1] + bv[:, 1] * t
  cross[:, 2] = torch.clamp(bp[:, 2] + bv[:, 2] * t - 0.5 * g * t * t, min=0.0)

  # Whole-body: reward ANY body link being at the interception point (a block is
  # any body part on the ball, not just a hand). This is the precise, adaptive
  # target the keeper should reach — independent of which fixed dive it uses.
  body = robot.data.body_link_pos_w                   # (B,L,3)
  d2 = (body - cross.unsqueeze(1)).pow(2).sum(dim=-1)  # (B,L)
  return torch.exp(-d2.min(dim=1).values / (std * std))


def goalkeeper_active_intercept_point(
  env: ManagerBasedRlEnv,
  std: float = 0.4,
  ee_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link", "right_wrist_yaw_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
  ),
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  reward = goalkeeper_intercept_point(
    env,
    std=std,
    ee_body_names=ee_body_names,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )
  return _apply_active_ball_mask(env, reward, ball_cfg)


def _selected_condition_ee_indices(
  target_yz: torch.Tensor,
  ee_count: int,
) -> torch.Tensor:
  """Choose a body-link slot from target side/height when no true region is exposed."""
  if ee_count <= 1:
    return torch.zeros(target_yz.shape[0], dtype=torch.long, device=target_yz.device)
  z = target_yz[:, 1]
  y = target_yz[:, 0]
  high = z > 1.25
  low = z < 0.75
  right = y >= 0.0
  # Default body order is left hand, right hand, left foot, right foot. This
  # mirrors the reference mapping where region 0 (positive-y side in our reset
  # convention) is matched against the left hand.
  left_hand = torch.zeros_like(y, dtype=torch.long)
  right_hand = torch.ones_like(left_hand).clamp(max=ee_count - 1)
  left_foot = torch.full_like(left_hand, min(2, ee_count - 1))
  right_foot = torch.full_like(left_hand, min(3, ee_count - 1))
  hand = torch.where(right, left_hand, right_hand)
  foot = torch.where(right, left_foot, right_foot)
  return torch.where(low & ~high, foot, hand)


def goalkeeper_active_condition_target_reach(
  env: ManagerBasedRlEnv,
  std: float = 0.35,
  ball_switch_x: float = 0.5,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ee_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ),
) -> torch.Tensor:
  """Active dense reward tied directly to the student's landing condition.

  Far from the keeper, the target is the predicted crossing/landing ``(y, z)``.
  When the ball is close to the keeper plane, the target switches to the live
  ball position, matching the paper's position-conditioned task reward shape.
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  from src.tasks.soccer.mdp.goalkeeper_student_obs import (
    GOALKEEPER_CONDITION_Y_SCALE,
    GOALKEEPER_CONDITION_Z_CENTER,
    GOALKEEPER_CONDITION_Z_SCALE,
    goalkeeper_prediction_condition,
  )

  condition = goalkeeper_prediction_condition(env)
  pred_y = condition[:, 0] * GOALKEEPER_CONDITION_Y_SCALE
  pred_z = condition[:, 1] * GOALKEEPER_CONDITION_Z_SCALE + GOALKEEPER_CONDITION_Z_CENTER
  target_yz = torch.stack([pred_y, pred_z], dim=-1)

  ball_pos_b = quat_apply_inverse(
    robot.data.root_link_quat_w,
    ball.data.root_link_pos_w - robot.data.root_link_pos_w,
  )
  near = ball_pos_b[:, 0].abs() <= ball_switch_x
  target_yz = torch.where(near.unsqueeze(-1), ball_pos_b[:, 1:3], target_yz)

  ee_indices = _resolve_ee_indices(robot, ee_body_names)
  ee_pos_w = robot.data.body_link_pos_w[:, ee_indices]
  num_ee = ee_pos_w.shape[1]
  ee_delta = ee_pos_w - robot.data.root_link_pos_w.unsqueeze(1)
  ee_pos_b = quat_apply_inverse(
    robot.data.root_link_quat_w.unsqueeze(1).expand(-1, num_ee, -1).reshape(-1, 4),
    ee_delta.reshape(-1, 3),
  ).view(env.num_envs, num_ee, 3)
  selected = _selected_condition_ee_indices(target_yz, ee_pos_b.shape[1])
  batch = torch.arange(env.num_envs, device=env.device)
  selected_yz = ee_pos_b[batch, selected, 1:3]
  d2 = torch.sum((selected_yz - target_yz) ** 2, dim=-1)
  reward = torch.exp(-d2 / (std * std))
  return _apply_active_ball_mask(env, reward, ball_cfg)


def goalkeeper_stop_ball(
  env: ManagerBasedRlEnv,
  velocity_drop_threshold: float = 2.0,
  behind_robot_x_threshold: float = 0.0,
  goal_x: float = -0.5,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """One-shot block reward when the ball is deflected before entering the goal.

  Coordinate system (see goalkeeper_ball_reset.py): robot faces +x; ball comes
  from +x (front) toward -x; the goal plane is at x = goal_x (-0.5). A successful
  save = the ball is decelerated significantly while it is still in front of the
  goal plane (i.e. the keeper has deflected/stopped it before it scores).

  Tracks per-environment max ball speed and awards 1.0 when:
    1. max_speed - current_speed > velocity_drop_threshold (2.0 m/s), AND
    2. the ball has NOT yet crossed the goal plane (ball_x > goal_x).

  The reward is awarded at most once per episode per environment.
  (Was previously gated on ``ball_x > robot_x`` — an inverted condition copied
  from a stale "ball comes from -x" coordinate convention.)

  Weight: 100.0 (matches paper's stopball=100).
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]

  ball_pos = ball.data.root_link_pos_w  # (B, 3)
  ball_vel = ball.data.root_link_lin_vel_w  # (B, 3)
  robot_pos = robot.data.root_link_pos_w  # (B, 3)

  current_speed = torch.norm(ball_vel, dim=-1)  # (B,)

  max_speed = _gk_get_or_init_state(env, "_gk_max_ball_speed", 0.0)
  block_awarded = _gk_get_or_init_state(env, "_gk_block_awarded", 0.0)

  # Update per-env max speed.
  max_speed = torch.maximum(max_speed, current_speed)
  setattr(env, "_gk_max_ball_speed", max_speed)

  # Ball origin is in env-local x; env_origins offset is added in world frame, so
  # compare against the goal plane in the robot/world-relative x of the ball.
  ball_x_rel = ball_pos[:, 0] - env.scene.env_origins[:, 0]
  speed_drop = (max_speed - current_speed) > velocity_drop_threshold
  ball_in_play = ball_x_rel > goal_x  # not yet scored
  block_detected = speed_drop & ball_in_play & (block_awarded < 0.5)

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  if torch.any(block_detected):
    ids = torch.nonzero(block_detected, as_tuple=False).squeeze(-1)
    reward[ids] = 1.0
    block_awarded[ids] = 1.0
    setattr(env, "_gk_block_awarded", block_awarded)

  return reward


def goalkeeper_active_stop_ball(
  env: ManagerBasedRlEnv,
  velocity_drop_threshold: float = 2.0,
  behind_robot_x_threshold: float = 0.0,
  goal_x: float = -0.5,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  reward = goalkeeper_stop_ball(
    env,
    velocity_drop_threshold=velocity_drop_threshold,
    behind_robot_x_threshold=behind_robot_x_threshold,
    goal_x=goal_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )
  return _apply_active_ball_mask(env, reward, ball_cfg)


def goalkeeper_goal_conceded(
  env: ManagerBasedRlEnv,
  goal_x: float = -0.5,
  goal_half_width: float = 1.5,
  goal_height: float = 1.8,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  """One-shot penalty (returns 1.0 once) when the ball enters the goal.

  Mirrors the eval block criterion exactly (ball crosses x ≤ goal_x inside the
  |y| ≤ goal_half_width, z ≤ goal_height frame), so optimizing -weight·this term
  directly optimizes the evaluated block rate. Awarded at most once per episode.
  Intended for PPO fine-tuning of an already-competent (distilled) policy, where
  it provides a precisely-aligned learning signal that dense shaping rewards lack.
  """
  ball: Entity = env.scene[ball_cfg.name]
  ball_pos = ball.data.root_link_pos_w
  ball_x_rel = ball_pos[:, 0] - env.scene.env_origins[:, 0]
  ball_y_rel = ball_pos[:, 1] - env.scene.env_origins[:, 1]
  in_goal = (
    (ball_x_rel <= goal_x)
    & (ball_y_rel.abs() <= goal_half_width)
    & (ball_pos[:, 2] <= goal_height)
  )

  conceded = _gk_get_or_init_state(env, "_gk_conceded", 0.0)
  fire = in_goal & (conceded < 0.5)
  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if torch.any(fire):
    ids = torch.nonzero(fire, as_tuple=False).squeeze(-1)
    reward[ids] = 1.0
    conceded[ids] = 1.0
    setattr(env, "_gk_conceded", conceded)
  return reward


def goalkeeper_active_goal_conceded(
  env: ManagerBasedRlEnv,
  goal_x: float = -0.5,
  goal_half_width: float = 1.5,
  goal_height: float = 1.8,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  reward = goalkeeper_goal_conceded(
    env,
    goal_x=goal_x,
    goal_half_width=goal_half_width,
    goal_height=goal_height,
    ball_cfg=ball_cfg,
  )
  return _apply_active_ball_mask(env, reward, ball_cfg)


def goalkeeper_active_action_rate(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  from src.tasks.soccer.mdp.shooter_rewards import action_rate_l2_clip

  reward = action_rate_l2_clip(env)
  return _apply_active_ball_mask(env, reward, ball_cfg)


def goalkeeper_active_upright(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward staying upright after launch so active RL has a dense stability signal."""
  reward = goalkeeper_posture_orientation(env, robot_cfg=robot_cfg)
  return _apply_active_ball_mask(env, reward, ball_cfg)


def goalkeeper_active_fall_penalty(
  env: ManagerBasedRlEnv,
  limit_angle: float = 1.2217304763960306,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """One-step fall indicator after launch, complementing idle fall penalty."""
  robot: Entity = env.scene[robot_cfg.name]
  upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0)
  limit_cos = torch.cos(torch.tensor(limit_angle, dtype=torch.float32, device=env.device))
  fallen = upright < limit_cos
  return fallen.float() * _active_ball_mask(env, ball_cfg).to(dtype=upright.dtype)


def goalkeeper_active_motion_prior_joint_pose(
  env: ManagerBasedRlEnv,
  motion_dir: str = "src/assets/soccer/motions/goalkeeper",
  motion_names: tuple[str, ...] | None = None,
  route_mode: str = "all",
  std: float = 0.5,
  launch_delay_s: float = 3.0,
  dt: float | None = None,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward active-stage joint poses that stay close to the goalkeeper motion prior.

  The six goalkeeper motions are state trajectories, not observation-action
  pairs, so this is a pose-tracking regularizer rather than an RL-stage BC loss.
  """
  robot: Entity = env.scene[robot_cfg.name]
  route_mode = route_mode.lower()
  if route_mode not in {"all", "region"}:
    raise ValueError(f"Unknown goalkeeper motion-prior route_mode: {route_mode!r}")
  selected_motion_names = motion_names
  if route_mode == "region":
    selected_motion_names = motion_names or GOALKEEPER_REGION_MOTION_PRIOR_FILES
    if len(selected_motion_names) != len(GOALKEEPER_REGION_MOTION_PRIOR_FILES):
      raise ValueError(
        "route_mode='region' requires exactly six motion prior files ordered as "
        "Right-Mid, Left-Mid, Right-Up, Left-Up, Right-Low, Left-Low"
      )
  motion_joint_pos, motion_lengths, joint_ids = _load_goalkeeper_motion_prior(
    env, motion_dir, robot, motion_names=selected_motion_names
  )
  if dt is None:
    dt = float(getattr(env, "step_dt", 0.02))
  launch_delay_steps = int(round(max(launch_delay_s, 0.0) / max(float(dt), 1e-6)))
  episode_steps = getattr(env, "episode_length_buf", None)
  if episode_steps is None:
    active_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  else:
    active_steps = (episode_steps.to(device=env.device, dtype=torch.long) - launch_delay_steps).clamp_min(0)

  current = robot.data.joint_pos[:, joint_ids].to(dtype=torch.float32)
  if route_mode == "region":
    region = getattr(env, "_gk_region", None)
    if region is None or region.shape[0] != env.num_envs:
      region_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    else:
      region_idx = region.to(device=env.device, dtype=torch.long).clamp(0, motion_joint_pos.shape[0] - 1)
    frame_idx = torch.minimum(active_steps, motion_lengths[region_idx] - 1)
    reference = motion_joint_pos[region_idx, frame_idx]
    best_err = torch.mean((current - reference) ** 2, dim=-1)
  else:
    motion_idx = torch.arange(motion_joint_pos.shape[0], device=env.device).view(1, -1)
    frame_idx = torch.minimum(
      active_steps.view(-1, 1),
      (motion_lengths - 1).view(1, -1),
    )
    reference = motion_joint_pos[motion_idx, frame_idx]  # (B, M, J)
    err = torch.mean((current.unsqueeze(1) - reference) ** 2, dim=-1)
    best_err = torch.min(err, dim=1).values
  reward = torch.exp(-best_err / max(std * std, 1e-6))
  return _apply_active_ball_mask(env, reward, ball_cfg)


def goalkeeper_idle_action_rate(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  from src.tasks.soccer.mdp.shooter_rewards import action_rate_l2_clip

  reward = action_rate_l2_clip(env)
  return reward * _idle_wait_mask(env, ball_cfg).to(dtype=reward.dtype)


def goalkeeper_stay_on_line(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize lateral deviation from goal center.

  Paper's 'stayonline' — discourages the robot from drifting sideways
  away from the center of the goal line.

  Weight: -2.0 (matches paper).
  """
  robot: Entity = env.scene[robot_cfg.name]
  robot_y = robot.data.root_link_pos_w[:, 1]
  return torch.abs(robot_y)


def goalkeeper_no_retreat(
  env: ManagerBasedRlEnv,
  goal_line_x: float = 0.0,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize retreating behind the goal line.

  Paper's 'noretreat' — discourages the robot from backing into its own goal.
  Coordinate system (see goalkeeper_ball_reset.py): robot faces +x at x≈0, the
  ball comes from +x, and the goal is behind at -x. "Retreating" therefore means
  moving toward -x. We penalize robot_x dropping below goal_line_x.
  (Was previously ``clamp(robot_x - goal_line_x)`` — penalizing forward motion
  toward the ball, the opposite of the intent, from a stale coordinate frame.)

  Weight: -2.0 (matches paper).
  """
  robot: Entity = env.scene[robot_cfg.name]
  robot_x = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  retreat = torch.clamp(goal_line_x - robot_x, min=0.0)
  return retreat * retreat


def goalkeeper_feet_slippage(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  foot_body_names: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ),
  contact_sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Penalize foot sliding when feet are in ground contact.

  Paper's 'feet_slippage' — returns the sum of squared xy-velocities
  for feet that are currently touching the ground. This discourages
  slipping while moving laterally.

  Weight: -3.0 (exp(-10*vel) → direct penalty).
  """
  robot: Entity = env.scene[robot_cfg.name]
  foot_indices = _resolve_ee_indices(robot, foot_body_names)
  foot_vel_w = robot.data.body_link_lin_vel_w[:, foot_indices]  # (B, 2, 3)
  foot_vel_xy_sq = torch.sum(foot_vel_w[:, :, :2] ** 2, dim=-1)  # (B, 2)

  # Check foot-ground contact via sensor.
  # feet_ground_contact uses reduce="netforce", single slot.
  contact = env.scene.sensors.get(contact_sensor_name, None)
  if contact is not None:
    found = contact.data.force[:, 0]  # (B, 3) — net force on feet
    contact_found = torch.norm(found, dim=-1) > 1.0  # (B,)
  else:
    contact_found = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  slip = foot_vel_xy_sq.sum(dim=-1)  # (B,)
  return slip * contact_found.float()


def goalkeeper_posture_orientation(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward upright posture via projected gravity.

  Paper's 'postorientation'. ``projected_gravity_b`` is the (normalized) gravity
  direction expressed in the robot base frame, so its z-component is ≈ -1.0 when
  the robot is perfectly upright and rises toward 0 as it tilts to horizontal.
  We therefore reward ``clamp(-grav_z, 0, 1)`` → 1.0 upright, 0.0 on its side.
  (Was ``clamp(grav_z, 0, 1)``, which is ≈0 for an upright robot and never fired
  — a sign error that removed the dense "stay upright" signal.)

  Weight: 3.0 (matches paper).
  """
  robot: Entity = env.scene[robot_cfg.name]
  grav_z = robot.data.projected_gravity_b[:, 2]
  return torch.clamp(-grav_z, 0.0, 1.0)


def goalkeeper_ang_vel_xy(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize base angular velocity in the xy-plane.

  Paper's 'ang_vel_xy' — high angular velocity in pitch/roll axes
  indicates instability during reactive movements.

  Weight: -0.1 (matches paper).
  """
  robot: Entity = env.scene[robot_cfg.name]
  ang_vel = robot.data.root_link_ang_vel_b[:, :2]  # (B, 2)
  return torch.sum(ang_vel * ang_vel, dim=-1)  # (B,)


def goalkeeper_idle_joint_pose(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize deviation from the goalkeeper ready pose while the ball is idle."""
  robot: Entity = env.scene[robot_cfg.name]
  default = torch.tensor(_REF_DEFAULT_DOF_POS, device=env.device, dtype=torch.float32)
  err = robot.data.joint_pos - default.view(1, -1)
  return torch.sum(err * err, dim=-1) * _idle_wait_mask(env, ball_cfg).float()


def goalkeeper_idle_low_base_height(
  env: ManagerBasedRlEnv,
  target_z: float = 0.73,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize a tall base during the prepare wait window."""
  robot: Entity = env.scene[robot_cfg.name]
  base_z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  high = torch.clamp(base_z - target_z, min=0.0)
  return high * high * _idle_wait_mask(env, ball_cfg).float()


def goalkeeper_idle_base_height_band(
  env: ManagerBasedRlEnv,
  target_z: float = 0.72,
  tolerance: float = 0.05,
  low_margin: float = 0.08,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward a slightly lower but not collapsed base height while waiting."""
  robot: Entity = env.scene[robot_cfg.name]
  base_z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  distance = torch.abs(base_z - target_z)
  reward = 1.0 - distance / max(tolerance, 1e-6)
  too_low = base_z < (target_z - low_margin)
  reward = torch.where(too_low, -1.0 - (target_z - low_margin - base_z), reward)
  return reward * _idle_wait_mask(env, ball_cfg).float()


def goalkeeper_idle_leg_ready_pose(
  env: ManagerBasedRlEnv,
  target_joint_pos: dict[str, float] | None = None,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Softly bias only hip/knee/ankle pitch joints toward a crouched ready pose."""
  robot: Entity = env.scene[robot_cfg.name]
  target_joint_pos = target_joint_pos or MILD_READY_JOINT_POS
  joint_names = tuple(target_joint_pos.keys())
  cache_key = "_gk_idle_leg_ready_joint_ids"
  joint_ids = getattr(env, cache_key, None)
  if joint_ids is None or joint_ids.numel() != len(joint_names):
    joint_ids = _resolve_joint_indices(robot, joint_names, env.device)
    setattr(env, cache_key, joint_ids)
  target = torch.tensor(
    [target_joint_pos[name] for name in joint_names],
    dtype=torch.float32,
    device=env.device,
  )
  err = robot.data.joint_pos[:, joint_ids] - target.view(1, -1)
  return torch.sum(err * err, dim=-1) * _idle_wait_mask(env, ball_cfg).float()


def goalkeeper_idle_base_still(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize root drift/spin while waiting for the shooter to kick."""
  robot: Entity = env.scene[robot_cfg.name]
  lin = torch.sum(robot.data.root_link_lin_vel_w[:, :2] ** 2, dim=-1)
  ang = torch.sum(robot.data.root_link_ang_vel_w[:, :2] ** 2, dim=-1)
  return (lin + ang) * _idle_wait_mask(env, ball_cfg).float()


def goalkeeper_idle_upright(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Dense prepare-stage upright reward so falls receive signal before the threshold."""
  reward = goalkeeper_posture_orientation(env, robot_cfg=robot_cfg)
  return reward * _idle_wait_mask(env, ball_cfg).float()


def goalkeeper_idle_fall_penalty(
  env: ManagerBasedRlEnv,
  limit_angle: float = 1.2217304763960306,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """One-step fall indicator used as an extra idle-stage penalty."""
  robot: Entity = env.scene[robot_cfg.name]
  upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0)
  limit_cos = torch.cos(torch.tensor(limit_angle, dtype=torch.float32, device=env.device))
  fallen = upright < limit_cos
  return fallen.float() * _idle_wait_mask(env, ball_cfg).float()


def goalkeeper_idle_alive(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  """Reward staying alive while waiting for a real incoming ball."""
  alive = ~env.termination_manager.terminated
  return alive.float() * _idle_wait_mask(env, ball_cfg).float()


def goalkeeper_idle_timeout_success(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  """Reward completing the idle waiting window without falling or ball threat."""
  success = env.termination_manager.time_outs & ~env.termination_manager.terminated
  return success.float() * _idle_wait_mask(env, ball_cfg).float()


def goalkeeper_idle_timeout(env: ManagerBasedRlEnv) -> torch.Tensor:
  """End each idle episode after its sampled prepare window."""
  cfg = getattr(env, "cfg", None)
  wait_s = _gk_get_or_init_state(env, "_gk_idle_timeout_s", float(getattr(cfg, "episode_length_s", 4.0)))
  wait_steps = torch.clamp((wait_s / float(env.step_dt)).round().long(), min=1)
  return env.episode_length_buf >= wait_steps


def goalkeeper_ball_started(
  env: ManagerBasedRlEnv,
  speed_threshold: float = 0.5,
  incoming_vx_threshold: float = -0.5,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  """Terminate the idle expert episode once the ball becomes an incoming threat."""
  ball: Entity = env.scene[ball_cfg.name]
  ball_vel = ball.data.root_link_lin_vel_w
  speed = torch.norm(ball_vel, dim=-1)
  return (speed >= speed_threshold) & (ball_vel[:, 0] < incoming_vx_threshold)
