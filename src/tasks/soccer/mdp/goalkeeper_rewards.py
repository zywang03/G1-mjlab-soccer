"""Goalkeeper-specific reward functions for training.

Implements the compact no-AMP task-reward subset used by this repository's
Humanoid-Goalkeeper ablation. Regularization terms (action_rate,
joint_pos_limits, etc.) are reused from existing modules.

State tracking follows the pattern in training_rewards.py: per-environment
tensors stored on the env object via _gk_get_or_init_state, reset via a
dedicated reset event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.entity import Entity


def _gk_get_or_init_state(
  env: ManagerBasedRlEnv, name: str, default: float, dtype=torch.float32,
) -> torch.Tensor:
  t = getattr(env, name, None)
  if t is None or t.shape[0] != env.num_envs:
    t = torch.full((env.num_envs,), default, dtype=dtype, device=env.device)
    setattr(env, name, t)
  return t


def _robot_tensor_device(robot: Entity):
  device = None
  for attr in ("body_link_pos_w", "body_link_lin_vel_w", "body_link_quat_w", "root_link_pos_w"):
    tensor = getattr(robot.data, attr, None)
    if tensor is not None:
      device = tensor.device
      break
  if device is None:
    device = "cpu"
  return device


def _resolve_ee_indices(
  env: ManagerBasedRlEnv,
  robot: Entity,
  ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  """Look up body indices once and reuse them across reward calls."""
  device = _robot_tensor_device(robot)
  cache = getattr(env, "_gk_body_index_cache", None)
  if cache is None:
    cache = {}
    setattr(env, "_gk_body_index_cache", cache)
  key = (robot.__class__.__name__, id(robot), tuple(ee_body_names))
  cached = cache.get(key)
  if cached is None or cached.device != torch.device(device):
    cached = torch.as_tensor(
      robot.find_bodies(ee_body_names, preserve_order=True)[0],
      device=device,
      dtype=torch.long,
    )
    cache[key] = cached
  return cached


def _resolve_joint_indices(
  env: ManagerBasedRlEnv,
  robot: Entity,
  joint_names: tuple[str, ...],
) -> torch.Tensor:
  device = robot.data.joint_pos.device
  cache = getattr(env, "_gk_joint_index_cache", None)
  if cache is None:
    cache = {}
    setattr(env, "_gk_joint_index_cache", cache)
  key = (robot.__class__.__name__, id(robot), tuple(joint_names))
  cached = cache.get(key)
  if cached is None or cached.device != device:
    cached = torch.as_tensor(
      robot.find_joints(joint_names, preserve_order=True)[0],
      device=device,
      dtype=torch.long,
    )
    cache[key] = cached
  return cached


def _goal_frame_mask(
  ball_pos: torch.Tensor,
  goal_half_width: float,
  goal_height: float,
) -> torch.Tensor:
  return (torch.abs(ball_pos[:, 1]) <= goal_half_width) & (ball_pos[:, 2] <= goal_height)


def _gk_region_tensor(
  env: ManagerBasedRlEnv,
  device: torch.device | str | None = None,
) -> torch.Tensor:
  if device is None:
    device = env.device
  region = getattr(env, "_gk_region", None)
  if region is None or region.shape[0] != env.num_envs:
    return torch.zeros(env.num_envs, dtype=torch.long, device=device)
  return region.to(device=device, dtype=torch.long)


def _goalkeeper_high_ball_mask(
  env: ManagerBasedRlEnv,
  target_or_ball_pos: torch.Tensor,
  high_ball_region_ids: tuple[int, ...],
  high_ball_target_z_threshold: float | None,
) -> torch.Tensor:
  region = _gk_region_tensor(env, device=target_or_ball_pos.device)
  high_ball = torch.zeros(env.num_envs, dtype=torch.bool, device=target_or_ball_pos.device)
  for region_id in high_ball_region_ids:
    high_ball = high_ball | (region == int(region_id))
  if high_ball_target_z_threshold is not None:
    high_ball = high_ball | (target_or_ball_pos[:, 2] >= float(high_ball_target_z_threshold))
  return high_ball


def _selected_hand_distance_to_position(
  env: ManagerBasedRlEnv,
  robot: Entity,
  target_pos: torch.Tensor,
  ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  selected = _region_selected_ee_indices(env, robot, ee_body_names)
  batch = torch.arange(env.num_envs, device=target_pos.device)
  ee_pos = robot.data.body_link_pos_w[batch, selected]
  return torch.norm(ee_pos - target_pos, dim=-1)


def goalkeeper_dynamic_target_pos(
  env: ManagerBasedRlEnv,
  approach_distance_threshold: float = 0.8,
  goal_line_x: float = 0.0,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Dynamic interception target from the paper's p_land / p_ball switch.

  Far from the goal line, the goalkeeper should move toward the sampled landing
  point. Near the line, it should track the actual ball for precise contact.
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  ball_pos = ball.data.root_link_pos_w
  landing_pos = getattr(env, "_gk_ball_end_pos", None)
  if landing_pos is None or landing_pos.shape[0] != env.num_envs:
    landing_pos = ball_pos

  ball_vel = ball.data.root_link_lin_vel_w
  slowed = ball_vel[:, 0] > -0.1
  close = (ball_pos[:, 0] < 0.5) & (ball_pos[:, 0] > 0.1) & (~slowed)
  official_target = torch.where(close.unsqueeze(-1), ball_pos, landing_pos)
  official_target_x = torch.clamp(
    official_target[:, 0],
    min=robot.data.root_link_pos_w[:, 0] + 0.1,
    max=robot.data.root_link_pos_w[:, 0] + 1.0,
  )
  official_target = official_target.clone()
  official_target[:, 0] = official_target_x
  del approach_distance_threshold, goal_line_x
  return official_target


def _region_selected_ee_indices(
  env: ManagerBasedRlEnv,
  robot: Entity,
  ee_body_names: tuple[str, ...],
  low_target_z: float = 0.35,
) -> torch.Tensor:
  """Choose the paper's region-conditioned limb for each environment.

  Region layout in settings.yaml: 0/2/4 are positive-y, 1/3/5 are negative-y.
  On the G1 asset, left limbs are positive-y and right limbs are negative-y.
  If feet are present in ``ee_body_names``, genuinely low targets use the
  matching foot.  The numeric region ID can also identify a motion prior
  (e.g. leftstep/rightstep), so limb selection must key off target height rather
  than treating every region >= 4 as a low kick.
  """
  all_indices = _resolve_ee_indices(env, robot, ee_body_names)
  name_to_pos = {name: index for index, name in enumerate(ee_body_names)}
  left_hand = name_to_pos.get("left_wrist_yaw_link", 0)
  right_hand = name_to_pos.get("right_wrist_yaw_link", min(1, len(ee_body_names) - 1))

  region = getattr(env, "_gk_region", None)
  if region is None or region.shape[0] != env.num_envs:
    region = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  else:
    region = region.to(device=env.device, dtype=torch.long)
  left_side = (region % 2) == 0
  selected_pos = torch.where(
    left_side,
    torch.full_like(region, left_hand),
    torch.full_like(region, right_hand),
  )
  del low_target_z
  return all_indices[selected_pos]


def goalkeeper_region_motion_modulation(
  env: ManagerBasedRlEnv,
  lateral_speed_scale: float = 0.8,
  vertical_speed_scale: float = 0.6,
  lateral_gain: float = 0.2,
  vertical_gain: float = 0.2,
  up_target_z: float = 1.2,
  approach_distance_threshold: float = 0.8,
  goal_line_x: float = 0.0,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Official region-conditioned velocity multiplier."""
  robot: Entity = env.scene[robot_cfg.name]
  base_vel = robot.data.root_link_lin_vel_w
  region = getattr(env, "_gk_region", None)
  if region is None or region.shape[0] != env.num_envs:
    region = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  else:
    region = region.to(device=env.device, dtype=torch.long)
  curriculum = float(getattr(env, "_gk_curriculum_update", 0))
  jump_scale = 3.0 + 3.0 * curriculum
  multiplier = torch.ones(env.num_envs, dtype=base_vel.dtype, device=env.device)
  left_step = (region == 0) | (region == 4)
  right_step = (region == 1) | (region == 5)
  jump = (region == 2) | (region == 3)
  multiplier = torch.where(
    left_step,
    1.0 + 3.0 * torch.clamp(base_vel[:, 1], min=0.0, max=3.0),
    multiplier,
  )
  multiplier = torch.where(
    right_step,
    1.0 - 3.0 * torch.clamp(base_vel[:, 1], min=-3.0, max=0.0),
    multiplier,
  )
  multiplier = torch.where(
    jump,
    1.0 + jump_scale * torch.clamp(base_vel[:, 2], min=0.0, max=3.0),
    multiplier,
  )
  ball: Entity = env.scene[ball_cfg.name]
  behind = ball.data.root_link_pos_w[:, 0] <= 0.0
  multiplier = torch.where(behind, torch.full_like(multiplier, 2.0), multiplier)
  del lateral_speed_scale, vertical_speed_scale, lateral_gain, vertical_gain, up_target_z
  del approach_distance_threshold, goal_line_x
  return multiplier


def goalkeeper_selected_hand_target_distance(
  env: ManagerBasedRlEnv,
  approach_distance_threshold: float = 0.8,
  goal_line_x: float = 0.0,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ee_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  ),
) -> torch.Tensor:
  """Distance from the region-selected hand to the dynamic target."""
  robot: Entity = env.scene[robot_cfg.name]
  target = goalkeeper_dynamic_target_pos(
    env,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )
  selected = _region_selected_ee_indices(env, robot, ee_body_names)
  batch = torch.arange(env.num_envs, device=env.device)
  ee_pos = robot.data.body_link_pos_w[batch, selected]
  return torch.norm(ee_pos - target, dim=-1).unsqueeze(-1)


# -- Reset event ---------------------------------------------------------------


def _reset_gk_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
) -> None:
  """Reset per-environment GK tracking state on episode start."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  for key in (
    "_gk_max_ball_speed",
    "_gk_block_awarded",
    "_gk_contact_block_awarded",
    "_gk_save_success",
    "_gk_success_flag",
    "_gk_stop_flag",
    "_gk_jump_flag",
    "_gk_invalid_high_ball_block",
    "_gk_invalid_high_ball_penalty_awarded",
  ):
    t = getattr(env, key, None)
    if t is not None and t.shape[0] == env.num_envs:
      t[env_ids] = 0.0 if "speed" in key else False
  prev_ball_pos = getattr(env, "_gk_prev_ball_pos", None)
  if prev_ball_pos is not None and prev_ball_pos.shape[:2] == (env.num_envs, 3):
    prev_ball_pos[env_ids] = torch.nan


# -- Reward functions ----------------------------------------------------------


def goalkeeper_ee_reach(
  env: ManagerBasedRlEnv,
  std: float = 0.3,
  catch_distance: float = 0.20,
  sigmoid_scale: float = 12.0,
  approach_distance_threshold: float = 0.8,
  goal_line_x: float = 0.0,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ee_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  ),
  high_ball_region_ids: tuple[int, ...] = (2, 3),
  high_ball_target_z_threshold: float | None = 1.2,
  high_ball_reward_scale: float = 1.5,
) -> torch.Tensor:
  """Region-conditioned end-effector reach reward to the dynamic target.

  The paper's task reward uses the landing point while the ball is far away,
  switches to the actual ball near the robot, and computes distance for the
  region-selected hand. This gives an early directional signal instead of only
  rewarding accidental proximity to the live ball.
  """
  robot: Entity = env.scene[robot_cfg.name]

  target = goalkeeper_dynamic_target_pos(
    env,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )
  selected = _region_selected_ee_indices(env, robot, ee_body_names)
  batch = torch.arange(env.num_envs, device=env.device)
  ee_pos = robot.data.body_link_pos_w[batch, selected]

  dist = torch.norm(ee_pos - target, dim=-1)
  ball: Entity = env.scene[ball_cfg.name]
  base_pos = robot.data.root_link_pos_w
  y_error = torch.abs(target[:, 1] - base_pos[:, 1])
  z_error = torch.abs(target[:, 2] - base_pos[:, 2])
  aside_goal = torch.clamp(1.0 - y_error / 1.5, min=0.0, max=1.0)
  vertical_goal = torch.clamp(1.0 - z_error / 1.5, min=0.0, max=1.0)
  phase1_rew = 0.5 * (aside_goal + vertical_goal)
  curriculumsigma = float(getattr(env, "_gk_curriculumsigma", sigmoid_scale))
  reach = 1.0 - 1.0 / (1.0 + torch.exp(-curriculumsigma * (dist - catch_distance)))
  modulation = goalkeeper_region_motion_modulation(
    env,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )
  task_rew = reach * modulation
  far_ball = ball.data.root_link_pos_w[:, 0] > 1.5
  task_rew = torch.where(far_ball, phase1_rew, task_rew)
  high_ball = _goalkeeper_high_ball_mask(
    env,
    target,
    high_ball_region_ids=high_ball_region_ids,
    high_ball_target_z_threshold=high_ball_target_z_threshold,
  )
  task_rew = torch.where(
    high_ball,
    task_rew * float(high_ball_reward_scale),
    task_rew,
  )
  upright = 1.0 - torch.clamp(torch.sum(robot.data.projected_gravity_b[:, :2] ** 2, dim=-1), 0.0, 1.0)
  del std
  return task_rew * upright


def goalkeeper_success(
  env: ManagerBasedRlEnv,
  strict_distance: float = 0.15,
  approach_distance_threshold: float = 0.8,
  goal_line_x: float = 0.0,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ee_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  ),
  high_ball_region_ids: tuple[int, ...] = (2, 3),
  high_ball_target_z_threshold: float | None = 1.2,
  high_ball_success_scale: float = 2.0,
) -> torch.Tensor:
  dist = goalkeeper_selected_hand_target_distance(
    env,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
    ee_body_names=ee_body_names,
  ).squeeze(-1)
  success_flag = _gk_get_or_init_state(env, "_gk_success_flag", 0.0)
  reward = (success_flag + 1.0) * (dist < strict_distance).float()
  ball: Entity = env.scene[ball_cfg.name]
  high_ball = _goalkeeper_high_ball_mask(
    env,
    ball.data.root_link_pos_w,
    high_ball_region_ids=high_ball_region_ids,
    high_ball_target_z_threshold=high_ball_target_z_threshold,
  )
  return torch.where(high_ball, reward * float(high_ball_success_scale), reward)


def goalkeeper_target_alignment(
  env: ManagerBasedRlEnv,
  std: float = 0.45,
  approach_distance_threshold: float = 0.8,
  goal_line_x: float = 0.0,
  region_conditioned: bool = True,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ee_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ),
) -> torch.Tensor:
  """Reward the region-selected end-effector matching the dynamic target in y/z.

  The target uses the sampled landing point while the ball is far away and the
  actual ball position near the robot. Region conditioning prevents the nearest
  wrong-side limb from receiving the task reward.
  """
  target = goalkeeper_dynamic_target_pos(
    env,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )

  robot: Entity = env.scene[robot_cfg.name]
  if region_conditioned:
    selected = _region_selected_ee_indices(env, robot, ee_body_names)
    batch = torch.arange(env.num_envs, device=env.device)
    ee_pos_yz = robot.data.body_link_pos_w[batch, selected, 1:3]
    best_dist_sq = torch.sum((ee_pos_yz - target[:, 1:3]) ** 2, dim=-1)
  else:
    ee_indices = _resolve_ee_indices(env, robot, ee_body_names)
    ee_pos_yz = robot.data.body_link_pos_w[:, ee_indices, 1:3]
    target_yz = target[:, 1:3].unsqueeze(1)
    dist_sq_yz = torch.sum((ee_pos_yz - target_yz) ** 2, dim=-1)
    best_dist_sq = torch.min(dist_sq_yz, dim=-1).values
  return torch.exp(-best_dist_sq / (std * std))


def goalkeeper_stop_ball(
  env: ManagerBasedRlEnv,
  velocity_drop_threshold: float = 2.0,
  behind_robot_x_threshold: float = 0.0,
  goal_x: float = -0.5,
  goal_half_width: float = 1.5,
  goal_height: float = 1.8,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ee_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  ),
  high_ball_region_ids: tuple[int, ...] = (2, 3),
  high_ball_target_z_threshold: float | None = 1.2,
  high_ball_hand_gate_radius: float = 0.35,
) -> torch.Tensor:
  """One-shot stop reward, gated by hand proximity for high-ball regions."""
  del goal_x, goal_half_width, goal_height
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]

  ball_pos = ball.data.root_link_pos_w  # (B, 3)
  ball_vel = ball.data.root_link_lin_vel_w  # (B, 3)
  initial_vx = _gk_get_or_init_state(env, "_gk_initial_ball_vx", 0.0)
  uninitialized = torch.abs(initial_vx) < 1.0e-6
  initial_vx = torch.where(uninitialized, ball_vel[:, 0], initial_vx)
  setattr(env, "_gk_initial_ball_vx", initial_vx)

  stop_flag = _gk_get_or_init_state(env, "_gk_stop_flag", 0.0)
  success_flag = _gk_get_or_init_state(env, "_gk_success_flag", 0.0)
  changevel = (initial_vx - ball_vel[:, 0] > velocity_drop_threshold) & (
    ball_pos[:, 0] > behind_robot_x_threshold
  )
  high_ball = _goalkeeper_high_ball_mask(
    env,
    ball_pos,
    high_ball_region_ids=high_ball_region_ids,
    high_ball_target_z_threshold=high_ball_target_z_threshold,
  )
  hand_gate = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  if torch.any(high_ball):
    hand_distance = _selected_hand_distance_to_position(env, robot, ball_pos, ee_body_names)
    hand_gate = (~high_ball) | (hand_distance <= high_ball_hand_gate_radius)
  valid_changevel = changevel & hand_gate
  invalid_high_ball_block = (stop_flag < 0.5) & changevel & high_ball & (~hand_gate)
  invalid_flag = _gk_get_or_init_state(env, "_gk_invalid_high_ball_block", 0.0)
  invalid_flag[invalid_high_ball_block] = 1.0
  stopped = changevel | (ball_pos[:, 0] < behind_robot_x_threshold)
  success_ids = (stop_flag < 0.5) & valid_changevel
  reward = (stop_flag < 0.5).float() * valid_changevel.float()
  success_flag[success_ids] = 1.0
  stop_flag[stopped] = 1.0
  setattr(env, "_gk_success_flag", success_flag)
  setattr(env, "_gk_stop_flag", stop_flag)
  setattr(env, "_gk_invalid_high_ball_block", invalid_flag)
  return reward


def goalkeeper_penalize_high_ball_foot_block(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """One-shot proxy penalty for high-ball speed drops not made by the hands."""
  invalid_flag = _gk_get_or_init_state(env, "_gk_invalid_high_ball_block", 0.0)
  awarded = _gk_get_or_init_state(env, "_gk_invalid_high_ball_penalty_awarded", 0.0)
  eligible = (invalid_flag > 0.5) & (awarded < 0.5)
  reward = eligible.float()
  awarded[eligible] = 1.0
  setattr(env, "_gk_invalid_high_ball_penalty_awarded", awarded)
  return reward


def goalkeeper_contact_block(
  env: ManagerBasedRlEnv,
  velocity_drop_threshold: float = 1.0,
  contact_radius: float = 0.35,
  std: float = 0.35,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ee_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ),
) -> torch.Tensor:
  """Dense one-shot reward for a plausible goalkeeper-ball contact.

  This provides learning signal before the final goal-line outcome: the ball
  must be near a hand/foot and have slowed down relative to its episode max.
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]

  ball_pos = ball.data.root_link_pos_w
  ball_vel = ball.data.root_link_lin_vel_w
  current_speed = torch.norm(ball_vel, dim=-1)
  max_speed = _gk_get_or_init_state(env, "_gk_max_ball_speed", 0.0)
  max_speed = torch.maximum(max_speed, current_speed)
  setattr(env, "_gk_max_ball_speed", max_speed)

  contact_awarded = _gk_get_or_init_state(env, "_gk_contact_block_awarded", 0.0)
  ee_indices = _resolve_ee_indices(env, robot, ee_body_names)
  ee_pos = robot.data.body_link_pos_w[:, ee_indices]
  dist = torch.norm(ee_pos - ball_pos.unsqueeze(1), dim=-1)
  min_dist = torch.min(dist, dim=-1).values

  speed_drop = (max_speed - current_speed) > velocity_drop_threshold
  close = min_dist <= contact_radius
  eligible = speed_drop & close & (contact_awarded < 0.5)

  reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  if torch.any(eligible):
    ids = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
    reward[ids] = torch.exp(-(min_dist[ids] * min_dist[ids]) / (std * std))
    contact_awarded[ids] = 1.0
    setattr(env, "_gk_contact_block_awarded", contact_awarded)

  return reward


def goalkeeper_stay_on_line(
  env: ManagerBasedRlEnv,
  goal_line_x: float = 0.0,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize depth deviation from the goalkeeper line.

  The goalkeeper must be free to move laterally along the line. Penalizing y
  caused a standing-in-the-center local optimum, so this term only keeps the
  base near the goal line in x.

  Weight: -2.0 (matches paper).
  """
  robot: Entity = env.scene[robot_cfg.name]
  robot_x = robot.data.root_link_pos_w[:, 0]
  return torch.abs(robot_x - goal_line_x)


def _goalkeeper_lateral_error(
  env: ManagerBasedRlEnv,
  approach_distance_threshold: float,
  goal_line_x: float,
  ball_cfg: SceneEntityCfg,
  robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot: Entity = env.scene[robot_cfg.name]
  target = goalkeeper_dynamic_target_pos(
    env,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )
  return target[:, 1] - robot.data.root_link_pos_w[:, 1]


def goalkeeper_lateral_intercept_velocity(
  env: ManagerBasedRlEnv,
  speed_scale: float = 0.8,
  lateral_deadzone: float = 0.15,
  approach_distance_threshold: float = 0.8,
  goal_line_x: float = 0.0,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward base velocity that moves the goalkeeper laterally toward target y."""
  robot: Entity = env.scene[robot_cfg.name]
  y_error = _goalkeeper_lateral_error(
    env,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )
  direction = torch.sign(y_error)
  progress_speed = robot.data.root_link_lin_vel_w[:, 1] * direction
  reward = torch.clamp(progress_speed / speed_scale, 0.0, 1.0)
  return reward * (torch.abs(y_error) > lateral_deadzone).float()


def goalkeeper_track_lateral_velocity(
  env: ManagerBasedRlEnv,
  position_gain: float = 2.0,
  max_speed: float = 1.2,
  std: float = 0.5,
  deadzone: float = 0.1,
  command_threshold: float = 0.05,
  approach_distance_threshold: float = 0.8,
  goal_line_x: float = 0.0,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward the base tracking the explicit sidestep velocity command."""
  from src.tasks.soccer.mdp.goalkeeper_obs import goalkeeper_sidestep_command

  robot: Entity = env.scene[robot_cfg.name]
  command = goalkeeper_sidestep_command(
    env,
    position_gain=position_gain,
    max_speed=max_speed,
    deadzone=deadzone,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )
  cmd_vy = command[:, 1]
  actual_vy = robot.data.root_link_lin_vel_w[:, 1]
  reward = torch.exp(-((actual_vy - cmd_vy) ** 2) / (std * std))
  active = torch.abs(cmd_vy) > command_threshold
  return reward * active.float()


def goalkeeper_lateral_intercept_position(
  env: ManagerBasedRlEnv,
  std: float = 0.5,
  lateral_deadzone: float = 0.05,
  approach_distance_threshold: float = 0.8,
  goal_line_x: float = 0.0,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward the base y position matching the dynamic interception target y."""
  y_error = _goalkeeper_lateral_error(
    env,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )
  reward = torch.exp(-(y_error * y_error) / (std * std))
  return torch.where(torch.abs(y_error) > lateral_deadzone, reward, torch.ones_like(reward))


def goalkeeper_no_retreat(
  env: ManagerBasedRlEnv,
  goal_line_x: float = 0.0,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize retreating behind the goal line.

  Paper's 'noretreat' — discourages the robot from moving backward into the
  goal. In this goalkeeper setup, the ball comes from +x and the goal is at
  -x, so retreating means decreasing x past the goal line.

  Weight: -2.0 (matches paper).
  """
  robot: Entity = env.scene[robot_cfg.name]
  robot_x = robot.data.root_link_pos_w[:, 0]
  retreat = torch.clamp(goal_line_x - robot_x, min=0.0)
  return retreat * retreat


def _goalkeeper_foot_contact_mask(
  env: ManagerBasedRlEnv,
  contact_sensor_name: str,
  num_feet: int,
) -> torch.Tensor:
  contact = _goalkeeper_get_sensor(env, contact_sensor_name)
  if contact is None:
    return torch.zeros((env.num_envs, num_feet), dtype=torch.bool, device=env.device)

  found = getattr(contact.data, "found", None)
  if found is not None:
    found = found.to(device=env.device)
    if found.ndim == 1:
      found = found.unsqueeze(-1)
    if found.shape[1] >= num_feet:
      return found[:, :num_feet] > 0.0
    if found.shape[1] == 1:
      return (found[:, :1] > 0.0).expand(-1, num_feet)

  force = getattr(contact.data, "force", None)
  if force is not None:
    force = force.to(device=env.device)
    if force.ndim == 2:
      force = force.unsqueeze(1)
    force_found = torch.norm(force, dim=-1) > 1.0
    if force_found.shape[1] >= num_feet:
      return force_found[:, :num_feet]
    if force_found.shape[1] == 1:
      return force_found[:, :1].expand(-1, num_feet)

  return torch.zeros((env.num_envs, num_feet), dtype=torch.bool, device=env.device)


def _goalkeeper_get_sensor(env: ManagerBasedRlEnv, sensor_name: str):
  sensors = getattr(env.scene, "sensors", None)
  if hasattr(sensors, "get"):
    sensor = sensors.get(sensor_name, None)
    if sensor is not None:
      return sensor
  try:
    return env.scene[sensor_name]
  except (KeyError, TypeError, AttributeError):
    return None


def _goalkeeper_lateral_active(
  env: ManagerBasedRlEnv,
  lateral_deadzone: float,
  approach_distance_threshold: float,
  goal_line_x: float,
  ball_cfg: SceneEntityCfg,
  robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
  y_error = _goalkeeper_lateral_error(
    env,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )
  return (torch.abs(y_error) > lateral_deadzone).float()


def goalkeeper_feet_slippage(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  foot_body_names: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ),
  contact_sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Official positive foot-slippage reward: exp(-10 * contact foot speed)."""
  robot: Entity = env.scene[robot_cfg.name]
  foot_indices = _resolve_ee_indices(env, robot, foot_body_names)
  foot_vel_w = robot.data.body_link_lin_vel_w[:, foot_indices]

  contact_found = _goalkeeper_foot_contact_mask(
    env,
    contact_sensor_name=contact_sensor_name,
    num_feet=len(foot_body_names),
  )
  contactvel = torch.sum(torch.norm(foot_vel_w, dim=-1) * contact_found.float(), dim=1)
  return torch.exp(contactvel * -10.0)


def goalkeeper_feet_orientation(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  foot_body_names: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ),
) -> torch.Tensor:
  """Official feetorientation reward."""
  robot: Entity = env.scene[robot_cfg.name]
  foot_indices = _resolve_ee_indices(env, robot, foot_body_names)
  foot_quat = robot.data.body_link_quat_w[:, foot_indices]
  gravity = torch.zeros(env.num_envs, len(foot_body_names), 3, device=env.device)
  gravity[..., 2] = -1.0
  from mjlab.utils.lab_api.math import quat_apply_inverse

  local_gravity = quat_apply_inverse(foot_quat, gravity)
  feet_orientation = torch.sum(local_gravity[..., :2] ** 2, dim=(1, 2))
  return torch.exp(feet_orientation * -5.0)


def goalkeeper_success_land(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  contact_sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Official successland for jump regions."""
  robot: Entity = env.scene[robot_cfg.name]
  jump_flag = _gk_get_or_init_state(env, "_gk_jump_flag", 0.0)
  jump_flag[:] = torch.logical_or(jump_flag > 0.5, robot.data.root_link_pos_w[:, 2] > 1.0).float()
  contact_found = _goalkeeper_foot_contact_mask(env, contact_sensor_name, 2)
  two_feet = torch.sum(contact_found.float(), dim=1) >= 2.0
  one_foot = (torch.sum(contact_found.float(), dim=1) == 1.0) & (jump_flag > 0.5)
  region = getattr(env, "_gk_region", None)
  if region is None or region.shape[0] != env.num_envs:
    region = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  else:
    region = region.to(device=env.device, dtype=torch.long)
  jump_ids = (region == 2) | (region == 3)
  reward = jump_flag + two_feet.float() * (jump_flag > 0.5).float() * 5.0 - one_foot.float()
  return reward * jump_ids.float()


def goalkeeper_penalize_sharp_contact(
  env: ManagerBasedRlEnv,
  max_contact_force: float = 100.0,
  contact_sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  contact = _goalkeeper_get_sensor(env, contact_sensor_name)
  if contact is None or not hasattr(contact.data, "force"):
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
  force = contact.data.force.to(device=env.device)
  if force.ndim == 2:
    force = force.unsqueeze(1)
  return (torch.mean(torch.norm(force, dim=-1), dim=-1) > max_contact_force).float()


def goalkeeper_penalize_knee_height(
  env: ManagerBasedRlEnv,
  min_height: float = 0.15,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  knee_body_names: tuple[str, ...] = (
    "left_knee_link",
    "right_knee_link",
  ),
) -> torch.Tensor:
  robot: Entity = env.scene[robot_cfg.name]
  try:
    knee_indices = _resolve_ee_indices(env, robot, knee_body_names)
  except (KeyError, ValueError):
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
  knee_height = robot.data.body_link_pos_w[:, knee_indices, 2]
  return (torch.min(knee_height, dim=-1).values < min_height).float()


def goalkeeper_feet_shuffle(
  env: ManagerBasedRlEnv,
  lateral_deadzone: float = 0.15,
  min_lateral_speed: float = 0.2,
  approach_distance_threshold: float = 0.8,
  goal_line_x: float = 0.0,
  contact_sensor_name: str = "feet_ground_contact",
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize double-stance lateral base motion during active interception.

  This targets the failure mode where the body translates sideways while both
  feet remain planted, which looks like ground shuffling instead of stepping.
  Single-stance steps and near-target standing are left alone.
  """
  robot: Entity = env.scene[robot_cfg.name]
  contact_found = _goalkeeper_foot_contact_mask(
    env,
    contact_sensor_name=contact_sensor_name,
    num_feet=2,
  )
  double_stance = torch.sum(contact_found.float(), dim=1) >= 2.0
  active = _goalkeeper_lateral_active(
    env,
    lateral_deadzone=lateral_deadzone,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  ).bool()
  lateral_speed = torch.abs(robot.data.root_link_lin_vel_w[:, 1])
  moving = lateral_speed > min_lateral_speed
  return lateral_speed * lateral_speed * double_stance.float() * active.float() * moving.float()


def goalkeeper_feet_air_time(
  env: ManagerBasedRlEnv,
  threshold: float = 0.35,
  lateral_deadzone: float = 0.15,
  approach_distance_threshold: float = 0.8,
  goal_line_x: float = 0.0,
  contact_sensor_name: str = "feet_ground_contact",
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward clean single-stance timing when a lateral interception is needed."""
  contact = _goalkeeper_get_sensor(env, contact_sensor_name)
  if contact is None:
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

  air_time = getattr(contact.data, "current_air_time", None)
  contact_time = getattr(contact.data, "current_contact_time", None)
  if air_time is None or contact_time is None:
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

  air_time = air_time.to(device=env.device)
  contact_time = contact_time.to(device=env.device)
  in_contact = contact_time > 0.0
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  swing_air_time = torch.max(torch.where(~in_contact, air_time, 0.0), dim=1)[0]
  reward = torch.clamp(threshold - torch.abs(swing_air_time - threshold), min=0.0)
  reward = reward * single_stance.float()
  active = _goalkeeper_lateral_active(
    env,
    lateral_deadzone=lateral_deadzone,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )
  return reward * active


def goalkeeper_feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float = 0.12,
  height_std: float = 0.04,
  lateral_deadzone: float = 0.15,
  approach_distance_threshold: float = 0.8,
  goal_line_x: float = 0.0,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  foot_body_names: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ),
  contact_sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Reward swing feet reaching a useful clearance height during lateral moves."""
  robot: Entity = env.scene[robot_cfg.name]
  foot_indices = _resolve_ee_indices(env, robot, foot_body_names)
  foot_pos = robot.data.body_link_pos_w[:, foot_indices]
  foot_vel = robot.data.body_link_lin_vel_w[:, foot_indices]
  contact_found = _goalkeeper_foot_contact_mask(
    env,
    contact_sensor_name=contact_sensor_name,
    num_feet=len(foot_body_names),
  )

  swing = (~contact_found).float()
  height_error = foot_pos[:, :, 2] - target_height
  height_score = torch.exp(-(height_error * height_error) / (height_std * height_std))
  swing_speed = torch.norm(foot_vel[:, :, :2], dim=-1).clamp(max=1.0)
  reward = torch.sum(height_score * swing_speed * swing, dim=-1)
  active = _goalkeeper_lateral_active(
    env,
    lateral_deadzone=lateral_deadzone,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )
  return reward * active


def goalkeeper_posture_orientation(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward upright posture via projected gravity.

  Paper's 'postorientation' — returns ``-projected_gravity[:, 2]`` clamped
  to [0, 1]. In mjlab/MuJoCo, a perfectly upright robot observes gravity as
  local -z, so projected_gravity_z ≈ -1. As the robot tilts, this value moves
  toward 0.

  Weight: 3.0 (matches paper).
  """
  robot: Entity = env.scene[robot_cfg.name]
  grav_z = robot.data.projected_gravity_b[:, 2]
  return torch.clamp(-grav_z, 0.0, 1.0)


def _goalkeeper_post_task_mask(
  env: ManagerBasedRlEnv,
  goal_x: float,
  velocity_drop_threshold: float,
  ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Official post-task gate: ball behind line or slowed sharply in front."""
  ball: Entity = env.scene[ball_cfg.name]
  ball_pos = ball.data.root_link_pos_w
  ball_vel = ball.data.root_link_lin_vel_w
  current_speed = torch.norm(ball_vel, dim=-1)
  max_speed = _gk_get_or_init_state(env, "_gk_max_ball_speed", 0.0)
  stopped_in_front = (max_speed - current_speed > velocity_drop_threshold) & (ball_pos[:, 0] > goal_x)
  behind = ball_pos[:, 0] < goal_x
  return stopped_in_front | behind


def goalkeeper_post_orientation(
  env: ManagerBasedRlEnv,
  goal_x: float = -0.5,
  velocity_drop_threshold: float = 2.0,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  """Paper's postorientation: uprightness only after the shot resolves."""
  robot: Entity = env.scene[robot_cfg.name]
  grav_xy_sq = torch.sum(robot.data.projected_gravity_b[:, :2] ** 2, dim=-1)
  reward = torch.exp(-3.0 * grav_xy_sq)
  mask = _goalkeeper_post_task_mask(env, goal_x, velocity_drop_threshold, ball_cfg)
  return reward * mask.float()


def goalkeeper_post_ang_vel(
  env: ManagerBasedRlEnv,
  goal_x: float = -0.5,
  velocity_drop_threshold: float = 2.0,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  """Paper's postangvel: low roll/pitch angular velocity post task."""
  robot: Entity = env.scene[robot_cfg.name]
  ang_vel_xy_sq = torch.sum(robot.data.root_link_ang_vel_b[:, :2] ** 2, dim=-1)
  reward = torch.exp(-3.0 * ang_vel_xy_sq)
  mask = _goalkeeper_post_task_mask(env, goal_x, velocity_drop_threshold, ball_cfg)
  return reward * mask.float()


def goalkeeper_post_lin_vel(
  env: ManagerBasedRlEnv,
  goal_x: float = -0.5,
  velocity_drop_threshold: float = 2.0,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  """Paper's postlinvel: low forward/backward base velocity post task."""
  robot: Entity = env.scene[robot_cfg.name]
  lin_vel_x_sq = robot.data.root_link_lin_vel_w[:, 0] ** 2
  reward = torch.exp(-3.0 * lin_vel_x_sq)
  mask = _goalkeeper_post_task_mask(env, goal_x, velocity_drop_threshold, ball_cfg)
  return reward * mask.float()


def _goalkeeper_named_joint_deviation(
  env: ManagerBasedRlEnv,
  joint_names: tuple[str, ...],
  robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
  from src.tasks.soccer.mdp.goalkeeper_obs import _GK_DEFAULT_JOINT_POS

  robot: Entity = env.scene[robot_cfg.name]
  joint_ids = _resolve_joint_indices(env, robot, joint_names)
  default_cache = getattr(env, "_gk_joint_default_cache", None)
  if default_cache is None:
    default_cache = {}
    setattr(env, "_gk_joint_default_cache", default_cache)
  default_key = (tuple(joint_names), robot.data.joint_pos.device, robot.data.joint_pos.dtype)
  default = default_cache.get(default_key)
  if default is None:
    default = torch.as_tensor(
      [_GK_DEFAULT_JOINT_POS[name] for name in joint_names],
      device=robot.data.joint_pos.device,
      dtype=robot.data.joint_pos.dtype,
    )
    default_cache[default_key] = default
  error = robot.data.joint_pos[:, joint_ids] - default
  return torch.sum(error * error, dim=-1)


def goalkeeper_post_upper_dof_pos(
  env: ManagerBasedRlEnv,
  goal_x: float = -0.5,
  velocity_drop_threshold: float = 2.0,
  joint_names: tuple[str, ...] = (
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
  ),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  """Paper's postupperdofpos: recover upper limb joints post task."""
  mse = _goalkeeper_named_joint_deviation(env, joint_names, robot_cfg)
  reward = torch.exp(-mse)
  mask = _goalkeeper_post_task_mask(env, goal_x, velocity_drop_threshold, ball_cfg)
  return reward * mask.float()


def goalkeeper_post_waist_dof_pos(
  env: ManagerBasedRlEnv,
  goal_x: float = -0.5,
  velocity_drop_threshold: float = 2.0,
  joint_names: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
  ),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  """Paper's postwaistdofpos: recover waist joints post task."""
  mse = _goalkeeper_named_joint_deviation(env, joint_names, robot_cfg)
  reward = torch.exp(-3.0 * mse)
  mask = _goalkeeper_post_task_mask(env, goal_x, velocity_drop_threshold, ball_cfg)
  return reward * mask.float()


def goalkeeper_post_task_stability(
  env: ManagerBasedRlEnv,
  goal_x: float = -0.5,
  target_base_height: float = 0.8,
  height_std: float = 0.2,
  ang_vel_std: float = 1.0,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
  """Reward whole-body recovery after contact attempts or shot resolution.

  In the paper, stability terms are post-task rewards multiplied by the
  ``ball stopped`` indicator. In this implementation we also enable the
  recovery signal after a plausible contact/block attempt or after the ball has
  crossed the goal plane. Otherwise failed-but-upright recoveries receive no
  learning signal, which encouraged late-stage policies to lunge and fall.
  """
  ball: Entity = env.scene[ball_cfg.name]
  ball_pos = ball.data.root_link_pos_w
  robot: Entity = env.scene[robot_cfg.name]

  save_success = _gk_get_or_init_state(env, "_gk_save_success", 0.0)
  contact_awarded = _gk_get_or_init_state(env, "_gk_contact_block_awarded", 0.0)
  block_awarded = _gk_get_or_init_state(env, "_gk_block_awarded", 0.0)
  resolved = (ball_pos[:, 0] <= goal_x) | (block_awarded > 0.5)
  recovery_window = (save_success > 0.5) | (contact_awarded > 0.5) | resolved
  upright = goalkeeper_posture_orientation(env, robot_cfg=robot_cfg)
  height_error = robot.data.root_link_pos_w[:, 2] - target_base_height
  height_score = torch.exp(-(height_error * height_error) / (height_std * height_std))
  ang_vel_xy = torch.sum(robot.data.root_link_ang_vel_b[:, :2] ** 2, dim=-1)
  ang_vel_score = torch.exp(-ang_vel_xy / (ang_vel_std * ang_vel_std))
  return upright * height_score * ang_vel_score * recovery_window.float()


def goalkeeper_stance_joint_deviation(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize deviation from the reference goalkeeper standing stance."""
  from src.tasks.soccer.mdp.goalkeeper_obs import _REF_DEFAULT_DOF_POS

  robot: Entity = env.scene[robot_cfg.name]
  jnt_ids = robot_cfg.joint_ids
  default = torch.tensor(
    _REF_DEFAULT_DOF_POS,
    device=robot.data.joint_pos.device,
    dtype=robot.data.joint_pos.dtype,
  )[jnt_ids]
  error = robot.data.joint_pos[:, jnt_ids] - default
  return torch.sum(error * error, dim=-1)


def goalkeeper_action_l2_clip(
  env: ManagerBasedRlEnv,
  max_value: float = 100.0,
) -> torch.Tensor:
  """Penalize absolute action magnitude, clamped for PPO stability."""
  action = env.action_manager.action
  return torch.sum(action * action, dim=1).clamp(max=max_value)


def goalkeeper_action_saturation(
  env: ManagerBasedRlEnv,
  threshold: float = 0.8,
  max_value: float = 100.0,
) -> torch.Tensor:
  """Penalize raw policy commands that push near or beyond the action clip."""
  action = env.action_manager.action
  excess = torch.clamp(torch.abs(action) - threshold, min=0.0)
  return torch.sum(excess * excess, dim=1).clamp(max=max_value)


def goalkeeper_action_smoothness(
  env: ManagerBasedRlEnv,
  max_value: float = 100.0,
) -> torch.Tensor:
  """Paper's second-order action smoothness penalty."""
  action_acc = (
    env.action_manager.action
    - 2.0 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  return torch.sum(action_acc * action_acc, dim=1).clamp(max=max_value)


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


def goalkeeper_dof_acc(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> torch.Tensor:
  robot: Entity = env.scene[robot_cfg.name]
  joint_acc = getattr(robot.data, "joint_acc", None)
  if joint_acc is None:
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
  return torch.sum(joint_acc[:, robot_cfg.joint_ids] ** 2, dim=1)


def goalkeeper_torques(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> torch.Tensor:
  robot: Entity = env.scene[robot_cfg.name]
  try:
    torques = robot.data.joint_torques
  except (RuntimeError, AttributeError):
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
  return torch.sum(torques[:, robot_cfg.joint_ids] ** 2, dim=1)


def goalkeeper_dof_vel(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> torch.Tensor:
  robot: Entity = env.scene[robot_cfg.name]
  return torch.sum(robot.data.joint_vel[:, robot_cfg.joint_ids] ** 2, dim=1)


def goalkeeper_dof_vel_limits(
  env: ManagerBasedRlEnv,
  soft_limit: float = 1.0,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> torch.Tensor:
  robot: Entity = env.scene[robot_cfg.name]
  limits = getattr(robot.data, "soft_joint_vel_limits", None)
  if limits is None:
    limits = getattr(robot.data, "joint_vel_limits", None)
  if limits is None:
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
  selected = limits[:, robot_cfg.joint_ids] if limits.ndim == 2 else limits[robot_cfg.joint_ids]
  return torch.sum(
    (torch.abs(robot.data.joint_vel[:, robot_cfg.joint_ids]) - selected * soft_limit).clamp(min=0.0),
    dim=1,
  )


def goalkeeper_torque_limits(
  env: ManagerBasedRlEnv,
  soft_limit: float = 1.0,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> torch.Tensor:
  robot: Entity = env.scene[robot_cfg.name]
  try:
    torques = robot.data.joint_torques
  except (RuntimeError, AttributeError):
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
  limits = getattr(robot.data, "joint_effort_limits", None)
  if limits is None:
    limits = getattr(robot.data, "effort_limits", None)
  if limits is None:
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
  selected = limits[:, robot_cfg.joint_ids] if limits.ndim == 2 else limits[robot_cfg.joint_ids]
  return torch.sum((torch.abs(torques[:, robot_cfg.joint_ids]) - selected * soft_limit).clamp(min=0.0), dim=1)


def goalkeeper_deviation_waist_pitch_joint(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  return _goalkeeper_named_joint_deviation(
    env,
    ("waist_pitch_joint",),
    robot_cfg,
  )
