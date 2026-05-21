"""Termination functions for the soccer task.

Includes basic terminations (timeout, fell-over) and motion-reference
terminations for shooter evaluation (anchor height, anchor orientation,
end-effector position).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# ---------------------------------------------------------------------------
# Basic terminations
# ---------------------------------------------------------------------------


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity_ = asset.data.projected_gravity_b
  return torch.acos(-projected_gravity_[:, 2]).abs() > limit_angle


# ---------------------------------------------------------------------------
# Motion-reference terminations (shooter — disabled by default)
# ---------------------------------------------------------------------------


def bad_anchor_pos_z(
  env: ManagerBasedRlEnv,
  threshold: float = 0.25,
  command_name: str = "motion_command",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when anchor (torso) height deviates too far from motion reference.

  Paper: |z_robot - z_ref| > 0.25m. Returns False if no motion command.
  """
  cmd = getattr(env, command_name, None)
  if cmd is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  asset: Entity = env.scene[asset_cfg.name]
  torso_idx = asset.body_names.index("torso_link")
  robot_z = asset.data.body_link_pos_w[:, torso_idx, 2]
  ref_z = cmd.anchor_pos_w_ref[:, 2]
  return torch.abs(robot_z - ref_z) > threshold


def bad_anchor_ori(
  env: ManagerBasedRlEnv,
  threshold: float = 0.8,
  command_name: str = "motion_command",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when anchor (torso) orientation deviates too far from motion reference.

  Paper: quat_error > 0.8. Uses 2*asin(norm(q_robot - q_ref)).
  Returns False if no motion command.
  """
  cmd = getattr(env, command_name, None)
  if cmd is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  asset: Entity = env.scene[asset_cfg.name]
  torso_idx = asset.body_names.index("torso_link")
  robot_q = asset.data.body_link_quat_w[:, torso_idx, :]
  ref_q = cmd.anchor_quat_w_ref
  delta = robot_q - ref_q
  delta_norm = torch.norm(delta, dim=-1).clamp(max=1.0)
  error = 2.0 * torch.asin(delta_norm)
  return error > threshold


def bad_ee_body_pos_z(
  env: ManagerBasedRlEnv,
  threshold: float = 0.25,
  command_name: str = "motion_command",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when any end-effector z deviates too far from motion reference.

  Paper: |z_ee - z_ref| > 0.25m for ankles and wrists.
  Returns False if no motion command.
  """
  cmd = getattr(env, command_name, None)
  if cmd is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  asset: Entity = env.scene[asset_cfg.name]
  ee_names = (
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
  )
  ee_robot_indices = [asset.body_names.index(n) for n in ee_names]

  # Full-body indices for the same end-effectors in the motion data (30 bodies).
  ee_motion_indices = [6, 12, 22, 29]

  robot_z = asset.data.body_link_pos_w[:, ee_robot_indices, 2]  # (E, 4)
  ref_z = torch.cat(
    [cmd.get_ee_pos_w_ref(bi)[:, 2:3] for bi in ee_motion_indices], dim=-1
  )  # (E, 4)
  deviations = torch.abs(robot_z - ref_z)
  return deviations.max(dim=-1).values > threshold
