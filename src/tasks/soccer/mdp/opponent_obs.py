"""Opponent observation terms for dual-robot adversarial training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def opponent_root_state_in_robot_frame(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  opponent_cfg: SceneEntityCfg = SceneEntityCfg("opponent"),
) -> torch.Tensor:
  """Opponent root position, linear velocity, and angular velocity in robot frame."""
  robot: Entity = env.scene[robot_cfg.name]
  opponent: Entity = env.scene[opponent_cfg.name]
  robot_quat_w = robot.data.root_link_quat_w
  rel_pos_w = opponent.data.root_link_pos_w - robot.data.root_link_pos_w
  rel_lin_vel_w = opponent.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
  rel_ang_vel_w = opponent.data.root_link_ang_vel_w - robot.data.root_link_ang_vel_w
  return torch.cat(
    (
      quat_apply_inverse(robot_quat_w, rel_pos_w),
      quat_apply_inverse(robot_quat_w, rel_lin_vel_w),
      quat_apply_inverse(robot_quat_w, rel_ang_vel_w),
    ),
    dim=-1,
  )


def opponent_joint_state(
  env: ManagerBasedRlEnv,
  opponent_cfg: SceneEntityCfg = SceneEntityCfg("opponent"),
) -> torch.Tensor:
  """Opponent joint position offset and velocity."""
  opponent: Entity = env.scene[opponent_cfg.name]
  jnt_ids = opponent_cfg.joint_ids
  default_joint_pos = opponent.data.default_joint_pos
  assert default_joint_pos is not None
  joint_pos = opponent.data.joint_pos[:, jnt_ids] - default_joint_pos[:, jnt_ids]
  joint_vel = opponent.data.joint_vel[:, jnt_ids]
  return torch.cat((joint_pos, joint_vel), dim=-1)
