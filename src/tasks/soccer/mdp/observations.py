"""Observation functions for the soccer task.

Includes proprioceptive observations (IMU, joints, actions), ball-local
observations (position, velocity), motion-reference observations, and
soccer-perception observations (world point in robot frame).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# ---------------------------------------------------------------------------
# Proprioceptive observations
# ---------------------------------------------------------------------------


def builtin_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return sensor.data


def projected_gravity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b


def joint_pos_rel(
  env: ManagerBasedRlEnv,
  biased: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  jnt_ids = asset_cfg.joint_ids
  joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  return joint_pos[:, jnt_ids] - default_joint_pos[:, jnt_ids]


def joint_vel_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, jnt_ids] - default_joint_vel[:, jnt_ids]


def last_action(
  env: ManagerBasedRlEnv, action_name: str | None = None
) -> torch.Tensor:
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action


# ---------------------------------------------------------------------------
# Ball-local observations (ball position / velocity in robot pelvis frame)
# ---------------------------------------------------------------------------


def ball_pos_in_robot_frame(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Ball position in robot pelvis frame.

  Returns:
    Tensor of shape (num_envs, 3) — ball (x, y, z) relative to robot pelvis,
    expressed in the pelvis coordinate frame.
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  ball_pos_w = ball.data.root_link_pos_w
  robot_pos_w = robot.data.root_link_pos_w
  robot_quat_w = robot.data.root_link_quat_w
  delta_w = ball_pos_w - robot_pos_w
  return quat_apply_inverse(robot_quat_w, delta_w)


def ball_vel_in_robot_frame(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Ball linear velocity in robot pelvis frame.

  Returns:
    Tensor of shape (num_envs, 3).
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  ball_vel_w = ball.data.root_link_lin_vel_w
  robot_quat_w = robot.data.root_link_quat_w
  return quat_apply_inverse(robot_quat_w, ball_vel_w)


# ---------------------------------------------------------------------------
# Motion reference observations (O^ref_t — for shooter eval)
# ---------------------------------------------------------------------------


def motion_ref_joint_pos(
  env: ManagerBasedRlEnv,
  command_name: str = "motion_command",
) -> torch.Tensor:
  cmd = getattr(env, command_name, None)
  if cmd is None:
    return torch.zeros(env.num_envs, 29, device=env.device)
  return cmd.joint_pos_ref


def motion_ref_joint_vel(
  env: ManagerBasedRlEnv,
  command_name: str = "motion_command",
) -> torch.Tensor:
  cmd = getattr(env, command_name, None)
  if cmd is None:
    return torch.zeros(env.num_envs, 29, device=env.device)
  return cmd.joint_vel_ref


def motion_ref_anchor_ang_vel(
  env: ManagerBasedRlEnv,
  command_name: str = "motion_command",
) -> torch.Tensor:
  cmd = getattr(env, command_name, None)
  if cmd is None:
    return torch.zeros(env.num_envs, 3, device=env.device)
  return cmd.anchor_ang_vel_ref


# ---------------------------------------------------------------------------
# Soccer-perception observations (O^soc_t — for shooter eval)
# ---------------------------------------------------------------------------


def world_point_in_robot_frame(
  env: ManagerBasedRlEnv,
  point: tuple[float, float, float],
  robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Convert a fixed world-frame point to robot pelvis frame."""
  robot: Entity = env.scene[robot_cfg.name]
  robot_pos_w = robot.data.root_link_pos_w
  robot_quat_w = robot.data.root_link_quat_w
  point_t = torch.tensor(point, device=env.device, dtype=torch.float32)
  delta_w = point_t.unsqueeze(0) - robot_pos_w
  return quat_apply_inverse(robot_quat_w, delta_w)
