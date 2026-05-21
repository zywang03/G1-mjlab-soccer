"""Domain randomization events for the soccer task.

Includes robot base push and ball velocity perturbation, used during
training. These are disabled in eval configs (matching paper protocols).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp import sample_uniform
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def push_robot_base(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  vel_xy_range: tuple[float, float],
  vel_z_range: tuple[float, float],
  ang_vel_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Apply random velocity push to the robot base."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  n = len(env_ids)

  asset: Entity = env.scene[asset_cfg.name]

  lin_vel_xy = sample_uniform(
    vel_xy_range[0], vel_xy_range[1], (n, 2), env.device
  )
  lin_vel_z = sample_uniform(
    vel_z_range[0], vel_z_range[1], (n, 1), env.device
  )
  lin_vel = torch.cat([lin_vel_xy, lin_vel_z], dim=-1)

  ang_vel = sample_uniform(
    ang_vel_range[0], ang_vel_range[1], (n, 3), env.device
  )

  velocities = torch.cat([lin_vel, ang_vel], dim=-1)
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def perturb_ball_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  vel_range: tuple[float, float],
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
  """Add random perturbation to ball linear velocity."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  n = len(env_ids)

  asset: Entity = env.scene[ball_cfg.name]
  current_vel = asset.data.root_link_vel_w[env_ids].clone()
  noise = sample_uniform(vel_range[0], vel_range[1], (n, 3), env.device)
  current_vel[:, :3] += noise
  asset.write_root_link_velocity_to_sim(current_vel, env_ids=env_ids)
