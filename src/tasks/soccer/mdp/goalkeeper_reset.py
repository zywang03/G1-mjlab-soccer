"""Official-style reset and timing utilities for goalkeeper training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp import sample_uniform
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.soccer.mdp.goalkeeper_obs import _REF_DEFAULT_DOF_POS

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def decrement_goalkeeper_catchstep(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> None:
  """Advance official catch-step timing by one policy step."""
  del env_ids
  catchstep = getattr(env, "_gk_catchstep", None)
  if catchstep is None or catchstep.shape[0] != env.num_envs:
    catchstep = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    setattr(env, "_gk_catchstep", catchstep)
  catchstep.sub_(1)


def reset_goalkeeper_root_official(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_z_range: tuple[float, float] = (-0.01, 0.01),
  velocity_range: tuple[float, float] = (-0.3, 0.3),
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Reset the base like the official implementation."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  asset: Entity = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()
  root_states[:, :3] += env.scene.env_origins[env_ids]
  root_states[:, 2] += sample_uniform(
    pose_z_range[0], pose_z_range[1], (len(env_ids),), env.device
  )
  root_states[:, 7:13] = sample_uniform(
    velocity_range[0], velocity_range[1], (len(env_ids), 6), env.device
  )
  asset.write_root_link_pose_to_sim(root_states[:, :7], env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(root_states[:, 7:13], env_ids=env_ids)


def reset_goalkeeper_joints_official(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float] = (0.5, 1.5),
  offset_range: tuple[float, float] = (-0.1, 0.1),
  continue_keep: bool = True,
  continue_keep_prob: float = 0.8,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> None:
  """Official post-task reset/recovery.

  With ``continue_keep`` enabled, most reset envs copy joint state from random
  non-reset envs. This exposes the policy to recovery states after saves,
  falls, and partial motions instead of always starting from the same stance.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

  n = len(env_ids)
  current_pos = asset.data.joint_pos[:, joint_ids]
  current_vel = asset.data.joint_vel[:, joint_ids]
  default_pos = torch.as_tensor(
    _REF_DEFAULT_DOF_POS,
    device=env.device,
    dtype=current_pos.dtype,
  )[joint_ids]

  use_continue = torch.zeros(n, dtype=torch.bool, device=env.device)
  if continue_keep and env.num_envs > 1:
    use_continue = torch.rand(n, device=env.device) < continue_keep_prob

  joint_pos = torch.empty(n, current_pos.shape[1], dtype=current_pos.dtype, device=env.device)
  joint_vel = torch.empty_like(joint_pos)

  if torch.any(use_continue):
    source_ids = torch.randint(0, env.num_envs, (int(use_continue.sum().item()),), device=env.device)
    joint_pos[use_continue] = current_pos[source_ids]
    joint_vel[use_continue] = current_vel[source_ids]

  if torch.any(~use_continue):
    count = int((~use_continue).sum().item())
    scale = sample_uniform(scale_range[0], scale_range[1], (count, current_pos.shape[1]), env.device)
    offset = sample_uniform(offset_range[0], offset_range[1], (count, current_pos.shape[1]), env.device)
    joint_pos[~use_continue] = default_pos.unsqueeze(0) * scale + offset
    joint_vel[~use_continue] = 0.0

  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  if soft_joint_pos_limits is not None:
    limits = soft_joint_pos_limits[env_ids][:, joint_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])

  asset.write_joint_state_to_sim(
    joint_pos.view(n, -1),
    joint_vel.view(n, -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )
