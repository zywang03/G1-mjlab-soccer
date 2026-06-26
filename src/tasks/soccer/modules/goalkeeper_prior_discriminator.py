"""Offline motion-prior discriminator for the goalkeeper student."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalkeeperPriorDataset:
  """Flat ``[student_obs, action]`` samples from successful teacher rollout shards."""

  def __init__(
    self,
    dataset_dir: str,
    *,
    device: str | torch.device = "cpu",
    max_samples: int | None = None,
  ) -> None:
    self.device = torch.device(device)
    root = Path(dataset_dir).expanduser().resolve()
    shard_dir = root / "shards" if (root / "shards").is_dir() else root
    shards = sorted(shard_dir.glob("shard_*.pt"))
    if not shards:
      raise FileNotFoundError(f"No shard_*.pt files under {shard_dir}")
    features: list[torch.Tensor] = []
    total = 0
    for shard in shards:
      payload = torch.load(shard, map_location="cpu", weights_only=False)
      obs = payload["student_obs"].float()
      actions = payload["teacher_action"].float()
      valid = payload["valid_mask"].bool()
      success = payload.get("metadata", {}).get("success")
      if success is not None:
        valid = valid & success.bool().view(-1, 1)
      flat = torch.cat([obs, actions], dim=-1)[valid]
      if flat.numel() == 0:
        continue
      if max_samples is not None:
        remaining = int(max_samples) - total
        if remaining <= 0:
          break
        flat = flat[:remaining]
      features.append(flat)
      total += int(flat.shape[0])
    if not features:
      raise RuntimeError(f"No successful prior samples found under {shard_dir}")
    self.features = torch.cat(features, dim=0).to(self.device)

  @property
  def num_samples(self) -> int:
    return int(self.features.shape[0])

  @property
  def input_dim(self) -> int:
    return int(self.features.shape[-1])

  def sample(self, batch_size: int) -> torch.Tensor:
    idx = torch.randint(self.num_samples, (int(batch_size),), device=self.device)
    return self.features[idx]


class GoalkeeperPriorDiscriminator(nn.Module):
  """Small MLP discriminator over concatenated student observations and actions."""

  def __init__(
    self,
    input_dim: int,
    hidden_dims: tuple[int, ...] = (256, 128),
    reward_clip: float = 5.0,
  ) -> None:
    super().__init__()
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
      layers.append(nn.Linear(last_dim, int(hidden_dim)))
      layers.append(nn.ELU())
      last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, 1))
    self.net = nn.Sequential(*layers)
    self.reward_clip = float(reward_clip)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    return self.net(features).squeeze(-1)

  def reward(self, features: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      logits = self.forward(features)
      reward = F.softplus(logits)
      if self.reward_clip > 0.0:
        reward = reward.clamp(max=self.reward_clip)
      return reward


def discriminator_loss(
  discriminator: GoalkeeperPriorDiscriminator,
  expert_features: torch.Tensor,
  policy_features: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
  expert_logits = discriminator(expert_features)
  policy_logits = discriminator(policy_features)
  expert_loss = F.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits))
  policy_loss = F.binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits))
  loss = expert_loss + policy_loss
  with torch.no_grad():
    expert_acc = (expert_logits > 0.0).float().mean().item()
    policy_acc = (policy_logits < 0.0).float().mean().item()
  return loss, {
    "expert_acc": float(expert_acc),
    "policy_acc": float(policy_acc),
  }
