"""Train a position-conditioned LSTM goalkeeper student with behavior cloning.

The dataset is produced by ``scripts/collect_goalkeeper_teacher_dataset.py``.
Each sample contains the normal MoE-teacher actor observation plus a 4D
prediction condition.  The condition is encoded separately and FiLM-modulates
the LSTM latent, so it is not mixed into the recurrent input.  The teacher
itself is not widened: rollout still uses the original gate + six region
experts + optional prepare expert.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro
import torch
import torch.nn.functional as F
from tensordict import TensorDict

try:
  import scripts.train_shooter_bc as _shooter_bc
except ModuleNotFoundError:  # Direct execution: python scripts/train_goalkeeper_student_bc.py
  import train_shooter_bc as _shooter_bc

from src.tasks.soccer.modules.goalkeeper_student_actor import GoalkeeperStudentFiLMActor
from src.tasks.soccer.mdp.goalkeeper_student_obs import (
  GOALKEEPER_PHASE_IDLE_INDEX,
  GOALKEEPER_STUDENT_OBS_DIM,
  GOALKEEPER_TEACHER_ACTOR_OBS_DIM,
  build_goalkeeper_student_obs,
)

_BaseBcConfig = _shooter_bc.BcConfig


@dataclass
class BcConfig(_BaseBcConfig):
  output_dir: str = "logs/bc/goalkeeper_student"
  """Root directory for goalkeeper student BC runs."""

  model: str = "rnn"
  """Goalkeeper student defaults to the shooter-style LSTM actor."""

  hidden_dims: tuple[int, ...] = (256, 128, 64)
  """Action-head MLP used by the goalkeeper PPO student."""

  rnn_hidden_dim: int = 160
  """LSTM hidden size used by the goalkeeper PPO student."""

  rnn_num_layers: int = 2
  """Number of recurrent layers used by the goalkeeper PPO student."""

  condition_hidden_dim: int = 64
  """Hidden size for the 4D landing-condition FiLM encoder."""

  ball_latent_dim: int = 6
  """Unsupervised ball latent size used by the route-conditioned FiLM branch."""

  region_aux_coef: float = 0.2
  """Weight for 7-class MoE route prediction during BC."""

  region_aux_active_only: bool = False
  """Apply route auxiliary loss only after launch when true."""

  teacher: str | None = "/data/Courses/[CS2810]EmbodiedAI/humanoid_soccer_proj/G1-mjlab-soccer/checkpoints/goalkeeper_moe7_hard3_default_idle.pt"
  """Optional MoE7 teacher checkpoint whose learned gate provides BC route labels."""

  batch_size: int = 256
  """Number of full episodes per RNN BC minibatch."""

  init_std: float = 0.2
  """Initial action std metadata to match the goalkeeper PPO student config."""

  normalizer_max_shards: int = 64
  """Maximum number of shards used to warm up observation normalization."""

  success_only: bool = False
  """Goalkeeper student BC defaults to all valid teacher timesteps."""

  dataloader_num_workers: int = 8
  """Number of DataLoader workers for parallel shard loading."""

  train_shards_per_epoch: int | None = 128
  """Random train shard subset used each epoch; set <=0 or None for all train shards."""

  val_interval: int = 10
  """Run validation and refresh model_best every N epochs; final epoch always validates."""

  active_action_loss_weight: float = 1.0
  """Extra BC action-loss weight for non-idle frames where the ball is incoming."""

  transition_prefix_prob: float = 0.0
  """Probability of prepending a prepare-only sequence to an active BC episode."""

  transition_prefix_min_steps: int = 10
  """Minimum prepare-prefix length used for transition augmentation."""

  transition_prefix_max_steps: int = 75
  """Maximum prepare-prefix length used for transition augmentation."""

  best_metric: str = "loss"
  """Validation metric used for model_best: loss, action_loss, active_action_loss, or idle_action_loss."""


def make_goalkeeper_student_actor(obs_dim: int, action_dim: int, cfg: BcConfig, device: str):
  """Create the FiLM-conditioned LSTM actor used by goalkeeper student BC."""
  dummy_obs = TensorDict({"student": torch.zeros(1, obs_dim)}, batch_size=[1])
  return GoalkeeperStudentFiLMActor(
    dummy_obs,
    obs_groups={"actor": ["student"]},
    obs_set="actor",
    output_dim=action_dim,
    hidden_dims=cfg.hidden_dims,
    activation=cfg.activation,
    obs_normalization=cfg.obs_normalization,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": cfg.init_std,
      "std_type": "scalar",
    },
    rnn_type=cfg.rnn_type,
    rnn_hidden_dim=cfg.rnn_hidden_dim,
    rnn_num_layers=cfg.rnn_num_layers,
    condition_hidden_dim=cfg.condition_hidden_dim,
    ball_latent_dim=cfg.ball_latent_dim,
  ).to(device)


def _build_episode_index(shards: list[Path], success_only: bool, action_clip: float | None) -> list[tuple[int, int]]:
  """Build (shard_idx, episode_idx) index without loading full data into memory."""
  index: list[tuple[int, int]] = []
  for s_idx, shard in enumerate(shards):
    payload = torch.load(shard, map_location="cpu", weights_only=False)
    N = int(payload["student_obs"].shape[0])
    for e_idx in range(N):
      if success_only and not bool(payload["metadata"]["success"][e_idx].item()):
        continue
      if not torch.any(payload["valid_mask"][e_idx].bool()):
        continue
      index.append((s_idx, e_idx))
    del payload
  return index


class _ShardIterableDataset(torch.utils.data.IterableDataset):
  """Iterable dataset that streams shards across DataLoader workers.

  Each worker is assigned a subset of shards. It loads one shard at a time,
  extracts valid episodes, and yields individual (obs, action) episodes.
  The DataLoader's batch_size + collate_fn assembles padded batches.
  """

  def __init__(
    self,
    shards: list[Path],
    success_only: bool,
    action_clip: float | None,
    seed: int = 0,
    transition_prefix_prob: float = 0.0,
    transition_prefix_min_steps: int = 10,
    transition_prefix_max_steps: int = 75,
  ):
    self.shards = shards
    self.success_only = success_only
    self.action_clip = action_clip
    self.seed = seed
    self.transition_prefix_prob = max(0.0, min(1.0, float(transition_prefix_prob)))
    self.transition_prefix_min_steps = max(1, int(transition_prefix_min_steps))
    self.transition_prefix_max_steps = max(self.transition_prefix_min_steps, int(transition_prefix_max_steps))

  def __iter__(self):
    import random
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      shard_ids = list(range(len(self.shards)))
      wid = 0
    else:
      w = worker_info.num_workers
      wid = worker_info.id
      shard_ids = list(range(wid, len(self.shards), w))
    rng = random.Random(self.seed + wid)
    rng.shuffle(shard_ids)
    for sid in shard_ids:
      payload = torch.load(self.shards[sid], map_location="cpu", weights_only=False)
      N = int(payload["student_obs"].shape[0])
      ep_ids = list(range(N))
      rng.shuffle(ep_ids)
      for eid in ep_ids:
        if self.success_only and not bool(payload["metadata"]["success"][eid].item()):
          continue
        mask = payload["valid_mask"][eid].bool()
        if not torch.any(mask):
          continue
        obs = payload["student_obs"][eid, mask, :].float()
        act = payload["teacher_action"][eid, mask, :].float()
        if self.action_clip is not None:
          act = torch.clamp(act, -self.action_clip, self.action_clip)
        finite = torch.isfinite(obs).all(dim=-1) & torch.isfinite(act).all(dim=-1)
        if torch.any(finite):
          obs = obs[finite]
          act = act[finite]
          prefix = self._sample_prepare_prefix(payload, rng, exclude=eid)
          if prefix is not None:
            prefix_obs, prefix_act = prefix
            yield obs, act, prefix_obs, prefix_act
          else:
            yield obs, act
      del payload

  def _sample_prepare_prefix(self, payload: dict, rng, *, exclude: int):
    if self.transition_prefix_prob <= 0.0 or rng.random() >= self.transition_prefix_prob:
      return None
    N = int(payload["student_obs"].shape[0])
    candidates = list(range(N))
    rng.shuffle(candidates)
    for prefix_eid in candidates:
      if prefix_eid == exclude and N > 1:
        continue
      mask = payload["valid_mask"][prefix_eid].bool()
      if not torch.any(mask):
        continue
      obs = payload["student_obs"][prefix_eid, mask, :].float()
      act = payload["teacher_action"][prefix_eid, mask, :].float()
      idle = _idle_mask_from_student_obs(obs)
      idle_idx = torch.nonzero(idle, as_tuple=False).flatten()
      if idle_idx.numel() < self.transition_prefix_min_steps:
        continue
      max_len = min(int(idle_idx.numel()), self.transition_prefix_max_steps)
      length = rng.randint(self.transition_prefix_min_steps, max_len)
      selected = idle_idx[:length]
      prefix_obs = obs[selected]
      prefix_act = act[selected]
      if self.action_clip is not None:
        prefix_act = torch.clamp(prefix_act, -self.action_clip, self.action_clip)
      finite = torch.isfinite(prefix_obs).all(dim=-1) & torch.isfinite(prefix_act).all(dim=-1)
      if torch.any(finite):
        return prefix_obs[finite], prefix_act[finite]
    return None


def _select_epoch_shards(
  shards: list[Path],
  count: int | None,
  seed: int,
  *,
  epoch: int,
  shuffle: bool,
) -> list[Path]:
  """Pick a deterministic per-epoch shard subset without replacement."""
  if not shards:
    return []
  if count is None or count <= 0 or count >= len(shards):
    selected = list(shards)
  else:
    generator = torch.Generator().manual_seed(seed + epoch * 1009)
    order = torch.randperm(len(shards), generator=generator).tolist()
    selected = [shards[idx] for idx in order[:count]]
  if shuffle and len(selected) > 1 and (count is None or count <= 0 or count >= len(shards)):
    generator = torch.Generator().manual_seed(seed + epoch * 1009)
    order = torch.randperm(len(selected), generator=generator).tolist()
    selected = [selected[idx] for idx in order]
  return selected


def _should_run_validation(epoch: int, cfg: BcConfig) -> bool:
  interval = int(cfg.val_interval)
  if interval <= 0:
    return epoch == cfg.epochs
  return epoch % interval == 0 or epoch == cfg.epochs


def _collate_padded(batch: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Pad variable-length episodes into (T_max, B, dim) with masks."""
  normalized_batch = []
  for item in batch:
    if len(item) == 2:
      obs, act = item
    elif len(item) == 4:
      obs, act, prefix_obs, prefix_act = item
      obs = torch.cat([prefix_obs, obs], dim=0)
      act = torch.cat([prefix_act, act], dim=0)
    else:
      raise ValueError(f"Unsupported BC batch item with {len(item)} tensors")
    normalized_batch.append((_upgrade_goalkeeper_student_obs(obs), act))
  max_t = max(obs.shape[0] for obs, _ in normalized_batch)
  dim_obs = GOALKEEPER_STUDENT_OBS_DIM
  dim_act = normalized_batch[0][1].shape[1]
  obs_padded = torch.zeros(max_t, len(normalized_batch), dim_obs)
  act_padded = torch.zeros(max_t, len(normalized_batch), dim_act)
  masks = torch.zeros(max_t, len(normalized_batch), dtype=torch.bool)
  for j, (obs, act) in enumerate(normalized_batch):
    t = obs.shape[0]
    obs_padded[:t, j] = obs
    act_padded[:t, j] = act
    masks[:t, j] = True
  return obs_padded, act_padded, masks


def _make_dataloader(
  shards: list[Path],
  cfg: BcConfig,
  shuffle: bool,
  *,
  seed: int | None = None,
) -> torch.utils.data.DataLoader:
  dataset = _ShardIterableDataset(
    shards,
    cfg.success_only,
    cfg.action_clip,
    seed=cfg.seed if seed is None else seed,
    transition_prefix_prob=cfg.transition_prefix_prob,
    transition_prefix_min_steps=cfg.transition_prefix_min_steps,
    transition_prefix_max_steps=cfg.transition_prefix_max_steps,
  )
  return torch.utils.data.DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    collate_fn=_collate_padded,
    num_workers=cfg.dataloader_num_workers,
    pin_memory=True,
    persistent_workers=cfg.dataloader_num_workers > 0,
  )


def _update_goalkeeper_obs_normalizer(actor, shards: list[Path], cfg: BcConfig, device: str) -> int:
  if not cfg.obs_normalization:
    return 0

  actor.train()
  total = 0
  warmup_shards = shards[:cfg.normalizer_max_shards] if cfg.normalizer_max_shards > 0 else shards
  with torch.no_grad():
    for shard in warmup_shards:
      obs, _ = _shooter_bc._load_flat_samples(shard, cfg.success_only, cfg.action_clip)
      if obs.numel() == 0:
        continue
      obs = _upgrade_goalkeeper_student_obs(obs)
      total += int(obs.shape[0])
      for start in range(0, obs.shape[0], cfg.batch_size):
        batch = obs[start:start + cfg.batch_size].to(device)
        actor.update_normalization(_shooter_bc._make_obs_tensordict(batch))
  return total


def _load_goalkeeper_route_gate(checkpoint_path: str | None, device: str):
  if not checkpoint_path:
    return None
  path = Path(checkpoint_path).expanduser()
  if not path.exists():
    return None
  loaded = torch.load(path, map_location=device, weights_only=False)
  bundle = (
    loaded["actor_state_dict"]
    if isinstance(loaded, dict) and "actor_state_dict" in loaded and "sr" in loaded["actor_state_dict"]
    else loaded
  )
  if not isinstance(bundle, dict):
    return None
  gate = bundle.get("gate")
  if not isinstance(gate, dict) or not gate.get("state"):
    return None

  from src.tasks.soccer.modules.moe6_goalkeeper_policy import _make_gate_net

  state = gate["state"]
  num_classes = int(gate.get("num_classes", state["4.weight"].shape[0]))
  net = _make_gate_net(num_classes, torch.device(device))
  net.load_state_dict(state)
  net.eval()
  for param in net.parameters():
    param.requires_grad_(False)
  return {
    "net": net,
    "mean": gate["mean"].to(device),
    "std": gate["std"].to(device).clamp_min(1.0e-6),
  }


def _goalkeeper_route_targets_from_obs(
  student_obs: torch.Tensor,
  route_gate,
  *,
  z_low: float = 0.85,
  z_up: float = 1.35,
  vz_low: float = -5.0,
  idle_speed_threshold: float = 0.5,
  idle_incoming_vx_threshold: float = -0.5,
  gravity: float = 9.81,
  dt: float = 0.02,
) -> torch.Tensor:
  """Return MoE7-style route labels from student obs: 0-5 active, 6 prepare."""
  ball_history = student_obs[..., :30].reshape(*student_obs.shape[:-1], 10, 3)
  pos = ball_history[..., -1, :]
  prev = ball_history[..., -2, :]
  vel = (pos - prev) / dt
  bx = pos[..., 0]
  vx = vel[..., 0]

  if isinstance(route_gate, dict):
    features = torch.cat([pos, vel], dim=-1)
    flat_features = features.reshape(-1, features.shape[-1])
    gate_mean = route_gate["mean"].to(device=flat_features.device, dtype=flat_features.dtype)
    gate_std = route_gate["std"].to(device=flat_features.device, dtype=flat_features.dtype).clamp_min(1.0e-6)
    flat_route = route_gate["net"]((flat_features - gate_mean) / gate_std).argmax(dim=-1).clamp(max=5)
    route = flat_route.reshape(bx.shape)
  else:
    t = torch.clamp(-bx / (vx - 1.0e-3), 0.0, 2.0)
    cy = pos[..., 1] + vel[..., 1] * t
    cz = pos[..., 2] + vel[..., 2] * t - 0.5 * gravity * t * t
    route = torch.zeros_like(bx, dtype=torch.long)
    route = torch.where(cz < z_low, torch.full_like(route, 4), route)
    route = torch.where(cz > z_up, torch.full_like(route, 2), route)
    route = torch.where(vel[..., 2] - gravity * t < vz_low, torch.full_like(route, 4), route)
    route = route + (cy < 0.0).long()

  speed = torch.linalg.vector_norm(vel, dim=-1)
  condition_idle = _idle_mask_from_student_obs(student_obs)
  kinematic_idle = (speed < idle_speed_threshold) | (vx >= idle_incoming_vx_threshold)
  idle = condition_idle | kinematic_idle
  return torch.where(idle, torch.full_like(route, 6), route)


def _bc_action_loss_per_element(pred: torch.Tensor, actions: torch.Tensor, cfg: BcConfig) -> torch.Tensor:
  if cfg.loss == "smooth_l1":
    return F.smooth_l1_loss(pred, actions, beta=cfg.smooth_l1_beta, reduction="none")
  if cfg.loss == "mse":
    return F.mse_loss(pred, actions, reduction="none")
  raise ValueError(f"Unsupported BC loss: {cfg.loss!r}. Use 'smooth_l1' or 'mse'.")


def _active_mask_from_student_obs(obs_flat: torch.Tensor) -> torch.Tensor:
  return ~_idle_mask_from_student_obs(obs_flat)


def _idle_mask_from_student_obs(obs_flat: torch.Tensor) -> torch.Tensor:
  if obs_flat.shape[-1] >= GOALKEEPER_STUDENT_OBS_DIM:
    return obs_flat[..., GOALKEEPER_PHASE_IDLE_INDEX] > 0.5
  return obs_flat[..., -1] > 0.5


def _upgrade_goalkeeper_student_obs(obs: torch.Tensor) -> torch.Tensor:
  if obs.shape[-1] == GOALKEEPER_STUDENT_OBS_DIM:
    return obs
  if obs.shape[-1] == GOALKEEPER_TEACHER_ACTOR_OBS_DIM + 4:
    return build_goalkeeper_student_obs(
      obs[..., :GOALKEEPER_TEACHER_ACTOR_OBS_DIM],
      obs[..., GOALKEEPER_TEACHER_ACTOR_OBS_DIM:],
    )
  return obs


def _weighted_goalkeeper_action_loss(
  pred: torch.Tensor,
  target: torch.Tensor,
  obs_flat: torch.Tensor,
  cfg: BcConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
  per_element = _bc_action_loss_per_element(pred, target, cfg)
  active = _active_mask_from_student_obs(obs_flat)
  weights = torch.ones_like(per_element)
  if cfg.active_action_loss_weight != 1.0:
    weights = torch.where(active.unsqueeze(-1), weights * cfg.active_action_loss_weight, weights)
  loss = (per_element * weights).sum() / weights.sum().clamp_min(1.0)

  with torch.no_grad():
    idle = ~active
    active_values = int(active.sum().item()) * int(target.shape[-1])
    idle_values = int(idle.sum().item()) * int(target.shape[-1])
    active_loss_sum = float(per_element[active].sum().item()) if torch.any(active) else 0.0
    idle_loss_sum = float(per_element[idle].sum().item()) if torch.any(idle) else 0.0
  stats = {
    "unweighted_loss_sum": float(per_element.sum().item()),
    "action_values": float(per_element.numel()),
    "active_action_loss_sum": active_loss_sum,
    "active_action_values": float(active_values),
    "idle_action_loss_sum": idle_loss_sum,
    "idle_action_values": float(idle_values),
  }
  return loss, stats


def _goalkeeper_region_aux_loss(
  actor,
  obs_chunk: torch.Tensor,
  masks_chunk: torch.Tensor,
  cfg: BcConfig,
  route_gate,
  *,
  reduction: str = "mean",
) -> tuple[torch.Tensor, int]:
  aux = getattr(actor, "region_condition_output", None)
  if aux is None:
    return torch.zeros((), device=obs_chunk.device), 0
  raw_obs = obs_chunk[masks_chunk]
  targets = _goalkeeper_route_targets_from_obs(raw_obs, route_gate)
  logits = aux["region_logits"]
  if cfg.region_aux_active_only:
    active = targets != 6
    if not torch.any(active):
      return torch.zeros((), device=obs_chunk.device), 0
    targets = targets[active]
    logits = logits[active]
  if targets.numel() == 0:
    return torch.zeros((), device=obs_chunk.device), 0
  return F.cross_entropy(logits, targets.detach(), reduction=reduction), int(targets.numel())


def _evaluate_goalkeeper_bc(actor, shards: list[Path], cfg: BcConfig, device: str, route_gate) -> dict[str, float]:
  if not shards:
    return {
      "loss": float("nan"),
      "action_loss": float("nan"),
      "active_action_loss": float("nan"),
      "idle_action_loss": float("nan"),
      "region_aux_loss": float("nan"),
      "samples": 0.0,
    }

  actor.eval()
  loader = _make_dataloader(shards, cfg, shuffle=False, seed=cfg.seed + 17)
  total_action_loss = 0.0
  total_action_values = 0
  total_active_action_loss = 0.0
  total_active_action_values = 0
  total_idle_action_loss = 0.0
  total_idle_action_values = 0
  total_region_loss = 0.0
  total_region_samples = 0
  batches = 0
  with torch.no_grad():
    for obs_padded, act_padded, masks in loader:
      obs_padded = obs_padded.to(device, non_blocking=True)
      act_padded = act_padded.to(device, non_blocking=True)
      masks = masks.to(device, non_blocking=True)
      T_max = obs_padded.shape[0]
      hidden_state = None
      for start in range(0, T_max, cfg.bptt_length):
        end = min(start + cfg.bptt_length, T_max)
        obs_chunk = obs_padded[start:end]
        act_chunk = act_padded[start:end]
        mask_chunk = masks[start:end]
        if not torch.any(mask_chunk):
          continue
        pred, hidden_state = _shooter_bc._forward_rnn_chunk(actor, obs_chunk, mask_chunk, hidden_state)
        target = act_chunk[mask_chunk]
        obs_flat = obs_chunk[mask_chunk]
        action_loss = _bc_action_loss_per_element(pred, target, cfg).sum()
        active = _active_mask_from_student_obs(obs_flat)
        idle = ~active
        region_loss, region_samples = _goalkeeper_region_aux_loss(
          actor, obs_chunk, mask_chunk, cfg, route_gate, reduction="sum",
        )
        action_values = int(target.numel())
        total_action_loss += float(action_loss.item())
        total_action_values += action_values
        if torch.any(active):
          total_active_action_loss += float(_bc_action_loss_per_element(pred[active], target[active], cfg).sum().item())
          total_active_action_values += int(active.sum().item()) * int(target.shape[-1])
        if torch.any(idle):
          total_idle_action_loss += float(_bc_action_loss_per_element(pred[idle], target[idle], cfg).sum().item())
          total_idle_action_values += int(idle.sum().item()) * int(target.shape[-1])
        if region_samples > 0:
          total_region_loss += float(region_loss.item())
          total_region_samples += region_samples
      batches += 1
      if cfg.max_val_batches is not None and batches >= cfg.max_val_batches:
        break

  if total_action_values == 0:
    return {
      "loss": float("nan"),
      "action_loss": float("nan"),
      "active_action_loss": float("nan"),
      "idle_action_loss": float("nan"),
      "region_aux_loss": float("nan"),
      "samples": 0.0,
    }
  action_mean = total_action_loss / total_action_values
  active_action_mean = total_active_action_loss / total_active_action_values if total_active_action_values > 0 else float("nan")
  idle_action_mean = total_idle_action_loss / total_idle_action_values if total_idle_action_values > 0 else float("nan")
  region_mean = total_region_loss / total_region_samples if total_region_samples > 0 else 0.0
  return {
    "loss": action_mean + cfg.region_aux_coef * region_mean,
    "action_loss": action_mean,
    "active_action_loss": active_action_mean,
    "idle_action_loss": idle_action_mean,
    "region_aux_loss": region_mean,
    "samples": float(total_action_values),
    "active_action_values": float(total_active_action_values),
    "idle_action_values": float(total_idle_action_values),
    "region_samples": float(total_region_samples),
    "batches": float(batches),
  }


def train_goalkeeper_student_bc_impl(cfg: BcConfig) -> None:
  import json
  from dataclasses import asdict
  from datetime import datetime
  import numpy as np

  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  if cfg.model != "rnn":
    raise ValueError("Goalkeeper student BC only supports --model rnn.")

  shards = _shooter_bc._find_shards(cfg.dataset_dir)
  train_shards, val_shards = _shooter_bc._split_shards(shards, cfg)
  dataset_obs_dim, action_dim = _shooter_bc._infer_dims(shards)
  obs_dim = GOALKEEPER_STUDENT_OBS_DIM
  run_tag = cfg.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  out_dir = Path(cfg.output_dir).expanduser().resolve() / run_tag
  if out_dir.exists() and any(out_dir.iterdir()) and not cfg.overwrite:
    raise FileExistsError(f"Output directory already exists and is not empty: {out_dir}")
  out_dir.mkdir(parents=True, exist_ok=True)

  actor = make_goalkeeper_student_actor(obs_dim, action_dim, cfg, device)
  normalizer_samples = _update_goalkeeper_obs_normalizer(actor, train_shards, cfg, device)
  route_gate = _load_goalkeeper_route_gate(cfg.teacher, device)
  optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

  metadata = {
    "dataset_dir": str(Path(cfg.dataset_dir).expanduser().resolve()),
    "num_shards": len(shards),
    "train_shards": len(train_shards),
    "val_shards": len(val_shards),
    "obs_dim": obs_dim,
    "dataset_obs_dim": dataset_obs_dim,
    "action_dim": action_dim,
    "model": cfg.model,
    "normalizer_samples": normalizer_samples,
    "route_gate": "teacher" if route_gate is not None else "heuristic",
    "config": asdict(cfg),
  }
  _shooter_bc._save_json(out_dir / "config.json", metadata)

  print(f"[INFO] BC training on {device}  model={cfg.model}")
  print(f"[INFO] Output: {out_dir}")
  print(f"[INFO] Shards: train={len(train_shards)}, val={len(val_shards)}, obs_dim={obs_dim}, action_dim={action_dim}")
  print(f"[INFO] Route labels: {'teacher gate' if route_gate is not None else 'heuristic'}")
  if cfg.obs_normalization:
    print(f"[INFO] Updated obs normalizer with {normalizer_samples} samples.")
  print(f"[INFO] DataLoader workers: {cfg.dataloader_num_workers}")
  print(
    f"[INFO] Train shard subset per epoch: "
    f"{cfg.train_shards_per_epoch if cfg.train_shards_per_epoch and cfg.train_shards_per_epoch > 0 else 'all'}"
  )
  print(f"[INFO] Validation interval: {cfg.val_interval} epochs, max_val_batches={cfg.max_val_batches}")

  best_metric = float("inf")
  last_val = {
    "loss": float("nan"),
    "action_loss": float("nan"),
    "active_action_loss": float("nan"),
    "idle_action_loss": float("nan"),
    "region_aux_loss": float("nan"),
    "samples": 0.0,
    "active_action_values": 0.0,
    "idle_action_values": 0.0,
    "region_samples": 0.0,
    "batches": 0.0,
  }
  history: list[dict] = []
  for epoch in range(1, cfg.epochs + 1):
    actor.train()
    total_action_loss = 0.0
    total_action_values = 0
    total_region_loss = 0.0
    total_region_samples = 0
    batches = 0
    epoch_train_shards = _select_epoch_shards(
      train_shards,
      cfg.train_shards_per_epoch,
      cfg.seed,
      epoch=epoch,
      shuffle=True,
    )
    train_loader = _make_dataloader(epoch_train_shards, cfg, shuffle=True, seed=cfg.seed + epoch * 1009)

    for obs_padded, act_padded, masks in train_loader:
      obs_padded = obs_padded.to(device, non_blocking=True)
      act_padded = act_padded.to(device, non_blocking=True)
      masks = masks.to(device, non_blocking=True)
      T_max = obs_padded.shape[0]
      hidden_state = None

      for start in range(0, T_max, cfg.bptt_length):
        end = min(start + cfg.bptt_length, T_max)
        obs_chunk = obs_padded[start:end]
        act_chunk = act_padded[start:end]
        mask_chunk = masks[start:end]
        if not torch.any(mask_chunk):
          continue

        pred, hidden_state = _shooter_bc._forward_rnn_chunk(actor, obs_chunk, mask_chunk, hidden_state)
        target = act_chunk[mask_chunk]
        obs_flat = obs_chunk[mask_chunk]
        action_loss, action_stats = _weighted_goalkeeper_action_loss(pred, target, obs_flat, cfg)
        region_loss, region_samples = _goalkeeper_region_aux_loss(actor, obs_chunk, mask_chunk, cfg, route_gate)
        loss = action_loss + cfg.region_aux_coef * region_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0.0:
          torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
        optimizer.step()

        action_values = int(action_stats["action_values"])
        total_action_loss += float(action_stats["unweighted_loss_sum"])
        total_action_values += action_values
        if region_samples > 0:
          total_region_loss += float(region_loss.item()) * region_samples
          total_region_samples += region_samples

      batches += 1
      if cfg.max_train_batches_per_epoch is not None and batches >= cfg.max_train_batches_per_epoch:
        break

    if batches == 0 or total_action_values == 0:
      raise RuntimeError("No BC samples found. Check dataset or valid masks.")

    train_action_loss = total_action_loss / total_action_values
    train_region_loss = total_region_loss / total_region_samples if total_region_samples > 0 else 0.0
    train_loss = train_action_loss + cfg.region_aux_coef * train_region_loss
    ran_val = _should_run_validation(epoch, cfg)
    if ran_val:
      last_val = _evaluate_goalkeeper_bc(actor, val_shards, cfg, device, route_gate)
    val = last_val
    metric_value = val.get(cfg.best_metric, float("nan"))
    metric = metric_value if ran_val and torch.isfinite(torch.tensor(metric_value)) else float("inf")
    row = {
      "epoch": epoch,
      "train_loss": train_loss,
      "train_action_loss": train_action_loss,
      "train_region_aux_loss": train_region_loss,
      "val_loss": val["loss"],
      "val_action_loss": val["action_loss"],
      "val_active_action_loss": val["active_action_loss"],
      "val_idle_action_loss": val["idle_action_loss"],
      "val_region_aux_loss": val["region_aux_loss"],
      "train_action_values": total_action_values,
      "train_region_samples": total_region_samples,
      "val_action_values": int(val["samples"]),
      "val_active_action_values": int(val.get("active_action_values", 0.0)),
      "val_idle_action_values": int(val.get("idle_action_values", 0.0)),
      "val_region_samples": int(val.get("region_samples", 0.0)),
      "val_batches": int(val.get("batches", 0.0)),
      "batches": batches,
      "train_shards": len(epoch_train_shards),
      "validated": ran_val,
      "best_metric": cfg.best_metric,
      "metric": metric,
    }
    history.append(row)

    print(
      f"[INFO] epoch={epoch:04d} train_loss={train_loss:.6f} "
      f"action={train_action_loss:.6f} region={train_region_loss:.6f} "
      f"val_loss={val['loss']:.6f} val_action={val['action_loss']:.6f} "
      f"val_active={val['active_action_loss']:.6f} val_idle={val['idle_action_loss']:.6f} "
      f"val_region={val['region_aux_loss']:.6f} batches={batches} "
      f"train_shards={len(epoch_train_shards)} val={'yes' if ran_val else 'no'}"
    )

    if ran_val and metric < best_metric:
      best_metric = metric
      _shooter_bc._save_checkpoint(out_dir / "model_best.pt", actor, epoch, cfg, row)
    if cfg.save_interval > 0 and epoch % cfg.save_interval == 0:
      _shooter_bc._save_checkpoint(out_dir / f"model_{epoch}.pt", actor, epoch, cfg, row)

  _shooter_bc._save_checkpoint(out_dir / "model_last.pt", actor, cfg.epochs, cfg, history[-1])
  _shooter_bc._save_json(out_dir / "history.json", {"history": history, "best_metric": cfg.best_metric, "best_loss": best_metric})
  print(f"[INFO] Done. best_{cfg.best_metric}={best_metric:.6f}")


def train_goalkeeper_student_bc(cfg: BcConfig) -> None:
  train_goalkeeper_student_bc_impl(cfg)


def main() -> None:
  cfg = tyro.cli(BcConfig, prog="train_goalkeeper_student_bc")
  train_goalkeeper_student_bc(cfg)


if __name__ == "__main__":
  main()
