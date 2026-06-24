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


def _iter_streaming_rnn_minibatches(
  shards: list[Path],
  cfg: BcConfig,
  device: str,
  generator: torch.Generator,
  shuffle: bool,
):
  """Stream goalkeeper RNN BC batches without loading every shard at once."""
  shard_order = torch.randperm(len(shards), generator=generator).tolist() if shuffle else list(range(len(shards)))
  pending: list[tuple[torch.Tensor, torch.Tensor]] = []
  yielded = 0

  for shard_idx in shard_order:
    shard = shards[shard_idx]
    episodes = _shooter_bc._load_episode_sequences(shard, cfg.success_only, cfg.action_clip)
    if not episodes:
      continue
    if shuffle:
      order = torch.randperm(len(episodes), generator=generator).tolist()
      episodes = [episodes[idx] for idx in order]
    pending.extend(episodes)

    while len(pending) >= cfg.batch_size:
      batch_eps = pending[:cfg.batch_size]
      pending = pending[cfg.batch_size:]
      yield _shooter_bc._build_padded_batch(batch_eps, device)
      yielded += 1
      if cfg.max_train_batches_per_epoch is not None and shuffle and yielded >= cfg.max_train_batches_per_epoch:
        return
      if cfg.max_val_batches is not None and not shuffle and yielded >= cfg.max_val_batches:
        return

  if pending:
    yield _shooter_bc._build_padded_batch(pending, device)


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
  condition_idle = student_obs[..., -1] > 0.5
  kinematic_idle = (speed < idle_speed_threshold) | (vx >= idle_incoming_vx_threshold)
  idle = condition_idle | kinematic_idle
  return torch.where(idle, torch.full_like(route, 6), route)


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
    return {"loss": float("nan"), "action_loss": float("nan"), "region_aux_loss": float("nan"), "samples": 0.0}

  actor.eval()
  generator = torch.Generator().manual_seed(cfg.seed + 17)
  total_loss = 0.0
  total_action_loss = 0.0
  total_action_values = 0
  total_region_loss = 0.0
  total_region_samples = 0
  with torch.no_grad():
    for obs_padded, act_padded, masks in _iter_streaming_rnn_minibatches(shards, cfg, device, generator, shuffle=False):
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
        action_loss = _shooter_bc._bc_loss(pred, target, cfg, reduction="sum")
        region_loss, region_samples = _goalkeeper_region_aux_loss(
          actor,
          obs_chunk,
          mask_chunk,
          cfg,
          route_gate,
          reduction="sum",
        )
        action_values = int(target.numel())
        total_action_loss += float(action_loss.item())
        total_action_values += action_values
        if region_samples > 0:
          total_region_loss += float(region_loss.item())
          total_region_samples += region_samples
        total_loss += float(action_loss.item()) + cfg.region_aux_coef * float(region_loss.item())

  if total_action_values == 0:
    return {"loss": float("nan"), "action_loss": float("nan"), "region_aux_loss": float("nan"), "samples": 0.0}
  action_mean = total_action_loss / total_action_values
  region_mean = total_region_loss / total_region_samples if total_region_samples > 0 else 0.0
  return {
    "loss": action_mean + cfg.region_aux_coef * region_mean,
    "action_loss": action_mean,
    "region_aux_loss": region_mean,
    "samples": float(total_action_values),
    "region_samples": float(total_region_samples),
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
  obs_dim, action_dim = _shooter_bc._infer_dims(shards)
  run_tag = cfg.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  out_dir = Path(cfg.output_dir).expanduser().resolve() / run_tag
  if out_dir.exists() and any(out_dir.iterdir()) and not cfg.overwrite:
    raise FileExistsError(f"Output directory already exists and is not empty: {out_dir}")
  out_dir.mkdir(parents=True, exist_ok=True)

  actor = make_goalkeeper_student_actor(obs_dim, action_dim, cfg, device)
  normalizer_samples = _update_goalkeeper_obs_normalizer(actor, train_shards, cfg, device)
  route_gate = _load_goalkeeper_route_gate(cfg.teacher, device)
  optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
  generator = torch.Generator().manual_seed(cfg.seed)

  metadata = {
    "dataset_dir": str(Path(cfg.dataset_dir).expanduser().resolve()),
    "num_shards": len(shards),
    "train_shards": len(train_shards),
    "val_shards": len(val_shards),
    "obs_dim": obs_dim,
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

  best_metric = float("inf")
  history: list[dict] = []
  for epoch in range(1, cfg.epochs + 1):
    actor.train()
    total_action_loss = 0.0
    total_action_values = 0
    total_region_loss = 0.0
    total_region_samples = 0
    batches = 0

    for obs_padded, act_padded, masks in _iter_streaming_rnn_minibatches(
      train_shards, cfg, device, generator, shuffle=True
    ):
      T_max = obs_padded.shape[0]
      hidden_state = None
      batch_loss = 0.0
      batch_action_values = 0
      batch_region_loss = 0.0
      batch_region_samples = 0

      for start in range(0, T_max, cfg.bptt_length):
        end = min(start + cfg.bptt_length, T_max)
        obs_chunk = obs_padded[start:end]
        act_chunk = act_padded[start:end]
        mask_chunk = masks[start:end]
        if not torch.any(mask_chunk):
          continue

        pred, hidden_state = _shooter_bc._forward_rnn_chunk(actor, obs_chunk, mask_chunk, hidden_state)
        target = act_chunk[mask_chunk]
        action_loss = _shooter_bc._bc_loss(pred, target, cfg)
        region_loss, region_samples = _goalkeeper_region_aux_loss(actor, obs_chunk, mask_chunk, cfg, route_gate)
        loss = action_loss + cfg.region_aux_coef * region_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0.0:
          torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
        optimizer.step()

        action_values = int(target.numel())
        total_action_loss += float(action_loss.item()) * action_values
        total_action_values += action_values
        batch_loss += float(loss.item()) * action_values
        batch_action_values += action_values
        if region_samples > 0:
          total_region_loss += float(region_loss.item()) * region_samples
          total_region_samples += region_samples
          batch_region_loss += float(region_loss.item()) * region_samples
          batch_region_samples += region_samples

      if batch_action_values > 0:
        batches += 1

    if batches == 0 or total_action_values == 0:
      raise RuntimeError("No BC samples found. Check dataset or valid masks.")

    train_action_loss = total_action_loss / total_action_values
    train_region_loss = total_region_loss / total_region_samples if total_region_samples > 0 else 0.0
    train_loss = train_action_loss + cfg.region_aux_coef * train_region_loss
    val = _evaluate_goalkeeper_bc(actor, val_shards, cfg, device, route_gate)
    metric = val["loss"] if torch.isfinite(torch.tensor(val["loss"])) else train_loss
    row = {
      "epoch": epoch,
      "train_loss": train_loss,
      "train_action_loss": train_action_loss,
      "train_region_aux_loss": train_region_loss,
      "val_loss": val["loss"],
      "val_action_loss": val["action_loss"],
      "val_region_aux_loss": val["region_aux_loss"],
      "train_action_values": total_action_values,
      "train_region_samples": total_region_samples,
      "val_action_values": int(val["samples"]),
      "val_region_samples": int(val.get("region_samples", 0.0)),
      "batches": batches,
    }
    history.append(row)

    print(
      f"[INFO] epoch={epoch:04d} train_loss={train_loss:.6f} "
      f"action={train_action_loss:.6f} region={train_region_loss:.6f} "
      f"val_loss={val['loss']:.6f} val_action={val['action_loss']:.6f} "
      f"val_region={val['region_aux_loss']:.6f} batches={batches}"
    )

    if metric < best_metric:
      best_metric = metric
      _shooter_bc._save_checkpoint(out_dir / "model_best.pt", actor, epoch, cfg, row)
    if cfg.save_interval > 0 and epoch % cfg.save_interval == 0:
      _shooter_bc._save_checkpoint(out_dir / f"model_{epoch}.pt", actor, epoch, cfg, row)

  _shooter_bc._save_checkpoint(out_dir / "model_last.pt", actor, cfg.epochs, cfg, history[-1])
  _shooter_bc._save_json(out_dir / "history.json", {"history": history, "best_loss": best_metric})
  print(f"[INFO] Done. best_loss={best_metric:.6f}")


def train_goalkeeper_student_bc(cfg: BcConfig) -> None:
  train_goalkeeper_student_bc_impl(cfg)


def main() -> None:
  cfg = tyro.cli(BcConfig, prog="train_goalkeeper_student_bc")
  train_goalkeeper_student_bc(cfg)


if __name__ == "__main__":
  main()
