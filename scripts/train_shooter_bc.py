"""Train a motion-free shooter student actor with behavior cloning.

The input dataset is produced by ``scripts/collect_shooter_teacher_dataset.py``.
Only successful episodes are used by default, and only timesteps selected by the
collector's ``valid_mask`` contribute to the loss.

For RNN models (default), training uses truncated BPTT over ``bptt_length``
(default 24) timesteps with hidden state carry across chunks.  This matches
the Stage II teacher LSTM training paradigm: the raw ``nn.LSTM`` is called
directly on padded chunk tensors, bypassing RSL-RL's ``unpad_trajectories``
(which requires uniform trajectory lengths for its ``view(-1, T_max, ...)``).

For MLP models (``--model mlp``), training flattens all timesteps and ignores
temporal order.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from rsl_rl.models import MLPModel, RNNModel
from tensordict import TensorDict


@dataclass
class BcConfig:
  dataset_dir: str
  """Dataset run directory, or the shards directory itself."""

  output_dir: str = "logs/bc/shooter_student"
  """Root directory for BC runs."""

  run_name: str | None = None
  overwrite: bool = False
  device: str | None = None
  seed: int = 2810
  epochs: int = 50
  batch_size: int = 4096
  learning_rate: float = 3.0e-4
  weight_decay: float = 0.0
  grad_clip: float = 1.0
  loss: str = "smooth_l1"
  smooth_l1_beta: float = 1.0

  model: str = "rnn"
  """Model type: 'rnn' (LSTM, default) or 'mlp'."""

  hidden_dims: tuple[int, ...] = (128, 64, 32)
  """MLP hidden dims after the (optional) RNN head."""

  rnn_type: str = "lstm"
  rnn_hidden_dim: int = 128
  rnn_num_layers: int = 2

  bptt_length: int = 24
  """Truncated BPTT chunk size (timesteps per forward/backward pass).

  Hidden states are detached between chunks — each chunk is an independent
  gradient computation.  Default 24 matches teacher PPO's ``num_steps_per_env``.
  """

  activation: str = "elu"
  obs_normalization: bool = True
  init_std: float = 0.5

  success_only: bool = True
  val_fraction: float = 0.1
  max_train_batches_per_epoch: int | None = None
  max_val_batches: int | None = 64
  action_clip: float | None = None
  save_interval: int = 10


def _find_shards(dataset_dir: str) -> list[Path]:
  root = Path(dataset_dir).expanduser().resolve()
  shard_dir = root / "shards" if (root / "shards").is_dir() else root
  shards = sorted(shard_dir.glob("shard_*.pt"))
  if not shards:
    raise FileNotFoundError(f"No shard_*.pt files under {shard_dir}")
  return shards


def _load_flat_samples(path: Path, success_only: bool, action_clip: float | None) -> tuple[torch.Tensor, torch.Tensor]:
  """Load all valid timesteps from a shard as flat tensors (used for MLP and normalizer warmup)."""
  payload = torch.load(path, map_location="cpu", weights_only=False)
  obs = payload["student_obs"].float()
  actions = payload["teacher_action"].float()
  mask = payload["valid_mask"].bool()

  if success_only:
    success = payload["metadata"]["success"].bool()
    mask = mask & success[:, None]

  obs_flat = obs[mask]
  action_flat = actions[mask]
  if action_clip is not None:
    action_flat = torch.clamp(action_flat, -action_clip, action_clip)

  finite = torch.isfinite(obs_flat).all(dim=-1) & torch.isfinite(action_flat).all(dim=-1)
  return obs_flat[finite], action_flat[finite]


def _load_episode_sequences(
  path: Path, success_only: bool, action_clip: float | None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
  """Return trimmed per-episode (obs, action) tensors for RNN training."""
  payload = torch.load(path, map_location="cpu", weights_only=False)
  N = int(payload["student_obs"].shape[0])
  episodes: list[tuple[torch.Tensor, torch.Tensor]] = []
  for i in range(N):
    if success_only and not bool(payload["metadata"]["success"][i].item()):
      continue
    mask = payload["valid_mask"][i].bool()
    if not torch.any(mask):
      continue
    obs = payload["student_obs"][i, mask, :].float()
    act = payload["teacher_action"][i, mask, :].float()
    if action_clip is not None:
      act = torch.clamp(act, -action_clip, action_clip)
    finite = torch.isfinite(obs).all(dim=-1) & torch.isfinite(act).all(dim=-1)
    if not torch.any(finite):
      continue
    episodes.append((obs[finite], act[finite]))
  return episodes


def _split_shards(shards: list[Path], cfg: BcConfig) -> tuple[list[Path], list[Path]]:
  if cfg.val_fraction <= 0.0 or len(shards) < 2:
    return shards, []

  rng = np.random.default_rng(cfg.seed)
  order = rng.permutation(len(shards)).tolist()
  n_val = max(1, int(round(len(shards) * cfg.val_fraction)))
  n_val = min(n_val, len(shards) - 1)
  val_ids = set(order[:n_val])
  train = [path for idx, path in enumerate(shards) if idx not in val_ids]
  val = [path for idx, path in enumerate(shards) if idx in val_ids]
  return train, val


def _infer_dims(shards: list[Path]) -> tuple[int, int]:
  payload = torch.load(shards[0], map_location="cpu", weights_only=False)
  return int(payload["student_obs"].shape[-1]), int(payload["teacher_action"].shape[-1])


def _make_obs_tensordict(obs: torch.Tensor) -> TensorDict:
  return TensorDict({"student": obs}, batch_size=list(obs.shape[:-1]))


def _make_actor(obs_dim: int, action_dim: int, cfg: BcConfig, device: str) -> MLPModel | RNNModel:
  dummy_obs = _make_obs_tensordict(torch.zeros(1, obs_dim))
  model_cls = RNNModel if cfg.model == "rnn" else MLPModel
  kwargs: dict[str, Any] = {
    "hidden_dims": cfg.hidden_dims,
    "activation": cfg.activation,
    "obs_normalization": cfg.obs_normalization,
    "distribution_cfg": {
      "class_name": "GaussianDistribution",
      "init_std": cfg.init_std,
      "std_type": "scalar",
    },
  }
  if cfg.model == "rnn":
    kwargs.update({
      "rnn_type": cfg.rnn_type,
      "rnn_hidden_dim": cfg.rnn_hidden_dim,
      "rnn_num_layers": cfg.rnn_num_layers,
    })
  actor = model_cls(
    dummy_obs,
    obs_groups={"actor": ["student"]},
    obs_set="actor",
    output_dim=action_dim,
    **kwargs,
  )
  return actor.to(device)


def _update_obs_normalizer(actor, shards: list[Path], cfg: BcConfig, device: str) -> int:
  if not cfg.obs_normalization:
    return 0

  actor.train()
  total = 0
  with torch.no_grad():
    for shard in shards:
      obs, _ = _load_flat_samples(shard, cfg.success_only, cfg.action_clip)
      if obs.numel() == 0:
        continue
      total += int(obs.shape[0])
      for start in range(0, obs.shape[0], cfg.batch_size):
        batch = obs[start:start + cfg.batch_size].to(device)
        actor.update_normalization(_make_obs_tensordict(batch))
  return total


def _build_padded_batch(
  episodes: list[tuple[torch.Tensor, torch.Tensor]], device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Pad a batch of variable-length episodes to the same max length.

  Returns:
    obs_padded:  (T_max, B, obs_dim)  — trailing zeros for short episodes
    act_padded:  (T_max, B, act_dim)  — trailing zeros, masked out in loss
    masks:       (T_max, B)           — True = valid timestep
  """
  max_t = max(ep[0].shape[0] for ep in episodes)
  dim_obs = episodes[0][0].shape[1]
  dim_act = episodes[0][1].shape[1]
  obs_padded = torch.zeros(max_t, len(episodes), dim_obs, device=device)
  act_padded = torch.zeros(max_t, len(episodes), dim_act, device=device)
  masks = torch.zeros(max_t, len(episodes), dtype=torch.bool, device=device)
  for j, (obs, act) in enumerate(episodes):
    t = obs.shape[0]
    obs_padded[:t, j] = obs.to(device)
    act_padded[:t, j] = act.to(device)
    masks[:t, j] = True
  return obs_padded, act_padded, masks


def _latin_hidden_state(actor, batch_size: int, device: str):
  """Return zero initial hidden state for a batch of B independent episodes.

  Shape: ``(num_layers, B, hidden_dim)`` for GRU, or a tuple of two such
  tensors for LSTM.
  """
  rnn = actor.rnn.rnn
  num_layers = rnn.num_layers
  hidden_dim = rnn.hidden_size
  h0 = torch.zeros(num_layers, batch_size, hidden_dim, device=device)
  if isinstance(rnn, nn.LSTM):
    return (h0.clone(), h0.clone())
  return h0


def _detach_hidden_state(hidden_state):
  """Detach hidden state for truncated BPTT boundary."""
  if isinstance(hidden_state, tuple):
    return tuple(h.detach() for h in hidden_state)
  return hidden_state.detach()


def _forward_rnn_chunk(
  actor, obs_chunk: torch.Tensor, masks_chunk: torch.Tensor,
  hidden_state=None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Run one chunk of truncated BPTT through the LSTM + MLP.

  Directly calls the raw ``nn.LSTM`` to get full temporal recurrence
  (bypassing RSL-RL's ``RNN`` wrapper and its ``unpad_trajectories``).

  Returns:
    pred:         (valid_in_chunk, action_dim)  — MLP output at valid timesteps
    new_hs:       detached hidden state at chunk end, for the next chunk
  """
  # Step 1: concat observation groups + normalize (same as MLPModel.get_latent)
  obs_list = [obs_chunk]  # single group "student"
  latent = torch.cat(obs_list, dim=-1)            # (L, B, obs_dim)
  if actor.obs_normalization:
    latent = actor.obs_normalizer(latent)          # (L, B, obs_dim)

  # Step 2: zero out padded positions so the LSTM sees clean zeros
  latent = latent * masks_chunk.unsqueeze(-1).float()

  # Step 3: raw LSTM forward — full recurrence within this chunk
  if hidden_state is None:
    hidden_state = _latin_hidden_state(actor, obs_chunk.shape[1], obs_chunk.device)
  rnn_out, new_hs = actor.rnn.rnn(latent, hidden_state)
  # rnn_out: (L, B, hidden_dim) — includes outputs at masked positions (discarded)

  # Step 4: MLP only on valid positions
  mlp_out = actor.mlp(rnn_out[masks_chunk])        # (valid_in_chunk, mlp_output_dim)

  # Step 5: distribution → mean
  pred = actor.distribution.deterministic_output(mlp_out)  # (valid_in_chunk, action_dim)

  return pred, _detach_hidden_state(new_hs)


def _forward_rnn_full(
  actor, obs_padded: torch.Tensor, act_padded: torch.Tensor, masks: torch.Tensor,
  cfg: BcConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Run truncated BPTT over all chunks of a padded batch, return (pred, target).

  Chunks iterator length ``bptt_length``, hidden states are detached at chunk
  boundaries (standard truncated BPTT).  Optimizer steps between chunks are
  handled by the caller.
  """
  T_max = obs_padded.shape[0]
  hidden_state = None
  all_preds: list[torch.Tensor] = []
  all_targets: list[torch.Tensor] = []

  for start in range(0, T_max, cfg.bptt_length):
    end = min(start + cfg.bptt_length, T_max)
    obs_chunk = obs_padded[start:end]               # (L, B, obs_dim)
    act_chunk = act_padded[start:end]               # (L, B, act_dim)
    mask_chunk = masks[start:end]                    # (L, B)

    if not torch.any(mask_chunk):
      continue

    pred, hidden_state = _forward_rnn_chunk(actor, obs_chunk, mask_chunk, hidden_state)
    tgt = act_chunk[mask_chunk]                      # (valid, act_dim)

    all_preds.append(pred)
    all_targets.append(tgt)

  if not all_preds:
    return (
      torch.zeros(0, act_padded.shape[-1], device=act_padded.device),
      torch.zeros(0, act_padded.shape[-1], device=act_padded.device),
    )
  return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)


def _iter_rnn_minibatches(
  shards: list[Path],
  cfg: BcConfig,
  device: str,
  generator: torch.Generator,
  shuffle: bool,
):
  """Yield padded batches ``(obs, act_padded, masks)`` for RNN BC training.

  Loads all episodes into memory for proper mini-batch composition.
  Each yield gives one mini-batch of B episodes padded to the same length.
  The caller handles chunked BPTT within each mini-batch.
  """
  episodes: list[tuple[torch.Tensor, torch.Tensor]] = []
  for shard in shards:
    episodes.extend(_load_episode_sequences(shard, cfg.success_only, cfg.action_clip))
  if not episodes:
    return

  order = torch.randperm(len(episodes), generator=generator) if shuffle else torch.arange(len(episodes))
  yielded = 0
  for start in range(0, len(episodes), cfg.batch_size):
    ids = order[start:start + cfg.batch_size].tolist()
    batch_eps = [episodes[i] for i in ids]
    yield _build_padded_batch(batch_eps, device)
    yielded += 1
    if cfg.max_train_batches_per_epoch is not None and shuffle and yielded >= cfg.max_train_batches_per_epoch:
      return
    if cfg.max_val_batches is not None and not shuffle and yielded >= cfg.max_val_batches:
      return


def _iter_mlp_minibatches(
  shards: list[Path],
  cfg: BcConfig,
  device: str,
  generator: torch.Generator,
  shuffle: bool,
):
  """Yield flat (obs, actions) batches for MLP BC training."""
  shard_order = torch.randperm(len(shards), generator=generator).tolist() if shuffle else list(range(len(shards)))
  yielded = 0
  for shard_idx in shard_order:
    obs, actions = _load_flat_samples(shards[shard_idx], cfg.success_only, cfg.action_clip)
    if obs.numel() == 0:
      continue

    sample_order = torch.randperm(obs.shape[0], generator=generator) if shuffle else torch.arange(obs.shape[0])
    for start in range(0, obs.shape[0], cfg.batch_size):
      ids = sample_order[start:start + cfg.batch_size]
      yield obs[ids].to(device), actions[ids].to(device)
      yielded += 1
      if cfg.max_train_batches_per_epoch is not None and shuffle and yielded >= cfg.max_train_batches_per_epoch:
        return
      if cfg.max_val_batches is not None and not shuffle and yielded >= cfg.max_val_batches:
        return


def _bc_loss(pred: torch.Tensor, actions: torch.Tensor, cfg: BcConfig, reduction: str = "mean") -> torch.Tensor:
  if cfg.loss == "smooth_l1":
    return F.smooth_l1_loss(pred, actions, beta=cfg.smooth_l1_beta, reduction=reduction)
  if cfg.loss == "mse":
    return F.mse_loss(pred, actions, reduction=reduction)
  raise ValueError(f"Unsupported BC loss: {cfg.loss!r}. Use 'smooth_l1' or 'mse'.")


def _evaluate(actor, shards: list[Path], cfg: BcConfig, device: str) -> dict[str, float]:
  if not shards:
    return {"loss": float("nan"), "samples": 0.0}

  actor.eval()
  is_rnn = cfg.model == "rnn"
  generator = torch.Generator().manual_seed(cfg.seed + 17)
  total_loss = 0.0
  total_samples = 0
  with torch.no_grad():
    if is_rnn:
      for obs_padded, act_padded, masks in _iter_rnn_minibatches(shards, cfg, device, generator, shuffle=False):
        pred, tgt = _forward_rnn_full(actor, obs_padded, act_padded, masks, cfg)
        if pred.numel() == 0:
          continue
        loss = _bc_loss(pred, tgt, cfg, reduction="sum")
        total_loss += float(loss.item())
        total_samples += int(tgt.numel())
    else:
      for obs, actions in _iter_mlp_minibatches(shards, cfg, device, generator, shuffle=False):
        pred = actor(_make_obs_tensordict(obs))
        loss = _bc_loss(pred, actions, cfg, reduction="sum")
        total_loss += float(loss.item())
        total_samples += int(actions.numel())

  if total_samples == 0:
    return {"loss": float("nan"), "samples": 0.0}
  return {"loss": total_loss / total_samples, "samples": float(total_samples)}


def _save_checkpoint(path: Path, actor, epoch: int, cfg: BcConfig, metrics: dict[str, Any]) -> None:
  torch.save(
    {
      "actor_state_dict": actor.state_dict(),
      "iter": epoch,
      "infos": {"bc": metrics},
      "config": asdict(cfg),
    },
    path,
  )


def _save_json(path: Path, payload: dict[str, Any]) -> None:
  path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def train_bc(cfg: BcConfig) -> None:
  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  is_rnn = cfg.model == "rnn"

  shards = _find_shards(cfg.dataset_dir)
  train_shards, val_shards = _split_shards(shards, cfg)
  obs_dim, action_dim = _infer_dims(shards)

  run_tag = cfg.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  out_dir = Path(cfg.output_dir).expanduser().resolve() / run_tag
  if out_dir.exists() and any(out_dir.iterdir()) and not cfg.overwrite:
    raise FileExistsError(f"Output directory already exists and is not empty: {out_dir}")
  out_dir.mkdir(parents=True, exist_ok=True)

  actor = _make_actor(obs_dim, action_dim, cfg, device)
  normalizer_samples = _update_obs_normalizer(actor, train_shards, cfg, device)
  optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
  generator = torch.Generator().manual_seed(cfg.seed)

  metadata: dict[str, Any] = {
    "dataset_dir": str(Path(cfg.dataset_dir).expanduser().resolve()),
    "num_shards": len(shards),
    "train_shards": len(train_shards),
    "val_shards": len(val_shards),
    "obs_dim": obs_dim,
    "action_dim": action_dim,
    "model": cfg.model,
    "normalizer_samples": normalizer_samples,
    "config": asdict(cfg),
  }
  _save_json(out_dir / "config.json", metadata)

  print(f"[INFO] BC training on {device}  model={cfg.model}")
  print(f"[INFO] Output: {out_dir}")
  print(f"[INFO] Shards: train={len(train_shards)}, val={len(val_shards)}, obs_dim={obs_dim}, action_dim={action_dim}")
  if cfg.success_only:
    print("[INFO] Filtering to successful episodes only.")
  if cfg.obs_normalization:
    print(f"[INFO] Updated obs normalizer with {normalizer_samples} samples.")

  _iter_train = _iter_rnn_minibatches if is_rnn else _iter_mlp_minibatches

  best_metric = float("inf")
  history: list[dict[str, Any]] = []
  for epoch in range(1, cfg.epochs + 1):
    actor.train()
    total_loss = 0.0
    total_samples = 0
    batches = 0

    for batch_args in _iter_train(train_shards, cfg, device, generator, shuffle=True):
      if is_rnn:
        obs_padded, act_padded, masks = batch_args
        T_max = obs_padded.shape[0]
        hidden_state = None
        batch_loss = 0.0
        batch_samples = 0
        chunk_count = 0

        for start in range(0, T_max, cfg.bptt_length):
          end = min(start + cfg.bptt_length, T_max)
          obs_chunk = obs_padded[start:end]
          act_chunk = act_padded[start:end]
          mask_chunk = masks[start:end]
          if not torch.any(mask_chunk):
            continue

          pred, hidden_state = _forward_rnn_chunk(actor, obs_chunk, mask_chunk, hidden_state)
          tgt = act_chunk[mask_chunk]
          chunk_loss = _bc_loss(pred, tgt, cfg)
          chunk_samples = int(tgt.numel())

          optimizer.zero_grad(set_to_none=True)
          chunk_loss.backward()
          if cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
          optimizer.step()

          batch_loss += float(chunk_loss.item()) * chunk_samples
          batch_samples += chunk_samples
          chunk_count += 1

        total_loss += batch_loss
        total_samples += batch_samples
        batches += 1
      else:
        obs_flat, actions_flat = batch_args
        pred = actor(_make_obs_tensordict(obs_flat))
        loss = _bc_loss(pred, actions_flat, cfg)
        sample_count = int(actions_flat.numel())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0.0:
          torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += float(loss.item()) * sample_count
        total_samples += sample_count
        batches += 1

    if batches == 0 or total_samples == 0:
      raise RuntimeError(
        "No BC samples found. Check dataset success rate or rerun with --success-only False."
      )

    train_loss = total_loss / total_samples
    val = _evaluate(actor, val_shards, cfg, device)
    metric = val["loss"] if np.isfinite(val["loss"]) else train_loss
    row = {
      "epoch": epoch,
      "train_loss": train_loss,
      "val_loss": val["loss"],
      "train_action_values": total_samples,
      "val_action_values": int(val["samples"]),
      "batches": batches,
    }
    history.append(row)

    print(
      f"[INFO] epoch={epoch:04d} train_loss={train_loss:.6f} "
      f"val_loss={val['loss']:.6f} batches={batches}"
    )

    if metric < best_metric:
      best_metric = metric
      _save_checkpoint(out_dir / "model_best.pt", actor, epoch, cfg, row)
    if cfg.save_interval > 0 and epoch % cfg.save_interval == 0:
      _save_checkpoint(out_dir / f"model_{epoch}.pt", actor, epoch, cfg, row)

  _save_checkpoint(out_dir / "model_last.pt", actor, cfg.epochs, cfg, history[-1])
  _save_json(out_dir / "history.json", {"history": history, "best_loss": best_metric})
  print(f"[INFO] Done. best_loss={best_metric:.6f}")


def main() -> None:
  cfg = tyro.cli(BcConfig, prog="train_shooter_bc")
  train_bc(cfg)


if __name__ == "__main__":
  main()
