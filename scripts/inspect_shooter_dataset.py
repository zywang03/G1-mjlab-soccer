"""Inspect shooter teacher rollout datasets collected for student BC."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro


@dataclass
class InspectConfig:
  dataset_dir: str
  """Dataset run directory containing a shards/ subdirectory."""

  output_json: str | None = None
  """Optional path to write the computed summary JSON."""

  bins_x: int = 5
  bins_y: int = 5


def _load_metadata(dataset_dir: Path) -> dict[str, torch.Tensor]:
  shard_dir = dataset_dir / "shards"
  shards = sorted(shard_dir.glob("shard_*.pt"))
  if not shards:
    raise FileNotFoundError(f"No shard_*.pt files under {shard_dir}")

  all_meta: dict[str, list[torch.Tensor]] = {}
  for shard in shards:
    payload = torch.load(shard, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    for key, value in metadata.items():
      if not isinstance(value, torch.Tensor):
        continue
      all_meta.setdefault(key, []).append(value.cpu())

  return {key: torch.cat(values, dim=0) for key, values in all_meta.items()}


def _finite_mean(values: torch.Tensor) -> float:
  finite = torch.isfinite(values)
  if not torch.any(finite):
    return 0.0
  return float(values[finite].mean().item())


def _finite_std(values: torch.Tensor) -> float:
  finite = torch.isfinite(values)
  if not torch.any(finite):
    return 0.0
  return float(values[finite].std(unbiased=False).item())


def _side_counts(side: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, int]:
  if mask is not None:
    side = side[mask]
  return {
    "none": int((side < 0).sum().item()),
    "left": int((side == 0).sum().item()),
    "right": int((side == 1).sum().item()),
  }


def _destination_heatmap(metadata: dict[str, torch.Tensor], bins_x: int, bins_y: int) -> list[list[dict[str, Any]]]:
  dest = metadata["destination"]
  success = metadata["success"]
  if dest.numel() == 0:
    return []

  x = dest[:, 0]
  y = dest[:, 1]
  x_edges = torch.linspace(float(x.min()), float(x.max()) + 1e-6, bins_x + 1)
  y_edges = torch.linspace(float(y.min()), float(y.max()) + 1e-6, bins_y + 1)
  heatmap: list[list[dict[str, Any]]] = []
  for iy in range(bins_y):
    row: list[dict[str, Any]] = []
    for ix in range(bins_x):
      mask = (x >= x_edges[ix]) & (x < x_edges[ix + 1]) & (y >= y_edges[iy]) & (y < y_edges[iy + 1])
      count = int(mask.sum().item())
      succ = int((success & mask).sum().item())
      row.append({"count": count, "success": succ, "rate": succ / max(count, 1)})
    heatmap.append(row)
  return heatmap


def inspect_dataset(cfg: InspectConfig) -> dict[str, Any]:
  dataset_dir = Path(cfg.dataset_dir).expanduser().resolve()
  metadata = _load_metadata(dataset_dir)

  total = int(metadata["success"].numel())
  success = metadata["success"]
  valid_kick = metadata["valid_kick_step"] >= 0
  crossed = metadata["ball_cross_step"] >= 0
  goal = metadata["goal_success"]
  early_term = metadata["early_terminated"]
  nonfoot = metadata["nonfoot_contact_any"]
  both_feet = metadata["both_feet_contact_any"]

  summary: dict[str, Any] = {
    "dataset_dir": str(dataset_dir),
    "episodes": total,
    "success": int(success.sum().item()),
    "success_rate": float(success.float().mean().item()) if total else 0.0,
    "valid_kick": int(valid_kick.sum().item()),
    "valid_kick_rate": float(valid_kick.float().mean().item()) if total else 0.0,
    "ball_crossed_goal_plane": int(crossed.sum().item()),
    "goal_inside_frame": int(goal.sum().item()),
    "early_terminated": int(early_term.sum().item()),
    "nonfoot_contact_any": int(nonfoot.sum().item()),
    "both_feet_contact_any": int(both_feet.sum().item()),
    "actual_kick_side_all": _side_counts(metadata["actual_kick_side"]),
    "actual_kick_side_success": _side_counts(metadata["actual_kick_side"], success),
    "target_error_all_mean": _finite_mean(metadata["target_error"]),
    "target_error_all_std": _finite_std(metadata["target_error"]),
    "target_error_success_mean": _finite_mean(metadata["target_error"][success]),
    "target_error_success_std": _finite_std(metadata["target_error"][success]),
    "kick_speed_all_mean": _finite_mean(metadata["kick_speed"]),
    "kick_speed_all_std": _finite_std(metadata["kick_speed"]),
    "kick_speed_success_mean": _finite_mean(metadata["kick_speed"][success]),
    "kick_speed_success_std": _finite_std(metadata["kick_speed"][success]),
    "destination_heatmap": _destination_heatmap(metadata, cfg.bins_x, cfg.bins_y),
  }

  print(json.dumps(summary, indent=2, ensure_ascii=False))
  if cfg.output_json is not None:
    out = Path(cfg.output_json).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
  return summary


def main() -> None:
  cfg = tyro.cli(InspectConfig, prog="inspect_shooter_dataset")
  inspect_dataset(cfg)


if __name__ == "__main__":
  main()
