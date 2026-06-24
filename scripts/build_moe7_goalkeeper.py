"""Build a deployable goalkeeper MoE bundle with an optional prepare expert."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro


@dataclass
class Cfg:
  moe6: str = "src/assets/soccer/weight/goalkeeper_moe6.pt"
  idle: str = "logs/adversarial/keeper_idle/round_001/keeper_idle_best.pt"
  out: str = "logs/repairs/goalkeeper_moe7_prepare.pt"
  gate: str = ""
  idle_speed_threshold: float = 0.5
  idle_incoming_vx_threshold: float = -0.5


def _actor_only_checkpoint(path: str) -> dict:
  ckpt = torch.load(path, map_location="cpu", weights_only=False)
  if "actor_state_dict" not in ckpt:
    raise KeyError(f"{path} does not contain actor_state_dict")
  return ckpt


def main(cfg: Cfg) -> None:
  bundle = torch.load(cfg.moe6, map_location="cpu", weights_only=False)
  if "sr" not in bundle or len(bundle["sr"]) != 6:
    raise ValueError(f"{cfg.moe6} is not a 6-region goalkeeper MoE bundle")

  bundle = dict(bundle)
  bundle["idle"] = _actor_only_checkpoint(cfg.idle)
  bundle["idle_speed_threshold"] = cfg.idle_speed_threshold
  bundle["idle_incoming_vx_threshold"] = cfg.idle_incoming_vx_threshold
  if cfg.gate:
    bundle["gate"] = torch.load(cfg.gate, map_location="cpu", weights_only=False)

  out = Path(cfg.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  torch.save(bundle, out)
  print(f"[INFO] saved MoE7 goalkeeper bundle to {out}", flush=True)


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="build_moe7_goalkeeper"))
