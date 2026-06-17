"""Bundle six goalkeeper experts into one deployable checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro


@dataclass
class Cfg:
  expert_dir: str = "logs/lyk/experts"
  prefix: str = "stable_sr"
  out: str = "src/assets/soccer/weight/goalkeeper_moe6_lyk.pt"
  z_low: float = 0.85
  z_up: float = 1.35
  vz_low: float = -99.0
  latch_hi: float = 5.0
  land_x: float = 0.0
  mirror_map: str = "1:0,3:2"


def main(cfg: Cfg) -> None:
  experts = []
  for idx in range(6):
    path = Path(cfg.expert_dir) / f"{cfg.prefix}{idx}.pt"
    if not path.exists():
      raise FileNotFoundError(path)
    experts.append(torch.load(path, map_location="cpu", weights_only=False))
  bundle = {
    "moe6": True,
    "sr": experts,
    "z_low": cfg.z_low,
    "z_up": cfg.z_up,
    "vz_low": cfg.vz_low,
    "latch_hi": cfg.latch_hi,
    "land_x": cfg.land_x,
    "mirror_map": cfg.mirror_map,
  }
  out = Path(cfg.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  torch.save(bundle, out)
  print(f"[INFO] saved MoE6 bundle to {out}", flush=True)


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="bundle_moe6"))
