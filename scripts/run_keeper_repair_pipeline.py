"""Run the goalkeeper repair-oracle pipeline end to end.

This is an automation wrapper around:
  1. diagnose_gk.py       -- measure the base failure structure
  2. repair_oracle.py     -- prove CEM repairs help on sampled scenarios
  3. repair_oracle.py     -- collect repaired (obs, action) pairs
  4. distill_repairs.py   -- distill those pairs into a native MLP checkpoint
  5. diagnose_gk.py       -- evaluate the distilled checkpoint

The script intentionally shells out to the individual tools so each step keeps
its normal logging and can be resumed independently.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro


_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Cfg:
  base: str = "src/assets/soccer/weight/goalkeeper_distilled_v3.pt"
  repair_data: str = "logs/repairs/repairs_lyk.pt"
  distilled_out: str = "logs/rsl_rl/g1_goalkeeper/distilled/model_repaired_lyk.pt"
  stages: tuple[str, ...] = ("diagnose-base", "prove", "collect", "distill", "diagnose-final")
  device: str = "cuda:0"
  num_envs: int = 2048
  diagnose_batches: int = 4
  final_batches: int = 8
  regions: tuple[int, ...] = (3, 2, 1, 0, 4, 5)
  prove_regions: tuple[int, ...] = (3, 2, 1, 0)
  G: int = 32
  P: int = 64
  iters: int = 8
  elites: int = 8
  knots: int = 12
  knot_span: int = 80
  horizon: int = 150
  collect_batches: int = 32
  epochs: int = 40
  batch_size: int = 16384
  lr: float = 5.0e-4
  lr_final: float = 5.0e-5
  continue_on_prove_failure: bool = False


def _script(name: str) -> str:
  return str(_REPO_ROOT / "scripts" / name)


def _run(label: str, args: list[str], env: dict[str, str]) -> None:
  print(f"\n[PIPELINE] {label}", flush=True)
  print("[PIPELINE] " + " ".join(args), flush=True)
  subprocess.run(args, cwd=_REPO_ROOT, env=env, check=True)


def _parse_stages(stages: tuple[str, ...]) -> set[str]:
  allowed = {"diagnose-base", "prove", "collect", "distill", "diagnose-final"}
  selected = set(stages)
  unknown = selected - allowed
  if unknown:
    raise ValueError(f"unknown stages: {sorted(unknown)}; allowed={sorted(allowed)}")
  return selected


def main(cfg: Cfg) -> None:
  selected = _parse_stages(cfg.stages)
  env = os.environ.copy()
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("MPLCONFIGDIR", "/tmp/mpl")

  if "diagnose-base" in selected:
    _run(
      "diagnose base goalkeeper",
      [
        sys.executable,
        _script("diagnose_gk.py"),
        "--checkpoint",
        cfg.base,
        "--num-envs",
        str(cfg.num_envs),
        "--batches",
        str(cfg.diagnose_batches),
        "--device",
        cfg.device,
      ],
      env,
    )

  if "prove" in selected:
    prove_cmd = [
      sys.executable,
      _script("repair_oracle.py"),
      "--checkpoint",
      cfg.base,
      "--mode",
      "prove",
      "--regions",
      *[str(r) for r in cfg.prove_regions],
      "--G",
      str(cfg.G),
      "--P",
      str(cfg.P),
      "--iters",
      str(cfg.iters),
      "--elites",
      str(cfg.elites),
      "--knots",
      str(cfg.knots),
      "--knot-span",
      str(cfg.knot_span),
      "--horizon",
      str(cfg.horizon),
      "--batches",
      str(max(1, min(4, cfg.collect_batches))),
      "--device",
      cfg.device,
    ]
    try:
      _run("prove repair oracle", prove_cmd, env)
    except subprocess.CalledProcessError:
      if not cfg.continue_on_prove_failure:
        raise
      print("[PIPELINE] prove failed; continuing because continue_on_prove_failure=True", flush=True)

  if "collect" in selected:
    _run(
      "collect repaired trajectories",
      [
        sys.executable,
        _script("repair_oracle.py"),
        "--checkpoint",
        cfg.base,
        "--mode",
        "collect",
        "--regions",
        *[str(r) for r in cfg.regions],
        "--G",
        str(cfg.G),
        "--P",
        str(cfg.P),
        "--iters",
        str(cfg.iters),
        "--elites",
        str(cfg.elites),
        "--knots",
        str(cfg.knots),
        "--knot-span",
        str(cfg.knot_span),
        "--horizon",
        str(cfg.horizon),
        "--batches",
        str(cfg.collect_batches),
        "--out",
        cfg.repair_data,
        "--device",
        cfg.device,
      ],
      env,
    )

  if "distill" in selected:
    _run(
      "distill repaired goalkeeper",
      [
        sys.executable,
        _script("distill_repairs.py"),
        "--data",
        cfg.repair_data,
        "--resume",
        cfg.base,
        "--out",
        cfg.distilled_out,
        "--epochs",
        str(cfg.epochs),
        "--batch-size",
        str(cfg.batch_size),
        "--lr",
        str(cfg.lr),
        "--lr-final",
        str(cfg.lr_final),
        "--device",
        cfg.device,
      ],
      env,
    )

  if "diagnose-final" in selected:
    _run(
      "diagnose repaired goalkeeper",
      [
        sys.executable,
        _script("diagnose_gk.py"),
        "--checkpoint",
        cfg.distilled_out,
        "--num-envs",
        str(cfg.num_envs),
        "--batches",
        str(cfg.final_batches),
        "--device",
        cfg.device,
      ],
      env,
    )


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="run_keeper_repair_pipeline"))
