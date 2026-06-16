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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import tyro


_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Cfg:
  base: str = "src/assets/soccer/weight/goalkeeper_distilled_v3.pt"
  repair_data: str = "logs/repairs/repairs_lyk.pt"
  distilled_out: str = "logs/rsl_rl/g1_goalkeeper/distilled/model_repaired_lyk.pt"
  stages: tuple[str, ...] = ("diagnose-base", "prove", "collect", "distill", "diagnose-final")
  device: str = "cuda:0"
  devices: tuple[str, ...] = ()
  num_envs: int = 2048
  diagnose_batches: int = 4
  final_batches: int = 8
  regions: tuple[int, ...] = (3, 2, 1, 0, 4, 5)
  region_weights: tuple[float, ...] = ()
  prove_regions: tuple[int, ...] = (3, 2, 1, 0)
  prove_region_weights: tuple[float, ...] = ()
  G: int = 32
  P: int = 64
  iters: int = 8
  elites: int = 8
  knots: int = 12
  knot_span: int = 80
  release_steps: int = 20
  horizon: int = 150
  w_stable: float = 20.0
  w_final_upright: float = 20.0
  collect_pre_steps: int = 35
  collect_post_steps: int = 12
  require_final_upright: bool = False
  collect_batches: int = 32
  collect_hours: float = 0.0
  collect_batches_per_shard: int = 8
  max_collect_shards: int = 999
  epochs: int = 40
  batch_size: int = 16384
  lr: float = 5.0e-4
  lr_final: float = 5.0e-5
  seed: int = 2810
  continue_on_prove_failure: bool = False


def _script(name: str) -> str:
  return str(_REPO_ROOT / "scripts" / name)


def _run(label: str, args: list[str], env: dict[str, str]) -> None:
  print(f"\n[PIPELINE] {label}", flush=True)
  print("[PIPELINE] " + " ".join(args), flush=True)
  subprocess.run(args, cwd=_REPO_ROOT, env=env, check=True)


def _popen(label: str, args: list[str], env: dict[str, str], log_path: Path) -> subprocess.Popen:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  print(f"\n[PIPELINE] {label}", flush=True)
  print("[PIPELINE] " + " ".join(args), flush=True)
  print(f"[PIPELINE] log -> {log_path}", flush=True)
  log = open(log_path, "w")
  proc = subprocess.Popen(args, cwd=_REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
  proc._pipeline_log = log  # keep the file handle alive until the process exits
  return proc


def _extend_region_args(args: list[str], regions: Sequence[int], weights: Sequence[float]) -> None:
  args.extend(["--regions", *[str(r) for r in regions]])
  if weights:
    args.extend(["--region-weights", *[str(w) for w in weights]])


def _parse_stages(stages: tuple[str, ...]) -> set[str]:
  allowed = {"diagnose-base", "prove", "collect", "distill", "diagnose-final"}
  selected = set(stages)
  unknown = selected - allowed
  if unknown:
    raise ValueError(f"unknown stages: {sorted(unknown)}; allowed={sorted(allowed)}")
  return selected


def _repair_shard_path(repair_data: str, shard_idx: int) -> str:
  path = Path(repair_data)
  return str(path.with_name(f"{path.stem}_shard{shard_idx:03d}{path.suffix}"))


def _existing_repair_data(repair_data: str) -> list[str]:
  path = Path(repair_data)
  shards = sorted(path.parent.glob(f"{path.stem}_shard*.pt"))
  if shards:
    return [str(p) for p in shards]
  return [repair_data]


def main(cfg: Cfg) -> None:
  selected = _parse_stages(cfg.stages)
  env = os.environ.copy()
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
  collected_data: list[str] = []
  collect_devices = tuple(cfg.devices) or (cfg.device,)

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
    ]
    _extend_region_args(prove_cmd, cfg.prove_regions, cfg.prove_region_weights)
    prove_cmd.extend([
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
      "--release-steps",
      str(cfg.release_steps),
      "--horizon",
      str(cfg.horizon),
      "--w-stable",
      str(cfg.w_stable),
      "--w-final-upright",
      str(cfg.w_final_upright),
      "--batches",
      str(max(1, min(4, cfg.collect_batches))),
      "--device",
      cfg.device,
    ])
    try:
      _run("prove repair oracle", prove_cmd, env)
    except subprocess.CalledProcessError:
      if not cfg.continue_on_prove_failure:
        raise
      print("[PIPELINE] prove failed; continuing because continue_on_prove_failure=True", flush=True)

  if "collect" in selected:
    collect_start = time.monotonic()
    collect_deadline = collect_start + max(0.0, cfg.collect_hours) * 3600.0
    shard_count = cfg.max_collect_shards if cfg.collect_hours > 0.0 else 1

    shard_idx = 0
    while shard_idx < shard_count:
      if cfg.collect_hours > 0.0 and shard_idx > 0 and time.monotonic() >= collect_deadline:
        break
      shard_batches = (
        cfg.collect_batches_per_shard
        if cfg.collect_hours > 0.0
        else cfg.collect_batches
      )
      procs: list[tuple[int, str, subprocess.Popen]] = []
      for dev_idx, dev in enumerate(collect_devices):
        if shard_idx >= shard_count:
          break
        if cfg.collect_hours > 0.0 and shard_idx > 0 and time.monotonic() >= collect_deadline:
          break
        shard_out = (
          _repair_shard_path(cfg.repair_data, shard_idx)
          if cfg.collect_hours > 0.0 or len(collect_devices) > 1
          else cfg.repair_data
        )
        collect_cmd = [
          sys.executable,
          _script("repair_oracle.py"),
          "--checkpoint",
          cfg.base,
          "--mode",
          "collect",
        ]
        _extend_region_args(collect_cmd, cfg.regions, cfg.region_weights)
        collect_cmd.extend([
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
          "--release-steps",
          str(cfg.release_steps),
          "--horizon",
          str(cfg.horizon),
          "--w-stable",
          str(cfg.w_stable),
          "--w-final-upright",
          str(cfg.w_final_upright),
          "--collect-pre-steps",
          str(cfg.collect_pre_steps),
          "--collect-post-steps",
          str(cfg.collect_post_steps),
          "--require-final-upright" if cfg.require_final_upright else "--no-require-final-upright",
          "--batches",
          str(shard_batches),
          "--seed",
          str(cfg.seed + shard_idx),
          "--out",
          shard_out,
          "--device",
          dev,
        ])
        if len(collect_devices) == 1:
          _run(f"collect repaired trajectories shard {shard_idx}", collect_cmd, env)
          collected_data.append(shard_out)
        else:
          log_path = Path(cfg.repair_data).parent / f"collect_shard{shard_idx:03d}_{dev.replace(':', '')}.log"
          proc = _popen(
            f"collect repaired trajectories shard {shard_idx} on {dev}",
            collect_cmd,
            env,
            log_path,
          )
          procs.append((shard_idx, shard_out, proc))
        shard_idx += 1
      for done_idx, shard_out, proc in procs:
        code = proc.wait()
        getattr(proc, "_pipeline_log").close()
        if code != 0:
          raise subprocess.CalledProcessError(code, f"collect shard {done_idx}")
        collected_data.append(shard_out)

    if cfg.collect_hours > 0.0:
      elapsed_h = (time.monotonic() - collect_start) / 3600.0
      print(
        f"[PIPELINE] collected {len(collected_data)} shards in {elapsed_h:.2f}h",
        flush=True,
      )

  if "distill" in selected:
    data_paths = collected_data or _existing_repair_data(cfg.repair_data)
    _run(
      "distill repaired goalkeeper",
      [
        sys.executable,
        _script("distill_repairs.py"),
        "--data",
        *data_paths,
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
