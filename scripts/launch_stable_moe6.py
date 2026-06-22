"""Launch six stable-save goalkeeper region experts across multiple GPUs."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tyro

_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Cfg:
  init: str = "src/assets/soccer/weight/model_repaired_lyk.pt"
  out_dir: str = "logs/lyk/experts"
  log_dir: str = "logs/lyk/expert_logs"
  devices: tuple[int, ...] = (0, 1, 2, 3)
  regions: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
  num_envs: int = 4096
  warmup: int = 30
  block_iters: int = 20
  blocks: int = 80
  eval_resets: int = 4
  lr: float = 3.0e-5
  std: float = 0.035
  residual_scale: float = 0.35
  stable_save_weight: float = 0.8
  rollback_drop: float = 0.005
  extra_args: tuple[str, ...] = ()


def _cmd(cfg: Cfg, region: int) -> list[str]:
  return [
    sys.executable,
    "scripts/train_ballistic_residual.py",
    "--init", cfg.init,
    "--out", f"{cfg.out_dir}/stable_sr{region}.pt",
    "--train-regions", str(region),
    "--num-envs", str(cfg.num_envs),
    "--warmup", str(cfg.warmup),
    "--block-iters", str(cfg.block_iters),
    "--blocks", str(cfg.blocks),
    "--eval-resets", str(cfg.eval_resets),
    "--lr", str(cfg.lr),
    "--std", str(cfg.std),
    "--residual-scale", str(cfg.residual_scale),
    "--w-conceded", "15.0",
    "--w-intercept", "3.0",
    "--w-body", "1.0",
    "--w-stop", "1.0",
    "--w-posture", "1.5",
    "--w-recovery", "6.0",
    "--w-post-save-ang-vel", "0.08",
    "--post-save-action-rate", "0.06",
    "--w-feet-slip", "0.08",
    "--w-ang-vel", "0.04",
    "--action-rate", "0.08",
    "--clip-param", "0.06",
    "--desired-kl", "0.003",
    "--stable-save-weight", str(cfg.stable_save_weight),
    "--rollback-drop", str(cfg.rollback_drop),
    "--device", "cuda:0",
    *cfg.extra_args,
  ]


def main(cfg: Cfg) -> None:
  if not cfg.devices:
    raise ValueError("at least one device is required")
  Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
  Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

  pending = list(cfg.regions)
  running: list[tuple[int, int, subprocess.Popen, object]] = []
  failed: list[tuple[int, int]] = []

  def launch(region: int, gpu: int) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MUJOCO_EGL_DEVICE_ID"] = "0"
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("WANDB_MODE", "disabled")
    log_path = Path(cfg.log_dir) / f"stable_sr{region}_gpu{gpu}.log"
    log = open(log_path, "w")
    cmd = _cmd(cfg, region)
    print(f"[LAUNCH] region {region} on gpu {gpu}: {' '.join(cmd)}")
    print(f"[LAUNCH] log -> {log_path}")
    proc = subprocess.Popen(cmd, cwd=_REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    running.append((region, gpu, proc, log))

  while pending or running:
    busy = {gpu for _, gpu, _, _ in running}
    free = [gpu for gpu in cfg.devices if gpu not in busy]
    while pending and free:
      launch(pending.pop(0), free.pop(0))

    time.sleep(10.0)
    still_running = []
    for region, gpu, proc, log in running:
      code = proc.poll()
      if code is None:
        still_running.append((region, gpu, proc, log))
        continue
      log.close()
      if code != 0:
        failed.append((region, code))
        print(f"[FAIL] region {region} exited with code {code}")
      else:
        print(f"[DONE] region {region} on gpu {gpu}")
    running = still_running

  if failed:
    raise SystemExit(f"failed experts: {failed}")
  print(f"[DONE] all experts saved under {cfg.out_dir}")


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="launch_stable_moe6"))
