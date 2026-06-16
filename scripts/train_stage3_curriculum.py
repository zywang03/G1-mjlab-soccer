"""Auto-curriculum for Shooter Stage 3: speed + goal-plane accuracy.

Iterates through speed levels, training the Stage 3 LSTM teacher and
evaluating with ``eval_shooter_parallel.py`` between phases.

Usage:
  # Full curriculum
  python scripts/train_stage3_curriculum.py \\
      --base-checkpoint logs/rsl_rl/g1_soccer/2026-06-12_16-34-10/model_100000.pt \\
      --motion-dir src/assets/soccer/motions/shooter \\
      --env.scene.num-envs 4096 --gpu-ids '[0,1]' \\
      --min-iters 3000

  # Smoke run — skip eval pass/fail gate, verify all 5 phases chain correctly
  python scripts/train_stage3_curriculum.py \\
      --base-checkpoint <model_100000.pt> \\
      --motion-dir src/assets/soccer/motions/shooter \\
      --min-iters 100 --num-trials 2 --num-eval-envs 2 \\
      --skip-eval-check
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tyro


_TASK_ID = "Unitree-G1-Shooter-Stage3"
_EVAL_TASK_ID = "Eval-Shooter-Stage3"
_SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class CurriculumPhase:
  ball_speed_std: float
  min_mean_kick_speed: float


DEFAULT_PHASES = [
  CurriculumPhase(ball_speed_std=2.0, min_mean_kick_speed=1.8),
  CurriculumPhase(ball_speed_std=2.5, min_mean_kick_speed=2.3),
  CurriculumPhase(ball_speed_std=3.0, min_mean_kick_speed=2.8),
  CurriculumPhase(ball_speed_std=3.5, min_mean_kick_speed=3.4),
  CurriculumPhase(ball_speed_std=4.0, min_mean_kick_speed=4.0),
]


@dataclass
class CurriculumConfig:
  base_checkpoint: str
  motion_dir: str
  min_iters: int = 3000
  max_attempts: int = 3
  num_trials: int = 50
  num_eval_envs: int = 64
  gpu_ids: list[int] | str | None = field(default_factory=lambda: [0])
  env_num_envs: int = 4096
  skip_eval_check: bool = False


def _find_python() -> str:
  return sys.executable


def _find_latest_checkpoint(run_dir: Path) -> Path:
  models = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  if not models:
    raise FileNotFoundError(f"No model_*.pt found in {run_dir}")
  return models[-1]


def _run_train(cfg: CurriculumConfig, ckpt_path: str, phase: CurriculumPhase,
               run_name: str) -> Path:
  cmd = [
    _find_python(), str(_SCRIPT_DIR / "train.py"),
    _TASK_ID,
    "--motion-dir", cfg.motion_dir,
    "--load-checkpoint-path", ckpt_path,
    "--ball-speed-std", str(phase.ball_speed_std),
    "--agent.max-iterations", str(cfg.min_iters),
    "--agent.run-name", run_name,
    "--env.scene.num-envs", str(cfg.env_num_envs),
    "--agent.save-interval", "100",
  ]

  gpu = cfg.gpu_ids
  if isinstance(gpu, str):
    cmd.extend(["--gpu-ids", gpu])
  elif isinstance(gpu, list):
    cmd.extend(["--gpu-ids", str(gpu)])

  print(f"\n[TRAIN] phase ball_speed_std={phase.ball_speed_std}, run={run_name}")
  print(f"[TRAIN] {' '.join(cmd)}\n", flush=True)

  subprocess.run(cmd, check=True)

  log_root = Path("logs") / "rsl_rl" / "g1_soccer"
  run_dirs = sorted(log_root.glob(f"*_{run_name}"), key=os.path.getmtime)
  if not run_dirs:
    raise RuntimeError(f"Could not find log dir for run_name={run_name}")
  return run_dirs[-1]


def _run_eval(ckpt_path: Path, num_trials: int, num_envs: int) -> dict[str, Any]:
  with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json_path = f.name

  cmd = [
    _find_python(), str(_SCRIPT_DIR / "eval_shooter_parallel.py"),
    "--task-id", _EVAL_TASK_ID,
    "--checkpoint", str(ckpt_path),
    "--headless",
    "--num-trials", str(num_trials),
    "--num-envs", str(num_envs),
    "--summary-json", json_path,
  ]

  print(f"\n[EVAL] checkpoint={ckpt_path.name}")
  print(f"[EVAL] {' '.join(cmd)}\n", flush=True)

  for retry in range(3):
    try:
      subprocess.run(cmd, check=True)
      with open(json_path, encoding="utf-8") as fh:
        metrics = json.load(fh)
      return metrics
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
      if retry == 2:
        raise RuntimeError(f"[EVAL] failed after 3 attempts: {exc}") from exc
      print(f"[EVAL] attempt {retry + 1} failed ({exc}), retrying in 5s ...")
      import time
      time.sleep(5)
    finally:
      try:
        if os.path.exists(json_path):
          os.unlink(json_path)
      except OSError:
        pass


def _check_pass(metrics: dict[str, Any], phase: CurriculumPhase) -> tuple[bool, list[str]]:
  reasons: list[str] = []

  sr = metrics.get("success_rate", 0.0)
  if sr < 90.0:
    reasons.append(f"success_rate={sr:.1f} < 90.0")

  mks = metrics.get("mean_kick_speed", 0.0)
  if mks < phase.min_mean_kick_speed:
    reasons.append(f"mean_kick_speed={mks:.2f} < {phase.min_mean_kick_speed}")

  hit = metrics.get("target_hit_rate_0_3m", 0.0)
  if hit < 80.0:
    reasons.append(f"target_hit_rate_0_3m={hit:.1f} < 80.0")

  mte = metrics.get("mean_target_error", float("inf"))
  if mte > 0.35:
    reasons.append(f"mean_target_error={mte:.3f} > 0.35")

  return len(reasons) == 0, reasons


def _print_metrics(metrics: dict[str, Any], phase: CurriculumPhase) -> None:
  print(f"  success_rate:        {metrics.get('success_rate', 0):.1f}%")
  print(f"  mean_kick_speed:     {metrics.get('mean_kick_speed', 0):.2f} m/s  (min: {phase.min_mean_kick_speed:.1f})")
  print(f"  mean_kick_accuracy:  {metrics.get('mean_kick_accuracy', 0):.4f}")
  print(f"  target_hit_rate_0_3m: {metrics.get('target_hit_rate_0_3m', 0):.1f}%")
  print(f"  mean_target_error:   {metrics.get('mean_target_error', 0):.3f} m")
  print(f"  mean_cross_z:        {metrics.get('mean_cross_z', 0):.3f} m")
  print(f"  mean_abs_z_speed:    {metrics.get('mean_abs_z_speed', 0):.3f} m/s")


def main():
  import mjlab.tasks  # noqa: F401  — populate registry
  import src.tasks    # noqa: F401

  args = tyro.cli(CurriculumConfig)

  if not Path(args.base_checkpoint).exists():
    raise FileNotFoundError(f"base checkpoint not found: {args.base_checkpoint}")

  if not Path(args.motion_dir).exists():
    raise FileNotFoundError(f"motion directory not found: {args.motion_dir}")

  phases = DEFAULT_PHASES
  current_ckpt = args.base_checkpoint

  for i, phase in enumerate(phases):
    phase_label = f"phase{i+1}_std{phase.ball_speed_std}"
    print(f"\n{'='*60}")
    print(f"  Phase {i+1}/{len(phases)}: ball_speed_std={phase.ball_speed_std}, "
          f"min_speed={phase.min_mean_kick_speed} m/s")
    print(f"{'='*60}")

    for attempt in range(1, args.max_attempts + 1):
      run_name = f"stage3_{phase_label}_att{attempt}"
      run_dir = _run_train(args, current_ckpt, phase, run_name)
      latest_ckpt = _find_latest_checkpoint(run_dir)
      metrics = _run_eval(latest_ckpt, args.num_trials, args.num_eval_envs)

      print(f"\n  [Phase {i+1} attempt {attempt}] Eval metrics:")
      _print_metrics(metrics, phase)

      passed, reasons = _check_pass(metrics, phase)
      if args.skip_eval_check:
        passed = True
        reasons = ["(skip_eval_check — smoke run)"]
      if passed:
        print(f"\n  *** Phase {i+1} PASSED (attempt {attempt}) ***")
        current_ckpt = str(latest_ckpt)
        break
      else:
        print(f"\n  --- Phase {i+1} FAILED attempt {attempt}: {reasons}")
        current_ckpt = str(latest_ckpt)
    else:
      raise RuntimeError(
        f"Phase {i+1} (ball_speed_std={phase.ball_speed_std}) "
        f"failed after {args.max_attempts} attempts."
      )

  print(f"\n{'='*60}")
  print("  Curriculum completed successfully.")
  print(f"  Final checkpoint: {current_ckpt}")
  print(f"{'='*60}")


if __name__ == "__main__":
  main()
