"""Auto-curriculum for Shooter Stage 4: high-speed goal-plane accuracy.

Iterates through speed levels (5→10 m/s), training the Stage 4 LSTM teacher
and evaluating with ``eval_shooter_parallel.py`` between phases.

Usage:
  # Full curriculum (resume from Stage 3 final checkpoint)
  python scripts/train_stage4_curriculum.py \\
      --base-checkpoint logs/rsl_rl/g1_soccer/<stage3_final>/model_111994.pt \\
      --motion-dir src/assets/soccer/motions/shooter \\
      --env-num-envs 4096 --gpu-ids '[0]' \\
      --min-iters 3000

  # Resume from Phase 5 (skip already-trained Phase 1-4)
  python scripts/train_stage4_curriculum.py \\
      --base-checkpoint logs/.../stage4_phase4_std8.0_att2/model_138985.pt \\
      --motion-dir src/assets/soccer/motions/shooter \\
      --start-phase 4 \\
      --min-iters 3000

  # Smoke run — skip eval pass/fail gate, verify all 9 phases chain correctly
  python scripts/train_stage4_curriculum.py \\
      --base-checkpoint <model.pt> \\
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


_TASK_ID = "Unitree-G1-Shooter-Stage4"
_EVAL_TASK_ID = "Eval-Shooter-Stage4"
_SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class CurriculumPhase:
  ball_speed_std: float
  min_mean_kick_speed: float


DEFAULT_PHASES = [
  CurriculumPhase(ball_speed_std=5.0,  min_mean_kick_speed=4.5),
  CurriculumPhase(ball_speed_std=6.0,  min_mean_kick_speed=5.5),
  CurriculumPhase(ball_speed_std=7.0,  min_mean_kick_speed=6.5),
  CurriculumPhase(ball_speed_std=8.0,  min_mean_kick_speed=7.5),
  CurriculumPhase(ball_speed_std=8.5,  min_mean_kick_speed=8.0),
  CurriculumPhase(ball_speed_std=9.0,  min_mean_kick_speed=8.5),
  CurriculumPhase(ball_speed_std=9.5,  min_mean_kick_speed=9.0),
  CurriculumPhase(ball_speed_std=10.0, min_mean_kick_speed=9.5),
  CurriculumPhase(ball_speed_std=10.0, min_mean_kick_speed=10.0),
]


@dataclass
class CurriculumConfig:
  base_checkpoint: str
  motion_dir: str
  min_iters: int = 3000
  max_attempts: int = 3
  num_trials: int = 5000
  num_eval_envs: int = 2048
  gpu_ids: list[int] | str | None = field(default_factory=lambda: [0])
  env_num_envs: int = 4096
  skip_eval_check: bool = False
  start_phase: int = 0


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


def _run_eval(ckpt_path: Path, num_trials: int, num_envs: int,
              run_dir: Path | None = None) -> dict[str, Any]:
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
  if run_dir is not None:
    cmd.extend(["--save-npz", str(run_dir / "eval_trials.npz")])

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
  if args.start_phase < 0 or args.start_phase >= len(phases):
    raise ValueError(f"start_phase={args.start_phase} out of range [0, {len(phases)})")
  current_ckpt = args.base_checkpoint

  for i, phase in enumerate(phases):
    if i < args.start_phase:
      print(f"  Skipping Phase {i+1} (start_phase={args.start_phase}).")
      continue
    phase_label = f"phase{i+1}_std{phase.ball_speed_std}"
    print(f"\n{'='*60}")
    print(f"  Phase {i+1}/{len(phases)}: ball_speed_std={phase.ball_speed_std}, "
          f"min_speed={phase.min_mean_kick_speed} m/s")
    print(f"{'='*60}")

    for attempt in range(1, args.max_attempts + 1):
      run_name = f"stage4_{phase_label}_att{attempt}"
      run_dir = _run_train(args, current_ckpt, phase, run_name)
      latest_ckpt = _find_latest_checkpoint(run_dir)
      metrics = _run_eval(latest_ckpt, args.num_trials, args.num_eval_envs, run_dir)

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
