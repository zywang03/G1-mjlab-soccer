"""Automatic Stage I → Stage II training pipeline.

Runs the PAiD paper-aligned two-stage shooter training sequentially:

  1. **Stage I**: LSTM motion tracking — 13 human kick motions,
     perception-free, adaptive sampling.
  2. **Stage II**: perception-guided kicking — transfers Stage I LSTM
     weights (partial: input layer expanded for new soccer observation
     terms), reduced tracking weights, added task rewards.

This is a thin subprocess wrapper around ``scripts/train.py``.
It does NOT replace train.py — every training run is still a fully
valid, loggable train.py invocation that can be resumed independently.

Signal handling
---------------
When the user presses Ctrl+C, this script:

  1. Forwards SIGTERM to the entire child process group (including
     any torchrunx workers).
  2. Waits up to 20 s for graceful shutdown; force-kills if needed.
  3. Detects the log directory created by the child and prints:
     - The latest checkpoint file and iteration.
     - The exact ``train.py`` resume command.

Usage
-----
Complete two-stage pipeline (recommended):
  python scripts/train_pipeline.py \\
      --motion-dir src/assets/soccer/motions/shooter \\
      --env-scene-num-envs 2048 --gpu-ids 0,1 \\
      --stage1-iterations 10001 --stage2-iterations 10001 \\
      --stage1-run-name my_s1 --stage2-run-name my_s2

Stage I only (debug / motion-quality check):
  python scripts/train_pipeline.py \\
      --motion-dir src/assets/soccer/motions/shooter \\
      --env-scene-num-envs 2048 --gpu-ids 0,1 --stage1-only

Skip Stage I, run Stage II from existing checkpoint:
  python scripts/train_pipeline.py \\
      --motion-dir src/assets/soccer/motions/shooter \\
      --env-scene-num-envs 2048 --gpu-ids 0,1 \\
      --skip-stage1 --load-run "2026-06-10_14-26-14" \\
      --load-checkpoint "model_10000.pt"

How it works (internal flow)
----------------------------
1. Stage I launched via::
     python scripts/train.py Unitree-G1-Shooter-Stage1 <args>
   Log directory detected by snapshotting ``logs/rsl_rl/g1_soccer/``
   before and after the run; the newest directory with a ``params/``
   sub-folder is selected.

2. The latest checkpoint in the Stage I log directory is identified
   by parsing ``model_<iter>.pt`` filenames — the one with the highest
   iteration number wins.

3. Stage II launched via::
     python scripts/train.py Unitree-G1-Shooter-Stage2 <args> \\
         --agent.resume True \\
         --agent.load-run <stage1_run_dir> \\
         --agent.load-checkpoint <latest_ckpt>
   ``train.py`` uses standard ``runner.load()`` since both stages share
   identical observation dimensions (160D).

Edge cases
----------
- Stage I interrupted (Ctrl+C): prints log dir + checkpoint path +
  resume command, then exits.
- Stage I crash: prints "[PIPELINE] Stage I crashed.", exits 1.
- Stage II crash: same pattern.
- No checkpoint found after Stage I: prints error, exits 1.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro

# ---------------------------------------------------------------------------
# Global: child process group and interrupt flag, for SIGINT forwarding
# ---------------------------------------------------------------------------

_child_pgrp: int | None = None
"""The process group ID of the currently running child process.

Set via ``os.getpgid(proc.pid)`` before the output stream is consumed,
cleared in ``finally`` so the next stage gets a fresh group ID.
"""

_interrupted: bool = False
"""Flag set by the SIGINT/SIGTERM handler.

When True, ``_stream()`` breaks out of its output-reading loop, and
``run_stage()`` skips the normal completion path to report the
checkpoint and print a resume command instead.
"""


def _sigint_handler(signum: int, _frame: object) -> None:
    """Signal handler for SIGINT/SIGTERM.

    Strategy
    --------
    Instead of immediately raising ``KeyboardInterrupt`` (which has
    tricky timing with blocking I/O in ``subprocess.Popen.stdout``),
    we set a global flag and forward SIGTERM to the child process
    group.  The child death closes the stdout pipe, unblocking the
    stream-read loop, which then sees ``_interrupted == True`` and
    breaks.  ``run_stage()`` detects the flag and reports the snapshot
    checkpoint.
    """
    del signum, _frame
    global _child_pgrp, _interrupted
    _interrupted = True
    if _child_pgrp is not None:
        print(
            f"\n[PIPELINE] Ctrl+C — forwarding SIGTERM to child PGID {_child_pgrp}…"
        )
        try:
            os.killpg(_child_pgrp, signal.SIGTERM)
        except OSError as exc:
            print(f"[PIPELINE] killpg: {exc}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LOG_ROOT = Path("logs") / "rsl_rl" / "g1_soccer"
"""CWD-relative path where train.py writes run directories."""

_TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"
"""Absolute path to the train.py entrypoint."""


@dataclass
class PipelineConfig:
    """All tunable knobs for the two-stage training pipeline.

    Every field maps to a ``scripts/train.py`` argument (via
    ``_build_train_cmd()``) or controls the pipeline flow.
    """

    # ---- Required ----------------------------------------------------------------

    motion_dir: str
    """**Required.**  Directory of ``.npz`` motion files.  See
    ``src/assets/soccer/motions/shooter/`` for the 13 shipped motions."""

    # ---- Environment -------------------------------------------------------------

    env_scene_num_envs: int = 2048
    """Number of parallel envs per GPU.  For two GPUs the total is
    ``2 × env_scene_num_envs``.  Default is 2048 on each GPU
    (total 4096), matching the paper."""

    gpu_ids: str = "0"
    """Comma-separated GPU IDs (e.g. ``"0"``, ``"0,1"``).  The pipeline
    translates this to ``CUDA_VISIBLE_DEVICES`` (if not already set by
    the caller) so ``train.py`` always receives ``--gpu-ids all`` and
    uses exactly the GPUs the user requested."""

    # ---- Stage I -----------------------------------------------------------------

    stage1_iterations: int = 10001
    """Maximum PPO iterations for Stage I motion tracking.  Set lower
    (e.g. 2000) for quick smoke-tests."""

    stage1_run_name: str = ""
    """Suffix appended to the timestamped Stage I log-directory name.
    Set to e.g. ``"baseline_s1"`` to make runs easily identifiable."""

    # ---- Stage II ----------------------------------------------------------------

    stage2_iterations: int = 10001
    """Maximum PPO iterations for Stage II perception-guided kicking."""

    stage2_run_name: str = ""
    """Suffix for Stage II log-directory name."""

    # ---- Control flow ------------------------------------------------------------

    skip_stage1: bool = False
    """Skip Stage I entirely.  When True, ``--load-run`` and
    ``--load-checkpoint`` are required.  Useful for continuing to
    Stage II after an independently-run Stage I."""

    stage1_only: bool = False
    """Run Stage I only and exit.  Mutually exclusive with
    ``--skip-stage1``."""

    load_run: str | None = None
    """When ``--skip-stage1`` is set, the name of the run directory
    under ``logs/rsl_rl/g1_soccer/`` that Stage II should resume from.
    Example: ``"2026-06-10_14-26-14"``."""

    load_checkpoint: str | None = None
    """When ``--skip-stage1`` is set, the checkpoint filename to load.
    Example: ``"model_10000.pt"``."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_gpu_ids(raw: str) -> str:
    """Translate the user's ``--gpu-ids`` value into a train.py argument.

    Returns ``"all"`` so that ``train.py``'s ``select_gpus()`` picks up
    whatever GPUs are visible via ``CUDA_VISIBLE_DEVICES``.

    As a side-effect, when the user specifies explicit GPU IDs (e.g.
    ``"0"`` or ``"0,1"``) and the environment variable
    ``CUDA_VISIBLE_DEVICES`` is *not* already set, this function sets it
    to the user's value so that ``train.py`` sees the correct set.
    """
    raw = raw.strip()
    if not raw or raw.lower() == "all":
        return "all"

    # User specified concrete GPU ids — if CUDA_VISIBLE_DEVICES is not
    # already set, inherit from --gpu-ids so train.py picks the right ones.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = raw

    return "all"


def _build_train_cmd(
    stage: Literal["Stage1", "Stage2"],
    cfg: PipelineConfig,
    resume: bool = False,
    load_run: str | None = None,
    load_ckpt: str | None = None,
) -> list[str]:
    """Build a ``sys.executable scripts/train.py <task> <args>`` command list.

    Parameters
    ----------
    stage : "Stage1" or "Stage2"
        Which registered task ID to train (``Unitree-G1-Shooter-{stage}``).
    cfg : PipelineConfig
        Source of all training hyper-parameters.
    resume : bool
        Whether to add ``--agent.resume True`` and ``--agent.load-*`` args.
    load_run : str | None
        Run directory name for load-run (only when ``resume=True``).
    load_ckpt : str | None
        Checkpoint filename for load-checkpoint (only when ``resume=True``).

    Returns
    -------
    list[str]
        Full command-line argument list ready for ``subprocess.Popen``.
    """
    task_id = f"Unitree-G1-Shooter-{stage}"
    iterations = cfg.stage1_iterations if stage == "Stage1" else cfg.stage2_iterations
    run_name = cfg.stage1_run_name if stage == "Stage1" else cfg.stage2_run_name

    # save-interval is dynamically clamped so large runs don't
    # generate (say) 200 checkpoints of 8 MB each.
    save_int = min(100, max(1, iterations // 20))

    # --gpu-ids in train.py accepts either a single int (e.g. 0)
    # or the literal string "all".  Multi-element lists like [0, 1]
    # are not supported via tyro CLI.  We map "0,1" → "all" since
    # the actual GPU selection is governed by CUDA_VISIBLE_DEVICES.
    gpu_arg = _normalize_gpu_ids(cfg.gpu_ids)

    cmd = [
        sys.executable, str(_TRAIN_SCRIPT), task_id,
        "--motion-dir", cfg.motion_dir,
        "--env.scene.num-envs", str(cfg.env_scene_num_envs),
        "--gpu-ids", gpu_arg,
        "--agent.max-iterations", str(iterations),
        "--agent.save-interval", str(save_int),
    ]
    if run_name:
        cmd += ["--agent.run-name", run_name]
    if resume:
        if load_run is None or load_ckpt is None:
            raise ValueError(
                "--load-run and --load-checkpoint are required when resume=True"
            )
        cmd += [
            "--agent.resume", "True",
            "--agent.load-run", load_run,
            "--agent.load-checkpoint", load_ckpt,
        ]
    return cmd


def _snapshot_log_dirs() -> set[str]:
    """Return the names of all directories currently under _LOG_ROOT."""
    if _LOG_ROOT.is_dir():
        return {d.name for d in _LOG_ROOT.iterdir() if d.is_dir()}
    return set()


def _detect_log_dir(
    snapshot_before: set[str], snapshot_after: set[str]
) -> Path:
    """Find the single new run directory created between two snapshots.

    Prefers directories that contain a ``params/`` sub-directory
    (a reliable marker of a train.py run).  Falls back to the newest
    candidate if no ``params/`` directory is found.
    """
    new = snapshot_after - snapshot_before
    if not new:
        raise RuntimeError(
            "No new log directory detected under "
            f"{_LOG_ROOT.resolve()} — train.py may have failed "
            "before creating one."
        )
    candidates = sorted(new)
    for name in reversed(candidates):
        candidate = _LOG_ROOT / name
        if candidate.is_dir() and (candidate / "params").is_dir():
            return candidate
    return _LOG_ROOT / candidates[-1]


def _stream(proc: subprocess.Popen) -> None:
    """Read child stdout line-by-line, tee to parent stdout.

    Checks the ``_interrupted`` flag after every line.  When the flag
    is set (Ctrl+C was forwarded to the child's process group), the
    loop breaks early so ``_wait_child()`` can run.
    """
    global _interrupted
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        if _interrupted:
            break
    proc.stdout.close()


def _wait_child(proc: subprocess.Popen) -> bool:
    """Block until the child process exits, with force-kill fallback.

    Returns
    -------
    bool
        ``True`` if the child exited normally (including via SIGTERM),
        ``False`` if we had to send SIGKILL after a 20 s timeout.
    """
    global _child_pgrp
    try:
        proc.wait(timeout=20)
        return True
    except subprocess.TimeoutExpired:
        if _child_pgrp is not None:
            print(
                f"[PIPELINE] Child still alive — sending SIGKILL to "
                f"PGID {_child_pgrp}"
            )
            os.killpg(_child_pgrp, signal.SIGKILL)
        proc.wait()
        return False


def find_latest_checkpoint(log_dir: Path) -> tuple[Path | None, int]:
    """Locate the checkpoint file with the highest iteration number.

    Parses filenames matching ``model_<int>.pt``.  Returns
    ``(path, iteration)``, or ``(None, -1)`` if no checkpoint exists.
    """
    best_path: Path | None = None
    best_iter = -1
    for pt in log_dir.glob("model_*.pt"):
        try:
            it = int(pt.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if it > best_iter:
            best_iter = it
            best_path = pt
    return best_path, best_iter


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage(
    stage: Literal["Stage1", "Stage2"],
    cfg: PipelineConfig,
    resume: bool = False,
    load_run: str | None = None,
    load_ckpt: str | None = None,
) -> Path | None:
    """Launch ``train.py`` as a subprocess for one training stage.

    Parameters
    ----------
    stage : "Stage1" or "Stage2"
    cfg : PipelineConfig
    resume : bool
        Passed through to ``_build_train_cmd()`` to control whether
        ``--agent.resume`` is included.
    load_run : str | None
        Log directory name for ``--agent.load-run``.
    load_ckpt : str | None
        Checkpoint filename for ``--agent.load-checkpoint``.

    Returns
    -------
    Path | None
        Path to the training run's log directory on success, or
        ``None`` if the user interrupted with Ctrl+C.

    Raises
    ------
    subprocess.CalledProcessError
        If the child process exits with a non-zero code (crash).
    """
    global _child_pgrp, _interrupted
    _interrupted = False

    cmd = _build_train_cmd(
        stage, cfg, resume=resume, load_run=load_run, load_ckpt=load_ckpt
    )
    marker = f"[PIPELINE:{stage}]"
    print(f"\n{marker} Launching: {' '.join(cmd)}\n")
    sys.stdout.flush()

    _LOG_ROOT.mkdir(parents=True, exist_ok=True)
    before = _snapshot_log_dirs()

    # │ preexec_fn=os.setsid  │  Places the child in its own session.
    # │                       │  os.killpg(child_pgrp, …) then kills
    # │                       │  the child AND all torchrunx workers
    # │                       │  in one shot.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
        bufsize=1,
    )
    _child_pgrp = os.getpgid(proc.pid)
    try:
        _stream(proc)
        _wait_child(proc)
    finally:
        _child_pgrp = None

    # ── Determine output directory ─────────────────────────────────────
    try:
        after = _snapshot_log_dirs()
        log_dir = _detect_log_dir(before, after)
    except RuntimeError:
        log_dir = None

    # ── Handle interrupt ────────────────────────────────────────────────
    if _interrupted:
        print(f"\n{marker} interrupted by user.")
        if log_dir is not None:
            ckpt_path, ckpt_iter = find_latest_checkpoint(log_dir)
            _print_ckpt_info(ckpt_path, ckpt_iter)
            _print_resume_command(stage, log_dir, cfg, ckpt_path)
        return None

    # ── Handle crash ────────────────────────────────────────────────────
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    # ── Success ─────────────────────────────────────────────────────────
    print(f"\n{marker} completed.  log_dir = {log_dir}")
    if log_dir is not None:
        ckpt_path, ckpt_iter = find_latest_checkpoint(log_dir)
        _print_ckpt_info(ckpt_path, ckpt_iter)
    return log_dir


def _print_ckpt_info(ckpt_path: Path | None, ckpt_iter: int) -> None:
    """Print a one-line summary of the latest checkpoint."""
    if ckpt_path is not None:
        sz_mb = ckpt_path.stat().st_size / 1024 / 1024
        print(
            f"[PIPELINE] Latest checkpoint: {ckpt_path.name} "
            f"(iter={ckpt_iter}, {sz_mb:.1f} MB)"
        )


def _print_resume_command(
    stage: Literal["Stage1", "Stage2"],
    log_dir: Path,
    cfg: PipelineConfig,
    ckpt_path: Path | None,
) -> None:
    """Print a copy-pasteable ``train.py`` command to resume from here."""
    task_id = f"Unitree-G1-Shooter-{stage}"
    ckpt_name = ckpt_path.name if ckpt_path else "model_*.pt"
    run_name = log_dir.name
    print(f"\n[PIPELINE] To resume this {stage} from the last checkpoint:")
    print(f"  python scripts/train.py {task_id} \\")
    print(f"      --motion-dir {cfg.motion_dir} \\")
    print(
        f"      --env.scene.num-envs {cfg.env_scene_num_envs} "
        f"--gpu-ids {cfg.gpu_ids} \\"
    )
    print(f"      --agent.resume True \\")
    print(
        f"      --agent.load-run \"{run_name}\" "
        f"--agent.load-checkpoint \"{ckpt_name}\""
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI, validate, run Stage I, then Stage II."""
    # Imports are deferred so ``--help`` doesn't spin up torch/Warp.
    import mjlab.tasks  # noqa: F401
    import src.tasks    # noqa: F401

    cfg = tyro.cli(PipelineConfig, description=__doc__)

    # ── Validate ────────────────────────────────────────────────────────
    if cfg.skip_stage1 and cfg.stage1_only:
        print("[PIPELINE] Conflicting flags: --skip-stage1 and --stage1-only")
        sys.exit(1)
    if cfg.skip_stage1 and (not cfg.load_run or not cfg.load_checkpoint):
        print(
            "[PIPELINE] --skip-stage1 requires both "
            "--load-run AND --load-checkpoint"
        )
        sys.exit(1)
    if not cfg.motion_dir:
        print("[PIPELINE] --motion-dir is required")
        sys.exit(1)

    # ── Stage I ─────────────────────────────────────────────────────────
    stage1_log_dir: Path | None = None

    if not cfg.skip_stage1:
        try:
            stage1_log_dir = run_stage("Stage1", cfg, resume=False)
        except subprocess.CalledProcessError:
            print("\n[PIPELINE] Stage I crashed.  See the output above for details.")
            sys.exit(1)
        if stage1_log_dir is None:
            sys.exit(1)  # interrupt already reported by run_stage()

    if cfg.stage1_only:
        print("\n[PIPELINE] Stage I only mode — done.")
        return

    # ── Stage II ────────────────────────────────────────────────────────
    # Resolve the checkpoint source.
    if cfg.skip_stage1:
        load_run_name: str = cfg.load_run  # type: ignore[assignment]
        load_ckpt: str = cfg.load_checkpoint  # type: ignore[assignment]
        print(
            f"\n[PIPELINE] Skipping Stage I — loading from "
            f"run={load_run_name}, ckpt={load_ckpt}"
        )
    else:
        load_run_name = stage1_log_dir.name  # type: ignore[union-attr]
        ckpt_path, ckpt_iter = find_latest_checkpoint(stage1_log_dir)  # type: ignore[arg-type]
        if ckpt_path is None:
            print(
                "[PIPELINE] No checkpoint found in the Stage I log directory "
                f"'{stage1_log_dir}'.  Aborting."
            )
            sys.exit(1)
        load_ckpt = ckpt_path.name
        print(
            f"\n[PIPELINE] Starting Stage II from Stage I run "
            f"'{load_run_name}', checkpoint '{load_ckpt}' "
            f"(iter {ckpt_iter})"
        )

    try:
        stage2_log_dir = run_stage(
            "Stage2", cfg,
            resume=True,
            load_run=load_run_name,
            load_ckpt=load_ckpt,
        )
    except subprocess.CalledProcessError:
        print("\n[PIPELINE] Stage II crashed.  See the output above for details.")
        sys.exit(1)
    if stage2_log_dir is None:
        sys.exit(1)

    print(
        f"\n[PIPELINE] Both stages completed.  "
        f"Stage II log: {stage2_log_dir}"
    )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)
    main()
