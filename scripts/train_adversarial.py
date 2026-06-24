"""Alternating adversarial-training framework.

This is only the outer alternating framework for now. It alternates shooter and
keeper phases, records the checkpoint chain in a manifest, and uses random
goal-plane target sampling for the shooter phase. The current phase backends are
direct Python calls into the existing local training entrypoints.

Examples:
  # Safe smoke test: writes only the schedule and placeholder checkpoints.
  python scripts/train_adversarial.py --rounds 2 --dry-run

  # Run the alternating shell.
  python scripts/train_adversarial.py --rounds 3 --no-dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal


Mode = Literal[
  "alternate",
  "train-shooter",
  "train-keeper",
  "train-keeper-idle",
  "train-keeper-moe7",
]

DEFAULT_OUT_DIR = "logs/adversarial"
DEFAULT_SHOOTER_INIT = "checkpoints/stage2/model_100000.pt"
DEFAULT_KEEPER_INIT = "src/assets/soccer/weight/goalkeeper_moe6.pt"
DEFAULT_MOTION_DIR = "src/assets/soccer/motions/shooter"
DEFAULT_SHOOTER_LOG_ROOT = "logs/rsl_rl/g1_soccer"
DEFAULT_KEEPER_LOG_ROOT = "logs/rsl_rl/g1_soccer"
SHOOTER_TASK_ID = "Unitree-G1-Shooter-Adversarial"
KEEPER_TASK_ID = "Unitree-G1-Goalkeeper-Adversarial"
KEEPER_EXPERT_TASK_ID = "Unitree-G1-Goalkeeper-Expert-Adversarial"
KEEPER_IDLE_TASK_ID = "Unitree-G1-Goalkeeper-Idle-Adversarial"
KEEPER_MOE7_PREPARE_TASK_ID = "Unitree-G1-Goalkeeper-MoE7-Prepare-Adversarial"
FROZEN_SHOOTER_TASK_ID = "Eval-Shooter-Stage3"
FROZEN_KEEPER_TASK_ID = "Eval-Goalkeeper"
OPPONENT_CURRENT_PROB = 0.5
OPPONENT_HISTORY_PROB = 0.4
OPPONENT_INITIAL_PROB = 0.1

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))


@dataclass
class AdversarialConfig:
  mode: Mode = "alternate"
  rounds: int = 3
  out_dir: str = DEFAULT_OUT_DIR
  dry_run: bool = True
  seed: int = 2810

  shooter_init: str = DEFAULT_SHOOTER_INIT
  keeper_init: str = DEFAULT_KEEPER_INIT
  keeper_idle_init: str = ""

  motion_dir: str = DEFAULT_MOTION_DIR
  shooter_log_root: str = DEFAULT_SHOOTER_LOG_ROOT
  keeper_log_root: str = DEFAULT_KEEPER_LOG_ROOT
  num_envs: int = 1024
  gpu_ids: str = "0"
  keeper_device: str = "cuda:0"

  shooter_iters_per_round: int = 3000
  keeper_blocks_per_round: int = 20
  keeper_block_iters: int = 5
  keeper_expert_iters_per_round: int = 1000
  keeper_idle_iters_per_round: int = 1000
  keeper_warmup: int = 20
  promotion_trials: int = 16
  promotion_device: str = "cuda:0"
  isolated_training_process: bool = True

  shooter_targets_per_round: int = 64
  target_x_min: float = -1.25
  target_x_max: float = 1.25
  target_y: float = -5.0
  target_z_min: float = 0.05
  target_z_max: float = 1.5


def sample_target_schedule(
  rounds: int,
  targets_per_round: int,
  seed: int,
  x_range: tuple[float, float] = (-1.25, 1.25),
  y: float = -5.0,
  z_range: tuple[float, float] = (0.05, 1.5),
) -> list[list[dict[str, float]]]:
  """Sample reproducible goal-plane targets for each adversarial round."""
  rng = random.Random(seed)
  return [
    [
      {"x": rng.uniform(*x_range), "y": y, "z": rng.uniform(*z_range)}
      for _ in range(targets_per_round)
    ]
    for _ in range(rounds)
  ]


def sample_opponent_ckpt(
  rng: random.Random,
  initial_ckpt: str,
  current_ckpt: str,
  history_ckpts: list[str],
) -> dict[str, Any]:
  eligible_history = [
    ckpt for ckpt in history_ckpts
    if ckpt != initial_ckpt and ckpt != current_ckpt
  ]
  choices: list[tuple[str, float]] = [
    ("current", OPPONENT_CURRENT_PROB),
    ("initial", OPPONENT_INITIAL_PROB),
  ]
  if eligible_history:
    choices.append(("history", OPPONENT_HISTORY_PROB))

  pick = rng.random() * sum(weight for _, weight in choices)
  bucket = choices[-1][0]
  cursor = 0.0
  for name, weight in choices:
    cursor += weight
    if pick <= cursor:
      bucket = name
      break

  if bucket == "history":
    ckpt = rng.choice(eligible_history)
  elif bucket == "initial":
    ckpt = initial_ckpt
  else:
    ckpt = current_ckpt

  return {
    "bucket": bucket,
    "ckpt": ckpt,
    "history_pool": eligible_history,
    "weights": {
      "current": OPPONENT_CURRENT_PROB,
      "history": OPPONENT_HISTORY_PROB,
      "initial": OPPONENT_INITIAL_PROB,
    },
  }


def parse_gpu_ids(raw: str) -> list[int] | Literal["all"] | None:
  raw = raw.strip().lower()
  if raw in ("", "none", "cpu"):
    return None
  if raw == "all":
    return "all"
  return [int(part.strip()) for part in raw.split(",") if part.strip()]


def snapshot_run_names(log_root: Path) -> set[str]:
  return {p.name for p in log_root.iterdir() if p.is_dir()} if log_root.is_dir() else set()


def latest_model_for_run(log_root: Path, run_name: str, before: set[str]) -> Path:
  after = snapshot_run_names(log_root)
  candidates = [log_root / name for name in sorted(after - before) if name.endswith(f"_{run_name}")]
  if not candidates:
    candidates = sorted(log_root.glob(f"*_{run_name}"), key=lambda p: p.stat().st_mtime)
  if not candidates:
    raise FileNotFoundError(f"No run dir found for run_name={run_name} under {log_root}")
  models = sorted(
    candidates[-1].glob("model_*.pt"),
    key=lambda p: int(p.stem.split("_", maxsplit=1)[1]),
  )
  if not models:
    raise FileNotFoundError(f"No model_*.pt found in {candidates[-1]}")
  return models[-1]


def model_step(path: str | Path) -> int | None:
  stem = Path(path).stem
  if not stem.startswith("model_"):
    return None
  try:
    return int(stem.split("_", maxsplit=1)[1])
  except ValueError:
    return None


def write_best_note(
  note_path: Path,
  *,
  role: str,
  round_idx: int,
  input_ckpt: str,
  output_ckpt: str,
  source_output_ckpt: str,
  dry_run: bool,
  candidate_ckpt: str | None = None,
  promotion_eval: dict[str, Any] | None = None,
) -> None:
  source_step = model_step(source_output_ckpt)
  input_step = model_step(input_ckpt)
  source_kind = "trained_model" if (not dry_run and source_step is not None) else "phase_input"
  best_step = source_step if source_kind == "trained_model" else None
  lines = [
    f"role: {role}",
    f"round: {round_idx}",
    f"output_ckpt: {output_ckpt}",
    f"input_ckpt: {input_ckpt}",
    f"source_output_ckpt: {source_output_ckpt}",
    f"source_kind: {source_kind}",
    f"source_step: {best_step if best_step is not None else 'null'}",
    f"input_step: {input_step if input_step is not None else 'null'}",
    f"best_is_phase_input: {str(source_kind == 'phase_input').lower()}",
    f"dry_run: {str(dry_run).lower()}",
  ]
  if candidate_ckpt is not None:
    lines.append(f"candidate_ckpt: {candidate_ckpt}")
  if promotion_eval is not None:
    winner = promotion_eval.get("winner")
    best_source_ckpt = (
      promotion_eval.get("candidate_ckpt")
      if winner == "candidate"
      else promotion_eval.get("previous_ckpt")
    )
    best_source_kind = "candidate" if winner == "candidate" else "previous_best"
    best_source_step = model_step(best_source_ckpt) if best_source_ckpt else None
    lines.extend([
      f"promotion_winner: {winner}",
      f"promotion_trials: {promotion_eval.get('trials')}",
      f"promotion_previous_score: {promotion_eval.get('previous_score')}",
      f"promotion_candidate_score: {promotion_eval.get('candidate_score')}",
      f"best_source_kind: {best_source_kind}",
      f"best_source_ckpt: {best_source_ckpt}",
      f"best_source_step: {best_source_step if best_source_step is not None else 'null'}",
    ])
  note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bind_log_root(agent_cfg: Any, log_root: str) -> Path:
  """Make scripts.train.launch_training write under the requested log root."""
  root = Path(log_root)
  agent_cfg.experiment_name = root.name
  return root


def _training_worker(task_id: str, train_cfg: Any) -> None:
  from scripts.train import launch_training

  launch_training(task_id, train_cfg)


def should_isolate_training_process(train_cfg: Any, cfg: AdversarialConfig) -> bool:
  return cfg.isolated_training_process and train_cfg.__class__.__module__ == "scripts.train"


def launch_training_phase(task_id: str, train_cfg: Any, cfg: AdversarialConfig) -> None:
  if not should_isolate_training_process(train_cfg, cfg):
    from scripts.train import launch_training

    launch_training(task_id, train_cfg)
    return

  import multiprocessing as mp

  proc = mp.get_context("spawn").Process(
    target=_training_worker,
    args=(task_id, train_cfg),
  )
  proc.start()
  proc.join()
  if proc.exitcode != 0:
    raise RuntimeError(f"training phase failed task_id={task_id} exitcode={proc.exitcode}")


def ensure_checkpoint_copy(src: str, dst: Path, role: str) -> None:
  src_path = Path(src)
  if src_path.exists():
    shutil.copy2(src_path, dst)
  else:
    dst.write_text(f"previous best placeholder: {src}\n", encoding="utf-8")


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
  import torch

  loaded = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(loaded, dict):
    raise ValueError(f"{path} is not a checkpoint dict")
  return loaded


def extract_idle_training_checkpoint(init_ckpt: str | Path, out_dir: Path) -> Path:
  """Return an actor checkpoint for idle-only training.

  A MoE7 bundle is deployable, but the idle training runner expects a single
  GoalkeeperActorCritic/AdversarialGoalkeeperActorCritic checkpoint. If a bundle
  is passed, extract only its prepare/idle expert.
  """
  loaded = _load_checkpoint(init_ckpt)
  if "sr" not in loaded:
    return Path(init_ckpt)
  idle = next((loaded[k] for k in ("idle", "prepare", "idle_expert") if k in loaded), None)
  if not isinstance(idle, dict) or "actor_state_dict" not in idle:
    raise KeyError(f"{init_ckpt} is a MoE bundle but has no idle actor checkpoint")
  out_dir.mkdir(parents=True, exist_ok=True)
  out = out_dir / "keeper_idle_train_init.pt"
  import torch

  torch.save(idle, out)
  return out


def extract_moe_expert_training_checkpoint(init_ckpt: str | Path, expert_idx: int, out_dir: Path) -> Path:
  loaded = _load_checkpoint(init_ckpt)
  experts = loaded.get("sr")
  if not isinstance(experts, list) or not 0 <= expert_idx < len(experts):
    raise KeyError(f"{init_ckpt} has no sr[{expert_idx}] expert checkpoint")
  expert = experts[expert_idx]
  if not isinstance(expert, dict) or "actor_state_dict" not in expert:
    raise KeyError(f"{init_ckpt} sr[{expert_idx}] does not contain actor_state_dict")
  out_dir.mkdir(parents=True, exist_ok=True)
  out = out_dir / f"keeper_expert_{expert_idx}_train_init.pt"
  import torch

  torch.save(expert, out)
  return out


def rebuild_moe_with_idle(
  base_ckpt: str | Path,
  idle_ckpt: str | Path,
  out_ckpt: Path,
) -> bool:
  loaded = _load_checkpoint(base_ckpt)
  if "sr" not in loaded:
    return False
  idle = _load_checkpoint(idle_ckpt)
  if "actor_state_dict" not in idle:
    raise KeyError(f"{idle_ckpt} does not contain actor_state_dict")
  bundle = dict(loaded)
  bundle["idle"] = idle
  out_ckpt.parent.mkdir(parents=True, exist_ok=True)
  import torch

  torch.save(bundle, out_ckpt)
  return True


def rebuild_moe_with_experts(
  base_ckpt: str | Path,
  expert_ckpts: dict[int, str | Path],
  idle_ckpt: str | Path,
  out_ckpt: Path,
) -> None:
  bundle = dict(_load_checkpoint(base_ckpt))
  experts = list(bundle.get("sr") or [])
  if len(experts) < 6:
    raise KeyError(f"{base_ckpt} must be a MoE bundle with at least 6 sr experts")
  for idx, ckpt in sorted(expert_ckpts.items()):
    if not 0 <= idx < len(experts):
      raise KeyError(f"sr[{idx}] is outside bundle expert range")
    experts[idx] = _load_checkpoint(ckpt)
  idle = _load_checkpoint(idle_ckpt)
  if "actor_state_dict" not in idle:
    raise KeyError(f"{idle_ckpt} does not contain actor_state_dict")
  bundle["sr"] = experts
  bundle["idle"] = idle
  out_ckpt.parent.mkdir(parents=True, exist_ok=True)
  import torch

  torch.save(bundle, out_ckpt)


class LocalCheckpointPolicy:
  """Compete-compatible local policy loaded from a checkpoint."""

  def __init__(self, role: Literal["shooter", "keeper"], checkpoint: str, device: str):
    import torch
    from src.tasks.soccer.adversarial import (
      FrozenGoalkeeperPolicy,
      FrozenShooterPolicy,
    )

    self.role = role
    self.device = torch.device(device)
    if role == "shooter":
      self.policy = FrozenShooterPolicy(
        checkpoint,
        device=device,
        num_envs=1,
        task_id=SHOOTER_TASK_ID,
      )
      self._last_action = torch.zeros(1, 29, device=self.device)
    else:
      self.policy = FrozenGoalkeeperPolicy(
        checkpoint,
        device=device,
        num_envs=1,
        task_id=KEEPER_TASK_ID,
      )
      self._last_action = torch.zeros(1, 29, device=self.device)

  def reset(self) -> None:
    import torch
    self._last_action = torch.zeros(self._last_action.shape, dtype=self._last_action.dtype, device=self.device)
    reset_fn = getattr(self.policy, "reset", None)
    if reset_fn is not None:
      reset_fn()

  def close(self) -> None:
    close_fn = getattr(self.policy, "close", None)
    if close_fn is not None:
      close_fn()

  def __call__(self, raw_state: dict[str, Any]) -> torch.Tensor:
    del raw_state
    if not hasattr(self, "dual_env"):
      raise RuntimeError("LocalCheckpointPolicy requires bind_env() before use")
    action = self.policy(self.dual_env, self._last_action)
    import torch
    with torch.inference_mode(False):
      self._last_action = action.detach().to(device=self.device).clone()
    return action

  def bind_env(self, env_base: Any):
    if self.role == "shooter":
      scene = {
        "opponent": env_base.scene["shooter"],
        "robot": env_base.scene["goalkeeper"],
        "ball": env_base.scene["ball"],
      }
    else:
      scene = {
        "opponent": env_base.scene["goalkeeper"],
        "robot": env_base.scene["shooter"],
        "ball": env_base.scene["ball"],
      }
    self.dual_env = SimpleNamespace(scene=scene, device=env_base.device, num_envs=1)
    return self


def compare_candidate_to_previous(
  role: Literal["shooter", "keeper"],
  previous_ckpt: str,
  candidate_ckpt: str,
  opponent_ckpt: str,
  trials: int,
  device: str,
  seed: int,
) -> dict[str, Any]:
  import torch
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends
  from scripts.compete import make_compete_env_cfg, run_trial

  configure_torch_backends()
  env_cfg = make_compete_env_cfg()
  env_cfg.scene.num_envs = 1
  metric = "goals" if role == "shooter" else "saves"
  print(
    f"[PROMOTE] start role={role} trials={trials} metric={metric} device={device} seed={seed}",
    flush=True,
  )
  print(f"[PROMOTE] previous_ckpt={previous_ckpt}", flush=True)
  print(f"[PROMOTE] candidate_ckpt={candidate_ckpt}", flush=True)
  print(f"[PROMOTE] opponent_ckpt={opponent_ckpt}", flush=True)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=100.0)

  def _score(label: Literal["previous", "candidate"], active_ckpt: str) -> tuple[int, int, int]:
    print(f"[PROMOTE] scoring {label} active_ckpt={active_ckpt}", flush=True)
    if role == "shooter":
      shooter = LocalCheckpointPolicy("shooter", active_ckpt, device).bind_env(env)
      keeper = LocalCheckpointPolicy("keeper", opponent_ckpt, device).bind_env(env)
    else:
      shooter = LocalCheckpointPolicy("shooter", opponent_ckpt, device).bind_env(env)
      keeper = LocalCheckpointPolicy("keeper", active_ckpt, device).bind_env(env)
    goals = blocks = crossed = 0
    try:
      for trial_idx in range(1, trials + 1):
        stats = run_trial(wrapped, env, shooter, keeper)
        goals += int(stats["goal_scored"])
        blocks += int(stats["blocked"])
        crossed += int(stats["ball_final_x"] <= -0.5)
        score = goals if role == "shooter" else (trial_idx - goals)
        print(
          "[PROMOTE] "
          f"{label} trial={trial_idx}/{trials} "
          f"goal={int(stats['goal_scored'])} block={int(stats['blocked'])} "
          f"final_x={float(stats['ball_final_x']):.3f} "
          f"cum_goals={goals} cum_blocks={blocks} cum_crossed={crossed} score={score}",
          flush=True,
        )
    finally:
      shooter.close()
      keeper.close()
    score = goals if role == "shooter" else (trials - goals)
    return score, goals, blocks

  try:
    torch.manual_seed(seed)
    previous_score, previous_goals, previous_blocks = _score("previous", previous_ckpt)
    torch.manual_seed(seed)
    candidate_score, candidate_goals, candidate_blocks = _score("candidate", candidate_ckpt)
  finally:
    env.close()

  winner = "candidate" if candidate_score > previous_score else "previous"
  print(
    f"[PROMOTE] result role={role} winner={winner} "
    f"previous_score={previous_score} candidate_score={candidate_score} "
    f"previous_goals={previous_goals} candidate_goals={candidate_goals} "
    f"previous_blocks={previous_blocks} candidate_blocks={candidate_blocks}",
    flush=True,
  )
  return {
    "role": role,
    "winner": winner,
    "previous_ckpt": previous_ckpt,
    "candidate_ckpt": candidate_ckpt,
    "opponent_ckpt": opponent_ckpt,
    "trials": trials,
    "device": device,
    "seed": seed,
    "previous_score": previous_score,
    "candidate_score": candidate_score,
    "previous_goals": previous_goals,
    "candidate_goals": candidate_goals,
    "previous_blocks": previous_blocks,
    "candidate_blocks": candidate_blocks,
    "metric": metric,
  }


def promote_checkpoint(
  *,
  role: Literal["shooter", "keeper"],
  previous_ckpt: str,
  candidate_ckpt: str,
  opponent_ckpt: str,
  output_ckpt: Path,
  cfg: AdversarialConfig,
  seed: int,
) -> dict[str, Any] | None:
  if cfg.dry_run or cfg.promotion_trials <= 0:
    shutil.copy2(candidate_ckpt, output_ckpt)
    return None
  result = compare_candidate_to_previous(
    role,
    previous_ckpt,
    candidate_ckpt,
    opponent_ckpt,
    cfg.promotion_trials,
    cfg.promotion_device,
    seed,
  )
  if result["winner"] == "candidate":
    shutil.copy2(candidate_ckpt, output_ckpt)
  else:
    ensure_checkpoint_copy(previous_ckpt, output_ckpt, role)
  return result


def run(cfg: AdversarialConfig) -> dict[str, Any]:
  if cfg.rounds < 1:
    raise ValueError("--rounds must be >= 1")
  if cfg.shooter_targets_per_round < 1:
    raise ValueError("--shooter-targets-per-round must be >= 1")
  if cfg.promotion_trials < 0:
    raise ValueError("--promotion-trials must be >= 0")
  if cfg.keeper_expert_iters_per_round < 1:
    raise ValueError("--keeper-expert-iters-per-round must be >= 1")
  if cfg.keeper_idle_iters_per_round < 1:
    raise ValueError("--keeper-idle-iters-per-round must be >= 1")
  if cfg.target_x_min > cfg.target_x_max:
    raise ValueError("target x min must be <= x max")
  if cfg.target_z_min > cfg.target_z_max:
    raise ValueError("target z min must be <= z max")

  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  shooter_init_ckpt = cfg.shooter_init
  keeper_init_ckpt = cfg.keeper_init
  keeper_idle_init_ckpt = cfg.keeper_idle_init or cfg.keeper_init
  current_shooter_ckpt = shooter_init_ckpt
  current_keeper_ckpt = keeper_init_ckpt
  current_keeper_idle_ckpt = keeper_idle_init_ckpt
  target_center = (
    0.5 * (cfg.target_x_min + cfg.target_x_max),
    cfg.target_y,
    0.5 * (cfg.target_z_min + cfg.target_z_max),
  )
  target_length = cfg.target_x_max - cfg.target_x_min
  target_sampling = {
    "x_range": [cfg.target_x_min, cfg.target_x_max],
    "y": cfg.target_y,
    "z_range": [cfg.target_z_min, cfg.target_z_max],
    "backend_note": "Stage3 currently consumes random x on the goal plane; z samples are recorded for the adversarial schedule.",
  }
  schedule = sample_target_schedule(
    rounds=cfg.rounds,
    targets_per_round=cfg.shooter_targets_per_round,
    seed=cfg.seed,
    x_range=(cfg.target_x_min, cfg.target_x_max),
    y=cfg.target_y,
    z_range=(cfg.target_z_min, cfg.target_z_max),
  )
  manifest: dict[str, Any] = {
    "config": asdict(cfg),
    "opponent_sampling": {
      "current": OPPONENT_CURRENT_PROB,
      "history": OPPONENT_HISTORY_PROB,
      "initial": OPPONENT_INITIAL_PROB,
      "history_rule": "uniform over previous round checkpoints, excluding the current latest checkpoint and the initial checkpoint",
    },
    "target_sampling": target_sampling,
    "target_schedule": schedule,
    "phases": [],
  }

  def write_manifest() -> None:
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
      json.dump(manifest, fh, indent=2, sort_keys=True)
      fh.write("\n")

  opponent_rng = random.Random(cfg.seed + 1009)
  shooter_history: list[str] = []
  keeper_history: list[str] = []

  for round_idx in range(1, cfg.rounds + 1):
    round_start_shooter_ckpt = current_shooter_ckpt
    round_start_keeper_ckpt = current_keeper_ckpt
    round_dir = out_dir / f"round_{round_idx:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    targets_file = round_dir / "targets.json"
    with targets_file.open("w", encoding="utf-8") as fh:
      json.dump(schedule[round_idx - 1], fh, indent=2, sort_keys=True)
      fh.write("\n")

    if cfg.mode in ("alternate", "train-shooter"):
      opponent_sample = sample_opponent_ckpt(
        opponent_rng,
        keeper_init_ckpt,
        current_keeper_ckpt,
        keeper_history,
      )
      run_name = f"adv_r{round_idx:03d}_shooter"
      shooter_out = round_dir / "shooter_best.pt"
      shooter_candidate = round_dir / "shooter_candidate.pt"
      shooter_note = round_dir / "shooter_best.note.txt"
      source_output = str(shooter_candidate)
      if cfg.dry_run:
        shooter_candidate.write_text(f"dry-run placeholder for {current_shooter_ckpt}\n", encoding="utf-8")
      else:
        import mjlab.tasks  # noqa: F401
        import src.tasks  # noqa: F401
        from scripts.train import TrainConfig, launch_training

        shooter_train_cfg = TrainConfig.from_task(SHOOTER_TASK_ID)
        shooter_train_cfg = replace(
          shooter_train_cfg,
          motion_dir=cfg.motion_dir,
          load_checkpoint_path=current_shooter_ckpt,
          frozen_opponent_checkpoint_path=opponent_sample["ckpt"],
          frozen_opponent_role="goalkeeper",
          frozen_opponent_task_id=KEEPER_TASK_ID,
          gpu_ids=parse_gpu_ids(cfg.gpu_ids),
        )
        shooter_train_cfg.env.scene.num_envs = cfg.num_envs
        shooter_train_cfg.agent.max_iterations = cfg.shooter_iters_per_round
        shooter_train_cfg.agent.run_name = run_name
        shooter_train_cfg.agent.save_interval = 100
        motion_cmd = shooter_train_cfg.env.commands.get("motion")
        if motion_cmd is not None:
          motion_cmd.destination_center = target_center
          motion_cmd.destination_length = target_length
          motion_cmd.destination_width = 0.0

        log_root = bind_log_root(shooter_train_cfg.agent, cfg.shooter_log_root)
        before = snapshot_run_names(log_root)
        launch_training_phase(SHOOTER_TASK_ID, shooter_train_cfg, cfg)
        source_model = latest_model_for_run(log_root, run_name, before)
        shutil.copy2(source_model, shooter_candidate)
        source_output = str(source_model)

      promotion_eval = promote_checkpoint(
        role="shooter",
        previous_ckpt=current_shooter_ckpt,
        candidate_ckpt=str(shooter_candidate),
        opponent_ckpt=round_start_keeper_ckpt,
        output_ckpt=shooter_out,
        cfg=cfg,
        seed=cfg.seed + round_idx * 100 + 1,
      )

      phase = {
        "round": round_idx,
        "role": "shooter",
        "backend": "scripts.train.launch_training",
        "task_id": SHOOTER_TASK_ID,
        "input_ckpt": current_shooter_ckpt,
        "opponent_ckpt": opponent_sample["ckpt"],
        "opponent_sample": opponent_sample,
        "opponent_consumed_by_backend": True,
        "output_ckpt": str(shooter_out),
        "candidate_ckpt": str(shooter_candidate),
        "source_output_ckpt": source_output,
        "best_note": str(shooter_note),
        "promotion_eval": promotion_eval,
        "promoted": promotion_eval is None or promotion_eval.get("winner") == "candidate",
        "targets_file": str(targets_file),
        "target_sampling": target_sampling,
        "dry_run": cfg.dry_run,
      }
      write_best_note(
        shooter_note,
        role="shooter",
        round_idx=round_idx,
        input_ckpt=phase["input_ckpt"],
        output_ckpt=phase["output_ckpt"],
        source_output_ckpt=phase["source_output_ckpt"],
        dry_run=cfg.dry_run,
        candidate_ckpt=phase["candidate_ckpt"],
        promotion_eval=promotion_eval,
      )
      manifest["phases"].append(phase)
      current_shooter_ckpt = phase["output_ckpt"]
      shooter_history.append(current_shooter_ckpt)
      write_manifest()

    if cfg.mode == "train-keeper-moe7":
      opponent_sample = sample_opponent_ckpt(
        opponent_rng,
        shooter_init_ckpt,
        current_shooter_ckpt,
        shooter_history,
      )
      moe_out = round_dir / "keeper_moe7_best.pt"
      moe_candidate = round_dir / "keeper_moe7_candidate.pt"
      moe_note = round_dir / "keeper_moe7_best.note.txt"
      expert_runs: list[dict[str, Any]] = []
      expert_candidates: dict[int, Path] = {}
      idle_expert_candidate = round_dir / "keeper_idle_expert_candidate.pt"
      if cfg.dry_run:
        for expert_idx in range(6):
          candidate = round_dir / f"keeper_expert_{expert_idx}_candidate.pt"
          train_input = round_dir / f"keeper_expert_{expert_idx}_train_init.pt"
          candidate.write_text(f"dry-run placeholder for sr{expert_idx} from {current_keeper_ckpt}\n", encoding="utf-8")
          expert_runs.append({
            "expert": f"sr{expert_idx}",
            "task_id": KEEPER_EXPERT_TASK_ID,
            "input_ckpt": str(train_input),
            "opponent_ckpt": opponent_sample["ckpt"],
            "candidate_ckpt": str(candidate),
            "source_output_ckpt": str(candidate),
            "max_iterations": cfg.keeper_expert_iters_per_round,
          })
        idle_expert_candidate.write_text(f"dry-run placeholder for idle from {current_keeper_idle_ckpt}\n", encoding="utf-8")
        expert_runs.append({
          "expert": "idle",
          "task_id": KEEPER_IDLE_TASK_ID,
          "input_ckpt": str(round_dir / "keeper_idle_train_init.pt"),
          "opponent_ckpt": opponent_sample["ckpt"],
          "candidate_ckpt": str(idle_expert_candidate),
          "source_output_ckpt": str(idle_expert_candidate),
          "max_iterations": cfg.keeper_idle_iters_per_round,
        })
        moe_candidate.write_text(f"dry-run placeholder for MoE7 from {current_keeper_ckpt}\n", encoding="utf-8")
      else:
        import mjlab.tasks  # noqa: F401
        import src.tasks  # noqa: F401
        from scripts.train import TrainConfig, launch_training

        for expert_idx in range(6):
          run_name = f"adv_r{round_idx:03d}_keeper_expert{expert_idx}"
          train_input = extract_moe_expert_training_checkpoint(current_keeper_ckpt, expert_idx, round_dir)
          expert_candidate = round_dir / f"keeper_expert_{expert_idx}_candidate.pt"
          expert_train_cfg = TrainConfig.from_task(KEEPER_EXPERT_TASK_ID)
          expert_train_cfg = replace(
            expert_train_cfg,
            load_actor_only=True,
            load_checkpoint_path=str(train_input),
            frozen_opponent_checkpoint_path=opponent_sample["ckpt"],
            frozen_opponent_role="shooter",
            frozen_opponent_task_id=SHOOTER_TASK_ID,
            gpu_ids=parse_gpu_ids(cfg.gpu_ids),
          )
          expert_train_cfg.env.scene.num_envs = cfg.num_envs
          expert_train_cfg.agent.max_iterations = cfg.keeper_expert_iters_per_round
          expert_train_cfg.agent.run_name = run_name
          expert_train_cfg.agent.save_interval = 100

          log_root = bind_log_root(expert_train_cfg.agent, cfg.keeper_log_root)
          before = snapshot_run_names(log_root)
          launch_training_phase(KEEPER_EXPERT_TASK_ID, expert_train_cfg, cfg)
          source_model = latest_model_for_run(log_root, run_name, before)
          shutil.copy2(source_model, expert_candidate)
          expert_candidates[expert_idx] = expert_candidate
          expert_runs.append({
            "expert": f"sr{expert_idx}",
            "task_id": KEEPER_EXPERT_TASK_ID,
            "input_ckpt": str(train_input),
            "opponent_ckpt": opponent_sample["ckpt"],
            "candidate_ckpt": str(expert_candidate),
            "source_output_ckpt": str(source_model),
            "max_iterations": cfg.keeper_expert_iters_per_round,
          })

        run_name = f"adv_r{round_idx:03d}_keeper_idle"
        idle_train_input = extract_idle_training_checkpoint(current_keeper_idle_ckpt, round_dir)
        idle_train_cfg = TrainConfig.from_task(KEEPER_IDLE_TASK_ID)
        idle_train_cfg = replace(
          idle_train_cfg,
          load_actor_only=True,
          load_checkpoint_path=str(idle_train_input),
          frozen_opponent_checkpoint_path=opponent_sample["ckpt"],
          frozen_opponent_role="shooter",
          frozen_opponent_task_id=SHOOTER_TASK_ID,
          gpu_ids=parse_gpu_ids(cfg.gpu_ids),
        )
        idle_train_cfg.env.scene.num_envs = cfg.num_envs
        idle_train_cfg.agent.max_iterations = cfg.keeper_idle_iters_per_round
        idle_train_cfg.agent.run_name = run_name
        idle_train_cfg.agent.save_interval = 100

        log_root = bind_log_root(idle_train_cfg.agent, cfg.keeper_log_root)
        before = snapshot_run_names(log_root)
        launch_training_phase(KEEPER_IDLE_TASK_ID, idle_train_cfg, cfg)
        source_model = latest_model_for_run(log_root, run_name, before)
        shutil.copy2(source_model, idle_expert_candidate)
        expert_runs.append({
          "expert": "idle",
          "task_id": KEEPER_IDLE_TASK_ID,
          "input_ckpt": str(idle_train_input),
          "opponent_ckpt": opponent_sample["ckpt"],
          "candidate_ckpt": str(idle_expert_candidate),
          "source_output_ckpt": str(source_model),
          "max_iterations": cfg.keeper_idle_iters_per_round,
        })
        rebuild_moe_with_experts(current_keeper_ckpt, expert_candidates, source_model, moe_candidate)

      promotion_eval = promote_checkpoint(
        role="keeper",
        previous_ckpt=current_keeper_ckpt,
        candidate_ckpt=str(moe_candidate),
        opponent_ckpt=round_start_shooter_ckpt,
        output_ckpt=moe_out,
        cfg=cfg,
        seed=cfg.seed + round_idx * 100 + 3,
      )
      phase = {
        "round": round_idx,
        "role": "keeper_moe7",
        "backend": "scripts.train.launch_training",
        "task_id": KEEPER_EXPERT_TASK_ID,
        "idle_task_id": KEEPER_IDLE_TASK_ID,
        "input_ckpt": current_keeper_ckpt,
        "idle_input_ckpt": current_keeper_idle_ckpt,
        "opponent_ckpt": opponent_sample["ckpt"],
        "opponent_sample": opponent_sample,
        "opponent_consumed_by_backend": True,
        "output_ckpt": str(moe_out),
        "candidate_ckpt": str(moe_candidate),
        "source_output_ckpt": str(moe_candidate),
        "expert_runs": expert_runs,
        "best_note": str(moe_note),
        "promotion_eval": promotion_eval,
        "promoted": promotion_eval is None or promotion_eval.get("winner") == "candidate",
        "selection_metric": "rebuilt_moe7_after_all_expert_training",
        "dry_run": cfg.dry_run,
      }
      write_best_note(
        moe_note,
        role="keeper_moe7",
        round_idx=round_idx,
        input_ckpt=phase["input_ckpt"],
        output_ckpt=phase["output_ckpt"],
        source_output_ckpt=phase["source_output_ckpt"],
        dry_run=cfg.dry_run,
        candidate_ckpt=phase["candidate_ckpt"],
        promotion_eval=promotion_eval,
      )
      manifest["phases"].append(phase)
      current_keeper_ckpt = phase["output_ckpt"]
      current_keeper_idle_ckpt = phase["output_ckpt"]
      keeper_history.append(current_keeper_ckpt)
      write_manifest()

    if cfg.mode == "train-keeper-idle":
      opponent_sample = sample_opponent_ckpt(
        opponent_rng,
        shooter_init_ckpt,
        current_shooter_ckpt,
        shooter_history,
      )
      idle_out = round_dir / "keeper_idle_best.pt"
      idle_candidate = round_dir / "keeper_idle_candidate.pt"
      idle_expert_candidate = round_dir / "keeper_idle_expert_candidate.pt"
      idle_note = round_dir / "keeper_idle_best.note.txt"
      source_output = str(idle_candidate)
      idle_train_input = Path(current_keeper_idle_ckpt)
      if cfg.dry_run:
        idle_candidate.write_text(f"dry-run placeholder for {current_keeper_idle_ckpt}\n", encoding="utf-8")
      else:
        import mjlab.tasks  # noqa: F401
        import src.tasks  # noqa: F401
        from scripts.train import TrainConfig, launch_training

        run_name = f"adv_r{round_idx:03d}_keeper_idle"
        idle_train_input = Path(current_keeper_idle_ckpt)
        idle_train_cfg = TrainConfig.from_task(KEEPER_MOE7_PREPARE_TASK_ID)
        idle_train_cfg = replace(
          idle_train_cfg,
          load_checkpoint_path=str(idle_train_input),
          load_actor_only=True,
          frozen_opponent_checkpoint_path=opponent_sample["ckpt"],
          frozen_opponent_role="shooter",
          frozen_opponent_task_id=SHOOTER_TASK_ID,
          gpu_ids=parse_gpu_ids(cfg.gpu_ids),
        )
        idle_train_cfg.env.scene.num_envs = cfg.num_envs
        idle_train_cfg.agent.max_iterations = cfg.keeper_idle_iters_per_round
        idle_train_cfg.agent.run_name = run_name
        idle_train_cfg.agent.save_interval = 100

        log_root = bind_log_root(idle_train_cfg.agent, cfg.keeper_log_root)
        before = snapshot_run_names(log_root)
        launch_training_phase(KEEPER_MOE7_PREPARE_TASK_ID, idle_train_cfg, cfg)
        source_model = latest_model_for_run(log_root, run_name, before)
        shutil.copy2(source_model, idle_expert_candidate)
        loaded_source = _load_checkpoint(source_model)
        actor_bundle = loaded_source.get("actor_state_dict") if isinstance(loaded_source, dict) else None
        if not isinstance(actor_bundle, dict) or "sr" not in actor_bundle:
          raise KeyError(f"{source_model} does not contain a full MoE7 actor_state_dict")
        import torch

        torch.save(actor_bundle, idle_candidate)
        source_output = str(source_model)

      shutil.copy2(idle_candidate, idle_out)
      promotion_eval = None

      phase = {
        "round": round_idx,
        "role": "keeper_idle",
        "backend": "scripts.train.launch_training",
        "task_id": KEEPER_MOE7_PREPARE_TASK_ID,
        "input_ckpt": current_keeper_idle_ckpt,
        "idle_train_input_ckpt": str(idle_train_input),
        "opponent_ckpt": opponent_sample["ckpt"],
        "opponent_sample": opponent_sample,
        "opponent_consumed_by_backend": True,
        "output_ckpt": str(idle_out),
        "candidate_ckpt": str(idle_candidate),
        "idle_expert_candidate_ckpt": str(idle_expert_candidate),
        "source_output_ckpt": source_output,
        "best_note": str(idle_note),
        "promotion_eval": promotion_eval,
        "promoted": True,
        "selection_metric": "latest_idle_training_output",
        "dry_run": cfg.dry_run,
      }
      write_best_note(
        idle_note,
        role="keeper_idle",
        round_idx=round_idx,
        input_ckpt=phase["input_ckpt"],
        output_ckpt=phase["output_ckpt"],
        source_output_ckpt=phase["source_output_ckpt"],
        dry_run=cfg.dry_run,
        candidate_ckpt=phase["candidate_ckpt"],
        promotion_eval=promotion_eval,
      )
      manifest["phases"].append(phase)
      current_keeper_idle_ckpt = phase["output_ckpt"]
      write_manifest()

    if cfg.mode in ("alternate", "train-keeper"):
      opponent_sample = sample_opponent_ckpt(
        opponent_rng,
        shooter_init_ckpt,
        current_shooter_ckpt,
        shooter_history,
      )
      keeper_out = round_dir / "keeper_best.pt"
      keeper_candidate = round_dir / "keeper_candidate.pt"
      keeper_note = round_dir / "keeper_best.note.txt"
      source_output = str(keeper_candidate)
      if cfg.dry_run:
        keeper_candidate.write_text(f"dry-run placeholder for {current_keeper_ckpt}\n", encoding="utf-8")
      else:
        import mjlab.tasks  # noqa: F401
        import src.tasks  # noqa: F401
        from scripts.train import TrainConfig, launch_training

        run_name = f"adv_r{round_idx:03d}_keeper"
        keeper_train_cfg = TrainConfig.from_task(KEEPER_TASK_ID)
        keeper_train_cfg = replace(
          keeper_train_cfg,
          load_checkpoint_path=current_keeper_ckpt,
          frozen_opponent_checkpoint_path=opponent_sample["ckpt"],
          frozen_opponent_role="shooter",
          frozen_opponent_task_id=SHOOTER_TASK_ID,
          gpu_ids=parse_gpu_ids(cfg.gpu_ids),
        )
        keeper_train_cfg.env.scene.num_envs = cfg.num_envs
        keeper_train_cfg.agent.max_iterations = cfg.keeper_blocks_per_round * cfg.keeper_block_iters
        keeper_train_cfg.agent.run_name = run_name
        keeper_train_cfg.agent.save_interval = 100

        log_root = bind_log_root(keeper_train_cfg.agent, cfg.keeper_log_root)
        before = snapshot_run_names(log_root)
        launch_training_phase(KEEPER_TASK_ID, keeper_train_cfg, cfg)
        source_model = latest_model_for_run(log_root, run_name, before)
        shutil.copy2(source_model, keeper_candidate)
        source_output = str(source_model)

      promotion_eval = promote_checkpoint(
        role="keeper",
        previous_ckpt=current_keeper_ckpt,
        candidate_ckpt=str(keeper_candidate),
        opponent_ckpt=round_start_shooter_ckpt,
        output_ckpt=keeper_out,
        cfg=cfg,
        seed=cfg.seed + round_idx * 100 + 2,
      )

      phase = {
        "round": round_idx,
        "role": "keeper",
        "backend": "scripts.train.launch_training",
        "task_id": KEEPER_TASK_ID,
        "input_ckpt": current_keeper_ckpt,
        "opponent_ckpt": opponent_sample["ckpt"],
        "opponent_sample": opponent_sample,
        "opponent_consumed_by_backend": True,
        "output_ckpt": str(keeper_out),
        "candidate_ckpt": str(keeper_candidate),
        "source_output_ckpt": source_output,
        "best_note": str(keeper_note),
        "promotion_eval": promotion_eval,
        "promoted": promotion_eval is None or promotion_eval.get("winner") == "candidate",
        "dry_run": cfg.dry_run,
      }
      write_best_note(
        keeper_note,
        role="keeper",
        round_idx=round_idx,
        input_ckpt=phase["input_ckpt"],
        output_ckpt=phase["output_ckpt"],
        source_output_ckpt=phase["source_output_ckpt"],
        dry_run=cfg.dry_run,
        candidate_ckpt=phase["candidate_ckpt"],
        promotion_eval=promotion_eval,
      )
      manifest["phases"].append(phase)
      current_keeper_ckpt = phase["output_ckpt"]
      keeper_history.append(current_keeper_ckpt)
      write_manifest()

  manifest["final"] = {
    "shooter_ckpt": current_shooter_ckpt,
    "keeper_ckpt": current_keeper_ckpt,
    "keeper_idle_ckpt": current_keeper_idle_ckpt,
  }
  write_manifest()
  print(f"[INFO] wrote adversarial manifest to {out_dir / 'manifest.json'}", flush=True)
  return manifest


def parse_config(argv: list[str] | None = None) -> AdversarialConfig:
  defaults = AdversarialConfig()
  parser = argparse.ArgumentParser(
    prog="train_adversarial",
    description="Alternate frozen-opponent shooter/keeper training phases.",
  )
  parser.add_argument("--mode", choices=[
    "alternate", "train-shooter", "train-keeper", "train-keeper-idle", "train-keeper-moe7",
  ], default=defaults.mode)
  parser.add_argument("--keeper-only", action="store_true", help="Only train the goalkeeper; keep the shooter frozen at --shooter-init.")
  parser.add_argument("--keeper-idle-only", action="store_true", help="Only train the goalkeeper idle expert in the dual-robot adversarial env.")
  parser.add_argument("--keeper-moe7-only", action="store_true", help="Train all 7 goalkeeper MoE experts against a frozen shooter.")
  parser.add_argument("--shooter-only", action="store_true", help="Only train the shooter; keep the goalkeeper frozen at --keeper-init.")
  parser.add_argument("--rounds", type=int, default=defaults.rounds)
  parser.add_argument("--out-dir", default=defaults.out_dir)
  parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=defaults.dry_run)
  parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
  parser.add_argument("--seed", type=int, default=defaults.seed)
  parser.add_argument("--shooter-init", default=defaults.shooter_init)
  parser.add_argument("--keeper-init", default=defaults.keeper_init)
  parser.add_argument("--keeper-idle-init", default=defaults.keeper_idle_init)
  parser.add_argument("--motion-dir", default=defaults.motion_dir)
  parser.add_argument("--shooter-log-root", default=defaults.shooter_log_root)
  parser.add_argument("--keeper-log-root", default=defaults.keeper_log_root)
  parser.add_argument("--num-envs", type=int, default=defaults.num_envs)
  parser.add_argument("--gpu-ids", default=defaults.gpu_ids)
  parser.add_argument("--keeper-device", default=defaults.keeper_device)
  parser.add_argument("--shooter-iters-per-round", type=int, default=defaults.shooter_iters_per_round)
  parser.add_argument("--keeper-blocks-per-round", type=int, default=defaults.keeper_blocks_per_round)
  parser.add_argument("--keeper-block-iters", type=int, default=defaults.keeper_block_iters)
  parser.add_argument("--keeper-expert-iters-per-round", type=int, default=defaults.keeper_expert_iters_per_round)
  parser.add_argument("--keeper-idle-iters-per-round", type=int, default=defaults.keeper_idle_iters_per_round)
  parser.add_argument("--keeper-warmup", type=int, default=defaults.keeper_warmup)
  parser.add_argument("--promotion-trials", type=int, default=defaults.promotion_trials)
  parser.add_argument("--promotion-device", default=defaults.promotion_device)
  parser.add_argument("--shooter-targets-per-round", type=int, default=defaults.shooter_targets_per_round)
  parser.add_argument("--target-x-min", type=float, default=defaults.target_x_min)
  parser.add_argument("--target-x-max", type=float, default=defaults.target_x_max)
  parser.add_argument("--target-y", type=float, default=defaults.target_y)
  parser.add_argument("--target-z-min", type=float, default=defaults.target_z_min)
  parser.add_argument("--target-z-max", type=float, default=defaults.target_z_max)
  parsed = vars(parser.parse_args(argv))
  keeper_only = parsed.pop("keeper_only")
  keeper_idle_only = parsed.pop("keeper_idle_only")
  keeper_moe7_only = parsed.pop("keeper_moe7_only")
  shooter_only = parsed.pop("shooter_only")
  if sum(bool(x) for x in (keeper_only, keeper_idle_only, keeper_moe7_only, shooter_only)) > 1:
    parser.error("--keeper-only, --keeper-idle-only, --keeper-moe7-only and --shooter-only are mutually exclusive")
  if keeper_only:
    parsed["mode"] = "train-keeper"
  if keeper_idle_only:
    parsed["mode"] = "train-keeper-idle"
  if keeper_moe7_only:
    parsed["mode"] = "train-keeper-moe7"
  if shooter_only:
    parsed["mode"] = "train-shooter"
  return AdversarialConfig(**parsed)


def main() -> None:
  run(parse_config())


if __name__ == "__main__":
  main()
