"""Train RL agents with PPO (RSL-RL) for G1 soccer skills.

Algorithm: clipped PPO with GAE, adaptive learning rate (KL-constrained).
  - clip_param=0.2, gamma=0.99, lam=0.95, entropy_coef=0.005
  - num_learning_epochs=5, num_mini_batches=4, learning_rate=1e-3

Task IDs:
  Unitree-G1-Shooter          Playable shooter (zero-agent only, no training)
  Unitree-G1-Goalkeeper       Playable goalkeeper (zero-agent only, no training)
  Unitree-G1-Shooter-Stage1   Training: motion tracking (LSTM, 128-64-32 + 2x128)
  Unitree-G1-Shooter-Stage2   Training: perception-guided kicking (same LSTM)
  Unitree-G1-Shooter-Student  Training: motion-free student PPO (MLP, BC-initialized)

The PAiD paper (arXiv:2602.05310) uses a two-stage training strategy.
Both stages share the same LSTM architecture and observation space (160D),
so Stage II resumes from Stage I via standard ``runner.load()`` with
no special weight transfer needed.

For the fully automated two-stage pipeline, see
``scripts/train_pipeline.py``.

Usage:
  # ---- Stage I: motion tracking ----
  python scripts/train.py Unitree-G1-Shooter-Stage1 \\
      --motion-dir src/assets/soccer/motions/shooter

  # Stage I + GPU selection
  CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py Unitree-G1-Shooter-Stage1 \\
      --motion-dir src/assets/soccer/motions/shooter \\
      --env.scene.num-envs 2048 --gpu-ids 0,1

  # ---- Stage II: perception-guided kicking (transfer from Stage I) ----
  python scripts/train.py Unitree-G1-Shooter-Stage2 \\
      --motion-dir src/assets/soccer/motions/shooter \\
      --env.scene.num-envs 2048 --gpu-ids 0,1 \\
      --agent.resume True \\
      --agent.load-run "2026-06-10_14-26-14" \\
      --agent.load-checkpoint "model_10000.pt" \\
      --agent.run-name shooter_stage2

  # ---- Resume interrupted training (same stage) ----
  python scripts/train.py Unitree-G1-Shooter-Stage2 \\
      --motion-dir src/assets/soccer/motions/shooter \\
      --env.scene.num-envs 2048 --gpu-ids 0,1 \\
      --agent.resume True \\
      --agent.load-run "2026-06-10_14-26-14" \\
      --agent.load-checkpoint "model_6600.pt"

  # ---- Initialize PPO actor from BC, with fresh critic/optimizer ----
  python scripts/train.py <student-task-id> \\
      --load-checkpoint-path logs/bc/shooter_student/<bc-run>/model_best.pt \\
      --load-actor-only True
"""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from src.tasks.soccer.mdp.shooter_commands import MultiMotionSoccerCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
  motion_dir: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  load_actor_only: bool = False
  load_model_only: bool = False
  load_checkpoint_path: str | None = None
  frozen_opponent_checkpoint_path: str | None = None
  frozen_opponent_role: Literal["shooter", "goalkeeper"] | None = None
  frozen_opponent_task_id: str | None = None
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  ball_speed_std: float | None = None
  actor_std_override: float | None = None
  init_at_random_ep_len: bool = True

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def _override_actor_std(runner, std: float) -> None:
  actor = getattr(getattr(runner, "alg", None), "actor", None)
  distribution = getattr(actor, "distribution", None)
  std_param = getattr(distribution, "std_param", None)
  if std_param is None:
    raise AttributeError("actor distribution does not expose std_param")
  with torch.no_grad():
    std_param.fill_(float(std))
  print(f"[INFO] Overriding loaded actor action std: {float(std):.4f}")


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  # Check for soccer multi-motion training tasks.
  is_soccer_training = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MultiMotionSoccerCommandCfg
  )

  if is_soccer_training:
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MultiMotionSoccerCommandCfg)
    if cfg.motion_dir:
      motion_dir = Path(cfg.motion_dir).expanduser().resolve()
      if not motion_dir.exists():
        raise FileNotFoundError(f"Motion directory not found: {motion_dir}")
      motion_cmd.motion_dir = str(motion_dir)
    if not motion_cmd.motion_dir:
      raise ValueError(
        "For soccer training tasks, --motion-dir must be set "
        "(or configured in the task registration)."
      )
    print(f"[INFO] Using motion directory: {motion_cmd.motion_dir}")

  # Override ball_speed std when --ball-speed-std is set (Stage 3 curriculum).
  if cfg.ball_speed_std is not None and "ball_speed" in cfg.env.rewards:
    cfg.env.rewards["ball_speed"] = replace(
      cfg.env.rewards["ball_speed"],
      params={**cfg.env.rewards["ball_speed"].params, "std": cfg.ball_speed_std},
    )
    print(f"[INFO] Overriding ball_speed std: {cfg.ball_speed_std}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.load_checkpoint_path is not None:
    resume_path = Path(cfg.load_checkpoint_path).expanduser().resolve()
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
  elif cfg.agent.resume:
    # Load checkpoint from local filesystem.
    resume_path = get_checkpoint_path(
      log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
    )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
  if cfg.frozen_opponent_checkpoint_path:
    if cfg.frozen_opponent_role is None or cfg.frozen_opponent_task_id is None:
      raise ValueError(
        "frozen_opponent_role and frozen_opponent_task_id must be set with "
        "frozen_opponent_checkpoint_path"
      )
    from src.tasks.soccer.adversarial import (
      FrozenOpponentSpec,
      FrozenOpponentVecEnvWrapper,
      build_frozen_opponent_policy,
    )

    opponent_spec = FrozenOpponentSpec(
      role=cfg.frozen_opponent_role,
      checkpoint=cfg.frozen_opponent_checkpoint_path,
      task_id=cfg.frozen_opponent_task_id,
    )
    opponent_policy = build_frozen_opponent_policy(
      opponent_spec, device=device, num_envs=env.num_envs,
    )
    env = FrozenOpponentVecEnvWrapper(
      env,
      opponent_policy=opponent_policy,
      opponent_role=cfg.frozen_opponent_role,
    )

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if cfg.load_actor_only:
      runner.load(
        str(resume_path),
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
      )
    elif cfg.load_model_only:
      runner.load(
        str(resume_path),
        load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False},
      )
    else:
      runner.load(str(resume_path))
    if cfg.actor_std_override is not None:
      _override_actor_std(runner, cfg.actor_std_override)

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations,
    init_at_random_ep_len=cfg.init_at_random_ep_len,
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
