"""Train the position-conditioned goalkeeper student with PPO.

This is a non-adversarial training entrypoint.  The actor is the FiLM LSTM
student used by ``train_goalkeeper_student_bc.py`` and remains loadable by
``eval_naive_goalkeeper.py``.  The critic is also recurrent, and PPO keeps an
online rollout distillation loss against a frozen MoE teacher.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

try:
  from scripts.train import TrainConfig, _override_actor_std, launch_training
except ModuleNotFoundError:  # Direct execution: python scripts/train_goalkeeper_student_ppo.py
  from train import TrainConfig, _override_actor_std, launch_training
from src.tasks.soccer.mdp.goalkeeper_ball_reset import (
  launch_staged_ball_after_delay,
  reset_ball_staged_delayed_launch,
)

DEFAULT_TEACHER = (
  "/data/Courses/[CS2810]EmbodiedAI/humanoid_soccer_proj/G1-mjlab-soccer/"
  "logs/repairs/goalkeeper_moe7_hard3_default_idle.pt"
)


@dataclass
class FineTuneConfig:
  init: str | None = None
  """Optional student checkpoint used to initialize the PPO actor."""

  resume_full: bool = False
  """When --init is set, load actor, critic, optimizer, and iteration."""

  task_id: str = "Unitree-G1-Goalkeeper-Student-PPO"
  num_envs: int = 1024
  max_iterations: int = 10000
  run_name: str = "goalkeeper_student_ppo"
  device_ids: list[int] | None = None
  delayed_launch: bool = True
  launch_delay_s: float = 3.0
  teacher: str = DEFAULT_TEACHER
  """Frozen MoE teacher checkpoint used for online rollout distillation."""

  distill_coef: float = 1.0
  """Weight on MSE(student mean, teacher action) during PPO updates."""

  distill_final_coef: float | None = None
  """Final online distillation weight after optional linear annealing."""

  distill_anneal_updates: int = 0
  """Number of PPO rollout updates used to linearly anneal distillation weight."""

  teacher_every_n_steps: int = 1
  """Query teacher every N rollout steps; non-teacher steps use current actor mean."""

  init_action_std: float | None = None
  """Override the loaded actor action std after checkpoint load."""

  min_action_std: float | None = None
  """Optional lower bound for the learned action std during PPO updates."""

  max_action_std: float | None = None
  """Optional upper bound for the learned action std during conservative polish."""

  actor_mean_clip: float | None = None
  """Optional clamp on actor deterministic means for recovery finetuning."""

  ppo_log_ratio_clip: float = 8.0
  """Clamp PPO log-prob ratios before exp to avoid NaN recovery updates."""

  idle_deterministic_actions: bool = True
  """Use deterministic actor means in prelaunch prepare states."""

  mask_idle_actor_loss: bool = False
  """Do not update the actor with PPO surrogate/entropy loss on prepare samples."""

  entropy_coef: float | None = None
  """Optional entropy coefficient override for low-noise finetuning."""

  learning_rate: float | None = None
  """Optional PPO learning rate override for conservative finetuning."""

  num_learning_epochs: int = 5
  """Number of PPO optimization epochs per rollout."""

  normalize_advantage_per_mini_batch: bool = False
  """Normalize PPO advantages inside each mini-batch for recovery finetuning."""

  distill_loss_type: str = "mse"
  """Online distillation loss: mse or huber."""

  profile: Literal["default", "bc_regularized_finetune", "student_polish"] = "default"
  """Reward/regularization preset for this finetune run."""

  offline_bc_dataset: str | None = None
  """Teacher rollout dataset used as an offline BC regularizer."""

  offline_bc_coef: float = 0.0
  """Weight for offline BC regularization during PPO updates."""

  offline_bc_final_coef: float | None = None
  """Final active/offline BC weight after optional linear annealing."""

  offline_bc_anneal_updates: int = 0
  """Number of PPO rollout updates used to linearly anneal offline BC weight."""

  offline_idle_bc_coef: float = 0.0
  """Extra BC weight on idle/prelaunch samples to preserve the ready stance."""

  offline_idle_bc_final_coef: float | None = None
  """Final idle-only BC weight after optional linear annealing."""

  offline_idle_bc_anneal_updates: int = 0
  """Number of PPO rollout updates used to linearly anneal idle-only BC weight."""

  offline_bc_active_fraction: float = 0.5
  """Fraction of non-idle offline BC samples drawn from active/post-launch segments."""

  offline_bc_batch_size: int = 64
  """Number of BC sequences sampled per PPO mini-batch."""

  offline_bc_seq_len: int = 24
  """Number of timesteps per sampled BC sequence."""

  offline_bc_every_n_updates: int = 1
  """Apply offline BC every N PPO optimizer updates."""

  offline_bc_cache_shards: int = 8
  """Number of offline BC shards prefetched once per PPO rollout update."""

  freeze_actor_obs_normalization: bool = True
  """Keep the BC actor observation normalizer fixed during RL finetuning."""

  condition_aux_coef: float = 0.05
  """Weight for auxiliary 7-class MoE route prediction from the student latent."""

  condition_aux_final_coef: float | None = None
  """Final auxiliary condition prediction weight after optional linear annealing."""

  condition_aux_anneal_updates: int = 0
  """Number of PPO rollout updates used to linearly anneal condition aux weight."""

  condition_aux_active_only: bool = True
  """Apply auxiliary route loss only to active/post-launch samples."""

  critic_warmup_iterations: int | None = None
  """Freeze the actor and train only the critic for this many initial iterations."""

  eval_interval: int = 0
  """Polish mode: train iterations between deterministic eval/rollback checks."""

  eval_resets: int = 1
  """Polish mode: number of vectorized resets per rollback eval."""

  eval_steps: int = 300
  """Polish mode: deterministic eval horizon in env steps."""

  rollback_drop: float = 0.0
  """Polish mode: rollback if block rate drops this much below best; 0 disables."""

  seed: int = 2810
  video: bool = False
  save_interval: int = 100


def _apply_delayed_launch_sampling(env_cfg, wait_s: float) -> None:
  reset_event = env_cfg.events["reset_ball"]
  ball_pos = (3.0, 0.0, 0.1)
  sampler_params = dict(reset_event.params)
  sampler_params["fixed_start_local"] = ball_pos
  env_cfg.episode_length_s += wait_s
  env_cfg.events.pop("push_robot", None)
  env_cfg.events.pop("perturb_ball_vel", None)
  env_cfg.events["reset_ball"] = EventTermCfg(
    func=reset_ball_staged_delayed_launch,
    mode="reset",
    params={
      "sampler_func": reset_event.func,
      "sampler_params": sampler_params,
      "ball_pos": ball_pos,
      "ball_cfg": SceneEntityCfg("ball"),
    },
  )
  env_cfg.events["launch_delayed_ball"] = EventTermCfg(
    func=launch_staged_ball_after_delay,
    mode="step",
    params={"wait_s": wait_s, "ball_cfg": SceneEntityCfg("ball")},
  )


def _apply_bc_regularized_finetune_profile(env_cfg) -> None:
  """Use save/block rewards with minimal prepare shaping for BC-initialized RL."""
  env_cfg.rewards.pop("posture", None)
  if "idle_fall_penalty" in env_cfg.rewards:
    env_cfg.rewards["idle_fall_penalty"].weight = -1000.0
  if "action_rate" in env_cfg.rewards:
    env_cfg.rewards["action_rate"].weight = -0.1
  if "idle_low_base_height" in env_cfg.rewards:
    env_cfg.rewards["idle_low_base_height"].weight = -20.0
  if "idle_base_height_band" in env_cfg.rewards:
    env_cfg.rewards["idle_base_height_band"].weight = 5.0
  if "idle_leg_ready_pose" in env_cfg.rewards:
    env_cfg.rewards["idle_leg_ready_pose"].weight = -5.0
  from src.tasks.soccer.mdp.goalkeeper_rewards import (
    goalkeeper_active_fall_penalty,
    goalkeeper_active_upright,
    goalkeeper_idle_base_still,
    goalkeeper_idle_upright,
  )

  env_cfg.rewards["idle_upright"] = RewardTermCfg(
    func=goalkeeper_idle_upright,
    weight=2.0,
  )
  env_cfg.rewards["idle_base_still"] = RewardTermCfg(
    func=goalkeeper_idle_base_still,
    weight=-0.5,
  )
  env_cfg.rewards["active_upright"] = RewardTermCfg(
    func=goalkeeper_active_upright,
    weight=0.5,
  )
  env_cfg.rewards["active_fall_penalty"] = RewardTermCfg(
    func=goalkeeper_active_fall_penalty,
    weight=-100.0,
  )


def _apply_student_polish_defaults(cfg: FineTuneConfig) -> None:
  """Conservative train_polish-style local refinement defaults."""
  if cfg.init_action_std is None:
    cfg.init_action_std = 0.06
  if cfg.min_action_std is None:
    cfg.min_action_std = 0.01
  if cfg.max_action_std is None:
    cfg.max_action_std = 0.06
  if cfg.learning_rate is None:
    cfg.learning_rate = 1.0e-4
  if cfg.entropy_coef is None:
    cfg.entropy_coef = 0.0
  if cfg.critic_warmup_iterations is None:
    cfg.critic_warmup_iterations = 50
  if cfg.eval_interval <= 0:
    cfg.eval_interval = 20
  if cfg.rollback_drop <= 0.0:
    cfg.rollback_drop = 0.02
  cfg.num_learning_epochs = 5
  cfg.offline_bc_coef = cfg.offline_bc_coef or 0.5
  cfg.offline_idle_bc_coef = cfg.offline_idle_bc_coef or 0.5
  cfg.offline_bc_every_n_updates = 1
  cfg.offline_bc_batch_size = max(cfg.offline_bc_batch_size, 64)
  cfg.offline_bc_cache_shards = max(cfg.offline_bc_cache_shards, 32)
  cfg.distill_loss_type = "huber"
  cfg.normalize_advantage_per_mini_batch = False
  cfg.mask_idle_actor_loss = True


def build_train_config(cfg: FineTuneConfig) -> TrainConfig:
  if cfg.profile == "student_polish":
    _apply_student_polish_defaults(cfg)
  env_cfg = load_env_cfg(cfg.task_id)
  agent_cfg = load_rl_cfg(cfg.task_id)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  agent_cfg.seed = cfg.seed
  agent_cfg.max_iterations = cfg.max_iterations
  agent_cfg.run_name = cfg.run_name
  agent_cfg.save_interval = cfg.save_interval
  agent_cfg.algorithm.teacher_checkpoint_path = cfg.teacher
  agent_cfg.algorithm.distill_coef = cfg.distill_coef
  agent_cfg.algorithm.distill_final_coef = cfg.distill_final_coef
  agent_cfg.algorithm.distill_anneal_updates = cfg.distill_anneal_updates
  agent_cfg.algorithm.teacher_every_n_steps = cfg.teacher_every_n_steps
  if cfg.entropy_coef is not None:
    agent_cfg.algorithm.entropy_coef = cfg.entropy_coef
  if cfg.learning_rate is not None:
    agent_cfg.algorithm.learning_rate = cfg.learning_rate
  if cfg.min_action_std is not None:
    agent_cfg.algorithm.min_action_std = cfg.min_action_std
  if cfg.max_action_std is not None:
    agent_cfg.algorithm.max_action_std = cfg.max_action_std
  if cfg.actor_mean_clip is not None:
    agent_cfg.algorithm.actor_mean_clip = cfg.actor_mean_clip
  agent_cfg.algorithm.ppo_log_ratio_clip = cfg.ppo_log_ratio_clip
  agent_cfg.algorithm.idle_deterministic_actions = cfg.idle_deterministic_actions
  agent_cfg.algorithm.mask_idle_actor_loss = cfg.mask_idle_actor_loss
  if cfg.critic_warmup_iterations is not None:
    agent_cfg.algorithm.critic_warmup_iterations = cfg.critic_warmup_iterations
  agent_cfg.algorithm.num_learning_epochs = cfg.num_learning_epochs
  agent_cfg.algorithm.normalize_advantage_per_mini_batch = cfg.normalize_advantage_per_mini_batch
  agent_cfg.algorithm.distill_loss_type = cfg.distill_loss_type
  agent_cfg.algorithm.offline_bc_dataset = cfg.offline_bc_dataset
  agent_cfg.algorithm.offline_bc_coef = cfg.offline_bc_coef
  agent_cfg.algorithm.offline_bc_final_coef = cfg.offline_bc_final_coef
  agent_cfg.algorithm.offline_bc_anneal_updates = cfg.offline_bc_anneal_updates
  agent_cfg.algorithm.offline_idle_bc_coef = cfg.offline_idle_bc_coef
  agent_cfg.algorithm.offline_idle_bc_final_coef = cfg.offline_idle_bc_final_coef
  agent_cfg.algorithm.offline_idle_bc_anneal_updates = cfg.offline_idle_bc_anneal_updates
  agent_cfg.algorithm.offline_bc_active_fraction = cfg.offline_bc_active_fraction
  agent_cfg.algorithm.offline_bc_batch_size = cfg.offline_bc_batch_size
  agent_cfg.algorithm.offline_bc_seq_len = cfg.offline_bc_seq_len
  agent_cfg.algorithm.offline_bc_every_n_updates = cfg.offline_bc_every_n_updates
  agent_cfg.algorithm.offline_bc_cache_shards = cfg.offline_bc_cache_shards
  agent_cfg.algorithm.freeze_actor_obs_normalization = cfg.freeze_actor_obs_normalization
  agent_cfg.algorithm.condition_aux_coef = cfg.condition_aux_coef
  agent_cfg.algorithm.condition_aux_final_coef = cfg.condition_aux_final_coef
  agent_cfg.algorithm.condition_aux_anneal_updates = cfg.condition_aux_anneal_updates
  agent_cfg.algorithm.condition_aux_active_only = cfg.condition_aux_active_only
  if cfg.delayed_launch:
    _apply_delayed_launch_sampling(env_cfg, wait_s=cfg.launch_delay_s)
  if cfg.profile in ("bc_regularized_finetune", "student_polish"):
    _apply_bc_regularized_finetune_profile(env_cfg)
  if cfg.profile == "student_polish":
    agent_cfg.algorithm.clip_param = 0.1
    agent_cfg.algorithm.desired_kl = 0.005

  return TrainConfig(
    env=env_cfg,
    agent=agent_cfg,
    video=cfg.video,
    load_actor_only=cfg.init is not None and not cfg.resume_full,
    load_checkpoint_path=cfg.init,
    actor_std_override=cfg.init_action_std,
    init_at_random_ep_len=False,
    gpu_ids=cfg.device_ids if cfg.device_ids is not None else [0],
  )


def _ball_entered_goal(ball_pos: torch.Tensor, origins: torch.Tensor) -> torch.Tensor:
  rel = ball_pos - origins
  return (rel[:, 0] <= -0.5) & (rel[:, 1].abs() <= 1.5) & (ball_pos[:, 2] <= 1.8)


def _eval_block_rate(env, policy, *, steps: int, resets: int) -> float:
  ball = env.unwrapped.scene["ball"]
  origins = env.unwrapped.scene.env_origins
  blocked = 0
  total = 0
  was_training = bool(getattr(policy, "training", False))
  policy.eval()
  with torch.inference_mode():
    for _ in range(max(1, resets)):
      obs = env.reset()
      if isinstance(obs, tuple):
        obs = obs[0]
      reset = getattr(policy, "reset", None)
      if callable(reset):
        reset()
      entered = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
      for _ in range(steps):
        action = policy(obs)
        result = env.step(action)
        obs = result[0]
        entered |= _ball_entered_goal(ball.data.root_link_pos_w, origins)
      blocked += int((~entered).sum().item())
      total += env.unwrapped.num_envs
  if was_training:
    policy.train()
  return blocked / max(total, 1)


def _save_runner_checkpoint(runner, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  saved = runner.alg.save()
  saved["iter"] = 0
  saved.setdefault("infos", {"env_state": {"common_step_counter": 0}})
  torch.save(saved, path)


def _load_initial_checkpoint(runner, train_cfg: TrainConfig) -> None:
  if train_cfg.load_checkpoint_path is None:
    return
  resume_path = Path(train_cfg.load_checkpoint_path).expanduser().resolve()
  if not resume_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
  print(f"[INFO]: Loading model checkpoint from: {resume_path}")
  if train_cfg.load_actor_only:
    runner.load(
      str(resume_path),
      load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
    )
  elif train_cfg.load_model_only:
    runner.load(
      str(resume_path),
      load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False},
    )
  else:
    runner.load(str(resume_path))
  if train_cfg.actor_std_override is not None:
    _override_actor_std(runner, train_cfg.actor_std_override)
  clamp = getattr(runner.alg, "_clamp_actor_std", None)
  if callable(clamp):
    clamp()


def _run_polish_training(task_id: str, cfg: FineTuneConfig, train_cfg: TrainConfig) -> None:
  configure_torch_backends()
  gpu_id = (cfg.device_ids or [0])[0]
  os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
  os.environ["MUJOCO_GL"] = "egl"
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  train_cfg.agent.seed = cfg.seed
  train_cfg.env.seed = cfg.seed
  log_root = Path("logs") / "rsl_rl" / train_cfg.agent.experiment_name
  log_dir = log_root / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{train_cfg.agent.run_name}")
  print(f"[INFO] Training with: device={device}, seed={cfg.seed}, rank=0")
  print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(cfg=train_cfg.env, device=device, render_mode="rgb_array" if cfg.video else None)
  env = RslRlVecEnvWrapper(env, clip_actions=train_cfg.agent.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(train_cfg.agent), str(log_dir), device)
  runner.add_git_repo_to_log(__file__)
  _load_initial_checkpoint(runner, train_cfg)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(train_cfg.env))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(train_cfg.agent))

  warmup = max(0, int(getattr(runner.alg, "critic_warmup_iterations", 0)))
  if warmup > 0:
    previous_trainable = runner.alg.set_actor_trainable(False)
    offline_bc_coef = runner.alg.offline_bc_coef
    offline_idle_bc_coef = runner.alg.offline_idle_bc_coef
    distill_coef = runner.alg.distill_coef
    condition_aux_coef = runner.alg.condition_aux_coef
    runner.alg.offline_bc_coef = 0.0
    runner.alg.offline_idle_bc_coef = 0.0
    runner.alg.distill_coef = 0.0
    runner.alg.condition_aux_coef = 0.0
    print(f"[INFO] Critic warmup: freezing actor for {warmup} iterations")
    runner.learn(num_learning_iterations=warmup, init_at_random_ep_len=False)
    runner.alg.restore_actor_trainable(previous_trainable)
    runner.alg.offline_bc_coef = offline_bc_coef
    runner.alg.offline_idle_bc_coef = offline_idle_bc_coef
    runner.alg.distill_coef = distill_coef
    runner.alg.condition_aux_coef = condition_aux_coef
    runner.alg._clamp_actor_std()

  policy = runner.get_inference_policy(device=device)
  best_rate = _eval_block_rate(env, policy, steps=cfg.eval_steps, resets=cfg.eval_resets)
  best_state = copy.deepcopy(runner.alg.actor.state_dict())
  best_path = log_dir / "model_best.pt"
  _save_runner_checkpoint(runner, best_path)
  print(f"[EVAL] init block {100.0 * best_rate:.1f}% (saved {best_path})")

  remaining = train_cfg.agent.max_iterations
  interval = max(1, cfg.eval_interval)
  total_blocks = (remaining + interval - 1) // interval
  for block in range(total_blocks):
    iters = min(interval, remaining)
    if iters <= 0:
      break
    runner.learn(num_learning_iterations=iters, init_at_random_ep_len=False)
    runner.alg._clamp_actor_std()
    rate = _eval_block_rate(env, policy, steps=cfg.eval_steps, resets=cfg.eval_resets)
    tag = ""
    if rate >= best_rate:
      best_rate = rate
      best_state = copy.deepcopy(runner.alg.actor.state_dict())
      _save_runner_checkpoint(runner, best_path)
      tag = " *best*"
    elif cfg.rollback_drop > 0.0 and rate < best_rate - cfg.rollback_drop:
      runner.alg.actor.load_state_dict(best_state)
      runner.alg._clamp_actor_std()
      tag = " rollback"
    print(f"[EVAL] block {block + 1}/{total_blocks}: {100.0 * rate:.1f}% (best {100.0 * best_rate:.1f}%){tag}")
    remaining -= iters

  runner.alg.actor.load_state_dict(best_state)
  runner.alg._clamp_actor_std()
  _save_runner_checkpoint(runner, best_path)
  print(f"[INFO] saved polish best {100.0 * best_rate:.1f}% to {best_path}")
  env.close()


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  cfg = tyro.cli(FineTuneConfig, prog="train_goalkeeper_student_ppo")
  train_cfg = build_train_config(cfg)
  if cfg.profile == "student_polish" or cfg.eval_interval > 0:
    _run_polish_training(cfg.task_id, cfg, train_cfg)
  else:
    launch_training(cfg.task_id, train_cfg)


if __name__ == "__main__":
  main()
