"""Train the position-conditioned goalkeeper student with PPO.

This is a non-adversarial training entrypoint.  The actor is the FiLM LSTM
student used by ``train_goalkeeper_student_bc.py`` and remains loadable by
``eval_naive_goalkeeper.py``.  The critic is also recurrent.  Training uses
PPO plus optional KL anchors (reference KL against the loaded checkpoint,
and/or teacher KL against a frozen MoE teacher on active samples).
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

DEFAULT_TEACHER_KL = (
  "/data/Courses/[CS2810]EmbodiedAI/humanoid_soccer_proj/G1-mjlab-soccer/"
  "checkpoints/goalkeeper_moe6_hard3_default.pt"
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

  init_action_std: float | None = None
  """Override the loaded actor action std after checkpoint load."""

  min_action_std: float | None = 0.03
  """Lower bound for the learned action std; prevents KL blowups at tiny std."""

  max_action_std: float | None = None
  """Optional upper bound for the learned action std during conservative polish."""

  actor_mean_clip: float | None = None
  """Optional clamp on actor deterministic means for recovery finetuning."""

  ppo_log_ratio_clip: float = 4.0
  """Clamp PPO log-prob ratios before exp to avoid NaN recovery updates."""

  idle_deterministic_actions: bool | None = None
  """Whether to use deterministic actor means in prelaunch prepare states."""

  mask_idle_actor_loss: bool = False
  """Do not update the actor with PPO surrogate/entropy loss on prepare samples."""

  idle_actor_loss_weight: float = 0.0
  """Small PPO actor-loss weight for prepare samples when mask_idle_actor_loss is enabled."""

  entropy_coef: float | None = 0.003
  """Entropy coefficient; keep some exploration without overwhelming the BC prior."""

  learning_rate: float | None = None
  """Optional PPO learning rate override for conservative finetuning."""

  num_learning_epochs: int = 5
  """Number of PPO optimization epochs per rollout."""

  normalize_advantage_per_mini_batch: bool = False
  """Normalize PPO advantages inside each mini-batch for recovery finetuning."""

  profile: Literal["default", "bc_regularized_finetune", "student_polish", "teacher_kl"] = "default"
  """Reward/regularization preset for this finetune run."""

  freeze_actor_obs_normalization: bool | None = None
  """Whether to keep the actor observation normalizer fixed during RL finetuning."""

  condition_aux_coef: float = 0.05
  """Weight for auxiliary 7-class MoE route prediction from the student latent."""

  condition_aux_final_coef: float | None = None
  """Final auxiliary condition prediction weight after optional linear annealing."""

  condition_aux_anneal_updates: int = 0
  """Number of PPO rollout updates used to linearly anneal condition aux weight."""

  condition_aux_active_only: bool = True
  """Apply auxiliary route loss only to active/post-launch samples."""

  reference_kl_coef: float = 0.0
  """KL anchor weight against the frozen policy captured after --init is loaded."""

  reference_kl_idle_coef: float | None = None
  """Prepare/idle KL anchor weight; defaults to reference_kl_coef when unset."""

  reference_kl_active_coef: float | None = None
  """Active/post-launch KL anchor weight; defaults to reference_kl_coef when unset."""

  reference_kl_std_floor: float = 0.1
  """Std floor used only for the reference KL anchor to avoid tiny-std blowups."""

  teacher_kl_checkpoint: str | None = None
  """Frozen MoE teacher checkpoint used for active-phase KL regularization."""

  teacher_kl_coef: float | None = None
  """Weight on KL(student || teacher) computed only on active/post-launch samples."""

  teacher_kl_std_floor: float = 0.05
  """Std floor for teacher KL to avoid tiny-std blowups."""

  prior_disc_dataset_dir: str | None = None
  """Successful MoE rollout dataset used to train a GAIL-style prior discriminator."""

  prior_disc_reward_coef: float = 0.0
  """Reward weight for the discriminator prior; 0 disables the prior."""

  prior_disc_updates: int = 1
  """Number of discriminator update batches per PPO update."""

  prior_disc_batch_size: int = 256
  """Expert/policy batch size used for each discriminator update."""

  prior_disc_learning_rate: float = 3.0e-4
  """Learning rate for the prior discriminator optimizer."""

  prior_disc_reward_clip: float = 5.0
  """Maximum per-step discriminator prior reward before applying the coefficient."""

  active_bc_dataset_dir: str | None = None
  """Successful active MoE rollout dataset used for online BC regularization."""

  active_bc_coef: float = 0.0
  """Weight for active-only BC loss against successful MoE rollout actions."""

  active_bc_final_coef: float | None = None
  """Final active BC weight after optional linear annealing."""

  active_bc_anneal_updates: int = 0
  """Number of PPO updates used to linearly anneal active BC weight."""

  active_bc_batch_size: int = 256
  """Number of successful active frames sampled for each PPO minibatch BC loss."""

  motion_prior_weight: float = 0.0
  """Active-stage joint-pose motion-prior reward weight; 0 disables it."""

  motion_prior_dir: str = "src/assets/soccer/motions/goalkeeper"
  """Directory containing goalkeeper motion-prior .pt files and joint_id.txt."""

  motion_prior_names: tuple[str, ...] | None = None
  """Optional motion-prior file names to use, e.g. leftjump.pt rightjump.pt."""

  motion_prior_route_mode: Literal["all", "region"] = "all"
  """Motion-prior routing: all=min over selected motions, region=use _gk_region mapping."""

  motion_prior_std: float = 0.5
  """Gaussian std for active motion-prior joint-pose tracking reward."""

  motion_prior_launch_delay_s: float | None = None
  """Active-frame alignment offset; defaults to --launch-delay-s."""

  idle_action_rate_weight: float | None = None
  """Optional override for prepare-stage action-rate smoothing."""

  idle_joint_pose_weight: float | None = None
  """Optional override for prepare-stage default joint-pose regularization."""

  action_rate_weight: float | None = None
  """Optional override for the base action-rate penalty."""

  active_upright_weight: float | None = None
  """Optional override for active/post-launch upright shaping."""

  active_fall_penalty_weight: float | None = None
  """Optional override for active/post-launch fall penalty."""

  intercept_weight: float | None = None
  """Optional override for active interception reward."""

  stop_ball_weight: float | None = None
  """Optional override for save/outcome reward."""

  critic_warmup_iterations: int | None = 50
  """Freeze the actor and train only the critic for this many initial iterations."""

  eval_interval: int = 0
  """Polish mode: train iterations between deterministic eval/rollback checks."""

  eval_resets: int = 1
  """Polish mode: number of vectorized resets per rollback eval."""

  eval_steps: int = 300
  """Polish mode: deterministic eval horizon in env steps."""

  rollback_drop: float = 0.0
  """Polish mode: rollback if block rate drops this much below best; 0 disables."""

  disable_wandb_for_polish: bool = False
  """Run polish finetuning with local logs only when explicitly enabled."""

  resume_wandb: bool = False
  """Resume the wandb run associated with --init instead of creating a new one."""

  wandb_resume_id: str | None = None
  """Explicit wandb run id to resume; overrides --resume-wandb auto-discovery."""

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
  if "idle_action_rate" in env_cfg.rewards:
    env_cfg.rewards["idle_action_rate"].weight = -0.05
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
    weight=-0.1,
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
    cfg.init_action_std = 0.1
  if cfg.min_action_std is None:
    cfg.min_action_std = 0.05
  if cfg.max_action_std is None:
    cfg.max_action_std = 0.2
  if cfg.learning_rate is None:
    cfg.learning_rate = 1.0e-4
  if cfg.entropy_coef is None:
    cfg.entropy_coef = 0.005
  if cfg.critic_warmup_iterations is None:
    cfg.critic_warmup_iterations = 50
  cfg.num_learning_epochs = 5
  if cfg.actor_mean_clip is None:
    cfg.actor_mean_clip = 6.0
  cfg.ppo_log_ratio_clip = min(cfg.ppo_log_ratio_clip, 4.0)
  cfg.condition_aux_coef = 0.0
  cfg.condition_aux_final_coef = 0.0
  cfg.reference_kl_coef = cfg.reference_kl_coef or 0.1
  if cfg.reference_kl_idle_coef is None:
    cfg.reference_kl_idle_coef = 0.1
  if cfg.reference_kl_active_coef is None:
    cfg.reference_kl_active_coef = 0.001
  cfg.reference_kl_std_floor = 0.05
  cfg.normalize_advantage_per_mini_batch = False
  cfg.mask_idle_actor_loss = True
  cfg.idle_actor_loss_weight = cfg.idle_actor_loss_weight or 0.2
  cfg.idle_deterministic_actions = True


def _apply_teacher_kl_defaults(cfg: FineTuneConfig) -> None:
  """Prepare-from-scratch defaults: pure reward/PPO for prepare, MoE6 KL for active."""
  if cfg.teacher_kl_checkpoint is None:
    cfg.teacher_kl_checkpoint = DEFAULT_TEACHER_KL
  # Respect an explicit 0.0 override so we can cleanly disable teacher KL.
  if cfg.teacher_kl_coef is None:
    cfg.teacher_kl_coef = 0.003
  if cfg.init_action_std is None:
    cfg.init_action_std = 0.12
  if cfg.max_action_std is None:
    cfg.max_action_std = 0.15
  if cfg.learning_rate is None:
    cfg.learning_rate = 3.0e-4
  if cfg.entropy_coef is None:
    cfg.entropy_coef = 0.003
  cfg.critic_warmup_iterations = 0
  cfg.condition_aux_coef = 0.01
  cfg.condition_aux_final_coef = 0.01
  cfg.mask_idle_actor_loss = False
  if cfg.idle_deterministic_actions is None:
    cfg.idle_deterministic_actions = False


def _apply_teacher_kl_profile(env_cfg) -> None:
  """Prepare-stage standing rewards/penalties + active interception rewards.

  Prepare samples are shaped purely by idle standing terms (upright, base height
  band, leg ready pose, joint default-pose penalty, fall penalty, action rate).
  Active samples use the goalkeeper interception rewards. No KL is computed on
  prepare samples; active samples are additionally anchored to the MoE6 teacher
  via the PPO-level teacher_kl loss.
  """
  from src.tasks.soccer.mdp.goalkeeper_rewards import (
    goalkeeper_active_fall_penalty,
    goalkeeper_active_upright,
    goalkeeper_idle_action_rate,
    goalkeeper_idle_alive,
    goalkeeper_idle_base_height_band,
    goalkeeper_idle_base_still,
    goalkeeper_idle_fall_penalty,
    goalkeeper_idle_joint_pose,
    goalkeeper_idle_leg_ready_pose,
    goalkeeper_idle_low_base_height,
    goalkeeper_idle_upright,
  )

  env_cfg.rewards.pop("posture", None)
  env_cfg.rewards["idle_fall_penalty"] = RewardTermCfg(
    func=goalkeeper_idle_fall_penalty,
    weight=-2000.0,
  )
  env_cfg.rewards["idle_upright"] = RewardTermCfg(
    func=goalkeeper_idle_upright,
    weight=2.0,
  )
  env_cfg.rewards["idle_base_still"] = RewardTermCfg(
    func=goalkeeper_idle_base_still,
    weight=-0.05,
  )
  env_cfg.rewards["idle_low_base_height"] = RewardTermCfg(
    func=goalkeeper_idle_low_base_height,
    weight=-20.0,
    params={"target_z": 0.72},
  )
  env_cfg.rewards["idle_base_height_band"] = RewardTermCfg(
    func=goalkeeper_idle_base_height_band,
    weight=5.0,
    params={"target_z": 0.72, "tolerance": 0.05, "low_margin": 0.08},
  )
  env_cfg.rewards["idle_leg_ready_pose"] = RewardTermCfg(
    func=goalkeeper_idle_leg_ready_pose,
    weight=0.0,
  )
  env_cfg.rewards["idle_joint_pose"] = RewardTermCfg(
    func=goalkeeper_idle_joint_pose,
    weight=0.0,
  )
  env_cfg.rewards["idle_alive"] = RewardTermCfg(
    func=goalkeeper_idle_alive,
    weight=0.1,
  )
  env_cfg.rewards["idle_action_rate"] = RewardTermCfg(
    func=goalkeeper_idle_action_rate,
    weight=-0.005,
  )
  env_cfg.rewards["active_upright"] = RewardTermCfg(
    func=goalkeeper_active_upright,
    weight=2.0,
  )
  env_cfg.rewards["active_fall_penalty"] = RewardTermCfg(
    func=goalkeeper_active_fall_penalty,
    weight=-100.0,
  )
  if "goal_conceded" in env_cfg.rewards:
    env_cfg.rewards["goal_conceded"].weight = -20.0
  if "intercept" in env_cfg.rewards:
    env_cfg.rewards["intercept"].weight = 1.0
  if "condition_target_reach" in env_cfg.rewards:
    env_cfg.rewards["condition_target_reach"].weight = 0.5
  if "body" in env_cfg.rewards:
    env_cfg.rewards["body"].weight = 1.0
  if "stop_ball" in env_cfg.rewards:
    env_cfg.rewards["stop_ball"].weight = 3.0


def build_train_config(cfg: FineTuneConfig) -> TrainConfig:
  if cfg.profile == "student_polish":
    _apply_student_polish_defaults(cfg)
  if cfg.profile == "teacher_kl":
    _apply_teacher_kl_defaults(cfg)
  env_cfg = load_env_cfg(cfg.task_id)
  agent_cfg = load_rl_cfg(cfg.task_id)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  agent_cfg.seed = cfg.seed
  agent_cfg.max_iterations = cfg.max_iterations
  agent_cfg.run_name = cfg.run_name
  agent_cfg.save_interval = cfg.save_interval
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
  idle_deterministic_actions = cfg.idle_deterministic_actions
  if idle_deterministic_actions is None:
    idle_deterministic_actions = False
  agent_cfg.algorithm.idle_deterministic_actions = idle_deterministic_actions
  agent_cfg.algorithm.mask_idle_actor_loss = cfg.mask_idle_actor_loss
  agent_cfg.algorithm.idle_actor_loss_weight = cfg.idle_actor_loss_weight
  if cfg.critic_warmup_iterations is not None:
    agent_cfg.algorithm.critic_warmup_iterations = cfg.critic_warmup_iterations
  agent_cfg.algorithm.num_learning_epochs = cfg.num_learning_epochs
  agent_cfg.algorithm.normalize_advantage_per_mini_batch = cfg.normalize_advantage_per_mini_batch
  freeze_actor_obs_normalization = cfg.freeze_actor_obs_normalization
  if freeze_actor_obs_normalization is None:
    freeze_actor_obs_normalization = cfg.init is not None
  agent_cfg.algorithm.freeze_actor_obs_normalization = freeze_actor_obs_normalization
  agent_cfg.algorithm.condition_aux_coef = cfg.condition_aux_coef
  agent_cfg.algorithm.condition_aux_final_coef = cfg.condition_aux_final_coef
  agent_cfg.algorithm.condition_aux_anneal_updates = cfg.condition_aux_anneal_updates
  agent_cfg.algorithm.condition_aux_active_only = cfg.condition_aux_active_only
  agent_cfg.algorithm.reference_kl_coef = cfg.reference_kl_coef
  agent_cfg.algorithm.reference_kl_idle_coef = cfg.reference_kl_idle_coef
  agent_cfg.algorithm.reference_kl_active_coef = cfg.reference_kl_active_coef
  agent_cfg.algorithm.reference_kl_std_floor = cfg.reference_kl_std_floor
  agent_cfg.algorithm.teacher_kl_checkpoint = cfg.teacher_kl_checkpoint
  agent_cfg.algorithm.teacher_kl_coef = cfg.teacher_kl_coef
  agent_cfg.algorithm.teacher_kl_std_floor = cfg.teacher_kl_std_floor
  agent_cfg.algorithm.prior_disc_dataset_dir = cfg.prior_disc_dataset_dir
  agent_cfg.algorithm.prior_disc_reward_coef = cfg.prior_disc_reward_coef
  agent_cfg.algorithm.prior_disc_updates = cfg.prior_disc_updates
  agent_cfg.algorithm.prior_disc_batch_size = cfg.prior_disc_batch_size
  agent_cfg.algorithm.prior_disc_learning_rate = cfg.prior_disc_learning_rate
  agent_cfg.algorithm.prior_disc_reward_clip = cfg.prior_disc_reward_clip
  agent_cfg.algorithm.active_bc_dataset_dir = cfg.active_bc_dataset_dir
  agent_cfg.algorithm.active_bc_coef = cfg.active_bc_coef
  agent_cfg.algorithm.active_bc_final_coef = cfg.active_bc_final_coef
  agent_cfg.algorithm.active_bc_anneal_updates = cfg.active_bc_anneal_updates
  agent_cfg.algorithm.active_bc_batch_size = cfg.active_bc_batch_size
  if cfg.delayed_launch:
    _apply_delayed_launch_sampling(env_cfg, wait_s=cfg.launch_delay_s)
  if cfg.profile in ("bc_regularized_finetune", "student_polish"):
    _apply_bc_regularized_finetune_profile(env_cfg)
  if cfg.profile == "teacher_kl":
    _apply_teacher_kl_profile(env_cfg)
  if cfg.profile == "student_polish":
    if "idle_action_rate" in env_cfg.rewards:
      env_cfg.rewards["idle_action_rate"].weight = -0.2
    agent_cfg.algorithm.clip_param = 0.1
    agent_cfg.algorithm.desired_kl = 0.01
  if cfg.motion_prior_weight != 0.0:
    from src.tasks.soccer.mdp.goalkeeper_rewards import goalkeeper_active_motion_prior_joint_pose

    env_cfg.rewards["motion_prior_joint_pose"] = RewardTermCfg(
      func=goalkeeper_active_motion_prior_joint_pose,
      weight=cfg.motion_prior_weight,
      params={
        "motion_dir": cfg.motion_prior_dir,
        "motion_names": cfg.motion_prior_names,
        "route_mode": cfg.motion_prior_route_mode,
        "std": cfg.motion_prior_std,
        "launch_delay_s": (
          cfg.launch_delay_s
          if cfg.motion_prior_launch_delay_s is None
          else cfg.motion_prior_launch_delay_s
        ),
      },
    )
  if cfg.idle_action_rate_weight is not None and "idle_action_rate" in env_cfg.rewards:
    env_cfg.rewards["idle_action_rate"].weight = cfg.idle_action_rate_weight
  if cfg.idle_joint_pose_weight is not None and "idle_joint_pose" in env_cfg.rewards:
    env_cfg.rewards["idle_joint_pose"].weight = cfg.idle_joint_pose_weight
  if cfg.action_rate_weight is not None and "action_rate" in env_cfg.rewards:
    env_cfg.rewards["action_rate"].weight = cfg.action_rate_weight
  if cfg.active_upright_weight is not None and "active_upright" in env_cfg.rewards:
    env_cfg.rewards["active_upright"].weight = cfg.active_upright_weight
  if cfg.active_fall_penalty_weight is not None and "active_fall_penalty" in env_cfg.rewards:
    env_cfg.rewards["active_fall_penalty"].weight = cfg.active_fall_penalty_weight
  if cfg.intercept_weight is not None and "intercept" in env_cfg.rewards:
    env_cfg.rewards["intercept"].weight = cfg.intercept_weight
  if cfg.stop_ball_weight is not None and "stop_ball" in env_cfg.rewards:
    env_cfg.rewards["stop_ball"].weight = cfg.stop_ball_weight
  if cfg.init_action_std is not None and cfg.init is None:
    agent_cfg.actor.distribution_cfg["init_std"] = cfg.init_action_std

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


def _eval_goalkeeper_diagnostics(
  env,
  policy,
  *,
  steps: int,
  resets: int,
  fall_limit_angle: float = 1.2217304763960306,
) -> dict[str, float]:
  ball = env.unwrapped.scene["ball"]
  robot = env.unwrapped.scene["robot"]
  origins = env.unwrapped.scene.env_origins
  device = env.unwrapped.device
  limit_cos = torch.cos(torch.tensor(fall_limit_angle, dtype=torch.float32, device=device))

  blocked = 0
  total = 0
  prelaunch_falls = 0
  active_falls = 0
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
      num_envs = env.unwrapped.num_envs
      entered = torch.zeros(num_envs, dtype=torch.bool, device=device)
      prelaunch_fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
      active_fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
      for _ in range(steps):
        action = policy(obs)
        result = env.step(action)
        obs = result[0]
        entered |= _ball_entered_goal(ball.data.root_link_pos_w, origins)

        upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0)
        fallen = upright < limit_cos
        launched = getattr(env.unwrapped, "_gk_delayed_ball_launched", None)
        if launched is None or launched.shape[0] != num_envs:
          idle_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
          active_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        else:
          launched = launched.to(device=device, dtype=torch.bool)
          idle_mask = ~launched
          active_mask = launched
        prelaunch_fell |= fallen & idle_mask
        active_fell |= fallen & active_mask
      blocked += int((~entered).sum().item())
      prelaunch_falls += int(prelaunch_fell.sum().item())
      active_falls += int(active_fell.sum().item())
      total += num_envs
  if was_training:
    policy.train()
  total = max(total, 1)
  return {
    "block_rate": blocked / total,
    "prelaunch_fall_rate": prelaunch_falls / total,
    "active_fall_rate": active_falls / total,
  }


def _resolve_wandb_resume(cfg: FineTuneConfig) -> dict[str, str]:
  if cfg.wandb_resume_id:
    return {
      "id": cfg.wandb_resume_id,
      "resume": "must",
      "source": "explicit",
      "dir": "",
    }
  if not cfg.resume_wandb:
    return {}
  if not cfg.init:
    raise ValueError("--resume-wandb requires --init or use --wandb-resume-id")
  init_path = Path(cfg.init).expanduser().resolve()
  run_dir = init_path.parent
  wandb_meta_path = run_dir / "params" / "wandb.yaml"
  if not wandb_meta_path.exists():
    raise FileNotFoundError(
      f"Wandb metadata not found for init checkpoint: {wandb_meta_path}. "
      "Use --wandb-resume-id to resume explicitly."
    )
  wandb_id = ""
  resume_mode = "must"
  for line in wandb_meta_path.read_text().splitlines():
    key, _, value = line.partition(":")
    key = key.strip()
    value = value.strip()
    if key == "id":
      wandb_id = value
    elif key == "resume" and value:
      resume_mode = value
  if not wandb_id:
    raise ValueError(
      f"Wandb metadata at {wandb_meta_path} does not contain an id. "
      "Use --wandb-resume-id to resume explicitly."
    )
  return {
    "id": wandb_id,
    "resume": resume_mode,
    "source": "init",
    "dir": str(run_dir),
  }


def _apply_wandb_resume_env(cfg: FineTuneConfig, log_dir: Path) -> dict[str, str]:
  meta = _resolve_wandb_resume(cfg)
  if not meta:
    return {}
  os.environ["WANDB_RESUME"] = meta["resume"]
  os.environ["WANDB_RUN_ID"] = meta["id"]
  print(
    f"[INFO] Resuming wandb run {meta['id']} "
    f"from {meta.get('source', 'unknown')}"
  )
  dump_yaml(log_dir / "params" / "wandb.yaml", {"id": meta["id"], "resume": meta["resume"]})
  return meta


def _record_wandb_run_metadata(log_dir: Path) -> None:
  run_id = os.environ.get("WANDB_RUN_ID", "").strip()
  if not run_id:
    return
  resume_mode = os.environ.get("WANDB_RESUME", "").strip() or "allow"
  dump_yaml(log_dir / "params" / "wandb.yaml", {"id": run_id, "resume": resume_mode})


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
  set_reference = getattr(runner.alg, "set_reference_actor_from_current", None)
  if callable(set_reference) and getattr(runner.alg, "reference_kl_coef", 0.0) > 0.0:
    set_reference()
    print(f"[INFO] Reference KL anchor frozen from: {resume_path}")


def _run_teacher_kl_training(task_id: str, cfg: FineTuneConfig, train_cfg: TrainConfig) -> None:
  """Teacher-KL training that loads an init checkpoint and freezes a reference actor."""
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
  _apply_wandb_resume_env(cfg, log_dir)

  env = ManagerBasedRlEnv(cfg=train_cfg.env, device=device, render_mode="rgb_array" if cfg.video else None)
  env = RslRlVecEnvWrapper(env, clip_actions=train_cfg.agent.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(train_cfg.agent), str(log_dir), device)
  runner.add_git_repo_to_log(__file__)
  _load_initial_checkpoint(runner, train_cfg)
  _record_wandb_run_metadata(log_dir)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(train_cfg.env))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(train_cfg.agent))

  remaining = train_cfg.agent.max_iterations
  interval = max(1, cfg.eval_interval)
  if cfg.eval_interval <= 0:
    runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
    runner.alg._clamp_actor_std()
    env.close()
    return

  policy = runner.get_inference_policy(device=device)
  best_metrics = _eval_goalkeeper_diagnostics(env, policy, steps=cfg.eval_steps, resets=cfg.eval_resets)
  best_rate = best_metrics["block_rate"]
  best_state = copy.deepcopy(runner.alg.actor.state_dict())
  best_path = log_dir / "model_best.pt"
  _save_runner_checkpoint(runner, best_path)
  print(
    "[EVAL] init "
    f"block {100.0 * best_metrics['block_rate']:.1f}% "
    f"prelaunch_fall {100.0 * best_metrics['prelaunch_fall_rate']:.1f}% "
    f"active_fall {100.0 * best_metrics['active_fall_rate']:.1f}% "
    f"(saved {best_path})"
  )

  total_blocks = (remaining + interval - 1) // interval
  for block in range(total_blocks):
    iters = min(interval, remaining)
    if iters <= 0:
      break
    runner.learn(num_learning_iterations=iters, init_at_random_ep_len=False)
    runner.alg._clamp_actor_std()
    metrics = _eval_goalkeeper_diagnostics(env, policy, steps=cfg.eval_steps, resets=cfg.eval_resets)
    rate = metrics["block_rate"]
    tag = ""
    if rate >= best_rate:
      best_rate = rate
      best_state = copy.deepcopy(runner.alg.actor.state_dict())
      best_metrics = metrics
      _save_runner_checkpoint(runner, best_path)
      tag = " *best*"
    print(
      f"[EVAL] block {block + 1}/{total_blocks}: "
      f"block {100.0 * rate:.1f}% "
      f"prelaunch_fall {100.0 * metrics['prelaunch_fall_rate']:.1f}% "
      f"active_fall {100.0 * metrics['active_fall_rate']:.1f}% "
      f"(best {100.0 * best_rate:.1f}%){tag}"
    )
    remaining -= iters

  runner.alg.actor.load_state_dict(best_state)
  runner.alg._clamp_actor_std()
  _save_runner_checkpoint(runner, best_path)
  print(
    "[INFO] saved teacher_kl best "
    f"block {100.0 * best_metrics['block_rate']:.1f}% "
    f"prelaunch_fall {100.0 * best_metrics['prelaunch_fall_rate']:.1f}% "
    f"active_fall {100.0 * best_metrics['active_fall_rate']:.1f}% "
    f"to {best_path}"
  )
  env.close()


def _run_polish_training(task_id: str, cfg: FineTuneConfig, train_cfg: TrainConfig) -> None:
  configure_torch_backends()
  if cfg.disable_wandb_for_polish:
    os.environ.setdefault("WANDB_MODE", "disabled")
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
  _apply_wandb_resume_env(cfg, log_dir)

  env = ManagerBasedRlEnv(cfg=train_cfg.env, device=device, render_mode="rgb_array" if cfg.video else None)
  env = RslRlVecEnvWrapper(env, clip_actions=train_cfg.agent.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(train_cfg.agent), str(log_dir), device)
  runner.add_git_repo_to_log(__file__)
  _load_initial_checkpoint(runner, train_cfg)
  _record_wandb_run_metadata(log_dir)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(train_cfg.env))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(train_cfg.agent))

  warmup = max(0, int(getattr(runner.alg, "critic_warmup_iterations", 0)))
  if warmup > 0:
    print(f"[INFO] Critic warmup: actor frozen inside PPO updates for {warmup} iterations")

  remaining = train_cfg.agent.max_iterations
  interval = max(1, cfg.eval_interval)
  if cfg.eval_interval <= 0:
    runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
    runner.alg._clamp_actor_std()
    env.close()
    return

  policy = runner.get_inference_policy(device=device)
  best_rate = _eval_block_rate(env, policy, steps=cfg.eval_steps, resets=cfg.eval_resets)
  best_state = copy.deepcopy(runner.alg.actor.state_dict())
  best_path = log_dir / "model_best.pt"
  _save_runner_checkpoint(runner, best_path)
  print(f"[EVAL] init block {100.0 * best_rate:.1f}% (saved {best_path})")

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
  elif cfg.profile == "teacher_kl" and cfg.init is not None:
    _run_teacher_kl_training(cfg.task_id, cfg, train_cfg)
  else:
    launch_training(cfg.task_id, train_cfg)


if __name__ == "__main__":
  main()
