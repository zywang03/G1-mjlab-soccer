"""Hybrid PPO + online teacher distillation for the goalkeeper student."""

from __future__ import annotations

from collections.abc import Generator
import copy
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import (
  resolve_callable,
  resolve_obs_groups,
  split_and_pad_trajectories,
  unpad_trajectories,
)
from src.tasks.soccer.modules.goalkeeper_prior_discriminator import (
  GoalkeeperPriorDataset,
  GoalkeeperPriorDiscriminator,
  discriminator_loss,
)
from src.tasks.soccer.mdp.goalkeeper_student_obs import (
  GOALKEEPER_PHASE_IDLE_INDEX,
  GOALKEEPER_STUDENT_OBS_DIM,
  GOALKEEPER_TEACHER_ACTOR_OBS_DIM,
  build_goalkeeper_student_obs,
)


@dataclass
class GoalkeeperStudentPpoAlgorithmCfg:
  """PPO config extended with optional online teacher distillation knobs."""

  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 3.0e-4
  schedule: str = "adaptive"
  gamma: float = 0.99
  lam: float = 0.95
  entropy_coef: float = 0.001
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  clip_param: float = 0.2
  normalize_advantage_per_mini_batch: bool = False
  optimizer: str = "adam"
  share_cnn_encoders: bool = False
  rnd_cfg: dict | None = None
  symmetry_cfg: dict | None = None
  class_name: str = "src.tasks.soccer.modules.goalkeeper_student_ppo:GoalkeeperStudentPPO"
  condition_aux_coef: float = 0.05
  condition_aux_final_coef: float | None = None
  condition_aux_anneal_updates: int = 0
  condition_aux_active_only: bool = True
  reference_kl_coef: float = 0.0
  reference_kl_idle_coef: float | None = None
  reference_kl_active_coef: float | None = None
  reference_kl_std_floor: float = 0.1
  teacher_kl_checkpoint: str | None = None
  """Frozen MoE teacher checkpoint used for active-phase KL regularization."""
  teacher_kl_coef: float = 0.0
  """Weight on KL(student || teacher) computed only on active/post-launch samples."""
  teacher_kl_std_floor: float = 0.05
  """Std floor for teacher KL to avoid tiny-std blowups."""
  prior_disc_dataset_dir: str | None = None
  prior_disc_reward_coef: float = 0.0
  prior_disc_updates: int = 1
  prior_disc_batch_size: int = 256
  prior_disc_learning_rate: float = 3.0e-4
  prior_disc_reward_clip: float = 5.0
  active_bc_dataset_dir: str | None = None
  active_bc_coef: float = 0.0
  active_bc_final_coef: float | None = None
  active_bc_anneal_updates: int = 0
  active_bc_batch_size: int = 256
  min_action_std: float = 1.0e-4
  max_action_std: float | None = None
  actor_mean_clip: float | None = None
  ppo_log_ratio_clip: float = 8.0
  idle_deterministic_actions: bool = True
  mask_idle_actor_loss: bool = False
  idle_actor_loss_weight: float = 0.0
  critic_warmup_iterations: int = 0
  freeze_actor_obs_normalization: bool = False


class GoalkeeperStudentRolloutStorage(RolloutStorage):
  """RL rollout storage with per-transition actor-loss masks."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.actor_loss_masks = torch.ones(self.num_transitions_per_env, self.num_envs, 1, device=self.device)

  def add_transition(self, transition: RolloutStorage.Transition) -> None:
    actor_loss_mask = getattr(transition, "actor_loss_mask", None)
    if actor_loss_mask is None:
      self.actor_loss_masks[self.step].fill_(1.0)
    else:
      self.actor_loss_masks[self.step].copy_(actor_loss_mask.view(-1, 1).to(device=self.device, dtype=torch.float32))
    super().add_transition(transition)

  def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[RolloutStorage.Batch, None, None]:
    batch_size = self.num_envs * self.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

    observations = self.observations.flatten(0, 1)
    actions = self.actions.flatten(0, 1)
    actor_loss_masks = self.actor_loss_masks.flatten(0, 1)
    values = self.values.flatten(0, 1)
    returns = self.returns.flatten(0, 1)
    old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
    advantages = self.advantages.flatten(0, 1)
    old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore[arg-type]

    for _ in range(num_epochs):
      for i in range(num_mini_batches):
        start = i * mini_batch_size
        stop = (i + 1) * mini_batch_size
        batch_idx = indices[start:stop]
        batch = RolloutStorage.Batch(
          observations=observations[batch_idx],
          actions=actions[batch_idx],
          values=values[batch_idx],
          advantages=advantages[batch_idx],
          returns=returns[batch_idx],
          old_actions_log_prob=old_actions_log_prob[batch_idx],
          old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
        )
        batch.actor_loss_mask = actor_loss_masks[batch_idx]
        yield batch

  def recurrent_mini_batch_generator(
    self,
    num_mini_batches: int,
    num_epochs: int = 8,
  ) -> Generator[RolloutStorage.Batch, None, None]:
    padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
    mini_batch_size = self.num_envs // num_mini_batches

    for _ in range(num_epochs):
      first_traj = 0
      for i in range(num_mini_batches):
        start = i * mini_batch_size
        stop = (i + 1) * mini_batch_size

        dones = self.dones.squeeze(-1)
        last_was_done = torch.zeros_like(dones, dtype=torch.bool)
        last_was_done[1:] = dones[:-1]
        last_was_done[0] = True
        trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
        last_traj = first_traj + trajectories_batch_size

        last_was_done_t = last_was_done.permute(1, 0)
        hidden_state_a_batch = None
        if self.saved_hidden_state_a is not None:
          hidden_state_a_batch = [
            saved_hidden_state.permute(2, 0, 1, 3)[last_was_done_t][first_traj:last_traj]
            .transpose(1, 0)
            .contiguous()
            for saved_hidden_state in self.saved_hidden_state_a
          ]
          hidden_state_a_batch = hidden_state_a_batch[0] if len(hidden_state_a_batch) == 1 else hidden_state_a_batch

        hidden_state_c_batch = None
        if self.saved_hidden_state_c is not None:
          hidden_state_c_batch = [
            saved_hidden_state.permute(2, 0, 1, 3)[last_was_done_t][first_traj:last_traj]
            .transpose(1, 0)
            .contiguous()
            for saved_hidden_state in self.saved_hidden_state_c
          ]
          hidden_state_c_batch = hidden_state_c_batch[0] if len(hidden_state_c_batch) == 1 else hidden_state_c_batch

        batch = RolloutStorage.Batch(
          observations=padded_obs_trajectories[:, first_traj:last_traj],
          actions=self.actions[:, start:stop],
          values=self.values[:, start:stop],
          advantages=self.advantages[:, start:stop],
          returns=self.returns[:, start:stop],
          old_actions_log_prob=self.actions_log_prob[:, start:stop],
          old_distribution_params=tuple(p[:, start:stop] for p in self.distribution_params),  # type: ignore[arg-type]
          hidden_states=(hidden_state_a_batch, hidden_state_c_batch),
          masks=trajectory_masks[:, first_traj:last_traj],
        )
        batch.actor_loss_mask = self.actor_loss_masks[:, start:stop]
        yield batch
        first_traj = last_traj


class GoalkeeperStudentPPO(PPO):
  """PPO that keeps an online MoE teacher imitation loss during updates."""

  def __init__(
    self,
    *args,
    condition_aux_coef: float = 0.05,
    condition_aux_final_coef: float | None = None,
    condition_aux_anneal_updates: int = 0,
    condition_aux_active_only: bool = True,
    reference_kl_coef: float = 0.0,
    reference_kl_idle_coef: float | None = None,
    reference_kl_active_coef: float | None = None,
    reference_kl_std_floor: float = 0.1,
    teacher_kl_checkpoint: str | None = None,
    teacher_kl_coef: float = 0.0,
    teacher_kl_std_floor: float = 0.05,
    prior_disc_dataset_dir: str | None = None,
    prior_disc_reward_coef: float = 0.0,
    prior_disc_updates: int = 1,
    prior_disc_batch_size: int = 256,
    prior_disc_learning_rate: float = 3.0e-4,
    prior_disc_reward_clip: float = 5.0,
    active_bc_dataset_dir: str | None = None,
    active_bc_coef: float = 0.0,
    active_bc_final_coef: float | None = None,
    active_bc_anneal_updates: int = 0,
    active_bc_batch_size: int = 256,
    min_action_std: float = 1.0e-4,
    max_action_std: float | None = None,
    actor_mean_clip: float | None = None,
    ppo_log_ratio_clip: float = 8.0,
    idle_deterministic_actions: bool = True,
    mask_idle_actor_loss: bool = False,
    idle_actor_loss_weight: float = 0.0,
    critic_warmup_iterations: int = 0,
    freeze_actor_obs_normalization: bool = False,
    env: VecEnv | None = None,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.condition_aux_coef = float(condition_aux_coef)
    self.condition_aux_final_coef = float(
      condition_aux_coef if condition_aux_final_coef is None else condition_aux_final_coef
    )
    self.condition_aux_anneal_updates = max(0, int(condition_aux_anneal_updates))
    self.condition_aux_active_only = bool(condition_aux_active_only)
    self.reference_kl_coef = float(reference_kl_coef)
    self.reference_kl_idle_coef = float(reference_kl_coef if reference_kl_idle_coef is None else reference_kl_idle_coef)
    self.reference_kl_active_coef = float(
      reference_kl_coef if reference_kl_active_coef is None else reference_kl_active_coef
    )
    self.reference_kl_std_floor = max(float(reference_kl_std_floor), 1.0e-6)
    self.reference_actor = None
    self.teacher_kl_coef = float(teacher_kl_coef)
    self.teacher_kl_std_floor = max(float(teacher_kl_std_floor), 1.0e-6)
    self.teacher_kl_policy = self._load_teacher_kl_policy(teacher_kl_checkpoint, env)
    self.prior_disc_dataset_dir = prior_disc_dataset_dir
    self.prior_disc_reward_coef = float(prior_disc_reward_coef)
    self.prior_disc_updates = max(0, int(prior_disc_updates))
    self.prior_disc_batch_size = max(1, int(prior_disc_batch_size))
    self.prior_disc_learning_rate = float(prior_disc_learning_rate)
    self.prior_disc_reward_clip = float(prior_disc_reward_clip)
    self.active_bc_dataset_dir = active_bc_dataset_dir
    self.active_bc_coef = float(active_bc_coef)
    self.active_bc_final_coef = float(active_bc_coef if active_bc_final_coef is None else active_bc_final_coef)
    self.active_bc_anneal_updates = max(0, int(active_bc_anneal_updates))
    self.active_bc_batch_size = max(1, int(active_bc_batch_size))
    self.prior_dataset = None
    self.prior_discriminator = None
    self.prior_disc_optimizer = None
    self.active_bc_dataset = None
    self.active_bc_obs_dim = None
    prior_source_dir = self.prior_disc_dataset_dir if self.prior_disc_reward_coef > 0.0 else None
    active_bc_source_dir = self.active_bc_dataset_dir if self.active_bc_coef > 0.0 else None
    shared_dataset = None
    if prior_source_dir and active_bc_source_dir and prior_source_dir == active_bc_source_dir:
      shared_dataset = GoalkeeperPriorDataset(prior_source_dir, device=self.device)
    if prior_source_dir:
      self.prior_dataset = shared_dataset or GoalkeeperPriorDataset(prior_source_dir, device=self.device)
      self.prior_discriminator = GoalkeeperPriorDiscriminator(
        self.prior_dataset.input_dim,
        reward_clip=self.prior_disc_reward_clip,
      ).to(self.device)
      self.prior_disc_optimizer = torch.optim.Adam(
        self.prior_discriminator.parameters(),
        lr=self.prior_disc_learning_rate,
      )
      print(
        f"[INFO] Loaded goalkeeper prior dataset from {self.prior_disc_dataset_dir} "
        f"({self.prior_dataset.num_samples} samples)"
      )
    if active_bc_source_dir:
      self.active_bc_dataset = shared_dataset or GoalkeeperPriorDataset(active_bc_source_dir, device=self.device)
      self.active_bc_obs_dim = self.active_bc_dataset.input_dim - self.actor.num_actions
      print(
        f"[INFO] Loaded goalkeeper active BC dataset from {self.active_bc_dataset_dir} "
        f"({self.active_bc_dataset.num_samples} samples)"
      )
    self.min_action_std = float(min_action_std)
    self.max_action_std = None if max_action_std is None else float(max_action_std)
    if hasattr(self.actor, "min_sample_std"):
      self.actor.min_sample_std = self.min_action_std
    self.actor_mean_clip = None if actor_mean_clip is None else float(actor_mean_clip)
    self.ppo_log_ratio_clip = float(ppo_log_ratio_clip)
    self.idle_deterministic_actions = bool(idle_deterministic_actions)
    self.mask_idle_actor_loss = bool(mask_idle_actor_loss)
    self.idle_actor_loss_weight = min(max(float(idle_actor_loss_weight), 0.0), 1.0)
    self.critic_warmup_iterations = max(0, int(critic_warmup_iterations))
    self.freeze_actor_obs_normalization = bool(freeze_actor_obs_normalization)
    self.env = env
    # Skip the dead condition_aux_head forward when no aux loss is computed.
    if float(condition_aux_coef) <= 0.0 and float(
      condition_aux_final_coef or 0.0
    ) <= 0.0:
      self.actor.condition_aux_enabled = False
    self._act_step = 0
    self._ppo_update_count = 0

  def set_reference_actor_from_current(self) -> None:
    """Freeze a copy of the current actor as the KL anchor policy."""
    self.reference_actor = copy.deepcopy(self.actor)
    self.reference_actor.eval()
    for param in self.reference_actor.parameters():
      param.requires_grad_(False)

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    """Record one environment step, with optional actor-normalizer freeze."""
    if not self.freeze_actor_obs_normalization:
      self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)
    if self.rnd:
      self.rnd.update_normalization(obs)

    self.transition.rewards = rewards.clone()
    self.transition.dones = dones

    if self.rnd:
      self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
      self.transition.rewards += self.intrinsic_rewards

    if "time_outs" in extras:
      self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
        1,
      )
    self.transition.rewards += self._prior_discriminator_reward(obs, self.transition.actions)

    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.reset(dones)
    self.critic.reset(dones)

  def _load_teacher_kl_policy(self, checkpoint_path: str | None, env: VecEnv | None):
    if not checkpoint_path or self.teacher_kl_coef <= 0.0:
      return None
    if env is None:
      raise ValueError("GoalkeeperStudentPPO needs env to build the teacher KL policy.")
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
      raise FileNotFoundError(f"Teacher KL checkpoint not found: {path}")
    from src.tasks.soccer.modules.moe7_prepare_actor import MoE7PrepareGoalkeeperActor

    loaded = torch.load(path, map_location=self.device, weights_only=False)
    bundle = (
      loaded["actor_state_dict"]
      if isinstance(loaded, dict) and "actor_state_dict" in loaded and "sr" in loaded["actor_state_dict"]
      else loaded
    )
    if not isinstance(bundle, dict) or "sr" not in bundle:
      raise ValueError(f"Teacher KL checkpoint is not a MoE goalkeeper bundle: {path}")
    obs = env.get_observations().to(self.device)
    teacher = MoE7PrepareGoalkeeperActor(
      obs,
      {"actor": ["actor"], "critic": ["critic"]},
      "actor",
      env.num_actions,
      hidden_dims=(512, 256, 256),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
        "freeze_idle_std": True,
        "idle_std_min": 0.15,
        "idle_std_max": 0.15,
      },
    ).to(self.device)
    teacher.load_moe_bundle(bundle)
    teacher.eval()
    for param in teacher.parameters():
      param.requires_grad_(False)
    print(f"[INFO] Loaded teacher KL policy from: {path}")
    return teacher

  def _teacher_kl_loss(self, batch: RolloutStorage.Batch) -> torch.Tensor:
    teacher = getattr(self, "teacher_kl_policy", None)
    if teacher is None:
      return torch.zeros((), device=self.device)
    student_params = tuple(
      p.reshape(-1, p.shape[-1]) if p.dim() > 2 else p
      for p in self.actor.output_distribution_params
    )
    observations = batch.observations
    if batch.masks is not None:
      observations = unpad_trajectories(observations, batch.masks)
    teacher_obs = self._teacher_kl_obs(observations)
    with torch.no_grad():
      teacher(teacher_obs)
      teacher_params = self._teacher_kl_params(teacher.output_distribution_params)
    n = min(student_params[0].shape[0], teacher_params[0].shape[0])
    student_params = tuple(p[:n] for p in student_params)
    teacher_params = tuple(p[:n] for p in teacher_params)
    per_sample_kl = self.actor.get_kl_divergence(teacher_params, student_params)
    per_sample_kl = torch.clamp(per_sample_kl, max=2.0)
    active_mask = getattr(batch, "actor_loss_mask", None)
    if active_mask is None:
      return per_sample_kl.mean()
    active_mask = active_mask.reshape(-1).to(device=per_sample_kl.device) > 0.5
    if active_mask.shape[0] < n:
      n = active_mask.shape[0]
      per_sample_kl = per_sample_kl[:n]
    elif active_mask.shape[0] > n:
      active_mask = active_mask[:n]
    if not torch.any(active_mask):
      return torch.zeros((), device=self.device)
    return per_sample_kl[active_mask].mean()

  @staticmethod
  def _teacher_kl_obs(observations: TensorDict) -> TensorDict:
    """Strip the 4D student condition and flatten to 2D for the non-recurrent teacher."""
    if not isinstance(observations, TensorDict):
      return observations
    for key in ("actor", "student"):
      if key not in observations.keys():
        continue
      tensor = observations[key]
      if tensor.shape[-1] in (GOALKEEPER_STUDENT_OBS_DIM, GOALKEEPER_TEACHER_ACTOR_OBS_DIM + 4):
        tensor = tensor[..., :960]
      if tensor.dim() == 3:
        tensor = tensor.reshape(-1, tensor.shape[-1])
      return TensorDict({key: tensor}, batch_size=list(tensor.shape[:-1]))
    return observations

  def _teacher_kl_params(self, params: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if len(params) != 2:
      return params
    mean, std = params
    clip = getattr(self, "actor_mean_clip", None)
    if clip is not None and clip > 0.0:
      mean = torch.nan_to_num(mean, nan=0.0, posinf=clip, neginf=-clip).clamp(min=-clip, max=clip)
    floor = max(float(getattr(self, "teacher_kl_std_floor", 0.0)), 0.0)
    if floor <= 0.0:
      return mean, std
    return mean, std.clamp_min(floor)

  def train_mode(self) -> None:
    super().train_mode()
    teacher_kl_policy = getattr(self, "teacher_kl_policy", None)
    if teacher_kl_policy is not None:
      teacher_kl_policy.eval()

  def act(self, obs: TensorDict) -> torch.Tensor:
    self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
    self._clamp_actor_std()
    sampled_actions = self._clip_actor_actions(self.actor(obs, stochastic_output=True).detach())
    action_mean = self._clip_actor_actions(self.actor.output_mean.detach())
    idle_mask = self._idle_action_mask(obs).unsqueeze(-1)
    if getattr(self, "idle_deterministic_actions", True):
      actions = torch.where(idle_mask, action_mean, sampled_actions)
    else:
      actions = sampled_actions
    actions = self._clip_actor_actions(actions)
    self.transition.actions = actions
    self.transition.values = self.critic(obs).detach()
    self.transition.actions_log_prob = self.actor.get_output_log_prob(actions).detach()
    self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
    self.transition.observations = obs
    self.transition.actor_loss_mask = (~idle_mask).to(dtype=torch.float32)
    self._act_step += 1
    return actions

  def _idle_condition_mask(self, obs: TensorDict) -> torch.Tensor:
    raw_student_obs = getattr(self.actor, "_raw_student_obs", None)
    if callable(raw_student_obs):
      raw = raw_student_obs(obs)
    elif "student" in obs.keys():
      raw = obs["student"]
    elif "actor" in obs.keys():
      raw = obs["actor"]
    else:
      raw = torch.cat([value for value in obs.values()], dim=-1)
    if raw.shape[-1] >= GOALKEEPER_STUDENT_OBS_DIM:
      return raw[..., GOALKEEPER_PHASE_IDLE_INDEX] > 0.5
    return raw[..., -1] > 0.5

  def _idle_action_mask(self, obs: TensorDict) -> torch.Tensor:
    env = getattr(self, "env", None)
    source_env = getattr(env, "unwrapped", env)
    launched = getattr(source_env, "_gk_delayed_ball_launched", None)
    num_envs = getattr(source_env, "num_envs", getattr(env, "num_envs", None))
    if launched is not None and num_envs is not None and launched.shape[0] == num_envs:
      return ~launched.to(device=self.device, dtype=torch.bool)
    return self._idle_condition_mask(obs)

  def update(self) -> dict[str, float]:
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_condition_aux_loss = 0.0
    mean_reference_kl_loss = 0.0
    mean_teacher_kl_loss = 0.0
    mean_active_bc_loss = 0.0
    mean_prior_disc_loss = 0.0
    mean_prior_disc_expert_acc = 0.0
    mean_prior_disc_policy_acc = 0.0

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    else:
      generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

    critic_warmup = self._ppo_update_count < self.critic_warmup_iterations
    previous_actor_trainable = self.set_actor_trainable(False) if critic_warmup else None
    update_index = 0
    try:
      for batch in generator:
        self._clamp_actor_std()
        original_batch_size = batch.observations.batch_size[0]

        if self.normalize_advantage_per_mini_batch:
          with torch.no_grad():
            batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

        self.actor(
          batch.observations,
          masks=batch.masks,
          hidden_state=batch.hidden_states[0],
          stochastic_output=True,
        )
        self._clip_actor_distribution_mean()
        actions_log_prob = self.actor.get_output_log_prob(batch.actions)
        values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
        distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
        entropy = self.actor.output_entropy[:original_batch_size]

        if self.desired_kl is not None and self.schedule == "adaptive" and not critic_warmup:
          self._update_learning_rate(batch.old_distribution_params, distribution_params)

        ratio = self._ppo_ratio(actions_log_prob, torch.squeeze(batch.old_actions_log_prob))
        advantages = torch.squeeze(batch.advantages)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss, entropy_loss = self._masked_actor_losses(
          batch,
          advantages,
          ratio,
          entropy,
          clipped_ratio=clipped_ratio,
        )

        if self.use_clipped_value_loss:
          value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
          value_losses = (values - batch.returns).pow(2)
          value_losses_clipped = (value_clipped - batch.returns).pow(2)
          value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
          value_loss = (batch.returns - values).pow(2).mean()

        condition_aux_weight = self._scheduled_coef(
          self.condition_aux_coef,
          self.condition_aux_final_coef,
          self.condition_aux_anneal_updates,
        )
        reference_kl_weight = 1.0 if self._uses_phase_reference_kl() else self.reference_kl_coef
        teacher_kl_weight = self.teacher_kl_coef
        active_bc_weight = self._active_bc_weight()
        if critic_warmup:
          zero = value_loss * 0.0
          surrogate_loss = zero
          entropy_loss = zero
          condition_aux_weight = 0.0
          reference_kl_weight = 0.0
          teacher_kl_weight = 0.0
          active_bc_weight = 0.0
        condition_aux_loss = (
          self._condition_aux_loss(batch) if condition_aux_weight > 0.0 else torch.zeros((), device=self.device)
        )
        reference_kl_loss = (
          self._reference_kl_loss(batch, phase_weighted=self._uses_phase_reference_kl())
          if reference_kl_weight > 0.0
          else torch.zeros((), device=self.device)
        )
        teacher_kl_loss = (
          self._teacher_kl_loss(batch)
          if teacher_kl_weight > 0.0
          else torch.zeros((), device=self.device)
        )
        active_bc_loss = (
          self._active_bc_loss()
          if active_bc_weight > 0.0
          else torch.zeros((), device=self.device)
        )
        loss = (
          surrogate_loss
          + self.value_loss_coef * value_loss
          - self.entropy_coef * entropy_loss
          + condition_aux_weight * condition_aux_loss
          + reference_kl_weight * reference_kl_loss
          + teacher_kl_weight * teacher_kl_loss
          + active_bc_weight * active_bc_loss
        )
        if not torch.isfinite(loss):
          self._clamp_actor_std()
          update_index += 1
          continue

        prior_metrics = self._update_prior_discriminator_from_batch(batch)
        self.optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
          self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._clamp_actor_std()

        mean_value_loss += value_loss.item()
        mean_surrogate_loss += surrogate_loss.item()
        mean_entropy += entropy_loss.item()
        mean_condition_aux_loss += condition_aux_loss.item()
        mean_reference_kl_loss += reference_kl_loss.item()
        mean_teacher_kl_loss += teacher_kl_loss.item()
        mean_active_bc_loss += active_bc_loss.item()
        mean_prior_disc_loss += prior_metrics["prior_disc"]
        mean_prior_disc_expert_acc += prior_metrics["prior_disc_expert_acc"]
        mean_prior_disc_policy_acc += prior_metrics["prior_disc_policy_acc"]
        update_index += 1
    finally:
      if previous_actor_trainable is not None:
        self.restore_actor_trainable(previous_actor_trainable)

    num_updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    self._ppo_update_count += 1
    return {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "condition_aux": mean_condition_aux_loss / num_updates,
      "reference_kl": mean_reference_kl_loss / num_updates,
      "teacher_kl": mean_teacher_kl_loss / num_updates,
      "active_bc": mean_active_bc_loss / num_updates,
      "prior_disc": mean_prior_disc_loss / num_updates,
      "prior_disc_expert_acc": mean_prior_disc_expert_acc / num_updates,
      "prior_disc_policy_acc": mean_prior_disc_policy_acc / num_updates,
    }

  def _prior_discriminator_features(self, observations: TensorDict, actions: torch.Tensor, masks=None) -> torch.Tensor:
    raw_student_obs = getattr(self.actor, "_raw_student_obs", None)
    if callable(raw_student_obs):
      raw = raw_student_obs(observations)
    elif isinstance(observations, TensorDict) and "student" in observations.keys():
      raw = observations["student"]
    elif isinstance(observations, TensorDict) and "actor" in observations.keys():
      raw = observations["actor"]
    else:
      raw = torch.cat([value for value in observations.values()], dim=-1)
    if masks is not None:
      raw = unpad_trajectories(raw, masks).reshape(-1, raw.shape[-1])
      actions = actions.reshape(-1, actions.shape[-1])
      if actions.shape[0] != raw.shape[0]:
        actions = actions[: raw.shape[0]]
    else:
      raw = raw.reshape(-1, raw.shape[-1])
      actions = actions.reshape(-1, actions.shape[-1])
    return torch.cat([raw.to(device=self.device), actions.to(device=self.device)], dim=-1)

  def _active_bc_loss(self) -> torch.Tensor:
    dataset = getattr(self, "active_bc_dataset", None)
    coef = float(getattr(self, "active_bc_coef", 0.0))
    if dataset is None or coef <= 0.0:
      return torch.zeros((), device=self.device)
    batch = dataset.sample(int(self.active_bc_batch_size)).to(device=self.device)
    obs_dim = getattr(self, "active_bc_obs_dim", None)
    if obs_dim is None:
      obs_dim = batch.shape[-1] - self.actor.num_actions
    obs_dim = int(obs_dim)
    obs_key = "student"
    obs_groups = getattr(self.actor, "obs_groups", None)
    if obs_groups:
      obs_key = obs_groups[0]
    obs_tensor = self._upgrade_student_obs(batch[:, :obs_dim])
    obs = TensorDict({obs_key: obs_tensor}, batch_size=[batch.shape[0]], device=self.device)
    expert_actions = batch[:, obs_dim:]
    forward_bc_chunk = getattr(self.actor, "forward_bc_chunk", None)
    if callable(forward_bc_chunk):
      masks = torch.ones(1, batch.shape[0], dtype=torch.bool, device=self.device)
      actor_mean, _ = forward_bc_chunk(obs_tensor.unsqueeze(0), masks, hidden_state=None)
    else:
      try:
        self.actor(obs, masks=None, hidden_state=None, stochastic_output=False)
      except TypeError:
        self.actor(obs, stochastic_output=False)
      actor_mean = self.actor.output_distribution_params[0]
    if actor_mean.shape[-1] != expert_actions.shape[-1]:
      n = min(actor_mean.shape[-1], expert_actions.shape[-1])
      actor_mean = actor_mean[..., :n]
      expert_actions = expert_actions[..., :n]
    return F.smooth_l1_loss(actor_mean, expert_actions)

  def _prior_discriminator_reward(self, obs: TensorDict, actions: torch.Tensor) -> torch.Tensor:
    discriminator = getattr(self, "prior_discriminator", None)
    coef = float(getattr(self, "prior_disc_reward_coef", 0.0))
    if discriminator is None or coef <= 0.0:
      return torch.zeros(actions.shape[0], device=self.device)
    features = self._prior_discriminator_features(obs, actions)
    rewards = discriminator.reward(features) * coef
    active = getattr(self.transition, "actor_loss_mask", None)
    if active is None:
      active = (~self._idle_action_mask(obs)).to(dtype=torch.float32).unsqueeze(-1)
    active = active.reshape(-1).to(device=rewards.device, dtype=rewards.dtype)
    n = min(rewards.shape[0], active.shape[0])
    return rewards[:n] * active[:n]

  def _update_prior_discriminator_from_batch(self, batch: RolloutStorage.Batch) -> dict[str, float]:
    discriminator = getattr(self, "prior_discriminator", None)
    dataset = getattr(self, "prior_dataset", None)
    if discriminator is None or dataset is None or self.prior_disc_updates <= 0:
      return {"prior_disc": 0.0, "prior_disc_expert_acc": 0.0, "prior_disc_policy_acc": 0.0}
    features = self._prior_discriminator_features(batch.observations, batch.actions, batch.masks).detach()
    active_mask = getattr(batch, "actor_loss_mask", None)
    if active_mask is not None:
      active_mask = active_mask.reshape(-1).to(device=features.device) > 0.5
      if active_mask.shape[0] != features.shape[0]:
        active_mask = active_mask[: features.shape[0]]
      if torch.any(active_mask):
        features = features[active_mask]
    return self._update_prior_discriminator(features)

  def _update_prior_discriminator(self, policy_features: torch.Tensor) -> dict[str, float]:
    discriminator = getattr(self, "prior_discriminator", None)
    dataset = getattr(self, "prior_dataset", None)
    optimizer = getattr(self, "prior_disc_optimizer", None)
    if discriminator is None or dataset is None or optimizer is None or policy_features.numel() == 0:
      return {"prior_disc": 0.0, "prior_disc_expert_acc": 0.0, "prior_disc_policy_acc": 0.0}
    total_loss = 0.0
    total_expert_acc = 0.0
    total_policy_acc = 0.0
    updates = max(1, int(self.prior_disc_updates))
    for _ in range(updates):
      batch_size = min(int(self.prior_disc_batch_size), int(policy_features.shape[0]))
      policy_idx = torch.randint(policy_features.shape[0], (batch_size,), device=policy_features.device)
      policy_batch = policy_features[policy_idx].detach()
      expert_batch = dataset.sample(batch_size).to(device=policy_features.device, dtype=policy_features.dtype)
      loss, metrics = discriminator_loss(discriminator, expert_batch, policy_batch)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      total_loss += float(loss.item())
      total_expert_acc += metrics["expert_acc"]
      total_policy_acc += metrics["policy_acc"]
    return {
      "prior_disc": total_loss / updates,
      "prior_disc_expert_acc": total_expert_acc / updates,
      "prior_disc_policy_acc": total_policy_acc / updates,
    }

  def _clamp_actor_std(self) -> None:
    distribution = getattr(self.actor, "distribution", None)
    std_param = getattr(distribution, "std_param", None)
    if std_param is None:
      return
    with torch.no_grad():
      max_std = getattr(self, "max_action_std", None)
      if max_std is None or max_std <= 0.0:
        std_param.nan_to_num_(nan=self.min_action_std, posinf=self.min_action_std, neginf=self.min_action_std)
        std_param.clamp_(min=self.min_action_std)
      else:
        std_param.nan_to_num_(nan=self.min_action_std, posinf=max_std, neginf=self.min_action_std)
        std_param.clamp_(min=self.min_action_std, max=max_std)

  def set_actor_trainable(self, trainable: bool) -> list[bool]:
    previous = [param.requires_grad for param in self.actor.parameters()]
    for param in self.actor.parameters():
      param.requires_grad_(trainable)
    return previous

  def restore_actor_trainable(self, previous: list[bool]) -> None:
    for param, requires_grad in zip(self.actor.parameters(), previous, strict=False):
      param.requires_grad_(requires_grad)

  def _actor_output_mean(self) -> torch.Tensor:
    mean = self.actor.output_mean
    clip = getattr(self, "actor_mean_clip", None)
    if clip is None or clip <= 0.0:
      return torch.nan_to_num(mean, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    mean = torch.nan_to_num(mean, nan=0.0, posinf=clip, neginf=-clip)
    return mean.clamp(min=-clip, max=clip)

  def _clip_actor_actions(self, actions: torch.Tensor) -> torch.Tensor:
    clip = getattr(self, "actor_mean_clip", None)
    if clip is None or clip <= 0.0:
      return torch.nan_to_num(actions, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    actions = torch.nan_to_num(actions, nan=0.0, posinf=clip, neginf=-clip)
    return actions.clamp(min=-clip, max=clip)

  def _clip_actor_distribution_mean(self) -> None:
    clip = getattr(self, "actor_mean_clip", None)
    if clip is None or clip <= 0.0:
      return
    distribution = getattr(self.actor, "distribution", None)
    mean = getattr(distribution, "mean", None)
    update = getattr(distribution, "update", None)
    if mean is None or not callable(update):
      return
    clipped = torch.nan_to_num(mean, nan=0.0, posinf=clip, neginf=-clip).clamp(min=-clip, max=clip)
    update(clipped)

  def _ppo_ratio(self, actions_log_prob: torch.Tensor, old_actions_log_prob: torch.Tensor) -> torch.Tensor:
    log_ratio = actions_log_prob - old_actions_log_prob
    clip = max(float(getattr(self, "ppo_log_ratio_clip", 8.0)), 1.0e-6)
    log_ratio = torch.nan_to_num(log_ratio, nan=0.0, posinf=clip, neginf=-clip)
    return torch.exp(log_ratio.clamp(min=-clip, max=clip))

  def _masked_actor_losses(
    self,
    batch_or_observations,
    advantages: torch.Tensor,
    ratio: torch.Tensor,
    entropy: torch.Tensor,
    *,
    clipped_ratio: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    clipped = clipped_ratio if clipped_ratio is not None else ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
    surrogate = -advantages * ratio
    surrogate_clipped = -advantages * clipped
    per_sample_surrogate = torch.max(surrogate, surrogate_clipped)
    if not getattr(self, "mask_idle_actor_loss", False):
      return per_sample_surrogate.mean(), entropy.mean()

    active_mask = getattr(batch_or_observations, "actor_loss_mask", None)
    if active_mask is None:
      active_mask = ~self._idle_condition_mask(batch_or_observations)
    else:
      active_mask = active_mask.to(device=per_sample_surrogate.device) > 0.5
    if active_mask.shape != per_sample_surrogate.shape:
      active_mask = active_mask.reshape(per_sample_surrogate.shape)
    active_mask = active_mask.to(device=per_sample_surrogate.device, dtype=torch.bool)
    idle_weight = min(max(float(getattr(self, "idle_actor_loss_weight", 0.0)), 0.0), 1.0)
    if idle_weight <= 0.0 and not torch.any(active_mask):
      zero = per_sample_surrogate.sum() * 0.0
      return zero, zero
    weights = torch.where(
      active_mask,
      torch.ones_like(per_sample_surrogate),
      torch.full_like(per_sample_surrogate, idle_weight),
    )
    denom = weights.sum().clamp_min(1.0e-6)
    return (per_sample_surrogate * weights).sum() / denom, (entropy * weights).sum() / denom

  def _scheduled_coef(self, start: float, final: float | None, anneal_updates: int) -> float:
    final_value = start if final is None else final
    if anneal_updates <= 0:
      return float(start)
    progress = min(max(float(getattr(self, "_ppo_update_count", 0)) / float(anneal_updates), 0.0), 1.0)
    return float(start + (final_value - start) * progress)

  def _active_bc_weight(self) -> float:
    return self._scheduled_coef(
      self.active_bc_coef,
      self.active_bc_final_coef,
      self.active_bc_anneal_updates,
    )

  def _uses_phase_reference_kl(self) -> bool:
    return (
      abs(float(getattr(self, "reference_kl_idle_coef", self.reference_kl_coef)) - self.reference_kl_coef) > 1.0e-12
      or abs(float(getattr(self, "reference_kl_active_coef", self.reference_kl_coef)) - self.reference_kl_coef)
      > 1.0e-12
    )

  def _reference_kl_loss(self, batch: RolloutStorage.Batch, *, phase_weighted: bool = False) -> torch.Tensor:
    reference_actor = getattr(self, "reference_actor", None)
    if reference_actor is None:
      return torch.zeros((), device=self.device)
    current_params = self._reference_kl_params(self.actor.output_distribution_params)
    with torch.no_grad():
      reference_actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      reference_params = self._reference_kl_params(
        tuple(param.detach() for param in reference_actor.output_distribution_params)
      )
    per_sample_kl = self.actor.get_kl_divergence(reference_params, current_params)
    if not phase_weighted:
      return per_sample_kl.mean()
    weights = self._reference_kl_phase_weights(batch, per_sample_kl)
    return (per_sample_kl * weights).mean()

  def _reference_kl_phase_weights(self, batch: RolloutStorage.Batch, like: torch.Tensor) -> torch.Tensor:
    active_mask = getattr(batch, "actor_loss_mask", None)
    if active_mask is None:
      active_mask = torch.ones_like(like, dtype=torch.bool)
    else:
      active_mask = active_mask.to(device=like.device) > 0.5
      if active_mask.shape != like.shape:
        active_mask = active_mask.reshape(like.shape)
    idle_coef = float(getattr(self, "reference_kl_idle_coef", self.reference_kl_coef))
    active_coef = float(getattr(self, "reference_kl_active_coef", self.reference_kl_coef))
    return torch.where(
      active_mask,
      torch.full_like(like, active_coef),
      torch.full_like(like, idle_coef),
    )

  def _reference_kl_params(self, params: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if len(params) != 2:
      return params
    mean, std = params
    clip = getattr(self, "actor_mean_clip", None)
    if clip is not None and clip > 0.0:
      mean = torch.nan_to_num(mean, nan=0.0, posinf=clip, neginf=-clip).clamp(min=-clip, max=clip)
    floor = max(float(getattr(self, "reference_kl_std_floor", 0.0)), 0.0)
    if floor <= 0.0:
      return mean, std
    return mean, std.clamp_min(floor)

  def _condition_aux_loss(self, batch: RolloutStorage.Batch) -> torch.Tensor:
    aux = getattr(self.actor, "condition_aux_output", None)
    if aux is None:
      return torch.zeros((), device=self.device)
    raw_student_obs = getattr(self.actor, "_raw_student_obs", None)
    if callable(raw_student_obs):
      raw = raw_student_obs(batch.observations)
    elif "student" in batch.observations.keys():
      raw = batch.observations["student"]
    else:
      raw = torch.cat([value for value in batch.observations.values()], dim=-1)
    if batch.masks is not None:
      raw = unpad_trajectories(raw, batch.masks).reshape(-1, raw.shape[-1])
    route_targets = self._route_targets(raw)
    active = route_targets != 6
    region_logits = aux["region_logits"].reshape(-1, aux["region_logits"].shape[-1])
    if route_targets.shape[0] != region_logits.shape[0]:
      n = min(route_targets.shape[0], region_logits.shape[0])
      route_targets = route_targets[:n]
      active = active[:n]
      region_logits = region_logits[:n]
    if self.condition_aux_active_only:
      if not torch.any(active):
        return torch.zeros((), device=self.device)
      route_targets = route_targets[active]
      region_logits = region_logits[active]
    return nn.functional.cross_entropy(region_logits, route_targets.detach())

  def _route_targets(self, student_obs: torch.Tensor) -> torch.Tensor:
    return self._route_targets_from_student_obs(student_obs)

  @staticmethod
  def _route_targets_from_student_obs(
    student_obs: torch.Tensor,
    *,
    z_low: float = 0.85,
    z_up: float = 1.35,
    vz_low: float = -5.0,
    idle_speed_threshold: float = 0.5,
    idle_incoming_vx_threshold: float = -0.5,
    gravity: float = 9.81,
    dt: float = 0.02,
    gate: nn.Module | None = None,
    gate_mean: torch.Tensor | None = None,
    gate_std: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Return MoE7-style route labels from student obs: 0-5 active, 6 prepare."""
    ball_history = student_obs[..., :30].reshape(*student_obs.shape[:-1], 10, 3)
    pos = ball_history[..., -1, :]
    prev = ball_history[..., -2, :]
    vel = (pos - prev) / dt
    bx = pos[..., 0]
    vx = vel[..., 0]
    if gate is not None and gate_mean is not None and gate_std is not None:
      features = torch.cat([pos, vel], dim=-1)
      flat_features = features.reshape(-1, features.shape[-1])
      gate_mean = gate_mean.to(device=flat_features.device, dtype=flat_features.dtype)
      gate_std = gate_std.to(device=flat_features.device, dtype=flat_features.dtype).clamp_min(1.0e-6)
      flat_route = gate((flat_features - gate_mean) / gate_std).argmax(dim=-1).clamp(max=5)
      route = flat_route.reshape(bx.shape)
    else:
      t = torch.clamp(-bx / (vx - 1.0e-3), 0.0, 2.0)
      cy = pos[..., 1] + vel[..., 1] * t
      cz = pos[..., 2] + vel[..., 2] * t - 0.5 * gravity * t * t
      route = torch.zeros_like(bx, dtype=torch.long)
      route = torch.where(cz < z_low, torch.full_like(route, 4), route)
      route = torch.where(cz > z_up, torch.full_like(route, 2), route)
      route = torch.where(vel[..., 2] - gravity * t < vz_low, torch.full_like(route, 4), route)
      route = route + (cy < 0.0).long()
    speed = torch.linalg.vector_norm(vel, dim=-1)
    if student_obs.shape[-1] >= GOALKEEPER_STUDENT_OBS_DIM:
      condition_idle = student_obs[..., GOALKEEPER_PHASE_IDLE_INDEX] > 0.5
    else:
      condition_idle = student_obs[..., -1] > 0.5
    kinematic_idle = (speed < idle_speed_threshold) | (vx >= idle_incoming_vx_threshold)
    idle = condition_idle | kinematic_idle
    return torch.where(idle, torch.full_like(route, 6), route)

  @staticmethod
  def _upgrade_student_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] == GOALKEEPER_STUDENT_OBS_DIM:
      return obs
    if obs.shape[-1] == GOALKEEPER_TEACHER_ACTOR_OBS_DIM + 4:
      return build_goalkeeper_student_obs(
        obs[..., :GOALKEEPER_TEACHER_ACTOR_OBS_DIM],
        obs[..., GOALKEEPER_TEACHER_ACTOR_OBS_DIM:],
      )
    return obs

  def _update_learning_rate(self, old_distribution_params, distribution_params) -> None:
    with torch.inference_mode():
      kl = self.actor.get_kl_divergence(old_distribution_params, distribution_params)
      kl_mean = torch.mean(kl)
      if self.is_multi_gpu:
        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
        kl_mean /= self.gpu_world_size
      if self.gpu_global_rank == 0:
        if kl_mean > self.desired_kl * 2.0:
          self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
          self.learning_rate = min(1e-2, self.learning_rate * 1.5)
      if self.is_multi_gpu:
        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
        torch.distributed.broadcast(lr_tensor, src=0)
        self.learning_rate = lr_tensor.item()
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = self.learning_rate

  @staticmethod
  def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "GoalkeeperStudentPPO":
    alg_class: type[GoalkeeperStudentPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore[assignment]
    actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore[assignment]
    critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore[assignment]

    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

    actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
    actor.num_actions = env.num_actions
    print(f"Actor Model: {actor}")
    if cfg["algorithm"].pop("share_cnn_encoders", None):
      cfg["critic"]["cnns"] = actor.cnns  # type: ignore[attr-defined]
    critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
    print(f"Critic Model: {critic}")

    storage = GoalkeeperStudentRolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
    return alg_class(
      actor,
      critic,
      storage,
      device=device,
      env=env,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg["multi_gpu"],
    )

  def broadcast_parameters(self) -> None:
    model_params = [self.actor.state_dict(), self.critic.state_dict()]
    torch.distributed.broadcast_object_list(model_params, src=0)
    self.actor.load_state_dict(model_params[0])
    self.critic.load_state_dict(model_params[1])

  def reduce_parameters(self) -> None:
    all_params = list(chain(self.actor.parameters(), self.critic.parameters()))
    grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
    all_grads = torch.cat(grads)
    torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
    all_grads /= self.gpu_world_size
    offset = 0
    for param in all_params:
      if param.grad is not None:
        numel = param.numel()
        param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
        offset += numel
