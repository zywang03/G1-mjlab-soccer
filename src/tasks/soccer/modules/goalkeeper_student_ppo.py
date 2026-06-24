"""Hybrid PPO + online teacher distillation for the goalkeeper student."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import torch
import torch.nn as nn
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


@dataclass
class GoalkeeperStudentPpoAlgorithmCfg:
  """PPO config extended with online teacher distillation knobs."""

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
  teacher_checkpoint_path: str | None = None
  distill_coef: float = 1.0
  distill_final_coef: float | None = None
  distill_anneal_updates: int = 0
  teacher_every_n_steps: int = 1
  distill_loss_type: str = "mse"
  offline_bc_dataset: str | None = None
  offline_bc_coef: float = 0.0
  offline_bc_final_coef: float | None = None
  offline_bc_anneal_updates: int = 0
  offline_idle_bc_coef: float = 0.0
  offline_idle_bc_final_coef: float | None = None
  offline_idle_bc_anneal_updates: int = 0
  offline_bc_active_fraction: float = 0.5
  offline_bc_batch_size: int = 64
  offline_bc_seq_len: int = 24
  offline_bc_every_n_updates: int = 1
  offline_bc_cache_shards: int = 8
  condition_aux_coef: float = 0.05
  condition_aux_final_coef: float | None = None
  condition_aux_anneal_updates: int = 0
  condition_aux_active_only: bool = True
  min_action_std: float = 1.0e-4
  max_action_std: float | None = None
  actor_mean_clip: float | None = None
  ppo_log_ratio_clip: float = 8.0
  idle_deterministic_actions: bool = True
  mask_idle_actor_loss: bool = False
  critic_warmup_iterations: int = 0
  freeze_actor_obs_normalization: bool = False


class GoalkeeperStudentRolloutStorage(RolloutStorage):
  """RL rollout storage with per-transition teacher actions."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.privileged_actions = torch.zeros(
      self.num_transitions_per_env,
      self.num_envs,
      *self.actions_shape,
      device=self.device,
    )

  def add_transition(self, transition: RolloutStorage.Transition) -> None:
    teacher_actions = getattr(transition, "privileged_actions", None)
    if teacher_actions is None:
      raise ValueError("GoalkeeperStudentPPO requires teacher actions for every rollout transition.")
    self.privileged_actions[self.step].copy_(teacher_actions)
    super().add_transition(transition)

  def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[RolloutStorage.Batch, None, None]:
    batch_size = self.num_envs * self.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

    observations = self.observations.flatten(0, 1)
    actions = self.actions.flatten(0, 1)
    teacher_actions = self.privileged_actions.flatten(0, 1)
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
        yield RolloutStorage.Batch(
          observations=observations[batch_idx],
          actions=actions[batch_idx],
          values=values[batch_idx],
          advantages=advantages[batch_idx],
          returns=returns[batch_idx],
          old_actions_log_prob=old_actions_log_prob[batch_idx],
          old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
          privileged_actions=teacher_actions[batch_idx],
        )

  def recurrent_mini_batch_generator(
    self,
    num_mini_batches: int,
    num_epochs: int = 8,
  ) -> Generator[RolloutStorage.Batch, None, None]:
    padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
    padded_teacher_trajectories, _ = split_and_pad_trajectories(self.privileged_actions, self.dones)
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

        yield RolloutStorage.Batch(
          observations=padded_obs_trajectories[:, first_traj:last_traj],
          actions=self.actions[:, start:stop],
          values=self.values[:, start:stop],
          advantages=self.advantages[:, start:stop],
          returns=self.returns[:, start:stop],
          old_actions_log_prob=self.actions_log_prob[:, start:stop],
          old_distribution_params=tuple(p[:, start:stop] for p in self.distribution_params),  # type: ignore[arg-type]
          hidden_states=(hidden_state_a_batch, hidden_state_c_batch),
          masks=trajectory_masks[:, first_traj:last_traj],
          privileged_actions=padded_teacher_trajectories[:, first_traj:last_traj],
        )
        first_traj = last_traj


class GoalkeeperStudentPPO(PPO):
  """PPO that keeps an online MoE teacher imitation loss during updates."""

  def __init__(
    self,
    *args,
    teacher_checkpoint_path: str | None = None,
    distill_coef: float = 1.0,
    distill_final_coef: float | None = None,
    distill_anneal_updates: int = 0,
    teacher_every_n_steps: int = 1,
    distill_loss_type: str = "mse",
    offline_bc_dataset: str | None = None,
    offline_bc_coef: float = 0.0,
    offline_bc_final_coef: float | None = None,
    offline_bc_anneal_updates: int = 0,
    offline_idle_bc_coef: float = 0.0,
    offline_idle_bc_final_coef: float | None = None,
    offline_idle_bc_anneal_updates: int = 0,
    offline_bc_active_fraction: float = 0.5,
    offline_bc_batch_size: int = 64,
    offline_bc_seq_len: int = 24,
    offline_bc_every_n_updates: int = 1,
    offline_bc_cache_shards: int = 8,
    condition_aux_coef: float = 0.05,
    condition_aux_final_coef: float | None = None,
    condition_aux_anneal_updates: int = 0,
    condition_aux_active_only: bool = True,
    min_action_std: float = 1.0e-4,
    max_action_std: float | None = None,
    actor_mean_clip: float | None = None,
    ppo_log_ratio_clip: float = 8.0,
    idle_deterministic_actions: bool = True,
    mask_idle_actor_loss: bool = False,
    critic_warmup_iterations: int = 0,
    freeze_actor_obs_normalization: bool = False,
    env: VecEnv | None = None,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.distill_coef = float(distill_coef)
    self.distill_final_coef = float(distill_coef if distill_final_coef is None else distill_final_coef)
    self.distill_anneal_updates = max(0, int(distill_anneal_updates))
    self.teacher_every_n_steps = max(1, int(teacher_every_n_steps))
    self.teacher_checkpoint_path = teacher_checkpoint_path
    self.teacher_policy = self._load_teacher_policy(teacher_checkpoint_path, env)
    self.teacher_route_gate = self._load_teacher_route_gate(teacher_checkpoint_path)
    self.loss_fn = nn.functional.smooth_l1_loss if distill_loss_type == "huber" else nn.functional.mse_loss
    self.offline_bc_coef = float(offline_bc_coef)
    self.offline_bc_final_coef = float(offline_bc_coef if offline_bc_final_coef is None else offline_bc_final_coef)
    self.offline_bc_anneal_updates = max(0, int(offline_bc_anneal_updates))
    self.offline_idle_bc_coef = float(offline_idle_bc_coef)
    self.offline_idle_bc_final_coef = float(
      offline_idle_bc_coef if offline_idle_bc_final_coef is None else offline_idle_bc_final_coef
    )
    self.offline_idle_bc_anneal_updates = max(0, int(offline_idle_bc_anneal_updates))
    self.offline_bc_active_fraction = min(max(float(offline_bc_active_fraction), 0.0), 1.0)
    self.offline_bc_batch_size = int(offline_bc_batch_size)
    self.offline_bc_seq_len = int(offline_bc_seq_len)
    self.offline_bc_every_n_updates = max(1, int(offline_bc_every_n_updates))
    self.offline_bc_cache_shards = max(1, int(offline_bc_cache_shards))
    self.condition_aux_coef = float(condition_aux_coef)
    self.condition_aux_final_coef = float(
      condition_aux_coef if condition_aux_final_coef is None else condition_aux_final_coef
    )
    self.condition_aux_anneal_updates = max(0, int(condition_aux_anneal_updates))
    self.condition_aux_active_only = bool(condition_aux_active_only)
    self.min_action_std = float(min_action_std)
    self.max_action_std = None if max_action_std is None else float(max_action_std)
    if hasattr(self.actor, "min_sample_std"):
      self.actor.min_sample_std = self.min_action_std
    self.actor_mean_clip = None if actor_mean_clip is None else float(actor_mean_clip)
    self.ppo_log_ratio_clip = float(ppo_log_ratio_clip)
    self.idle_deterministic_actions = bool(idle_deterministic_actions)
    self.mask_idle_actor_loss = bool(mask_idle_actor_loss)
    self.critic_warmup_iterations = max(0, int(critic_warmup_iterations))
    self.freeze_actor_obs_normalization = bool(freeze_actor_obs_normalization)
    self._offline_bc_shards: list[Path] = []
    self._offline_bc_active_shards: list[Path] = []
    self._offline_bc_cache: dict[Path, dict[str, torch.Tensor]] = {}
    self._init_offline_bc_dataset(offline_bc_dataset)
    self._act_step = 0
    self._ppo_update_count = 0

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

    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.reset(dones)
    self.critic.reset(dones)

  def _load_teacher_policy(self, checkpoint_path: str | None, env: VecEnv | None):
    if not checkpoint_path or max(self.distill_coef, self.distill_final_coef) <= 0.0:
      return None
    if env is None:
      raise ValueError("GoalkeeperStudentPPO needs env to build the online teacher policy.")
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
      raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    from src.tasks.soccer.modules.moe7_prepare_actor import MoE7PrepareGoalkeeperActor

    loaded = torch.load(path, map_location=self.device, weights_only=False)
    bundle = loaded["actor_state_dict"] if isinstance(loaded, dict) and "actor_state_dict" in loaded and "sr" in loaded["actor_state_dict"] else loaded
    if not isinstance(bundle, dict) or "sr" not in bundle:
      raise ValueError(f"Teacher checkpoint is not a MoE goalkeeper bundle: {path}")
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
    return teacher

  def _load_teacher_route_gate(self, checkpoint_path: str | None):
    if not checkpoint_path:
      return None
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
      return None
    loaded = torch.load(path, map_location=self.device, weights_only=False)
    bundle = loaded["actor_state_dict"] if isinstance(loaded, dict) and "actor_state_dict" in loaded and "sr" in loaded["actor_state_dict"] else loaded
    if not isinstance(bundle, dict):
      return None
    gate = bundle.get("gate")
    if not isinstance(gate, dict) or not gate.get("state"):
      return None
    from src.tasks.soccer.modules.moe6_goalkeeper_policy import _make_gate_net

    num_classes = int(gate.get("num_classes", gate["state"]["4.weight"].shape[0]))
    net = _make_gate_net(num_classes, torch.device(self.device))
    net.load_state_dict(gate["state"])
    net.eval()
    for param in net.parameters():
      param.requires_grad_(False)
    return {
      "net": net,
      "mean": gate["mean"].to(self.device),
      "std": gate["std"].to(self.device).clamp_min(1.0e-6),
    }

  def train_mode(self) -> None:
    super().train_mode()
    teacher_policy = getattr(self, "teacher_policy", None)
    if teacher_policy is not None:
      teacher_policy.eval()

  def act(self, obs: TensorDict) -> torch.Tensor:
    self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
    self._clamp_actor_std()
    sampled_actions = self._clip_actor_actions(self.actor(obs, stochastic_output=True).detach())
    action_mean = self._clip_actor_actions(self.actor.output_mean.detach())
    idle_mask = self._idle_condition_mask(obs).unsqueeze(-1)
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
    if self.teacher_policy is not None and self._act_step % self.teacher_every_n_steps == 0:
      teacher_actions = self.teacher_policy(obs).detach()
    else:
      teacher_actions = action_mean
    self.transition.privileged_actions = teacher_actions
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
    return raw[..., -1] > 0.5

  def update(self) -> dict[str, float]:
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_distill_loss = 0.0
    mean_offline_bc_loss = 0.0
    mean_offline_idle_bc_loss = 0.0
    mean_condition_aux_loss = 0.0

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    else:
      generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

    self._prefetch_offline_bc_shards()
    update_index = 0
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
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
      distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        self._update_learning_rate(batch.old_distribution_params, distribution_params)

      ratio = self._ppo_ratio(actions_log_prob, torch.squeeze(batch.old_actions_log_prob))
      advantages = torch.squeeze(batch.advantages)
      clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
      surrogate_loss, entropy_loss = self._masked_actor_losses(
        batch.observations,
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

      distill_weight = self._scheduled_coef(
        self.distill_coef,
        self.distill_final_coef,
        self.distill_anneal_updates,
      )
      offline_bc_weight = self._scheduled_coef(
        self.offline_bc_coef,
        self.offline_bc_final_coef,
        self.offline_bc_anneal_updates,
      )
      offline_idle_bc_weight = self._scheduled_coef(
        self.offline_idle_bc_coef,
        self.offline_idle_bc_final_coef,
        self.offline_idle_bc_anneal_updates,
      )
      condition_aux_weight = self._scheduled_coef(
        self.condition_aux_coef,
        self.condition_aux_final_coef,
        self.condition_aux_anneal_updates,
      )
      distill_loss = self._distill_loss(batch) if distill_weight > 0.0 else torch.zeros((), device=self.device)
      offline_bc_loss = self._scheduled_offline_bc_loss(update_index, offline_bc_weight)
      offline_idle_bc_loss = (
        self._offline_idle_bc_loss() if offline_idle_bc_weight > 0.0 else torch.zeros((), device=self.device)
      )
      condition_aux_loss = (
        self._condition_aux_loss(batch) if condition_aux_weight > 0.0 else torch.zeros((), device=self.device)
      )
      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy_loss
        + distill_weight * distill_loss
        + offline_bc_weight * offline_bc_loss
        + offline_idle_bc_weight * offline_idle_bc_loss
        + condition_aux_weight * condition_aux_loss
      )
      if not torch.isfinite(loss):
        self._clamp_actor_std()
        update_index += 1
        continue

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
      mean_distill_loss += distill_loss.item()
      mean_offline_bc_loss += offline_bc_loss.item()
      mean_offline_idle_bc_loss += offline_idle_bc_loss.item()
      mean_condition_aux_loss += condition_aux_loss.item()
      update_index += 1

    num_updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    self._ppo_update_count += 1
    return {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "distill": mean_distill_loss / num_updates,
      "offline_bc": mean_offline_bc_loss / num_updates,
      "offline_idle_bc": mean_offline_idle_bc_loss / num_updates,
      "condition_aux": mean_condition_aux_loss / num_updates,
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

  def _ppo_ratio(self, actions_log_prob: torch.Tensor, old_actions_log_prob: torch.Tensor) -> torch.Tensor:
    log_ratio = actions_log_prob - old_actions_log_prob
    clip = max(float(getattr(self, "ppo_log_ratio_clip", 8.0)), 1.0e-6)
    log_ratio = torch.nan_to_num(log_ratio, nan=0.0, posinf=clip, neginf=-clip)
    return torch.exp(log_ratio.clamp(min=-clip, max=clip))

  def _masked_actor_losses(
    self,
    observations,
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

    active_mask = ~self._idle_condition_mask(observations)
    if active_mask.shape != per_sample_surrogate.shape:
      active_mask = active_mask.reshape(per_sample_surrogate.shape)
    active_mask = active_mask.to(device=per_sample_surrogate.device, dtype=torch.bool)
    if not torch.any(active_mask):
      zero = per_sample_surrogate.sum() * 0.0
      return zero, zero
    return per_sample_surrogate[active_mask].mean(), entropy[active_mask].mean()

  def _init_offline_bc_dataset(self, dataset_dir: str | None) -> None:
    self._offline_bc_shards = []
    self._offline_bc_active_shards = []
    self._offline_bc_cache = {}
    offline_idle_bc_coef = getattr(self, "offline_idle_bc_coef", 0.0)
    offline_bc_final_coef = getattr(self, "offline_bc_final_coef", self.offline_bc_coef)
    offline_idle_bc_final_coef = getattr(self, "offline_idle_bc_final_coef", offline_idle_bc_coef)
    if not dataset_dir or (
      max(self.offline_bc_coef, offline_bc_final_coef) <= 0.0
      and max(offline_idle_bc_coef, offline_idle_bc_final_coef) <= 0.0
    ):
      return
    root = Path(dataset_dir).expanduser().resolve()
    shard_dir = root / "shards" if (root / "shards").is_dir() else root
    shards = sorted(shard_dir.glob("shard_*.pt"))
    if not shards:
      raise FileNotFoundError(f"No offline BC shard_*.pt files under {shard_dir}")
    self._offline_bc_shards = shards
    self._offline_bc_active_shards = shards[: min(len(shards), self.offline_bc_cache_shards)]

  def _prefetch_offline_bc_shards(self) -> None:
    offline_idle_bc_coef = getattr(self, "offline_idle_bc_coef", 0.0)
    offline_bc_final_coef = getattr(self, "offline_bc_final_coef", self.offline_bc_coef)
    offline_idle_bc_final_coef = getattr(self, "offline_idle_bc_final_coef", offline_idle_bc_coef)
    if (
      max(self.offline_bc_coef, offline_bc_final_coef) <= 0.0
      and max(offline_idle_bc_coef, offline_idle_bc_final_coef) <= 0.0
    ) or not self._offline_bc_shards:
      self._offline_bc_active_shards = []
      self._offline_bc_cache = {}
      return
    count = min(self.offline_bc_cache_shards, len(self._offline_bc_shards))
    perm = torch.randperm(len(self._offline_bc_shards), device="cpu")[:count].tolist()
    self._offline_bc_active_shards = [self._offline_bc_shards[i] for i in perm]
    self._offline_bc_cache = {}
    for shard in self._offline_bc_active_shards:
      self._load_offline_bc_shard(shard)

  def _scheduled_coef(self, start: float, final: float | None, anneal_updates: int) -> float:
    final_value = start if final is None else final
    if anneal_updates <= 0:
      return float(start)
    progress = min(max(float(getattr(self, "_ppo_update_count", 0)) / float(anneal_updates), 0.0), 1.0)
    return float(start + (final_value - start) * progress)

  def _scheduled_offline_bc_loss(self, update_index: int, coef: float | None = None) -> torch.Tensor:
    effective_coef = self.offline_bc_coef if coef is None else coef
    if (
      effective_coef <= 0.0
      or not self._offline_bc_shards
      or update_index % self.offline_bc_every_n_updates != 0
    ):
      return torch.zeros((), device=self.device)
    return self._offline_bc_loss()

  def _load_offline_bc_shard(self, shard: Path) -> dict[str, torch.Tensor]:
    cached = self._offline_bc_cache.get(shard)
    if cached is not None:
      return cached
    payload = torch.load(shard, map_location="cpu", weights_only=False)
    data = {
      "student_obs": payload["student_obs"].float(),
      "teacher_action": payload["teacher_action"].float(),
      "valid_mask": payload["valid_mask"].bool(),
    }
    self._offline_bc_cache[shard] = data
    return data

  def _sample_offline_bc_batch(self, idle_only: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not self._offline_bc_shards:
      raise RuntimeError("Offline BC dataset is not initialized.")
    sample_shards = self._offline_bc_active_shards or self._offline_bc_shards
    obs_chunks: list[torch.Tensor] = []
    action_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    attempts = 0
    while len(obs_chunks) < self.offline_bc_batch_size:
      attempts += 1
      if attempts > self.offline_bc_batch_size * 20:
        raise RuntimeError("Unable to sample enough valid offline BC sequences.")
      shard = sample_shards[
        torch.randint(len(sample_shards), (1,), device="cpu").item()
      ]
      data = self._load_offline_bc_shard(shard)
      num_eps, max_t = data["student_obs"].shape[:2]
      if num_eps == 0 or max_t == 0:
        continue
      ep_idx = torch.randint(num_eps, (1,), device="cpu").item()
      valid = data["valid_mask"][ep_idx]
      active_fraction = getattr(self, "offline_bc_active_fraction", 0.0)
      active_only = (
        (not idle_only)
        and active_fraction > 0.0
        and torch.rand((), device="cpu").item() < active_fraction
      )
      if idle_only or active_only:
        idle = data["student_obs"][ep_idx, :, -1] > 0.5
        valid_indices_all = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_indices_all.numel() <= 0:
          continue
        flags = idle[valid_indices_all]
        if idle_only:
          inactive = torch.nonzero(~flags, as_tuple=False).flatten()
          prefix_len = int(inactive[0].item()) if inactive.numel() > 0 else int(flags.numel())
          if prefix_len <= 0:
            continue
          valid_indices = valid_indices_all[:prefix_len]
        else:
          valid_indices = valid_indices_all[~flags]
        if valid_indices.numel() <= 0:
          continue
        seq_len = min(self.offline_bc_seq_len, int(valid_indices.numel()))
        start_high = max(int(valid_indices.numel()) - seq_len + 1, 1)
        start = torch.randint(start_high, (1,), device="cpu").item()
        indices = valid_indices[start : start + seq_len]
        obs = torch.zeros(self.offline_bc_seq_len, data["student_obs"].shape[-1])
        actions = torch.zeros(self.offline_bc_seq_len, data["teacher_action"].shape[-1])
        masks = torch.zeros(self.offline_bc_seq_len, dtype=torch.bool)
        obs[:seq_len] = data["student_obs"][ep_idx, indices]
        actions[:seq_len] = data["teacher_action"][ep_idx, indices]
        masks[:seq_len] = True
        obs_chunks.append(obs)
        action_chunks.append(actions)
        mask_chunks.append(masks)
        continue
      valid_len = int(valid.sum().item())
      if valid_len <= 0:
        continue
      seq_len = min(self.offline_bc_seq_len, valid_len)
      start_high = max(valid_len - seq_len + 1, 1)
      start = torch.randint(start_high, (1,), device="cpu").item()
      end = start + seq_len
      obs = torch.zeros(self.offline_bc_seq_len, data["student_obs"].shape[-1])
      actions = torch.zeros(self.offline_bc_seq_len, data["teacher_action"].shape[-1])
      masks = torch.zeros(self.offline_bc_seq_len, dtype=torch.bool)
      obs[:seq_len] = data["student_obs"][ep_idx, start:end]
      actions[:seq_len] = data["teacher_action"][ep_idx, start:end]
      masks[:seq_len] = True
      obs_chunks.append(obs)
      action_chunks.append(actions)
      mask_chunks.append(masks)
    obs_batch = torch.stack(obs_chunks, dim=1).to(self.device)
    action_batch = torch.stack(action_chunks, dim=1).to(self.device)
    mask_batch = torch.stack(mask_chunks, dim=1).to(self.device)
    return obs_batch, action_batch, mask_batch

  def _offline_bc_loss(self) -> torch.Tensor:
    if not self._offline_bc_shards:
      return torch.zeros((), device=self.device)
    obs, teacher_actions, masks = self._sample_offline_bc_batch()
    return self._bc_loss_for_batch(obs, teacher_actions, masks)

  def _offline_idle_bc_loss(self) -> torch.Tensor:
    if not self._offline_bc_shards:
      return torch.zeros((), device=self.device)
    obs, teacher_actions, masks = self._sample_offline_bc_batch(idle_only=True)
    return self._bc_loss_for_batch(obs, teacher_actions, masks)

  def _bc_loss_for_batch(
    self,
    obs: torch.Tensor,
    teacher_actions: torch.Tensor,
    masks: torch.Tensor,
  ) -> torch.Tensor:
    if hasattr(self.actor, "forward_bc_chunk"):
      pred, _ = self.actor.forward_bc_chunk(obs, masks, None)
    else:
      pred = self.actor(TensorDict({"student": obs}, batch_size=list(obs.shape[:-1])), masks=masks)
    target = teacher_actions[masks]
    return self.loss_fn(pred, target.detach())

  def _distill_loss(self, batch: RolloutStorage.Batch) -> torch.Tensor:
    if batch.privileged_actions is None:
      return torch.zeros((), device=self.device)
    pred_mean = self._actor_output_mean()
    teacher_actions = batch.privileged_actions
    if batch.masks is not None:
      teacher_actions = unpad_trajectories(teacher_actions, batch.masks)
    return self.loss_fn(pred_mean, teacher_actions.detach())

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
    gate = getattr(self, "teacher_route_gate", None)
    if isinstance(gate, dict):
      return self._route_targets_from_student_obs(
        student_obs,
        gate=gate["net"],
        gate_mean=gate["mean"],
        gate_std=gate["std"],
      )
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
    condition_idle = student_obs[..., -1] > 0.5
    kinematic_idle = (speed < idle_speed_threshold) | (vx >= idle_incoming_vx_threshold)
    idle = condition_idle | kinematic_idle
    return torch.where(idle, torch.full_like(route, 6), route)

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
