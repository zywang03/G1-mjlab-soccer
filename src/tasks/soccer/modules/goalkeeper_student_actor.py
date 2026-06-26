"""LSTM goalkeeper student actor with phase-aware task conditioning."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import EmpiricalNormalization, MLP
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.modules.rnn import RNN
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

from src.tasks.soccer.mdp.goalkeeper_student_obs import (
  GOALKEEPER_TASK_CONTEXT_DIM,
  GOALKEEPER_TEACHER_ACTOR_OBS_DIM,
)


class GoalkeeperStudentFiLMActor(nn.Module):
  """Recurrent student actor conditioned by explicit goalkeeper task context.

  The LSTM only sees the normal 960D goalkeeper actor observation history.
  A separate task context controls phase-specific prepare/active action heads.
  """

  is_recurrent = True

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (128, 64, 32),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict[str, Any] | None = None,
    rnn_type: str = "lstm",
    rnn_hidden_dim: int = 128,
    rnn_num_layers: int = 2,
    condition_hidden_dim: int = 32,
    ball_latent_dim: int = 6,
    min_sample_std: float | None = None,
    **kwargs,
  ) -> None:
    if kwargs:
      print(
        "GoalkeeperStudentFiLMActor.__init__ got unexpected arguments: "
        + str([key for key in kwargs.keys()])
      )
    super().__init__()
    self.obs_groups = obs_groups[obs_set]
    self.obs_dim = sum(int(obs[group].shape[-1]) for group in self.obs_groups)
    expected_dim = GOALKEEPER_TEACHER_ACTOR_OBS_DIM + GOALKEEPER_TASK_CONTEXT_DIM
    if self.obs_dim != expected_dim:
      raise ValueError(f"Goalkeeper student obs dim must be {expected_dim}, got {self.obs_dim}")

    self.obs_normalization = obs_normalization
    self.ball_latent_dim = int(ball_latent_dim)
    self.min_sample_std = None if min_sample_std is None else float(min_sample_std)
    self.num_route_classes = 7
    self.obs_normalizer = (
      EmpiricalNormalization(GOALKEEPER_TEACHER_ACTOR_OBS_DIM)
      if obs_normalization
      else nn.Identity()
    )
    self.rnn = RNN(
      GOALKEEPER_TEACHER_ACTOR_OBS_DIM,
      rnn_hidden_dim,
      rnn_num_layers,
      rnn_type,
    )
    self.phase_encoder = nn.Sequential(
      nn.Linear(4, condition_hidden_dim),
      nn.ELU(),
      nn.Linear(condition_hidden_dim, condition_hidden_dim),
      nn.ELU(),
    )
    self.ball_encoder = nn.Sequential(
      nn.Linear(6, condition_hidden_dim),
      nn.ELU(),
      nn.Linear(condition_hidden_dim, condition_hidden_dim),
      nn.ELU(),
    )
    self.target_encoder = nn.Sequential(
      nn.Linear(9, condition_hidden_dim),
      nn.ELU(),
      nn.Linear(condition_hidden_dim, condition_hidden_dim),
      nn.ELU(),
    )
    self.context_fusion = nn.Sequential(
      nn.Linear(rnn_hidden_dim + 3 * condition_hidden_dim, rnn_hidden_dim),
      nn.ELU(),
      nn.Linear(rnn_hidden_dim, rnn_hidden_dim),
      nn.ELU(),
    )
    self.region_estimator = nn.Sequential(
      nn.Linear(rnn_hidden_dim, condition_hidden_dim),
      nn.ELU(),
      nn.Linear(condition_hidden_dim, self.num_route_classes),
    )
    self.ball_estimator = nn.Sequential(
      nn.Linear(rnn_hidden_dim, condition_hidden_dim),
      nn.ELU(),
      nn.Linear(condition_hidden_dim, self.ball_latent_dim),
    )
    self.condition_aux_head = nn.Sequential(
      nn.Linear(rnn_hidden_dim, condition_hidden_dim),
      nn.ELU(),
      nn.Linear(condition_hidden_dim, 8),
    )
    self.prepare_head = MLP(rnn_hidden_dim, output_dim, hidden_dims, activation)
    self.active_head = MLP(rnn_hidden_dim, output_dim, hidden_dims, activation)

    distribution_cfg = dict(distribution_cfg or {
      "class_name": "GaussianDistribution",
      "init_std": 0.5,
      "std_type": "scalar",
    })
    distribution_cfg.pop("class_name", None)
    self.distribution = GaussianDistribution(output_dim, **distribution_cfg)
    self.distribution.init_mlp_weights(self.prepare_head)
    self.distribution.init_mlp_weights(self.active_head)
    self._last_condition_aux: torch.Tensor | None = None
    self._last_region_aux: dict[str, torch.Tensor] | None = None
    self.condition_aux_enabled: bool = True

  def _split_student_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
      obs[..., :GOALKEEPER_TEACHER_ACTOR_OBS_DIM],
      obs[..., GOALKEEPER_TEACHER_ACTOR_OBS_DIM:],
    )

  def _raw_student_obs(self, obs: TensorDict) -> torch.Tensor:
    return torch.cat([obs[group] for group in self.obs_groups], dim=-1)

  def _encode_task_context(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    phase_features = context[..., :4]
    ball_features = context[..., 4:10]
    target_features = context[..., 10:19]
    encoded = torch.cat([
      self.phase_encoder(phase_features),
      self.ball_encoder(ball_features),
      self.target_encoder(target_features),
    ], dim=-1)
    active_blend = context[..., 0:1].clamp(0.0, 1.0)
    return encoded, active_blend

  def _contextual_latent(self, actor_latent: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    context_latent, active_blend = self._encode_task_context(context)
    latent = self.context_fusion(torch.cat([actor_latent, context_latent], dim=-1))
    region_logits = self.region_estimator(latent)
    ball_latent = self.ball_estimator(latent)
    self._last_region_aux = {
      "region_logits": region_logits,
      "ball_latent": ball_latent,
    }
    return latent, active_blend

  def _action_head_output(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    _, active_blend = self._encode_task_context(context)
    prepare_action = self.prepare_head(latent)
    active_action = self.active_head(latent)
    return prepare_action * (1.0 - active_blend) + active_action * active_blend

  @staticmethod
  def _split_condition_aux(aux: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"landing_yz": aux[..., :2], "region_logits": aux[..., 2:]}

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    raw = self._raw_student_obs(obs)
    actor_obs, condition = self._split_student_obs(raw)
    actor_obs = self.obs_normalizer(actor_obs)
    base_latent = self.rnn(actor_obs, masks, hidden_state).squeeze(0)
    if masks is not None:
      condition = unpad_trajectories(condition, masks)
    latent, _ = self._contextual_latent(base_latent, condition)
    return latent

  def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
    raw = self._raw_student_obs(obs)
    actor_obs, condition = self._split_student_obs(raw)
    actor_obs = self.obs_normalizer(actor_obs)
    base_latent = self.rnn(actor_obs, masks, hidden_state).squeeze(0)
    if masks is not None:
      condition = unpad_trajectories(condition, masks)
    latent, active_blend = self._contextual_latent(base_latent, condition)
    if self.condition_aux_enabled:
      self._last_condition_aux = self.condition_aux_head(latent)
    prepare_action = self.prepare_head(latent)
    active_action = self.active_head(latent)
    mlp_output = prepare_action * (1.0 - active_blend) + active_action * active_blend
    if stochastic_output:
      self._clamp_distribution_std_param()
      self.distribution.update(mlp_output)
      return self.distribution.sample()
    return self.distribution.deterministic_output(mlp_output)

  def forward_bc_chunk(
    self,
    obs_chunk: torch.Tensor,
    masks_chunk: torch.Tensor,
    hidden_state=None,
  ) -> tuple[torch.Tensor, Any]:
    actor_obs, condition = self._split_student_obs(obs_chunk)
    if self.obs_normalization:
      actor_obs = self.obs_normalizer(actor_obs)
    actor_obs = actor_obs * masks_chunk.unsqueeze(-1).float()
    condition = condition * masks_chunk.unsqueeze(-1).float()
    rnn_out, new_hs = self.rnn.rnn(actor_obs, hidden_state)
    base_latent = rnn_out[masks_chunk]
    condition = condition[masks_chunk]
    latent, active_blend = self._contextual_latent(base_latent, condition)
    if self.condition_aux_enabled:
      self._last_condition_aux = self.condition_aux_head(latent)
    prepare_action = self.prepare_head(latent)
    active_action = self.active_head(latent)
    mlp_output = prepare_action * (1.0 - active_blend) + active_action * active_blend
    return self.distribution.deterministic_output(mlp_output), self._detach_hidden_state(new_hs)

  def predict_condition_aux(self, obs: TensorDict, masks=None, hidden_state=None) -> dict[str, torch.Tensor]:
    latent = self.get_latent(obs, masks, hidden_state)
    return self._split_condition_aux(self.condition_aux_head(latent))

  @property
  def condition_aux_output(self) -> dict[str, torch.Tensor] | None:
    if self._last_region_aux is not None:
      return self._last_region_aux
    if self._last_condition_aux is None:
      return None
    return self._split_condition_aux(self._last_condition_aux)

  @property
  def region_condition_output(self) -> dict[str, torch.Tensor] | None:
    return self._last_region_aux

  @staticmethod
  def _detach_hidden_state(hidden_state):
    if isinstance(hidden_state, tuple):
      return tuple(h.detach() for h in hidden_state)
    return hidden_state.detach()

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      actor_obs, _ = self._split_student_obs(self._raw_student_obs(obs))
      self.obs_normalizer.update(actor_obs)

  def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
    self.rnn.reset(dones, hidden_state)

  def get_hidden_state(self):
    return self.rnn.hidden_state

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    self.rnn.detach_hidden_state(dones)

  @property
  def output_mean(self) -> torch.Tensor:
    return self.distribution.mean

  @property
  def output_std(self) -> torch.Tensor:
    return self.distribution.std

  @property
  def output_entropy(self) -> torch.Tensor:
    return self.distribution.entropy

  @property
  def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
    return self.distribution.params

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    return self.distribution.log_prob(outputs)

  def get_kl_divergence(self, old_params, new_params) -> torch.Tensor:
    return self.distribution.kl_divergence(old_params, new_params)

  def _clamp_distribution_std_param(self) -> None:
    min_std = getattr(self, "min_sample_std", None)
    if min_std is None or min_std <= 0.0:
      return
    std_param = getattr(self.distribution, "std_param", None)
    if std_param is None:
      return
    with torch.no_grad():
      std_param.nan_to_num_(nan=min_std, posinf=min_std, neginf=min_std)
      std_param.clamp_(min=min_std)
