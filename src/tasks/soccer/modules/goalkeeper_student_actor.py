"""LSTM goalkeeper student actor with FiLM prediction conditioning."""

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
  GOALKEEPER_PREDICTION_CONDITION_DIM,
  GOALKEEPER_TEACHER_ACTOR_OBS_DIM,
)


class GoalkeeperStudentFiLMActor(nn.Module):
  """Recurrent student actor conditioned by predicted ball crossing features.

  The LSTM only sees the normal 960D goalkeeper actor observation history.  The
  4D prediction condition ``[pred_y, pred_z, time_norm, idle]`` is encoded
  separately and used to FiLM-modulate the LSTM latent before the action head.
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
    expected_dim = GOALKEEPER_TEACHER_ACTOR_OBS_DIM + GOALKEEPER_PREDICTION_CONDITION_DIM
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
    self.condition_encoder = nn.Sequential(
      nn.Linear(GOALKEEPER_PREDICTION_CONDITION_DIM, condition_hidden_dim),
      nn.ELU(),
      nn.Linear(condition_hidden_dim, condition_hidden_dim),
      nn.ELU(),
    )
    self.film = nn.Linear(condition_hidden_dim, 2 * rnn_hidden_dim)
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
    self.region_condition_encoder = nn.Sequential(
      nn.Linear(self.num_route_classes + self.ball_latent_dim, condition_hidden_dim),
      nn.ELU(),
      nn.Linear(condition_hidden_dim, condition_hidden_dim),
      nn.ELU(),
    )
    self.region_film = nn.Linear(condition_hidden_dim, 2 * rnn_hidden_dim)
    self.prepare_region_film = nn.Linear(condition_hidden_dim, 2 * rnn_hidden_dim)
    self.condition_aux_head = nn.Sequential(
      nn.Linear(rnn_hidden_dim, condition_hidden_dim),
      nn.ELU(),
      nn.Linear(condition_hidden_dim, 8),
    )
    self.mlp = MLP(rnn_hidden_dim, output_dim, hidden_dims, activation)

    distribution_cfg = dict(distribution_cfg or {
      "class_name": "GaussianDistribution",
      "init_std": 0.5,
      "std_type": "scalar",
    })
    distribution_cfg.pop("class_name", None)
    self.distribution = GaussianDistribution(output_dim, **distribution_cfg)
    self.distribution.init_mlp_weights(self.mlp)
    self._last_condition_aux: torch.Tensor | None = None
    self._last_region_aux: dict[str, torch.Tensor] | None = None
    self._init_identity_film()

  def _init_identity_film(self) -> None:
    for film in (self.film, self.region_film, self.prepare_region_film):
      nn.init.zeros_(film.weight)
      nn.init.zeros_(film.bias)

  def _split_student_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
      obs[..., :GOALKEEPER_TEACHER_ACTOR_OBS_DIM],
      obs[..., GOALKEEPER_TEACHER_ACTOR_OBS_DIM:],
    )

  def _raw_student_obs(self, obs: TensorDict) -> torch.Tensor:
    return torch.cat([obs[group] for group in self.obs_groups], dim=-1)

  def _conditioned_latent(self, actor_latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    encoded = self.condition_encoder(condition)
    gamma_beta = self.film(encoded)
    gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
    return (1.0 + gamma) * actor_latent + beta

  def _route_conditioned_latent(
    self,
    latent: torch.Tensor,
    condition: torch.Tensor,
    route_source_latent: torch.Tensor,
  ) -> torch.Tensor:
    region_logits = self.region_estimator(route_source_latent)
    ball_latent = self.ball_estimator(route_source_latent)
    route_probs = torch.softmax(region_logits, dim=-1)
    route_features = torch.cat([route_probs, ball_latent], dim=-1)
    encoded = self.region_condition_encoder(route_features)
    gamma_beta = self.region_film(encoded)
    prepare_gamma_beta = self.prepare_region_film(encoded)
    idle = condition[..., -1:].to(dtype=latent.dtype) > 0.5
    gamma_beta = torch.where(idle, prepare_gamma_beta, gamma_beta)
    gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
    self._last_region_aux = {
      "region_logits": region_logits,
      "ball_latent": ball_latent,
    }
    return (1.0 + gamma) * latent + beta

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
    latent = self._conditioned_latent(base_latent, condition)
    return self._route_conditioned_latent(latent, condition, base_latent)

  def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
    latent = self.get_latent(obs, masks, hidden_state)
    self._last_condition_aux = self.condition_aux_head(latent)
    mlp_output = self.mlp(latent)
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
    latent = self._conditioned_latent(base_latent, condition)
    latent = self._route_conditioned_latent(latent, condition, base_latent)
    self._last_condition_aux = self.condition_aux_head(latent)
    mlp_output = self.mlp(latent)
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
