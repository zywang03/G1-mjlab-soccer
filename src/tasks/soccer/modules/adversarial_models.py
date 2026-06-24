"""Zero-initialized adversarial policy model extensions."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import RNNModel
from rsl_rl.modules import EmpiricalNormalization, MLP
from rsl_rl.modules.rnn import RNN
from rsl_rl.utils import unpad_trajectories

from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic


class AdversarialRNNModel(RNNModel):
  """RNN model that can append opponent features without changing old behavior."""

  def __init__(self, *args, base_obs_dim: int = 160, residual_hidden_dims=(128, 64), residual_scale=1.0, **kwargs):
    obs = args[0]
    obs_groups = args[1]
    obs_set = args[2]
    output_dim = args[3] if len(args) > 3 else kwargs.get("output_dim")
    self.base_obs_dim = base_obs_dim
    self.raw_obs_dim = self._infer_obs_dim(obs, obs_groups, obs_set)
    self.extra_obs_dim = max(0, self.raw_obs_dim - self.base_obs_dim)
    self.residual_scale = residual_scale
    obs_normalization = bool(kwargs.get("obs_normalization", False))
    super().__init__(*args, **kwargs)
    self.obs_dim = self.base_obs_dim
    self.obs_normalizer = EmpiricalNormalization(self.base_obs_dim) if obs_normalization else nn.Identity()
    self.rnn = RNN(
      self.base_obs_dim,
      kwargs.get("rnn_hidden_dim", 256),
      kwargs.get("rnn_num_layers", 1),
      kwargs.get("rnn_type", "lstm"),
    )
    self.output_residual = self._make_zero_residual(self.extra_obs_dim, int(output_dim), residual_hidden_dims)

  @staticmethod
  def _infer_obs_dim(obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> int:
    return sum(int(obs[group].shape[-1]) for group in obs_groups[obs_set])

  @staticmethod
  def _make_zero_residual(input_dim: int, output_dim: int, hidden_dims) -> nn.Module:
    if input_dim <= 0:
      return nn.Identity()
    residual = MLP(input_dim, output_dim, hidden_dims, "elu")
    for module in reversed(list(residual.modules())):
      if isinstance(module, nn.Linear):
        nn.init.zeros_(module.weight)
        nn.init.zeros_(module.bias)
        break
    return residual

  def _raw_base_obs(self, obs: TensorDict) -> torch.Tensor:
    latent = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
    return latent[..., : self.base_obs_dim]

  def _raw_extra_obs(self, obs: TensorDict, masks=None) -> torch.Tensor | None:
    latent = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
    extra = latent[..., self.base_obs_dim :]
    if masks is not None:
      extra = unpad_trajectories(extra, masks)
    return extra if extra.numel() else None

  def _base_latent(self, obs: TensorDict) -> torch.Tensor:
    return self.obs_normalizer(self._raw_base_obs(obs))

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    return self.rnn(self._base_latent(obs), masks, hidden_state).squeeze(0)

  def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
    out = super().forward(obs, masks, hidden_state, stochastic_output)
    extra = self._raw_extra_obs(obs, masks)
    if extra is not None and self.extra_obs_dim > 0:
      out = out + self.residual_scale * self.output_residual(extra)
    return out

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      self.obs_normalizer.update(self._raw_base_obs(obs))  # type: ignore[attr-defined]

  def load_base_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
    self.load_state_dict({
      key: value
      for key, value in state_dict.items()
      if key in self.state_dict() and self.state_dict()[key].shape == value.shape
    }, strict=False)


class AdversarialGoalkeeperActorCritic(GoalkeeperActorCritic):
  """Goalkeeper actor-critic with zero-initialized opponent residual heads."""

  _BASE_ACTOR_OBS_DIM = 960
  _BASE_CRITIC_OBS_DIM = 113

  def __init__(self, *args, residual_hidden_dims=(128, 64), residual_scale=1.0, **kwargs):
    obs = args[0]
    self._raw_actor_obs_dim = self._infer_group_dim(obs, "actor")
    self._raw_critic_obs_dim = self._infer_group_dim(obs, "critic")
    super().__init__(
      *args,
      num_one_step_obs=96,
      num_critic_obs=self._BASE_CRITIC_OBS_DIM,
      num_actor_obs=self._BASE_ACTOR_OBS_DIM,
      **kwargs,
    )
    self.residual_scale = residual_scale
    self.actor_extra_dim = max(0, self._raw_actor_obs_dim - self._BASE_ACTOR_OBS_DIM)
    self.critic_extra_dim = max(0, self._raw_critic_obs_dim - self._BASE_CRITIC_OBS_DIM)

    self.actor_residual = self._make_zero_residual(self.actor_extra_dim, self.num_actions, residual_hidden_dims)
    self.critic_residual = self._make_zero_residual(self.critic_extra_dim, 1, residual_hidden_dims)

  @staticmethod
  def _infer_group_dim(obs, group: str) -> int:
    if hasattr(obs, "spaces"):
      space = obs.spaces.get(group)
      if space is not None and hasattr(space, "shape"):
        return int(space.shape[0])
    if isinstance(obs, dict) or hasattr(obs, "get"):
      tensor = obs.get(group)
      if tensor is not None and hasattr(tensor, "shape"):
        return int(tensor.shape[-1])
    return 0

  @staticmethod
  def _make_zero_residual(input_dim: int, output_dim: int, hidden_dims) -> nn.Module:
    if input_dim <= 0:
      return nn.Identity()
    residual = MLP(input_dim, output_dim, hidden_dims, "elu")
    for module in reversed(list(residual.modules())):
      if isinstance(module, nn.Linear):
        nn.init.zeros_(module.weight)
        nn.init.zeros_(module.bias)
        break
    return residual

  def load_base_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
    own = self.state_dict()
    compatible = {
      key: value
      for key, value in state_dict.items()
      if key in own and own[key].shape == value.shape
    }
    self.load_state_dict(compatible, strict=False)

  def _split_actor_obs(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    base = obs_history[:, : self._BASE_ACTOR_OBS_DIM]
    extra = obs_history[:, self._BASE_ACTOR_OBS_DIM :]
    return base, extra if extra.numel() else None

  def _split_critic_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    base = obs[:, : self._BASE_CRITIC_OBS_DIM]
    extra = obs[:, self._BASE_CRITIC_OBS_DIM :]
    return base, extra if extra.numel() else None

  def _actor_mean(self, obs_history: torch.Tensor) -> torch.Tensor:
    base_obs, extra = self._split_actor_obs(obs_history)
    base_obs = self._reorder_obs_history(base_obs)
    history_latent = self.history_encoder(base_obs)
    estimate_ball = self.ball_estimator(base_obs)
    estimate_region = self.region_estimator(base_obs)
    actor_input = torch.cat(
      (
        base_obs[:, -self.num_one_step_obs :],
        history_latent,
        estimate_ball,
        torch.argmax(estimate_region, dim=-1, keepdim=True),
      ),
      dim=-1,
    )
    mean = self.actor(actor_input)
    if extra is not None and self.actor_extra_dim > 0:
      mean = mean + self.residual_scale * self.actor_residual(extra)
    self.estimate_ball = estimate_ball
    self.estimate_region = estimate_region
    return mean

  def update_distribution(self, obs_history):
    action_mean = self._actor_mean(obs_history)
    self.distribution = torch.distributions.Normal(action_mean, action_mean * 0.0 + self.std)

  def act_inference(self, obs_history, observations=None):
    return self._actor_mean(obs_history)

  def evaluate(self, critic_observations, **kwargs):
    x = self._extract_tensor(critic_observations, group="critic")
    base, extra = self._split_critic_obs(x)
    value = self.critic(base)
    if extra is not None and self.critic_extra_dim > 0:
      value = value + self.residual_scale * self.critic_residual(extra)
    return value
