"""Goalkeeper ActorCritic — matches Humanoid-Goalkeeper paper architecture.

HIMPPO-style ActorCritic matching the official goalkeeper model.
The model takes 960D history-stacked actor observations and produces:
  - 29D action means (via history encoder + ball/region estimators → actor MLP)
  - 1D value (critic MLP on 113D privileged obs)

Architecture:
  history_encoder: 960 → 128 → 64 → 16  (ReLU)
  ball_estimator:  960 → 128 → 32 → 6   (ReLU)
  region_estimator: 960 → 128 → 32 → 6  (ReLU)
  actor:  119 → 512 → 256 → 256 → 29    (ELU)
  critic: 113 → 512 → 256 → 256 → 1     (ELU)
  std: (29,) learnable scalar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _get_activation(act_name: str):
  if act_name == "elu":
    return nn.ELU()
  elif act_name == "relu":
    return nn.ReLU()
  elif act_name == "selu":
    return nn.SELU()
  elif act_name == "lrelu":
    return nn.LeakyReLU()
  elif act_name == "tanh":
    return nn.Tanh()
  elif act_name == "sigmoid":
    return nn.Sigmoid()
  else:
    return nn.ELU()


class _GoalkeeperGaussianDistribution(nn.Module):
  """RSL-RL-compatible diagonal Gaussian distribution for checkpoint keys."""

  def __init__(
    self,
    output_dim: int,
    init_std: float = 1.0,
    min_std: float | None = None,
    max_std: float | None = None,
  ):
    super().__init__()
    self.std_param = nn.Parameter(init_std * torch.ones(output_dim))
    self.min_std = min_std
    self.max_std = max_std
    self._distribution = None
    Normal.set_default_validate_args(False)

  def update(self, mean: torch.Tensor):
    std = self.std_param
    if self.min_std is not None or self.max_std is not None:
      std = torch.clamp(std, min=self.min_std, max=self.max_std)
    std = std.expand_as(mean)
    self._distribution = Normal(mean, std)

  def sample(self) -> torch.Tensor:
    return self._distribution.sample()

  @property
  def mean(self) -> torch.Tensor:
    return self._distribution.mean

  @property
  def std(self) -> torch.Tensor:
    return self._distribution.stddev

  @property
  def entropy(self) -> torch.Tensor:
    return self._distribution.entropy().sum(dim=-1)

  @property
  def params(self) -> tuple[torch.Tensor, torch.Tensor]:
    return (self.mean, self.std)

  def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    return self._distribution.log_prob(outputs).sum(dim=-1)

  def kl_divergence(self, old_params, new_params) -> torch.Tensor:
    old_mean, old_std = old_params
    new_mean, new_std = new_params
    old_dist = Normal(old_mean, old_std)
    new_dist = Normal(new_mean, new_std)
    return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)


class GoalkeeperActorCritic(nn.Module):
  """Actor-Critic with history encoder, ball/region estimators.

  Designed to load reference Humanoid-Goalkeeper checkpoints directly.
  Compatible with RSL-RL's model interface (MLPModel-like).

  Observation history reordering:
    mjlab's ObservationManager stacks history as term-major:
      [ball_f0..ball_f9, ang_f0..ang_f9, ..., actions_f0..actions_f9]
    The actor MLP consumes frame-major history:
      [f0(96D), f1(96D), ..., f9(96D)]
    We transpose on-the-fly so the model always receives frame-major input.
  """

  is_recurrent = False

  # Term sizes and history length — used by _reorder_obs_history.
  _TERM_SIZES: tuple[int, ...] = (3, 3, 3, 29, 29, 29)
  _HISTORY_LEN: int = 10
  _ONE_STEP_DIM: int = sum(_TERM_SIZES)  # 96

  def __init__(
    self,
    obs,
    obs_groups=None,
    group_name="actor",
    num_actions=29,
    num_one_step_obs=96,
    num_critic_obs=113,
    num_actor_obs=960,
    actor_history_length=10,
    hidden_dims=None,
    actor_hidden_dims=(512, 256, 256),
    critic_hidden_dims=(512, 256, 256),
    activation="elu",
    obs_normalization=False,
    distribution_cfg=None,
    init_noise_std=1.0,
    verbose=False,
    **kwargs,
  ):
    if kwargs:
      print(
        "GoalkeeperActorCritic.__init__ got unexpected arguments: "
        + str([key for key in kwargs.keys()])
      )
    super().__init__()
    self.group_name = group_name
    self.obs_normalization = obs_normalization

    if hidden_dims is not None:
      if group_name == "critic":
        critic_hidden_dims = tuple(hidden_dims)
      else:
        actor_hidden_dims = tuple(hidden_dims)
    min_noise_std = None
    max_noise_std = None
    if distribution_cfg is not None:
      init_noise_std = distribution_cfg.get("init_std", init_noise_std)
      min_noise_std = distribution_cfg.get("min_std", None)
      max_noise_std = distribution_cfg.get("max_std", None)

    # Try to infer dimensions from obs space if passed.
    def _feature_dim(space_or_tensor, fallback: int) -> int:
      shape = getattr(space_or_tensor, "shape", None)
      if shape is None or len(shape) == 0:
        return fallback
      return int(shape[-1])

    if hasattr(obs, "spaces"):
      actor_space = obs.spaces.get("actor")
      critic_space = obs.spaces.get("critic")
      if actor_space is not None and hasattr(actor_space, "shape"):
        num_actor_obs = _feature_dim(actor_space, num_actor_obs)
        num_one_step_obs = num_actor_obs // actor_history_length
      if critic_space is not None and hasattr(critic_space, "shape"):
        num_critic_obs = _feature_dim(critic_space, num_critic_obs)
    elif isinstance(obs, dict):
      actor_space = obs.get("actor")
      critic_space = obs.get("critic")
      if actor_space is not None and hasattr(actor_space, "shape"):
        num_actor_obs = _feature_dim(actor_space, num_actor_obs)
        num_one_step_obs = num_actor_obs // actor_history_length
      if critic_space is not None and hasattr(critic_space, "shape"):
        num_critic_obs = _feature_dim(critic_space, num_critic_obs)

    self.num_actor_obs = num_actor_obs
    self.num_critic_obs = num_critic_obs
    self.num_one_step_obs = num_one_step_obs
    self.actor_history_length = actor_history_length
    self.num_actions = num_actions
    if (
      num_one_step_obs != self._ONE_STEP_DIM
      or actor_history_length != self._HISTORY_LEN
      or num_actor_obs != self._ONE_STEP_DIM * self._HISTORY_LEN
    ):
      raise ValueError(
        "goalkeeper actor history layout must be "
        f"{self._HISTORY_LEN}x{self._ONE_STEP_DIM} "
        f"({self._ONE_STEP_DIM * self._HISTORY_LEN}D); got "
        f"{actor_history_length}x{num_one_step_obs} ({num_actor_obs}D)"
      )
    self.history_latent_dim = 16
    self.estimate_ball_dim = 6
    self.num_regions = 6
    # Register the std parameter before other modules so optimizer state from
    # old checkpoints with a root-level ``std`` parameter keeps the same order.
    self.distribution = _GoalkeeperGaussianDistribution(
      num_actions,
      init_noise_std,
      min_std=min_noise_std,
      max_std=max_noise_std,
    )

    # History encoder: 960 → 128 → 64 → 16  (ReLU)
    history_input_dim = num_one_step_obs * actor_history_length
    self.history_encoder = nn.Sequential(
      nn.Linear(history_input_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, self.history_latent_dim),
    )

    # Ball estimator: 960 → 128 → 32 → 6  (ReLU)
    self.ball_estimator = nn.Sequential(
      nn.Linear(history_input_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Linear(32, self.estimate_ball_dim),
    )

    # Region estimator: 960 → 128 → 32 → 6  (ReLU)
    self.region_estimator = nn.Sequential(
      nn.Linear(history_input_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Linear(32, self.num_regions),
    )

    # Actor: 119 → 512 → 256 → 256 → 29  (ELU)
    act_fn = _get_activation(activation)
    mlp_input_dim_a = (
      num_one_step_obs + self.history_latent_dim + self.estimate_ball_dim + 1
    )
    self.num_actor_input = mlp_input_dim_a

    actor_layers = []
    actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
    actor_layers.append(act_fn)
    for l in range(len(actor_hidden_dims)):
      if l == len(actor_hidden_dims) - 1:
        actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
      else:
        actor_layers.append(
          nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])
        )
        actor_layers.append(act_fn)
    self.actor = nn.Sequential(*actor_layers)

    # Critic: 113 → 512 → 256 → 256 → 1  (ELU)
    mlp_input_dim_c = num_critic_obs
    critic_layers = []
    critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
    critic_layers.append(act_fn)
    for l in range(len(critic_hidden_dims)):
      if l == len(critic_hidden_dims) - 1:
        critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
      else:
        critic_layers.append(
          nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1])
        )
        critic_layers.append(act_fn)
    self.critic = nn.Sequential(*critic_layers)

    self.estimate_ball = None
    self.estimate_region = None

    if verbose:
      print(f"Actor MLP: {self.actor}")
      print(f"Critic MLP: {self.critic}")
      print(f"History MLP: {self.history_encoder}")
      print(f"Ball MLP: {self.ball_estimator}")
      print(f"Region MLP: {self.region_estimator}")

  def reset(self, dones=None):
    pass

  def load_state_dict(self, state_dict, strict=True, assign=False):
    state_dict = state_dict.copy()
    if "std" in state_dict and "distribution.std_param" not in state_dict:
      state_dict["distribution.std_param"] = state_dict.pop("std")
    else:
      state_dict.pop("std", None)
    if "log_std" in state_dict:
      log_std = state_dict.pop("log_std")
      if "distribution.std_param" not in state_dict:
        state_dict["distribution.std_param"] = log_std.exp()
    if "distribution.log_std_param" in state_dict:
      log_std = state_dict.pop("distribution.log_std_param")
      if "distribution.std_param" not in state_dict:
        state_dict["distribution.std_param"] = log_std.exp()
    try:
      return super().load_state_dict(state_dict, strict=strict, assign=assign)
    except TypeError:
      return super().load_state_dict(state_dict, strict=strict)

  def _extract_tensor(self, obs, group="actor"):
    """Extract tensor from observation dict/TensorDict for given group.

    Handles plain dicts, tuples (obs, critic_obs), and TensorDict.
    """
    if isinstance(obs, (tuple, list)):
      return self._extract_tensor(obs[0], group)
    if isinstance(obs, dict):
      x = obs.get(group, obs)
    elif hasattr(obs, "get") and group in obs:
      x = obs[group]
    else:
      x = obs
    if hasattr(x, "get") and not isinstance(x, torch.Tensor):
      x = x[group] if group in x else x
    if isinstance(x, torch.Tensor) and x.dim() == 1:
      x = x.unsqueeze(0)
    return x

  def _reorder_obs_history(self, obs_history: torch.Tensor) -> torch.Tensor:
    """Transpose term-major 960D → frame-major 960D.

    mjlab history stacking (term-major):
      [ball_f0..ball_f9, ang_f0..ang_f9, grav_f0..grav_f9, jpos_f0..jpos_f9,
       jvel_f0..jvel_f9, actions_f0..actions_f9]

    Reference model expects (frame-major):
      [f0(ball,ang,grav,jpos,jvel,act), ..., f9(...)]
    """
    B = obs_history.shape[0]
    chunks = []
    offset = 0
    for sz in self._TERM_SIZES:
      block = obs_history[:, offset : offset + self._HISTORY_LEN * sz]
      chunks.append(block.view(B, self._HISTORY_LEN, sz))
      offset += self._HISTORY_LEN * sz
    # (B, 10, 3+3+3+29+29+29) = (B, 10, 96) → (B, 960)
    return torch.cat(chunks, dim=-1).reshape(B, -1)

  def forward(
    self,
    obs,
    masks=None,
    hidden_state=None,
    stochastic_output=False,
    **kwargs,
  ):
    del hidden_state, kwargs
    if self.group_name == "critic":
      return self.evaluate(obs)

    obs_history = self._extract_tensor(obs, group="actor")
    if masks is not None and not self.is_recurrent:
      from rsl_rl.utils import unpad_trajectories
      obs_history = unpad_trajectories(obs_history, masks)
    if stochastic_output:
      self.update_distribution(obs_history)
      return self.distribution.sample()
    return self.act_inference(obs_history)

  @property
  def action_mean(self):
    return self.distribution.mean

  @property
  def action_std(self):
    return self.distribution.std

  @property
  def entropy(self):
    return self.distribution.entropy

  @property
  def output_mean(self):
    return self.action_mean

  @property
  def output_std(self):
    return self.action_std

  @property
  def output_entropy(self):
    return self.entropy

  @property
  def output_distribution_params(self):
    return self.distribution.params

  def get_hidden_state(self):
    return None

  def detach_hidden_state(self, dones=None):
    del dones

  def update_normalization(self, obs):
    del obs

  def get_output_log_prob(self, outputs):
    return self.get_actions_log_prob(outputs)

  def get_kl_divergence(self, old_params, new_params):
    return self.distribution.kl_divergence(old_params, new_params)

  def update_distribution(self, obs_history):
    obs_history = self._reorder_obs_history(obs_history)
    history_latent = self.history_encoder(obs_history)
    self.estimate_ball = self.ball_estimator(obs_history)
    self.estimate_region = self.region_estimator(obs_history)
    estimated_region = torch.argmax(self.estimate_region, dim=-1, keepdim=True).to(
      dtype=obs_history.dtype
    ) / 3.0
    actor_input = torch.cat(
      (
        obs_history[:, -self.num_one_step_obs :],
        history_latent,
        self.estimate_ball,
        estimated_region,
      ),
      dim=-1,
    )
    action_mean = self.actor(actor_input)
    self.distribution.update(action_mean)

  def act(self, obs_history=None, **kwargs):
    self.update_distribution(obs_history)
    return self.distribution.sample(), self.estimate_ball, self.estimate_region

  def get_actions_log_prob(self, actions):
    return self.distribution.log_prob(actions)

  def estimate_ball_and_region(self, obs_history):
    """Return supervised estimator predictions for actor observation history."""
    obs_history = self._reorder_obs_history(obs_history)
    return self.ball_estimator(obs_history), self.region_estimator(obs_history)

  def compute_estimator_loss(
    self,
    obs_history,
    ball_target: torch.Tensor,
    region_target: torch.Tensor,
    ball_loss_coef: float = 1.0,
    region_loss_coef: float = 1.0,
  ) -> dict[str, torch.Tensor]:
    """Compute paper-style ball MSE and region CE supervision losses."""
    estimate_ball, estimate_region = self.estimate_ball_and_region(obs_history)
    region_target = region_target.to(device=estimate_region.device, dtype=torch.long)
    ball_target = ball_target.to(device=estimate_ball.device, dtype=estimate_ball.dtype)
    ball_loss = F.mse_loss(estimate_ball, ball_target)
    region_loss = F.cross_entropy(estimate_region, region_target)
    total = ball_loss_coef * ball_loss + region_loss_coef * region_loss
    return {"ball": ball_loss, "region": region_loss, "total": total}

  def act_inference(self, obs_history, observations=None):
    """Deterministic inference — used at eval/play time."""
    obs_history = self._reorder_obs_history(obs_history)
    history_latent = self.history_encoder(obs_history)
    estimate_ball = self.ball_estimator(obs_history)
    estimate_region = self.region_estimator(obs_history)
    estimated_region = torch.argmax(estimate_region, dim=-1, keepdim=True).to(
      dtype=obs_history.dtype
    ) / 3.0
    actor_input = torch.cat(
      (
        obs_history[:, -self.num_one_step_obs :],
        history_latent,
        estimate_ball,
        estimated_region,
      ),
      dim=-1,
    )
    return self.actor(actor_input)

  def evaluate(self, critic_observations, **kwargs):
    x = self._extract_tensor(critic_observations, group="critic")
    return self.critic(x)
