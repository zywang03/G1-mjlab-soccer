"""Single-policy recurrent goalkeeper actor."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable


class GoalkeeperLSTMStudent(nn.Module):
  """Projection + LSTM/GRU + MLP action head for the 960D keeper actor obs."""

  is_recurrent: bool = False

  def __init__(
    self,
    obs,
    obs_groups,
    obs_set,
    output_dim,
    hidden_dims=(256, 128),
    activation="elu",
    obs_normalization=False,
    distribution_cfg=None,
    rnn_type="lstm",
    rnn_hidden_dim=256,
    rnn_num_layers=1,
    **kwargs,
  ):
    del obs_normalization
    if kwargs:
      print(
        "GoalkeeperLSTMStudent.__init__ got unexpected arguments: "
        + str([key for key in kwargs.keys()])
      )
    super().__init__()
    self.group = obs_groups[obs_set][0]
    in_dim = obs[self.group].shape[-1]
    self.rnn_type = rnn_type.lower()
    self.rnn_hidden_dim = int(rnn_hidden_dim)
    self.rnn_num_layers = int(rnn_num_layers)

    self.input = nn.Sequential(nn.Linear(in_dim, self.rnn_hidden_dim), nn.ELU())
    if self.rnn_type == "gru":
      self.rnn = nn.GRU(
        self.rnn_hidden_dim,
        self.rnn_hidden_dim,
        num_layers=self.rnn_num_layers,
      )
    elif self.rnn_type == "lstm":
      self.rnn = nn.LSTM(
        self.rnn_hidden_dim,
        self.rnn_hidden_dim,
        num_layers=self.rnn_num_layers,
      )
    else:
      raise ValueError(f"Unsupported rnn_type={rnn_type!r}; use 'lstm' or 'gru'")

    self.head = MLP(self.rnn_hidden_dim, output_dim, hidden_dims, activation)
    dcfg = dict(
      distribution_cfg
      or {"class_name": "GaussianDistribution", "init_std": 0.05, "std_type": "scalar"}
    )
    dist_class: type[Distribution] = resolve_callable(dcfg.pop("class_name"))
    self.distribution: Distribution = dist_class(output_dim, **dcfg)
    self._hidden_state = None

  def _zero_state(self, batch: int, device: torch.device):
    shape = (self.rnn_num_layers, batch, self.rnn_hidden_dim)
    h = torch.zeros(shape, device=device)
    if self.rnn_type == "lstm":
      return h, torch.zeros_like(h)
    return h

  def _fix_hidden(self, batch: int, device: torch.device):
    if self._hidden_state is None:
      self._hidden_state = self._zero_state(batch, device)
      return
    h = self._hidden_state[0] if self.rnn_type == "lstm" else self._hidden_state
    if h.shape[1] != batch or h.device != device:
      self._hidden_state = self._zero_state(batch, device)

  def reset(self, dones=None, hidden_state=None):
    del hidden_state
    if dones is None or self._hidden_state is None:
      self._hidden_state = None
      return
    done = dones.bool().view(-1)
    if self.rnn_type == "lstm":
      h, c = self._hidden_state
      h = h.detach().clone()
      c = c.detach().clone()
      h[:, done] = 0.0
      c[:, done] = 0.0
      self._hidden_state = (h, c)
    else:
      h = self._hidden_state.detach().clone()
      h[:, done] = 0.0
      self._hidden_state = h

  def _step_mean(self, actor_obs: torch.Tensor) -> torch.Tensor:
    self._fix_hidden(actor_obs.shape[0], actor_obs.device)
    x = self.input(actor_obs).unsqueeze(0)
    y, self._hidden_state = self.rnn(x, self._hidden_state)
    if self.rnn_type == "lstm":
      self._hidden_state = tuple(t.detach() for t in self._hidden_state)
    else:
      self._hidden_state = self._hidden_state.detach()
    return self.head(y.squeeze(0))

  def sequence_mean(self, actor_obs: torch.Tensor) -> torch.Tensor:
    """Return action means for actor_obs shaped (T, B, obs_dim)."""
    t, b, d = actor_obs.shape
    x = self.input(actor_obs.reshape(t * b, d)).reshape(t, b, -1)
    y, _ = self.rnn(x, self._zero_state(b, actor_obs.device))
    return self.head(y.reshape(t * b, -1)).reshape(t, b, -1)

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
    del masks, hidden_state
    mean = self._step_mean(obs[self.group])
    self.distribution.update(mean)
    if stochastic_output:
      return self.distribution.sample()
    return self.distribution.deterministic_output(mean)

  def get_hidden_state(self):
    return self._hidden_state

  def detach_hidden_state(self, dones=None):
    del dones
    if self._hidden_state is None:
      return
    if self.rnn_type == "lstm":
      self._hidden_state = tuple(t.detach() for t in self._hidden_state)
    else:
      self._hidden_state = self._hidden_state.detach()

  def update_normalization(self, obs):
    del obs

  @property
  def output_mean(self):
    return self.distribution.mean

  @property
  def output_std(self):
    return self.distribution.std

  @property
  def output_entropy(self):
    return self.distribution.entropy

  @property
  def output_distribution_params(self):
    return self.distribution.params

  def get_output_log_prob(self, outputs):
    return self.distribution.log_prob(outputs)

  def get_kl_divergence(self, old, new):
    return self.distribution.kl_divergence(old, new)
