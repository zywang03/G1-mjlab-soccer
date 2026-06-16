"""Ballistic-feature residual actor for goalkeeper fine-tuning.

The distilled goalkeeper already contains a coordinated diving skill.  This
module keeps that policy frozen and trains only a small residual head.  The
residual gets the original 960D actor history plus a compact ballistic feature
vector computed from the ball-position history:

  [t_to_keeper, y_keeper, z_keeper, t_to_goal, y_goal, z_goal, vx, speed]

The base policy still sees the exact observation shape it was distilled on, so
loading the existing checkpoint remains stable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable


BASE_CKPT = None
BASE_HIDDEN = (1024, 512, 256)
RESIDUAL_SCALE = 0.25

_HISTORY_LEN = 10
_BALL_TERM_DIM = 3
_BASE_OBS_DIM = 960
_FEATURE_DIM = 8
_CONTROL_DT = 0.02
_GRAVITY = 9.81


def _feature_dim() -> int:
  return _FEATURE_DIM


class GoalkeeperBallisticResidual(nn.Module):
  """Frozen distilled base policy plus trainable ballistic residual."""

  is_recurrent: bool = False

  def __init__(
    self,
    obs,
    obs_groups,
    obs_set,
    output_dim,
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=False,
    distribution_cfg=None,
    base_ckpt=None,
    base_hidden_dims=(1024, 512, 256),
    residual_scale=0.25,
    **kwargs,
  ):
    del obs_normalization
    if kwargs:
      print(
        "GoalkeeperBallisticResidual.__init__ got unexpected arguments: "
        + str([key for key in kwargs.keys()])
      )
    super().__init__()
    self.group = obs_groups[obs_set][0]
    if base_ckpt is None:
      base_ckpt = BASE_CKPT
      base_hidden_dims = BASE_HIDDEN
      residual_scale = RESIDUAL_SCALE
    self.residual_scale = float(residual_scale)
    dcfg = dict(
      distribution_cfg
      or {"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"}
    )

    self.base = MLPModel(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      base_hidden_dims,
      activation,
      False,
      dict(dcfg),
    )
    if base_ckpt:
      ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
      self.base.load_state_dict(ck["actor_state_dict"])
      print(f"[GoalkeeperBallisticResidual] loaded frozen base from {base_ckpt}")
    self.freeze_base()

    in_dim = obs[self.group].shape[-1]
    if in_dim != _BASE_OBS_DIM:
      raise ValueError(
        f"GoalkeeperBallisticResidual expects {_BASE_OBS_DIM}D actor obs; got {in_dim}"
      )
    self.residual = MLP(in_dim + _FEATURE_DIM, output_dim, hidden_dims, activation)
    for module in reversed([m for m in self.residual.modules() if isinstance(m, nn.Linear)]):
      nn.init.zeros_(module.weight)
      nn.init.zeros_(module.bias)
      break

    dist_class: type[Distribution] = resolve_callable(dcfg.pop("class_name"))
    self.distribution: Distribution = dist_class(output_dim, **dcfg)
    self.register_buffer("_ballistic_marker", torch.ones(1), persistent=True)

  def freeze_base(self) -> None:
    for param in self.base.parameters():
      param.requires_grad_(False)
    self.base.eval()

  def train(self, mode: bool = True):
    super().train(mode)
    self.base.eval()
    return self

  @staticmethod
  def ballistic_features_from_history(obs_history: torch.Tensor) -> torch.Tensor:
    """Compute normalized ballistic features from term-major 960D actor history."""
    if obs_history.shape[-1] < _HISTORY_LEN * _BALL_TERM_DIM:
      return torch.zeros(
        obs_history.shape[0],
        _FEATURE_DIM,
        dtype=obs_history.dtype,
        device=obs_history.device,
      )

    ball_hist = obs_history[:, : _HISTORY_LEN * _BALL_TERM_DIM].reshape(
      obs_history.shape[0], _HISTORY_LEN, _BALL_TERM_DIM
    )
    pos = ball_hist[:, -1]
    prev = ball_hist[:, -4]
    vel = (pos - prev) / (3.0 * _CONTROL_DT)
    vx = vel[:, 0]

    def plane_time(plane_x: float) -> torch.Tensor:
      safe_vx = torch.where(vx < -1.0e-3, vx, torch.full_like(vx, -1.0e-3))
      raw_t = (plane_x - pos[:, 0]) / safe_vx
      valid = (vx < -1.0e-3) & (raw_t >= 0.0)
      t = torch.where(valid, raw_t, torch.full_like(raw_t, 2.0))
      return torch.clamp(t, 0.0, 2.0)

    t_keeper = plane_time(0.0)
    t_goal = plane_time(-0.5)

    def cross(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      y = pos[:, 1] + vel[:, 1] * t
      z = pos[:, 2] + vel[:, 2] * t - 0.5 * _GRAVITY * t * t
      return y, torch.clamp(z, min=0.0)

    y_keeper, z_keeper = cross(t_keeper)
    y_goal, z_goal = cross(t_goal)
    speed = torch.linalg.norm(vel, dim=-1)

    features = torch.stack(
      (
        t_keeper / 2.0,
        y_keeper / 1.5,
        z_keeper / 1.8,
        t_goal / 2.0,
        y_goal / 1.5,
        z_goal / 1.8,
        vx / 8.0,
        speed / 8.0,
      ),
      dim=-1,
    )
    return torch.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0).clamp(-3.0, 3.0)

  def _mean(self, obs):
    actor_obs = obs[self.group]
    with torch.no_grad():
      base = self.base.forward({self.group: actor_obs}, stochastic_output=False)
    features = self.ballistic_features_from_history(actor_obs)
    residual_in = torch.cat((actor_obs, features), dim=-1)
    residual = self.residual(residual_in)
    return base + self.residual_scale * torch.tanh(residual)

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
    del masks, hidden_state
    mean = self._mean(obs)
    self.distribution.update(mean)
    if stochastic_output:
      return self.distribution.sample()
    return self.distribution.deterministic_output(mean)

  def reset(self, dones=None, hidden_state=None):
    del dones, hidden_state

  def get_hidden_state(self):
    return None

  def detach_hidden_state(self, dones=None):
    del dones

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
