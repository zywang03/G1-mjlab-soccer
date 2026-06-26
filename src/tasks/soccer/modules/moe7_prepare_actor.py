"""Trainable full-MoE goalkeeper actor for prepare-only fine-tuning."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import unpad_trajectories

from src.tasks.soccer.modules.adversarial_models import AdversarialGoalkeeperActorCritic
from src.tasks.soccer.modules.gk_ballistic_residual import GoalkeeperBallisticResidual
from src.tasks.soccer.modules.moe6_goalkeeper_policy import _make_gate_net


class FrozenMlpGoalkeeperExpert(nn.Module):
  """Native MLP goalkeeper expert used by the deployed MoE bundle."""

  def __init__(self, obs_dim: int = 960, num_actions: int = 29):
    super().__init__()
    self.mlp = MLP(obs_dim, num_actions, (1024, 512, 256), "elu")
    self.distribution = GaussianDistribution(num_actions, init_std=1.0, std_type="scalar")

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mlp(obs)

  @property
  def std(self) -> torch.Tensor:
    return self.distribution.std_param.clamp_min(1.0e-4)


def _is_ballistic_residual_checkpoint(state: dict[str, Any]) -> bool:
  actor_state = state.get("actor_state_dict", state)
  return bool(state.get("ballistic_residual")) or any(
    key.startswith(("base.", "residual.")) or key == "_ballistic_marker"
    for key in actor_state
  )


def _make_ballistic_region_expert(
  obs: Any,
  obs_groups: dict[str, list[str]],
  group_name: str,
  num_actions: int,
  checkpoint: dict[str, Any],
) -> GoalkeeperBallisticResidual:
  from tensordict import TensorDict

  actor_obs = obs["actor"] if hasattr(obs, "get") and "actor" in obs else obs
  device = actor_obs.device if isinstance(actor_obs, torch.Tensor) else torch.device("cpu")
  dummy_obs = TensorDict(
    {"actor": torch.zeros(1, MoE7PrepareGoalkeeperActor._BASE_ACTOR_OBS_DIM, device=device)},
    batch_size=[1],
  )
  meta = checkpoint.get("ballistic_residual", {})
  actor_state = checkpoint.get("actor_state_dict", checkpoint)
  expert = GoalkeeperBallisticResidual(
    dummy_obs,
    {"actor": ["actor"]},
    "actor",
    num_actions,
    hidden_dims=tuple(meta.get("hidden_dims", (512, 256, 128))),
    activation="elu",
    obs_normalization=False,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 0.08,
      "std_type": "scalar",
    },
    base_ckpt=None if any(key.startswith("base.") for key in actor_state) else meta.get("base"),
    base_hidden_dims=tuple(meta.get("base_hidden", (1024, 512, 256))),
    residual_scale=float(meta.get("residual_scale", 0.25)),
  ).to(device)
  expert.load_state_dict(actor_state, strict=False)
  return expert


class MoE7PrepareGoalkeeperActor(nn.Module):
  """Run the full goalkeeper MoE while only the prepare expert is trainable.

  The six region experts and the learned gate are frozen. The actor still
  routes every rollout/update sample through the same MoE decision surface, so
  prepare actions are trained against the downstream outcome produced by the
  frozen region experts after the ball starts moving.
  """

  is_recurrent = False
  _BASE_ACTOR_OBS_DIM = 960
  _BALL_HISTORY_DIM = 30
  _HISTORY_LEN = 10
  _DT = 0.02

  def __init__(
    self,
    obs,
    obs_groups,
    group_name,
    num_actions=29,
    **kwargs,
  ):
    super().__init__()
    self.group_name = group_name
    self.num_actions = num_actions
    self.z_low = kwargs.pop("z_low", 0.85)
    self.z_up = kwargs.pop("z_up", 1.35)
    self.vz_low = kwargs.pop("vz_low", -5.0)
    self.idle_speed_threshold = kwargs.pop("idle_speed_threshold", 0.5)
    self.idle_incoming_vx_threshold = kwargs.pop("idle_incoming_vx_threshold", -0.5)
    self.g = kwargs.pop("gravity", 9.81)
    distribution_cfg = dict(kwargs.get("distribution_cfg") or {})
    self.freeze_idle_std = bool(distribution_cfg.get("freeze_idle_std", True))
    self.idle_std_min = float(distribution_cfg.get("idle_std_min", 0.15))
    self.idle_std_max = float(distribution_cfg.get("idle_std_max", 0.15))
    self._obs = obs
    self._obs_groups = obs_groups
    self.sr_experts = nn.ModuleList([
      FrozenMlpGoalkeeperExpert(self._BASE_ACTOR_OBS_DIM, num_actions)
      for _ in range(6)
    ])
    self.idle_expert = AdversarialGoalkeeperActorCritic(
      obs,
      obs_groups,
      group_name,
      num_actions,
      **kwargs,
    )
    self.gate = _make_gate_net(7, torch.device("cpu"))
    self.has_learned_gate = False
    self.gate_mean = torch.zeros(6)
    self.gate_std = torch.ones(6)
    self._meta: dict[str, Any] = {}
    self.distribution: torch.distributions.Normal | None = None
    self._freeze_static_modules()

  def _freeze_static_modules(self) -> None:
    for module in (self.sr_experts, self.gate):
      for param in module.parameters():
        param.requires_grad_(False)
    for param in self.idle_expert.parameters():
      param.requires_grad_(True)
    if self.freeze_idle_std:
      self.idle_expert.std.requires_grad_(False)
    self._clamp_idle_std_()

  def _clamp_idle_std_(self) -> None:
    with torch.no_grad():
      self.idle_expert.std.clamp_(self.idle_std_min, self.idle_std_max)

  def _actor_tensor(self, obs: Any) -> torch.Tensor:
    if isinstance(obs, (tuple, list)):
      return self._actor_tensor(obs[0])
    if isinstance(obs, dict):
      x = obs.get("actor", obs)
    elif hasattr(obs, "get") and "actor" in obs:
      x = obs["actor"]
    else:
      x = obs
    if hasattr(x, "get") and not isinstance(x, torch.Tensor):
      x = x["actor"] if "actor" in x else x
    return x.unsqueeze(0) if isinstance(x, torch.Tensor) and x.dim() == 1 else x

  def _base_actor_obs(self, actor_obs: torch.Tensor) -> torch.Tensor:
    return actor_obs[:, : self._BASE_ACTOR_OBS_DIM]

  def _ball_from_actor_obs(self, actor_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ball_history = actor_obs[:, : self._BALL_HISTORY_DIM].reshape(-1, self._HISTORY_LEN, 3)
    pos = ball_history[:, -1]
    prev = ball_history[:, -2]
    return pos, (pos - prev) / self._DT

  def _heuristic_region(self, ball_pos: torch.Tensor, ball_vel: torch.Tensor) -> torch.Tensor:
    bx = ball_pos[:, 0]
    vx = ball_vel[:, 0]
    t = torch.clamp(-bx / (vx - 1.0e-3), 0.0, 2.0)
    cy = ball_pos[:, 1] + ball_vel[:, 1] * t
    cz = ball_pos[:, 2] + ball_vel[:, 2] * t - 0.5 * self.g * t * t
    region = torch.zeros(ball_pos.shape[0], dtype=torch.long, device=ball_pos.device)
    region = torch.where(cz < self.z_low, torch.full_like(region, 4), region)
    region = torch.where(cz > self.z_up, torch.full_like(region, 2), region)
    region = torch.where(ball_vel[:, 2] - self.g * t < self.vz_low, torch.full_like(region, 4), region)
    return region + (cy < 0).long()

  def _route(self, actor_obs: torch.Tensor) -> torch.Tensor:
    ball_pos, ball_vel = self._ball_from_actor_obs(actor_obs)
    speed = torch.norm(ball_vel, dim=-1)
    idle = (speed < self.idle_speed_threshold) | (ball_vel[:, 0] >= self.idle_incoming_vx_threshold)
    if self.has_learned_gate:
      features = torch.cat([ball_pos, ball_vel], dim=-1)
      gate_mean = self.gate_mean.to(actor_obs.device)
      gate_std = self.gate_std.to(actor_obs.device).clamp_min(1.0e-6)
      region = self.gate((features - gate_mean) / gate_std).argmax(dim=1).clamp(max=5)
    else:
      region = self._heuristic_region(ball_pos, ball_vel)
    idle_idx = torch.full_like(region, 6)
    return torch.where(idle, idle_idx, region)

  @staticmethod
  def _expert_std(expert: nn.Module, actor_obs: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    std = getattr(expert, "std", None)
    if std is None and hasattr(expert, "output_std"):
      std = expert.output_std
    if std is None and hasattr(expert, "distribution"):
      std = getattr(expert.distribution, "std", None)
    if std is None:
      return torch.ones_like(mean)
    std = std.to(actor_obs.device) if isinstance(std, torch.Tensor) else torch.as_tensor(std, device=actor_obs.device)
    return std.expand(actor_obs.shape[0], -1)

  @staticmethod
  def _expert_mean(expert: nn.Module, base_obs: torch.Tensor) -> torch.Tensor:
    if isinstance(expert, FrozenMlpGoalkeeperExpert):
      return expert(base_obs)
    return expert({"actor": base_obs})

  def _means_and_stds(self, actor_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    base_obs = self._base_actor_obs(actor_obs)
    means = [self._expert_mean(expert, base_obs) for expert in self.sr_experts]
    stds = [
      self._expert_std(expert, actor_obs, mean)
      for expert, mean in zip(self.sr_experts, means)
    ]
    means.append(self.idle_expert.act_inference(actor_obs))
    self._clamp_idle_std_()
    stds.append(self.idle_expert.std.clamp(self.idle_std_min, self.idle_std_max).expand(actor_obs.shape[0], -1))
    routes = self._route(actor_obs)
    batch = torch.arange(actor_obs.shape[0], device=actor_obs.device)
    return torch.stack(means, dim=1)[batch, routes], torch.stack(stds, dim=1)[batch, routes]

  def update_distribution(self, actor_obs: torch.Tensor) -> None:
    mean, std = self._means_and_stds(actor_obs)
    self.distribution = torch.distributions.Normal(mean, std)

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
    obs = unpad_trajectories(obs, masks) if masks is not None else obs
    actor_obs = self._actor_tensor(obs)
    self.update_distribution(actor_obs)
    assert self.distribution is not None
    return self.distribution.sample() if stochastic_output else self.distribution.mean

  def reset(self, dones=None, hidden_state=None) -> None:
    self.idle_expert.reset(dones)

  def get_hidden_state(self):
    return None

  def detach_hidden_state(self, dones=None) -> None:
    pass

  def update_normalization(self, obs) -> None:
    pass

  @property
  def output_mean(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.mean

  @property
  def output_std(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.stddev

  @property
  def output_entropy(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.entropy().sum(dim=-1)

  @property
  def output_distribution_params(self):
    return (self.output_mean, self.output_std)

  def get_output_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.log_prob(actions).sum(dim=-1)

  def get_kl_divergence(self, old_params, new_params) -> torch.Tensor:
    old_mean, old_std = old_params
    new_mean, new_std = new_params
    old_dist = torch.distributions.Normal(old_mean, old_std)
    new_dist = torch.distributions.Normal(new_mean, new_std)
    return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

  def load_moe_bundle(self, bundle: dict[str, Any]) -> None:
    experts = bundle.get("sr")
    if not isinstance(experts, list) or len(experts) < 6:
      raise KeyError("MoE7 prepare actor expects a bundle with six sr experts")
    for idx in range(6):
      checkpoint = experts[idx]
      state = checkpoint.get("actor_state_dict", checkpoint)
      if _is_ballistic_residual_checkpoint(checkpoint):
        self.sr_experts[idx] = _make_ballistic_region_expert(
          self._obs,
          self._obs_groups,
          self.group_name,
          self.num_actions,
          checkpoint,
        )
      else:
        self.sr_experts[idx].load_state_dict(state, strict=False)
      for param in self.sr_experts[idx].parameters():
        param.requires_grad_(False)
    idle = next((bundle[k] for k in ("idle", "prepare", "idle_expert") if k in bundle), None)
    if isinstance(idle, dict) and "actor_state_dict" in idle:
      self.idle_expert.load_state_dict(idle["actor_state_dict"], strict=False)
    else:
      print(
        "[MoE7PrepareGoalkeeperActor] No idle/prepare expert in bundle; "
        "idle samples will use the randomly-initialized idle expert (irrelevant "
        "for active-only KL/distillation)."
      )
    gate = bundle.get("gate")
    self.has_learned_gate = gate is not None
    if gate is not None:
      num_classes = int(gate.get("num_classes", gate["state"]["4.weight"].shape[0]))
      self.gate = _make_gate_net(num_classes, torch.device("cpu")).to(next(self.parameters()).device)
      self.gate.load_state_dict(gate["state"])
      self.gate_mean = gate["mean"].detach().clone()
      self.gate_std = gate["std"].detach().clone()
    self.z_low = bundle.get("z_low", self.z_low)
    self.z_up = bundle.get("z_up", self.z_up)
    self.vz_low = bundle.get("vz_low", self.vz_low)
    self.idle_speed_threshold = bundle.get("idle_speed_threshold", self.idle_speed_threshold)
    self.idle_incoming_vx_threshold = bundle.get(
      "idle_incoming_vx_threshold",
      self.idle_incoming_vx_threshold,
    )
    self._meta = {k: v for k, v in bundle.items() if k not in ("sr", "idle", "prepare", "idle_expert", "gate")}
    self._freeze_static_modules()

  def export_moe_bundle(self) -> dict[str, Any]:
    self._clamp_idle_std_()
    bundle = dict(self._meta)
    bundle["freeze_idle_std"] = self.freeze_idle_std
    bundle["idle_std_min"] = self.idle_std_min
    bundle["idle_std_max"] = self.idle_std_max
    bundle["sr"] = [{"actor_state_dict": expert.state_dict()} for expert in self.sr_experts]
    bundle["idle"] = {"actor_state_dict": self.idle_expert.state_dict()}
    if self.has_learned_gate:
      bundle["gate"] = {
        "state": self.gate.state_dict(),
        "mean": self.gate_mean.detach().cpu(),
        "std": self.gate_std.detach().cpu(),
        "num_classes": int(self.gate[-1].out_features),
      }
    return bundle

  def state_dict(self, *args, **kwargs):  # noqa: D401 - preserve deployable MoE shape in checkpoints.
    return self.export_moe_bundle()

  def load_state_dict(self, state_dict, strict: bool = True):
    if isinstance(state_dict, dict) and "sr" in state_dict:
      self.load_moe_bundle(state_dict)
      return torch.nn.modules.module._IncompatibleKeys([], [])
    return super().load_state_dict(state_dict, strict=strict)
