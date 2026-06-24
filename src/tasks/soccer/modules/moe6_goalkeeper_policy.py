"""Shared MoE goalkeeper policy adapter.

The checkpoint is a bundle of six region-specialist goalkeeper actors plus an
optional prepare/idle expert. This adapter loads those experts with the native
goalkeeper runner config and exposes a batch-safe policy callable.
"""

from __future__ import annotations

from dataclasses import asdict
import tempfile
from typing import Any

import torch
from mjlab.rl import MjlabOnPolicyRunner
from src.tasks.soccer.config.g1.gk_train_cfg import (
  goalkeeper_ballistic_residual_runner_cfg,
  goalkeeper_train_runner_cfg,
)
from src.tasks.soccer.config.g1.rl_cfg import (
  AdversarialGoalkeeperRunner,
  GoalkeeperRunner,
  unitree_g1_goalkeeper_adversarial_ppo_runner_cfg,
  unitree_g1_goalkeeper_ppo_runner_cfg,
)


def _as_batched_vec(values: Any, device: torch.device, width: int, n_envs: int) -> torch.Tensor:
  tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
  if tensor.numel() == width:
    return tensor.view(1, width).expand(n_envs, -1)
  return tensor.view(n_envs, width)


def _make_gate_net(num_classes: int, device: torch.device) -> torch.nn.Module:
  return torch.nn.Sequential(
    torch.nn.Linear(6, 128), torch.nn.ReLU(),
    torch.nn.Linear(128, 128), torch.nn.ReLU(),
    torch.nn.Linear(128, num_classes),
  ).to(device)


def _is_goalkeeper_actor_critic_checkpoint(state_dict: dict[str, Any]) -> bool:
  actor_state = state_dict.get("actor_state_dict", {})
  return any(
    key.startswith(("history_encoder.", "ball_estimator.", "region_estimator.", "actor."))
    for key in actor_state
  )


def _is_adversarial_goalkeeper_checkpoint(state_dict: dict[str, Any]) -> bool:
  actor_state = state_dict.get("actor_state_dict", {})
  return any(key.startswith(("actor_residual.", "critic_residual.")) for key in actor_state)


def _is_ballistic_residual_checkpoint(state_dict: dict[str, Any]) -> bool:
  actor_state = state_dict.get("actor_state_dict", {})
  return bool(state_dict.get("ballistic_residual")) or any(
    key.startswith(("base.", "residual.")) or key == "_ballistic_marker"
    for key in actor_state
  )


def _load_expert_policy(state_dict: dict[str, Any], env: Any, device: str):
  checkpoint = dict(state_dict)
  checkpoint.setdefault("infos", {})
  checkpoint.setdefault("iter", 0)
  if _is_ballistic_residual_checkpoint(state_dict):
    import src.tasks.soccer.modules.gk_ballistic_residual as gkbr

    meta = checkpoint.get("ballistic_residual", {})
    actor_state = checkpoint.get("actor_state_dict", {})
    embeds_base = any(key.startswith("base.") for key in actor_state)
    gkbr.BASE_CKPT = None if embeds_base else meta.get("base")
    gkbr.BASE_HIDDEN = tuple(meta.get("base_hidden", (1024, 512, 256)))
    gkbr.RESIDUAL_SCALE = float(meta.get("residual_scale", 0.25))
    runner = MjlabOnPolicyRunner(
      env,
      asdict(goalkeeper_ballistic_residual_runner_cfg()),
      device=device,
    )
  elif _is_adversarial_goalkeeper_checkpoint(state_dict):
    runner = AdversarialGoalkeeperRunner(
      env,
      asdict(unitree_g1_goalkeeper_adversarial_ppo_runner_cfg()),
      device=device,
    )
  elif _is_goalkeeper_actor_critic_checkpoint(state_dict):
    runner = GoalkeeperRunner(env, asdict(unitree_g1_goalkeeper_ppo_runner_cfg()), device=device)
  else:
    runner = MjlabOnPolicyRunner(env, asdict(goalkeeper_train_runner_cfg()), device=device)
  with tempfile.NamedTemporaryFile(suffix=".pt") as ckpt_file:
    torch.save(checkpoint, ckpt_file.name)
    runner.load(ckpt_file.name, load_cfg={"actor": True})
  return runner.get_inference_policy(device=device)


class MoE6GoalkeeperPolicy:
  """Batch-safe goalkeeper mixture-of-experts policy."""

  def __init__(self, bundle: dict[str, Any], env: Any, device: str, idle_env: Any | None = None):
    self.z_low = bundle.get("z_low", 0.85)
    self.z_up = bundle.get("z_up", 1.35)
    self.vz_low = bundle.get("vz_low", -5.0)
    self.latch_hi = bundle.get("latch_hi", 5.0)
    self.idle_speed_threshold = bundle.get("idle_speed_threshold", 0.5)
    self.idle_incoming_vx_threshold = bundle.get("idle_incoming_vx_threshold", -0.5)
    self.dev = torch.device(device)
    self.n_envs = env.unwrapped.num_envs
    self.g = 9.81
    self.ball = env.unwrapped.scene["ball"]
    self.env_origins = getattr(env.unwrapped.scene, "env_origins", torch.zeros(self.n_envs, 3, device=self.dev))
    self.idle_env = idle_env

    self.experts = [_load_expert_policy(bundle["sr"][i], env, device) for i in range(6)]
    self._apply_mirror_map(bundle.get("mirror_map", ""))
    idle_state = next((bundle[k] for k in ("idle", "prepare", "idle_expert") if k in bundle), None)
    self.idle_expert_index = None
    if idle_state is not None:
      self.idle_expert_index = len(self.experts)
      self.experts.append(_load_expert_policy(idle_state, idle_env or env, device))
    self.learned_gate = None
    gate = bundle.get("gate")
    if gate is not None and gate.get("state"):
      num_classes = int(gate["num_classes"]) if "num_classes" in gate else int(gate["state"]["4.weight"].shape[0])
      self.learned_gate = _make_gate_net(num_classes, self.dev)
      self.learned_gate.load_state_dict(gate["state"])
      self.learned_gate.eval()
      self.gate_mean = gate["mean"].to(self.dev)
      self.gate_std = gate["std"].to(self.dev)
    self.reset()

  def _apply_mirror_map(self, mirror_map: str) -> None:
    if not mirror_map:
      return
    from src.tasks.soccer.modules.symmetry import mirror_action, mirror_obs

    base_experts = list(self.experts)

    def _mirror(policy):
      return lambda obs: mirror_action(policy({"actor": mirror_obs(obs["actor"])}))

    for pair in mirror_map.split(","):
      dst, src = (int(x) for x in pair.split(":"))
      self.experts[dst] = _mirror(base_experts[src])

  def reset(self) -> None:
    self.latched = torch.full((self.n_envs,), -1, dtype=torch.long, device=self.dev)
    idle_env = getattr(self, "idle_env", None)
    if idle_env is not None:
      reset_fn = getattr(self.idle_env, "reset", None)
      if reset_fn is not None:
        reset_fn()
    for expert in self.experts:
      reset_fn = getattr(expert, "reset", None)
      if reset_fn is not None:
        reset_fn()

  def close(self) -> None:
    if self.idle_env is not None:
      close_fn = getattr(self.idle_env, "close", None)
      if close_fn is not None:
        close_fn()

  def _ball_from_raw(self, raw_state: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    ball = raw_state["ball"]
    return (
      _as_batched_vec(ball["pos"], self.dev, 3, self.n_envs),
      _as_batched_vec(ball["vel"], self.dev, 3, self.n_envs),
    )

  def _ball_from_env(self) -> tuple[torch.Tensor, torch.Tensor]:
    pos = self.ball.data.root_link_pos_w.to(self.dev) - self.env_origins.to(self.dev)
    vel = self.ball.data.root_link_lin_vel_w.to(self.dev)
    return pos, vel

  def _gate(self, ball_pos: torch.Tensor, ball_vel: torch.Tensor) -> torch.Tensor:
    bx = ball_pos[:, 0]
    vx = ball_vel[:, 0]
    valid = (vx < -1.0) & (bx > 0.2) & (bx < self.latch_hi)
    learned_gate = getattr(self, "learned_gate", None)
    if learned_gate is not None:
      features = torch.cat([ball_pos, ball_vel], dim=-1)
      region = learned_gate((features - self.gate_mean) / self.gate_std).argmax(1)
      region = region.clamp(max=5)
    else:
      t = torch.clamp(-bx / (vx - 1e-3), 0.0, 2.0)
      cy = ball_pos[:, 1] + ball_vel[:, 1] * t
      cz = ball_pos[:, 2] + ball_vel[:, 2] * t - 0.5 * self.g * t * t
      base = torch.zeros(self.n_envs, dtype=torch.long, device=self.dev)
      base = torch.where(cz < self.z_low, torch.full_like(base, 4), base)
      base = torch.where(cz > self.z_up, torch.full_like(base, 2), base)
      base = torch.where(ball_vel[:, 2] - self.g * t < self.vz_low, torch.full_like(base, 4), base)
      region = base + (cy < 0).long()
    self.latched = torch.where(valid & (self.latched < 0), region, self.latched)
    default = torch.zeros_like(self.latched)
    idle_expert_index = getattr(self, "idle_expert_index", None)
    if idle_expert_index is not None:
      speed = torch.norm(ball_vel, dim=-1)
      idle = (speed < self.idle_speed_threshold) | (vx >= self.idle_incoming_vx_threshold)
      idle_idx = torch.full_like(default, idle_expert_index)
      default = torch.where(idle, idle_idx, default)
    return torch.where(self.latched < 0, default, self.latched)

  def __call__(self, obs: dict[str, torch.Tensor], raw_state: dict[str, Any] | None = None) -> torch.Tensor:
    ball_pos, ball_vel = self._ball_from_raw(raw_state) if raw_state is not None else self._ball_from_env()
    use = self._gate(ball_pos, ball_vel)
    expert_actions = [expert(obs) for expert in self.experts[:6]]
    idle_expert_index = getattr(self, "idle_expert_index", None)
    if idle_expert_index is not None:
      idle_env = getattr(self, "idle_env", None)
      idle_obs = idle_env.get_observations() if idle_env is not None else obs
      expert_actions.append(self.experts[idle_expert_index](idle_obs))
    actions = torch.stack(expert_actions, dim=0)
    return actions[use, torch.arange(self.n_envs, device=self.dev)]
