"""Deployable 6-way goalkeeper mixture-of-experts policy."""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.rl import MjlabOnPolicyRunner


def _load_expert(env, checkpoint: dict, device: str, temp_dir: str):
  from src.tasks.soccer.config.g1.gk_train_cfg import (
    goalkeeper_ballistic_residual_runner_cfg,
    goalkeeper_train_runner_cfg,
  )

  if checkpoint.get("ballistic_residual"):
    import src.tasks.soccer.modules.gk_ballistic_residual as gkbr

    meta = checkpoint["ballistic_residual"]
    gkbr.BASE_CKPT = meta.get("base")
    gkbr.BASE_HIDDEN = tuple(meta.get("base_hidden", (1024, 512, 256)))
    gkbr.RESIDUAL_SCALE = float(meta.get("residual_scale", 0.25))
    agent_cfg = goalkeeper_ballistic_residual_runner_cfg()
  else:
    agent_cfg = goalkeeper_train_runner_cfg()

  path = Path(temp_dir) / f"expert_{id(checkpoint)}.pt"
  torch.save(checkpoint, path)
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
  runner.load(str(path), load_cfg={"actor": True})
  return runner.get_inference_policy(device=device)


class GoalkeeperMoE6Policy:
  """Single-file 6-expert goalkeeper policy with a ballistic region gate."""

  def __init__(self, bundle: dict, env, device: str):
    self.device = device
    self.z_low = float(bundle.get("z_low", 0.85))
    self.z_up = float(bundle.get("z_up", 1.35))
    self.vz_low = float(bundle.get("vz_low", -99.0))
    self.latch_hi = float(bundle.get("latch_hi", 5.0))
    self.land_x = float(bundle.get("land_x", 0.0))
    self.g = 9.81
    self._temp_dir = tempfile.mkdtemp(prefix="gk_moe6_")

    experts = bundle.get("sr")
    if not isinstance(experts, (list, tuple)) or len(experts) != 6:
      raise ValueError("MoE6 checkpoint must contain key 'sr' with 6 experts")
    self.experts = [_load_expert(env, expert, device, self._temp_dir) for expert in experts]

    mirror_map = bundle.get("mirror_map", "")
    if mirror_map:
      from src.tasks.soccer.modules.symmetry import mirror_action, mirror_obs

      base_experts = list(self.experts)

      def mirror_policy(policy):
        def wrapped(obs):
          return mirror_action(policy({"actor": mirror_obs(obs["actor"])}))
        return wrapped

      for item in str(mirror_map).split(","):
        if not item:
          continue
        dst, src = (int(x) for x in item.split(":"))
        self.experts[dst] = mirror_policy(base_experts[src])

    base_env = getattr(env, "unwrapped", env)
    self.ball = base_env.scene["ball"]
    self.origins = base_env.scene.env_origins
    self.num_envs = base_env.num_envs
    self._raw_ball_pos = None
    self._raw_ball_vel = None
    self.reset()

  def reset(self, dones=None):
    del dones
    self.latched = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device,
    )
    self._raw_ball_pos = None
    self._raw_ball_vel = None
    for expert in self.experts:
      reset = getattr(expert, "reset", None)
      if reset is not None:
        reset()

  def set_raw_ball_state(self, pos, vel) -> None:
    self._raw_ball_pos = torch.as_tensor(pos, dtype=torch.float32, device=self.device).reshape(1, 3)
    self._raw_ball_vel = torch.as_tensor(vel, dtype=torch.float32, device=self.device).reshape(1, 3)

  def _ball_state(self):
    if self._raw_ball_pos is not None and self._raw_ball_vel is not None:
      pos = self._raw_ball_pos
      vel = self._raw_ball_vel
      origins = torch.zeros_like(pos)
    else:
      pos = self.ball.data.root_link_pos_w
      vel = self.ball.data.root_link_lin_vel_w
      origins = self.origins
    return pos, vel, origins

  def _gate_region(self) -> tuple[torch.Tensor, torch.Tensor]:
    pos, vel, origins = self._ball_state()
    bx = pos[:, 0] - origins[:, 0]
    vx = vel[:, 0]
    valid = (vx < -1.0) & (bx > 0.2) & (bx < self.latch_hi)
    t = torch.clamp(-(bx - self.land_x) / (vx - 1.0e-3), 0.0, 2.0)
    cy = (pos[:, 1] - origins[:, 1]) + vel[:, 1] * t
    cz = pos[:, 2] + vel[:, 2] * t - 0.5 * self.g * t * t
    vz_cross = vel[:, 2] - self.g * t

    base = torch.zeros_like(bx, dtype=torch.long)
    base = torch.where(cz < self.z_low, torch.full_like(base, 4), base)
    base = torch.where(cz > self.z_up, torch.full_like(base, 2), base)
    base = torch.where(vz_cross < self.vz_low, torch.full_like(base, 4), base)
    return base + (cy < 0).long(), valid

  def __call__(self, obs):
    region, valid = self._gate_region()
    if region.shape[0] != self.latched.shape[0]:
      self.num_envs = region.shape[0]
      self.latched = torch.full_like(region, -1)
    new_latch = valid & (self.latched < 0)
    self.latched = torch.where(new_latch, region, self.latched)
    use_region = torch.where(self.latched < 0, torch.zeros_like(self.latched), self.latched)
    actions = torch.stack([expert(obs) for expert in self.experts], dim=0)
    return actions[use_region, torch.arange(actions.shape[1], device=self.device)]
