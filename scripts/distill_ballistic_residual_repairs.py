"""Distill repair-oracle data into a frozen-base ballistic residual keeper.

The plain MLP repair distillation can forget useful base behavior and often
loses a large gap between the CEM oracle and the deployed policy.  This script
keeps the original distilled goalkeeper frozen and trains only the ballistic
residual head, preserving base saves while learning corrective actions for hard
trajectories.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from src.tasks.soccer.config.g1.gk_train_cfg import goalkeeper_ballistic_residual_runner_cfg

import mjlab.tasks  # noqa: F401
import src.tasks.soccer.config.eval  # noqa: F401


@dataclass
class Cfg:
  data: tuple[str, ...] = ("logs/repairs/repairs_lyk_balanced_4gpu.pt",)
  out: str = "logs/rsl_rl/g1_goalkeeper/distilled/model_repaired_residual_lyk.pt"
  base: str = "src/assets/soccer/weight/goalkeeper_distilled_v3.pt"
  epochs: int = 100
  batch_size: int = 32768
  lr: float = 3.0e-4
  lr_final: float = 3.0e-5
  residual_scale: float = 0.45
  residual_l2: float = 1.0e-4
  val_frac: float = 0.02
  blocked_only: bool = True
  seed: int = 2810
  device: str = "cuda:0"


def _load_data(paths: tuple[str, ...], blocked_only: bool) -> tuple[torch.Tensor, torch.Tensor]:
  obs_parts, act_parts = [], []
  for path in paths:
    data = torch.load(path, map_location="cpu", weights_only=False)
    obs = data["obs"]
    act = data["act"]
    if blocked_only:
      mask = data["blocked"].bool()
      obs = obs[mask]
      act = act[mask]
    obs_parts.append(obs)
    act_parts.append(act)
    print(f"  loaded {path}: {obs.shape[0]} frames (blocked_only={blocked_only})", flush=True)
  return torch.cat(obs_parts), torch.cat(act_parts)


def main(cfg: Cfg) -> None:
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  dev = cfg.device

  X, Y = _load_data(cfg.data, cfg.blocked_only)
  n = X.shape[0]
  print(f"[INFO] total {n} pairs, obs {X.shape[1]}, act {Y.shape[1]}", flush=True)

  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = 2
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=dev), clip_actions=100.0)

  import src.tasks.soccer.modules.gk_ballistic_residual as gkbr

  gkbr.BASE_CKPT = cfg.base
  gkbr.BASE_HIDDEN = (1024, 512, 256)
  gkbr.RESIDUAL_SCALE = cfg.residual_scale

  agent = goalkeeper_ballistic_residual_runner_cfg()
  runner = MjlabOnPolicyRunner(env, asdict(agent), device=dev)
  actor = runner.alg.actor
  actor.freeze_base()
  optim = torch.optim.Adam(actor.residual.parameters(), lr=cfg.lr)

  perm = torch.randperm(n)
  n_val = int(max(0, min(n - 1, round(n * cfg.val_frac)))) if n > 1 else 0
  val_idx = perm[:n_val]
  train_idx = perm[n_val:]
  print(f"[INFO] train={train_idx.numel()} val={val_idx.numel()}", flush=True)

  for epoch in range(cfg.epochs):
    frac = epoch / max(1, cfg.epochs - 1)
    lr = cfg.lr + (cfg.lr_final - cfg.lr) * frac
    for group in optim.param_groups:
      group["lr"] = lr

    actor.train()
    train_perm = train_idx[torch.randperm(train_idx.numel())]
    last = 0.0
    for start in range(0, train_perm.numel(), cfg.batch_size):
      idx = train_perm[start : start + cfg.batch_size]
      xb = X[idx].to(dev, non_blocking=True)
      yb = Y[idx].to(dev, non_blocking=True)
      pred = actor.forward({"actor": xb}, stochastic_output=False)
      with torch.no_grad():
        base = actor.base.forward({"actor": xb}, stochastic_output=False)
      loss_bc = torch.nn.functional.smooth_l1_loss(pred, yb)
      loss_reg = (pred - base).pow(2).mean()
      loss = loss_bc + cfg.residual_l2 * loss_reg
      optim.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(actor.residual.parameters(), 1.0)
      optim.step()
      last = float(loss_bc.detach().cpu())

    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == cfg.epochs:
      msg = f"[INFO] epoch {epoch + 1}/{cfg.epochs} lr={lr:.1e} train_huber={last:.5f}"
      if n_val > 0:
        actor.eval()
        losses = []
        with torch.inference_mode():
          for start in range(0, val_idx.numel(), cfg.batch_size):
            idx = val_idx[start : start + cfg.batch_size]
            xb = X[idx].to(dev, non_blocking=True)
            yb = Y[idx].to(dev, non_blocking=True)
            pred = actor.forward({"actor": xb}, stochastic_output=False)
            losses.append(torch.nn.functional.smooth_l1_loss(pred, yb).detach().cpu())
        msg += f" val_huber={float(torch.stack(losses).mean()):.5f}"
      print(msg, flush=True)

  os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
  saved = runner.alg.save()
  saved["iter"] = 0
  saved["infos"] = {"env_state": {"common_step_counter": 0}}
  saved["ballistic_residual"] = {
    "base": cfg.base,
    "base_hidden": (1024, 512, 256),
    "residual_scale": cfg.residual_scale,
  }
  torch.save(saved, cfg.out)
  print(f"[INFO] saved ballistic residual repaired keeper to {cfg.out}", flush=True)
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="distill_ballistic_residual_repairs"))
