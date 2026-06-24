"""Train a learned gate for the goalkeeper MoE: ball (pos+vel) -> expert class.

Collects (ball_pos[3], ball_vel[3]) at the first step the ball is clearly
approaching (bx<obs_x), labelled by the true region. With ``--include-idle`` it
also adds zero-speed ball states labelled as class 6, the prepare/idle expert.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch, tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends


@dataclass
class Cfg:
  num_envs: int = 512
  batches: int = 40           # resets; samples ~= num_envs * batches
  obs_x: float = 3.0          # record ball state when bx first drops below this
  epochs: int = 60
  out: str = "logs/repairs/gate.pt"
  include_idle: bool = False
  idle_samples: int = 8192
  idle_class: int = 6
  idle_ball_pos: tuple[float, float, float] = (3.0, 0.0, 0.1)
  idle_y_range: tuple[float, float] = (-0.2, 0.2)
  idle_z_range: tuple[float, float] = (0.1, 0.1)
  idle_speed_threshold: float = 0.5
  idle_incoming_vx_threshold: float = -0.5
  seed: int = 1234
  device: str = "cuda:0"


def make_gate_net(num_classes: int, device: torch.device | str):
  return torch.nn.Sequential(
    torch.nn.Linear(6, 128), torch.nn.ReLU(),
    torch.nn.Linear(128, 128), torch.nn.ReLU(),
    torch.nn.Linear(128, num_classes),
  ).to(device)


def make_idle_samples(
  count: int,
  idle_class: int,
  ball_pos: tuple[float, float, float],
  y_range: tuple[float, float],
  z_range: tuple[float, float],
  device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
  pos = torch.tensor(ball_pos, dtype=torch.float32, device=device).view(1, 3).repeat(count, 1)
  pos[:, 1] = y_range[0] + torch.rand(count, device=device) * (y_range[1] - y_range[0])
  pos[:, 2] = z_range[0] + torch.rand(count, device=device) * (z_range[1] - z_range[0])
  vel = torch.zeros(count, 3, dtype=torch.float32, device=device)
  label = torch.full((count,), idle_class, dtype=torch.long, device=device)
  return torch.cat([pos, vel], dim=-1), label


def main(cfg: Cfg):
  configure_torch_backends()
  dev = cfg.device; torch.manual_seed(cfg.seed)
  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = cfg.num_envs; env_cfg.seed = cfg.seed
  if "fell_over" in env_cfg.terminations: env_cfg.terminations["fell_over"] = None
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=dev), clip_actions=100.0)
  ball = env.unwrapped.scene["ball"]; org = env.unwrapped.scene.env_origins
  N = cfg.num_envs; A = env.num_actions
  X, Y = [], []
  for b in range(cfg.batches):
    env.reset()
    reg = env.unwrapped._gk_region.clone().long()
    rec = torch.zeros(N, dtype=torch.bool, device=dev)
    with torch.no_grad():
      for _ in range(40):
        bp = ball.data.root_link_pos_w - org; bv = ball.data.root_link_lin_vel_w
        bx = bp[:, 0]; vx = bv[:, 0]
        take = (vx < -1.0) & (bx < cfg.obs_x) & (bx > 0.2) & (~rec)
        if take.any():
          ti = take.nonzero(as_tuple=False).squeeze(-1)
          X.append(torch.cat([bp[ti], bv[ti]], -1).cpu()); Y.append(reg[ti].cpu())
          rec[ti] = True
        env.step(torch.zeros(N, A, device=dev))
    print(f"  batch {b+1}/{cfg.batches}: {sum(x.shape[0] for x in X)} samples", flush=True)
  X = torch.cat(X).to(dev); Y = torch.cat(Y).to(dev)
  num_classes = cfg.idle_class + 1 if cfg.include_idle else 6
  if cfg.include_idle:
    xi, yi = make_idle_samples(cfg.idle_samples, cfg.idle_class, cfg.idle_ball_pos,
                               cfg.idle_y_range, cfg.idle_z_range, dev)
    X = torch.cat([X, xi], dim=0); Y = torch.cat([Y, yi], dim=0)
    print(f"  added {cfg.idle_samples} idle/prepare samples as class {cfg.idle_class}", flush=True)
  mean = X.mean(0); std = X.std(0).clamp(min=1e-3)
  Xn = (X - mean) / std
  net = make_gate_net(num_classes, dev)
  opt = torch.optim.Adam(net.parameters(), 1e-3)
  n = X.shape[0]
  for ep in range(cfg.epochs):
    perm = torch.randperm(n, device=dev); last = 0.0
    for s in range(0, n, 8192):
      idx = perm[s:s + 8192]
      loss = torch.nn.functional.cross_entropy(net(Xn[idx]), Y[idx])
      opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
    if (ep + 1) % 10 == 0:
      acc = (net(Xn).argmax(1) == Y).float().mean().item()
      print(f"  epoch {ep+1} loss={last:.4f} train_acc={acc:.4f}", flush=True)
  acc = (net(Xn).argmax(1) == Y).float().mean().item()
  torch.save({
    "state": net.state_dict(),
    "mean": mean.cpu(),
    "std": std.cpu(),
    "obs_x": cfg.obs_x,
    "acc": acc,
    "num_classes": num_classes,
    "idle_class": cfg.idle_class if cfg.include_idle else None,
    "idle_speed_threshold": cfg.idle_speed_threshold,
    "idle_incoming_vx_threshold": cfg.idle_incoming_vx_threshold,
  }, cfg.out)
  print(f"[INFO] gate train_acc={acc:.4f} saved to {cfg.out}", flush=True)
  env.close()


if __name__ == "__main__":
  import mjlab.tasks, src.tasks  # noqa
  main(tyro.cli(Cfg, prog="train_gate"))
