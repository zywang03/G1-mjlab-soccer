"""Sweep MoE6 goalkeeper gate parameters without retraining experts."""

from __future__ import annotations

import csv
import itertools
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


@dataclass
class Cfg:
  expert_dir: str = "logs/lyk/fuse_8gpu/base"
  prefix: str = "stable_sr"
  mirror_map: str = "1:0,3:2"
  out_csv: str = "logs/lyk/gate_sweep.csv"
  num_envs: int = 256
  batches: int = 32
  steps: int = 149
  seed: int = 2810
  z_lows: tuple[float, ...] = (0.75, 0.85, 0.95)
  z_ups: tuple[float, ...] = (1.25, 1.35, 1.45)
  latch_his: tuple[float, ...] = (4.0, 5.0, 6.0)
  vz_lows: tuple[float, ...] = (-99.0,)
  land_x: float = 0.0
  device: str = "cuda:0"


def _expert_paths(cfg: Cfg) -> tuple[str, ...]:
  return tuple(str(Path(cfg.expert_dir) / f"{cfg.prefix}{idx}.pt") for idx in range(6))


def _load_policy(env, checkpoint: str, device: str):
  loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
  from src.tasks.soccer.config.g1.gk_train_cfg import (
    goalkeeper_ballistic_residual_runner_cfg,
    goalkeeper_train_runner_cfg,
  )

  if loaded.get("ballistic_residual"):
    import src.tasks.soccer.modules.gk_ballistic_residual as gkbr

    meta = loaded["ballistic_residual"]
    gkbr.BASE_CKPT = meta.get("base")
    gkbr.BASE_HIDDEN = tuple(meta.get("base_hidden", (1024, 512, 256)))
    gkbr.RESIDUAL_SCALE = float(meta.get("residual_scale", 0.25))
    agent_cfg = goalkeeper_ballistic_residual_runner_cfg()
  else:
    agent_cfg = goalkeeper_train_runner_cfg()
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
  runner.load(checkpoint, load_cfg={"actor": True})
  return runner.get_inference_policy(device=device)


def _parse_mirror_map(raw: str) -> list[tuple[int, int]]:
  if not raw:
    return []
  out = []
  for item in raw.split(","):
    dst, src = (int(x) for x in item.split(":"))
    out.append((dst, src))
  return out


def _apply_mirror(experts, raw: str):
  if not raw:
    return experts
  from src.tasks.soccer.modules.symmetry import mirror_action, mirror_obs

  base = list(experts)

  def mirror_policy(policy):
    def wrapped(obs):
      return mirror_action(policy({"actor": mirror_obs(obs["actor"])}))
    return wrapped

  experts = list(experts)
  for dst, src in _parse_mirror_map(raw):
    experts[dst] = mirror_policy(base[src])
    print(f"[INFO] expert {dst} := mirror(expert {src})", flush=True)
  return experts


def main(cfg: Cfg) -> None:
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  device = cfg.device

  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env_cfg.terminations.pop("fell_over", None)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=100.0)

  experts = [_load_policy(env, path, device) for path in _expert_paths(cfg)]
  experts = _apply_mirror(experts, cfg.mirror_map)

  ball = env.unwrapped.scene["ball"]
  origins = env.unwrapped.scene.env_origins
  num_envs = cfg.num_envs
  gravity = 9.81

  def gate(z_low: float, z_up: float, latch_hi: float, vz_low: float):
    pos = ball.data.root_link_pos_w
    vel = ball.data.root_link_lin_vel_w
    bx = pos[:, 0] - origins[:, 0]
    vx = vel[:, 0]
    valid = (vx < -1.0) & (bx > 0.2) & (bx < latch_hi)
    t = torch.clamp(-(bx - cfg.land_x) / (vx - 1.0e-3), 0.0, 2.0)
    cy = (pos[:, 1] - origins[:, 1]) + vel[:, 1] * t
    cz = pos[:, 2] + vel[:, 2] * t - 0.5 * gravity * t * t
    vz_cross = vel[:, 2] - gravity * t
    base = torch.zeros_like(bx, dtype=torch.long)
    base = torch.where(cz < z_low, torch.full_like(base, 4), base)
    base = torch.where(cz > z_up, torch.full_like(base, 2), base)
    base = torch.where(vz_cross < vz_low, torch.full_like(base, 4), base)
    return base + (cy < 0).long(), valid

  rows = []
  grid = itertools.product(cfg.z_lows, cfg.z_ups, cfg.latch_his, cfg.vz_lows)
  for z_low, z_up, latch_hi, vz_low in grid:
    if z_low >= z_up:
      continue
    # Use the same sampled trajectories for every gate setting so the sweep
    # compares routing changes instead of random reset noise.
    torch.manual_seed(cfg.seed)
    total = blocked = routed_ok = 0
    per_region = [[0, 0] for _ in range(6)]
    with torch.inference_mode():
      for _ in range(cfg.batches):
        obs = env.reset()
        if isinstance(obs, tuple):
          obs = obs[0]
        true_region = getattr(env.unwrapped, "_gk_region").clone().long()
        latched = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        entered = torch.zeros(num_envs, dtype=torch.bool, device=device)
        for _ in range(cfg.steps):
          region, valid = gate(z_low, z_up, latch_hi, vz_low)
          latched = torch.where(valid & (latched < 0), region, latched)
          use = torch.where(latched < 0, torch.zeros_like(latched), latched)
          actions = torch.stack([expert(obs) for expert in experts], dim=0)
          action = actions[use, torch.arange(num_envs, device=device)]
          obs = env.step(action)[0]
          pos = ball.data.root_link_pos_w
          bx = pos[:, 0] - origins[:, 0]
          by = pos[:, 1] - origins[:, 1]
          entered |= (bx <= -0.5) & (by.abs() <= 1.5) & (pos[:, 2] <= 1.8)
        use = torch.where(latched < 0, torch.zeros_like(latched), latched)
        blocked_mask = ~entered
        total += num_envs
        blocked += int(blocked_mask.sum())
        routed_ok += int((use == true_region).sum())
        for region in range(6):
          mask = true_region == region
          per_region[region][0] += int((blocked_mask & mask).sum())
          per_region[region][1] += int(mask.sum())

    row = {
      "z_low": z_low,
      "z_up": z_up,
      "latch_hi": latch_hi,
      "vz_low": vz_low,
      "blocked": blocked,
      "total": total,
      "block_rate": blocked / max(1, total),
      "gate_acc": routed_ok / max(1, total),
    }
    for region, (ok, n) in enumerate(per_region):
      row[f"r{region}_block_rate"] = ok / max(1, n)
    rows.append(row)
    print(
      f"z_low={z_low:.2f} z_up={z_up:.2f} latch_hi={latch_hi:.1f} "
      f"vz_low={vz_low:.1f}: block={100 * row['block_rate']:.2f}% "
      f"gate={100 * row['gate_acc']:.2f}%",
      flush=True,
    )

  rows.sort(key=lambda row: row["block_rate"], reverse=True)
  if not rows:
    raise ValueError("empty sweep grid; check z_lows/z_ups/latch_his/vz_lows")
  out = Path(cfg.out_csv)
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  best = rows[0]
  print(
    "[BEST] "
    f"block={100 * best['block_rate']:.2f}% "
    f"z_low={best['z_low']} z_up={best['z_up']} "
    f"latch_hi={best['latch_hi']} vz_low={best['vz_low']} "
    f"csv={out}",
    flush=True,
  )
  env.close()


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  import src.tasks.soccer.config.eval  # noqa: F401

  main(tyro.cli(Cfg, prog="sweep_moe6_gate"))
