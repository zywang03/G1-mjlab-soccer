"""Collect failure cases for a bundled MoE6 goalkeeper checkpoint."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.soccer.modules.gk_moe6 import GoalkeeperMoE6Policy

_REGION_NAMES = ["Right-Mid", "Left-Mid", "Right-Up", "Left-Up", "Right-Low", "Left-Low"]


@dataclass
class Cfg:
  checkpoint: str = "src/assets/soccer/weight/goalkeeper_moe6_8gpu_base.pt"
  out_csv: str = "logs/lyk/moe6_failures.csv"
  num_envs: int = 256
  batches: int = 32
  steps: int = 150
  seed: int = 2810
  device: str = "cuda:0"


def main(cfg: Cfg) -> None:
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  device = cfg.device

  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env_cfg.terminations.pop("fell_over", None)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=100.0)

  bundle = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
  policy = GoalkeeperMoE6Policy(bundle, env, device)
  ball = env.unwrapped.scene["ball"]
  origins = env.unwrapped.scene.env_origins

  rows = []
  region_total = [0] * 6
  region_fail = [0] * 6
  routed_total = [[0] * 6 for _ in range(6)]
  routed_fail = [[0] * 6 for _ in range(6)]

  with torch.inference_mode():
    for batch in range(cfg.batches):
      obs = env.reset()
      if isinstance(obs, tuple):
        obs = obs[0]
      policy.reset()
      true_region = getattr(env.unwrapped, "_gk_region").clone().long()
      start_pos = ball.data.root_link_pos_w.clone()
      start_vel = ball.data.root_link_lin_vel_w.clone()
      entered = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
      enter_step = torch.full((cfg.num_envs,), -1, dtype=torch.long, device=device)
      min_goal_x = torch.full((cfg.num_envs,), 99.0, dtype=torch.float32, device=device)
      final_pos = start_pos.clone()
      final_vel = start_vel.clone()

      for step in range(cfg.steps):
        obs = env.step(policy(obs))[0]
        pos = ball.data.root_link_pos_w
        vel = ball.data.root_link_lin_vel_w
        bx = pos[:, 0] - origins[:, 0]
        by = pos[:, 1] - origins[:, 1]
        in_goal = (bx <= -0.5) & (by.abs() <= 1.5) & (pos[:, 2] <= 1.8)
        new_enter = in_goal & ~entered
        enter_step = torch.where(new_enter, torch.full_like(enter_step, step), enter_step)
        entered |= in_goal
        min_goal_x = torch.minimum(min_goal_x, bx)
        final_pos = pos.clone()
        final_vel = vel.clone()

      used_region = torch.where(policy.latched < 0, torch.zeros_like(policy.latched), policy.latched)
      for idx in range(cfg.num_envs):
        tr = int(true_region[idx])
        ur = int(used_region[idx])
        fail = bool(entered[idx])
        region_total[tr] += 1
        region_fail[tr] += int(fail)
        routed_total[tr][ur] += 1
        routed_fail[tr][ur] += int(fail)
        if fail:
          sp = start_pos[idx] - origins[idx]
          sv = start_vel[idx]
          fp = final_pos[idx] - origins[idx]
          fv = final_vel[idx]
          rows.append({
            "batch": batch,
            "env": idx,
            "true_region": tr,
            "true_region_name": _REGION_NAMES[tr],
            "used_region": ur,
            "used_region_name": _REGION_NAMES[ur],
            "enter_step": int(enter_step[idx]),
            "min_ball_x": float(min_goal_x[idx]),
            "start_x": float(sp[0]),
            "start_y": float(sp[1]),
            "start_z": float(sp[2]),
            "vel_x": float(sv[0]),
            "vel_y": float(sv[1]),
            "vel_z": float(sv[2]),
            "final_x": float(fp[0]),
            "final_y": float(fp[1]),
            "final_z": float(fp[2]),
            "final_vx": float(fv[0]),
            "final_vy": float(fv[1]),
            "final_vz": float(fv[2]),
          })
      print(
        f"batch {batch + 1}/{cfg.batches}: "
        f"failures={len(rows)} total={(batch + 1) * cfg.num_envs}",
        flush=True,
      )

  out = Path(cfg.out_csv)
  out.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = list(rows[0].keys()) if rows else [
    "batch", "env", "true_region", "true_region_name", "used_region",
    "used_region_name", "enter_step", "min_ball_x", "start_x", "start_y",
    "start_z", "vel_x", "vel_y", "vel_z", "final_x", "final_y", "final_z",
    "final_vx", "final_vy", "final_vz",
  ]
  with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

  total = sum(region_total)
  fails = sum(region_fail)
  print(f"\nFailures: {fails}/{total} = {100 * fails / max(1, total):.2f}%")
  for region in range(6):
    n = region_total[region]
    f = region_fail[region]
    print(f"{_REGION_NAMES[region]:<11}: fail {f}/{n} = {100 * f / max(1, n):.2f}%")
  print("\nWorst true->used routes:")
  route_rows = []
  for true in range(6):
    for used in range(6):
      n = routed_total[true][used]
      if n:
        route_rows.append((routed_fail[true][used] / n, routed_fail[true][used], n, true, used))
  for rate, f, n, true, used in sorted(route_rows, reverse=True)[:12]:
    print(
      f"{_REGION_NAMES[true]} -> {_REGION_NAMES[used]}: "
      f"fail {f}/{n} = {100 * rate:.2f}%"
    )
  print(f"\n[INFO] wrote failures to {out}")
  env.close()


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  import src.tasks.soccer.config.eval  # noqa: F401

  main(tyro.cli(Cfg, prog="collect_moe6_failures"))
