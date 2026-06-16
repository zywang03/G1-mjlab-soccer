"""Render goalkeeper evaluation cases grouped by ball landing region."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio
import torch
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from src.tasks.soccer.config.g1.gk_train_cfg import (
  goalkeeper_ballistic_residual_runner_cfg,
  goalkeeper_train_runner_cfg,
)

import mjlab.tasks  # noqa: F401
import src.tasks.soccer.config.eval  # noqa: F401


_REGION_NAMES = ["Right-Mid", "Left-Mid", "Right-Up", "Left-Up", "Right-Low", "Left-Low"]


@dataclass
class Cfg:
  checkpoint: str = "src/assets/soccer/weight/model_repaired_lyk.pt"
  out_dir: str = "videos/keeper_regions"
  mode: str = "regions"  # "regions": forced per region, "random": official eval reset
  regions: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
  cases_per_region: int = 3
  random_cases: int = 24
  steps: int = 150
  width: int = 1280
  height: int = 720
  seed: int = 2810
  device: str = "cuda:0"


def _load_policy(checkpoint: str, env, device: str):
  loaded = torch.load(checkpoint, map_location=device, weights_only=False)
  meta = loaded.get("ballistic_residual")
  if meta:
    import src.tasks.soccer.modules.gk_ballistic_residual as gkbr

    gkbr.BASE_CKPT = meta.get("base")
    gkbr.BASE_HIDDEN = tuple(meta.get("base_hidden", (1024, 512, 256)))
    gkbr.RESIDUAL_SCALE = float(meta.get("residual_scale", 0.25))
    agent = goalkeeper_ballistic_residual_runner_cfg()
  else:
    agent = goalkeeper_train_runner_cfg()
  runner = MjlabOnPolicyRunner(env, asdict(agent), device=device)
  runner.load(checkpoint, load_cfg={"actor": True})
  return runner.get_inference_policy(device=device)


def _force_region(env, region: int, generator: torch.Generator):
  rb = env.unwrapped.cfg.events["reset_ball"]
  vc = rb.params["vel_cfg"]
  dev = env.unwrapped.device
  org = env.unwrapped.scene.env_origins
  u = lambda lo, hi: lo + torch.rand(1, generator=generator, device=dev) * (hi - lo)
  start = torch.stack(
    [
      u(*vc.ball_start_x_range),
      u(*vc.ball_start_y_range),
      u(*vc.ball_start_z_range),
    ],
    dim=-1,
  ).reshape(1, 3)
  reg = vc.regions[region]
  end = torch.stack(
    [
      -u(*vc.ball_end_x_range),
      u(reg["width"][0], reg["width"][1]),
      u(reg["height"][0], reg["height"][1]),
    ],
    dim=-1,
  ).reshape(1, 3)
  tf = u(*vc.t_flight_range).reshape(1)
  g = 9.81
  vel_xy = (end[:, :2] - start[:, :2]) / tf[:, None]
  vel_z = ((end[:, 2] - start[:, 2]) + 0.5 * g * tf * tf) / tf
  vel = torch.cat([vel_xy, vel_z[:, None]], dim=-1)
  env.unwrapped._gk_forced = {
    "start": start + org,
    "vel": vel,
    "region": torch.tensor([region], dtype=torch.float32, device=dev),
  }
  return start[0].detach().cpu(), end[0].detach().cpu(), tf[0].detach().cpu()


def _goal_entered(ball, org) -> bool:
  bp = ball.data.root_link_pos_w[0]
  bx = bp[0] - org[0]
  by = bp[1] - org[1]
  return bool(bx <= -0.5 and by.abs() <= 1.5 and bp[2] <= 1.8)


def _height_band(cross_z: float) -> str:
  if cross_z < 0.3:
    return "low"
  if cross_z >= 1.2:
    return "up"
  return "mid"


def _render_one(wrapped, env, policy, ball, org, steps: int):
  obs = wrapped.reset()
  if isinstance(obs, tuple):
    obs = obs[0]
  frames = []
  entered = False
  best_xabs = float("inf")
  cross_y = 0.0
  cross_z = 0.0
  for _ in range(steps):
    with torch.inference_mode():
      action = policy(obs)
    res = wrapped.step(action)
    obs = res[0]
    frames.append(env.render())
    bp = ball.data.root_link_pos_w[0]
    bx = float((bp[0] - org[0]).detach().cpu())
    by = float((bp[1] - org[1]).detach().cpu())
    bz = float(bp[2].detach().cpu())
    if abs(bx) < best_xabs:
      best_xabs = abs(bx)
      cross_y = by
      cross_z = bz
    entered = entered or _goal_entered(ball, org)
    if res[2].item():
      break
  return frames, entered, cross_y, cross_z


def main(cfg: Cfg) -> None:
  configure_torch_backends()
  if cfg.mode not in ("regions", "random"):
    raise ValueError("--mode must be 'regions' or 'random'")
  Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = cfg.seed
  env_cfg.viewer.width = cfg.width
  env_cfg.viewer.height = cfg.height
  if "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode="rgb_array")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=100.0)
  policy = _load_policy(cfg.checkpoint, wrapped, cfg.device)
  ball = env.scene["ball"]
  org = env.scene.env_origins[0]
  gen = torch.Generator(device=cfg.device)
  gen.manual_seed(cfg.seed)

  total_save = 0
  total = 0
  region_counts = {i: 0 for i in range(len(_REGION_NAMES))}
  band_counts = {"low": 0, "mid": 0, "up": 0}

  if cfg.mode == "regions":
    for region in cfg.regions:
      for case_idx in range(cfg.cases_per_region):
        start, end, tf = _force_region(wrapped, region, gen)
        frames, entered, cross_y, cross_z = _render_one(wrapped, env, policy, ball, org, cfg.steps)
        label = "goal" if entered else "save"
        band = _height_band(cross_z)
        total += 1
        total_save += int(not entered)
        region_counts[region] += 1
        band_counts[band] += 1
        name = _REGION_NAMES[region]
        path = Path(cfg.out_dir) / f"region{region}_{name}_case{case_idx:02d}_{label}.mp4"
        imageio.mimsave(path, frames, fps=30, macro_block_size=1)
        print(
          f"region {region} {name}: {label} start={start.tolist()} "
          f"end={end.tolist()} tf={float(tf):.2f} cross=({cross_y:+.2f},{cross_z:.2f}) "
          f"band={band} -> {path}",
          flush=True,
        )
  else:
    if hasattr(env, "_gk_forced"):
      delattr(env, "_gk_forced")
    for case_idx in range(cfg.random_cases):
      frames, entered, cross_y, cross_z = _render_one(wrapped, env, policy, ball, org, cfg.steps)
      region = int(getattr(env, "_gk_region", torch.zeros(1, device=cfg.device))[0].item())
      name = _REGION_NAMES[region] if 0 <= region < len(_REGION_NAMES) else f"Region-{region}"
      label = "goal" if entered else "save"
      band = _height_band(cross_z)
      total += 1
      total_save += int(not entered)
      if 0 <= region < len(_REGION_NAMES):
        region_counts[region] += 1
      band_counts[band] += 1
      path = Path(cfg.out_dir) / f"random_case{case_idx:02d}_region{region}_{name}_{label}.mp4"
      imageio.mimsave(path, frames, fps=30, macro_block_size=1)
      print(
        f"random case {case_idx:02d}: region {region} {name} {label} "
        f"cross=({cross_y:+.2f},{cross_z:.2f}) band={band} -> {path}",
        flush=True,
      )

  print(f"DONE: {total_save}/{total} saves rendered to {cfg.out_dir}", flush=True)
  print("Region counts:", flush=True)
  for region, count in region_counts.items():
    if count:
      print(f"  {region} {_REGION_NAMES[region]}: {count}", flush=True)
  print("Height bands at keeper plane:", flush=True)
  for band, count in band_counts.items():
    print(f"  {band}: {count}", flush=True)
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="render_goalkeeper_regions"))
