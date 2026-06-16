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
  regions: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
  cases_per_region: int = 3
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


def main(cfg: Cfg) -> None:
  configure_torch_backends()
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
  for region in cfg.regions:
    for case_idx in range(cfg.cases_per_region):
      start, end, tf = _force_region(wrapped, region, gen)
      obs = wrapped.reset()
      if isinstance(obs, tuple):
        obs = obs[0]
      frames = []
      entered = False
      for _ in range(cfg.steps):
        with torch.inference_mode():
          action = policy(obs)
        res = wrapped.step(action)
        obs = res[0]
        frames.append(env.render())
        bp = ball.data.root_link_pos_w[0]
        bx = bp[0] - org[0]
        by = bp[1] - org[1]
        if bx <= -0.5 and by.abs() <= 1.5 and bp[2] <= 1.8:
          entered = True
        if res[2].item():
          break
      label = "goal" if entered else "save"
      total += 1
      total_save += int(not entered)
      name = _REGION_NAMES[region]
      path = Path(cfg.out_dir) / f"region{region}_{name}_case{case_idx:02d}_{label}.mp4"
      imageio.mimsave(path, frames, fps=30, macro_block_size=1)
      print(
        f"region {region} {name}: {label} start={start.tolist()} "
        f"end={end.tolist()} tf={float(tf):.2f} -> {path}",
        flush=True,
      )

  print(f"DONE: {total_save}/{total} saves rendered to {cfg.out_dir}", flush=True)
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="render_goalkeeper_regions"))
