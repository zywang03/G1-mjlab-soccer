"""Evaluate a stable-save 6-way goalkeeper mixture-of-experts."""

from __future__ import annotations

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

_REGION_NAMES = ["Right-Mid", "Left-Mid", "Right-Up", "Left-Up", "Right-Low", "Left-Low"]


@dataclass
class Cfg:
  experts: tuple[str, ...] = ()
  expert_dir: str = "logs/lyk/experts"
  prefix: str = "stable_sr"
  default_ckpt: str = ""
  mirror_map: str = ""
  num_envs: int = 256
  batches: int = 16
  steps: int = 150
  episode_length_s: float = 0.0
  seed: int = 2810
  z_low: float = 0.85
  z_up: float = 1.35
  land_x: float = 0.0
  latch_hi: float = 5.0
  final_ang_vel_xy: float = 3.0
  upright_gravity_z: float = -0.35
  device: str = "cuda:0"


def _expert_paths(cfg: Cfg) -> tuple[str, ...]:
  if cfg.experts:
    if len(cfg.experts) != 6:
      raise ValueError("--experts must provide exactly 6 checkpoint paths")
    return cfg.experts
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
  out: list[tuple[int, int]] = []
  for item in raw.split(","):
    dst, src = (int(x) for x in item.split(":"))
    if not (0 <= dst < 6 and 0 <= src < 6):
      raise ValueError(f"mirror_map entries must be in [0, 5], got {item}")
    out.append((dst, src))
  return out


def main(cfg: Cfg) -> None:
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  device = cfg.device

  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  if cfg.episode_length_s > 0.0:
    env_cfg.episode_length_s = cfg.episode_length_s
  if "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=100.0)
  steps = cfg.steps
  max_steps = int(getattr(env.unwrapped, "max_episode_length", 0))
  if max_steps > 0 and steps >= max_steps:
    steps = max(1, max_steps - 1)
    print(
      f"[WARN] --steps reached timeout ({max_steps}); using {steps}. "
      "Set --episode-length-s larger for long recovery eval.",
      flush=True,
    )

  experts = [_load_policy(env, path, device) for path in _expert_paths(cfg)]
  if cfg.mirror_map:
    from src.tasks.soccer.modules.symmetry import mirror_action, mirror_obs

    base_experts = list(experts)

    def mirror_policy(policy):
      def wrapped(obs):
        return mirror_action(policy({"actor": mirror_obs(obs["actor"])}))
      return wrapped

    for dst, src in _parse_mirror_map(cfg.mirror_map):
      experts[dst] = mirror_policy(base_experts[src])
      print(f"[INFO] expert {dst} := mirror(expert {src})", flush=True)

  default_policy = _load_policy(env, cfg.default_ckpt, device) if cfg.default_ckpt else None

  ball = env.unwrapped.scene["ball"]
  robot = env.unwrapped.scene["robot"]
  origins = env.unwrapped.scene.env_origins
  num_envs = cfg.num_envs
  gravity = 9.81

  def gate_region():
    ball_pos = ball.data.root_link_pos_w
    ball_vel = ball.data.root_link_lin_vel_w
    ball_x = ball_pos[:, 0] - origins[:, 0]
    ball_y = ball_pos[:, 1] - origins[:, 1]
    vx = ball_vel[:, 0]
    valid = (vx < -1.0) & (ball_x > 0.2) & (ball_x < cfg.latch_hi)
    t = torch.clamp(-(ball_x - cfg.land_x) / (vx - 1.0e-3), 0.0, 2.0)
    cross_y = ball_y + ball_vel[:, 1] * t
    cross_z = ball_pos[:, 2] + ball_vel[:, 2] * t - 0.5 * gravity * t * t
    base = torch.full_like(cross_y, 0, dtype=torch.long)
    base = torch.where(cross_z < cfg.z_low, torch.full_like(base, 4), base)
    base = torch.where(cross_z > cfg.z_up, torch.full_like(base, 2), base)
    return base + (cross_y < 0).long(), valid

  total = blocked = stable = upright = 0
  rec: list[tuple[int, int, bool, bool]] = []
  with torch.inference_mode():
    for batch in range(cfg.batches):
      obs = env.reset()
      if isinstance(obs, tuple):
        obs = obs[0]
      latched = torch.full((num_envs,), -1, dtype=torch.long, device=device)
      entered = torch.zeros(num_envs, dtype=torch.bool, device=device)
      true_region = (
        env.unwrapped._gk_region.clone().long()
        if hasattr(env.unwrapped, "_gk_region")
        else torch.zeros(num_envs, dtype=torch.long, device=device)
      )
      for _ in range(steps):
        region, valid = gate_region()
        new_latch = valid & (latched < 0)
        latched = torch.where(new_latch, region, latched)
        use_region = torch.where(latched < 0, torch.zeros_like(latched), latched)
        actions = torch.stack([expert(obs) for expert in experts], dim=0)
        action = actions[use_region, torch.arange(num_envs, device=device)]
        if default_policy is not None:
          action = torch.where((latched < 0).unsqueeze(1), default_policy(obs), action)
        obs = env.step(action)[0]
        ball_pos = ball.data.root_link_pos_w
        ball_x = ball_pos[:, 0] - origins[:, 0]
        ball_y = ball_pos[:, 1] - origins[:, 1]
        entered |= (ball_x <= -0.5) & (ball_y.abs() <= 1.5) & (ball_pos[:, 2] <= 1.8)

      blocked_mask = ~entered
      upright_mask = robot.data.projected_gravity_b[:, 2] < cfg.upright_gravity_z
      final_ang_vel = torch.linalg.norm(robot.data.root_link_ang_vel_b[:, :2], dim=-1)
      stable_mask = blocked_mask & upright_mask & (final_ang_vel < cfg.final_ang_vel_xy)
      use_region = torch.where(latched < 0, torch.zeros_like(latched), latched)

      total += num_envs
      blocked += int(blocked_mask.sum())
      upright += int(upright_mask.sum())
      stable += int(stable_mask.sum())
      for idx in range(num_envs):
        rec.append((
          int(true_region[idx]),
          int(use_region[idx]),
          bool(blocked_mask[idx]),
          bool(stable_mask[idx]),
        ))
      print(
        f"  batch {batch + 1}/{cfg.batches}: "
        f"stable {100 * stable / total:.1f}% block {100 * blocked / total:.1f}%",
        flush=True,
      )

  print(f"\nMoE6 stable-save: {stable}/{total} = {100 * stable / total:.1f}%")
  print(f"MoE6 block:       {blocked}/{total} = {100 * blocked / total:.1f}%")
  print(f"Final upright:    {upright}/{total} = {100 * upright / total:.1f}%")
  routed_ok = sum(1 for true, used, _, _ in rec if true == used)
  print(f"Gate accuracy:    {100 * routed_ok / max(1, len(rec)):.1f}%")
  for region in range(6):
    rows = [row for row in rec if row[0] == region]
    if not rows:
      continue
    block_rate = 100 * sum(row[2] for row in rows) / len(rows)
    stable_rate = 100 * sum(row[3] for row in rows) / len(rows)
    print(f"  {_REGION_NAMES[region]:<11} stable {stable_rate:5.1f}% block {block_rate:5.1f}% n={len(rows)}")
  env.close()


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  import src.tasks.soccer.config.eval  # noqa: F401

  main(tyro.cli(Cfg, prog="eval_stable_moe6"))
