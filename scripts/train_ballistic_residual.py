"""Train a ballistic-feature residual goalkeeper.

This is the high-value keeper improvement path for ``keeper-lyk``:
freeze the distilled goalkeeper and train only a small residual head.  The
residual observes the same 960D actor history, but internally augments it with
ballistic features predicted from the ball-position history.  PPO therefore
learns timing/reach corrections without destroying the base diving skill.
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.soccer import mdp
from src.tasks.soccer.config.g1.gk_train_cfg import (
  goalkeeper_ballistic_residual_runner_cfg,
)
from src.tasks.soccer.mdp.goalkeeper_rewards import (
  _reset_gk_state,
  goalkeeper_ang_vel_xy,
  goalkeeper_body_intercept,
  goalkeeper_feet_slippage,
  goalkeeper_goal_conceded,
  goalkeeper_intercept_point,
  goalkeeper_no_retreat,
  goalkeeper_posture_orientation,
  goalkeeper_recovery_upright,
  goalkeeper_stay_on_line,
  goalkeeper_stop_ball,
)
from src.tasks.soccer.mdp.shooter_rewards import action_rate_l2_clip


@dataclass
class Cfg:
  base: str = "src/assets/soccer/weight/goalkeeper_distilled_v3.pt"
  init: str = ""
  out: str = "logs/lyk/goalkeeper_ballistic_residual.pt"
  num_envs: int = 1024
  warmup: int = 30
  block_iters: int = 20
  blocks: int = 40
  eval_resets: int = 3
  lr: float = 1.0e-4
  std: float = 0.06
  residual_scale: float = 0.0
  region_weights: tuple[float, ...] = ()
  bc_data: tuple[str, ...] = ()
  bc_coef: float = 0.0
  bc_coef_final: float = 0.0
  bc_steps: int = 1
  bc_batch: int = 8192
  bc_max_frames_per_file: int = 0
  bc_max_frames: int = 0
  w_conceded: float = 15.0
  w_intercept: float = 3.0
  w_body: float = 2.0
  w_stop: float = 1.0
  w_posture: float = 1.0
  w_recovery: float = 2.0
  w_line: float = 0.3
  w_no_retreat: float = 0.5
  w_feet_slip: float = 0.05
  w_ang_vel: float = 0.02
  action_rate: float = 0.05
  clip_param: float = 0.08
  desired_kl: float = 0.004
  rollback_drop: float = 0.01
  seed: int = 2810
  device: str = "cuda:0"


def _vel_cfg(env):
  return env.unwrapped.cfg.events["reset_ball"].params["vel_cfg"]


def _set_region_weights(env, weights: tuple[float, ...]) -> tuple[float, ...]:
  vc = _vel_cfg(env)
  old = tuple(getattr(vc, "region_weights", ()))
  vc.region_weights = tuple(weights)
  return old


def _eval(env, policy, ball, n_steps: int = 150, n_resets: int = 3) -> float:
  old_weights = _set_region_weights(env, ())
  num_envs = env.unwrapped.num_envs
  origins = env.unwrapped.scene.env_origins
  blocked = 0
  total = 0
  try:
    with torch.inference_mode():
      for _ in range(n_resets):
        obs, _ = env.reset()
        entered = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)
        for _ in range(n_steps):
          obs = env.step(policy(obs))[0]
          ball_pos = ball.data.root_link_pos_w
          entered |= (
            ((ball_pos[:, 0] - origins[:, 0]) <= -0.5)
            & ((ball_pos[:, 1] - origins[:, 1]).abs() <= 1.5)
            & (ball_pos[:, 2] <= 1.8)
          )
        blocked += int((~entered).sum())
        total += num_envs
    return blocked / max(1, total)
  finally:
    _set_region_weights(env, old_weights)


def _subsample(obs, act, max_frames: int, gen: torch.Generator):
  if max_frames <= 0 or obs.shape[0] <= max_frames:
    return obs, act
  idx = torch.randperm(obs.shape[0], generator=gen)[:max_frames]
  return obs[idx], act[idx]


def _load_bc_data(cfg: Cfg) -> tuple[torch.Tensor, torch.Tensor] | None:
  if not cfg.bc_data or cfg.bc_coef <= 0.0:
    return None
  gen = torch.Generator(device="cpu")
  gen.manual_seed(cfg.seed)
  obs_parts, act_parts = [], []
  for path in cfg.bc_data:
    data = torch.load(path, map_location="cpu", weights_only=False)
    obs = data["obs"]
    act = data["act"]
    if "blocked" in data:
      mask = data["blocked"].bool()
      obs = obs[mask]
      act = act[mask]
    obs, act = _subsample(obs, act, cfg.bc_max_frames_per_file, gen)
    obs_parts.append(obs)
    act_parts.append(act)
    print(f"[INFO] BC anchor loaded {path}: {obs.shape[0]} frames", flush=True)
  obs = torch.cat(obs_parts)
  act = torch.cat(act_parts)
  obs, act = _subsample(obs, act, cfg.bc_max_frames, gen)
  print(f"[INFO] BC anchor total {obs.shape[0]} frames", flush=True)
  return obs, act


def _configure_residual_init(cfg: Cfg) -> tuple[str, float, bool]:
  """Return frozen-base path, residual scale, and whether to load cfg.init as actor."""
  if not cfg.init:
    scale = cfg.residual_scale if cfg.residual_scale > 0.0 else 0.25
    return cfg.base, scale, False

  ck = torch.load(cfg.init, map_location="cpu", weights_only=False)
  meta = ck.get("ballistic_residual")
  if meta:
    scale = cfg.residual_scale if cfg.residual_scale > 0.0 else float(meta.get("residual_scale", 0.45))
    return meta.get("base", cfg.base), scale, True

  scale = cfg.residual_scale if cfg.residual_scale > 0.0 else 0.25
  return cfg.init, scale, False


def main(cfg: Cfg) -> None:
  configure_torch_backends()
  dev = cfg.device

  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  if "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None
  env_cfg.rewards = {
    "goal_conceded": RewardTermCfg(
      func=goalkeeper_goal_conceded,
      weight=-cfg.w_conceded,
      params={},
    ),
    "intercept": RewardTermCfg(
      func=goalkeeper_intercept_point,
      weight=cfg.w_intercept,
      params={"std": 0.35},
    ),
    "body": RewardTermCfg(
      func=goalkeeper_body_intercept,
      weight=cfg.w_body,
      params={"std": 0.30},
    ),
    "stop_ball": RewardTermCfg(
      func=goalkeeper_stop_ball,
      weight=cfg.w_stop,
      params={"velocity_drop_threshold": 2.0, "goal_x": -0.5},
    ),
    "posture": RewardTermCfg(func=goalkeeper_posture_orientation, weight=cfg.w_posture),
    "recovery_upright": RewardTermCfg(func=goalkeeper_recovery_upright, weight=-cfg.w_recovery),
    "stay_on_line": RewardTermCfg(func=goalkeeper_stay_on_line, weight=-cfg.w_line),
    "no_retreat": RewardTermCfg(func=goalkeeper_no_retreat, weight=-cfg.w_no_retreat),
    "feet_slippage": RewardTermCfg(func=goalkeeper_feet_slippage, weight=-cfg.w_feet_slip),
    "ang_vel_xy": RewardTermCfg(func=goalkeeper_ang_vel_xy, weight=-cfg.w_ang_vel),
    "action_rate": RewardTermCfg(func=action_rate_l2_clip, weight=-cfg.action_rate),
  }
  env_cfg.events["reset_gk_state"] = EventTermCfg(
    func=_reset_gk_state,
    mode="reset",
    params={},
  )
  env_cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=dev),
    clip_actions=100.0,
  )

  import src.tasks.soccer.modules.gk_ballistic_residual as gkbr

  base_ckpt, residual_scale, load_init_actor = _configure_residual_init(cfg)
  gkbr.BASE_CKPT = base_ckpt
  gkbr.BASE_HIDDEN = (1024, 512, 256)
  gkbr.RESIDUAL_SCALE = residual_scale

  agent = goalkeeper_ballistic_residual_runner_cfg()
  agent.algorithm.class_name = "src.tasks.soccer.modules.bc_anchor_ppo.BCAnchorPPO"
  agent.algorithm.learning_rate = cfg.lr
  agent.algorithm.entropy_coef = 0.0
  agent.algorithm.clip_param = cfg.clip_param
  agent.algorithm.desired_kl = cfg.desired_kl
  agent.actor.distribution_cfg["init_std"] = cfg.std
  runner = MjlabOnPolicyRunner(env, asdict(agent), device=dev)
  if load_init_actor:
    runner.load(cfg.init, load_cfg={"actor": True})
    print(f"[INFO] residual actor init from {cfg.init}", flush=True)
  else:
    print(f"[INFO] frozen base init from {base_ckpt}", flush=True)
  alg = runner.alg
  alg.std_clamp = cfg.std
  alg.bc_coef = 0.0
  alg.bc_steps = cfg.bc_steps
  alg.bc_batch = cfg.bc_batch
  bc = _load_bc_data(cfg)
  if bc is not None:
    alg.bc_obs = bc[0].to(dev, non_blocking=True)
    alg.bc_act = bc[1].to(dev, non_blocking=True)
  with torch.no_grad():
    alg.actor.distribution.std_param.fill_(cfg.std)
  if cfg.region_weights:
    _set_region_weights(env, cfg.region_weights)
    print(f"[INFO] training region_weights={cfg.region_weights}; eval remains uniform", flush=True)

  ball = env.unwrapped.scene["ball"]
  policy = runner.get_inference_policy(device=dev)

  if cfg.warmup > 0:
    for param in alg.actor.residual.parameters():
      param.requires_grad_(False)
    bc0 = alg.bc_coef
    alg.bc_coef = 0.0
    runner.learn(num_learning_iterations=cfg.warmup, init_at_random_ep_len=True)
    for param in alg.actor.residual.parameters():
      param.requires_grad_(True)
    alg.bc_coef = bc0
    with torch.no_grad():
      alg.actor.distribution.std_param.fill_(cfg.std)

  best = _eval(env, policy, ball, n_resets=cfg.eval_resets)
  best_state = copy.deepcopy(alg.actor.state_dict())
  print(f"[EVAL] init block {100 * best:.1f}%", flush=True)
  for block in range(cfg.blocks):
    if cfg.bc_coef > 0.0 and alg.bc_obs is not None:
      frac = block / max(1, cfg.blocks - 1)
      alg.bc_coef = cfg.bc_coef + (cfg.bc_coef_final - cfg.bc_coef) * frac
    runner.learn(num_learning_iterations=cfg.block_iters, init_at_random_ep_len=False)
    with torch.no_grad():
      alg.actor.distribution.std_param.clamp_(min=1.0e-3, max=cfg.std)
    rate = _eval(env, policy, ball, n_resets=cfg.eval_resets)
    tag = ""
    if rate >= best:
      best = rate
      best_state = copy.deepcopy(alg.actor.state_dict())
      tag = " *best*"
    elif rate < best - cfg.rollback_drop:
      alg.actor.load_state_dict(best_state)
      tag = " rollback"
    print(
      f"[EVAL] block {block + 1}/{cfg.blocks}: {100 * rate:.1f}% "
      f"(best {100 * best:.1f}%){tag}",
      flush=True,
    )

  alg.actor.load_state_dict(best_state)
  os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
  saved = alg.save()
  saved["iter"] = 0
  saved["infos"] = {"env_state": {"common_step_counter": 0}}
  saved["ballistic_residual"] = {
    "base": base_ckpt,
    "base_hidden": (1024, 512, 256),
    "residual_scale": residual_scale,
  }
  torch.save(saved, cfg.out)
  print(f"[INFO] saved ballistic residual (best {100 * best:.1f}%) to {cfg.out}")
  env.close()


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  main(tyro.cli(Cfg, prog="train_ballistic_residual"))
