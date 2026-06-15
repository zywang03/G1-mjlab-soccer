"""Evaluate shooter with multiple environments in parallel.

Same metrics and goal-detection logic as ``eval_naive_shooter.py``
(the official Phase 1 evaluation script), but runs ``--num-envs``
environments concurrently for ~25× throughput.

Usage:
  # 50 trials, 32 parallel envs (= 2 batches)
  python scripts/eval_shooter_parallel.py \\
      --checkpoint logs/rsl_rl/g1_soccer/<run>/model_100000.pt \\
      --headless --num-trials 50 --num-envs 32

  # Zero-agent baseline
  python scripts/eval_shooter_parallel.py --headless --num-trials 50 --num-envs 16

  # Interactive viewer (forces num-envs=1)
  python scripts/eval_shooter_parallel.py \\
      --checkpoint <model.pt> --viewer native --num-trials 1
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.soccer.config.g1.rl_cfg import (
  SoccerRecurrentRunner,
  unitree_g1_soccer_recurrent_runner_cfg,
)

_GOAL_HALF_WIDTH = 1.5
_GOAL_HEIGHT = 1.8
_GOAL_Y = -5.0
_GOAL_CENTER = (0.0, -5.0, 0.9)
_KICK_SPEED_THRESHOLD = 1.0


@dataclass
class EvalConfig:
  checkpoint: str | None = None
  task_id: str = "Eval-Shooter"
  device: str | None = None
  seed: int = 2810
  headless: bool = False
  num_trials: int = 0
  num_envs: int = 32
  viewer: str = "auto"
  video: bool = False
  video_length: int = 500
  video_height: int = 480
  video_width: int = 640
  summary_json: str | None = None


class ZeroPolicy:
  def __init__(self, action_shape, device: str):
    self._action = torch.zeros(action_shape, device=device)

  def __call__(self, obs):
    del obs
    return self._action

  def reset(self, dones=None):
    pass


def _load_policy(checkpoint_path: str, env, device: str):
  print(f"[INFO] Loading policy from: {checkpoint_path}")
  agent_cfg = unitree_g1_soccer_recurrent_runner_cfg()
  runner = SoccerRecurrentRunner(env, asdict(agent_cfg), log_dir=None, device=device)
  runner.load(checkpoint_path)
  policy = runner.get_inference_policy(device=env.unwrapped.device)
  print("[INFO] Policy loaded successfully.")
  return policy


# -- Vectorised metrics -------------------------------------------------------


def _is_goal(ball_pos: torch.Tensor) -> torch.Tensor:
  """Return bool mask: ball has crossed the goal plane inside the frame."""
  return (
    (ball_pos[:, 1] <= _GOAL_Y)
    & (torch.abs(ball_pos[:, 0]) <= _GOAL_HALF_WIDTH)
    & (ball_pos[:, 2] >= 0.0)
    & (ball_pos[:, 2] <= _GOAL_HEIGHT)
  )


def _kick_accuracy_vec(ball_vel: torch.Tensor, ball_pos: torch.Tensor) -> torch.Tensor:
  """Cosine similarity between ball vel (XY) and ball→goal-centre vector.

  Batched: each row is one environment.
  """
  v_xy = ball_vel[:, :2]
  target_xy = torch.stack([
    _GOAL_CENTER[0] - ball_pos[:, 0],
    _GOAL_CENTER[1] - ball_pos[:, 1],
  ], dim=-1)
  v_norm = torch.linalg.vector_norm(v_xy, dim=-1)
  t_norm = torch.linalg.vector_norm(target_xy, dim=-1)
  valid = (v_norm > 1e-6) & (t_norm > 1e-6)
  cos = torch.zeros(ball_pos.shape[0], dtype=torch.float32, device=ball_pos.device)
  cos[valid] = torch.sum(v_xy[valid] * target_xy[valid], dim=-1) / (v_norm[valid] * t_norm[valid])
  return cos


def _run_batch(env, policy, max_steps: int, num_envs: int) -> dict[str, torch.Tensor]:
  """Run one batch of trials and return per-env metric tensors.

  Uses local coords (ball_world - env_origins) for goal-line checks so
  multi-env parallel eval is correct.
  """
  obs = env.reset()
  if isinstance(obs, tuple):
    obs = obs[0]
  if hasattr(policy, "reset"):
    policy.reset()

  base_env = env.unwrapped
  ball = base_env.scene["ball"]
  device = base_env.device
  env_origins = base_env.scene.env_origins

  command = base_env.command_manager.get_term("motion")
  target_x = command.target_destination_pos[:, 0].detach().clone()

  kicked = torch.zeros(num_envs, dtype=torch.bool, device=device)
  kick_speed = torch.zeros(num_envs, dtype=torch.float32, device=device)
  kick_acc = torch.zeros(num_envs, dtype=torch.float32, device=device)
  abs_z_speed = torch.zeros(num_envs, dtype=torch.float32, device=device)
  goal_scored = torch.zeros(num_envs, dtype=torch.bool, device=device)
  ball_final_y = torch.zeros(num_envs, dtype=torch.float32, device=device)
  step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
  terminated_flag = torch.zeros(num_envs, dtype=torch.bool, device=device)

  cross_recorded = torch.zeros(num_envs, dtype=torch.bool, device=device)
  target_error = torch.full((num_envs,), float("inf"), dtype=torch.float32, device=device)
  cross_pos_z = torch.zeros(num_envs, dtype=torch.float32, device=device)
  active = torch.ones(num_envs, dtype=torch.bool, device=device)

  prev_ball_local = ball.data.root_link_pos_w.detach().clone() - env_origins

  for _ in range(max_steps):
    step_active = active.clone()

    with torch.inference_mode():
      action = policy(obs)
    result = env.step(action)
    obs = result[0]
    terminated = result[2]
    if not isinstance(terminated, torch.Tensor):
      terminated = torch.as_tensor(terminated, device=device)
    terminated = terminated.to(device=device, dtype=torch.bool).view(-1)

    step_count[step_active] += 1

    ball_pos_w = ball.data.root_link_pos_w.detach().clone()
    ball_local = ball_pos_w - env_origins
    ball_vel = ball.data.root_link_vel_w.detach().clone()[:, :3]
    speed = torch.linalg.vector_norm(ball_vel, dim=-1)

    new_kick = step_active & (~kicked) & (speed > _KICK_SPEED_THRESHOLD)
    if torch.any(new_kick):
      kicked[new_kick] = True
      kick_speed[new_kick] = speed[new_kick]
      kick_acc[new_kick] = _kick_accuracy_vec(ball_vel, ball_local)[new_kick]
      abs_z_speed[new_kick] = ball_vel[new_kick, 2].abs()

    new_goal = step_active & _is_goal(ball_local)
    if torch.any(new_goal):
      goal_scored[new_goal] = True

    ball_final_y[step_active] = ball_local[:, 1][step_active]

    # Goal-plane crossing detection (local coords).
    can_cross = step_active & (~cross_recorded)
    if torch.any(can_cross):
      crossed = can_cross & (prev_ball_local[:, 1] > _GOAL_Y) & (ball_local[:, 1] <= _GOAL_Y)
      if torch.any(crossed):
        dy = ball_local[:, 1] - prev_ball_local[:, 1]
        safe = torch.where(dy >= 0, torch.tensor(1e-6, device=device),
                           torch.tensor(-1e-6, device=device))
        alpha = ((_GOAL_Y - prev_ball_local[:, 1]) / (dy + safe)).clamp(0.0, 1.0).unsqueeze(-1)
        cross_pos = prev_ball_local + alpha * (ball_local - prev_ball_local)
        target_error[crossed] = (cross_pos[crossed, 0] - target_x[crossed]).abs()
        cross_pos_z[crossed] = cross_pos[crossed, 2]
        cross_recorded[crossed] = True

    new_term = step_active & terminated
    if torch.any(new_term):
      terminated_flag[new_term] = True
      if hasattr(policy, "reset"):
        with torch.inference_mode():
          policy.reset(dones=new_term)

    prev_ball_local[step_active] = ball_local[step_active]
    active = step_active & (~terminated)
    if not torch.any(active):
      break

  return {
    "goal": goal_scored.cpu(),
    "kick_speed": kick_speed.cpu(),
    "kick_accuracy": kick_acc.cpu(),
    "abs_z_speed": abs_z_speed.cpu(),
    "ball_final_y": ball_final_y.cpu(),
    "steps": step_count.cpu(),
    "terminated": terminated_flag.cpu(),
    "cross_recorded": cross_recorded.cpu(),
    "target_error": target_error.cpu(),
    "cross_pos_z": cross_pos_z.cpu(),
  }


def _summarise(metrics: dict[str, torch.Tensor]) -> dict:
  """Aggregate per-env metrics across all batches."""
  total = int(metrics["goal"].numel())
  goals = int(metrics["goal"].sum().item())
  success_rate = goals / total * 100.0 if total > 0 else 0.0

  kicked_mask = metrics["kick_speed"] > 0.0
  if torch.any(kicked_mask):
    acc_values = metrics["kick_accuracy"][kicked_mask].float().numpy()
    mean_acc = float(np.mean(acc_values))
    std_acc = float(np.std(acc_values))
    mean_speed = float(metrics["kick_speed"][kicked_mask].float().mean().item())
    mean_abs_z = float(metrics["abs_z_speed"][kicked_mask].float().mean().item())
  else:
    mean_acc = 0.0
    std_acc = 0.0
    mean_speed = 0.0
    mean_abs_z = 0.0

  ball_past = int((metrics["ball_final_y"] <= _GOAL_Y).sum().item())
  crossed = int(metrics["cross_recorded"].sum().item())

  finite = torch.isfinite(metrics["target_error"])
  if torch.any(finite):
    target_err_np = metrics["target_error"][finite].float().numpy()
    mean_target_err = float(np.mean(target_err_np))
    median_target_err = float(np.median(target_err_np))
    hit_0_2m = float((target_err_np <= 0.2).mean()) * 100.0
    hit_0_3m = float((target_err_np <= 0.3).mean()) * 100.0
  else:
    mean_target_err = 0.0
    median_target_err = 0.0
    hit_0_2m = 0.0
    hit_0_3m = 0.0

  crossed_z = metrics["cross_pos_z"] > 0.0
  mean_cross_z = float(metrics["cross_pos_z"][crossed_z].float().mean().item()) if torch.any(crossed_z) else 0.0

  return {
    "total": total,
    "goals": goals,
    "success_rate": success_rate,
    "mean_kick_accuracy": mean_acc,
    "std_kick_accuracy": std_acc,
    "mean_kick_speed": mean_speed,
    "ball_past": ball_past,
    "goal_plane_crossed": crossed,
    "mean_target_error": mean_target_err,
    "median_target_error": median_target_err,
    "target_hit_rate_0_2m": hit_0_2m,
    "target_hit_rate_0_3m": hit_0_3m,
    "mean_cross_z": mean_cross_z,
    "mean_abs_z_speed": mean_abs_z,
  }


# -- Viewer -------------------------------------------------------------------


def run_viewer(cfg: EvalConfig, env, policy):
  import os
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  viewer_type = cfg.viewer
  if viewer_type == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    viewer_type = "native" if has_display else "viser"

  if viewer_type == "native":
    NativeMujocoViewer(env, policy).run()
  elif viewer_type == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer: {viewer_type}")


# -- Main ---------------------------------------------------------------------


def run_eval(cfg: EvalConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Viewer mode uses single env.
  use_viewer = not cfg.headless and cfg.num_trials <= 0
  num_envs = 1 if use_viewer else min(cfg.num_envs, cfg.num_trials or 1)

  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.scene.num_envs = num_envs
  env_cfg.viewer.height = cfg.video_height
  env_cfg.viewer.width = cfg.video_width

  actor_terms = list(env_cfg.observations["actor"].terms.keys())
  print(f"Task: {cfg.task_id}")
  print(f"Actor obs ({len(actor_terms)} terms): {actor_terms}")
  print(f"Terminations: {list(env_cfg.terminations.keys())}")
  print(f"Episode length: {env_cfg.episode_length_s}s")
  print(f"Environments: {num_envs}")

  render_mode = "rgb_array" if cfg.video else None
  env_base = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if cfg.video:
    from pathlib import Path
    from mjlab.utils.wrappers import VideoRecorder
    video_folder = Path("videos") / "eval"
    video_folder.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Recording video to: {video_folder}")
    env_base = VideoRecorder(env_base, video_folder=video_folder,
                              step_trigger=lambda step: step == 0,
                              video_length=cfg.video_length, disable_logger=True)

  env = RslRlVecEnvWrapper(env_base, clip_actions=100.0)

  if cfg.checkpoint:
    policy = _load_policy(cfg.checkpoint, env, device)
  else:
    action_shape = env.unwrapped.action_space.shape
    policy = ZeroPolicy(action_shape, device)
    print("[INFO] Using zero-agent fallback (no checkpoint provided).")

  obs_space = env.unwrapped.single_observation_space
  print(f"Actor obs dim:  {obs_space.spaces['actor'].shape}")
  print(f"Critic obs dim: {obs_space.spaces['critic'].shape}")

  if use_viewer:
    if cfg.num_trials > 0:
      print("[INFO] --num-trials set without --headless; running viewer.")
    run_viewer(cfg, env, policy)
    env.close()
    return

  # Headless evaluation with parallel environments.
  if cfg.num_trials <= 0:
    print("[WARN] --headless without --num-trials: nothing to evaluate.")
    env.close()
    return

  print(f"\n[INFO] Running {cfg.num_trials} headless trials ({num_envs} envs/batch)...\n")

  # Collect metrics across batches.
  all_metrics: dict[str, list[torch.Tensor]] = {}
  remaining = cfg.num_trials
  trial_offset = 0

  while remaining > 0:
    env_base.seed = cfg.seed + trial_offset  # unique seed per batch for diversity
    take = min(num_envs, remaining)
    metrics = _run_batch(env, policy, max_steps=500, num_envs=num_envs)
    metrics = {k: v[:take] for k, v in metrics.items()}
    for k, v in metrics.items():
      all_metrics.setdefault(k, []).append(v)

    interval = 1 if cfg.num_trials <= 10 else (cfg.num_trials // 10)
    for i in range(take):
      t = trial_offset + i + 1
      if t % interval == 0 or t == 1:
        g = metrics["goal"][i].item()
        s = metrics["kick_speed"][i].item()
        a = metrics["kick_accuracy"][i].item()
        steps = metrics["steps"][i].item()
        term = metrics["terminated"][i].item()
        print(f"  Trial {t:3d}/{cfg.num_trials}: "
              f"goal={g}, speed={s:.2f}, acc={a:.3f}, steps={steps}, term={term}")

    trial_offset += take
    remaining -= take

  merged = {k: torch.cat(v, dim=0) for k, v in all_metrics.items()}
  summary = _summarise(merged)

  print(f"\n{'='*55}")
  print(f"  Eval Summary ({summary['total']} trials)")
  print(f"{'='*55}")
  print(f"  Success Rate:            {summary['goals']}/{summary['total']} = {summary['success_rate']:.1f}%")
  print(f"  Kick Accuracy (cos):    {summary['mean_kick_accuracy']:.4f} ± {summary['std_kick_accuracy']:.4f}")
  print(f"  Mean Kick Speed:         {summary['mean_kick_speed']:.2f} m/s")
  print(f"  Ball past goal line:     {summary['ball_past']}/{summary['total']}")
  print(f"  Goal-plane crossed:      {summary['goal_plane_crossed']}/{summary['total']}")
  print(f"  Mean Target Error:       {summary['mean_target_error']:.3f} m")
  print(f"  Median Target Error:     {summary['median_target_error']:.3f} m")
  print(f"  Target Hit Rate ≤0.2m:  {summary['target_hit_rate_0_2m']:.1f}%")
  print(f"  Target Hit Rate ≤0.3m:  {summary['target_hit_rate_0_3m']:.1f}%")
  print(f"  Mean Cross Z:            {summary['mean_cross_z']:.3f} m")
  print(f"  Mean |Z-Speed|:          {summary['mean_abs_z_speed']:.3f} m/s")
  print(f"{'='*55}\n")

  if cfg.summary_json:
    import json
    from pathlib import Path
    out = Path(cfg.summary_json)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote summary to: {out.resolve()}")

  env.close()


def main():
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  args = tyro.cli(EvalConfig, prog="eval_shooter_parallel")
  run_eval(args)


if __name__ == "__main__":
  main()
