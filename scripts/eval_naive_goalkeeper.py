"""Evaluate goalkeeper — matches Humanoid-Goalkeeper paper eval protocol.

Runs the goalkeeper environment with a trained policy (or zero-agent fallback),
records video, and reports observation dimensions. In headless mode, runs
multiple trials and collects interception statistics (matching the paper's
evaluation protocol in Section IV).

Ball trajectory: 6-region parabolic model (matching paper's assign_ball_states).
Each episode randomly selects a region and samples a ball trajectory.

Usage:
  # Interactive viewer (zero agent)
  python scripts/eval_naive_goalkeeper.py

  # Interactive viewer (trained policy)
  python scripts/eval_naive_goalkeeper.py --checkpoint logs/rsl_rl/g1_soccer/model_5000.pt

  # Headless multi-trial eval with stats
  python scripts/eval_naive_goalkeeper.py --headless --num-trials=50
  python scripts/eval_naive_goalkeeper.py --headless --num-trials=500 --checkpoint <path>

  # With video
  python scripts/eval_naive_goalkeeper.py --video --video-length=300
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
  RslRlVecEnvWrapper,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from rsl_rl.runners import OnPolicyRunner

# ----- Paper eval thresholds (Section IV, default settings) -----
# Ball flight: 0.5-1.0s, distance 3-5m, goal 3.0×1.8m
# Success: ball blocked/intercepted (significant velocity drop when behind robot)

# Velocity drop threshold for block detection (m/s) — paper uses 2.0.
_BLOCK_VEL_DROP = 2.0
# Ball is "behind robot" (crossed the goal line) when x > 0.0 in world frame.
_BEHIND_ROBOT_X = 0.0


@dataclass
class EvalConfig:
  video: bool = False
  video_length: int = 300  # steps (6s at 50Hz, double the 3s episode)
  video_height: int = 480
  video_width: int = 640
  viewer: str = "auto"  # "auto", "native", "viser"
  device: str | None = None
  checkpoint: str | None = None  # path to .pt checkpoint file
  seed: int = 2810
  headless: bool = False   # run without viewer, collect stats
  num_trials: int = 0      # number of eval episodes (>0 implies headless)

  # Internal
  task_id: str = "Eval-Naive-Goalkeeper"


# ----- PPO config for checkpoint loading -----


def _make_agent_cfg():
  """Minimal PPO config sufficient for loading a policy checkpoint."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_soccer_eval",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def _load_policy(checkpoint_path: str, env, device: str):
  """Load a PPO checkpoint and return the inference policy."""
  print(f"[INFO] Loading policy from: {checkpoint_path}")
  agent_cfg = _make_agent_cfg()
  runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
  runner.load(checkpoint_path)
  policy = runner.get_inference_policy(device=env.unwrapped.device)
  print("[INFO] Policy loaded successfully.")
  return policy


def _make_zero_policy(env, device):
  """Return a zero-action policy for baseline evaluation."""
  act_dim = env.num_actions  # total_action_dim (29), not action_space.shape
  class ZeroPolicy:
    def __call__(self, obs):
      del obs
      return torch.zeros(1, act_dim, device=device)
    def reset(self):
      pass
  return ZeroPolicy()


# ----- Eval metrics (matching paper Section IV) -----


def _is_blocked(ball_vel_x_history: list, ball_pos_x_history: list) -> bool:
  """Check if ball was blocked: significant deceleration when near/behind robot.

  Paper criterion: x-velocity drops > 2 m/s when ball is behind robot (x > 0
  in local frame). We adapt this to world frame.
  """
  if len(ball_vel_x_history) < 2:
    return False
  for i in range(len(ball_vel_x_history)):
    x = ball_pos_x_history[i]
    if x > _BEHIND_ROBOT_X:
      # Ball is behind robot — check if velocity dropped significantly.
      max_vel = max(v for v in ball_vel_x_history[: i + 1] if v > 0)
      current_vel = ball_vel_x_history[i]
      if max_vel - current_vel > _BLOCK_VEL_DROP:
        return True
  return False


def _min_ball_robot_dist(ball_pos_w: torch.Tensor, robot_pos_w: torch.Tensor) -> float:
  """Minimum distance between ball and robot pelvis (xy-plane)."""
  delta = ball_pos_w[:2] - robot_pos_w[:2]
  return float(torch.norm(delta))


def run_trial(env, policy, max_steps: int = 150) -> dict:
  """Run one eval episode and return stats.

  Returns keys:
    blocked (bool), ball_past_robot (bool), ball_final_x (float),
    min_ball_robot_dist (float), max_ball_speed (float),
    ball_speed_at_robot (float | None), steps (int), terminated (bool)
  """
  obs = env.reset()
  if isinstance(obs, tuple):
    obs = obs[0]

  ball = env.unwrapped.scene["ball"]
  robot = env.unwrapped.scene["robot"]

  blocked = False
  ball_past_robot = False
  ball_final_x = 0.0
  min_dist = float("inf")
  max_ball_speed = 0.0
  ball_speed_at_robot = None
  ball_vel_x_history: list[float] = []
  ball_pos_x_history: list[float] = []
  steps = 0

  for _ in range(max_steps):
    with torch.inference_mode():
      action = policy(obs)
    result = env.step(action)
    obs = result[0]
    terminated = bool(result[2].item())
    steps += 1

    ball_pos = ball.data.root_link_pos_w[0].cpu()
    ball_vel = ball.data.root_link_lin_vel_w[0].cpu()
    robot_pos = robot.data.root_link_pos_w[0].cpu()
    speed = float(torch.norm(ball_vel))

    ball_vel_x_history.append(float(ball_vel[0]))
    ball_pos_x_history.append(float(ball_pos[0]))

    ball_final_x = float(ball_pos[0])
    max_ball_speed = max(max_ball_speed, speed)

    dist = _min_ball_robot_dist(ball_pos, robot_pos)
    if dist < min_dist:
      min_dist = dist

    # Detect ball crossing behind robot.
    if ball_pos[0] > _BEHIND_ROBOT_X and ball_speed_at_robot is None:
      ball_speed_at_robot = speed

    if ball_pos[0] > _BEHIND_ROBOT_X:
      ball_past_robot = True

    if not blocked:
      blocked = _is_blocked(ball_vel_x_history, ball_pos_x_history)

    if terminated:
      break

  return {
    "blocked": blocked,
    "ball_past_robot": ball_past_robot,
    "ball_final_x": ball_final_x,
    "min_ball_robot_dist": min_dist,
    "max_ball_speed": max_ball_speed,
    "ball_speed_at_robot": ball_speed_at_robot,
    "steps": steps,
    "terminated": terminated,
  }


# ----- Headless multi-trial eval -----


def run_headless_eval(cfg: EvalConfig, env, policy):
  """Run multiple trials headless and print summary statistics."""
  if cfg.num_trials <= 0:
    print("[WARN] --headless without --num-trials: nothing to evaluate.")
    return
  print(f"\n[INFO] Running {cfg.num_trials} headless eval trials...\n")

  results: list[dict] = []
  blocked_count = 0
  past_robot_count = 0
  min_dists: list[float] = []
  speeds_at_robot: list[float] = []

  for trial in range(cfg.num_trials):
    stats = run_trial(env, policy)
    results.append(stats)
    if stats["blocked"]:
      blocked_count += 1
    if stats["ball_past_robot"]:
      past_robot_count += 1
    if stats["min_ball_robot_dist"] < float("inf"):
      min_dists.append(stats["min_ball_robot_dist"])
    if stats["ball_speed_at_robot"] is not None:
      speeds_at_robot.append(stats["ball_speed_at_robot"])

    print_interval = 1 if cfg.num_trials <= 10 else (cfg.num_trials // 10)
    if (trial + 1) % print_interval == 0 or trial == 0:
      print(
        f"  Trial {trial + 1:3d}/{cfg.num_trials}: "
        f"blocked={stats['blocked']}, "
        f"past_robot={stats['ball_past_robot']}, "
        f"min_dist={stats['min_ball_robot_dist']:.3f}, "
        f"steps={stats['steps']}"
      )

  # Summary (matching paper Table II, Section IV).
  total = cfg.num_trials
  success_rate = blocked_count / total * 100 if total > 0 else 0
  mean_min_dist = float(np.mean(min_dists)) if min_dists else 0.0
  std_min_dist = float(np.std(min_dists)) if min_dists else 0.0
  mean_speed_at_robot = float(np.mean(speeds_at_robot)) if speeds_at_robot else 0.0
  pass_through_rate = past_robot_count / total * 100 if total > 0 else 0

  print(f"\n{'='*55}")
  print(f"  Eval Summary ({total} trials)")
  print(f"{'='*55}")
  print(f"  Blocked (Esucc):      {blocked_count}/{total} = {success_rate:.1f}%")
  print(f"  Ball behind robot:    {past_robot_count}/{total} = {pass_through_rate:.1f}%")
  print(f"  Min ball-robot dist:  {mean_min_dist:.3f} ± {std_min_dist:.3f} m")
  print(f"  Mean speed at robot:  {mean_speed_at_robot:.2f} m/s")
  print(f"{'='*55}\n")


# ----- Main -----


def run_eval(cfg: EvalConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.height = cfg.video_height
  env_cfg.viewer.width = cfg.video_width

  # Print env info.
  actor_terms = list(env_cfg.observations["actor"].terms.keys())
  critic_terms = list(env_cfg.observations["critic"].terms.keys())
  events = list(env_cfg.events.keys())
  term_names = list(env_cfg.terminations.keys())
  actor_hist = env_cfg.observations["actor"].history_length

  print(f"Task: {cfg.task_id}")
  print(f"Actor obs  ({len(actor_terms)} terms × {actor_hist} history): {actor_terms}")
  print(f"Critic obs ({len(critic_terms)} terms): {critic_terms}")
  print(f"Terminations ({len(term_names)}): {term_names}")
  print(f"Events     ({len(events)}): {events}")
  print(f"Episode length: {env_cfg.episode_length_s}s")
  print(f"Obs noise: {env_cfg.observations['actor'].enable_corruption}")

  # Print trajectory params.
  from src.tasks.soccer.config.soccer_settings import SETTINGS

  print(f"\nBall trajectory ({len(SETTINGS.goalkeeper_regions)} regions, parabolic model):")
  bt = SETTINGS.ball_trajectory
  print(f"  start: [{bt.ball_start_distance[0]}, {bt.ball_start_distance[1]}] m")
  print(f"  end:   [{bt.ball_end_distance[0]}, {bt.ball_end_distance[1]}] m")
  print(f"  t:     [{bt.t_flight[0]}, {bt.t_flight[1]}] s")

  region_names = [
    "Right-Mid", "Left-Mid", "Right-Up", "Left-Up", "Right-Low", "Left-Low",
  ]
  for i, r in enumerate(SETTINGS.goalkeeper_regions):
    print(f"  Region {i} ({region_names[i]}): h={r.height}, w={r.width}")

  render_mode = "rgb_array" if cfg.video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if cfg.video:
    video_folder = Path("videos") / "eval"
    video_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Recording video to: {video_folder}")
    env = VideoRecorder(
      env,
      video_folder=video_folder,
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  # Wrap with RSL-RL.
  env = RslRlVecEnvWrapper(env, clip_actions=100.0)

  # Policy.
  if cfg.checkpoint:
    policy = _load_policy(cfg.checkpoint, env, device)
  else:
    policy = _make_zero_policy(env, device)
    print("[INFO] Using zero-agent fallback (no checkpoint provided).")

  # Print runtime shapes.
  obs_space = env.unwrapped.single_observation_space
  actor_shape = obs_space.spaces["actor"].shape
  critic_shape = obs_space.spaces["critic"].shape
  action_dim = env.num_actions
  print(f"\nRuntime shapes:")
  print(f"  Actor obs dim:  {actor_shape}")
  print(f"  Critic obs dim: {critic_shape}")
  print(f"  Action dim:     {action_dim}")

  # Run headless or viewer.
  if cfg.headless:
    run_headless_eval(cfg, env, policy)
  else:
    if cfg.num_trials > 0:
      print("[INFO] --num-trials is set but --headless is not; "
            "running viewer (use --headless for batch eval stats).")

    # Select viewer.
    if cfg.viewer == "auto":
      has_display = bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
      )
      viewer_type = "native" if has_display else "viser"
    else:
      viewer_type = cfg.viewer

    if viewer_type == "native":
      NativeMujocoViewer(env, policy).run()
    elif viewer_type == "viser":
      ViserPlayViewer(env, policy).run()
    else:
      raise RuntimeError(f"Unsupported viewer: {viewer_type}")

  env.close()


def main():
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  eval_tasks = [t for t in all_tasks if "Eval" in t]
  if not eval_tasks:
    print("No eval tasks registered. Run: import src.tasks")
    return

  args = tyro.cli(EvalConfig, prog="eval_naive_goalkeeper")
  run_eval(args)


if __name__ == "__main__":
  main()
