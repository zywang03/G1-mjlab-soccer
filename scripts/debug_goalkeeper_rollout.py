"""Run goalkeeper rollouts with per-trial failure diagnostics.

This script is intentionally evaluation-focused: it does not train, and it does
not change rewards. It exposes the state needed to decide whether failures come
from poor positioning, missed contact, falls, or suspicious reward conditions.

Usage:
  python scripts/debug_goalkeeper_rollout.py --checkpoint src/assets/soccer/weight/goalkeeper.pt --num-trials 20
  python scripts/debug_goalkeeper_rollout.py --num-trials 5 --csv goalkeeper_debug.csv
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from eval_naive_goalkeeper import (
  _GOAL_HALF_WIDTH,
  _GOAL_HEIGHT,
  _GOAL_X,
  _load_policy,
  _make_zero_policy,
)
from src.tasks.soccer.config.soccer_settings import SETTINGS


_HAND_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_FALL_LIMIT_GRAV_Z = math.cos(math.radians(70.0))


@dataclass
class DebugConfig:
  checkpoint: str | None = None
  device: str | None = None
  task_id: str = "Eval-Goalkeeper"
  num_trials: int = 10
  max_steps: int = 150
  seed: int = 2810
  csv: str | None = None
  contact_margin: float = 0.08
  velocity_drop_threshold: float = 2.0
  behind_robot_x_threshold: float = 0.0
  keep_fall_termination: bool = False


def _ball_entered_goal(ball_pos: torch.Tensor) -> bool:
  x, y, z = ball_pos.tolist()
  return x <= _GOAL_X and abs(y) <= _GOAL_HALF_WIDTH and z <= _GOAL_HEIGHT


def _find_body_indices(robot, names: tuple[str, ...], device: str) -> torch.Tensor:
  return torch.as_tensor(
    robot.find_bodies(names, preserve_order=True)[0],
    device=device,
    dtype=torch.long,
  )


def _fmt_pos(pos: tuple[float, float, float] | None) -> str:
  if pos is None:
    return "None"
  return f"({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})"


def _region_name(region: int) -> str:
  names = {
    0: "Right-Mid",
    1: "Left-Mid",
    2: "Right-Up",
    3: "Left-Up",
    4: "Right-Low",
    5: "Left-Low",
  }
  return names.get(region, f"Region-{region}")


def run_trial(env, policy, cfg: DebugConfig, trial_index: int) -> dict[str, object]:
  if hasattr(policy, "reset"):
    policy.reset()

  obs = env.reset()
  if isinstance(obs, tuple):
    obs = obs[0]

  raw_env = env.unwrapped
  ball = raw_env.scene["ball"]
  robot = raw_env.scene["robot"]
  hand_ids = _find_body_indices(robot, _HAND_BODY_NAMES, raw_env.device)
  foot_ids = _find_body_indices(robot, _FOOT_BODY_NAMES, raw_env.device)

  region_t = getattr(raw_env, "_gk_region", None)
  region = int(region_t[0].item()) if region_t is not None else -1

  initial_ball_pos = tuple(float(x) for x in ball.data.root_link_pos_w[0].cpu())
  initial_ball_vel = tuple(float(x) for x in ball.data.root_link_lin_vel_w[0].cpu())

  min_hand_dist = float("inf")
  min_foot_dist = float("inf")
  min_ee_dist = float("inf")
  min_target_hand_dist = float("inf")
  min_ball_robot_x = float("inf")
  min_projected_gravity_z = float("inf")
  max_speed = 0.0
  max_speed_drop = 0.0
  goal_cross_pos: tuple[float, float, float] | None = None
  robot_pass_pos: tuple[float, float, float] | None = None
  old_front_stop_step: int | None = None
  reward_stop_step: int | None = None
  terminated = False
  steps = 0

  for step in range(cfg.max_steps):
    with torch.inference_mode():
      action = policy(obs)
    result = env.step(action)
    obs = result[0]
    done = result[2]
    terminated = bool(done.item()) if hasattr(done, "item") else bool(done)
    steps = step + 1

    ball_pos_t = ball.data.root_link_pos_w[0]
    ball_vel_t = ball.data.root_link_lin_vel_w[0]
    robot_pos_t = robot.data.root_link_pos_w[0]
    ball_pos = tuple(float(x) for x in ball_pos_t.cpu())

    speed = float(torch.norm(ball_vel_t).item())
    max_speed = max(max_speed, speed)
    speed_drop = max_speed - speed
    max_speed_drop = max(max_speed_drop, speed_drop)

    hand_pos = robot.data.body_link_pos_w[0, hand_ids]
    foot_pos = robot.data.body_link_pos_w[0, foot_ids]
    hand_dist = torch.norm(hand_pos - ball_pos_t.unsqueeze(0), dim=-1)
    foot_dist = torch.norm(foot_pos - ball_pos_t.unsqueeze(0), dim=-1)
    min_hand_dist = min(min_hand_dist, float(torch.min(hand_dist).item()))
    min_foot_dist = min(min_foot_dist, float(torch.min(foot_dist).item()))
    min_ee_dist = min(min_hand_dist, min_foot_dist)

    target_w = getattr(raw_env, "_gk_end_target_w", None)
    if target_w is not None and target_w.shape[0] == raw_env.num_envs:
      target = target_w[0]
      target_hand_idx = 0 if region % 2 == 0 else 1
      target_hand_dist = torch.norm(hand_pos[target_hand_idx] - target)
      min_target_hand_dist = min(min_target_hand_dist, float(target_hand_dist.item()))

    ball_robot_x = float((ball_pos_t[0] - robot_pos_t[0]).item())
    min_ball_robot_x = min(min_ball_robot_x, ball_robot_x)

    grav_z = float(robot.data.projected_gravity_b[0, 2].item())
    min_projected_gravity_z = min(min_projected_gravity_z, grav_z)

    if goal_cross_pos is None and _ball_entered_goal(ball_pos_t.cpu()):
      goal_cross_pos = ball_pos

    if robot_pass_pos is None and ball_pos_t[0] <= robot_pos_t[0]:
      robot_pass_pos = ball_pos

    dropped = speed_drop > cfg.velocity_drop_threshold
    old_front_side = ball_pos_t[0] > robot_pos_t[0] + cfg.behind_robot_x_threshold
    reward_behind_side = ball_pos_t[0] < robot_pos_t[0] - cfg.behind_robot_x_threshold
    if old_front_stop_step is None and dropped and bool(old_front_side):
      old_front_stop_step = steps
    if reward_stop_step is None and dropped and bool(reward_behind_side):
      reward_stop_step = steps

    if terminated:
      break

  final_ball_pos = tuple(float(x) for x in ball.data.root_link_pos_w[0].cpu())
  final_ball_vel = tuple(float(x) for x in ball.data.root_link_lin_vel_w[0].cpu())
  final_speed = float(torch.norm(ball.data.root_link_lin_vel_w[0]).item())
  entered_goal = goal_cross_pos is not None
  contact_like = min_ee_dist <= SETTINGS.ball.radius + cfg.contact_margin
  fell_like = min_projected_gravity_z < _FALL_LIMIT_GRAV_Z

  return {
    "trial": trial_index,
    "region": region,
    "region_name": _region_name(region),
    "blocked": not entered_goal,
    "ball_entered_goal": entered_goal,
    "steps": steps,
    "terminated": terminated,
    "contact_like": contact_like,
    "fell_like": fell_like,
    "min_ee_dist": min_ee_dist,
    "min_hand_dist": min_hand_dist,
    "min_foot_dist": min_foot_dist,
    "min_target_hand_dist": min_target_hand_dist,
    "max_speed": max_speed,
    "final_speed": final_speed,
    "max_speed_drop": max_speed_drop,
    "min_ball_robot_x": min_ball_robot_x,
    "min_projected_gravity_z": min_projected_gravity_z,
    "old_front_stop_step": old_front_stop_step,
    "reward_stop_step": reward_stop_step,
    "initial_ball_pos": initial_ball_pos,
    "initial_ball_vel": initial_ball_vel,
    "robot_pass_pos": robot_pass_pos,
    "goal_cross_pos": goal_cross_pos,
    "final_ball_pos": final_ball_pos,
    "final_ball_vel": final_ball_vel,
  }


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
  if not rows:
    return
  csv_path = Path(path)
  csv_path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = list(rows[0].keys())
  with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _print_summary(rows: list[dict[str, object]]) -> None:
  total = len(rows)
  blocked = sum(1 for row in rows if row["blocked"])
  contact_like = sum(1 for row in rows if row["contact_like"])
  fell_like = sum(1 for row in rows if row["fell_like"])
  old_front_stop = sum(1 for row in rows if row["old_front_stop_step"] is not None)
  reward_stop = sum(1 for row in rows if row["reward_stop_step"] is not None)

  by_region: dict[int, list[dict[str, object]]] = {}
  for row in rows:
    by_region.setdefault(int(row["region"]), []).append(row)

  mean_min_ee = sum(float(row["min_ee_dist"]) for row in rows) / total
  finite_target_dists = [
    float(row["min_target_hand_dist"])
    for row in rows
    if float(row["min_target_hand_dist"]) < float("inf")
  ]
  mean_target_hand = (
    sum(finite_target_dists) / len(finite_target_dists)
    if finite_target_dists else float("inf")
  )
  mean_speed_drop = sum(float(row["max_speed_drop"]) for row in rows) / total

  print("\n" + "=" * 72)
  print(f"Goalkeeper debug summary ({total} trials)")
  print("=" * 72)
  print(f"Block rate:                 {blocked}/{total} = {blocked / total * 100:.1f}%")
  print(f"Contact-like close calls:   {contact_like}/{total} = {contact_like / total * 100:.1f}%")
  print(f"Fall-like posture events:   {fell_like}/{total} = {fell_like / total * 100:.1f}%")
  print(f"Mean closest EE distance:   {mean_min_ee:.3f} m")
  print(f"Mean target-hand distance:  {mean_target_hand:.3f} m")
  print(f"Mean max ball speed drop:   {mean_speed_drop:.3f} m/s")
  print(f"Old front-side stop triggers: {old_front_stop}/{total}")
  print(f"Reward stop_ball triggers:   {reward_stop}/{total}")
  print("\nBy region:")
  for region in sorted(by_region):
    items = by_region[region]
    n = len(items)
    b = sum(1 for row in items if row["blocked"])
    c = sum(1 for row in items if row["contact_like"])
    print(
      f"  {region:>2d} {_region_name(region):<10} "
      f"block={b}/{n} ({b / n * 100:5.1f}%), "
      f"contact_like={c}/{n}"
    )
  print("=" * 72 + "\n")


def run_debug(cfg: DebugConfig) -> None:
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.scene.num_envs = 1
  if not cfg.keep_fall_termination and "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=100.0)

  if cfg.checkpoint:
    policy = _load_policy(cfg.checkpoint, env, device)
  else:
    policy = _make_zero_policy(env, device)
    print("[INFO] Using zero-agent fallback (no checkpoint provided).")

  rows = []
  for idx in range(cfg.num_trials):
    row = run_trial(env, policy, cfg, idx + 1)
    rows.append(row)
    print(
      f"Trial {idx + 1:3d}/{cfg.num_trials}: "
      f"blocked={row['blocked']}, region={row['region_name']}, "
      f"contact_like={row['contact_like']}, fell_like={row['fell_like']}, "
      f"min_ee={float(row['min_ee_dist']):.3f}m, "
      f"target_hand={float(row['min_target_hand_dist']):.3f}m, "
      f"speed_drop={float(row['max_speed_drop']):.2f}m/s, "
      f"old_front_stop={row['old_front_stop_step']}, reward_stop={row['reward_stop_step']}, "
      f"goal_cross={_fmt_pos(row['goal_cross_pos'])}"
    )

  _print_summary(rows)
  if cfg.csv:
    _write_csv(cfg.csv, rows)
    print(f"[INFO] Wrote CSV diagnostics to: {cfg.csv}")

  env.close()


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  args = tyro.cli(DebugConfig, prog="debug_goalkeeper_rollout")
  run_debug(args)


if __name__ == "__main__":
  main()
