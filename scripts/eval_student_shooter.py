"""Evaluate motion-free shooter student policies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.soccer.config.g1.rl_cfg import (
  SoccerRecurrentRunner,
  unitree_g1_student_shooter_ppo_runner_cfg,
)
from src.tasks.soccer.mdp.student_shooter_commands import StudentShooterCommand


@dataclass
class EvalStudentConfig:
  checkpoint: str | None = None
  task_id: str = "Unitree-G1-Shooter-Student"
  device: str | None = None
  seed: int = 2810
  num_trials: int = 256
  num_envs: int = 32
  max_steps: int = 250
  min_kick_speed: float = 1.0
  target_error_tolerance: float = 0.35
  horizontal_force_threshold: float = 0.0


class ZeroPolicy:
  def __init__(self, action_shape, device: str):
    self._action = torch.zeros(action_shape, device=device)

  def __call__(self, obs):
    del obs
    return self._action

  def reset(self, dones=None):
    pass


def _load_policy(checkpoint: str, env, device: str):
  agent_cfg = unitree_g1_student_shooter_ppo_runner_cfg()
  runner = SoccerRecurrentRunner(env, asdict(agent_cfg), log_dir=None, device=device)
  runner.load(
    checkpoint,
    load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
  )
  return runner.get_inference_policy(device=device)


def _cross_goal_plane(prev_pos: torch.Tensor, curr_pos: torch.Tensor, goal_y: float) -> tuple[torch.Tensor, torch.Tensor]:
  crossed = (prev_pos[:, 1] > goal_y) & (curr_pos[:, 1] <= goal_y)
  dy = curr_pos[:, 1] - prev_pos[:, 1]
  alpha = (goal_y - prev_pos[:, 1]) / (dy + torch.where(dy >= 0, 1e-6, -1e-6))
  alpha = torch.clamp(alpha, 0.0, 1.0).unsqueeze(-1)
  return crossed, prev_pos + alpha * (curr_pos - prev_pos)


def _inside_goal(cross_pos: torch.Tensor) -> torch.Tensor:
  return (torch.abs(cross_pos[:, 0]) <= 1.5) & (cross_pos[:, 2] >= 0.0) & (cross_pos[:, 2] <= 1.8)


def _run_batch(env, policy, cfg: EvalStudentConfig) -> dict[str, torch.Tensor]:
  obs = env.reset()
  if isinstance(obs, tuple):
    obs = obs[0]
  if hasattr(policy, "reset"):
    policy.reset()

  base_env = env.unwrapped
  command: StudentShooterCommand = base_env.command_manager.get_term("motion")
  ball = base_env.scene[command.cfg.ball_entity_name]
  tracker = command.kick_contact_tracker
  device = base_env.device
  num_envs = base_env.num_envs
  goal_y = float(command.cfg.destination_y)

  active = torch.ones(num_envs, dtype=torch.bool, device=device)
  early_terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
  valid_kick_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
  ball_cross_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
  actual_kick_side = torch.full((num_envs,), -1, dtype=torch.int8, device=device)
  nonfoot_contact_any = torch.zeros(num_envs, dtype=torch.bool, device=device)
  both_feet_contact_any = torch.zeros(num_envs, dtype=torch.bool, device=device)
  goal_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
  target_error = torch.full((num_envs,), float("inf"), dtype=torch.float32, device=device)
  kick_speed = torch.zeros(num_envs, dtype=torch.float32, device=device)
  kick_speed_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

  target_x = command.target_destination_pos[:, 0].detach().clone()
  prev_ball_pos = ball.data.root_link_pos_w.detach().clone() - base_env.scene.env_origins

  for step in range(cfg.max_steps):
    step_active = active.clone()
    with torch.inference_mode():
      action = policy(obs)
    result = env.step(action)
    obs = result[0]
    terminated = result[2]
    if not isinstance(terminated, torch.Tensor):
      terminated = torch.as_tensor(terminated, device=device)
    terminated = terminated.to(device=device, dtype=torch.bool).view(-1)

    newly_terminated = step_active & terminated
    if torch.any(newly_terminated):
      early_terminated[newly_terminated] = True
      if hasattr(policy, "reset"):
        policy.reset(dones=newly_terminated)

    event = tracker.detect_any_single_foot("ball_robot_contact", cfg.horizontal_force_threshold)
    new_valid = step_active & event.new_contact & (valid_kick_step < 0)
    if torch.any(new_valid):
      valid_kick_step[new_valid] = step
      side = torch.where(event.right_foot_contact, torch.ones_like(actual_kick_side), torch.zeros_like(actual_kick_side))
      actual_kick_side[new_valid] = side[new_valid]

    nonfoot_contact_any |= step_active & event.nonfoot_contact
    both_feet_contact_any |= step_active & event.left_foot_contact & event.right_foot_contact

    curr_ball_pos = ball.data.root_link_pos_w.detach().clone() - base_env.scene.env_origins
    speed = torch.linalg.vector_norm(ball.data.root_link_lin_vel_w[:, :2], dim=-1)
    new_speed = step_active & (kick_speed_step < 0) & (speed > cfg.min_kick_speed)
    if torch.any(new_speed):
      kick_speed_step[new_speed] = step
      kick_speed[new_speed] = speed[new_speed]

    crossed, cross_pos = _cross_goal_plane(prev_ball_pos, curr_ball_pos, goal_y)
    new_cross = step_active & crossed & (ball_cross_step < 0)
    if torch.any(new_cross):
      ball_cross_step[new_cross] = step
      target_error[new_cross] = torch.abs(cross_pos[new_cross, 0] - target_x[new_cross])
      goal_success[new_cross] = _inside_goal(cross_pos)[new_cross]
    prev_ball_pos = curr_ball_pos

    active = step_active & ~terminated
    if not torch.any(active):
      break

  success = (
    (valid_kick_step >= 0)
    & goal_success
    & (target_error < cfg.target_error_tolerance)
    & (kick_speed > cfg.min_kick_speed)
    & (~early_terminated)
    & (~nonfoot_contact_any)
    & (~both_feet_contact_any)
  )
  return {
    "success": success.cpu(),
    "valid_kick_step": valid_kick_step.cpu(),
    "ball_cross_step": ball_cross_step.cpu(),
    "goal_success": goal_success.cpu(),
    "target_error": target_error.cpu(),
    "kick_speed": kick_speed.cpu(),
    "early_terminated": early_terminated.cpu(),
    "nonfoot_contact_any": nonfoot_contact_any.cpu(),
    "both_feet_contact_any": both_feet_contact_any.cpu(),
    "actual_kick_side": actual_kick_side.cpu(),
  }


def _summarize(metadata: dict[str, torch.Tensor]) -> dict[str, Any]:
  total = int(metadata["success"].numel())
  success = metadata["success"]
  valid = metadata["valid_kick_step"] >= 0
  crossed = metadata["ball_cross_step"] >= 0
  finite_error = torch.isfinite(metadata["target_error"])
  summary = {
    "trials": total,
    "success": int(success.sum().item()),
    "success_rate": float(success.float().mean().item()) if total else 0.0,
    "valid_kick": int(valid.sum().item()),
    "ball_crossed_goal_plane": int(crossed.sum().item()),
    "goal_inside_frame": int(metadata["goal_success"].sum().item()),
    "early_terminated": int(metadata["early_terminated"].sum().item()),
    "nonfoot_contact_any": int(metadata["nonfoot_contact_any"].sum().item()),
    "both_feet_contact_any": int(metadata["both_feet_contact_any"].sum().item()),
    "left_kicks": int((metadata["actual_kick_side"] == 0).sum().item()),
    "right_kicks": int((metadata["actual_kick_side"] == 1).sum().item()),
  }
  summary["mean_target_error"] = float(metadata["target_error"][finite_error].mean().item()) if torch.any(finite_error) else 0.0
  summary["mean_success_target_error"] = float(metadata["target_error"][success].mean().item()) if torch.any(success) else 0.0
  summary["mean_kick_speed"] = float(metadata["kick_speed"][metadata["kick_speed"] > 0].mean().item()) if torch.any(metadata["kick_speed"] > 0) else 0.0
  summary["mean_success_kick_speed"] = float(metadata["kick_speed"][success].mean().item()) if torch.any(success) else 0.0
  return summary


def eval_student(cfg: EvalStudentConfig) -> dict[str, Any]:
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task_id)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=100.0)
  try:
    if cfg.checkpoint is None:
      policy = ZeroPolicy(env.unwrapped.action_space.shape, device)
      print("[INFO] Using zero policy.")
    else:
      policy = _load_policy(cfg.checkpoint, env, device)
      print(f"[INFO] Loaded policy: {cfg.checkpoint}")

    batches = []
    remaining = cfg.num_trials
    while remaining > 0:
      batch = _run_batch(env, policy, cfg)
      take = min(cfg.num_envs, remaining)
      batches.append({key: value[:take] for key, value in batch.items()})
      remaining -= take

    metadata = {key: torch.cat([batch[key] for batch in batches], dim=0) for key in batches[0]}
    summary = _summarize(metadata)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
  finally:
    env.close()


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  cfg = tyro.cli(EvalStudentConfig, prog="eval_student_shooter")
  eval_student(cfg)


if __name__ == "__main__":
  main()
