"""Collect Stage IV teacher rollouts for shooter student BC.

Rewritten for Stage IV: high-speed filtering (>=10 m/s), large-scale parallel
collection (256-2048 envs), RSL-RL tuple compat, OOM protection.

Example:
  python scripts/collect_shooter_teacher_dataset.py \
      --checkpoint checkpoints/stage4/model_138985.pt \
      --num-episodes 102400 --num-envs 256
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
from src.tasks.soccer.mdp.shooter_commands import MultiMotionSoccerCommand, MultiMotionSoccerCommandCfg
from src.tasks.soccer.mdp.student_shooter_obs import student_shooter_obs

GOAL_Y = -5.0
GOAL_HALF_WIDTH = 1.5
GOAL_HEIGHT = 1.8


@dataclass
class CollectConfig:
    checkpoint: str
    """Stage IV teacher checkpoint path."""

    output_dir: str = "data/shooter_student/teacher_rollouts"
    """Root directory for dataset shards."""

    task_id: str = "Eval-Shooter"
    motion_dir: str | None = None
    device: str | None = None
    seed: int = 2810
    num_episodes: int = 102400
    num_envs: int = 256
    max_steps: int = 500
    post_kick_steps: int = 40
    min_kick_speed: float = 10.0
    target_error_tolerance: float = 0.3
    horizontal_force_threshold: float = 0.0
    run_name: str | None = None
    overwrite: bool = False


def _to_item_list(values: torch.Tensor) -> list[Any]:
    return values.detach().cpu().tolist()


def _load_teacher_policy(checkpoint_path: str, env, device: str):
    agent_cfg = unitree_g1_soccer_recurrent_runner_cfg()
    runner = SoccerRecurrentRunner(env, asdict(agent_cfg), log_dir=None, device=device)
    runner.load(checkpoint_path)
    return runner.get_inference_policy(device=device)


def _set_motion_dir(env_cfg, motion_dir: str | None) -> None:
    if motion_dir is None:
        return
    motion_cmd = env_cfg.commands.get("motion")
    if isinstance(motion_cmd, MultiMotionSoccerCommandCfg):
        motion_cmd.motion_dir = str(Path(motion_dir).expanduser().resolve())


def _expected_cross_x(initial_ball: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    dy = destination[:, 1] - initial_ball[:, 1]
    safe = torch.where(dy >= 0, 1e-6, -1e-6)
    alpha = (GOAL_Y - initial_ball[:, 1]) / (dy + safe)
    return initial_ball[:, 0] + alpha * (destination[:, 0] - initial_ball[:, 0])


def _cross_goal_plane(prev_pos: torch.Tensor, curr_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    crossed = (prev_pos[:, 1] > GOAL_Y) & (curr_pos[:, 1] <= GOAL_Y)
    dy = curr_pos[:, 1] - prev_pos[:, 1]
    safe = torch.where(dy >= 0, 1e-6, -1e-6)
    alpha = ((GOAL_Y - prev_pos[:, 1]) / (dy + safe)).clamp(0.0, 1.0).unsqueeze(-1)
    cross_pos = prev_pos + alpha * (curr_pos - prev_pos)
    return crossed, cross_pos


def _inside_goal(cross_pos: torch.Tensor) -> torch.Tensor:
    return (
        (torch.abs(cross_pos[:, 0]) <= GOAL_HALF_WIDTH)
        & (cross_pos[:, 2] >= 0.0)
        & (cross_pos[:, 2] <= GOAL_HEIGHT)
    )


def _summarize_metadata(metadata: dict[str, torch.Tensor]) -> dict[str, Any]:
    total = int(metadata["success"].numel())
    success = int(metadata["success"].sum().item())
    valid_kick = int((metadata["valid_kick_step"] >= 0).sum().item())
    crossed = int((metadata["ball_cross_step"] >= 0).sum().item())
    goal = int(metadata["goal_success"].sum().item())
    left = int((metadata["actual_kick_side"] == 0).sum().item())
    right = int((metadata["actual_kick_side"] == 1).sum().item())
    success_mask = metadata["success"]
    if torch.any(success_mask):
        mean_error = float(metadata["target_error"][success_mask].mean().item())
        mean_speed = float(metadata["kick_speed"][success_mask].mean().item())
    else:
        mean_error = 0.0
        mean_speed = 0.0
    return {
        "episodes": total,
        "success": success,
        "success_rate": success / max(total, 1),
        "valid_kick": valid_kick,
        "ball_crossed_goal_plane": crossed,
        "goal_inside_frame": goal,
        "actual_left_kicks": left,
        "actual_right_kicks": right,
        "mean_success_target_error": mean_error,
        "mean_success_kick_speed": mean_speed,
    }


def _slice_batch(batch: dict[str, Any], count: int) -> dict[str, Any]:
    metadata = {
        key: value[:count] if isinstance(value, torch.Tensor) else value
        for key, value in batch["metadata"].items()
    }
    return {
        "student_obs": batch["student_obs"][:count],
        "teacher_action": batch["teacher_action"][:count],
        "valid_mask": batch["valid_mask"][:count],
        "metadata": metadata,
        "motion_names": batch["motion_names"],
        "summary": _summarize_metadata(metadata),
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _collect_batch(env, policy, cfg: CollectConfig) -> dict[str, Any]:
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    if hasattr(policy, "reset"):
        policy.reset()

    base_env = env.unwrapped
    command: MultiMotionSoccerCommand = base_env.command_manager.get_term("motion")
    ball = base_env.scene[command.cfg.ball_entity_name]
    tracker = command.kick_contact_tracker
    device = base_env.device
    num_envs = base_env.num_envs

    motion_id = command.motion_idx.detach().clone()
    expected_kick_leg = command.kick_leg.detach().clone()
    destination = command.target_destination_pos.detach().clone()
    initial_ball_pos = (ball.data.root_link_pos_w - base_env.scene.env_origins).detach().clone()
    expected_cross_x = _expected_cross_x(initial_ball_pos, destination)

    actual_kick_side = torch.full((num_envs,), -1, dtype=torch.int8, device=device)
    valid_kick_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    ball_cross_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    cross_pos = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    target_error = torch.full((num_envs,), float("inf"), dtype=torch.float32, device=device)
    kick_speed = torch.zeros(num_envs, dtype=torch.float32, device=device)
    max_ball_speed = torch.zeros(num_envs, dtype=torch.float32, device=device)
    early_terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
    nonfoot_contact_any = torch.zeros(num_envs, dtype=torch.bool, device=device)
    both_feet_contact_any = torch.zeros(num_envs, dtype=torch.bool, device=device)
    left_foot_contact_any = torch.zeros(num_envs, dtype=torch.bool, device=device)
    right_foot_contact_any = torch.zeros(num_envs, dtype=torch.bool, device=device)
    goal_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    active_steps = torch.full((num_envs,), cfg.max_steps, dtype=torch.long, device=device)

    student_obs_steps: list[torch.Tensor] = []
    teacher_action_steps: list[torch.Tensor] = []
    valid_mask_steps: list[torch.Tensor] = []

    prev_ball_pos = ball.data.root_link_pos_w.detach().clone() - base_env.scene.env_origins

    for step in range(cfg.max_steps):
        step_active = active.clone()

        with torch.inference_mode():
            student_frame = student_shooter_obs(base_env)
            action = policy(obs)

        student_obs_steps.append(student_frame.detach().cpu())
        teacher_action_steps.append(action.detach().cpu())
        valid_mask_steps.append(active.detach().cpu())

        result = env.step(action)
        obs = result[0]
        raw_term = result[2]
        if isinstance(raw_term, torch.Tensor):
            terminated = raw_term.to(device=device, dtype=torch.bool).view(-1)
        else:
            terminated = torch.as_tensor(raw_term, device=device, dtype=torch.bool).view(-1)

        event = tracker.detect(command, "ball_robot_contact", cfg.horizontal_force_threshold)
        new_valid = step_active & event.new_contact & (valid_kick_step < 0)
        if torch.any(new_valid):
            valid_kick_step[new_valid] = step
            side = torch.where(
                event.right_foot_contact,
                torch.ones_like(actual_kick_side),
                torch.zeros_like(actual_kick_side),
            )
            actual_kick_side[new_valid] = side[new_valid]

        nonfoot_contact_any |= step_active & event.nonfoot_contact
        both_feet_contact_any |= step_active & event.left_foot_contact & event.right_foot_contact
        left_foot_contact_any |= step_active & event.left_foot_contact
        right_foot_contact_any |= step_active & event.right_foot_contact

        curr_ball_pos = ball.data.root_link_pos_w.detach().clone() - base_env.scene.env_origins
        curr_ball_vel = ball.data.root_link_lin_vel_w.detach().clone()
        speed = torch.linalg.vector_norm(curr_ball_vel, dim=-1)
        max_ball_speed = torch.where(step_active, torch.maximum(max_ball_speed, speed), max_ball_speed)

        new_speed = step_active & (kick_speed == 0.0) & (speed > cfg.min_kick_speed)
        if torch.any(new_speed):
            kick_speed[new_speed] = speed[new_speed]

        crossed, interpolated_cross = _cross_goal_plane(prev_ball_pos, curr_ball_pos)
        new_cross = step_active & crossed & (ball_cross_step < 0)
        if torch.any(new_cross):
            ball_cross_step[new_cross] = step
            cross_pos[new_cross] = interpolated_cross[new_cross]
            target_error[new_cross] = torch.abs(interpolated_cross[new_cross, 0] - expected_cross_x[new_cross])
            goal_success[new_cross] = _inside_goal(interpolated_cross)[new_cross]

        prev_ball_pos = curr_ball_pos

        newly_terminated = step_active & terminated
        if torch.any(newly_terminated):
            early_terminated[newly_terminated] = True
            active_steps[newly_terminated] = step + 1
            if hasattr(policy, "reset"):
                with torch.inference_mode():
                    policy.reset(dones=newly_terminated)

        active = step_active & ~terminated
        if not torch.any(active):
            break

    student_obs_tensor = torch.stack(student_obs_steps, dim=1)
    teacher_action_tensor = torch.stack(teacher_action_steps, dim=1)
    valid_mask = torch.stack(valid_mask_steps, dim=1)
    seq_len = student_obs_tensor.shape[1]

    time_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    has_kick = valid_kick_step >= 0
    kick_end = (valid_kick_step + cfg.post_kick_steps + 1).clamp(max=seq_len)
    fallback_end = active_steps
    keep_end = torch.where(has_kick, torch.minimum(kick_end, fallback_end), fallback_end)
    valid_mask = valid_mask & (time_ids < keep_end.unsqueeze(-1))

    success = (
        has_kick
        & goal_success
        & (target_error < cfg.target_error_tolerance)
        & (kick_speed >= cfg.min_kick_speed)
        & (~early_terminated)
        & (~nonfoot_contact_any)
        & (~both_feet_contact_any)
    )

    metadata = {
        "motion_id": motion_id.cpu(),
        "expected_kick_leg": expected_kick_leg.cpu(),
        "actual_kick_side": actual_kick_side.cpu(),
        "valid_kick_step": valid_kick_step.cpu(),
        "ball_cross_step": ball_cross_step.cpu(),
        "goal_success": goal_success.cpu(),
        "success": success.cpu(),
        "target_error": target_error.cpu(),
        "kick_speed": kick_speed.cpu(),
        "max_ball_speed": max_ball_speed.cpu(),
        "early_terminated": early_terminated.cpu(),
        "nonfoot_contact_any": nonfoot_contact_any.cpu(),
        "both_feet_contact_any": both_feet_contact_any.cpu(),
        "left_foot_contact_any": left_foot_contact_any.cpu(),
        "right_foot_contact_any": right_foot_contact_any.cpu(),
        "keep_length": keep_end.cpu(),
        "destination": destination.cpu(),
        "initial_ball_pos": initial_ball_pos.cpu(),
        "cross_pos": cross_pos.cpu(),
        "expected_cross_x": expected_cross_x.cpu(),
    }

    return {
        "student_obs": student_obs_tensor,
        "teacher_action": teacher_action_tensor,
        "valid_mask": valid_mask,
        "metadata": metadata,
        "motion_names": tuple(command.motion.motion_names),
        "summary": _summarize_metadata(metadata),
    }


def run_collection(cfg: CollectConfig) -> None:
    configure_torch_backends()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(cfg.task_id, play=False)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    _set_motion_dir(env_cfg, cfg.motion_dir)

    run_tag = cfg.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(cfg.output_dir).expanduser().resolve() / run_tag
    shard_dir = out_dir / "shards"
    if out_dir.exists() and any(out_dir.iterdir()) and not cfg.overwrite:
        raise FileExistsError(f"Output directory already exists and is not empty: {out_dir}")
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Collecting shooter teacher dataset on {device}")
    print(f"[INFO] Output: {out_dir}")
    print(f"[INFO] Episodes: {cfg.num_episodes}, envs/batch: {cfg.num_envs}")
    print(f"[INFO] Filter: min_kick_speed={cfg.min_kick_speed}, target_error<={cfg.target_error_tolerance}m")

    env_base = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env_base, clip_actions=100.0)
    policy = _load_teacher_policy(cfg.checkpoint, env, device)

    total_summary: dict[str, float] = {
        "episodes": 0,
        "success": 0,
        "valid_kick": 0,
        "ball_crossed_goal_plane": 0,
        "goal_inside_frame": 0,
        "actual_left_kicks": 0,
        "actual_right_kicks": 0,
    }
    success_errors: list[float] = []
    success_speeds: list[float] = []
    total_episodes_collected = 0

    try:
        remaining = cfg.num_episodes
        shard_idx = 0
        last_progress_pct = -1
        while remaining > 0:
            batch = _collect_batch(env, policy, cfg)
            take = min(cfg.num_envs, remaining)
            if take < cfg.num_envs:
                batch = _slice_batch(batch, take)

            shard_path = shard_dir / f"shard_{shard_idx:06d}.pt"
            torch.save({
                "student_obs": batch["student_obs"],
                "teacher_action": batch["teacher_action"],
                "valid_mask": batch["valid_mask"],
                "metadata": batch["metadata"],
                "motion_names": batch["motion_names"],
                "config": asdict(cfg),
            }, shard_path)

            summary = batch["summary"]
            for key in total_summary:
                total_summary[key] += float(summary.get(key, 0.0))
            success_mask = batch["metadata"]["success"]
            if torch.any(success_mask):
                success_errors.extend(_to_item_list(batch["metadata"]["target_error"][success_mask]))
                success_speeds.extend(_to_item_list(batch["metadata"]["kick_speed"][success_mask]))

            shard_idx += 1
            remaining -= take
            total_episodes_collected += take
            pct = int(100 * (cfg.num_episodes - remaining) / cfg.num_episodes)
            if pct > last_progress_pct:
                last_progress_pct = pct

            yield_rate = (total_summary["success"] / max(total_episodes_collected, 1)) * 100.0
            print(
                f"[INFO] shard={shard_idx:04d} progress={pct:3d}% "
                f"success={int(total_summary['success'])}/{int(total_summary['episodes'])} "
                f"yield={yield_rate:.1f}% "
                f"speed={summary.get('mean_success_kick_speed', 0.0):.2f}m/s"
            )

            gc.collect()
            torch.cuda.empty_cache()
    finally:
        env.close()

    total_summary["success_rate"] = total_summary["success"] / max(total_summary["episodes"], 1)
    total_summary["yield_rate"] = total_summary["success"] / max(total_episodes_collected, 1)
    total_summary["mean_success_target_error"] = float(np.mean(success_errors)) if success_errors else 0.0
    total_summary["mean_success_kick_speed"] = float(np.mean(success_speeds)) if success_speeds else 0.0

    _save_json(out_dir / "summary.json", total_summary)
    _save_json(out_dir / "config.json", asdict(cfg))
    print(f"[INFO] Done. Summary: {json.dumps(total_summary, indent=2)}")


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    cfg = tyro.cli(CollectConfig, prog="collect_shooter_teacher_dataset")
    run_collection(cfg)


if __name__ == "__main__":
    main()
