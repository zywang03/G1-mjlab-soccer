"""Measure kick-trigger frame for each motion using trained policy (parallel).

For each motion, runs N trials with num_envs parallel envs and records the
simulation step at which ball speed first exceeds 1.0 m/s.

Output: KICK_FRAME_MAP dict usable by api_server.py for adaptive locking.

Usage:
    python scripts/analyze_kick_timing.py \
        --checkpoint checkpoints/stage6/model_20000.pt \
        --motion-dir src/assets/soccer/motions/shooter \
        --num-trials 50 --num-envs 32
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import src.tasks  # noqa: F401
from src.tasks.soccer.config.g1.rl_cfg import (
    SoccerRecurrentRunner,
    unitree_g1_soccer_recurrent_runner_cfg,
)

_KICK_SPEED_THRESHOLD = 1.0


@dataclass
class Config:
    checkpoint: str = "checkpoints/stage6/model_20000.pt"
    motion_dir: str = "src/assets/soccer/motions/shooter"
    num_trials: int = 50
    num_envs: int = 32
    device: str | None = None
    seed: int = 2810
    output: str | None = None
    """Optional JSON output path for KICK_FRAME_MAP."""


def _load_policy(checkpoint_path: str, env, device: str):
    print(f"[INFO] Loading policy from: {checkpoint_path}")
    agent_cfg = unitree_g1_soccer_recurrent_runner_cfg()
    runner = SoccerRecurrentRunner(env, asdict(agent_cfg), log_dir=None, device=device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print("[INFO] Policy loaded.")
    return policy


def _init_command_for_motion(command, motion_idx: int, num_envs: int, device: str):
    """Lock command to a specific motion, position robot at frame 0."""
    env_ids = torch.arange(num_envs, device=device)

    command.motion_idx[:] = motion_idx
    command.motion_length[:] = command.motion.file_lengths[motion_idx]
    command.time_steps[:] = 0

    command._compute_soccer_ball_positions(env_ids)
    command._update_soccer_ball(env_ids)
    command._update_target_points(env_ids)

    dest = torch.tensor([[-0.5, 0.0, 0.10]], device=device).expand(num_envs, -1)
    command.target_destination_pos[:] = dest

    jp = command.joint_pos.clone()
    jv = command.joint_vel.clone()

    root_pos = command.body_pos_w[:, 0].clone()
    root_ori = command.body_quat_w[:, 0].clone()
    root_lin_vel = command.body_lin_vel_w[:, 0].clone()
    root_ang_vel = command.body_ang_vel_w[:, 0].clone()

    command.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    command.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    command.robot.clear_state(env_ids=env_ids)

    action_term = command._env.action_manager._terms.get("joint_pos")
    if action_term is not None and hasattr(action_term, "_offset"):
        action_term._offset[env_ids] = jp

    command._compute_relative_transforms()

    if hasattr(command, "kick_contact_tracker"):
        flags = torch.zeros(num_envs, dtype=torch.bool, device=device)
        flags[env_ids] = True
        command.kick_contact_tracker._handle_resample(flags)

    command.cfg.sampling_mode = "start"


def _run_batch(env, policy, command, motion_idx, motion_length,
               num_envs: int, device: str) -> torch.Tensor:
    """Run one batch of trials, return kick_frame per env (-1 = no kick)."""
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    if hasattr(policy, "reset"):
        policy.reset()

    base_env = env.unwrapped
    ball = base_env.scene["ball"]

    _init_command_for_motion(command, motion_idx, num_envs, device)

    kicked = torch.zeros(num_envs, dtype=torch.bool, device=device)
    kick_frame = torch.full((num_envs,), -1, dtype=torch.int32, device=device)
    step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    max_steps = motion_length + 80

    for step in range(max_steps):
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

        motion_ended = step_active & (step_count >= motion_length)
        terminated = terminated | motion_ended

        ball_vel = ball.data.root_link_vel_w.detach().clone()[:, :3]
        speed = torch.linalg.vector_norm(ball_vel, dim=-1)

        new_kick = step_active & (~kicked) & (speed > _KICK_SPEED_THRESHOLD)
        if torch.any(new_kick):
            kicked[new_kick] = True
            kick_frame[new_kick] = step_count[new_kick]

        new_term = step_active & terminated
        if torch.any(new_term):
            if hasattr(policy, "reset"):
                with torch.inference_mode():
                    policy.reset(dones=new_term)

        active = step_active & (~terminated)
        if not torch.any(active):
            break

    return kick_frame


def main():
    cfg = tyro.cli(Config)
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Trials per motion: {cfg.num_trials}, Parallel envs: {cfg.num_envs}")

    num_envs = min(cfg.num_envs, cfg.num_trials)

    env_cfg = load_env_cfg("Eval-Shooter-Stage6", play=False)
    env_cfg.scene.num_envs = num_envs
    motion_cfg = env_cfg.commands.get("motion")
    if motion_cfg is not None:
        motion_cfg.sampling_mode = "start"
        motion_cfg.pose_range = {}
        motion_cfg.velocity_range = {}
        motion_cfg.joint_position_range = (0.0, 0.0)
        motion_cfg.motion_dir = cfg.motion_dir

    env_base = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env_base, clip_actions=100.0)
    command = env_base.command_manager.get_term("motion")
    policy = _load_policy(cfg.checkpoint, env, device)

    num_motions = command.motion.num_files
    print(f"[INFO] Loaded {num_motions} motions.  Each batch = {num_envs} parallel trials.\n")

    results: dict[str, Any] = {}
    print(f"{'Motion':<40s} {'Frames':>7} {'Kick±std':>14} {'Rate':>6} {'Deadline40%':>12} {'Deadline60%':>12}")
    print("-" * 93)

    for mi in range(num_motions):
        name = command.motion.motion_names[mi]
        nframes = int(command.motion.file_lengths[mi].item())

        all_kick_frames: list[int] = []
        remaining = cfg.num_trials
        trial_offset = 0

        while remaining > 0:
            take = min(num_envs, remaining)
            if num_envs > take:
                env_base.seed = cfg.seed + mi * 1000 + trial_offset

            batch_kf = _run_batch(env, policy, command, mi, nframes, num_envs, device)
            for i in range(take):
                kf = batch_kf[i].item()
                if kf >= 0:
                    all_kick_frames.append(int(kf))

            trial_offset += take
            remaining -= take
            gc.collect()
            torch.cuda.empty_cache()

        if all_kick_frames:
            mean_frame = np.mean(all_kick_frames)
            std_frame = np.std(all_kick_frames)
            kick_rate = len(all_kick_frames) / cfg.num_trials
        else:
            mean_frame = -1
            std_frame = -1
            kick_rate = 0

        deadline_40 = int(mean_frame * 0.4) if mean_frame > 0 else -1
        deadline_60 = int(mean_frame * 0.6) if mean_frame > 0 else -1

        results[name] = {
            "total_frames": nframes,
            "kick_rate": round(kick_rate, 4),
            "mean_kick_frame": round(mean_frame, 1),
            "std_kick_frame": round(std_frame, 1),
            "deadline_40pct": deadline_40,
            "deadline_60pct": deadline_60,
        }

        print(f"{name:<40s} {nframes:>7d} {mean_frame:>7.0f}±{std_frame:<6.0f} "
              f"{kick_rate:>5.0%} {deadline_40:>12d} {deadline_60:>12d}")

        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== KICK_FRAME_MAP (JSON) ===")
    print(json.dumps(results, indent=2))

    if cfg.output:
        output_path = Path(cfg.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\n[INFO] Saved to: {cfg.output}")

    env.close()


if __name__ == "__main__":
    main()
