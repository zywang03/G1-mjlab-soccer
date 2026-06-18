"""Evaluate shooter with fixed motion and target destination.

For each (motion, target_x) pair, runs ``--trials-per-pair`` trials and
records goal rate, kick speed, and target accuracy.

Usage:
  python scripts/eval_motion_analysis.py \\
      --checkpoint logs/rsl_rl/g1_soccer/<run>/model_100000.pt \\
      --motion-dir src/assets/soccer/motions/shooter \\
      --headless --trials-per-pair 10 --num-envs 32

Output:
  Table 1 — Per-motion aggregate (averaged across all targets).
  Table 2a-2d — Per-metric grid (rows=motion, columns=target_x).
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import src.tasks  # noqa: F401 — populate task registry
from src.tasks.soccer.config.g1.rl_cfg import (
    SoccerRecurrentRunner,
    unitree_g1_soccer_recurrent_runner_cfg,
)

_GOAL_HALF_WIDTH = 1.5
_GOAL_HEIGHT = 1.8
_GOAL_X = -0.5
_GOAL_CENTER = (-0.5, 0.0, 0.9)
_KICK_SPEED_THRESHOLD = 1.0
_MAX_STEPS = 500


def parse_args():
    parser = argparse.ArgumentParser(description="Motion analysis for shooter")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model .pt checkpoint")
    parser.add_argument("--motion-dir", type=str, required=True,
                        help="Path to motion .npz files")
    parser.add_argument("--num-targets", type=int, default=5,
                        help="Number of equally spaced target x positions on goal line")
    parser.add_argument("--trials-per-pair", type=int, default=5000,
                        help="Trials per (motion, target) pair")
    parser.add_argument("--num-envs", type=int, default=32,
                        help="Parallel environments")
    parser.add_argument("--headless", action="store_true",
                        help="Run headless (no viewer)")
    parser.add_argument("--seed", type=int, default=2810,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto-detect)")
    return parser.parse_args()


# -- Reusable helpers from eval_shooter_parallel --------------------------------


def _is_goal(ball_pos: torch.Tensor) -> torch.Tensor:
    return (
        (ball_pos[:, 0] <= _GOAL_X)
        & (torch.abs(ball_pos[:, 1]) <= _GOAL_HALF_WIDTH)
        & (ball_pos[:, 2] >= 0.0)
        & (ball_pos[:, 2] <= _GOAL_HEIGHT)
    )


def _kick_accuracy_vec(ball_vel: torch.Tensor, ball_pos: torch.Tensor) -> torch.Tensor:
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


def _load_policy(checkpoint_path: str, env, device: str):
    print(f"[INFO] Loading policy from: {checkpoint_path}")
    agent_cfg = unitree_g1_soccer_recurrent_runner_cfg()
    runner = SoccerRecurrentRunner(env, asdict(agent_cfg), log_dir=None, device=device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print("[INFO] Policy loaded successfully.")
    return policy


# -- Command state locking ------------------------------------------------------


def _init_command_for_motion(command, motion_idx, target_y, num_envs, device):
    """Lock command to a specific motion and destination (compete coords), then position robot."""
    env_ids = torch.arange(num_envs, device=device)

    command.motion_idx[:] = motion_idx
    command.motion_length[:] = command.motion.file_lengths[motion_idx]
    command.time_steps[:] = 0

    # Ball (fixed_ball_pos is set in env_cfg — uses that).
    command._compute_soccer_ball_positions(env_ids)
    command._update_soccer_ball(env_ids)
    command._update_target_points(env_ids)

    # Destination: fixed x=-0.5, sample y.
    dest = torch.tensor([[-0.5, target_y, 0.10]], device=device).expand(num_envs, -1)
    command.target_destination_pos[:] = dest

    # Position robot at motion frame 0 (already in compete coords via property overrides).
    jp = command.joint_pos.clone()
    jv = command.joint_vel.clone()

    root_pos = command.body_pos_w[:, 0].clone()
    root_ori = command.body_quat_w[:, 0].clone()
    root_lin_vel = command.body_lin_vel_w[:, 0].clone()
    root_ang_vel = command.body_ang_vel_w[:, 0].clone()

    # Stage5CompeteSoccerCommand handles the compete transform internally,
    # so we do NOT apply motion_origin_offset/motion_yaw_offset here.

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

    # Switch to "start" mode so future resamples only reset time_steps.
    command.cfg.sampling_mode = "start"


# -- Batch evaluation ----------------------------------------------------------


def _run_batch(env, policy, command, motion_idx, target_y,
               max_steps, num_envs, device) -> dict[str, torch.Tensor]:
    """Run one batch of trials for a fixed (motion, target) pair (compete coords)."""
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    if hasattr(policy, "reset"):
        policy.reset()

    base_env = env.unwrapped
    ball = base_env.scene["ball"]
    env_origins = base_env.scene.env_origins

    _init_command_for_motion(command, motion_idx, target_y, num_envs, device)

    kicked = torch.zeros(num_envs, dtype=torch.bool, device=device)
    kick_speed = torch.zeros(num_envs, dtype=torch.float32, device=device)
    kick_acc = torch.zeros(num_envs, dtype=torch.float32, device=device)
    abs_z_speed = torch.zeros(num_envs, dtype=torch.float32, device=device)
    goal_scored = torch.zeros(num_envs, dtype=torch.bool, device=device)
    ball_final_x = torch.zeros(num_envs, dtype=torch.float32, device=device)
    step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
    terminated_flag = torch.zeros(num_envs, dtype=torch.bool, device=device)
    motion_ended_flag = torch.zeros(num_envs, dtype=torch.bool, device=device)

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

        # Motion ended when cumulative steps reach motion_length.
        # In "start" mode, command.time_steps loops (T-1 → 0), so we
        # track total steps since trial start rather than relying on wrap.
        motion_ended = step_active & (step_count >= command.motion_length)
        if torch.any(motion_ended):
            motion_ended_flag[motion_ended] = True
            terminated = terminated | motion_ended

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

        ball_final_x[step_active] = ball_local[:, 0][step_active]

        can_cross = step_active & (~cross_recorded)
        if torch.any(can_cross):
            crossed = can_cross & (prev_ball_local[:, 0] > _GOAL_X) & (ball_local[:, 0] <= _GOAL_X)
            if torch.any(crossed):
                dx = ball_local[:, 0] - prev_ball_local[:, 0]
                safe = torch.where(dx >= 0, torch.tensor(1e-6, device=device),
                                   torch.tensor(-1e-6, device=device))
                alpha = ((_GOAL_X - prev_ball_local[:, 0]) / (dx + safe)).clamp(0.0, 1.0).unsqueeze(-1)
                cross_pos = prev_ball_local + alpha * (ball_local - prev_ball_local)
                ty = command.target_destination_pos[crossed, 1]
                target_error[crossed] = (cross_pos[crossed, 1] - ty).abs()
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
        "ball_final_x": ball_final_x.cpu(),
        "steps": step_count.cpu(),
        "terminated": terminated_flag.cpu(),
        "motion_ended": motion_ended_flag.cpu(),
        "cross_recorded": cross_recorded.cpu(),
        "target_error": target_error.cpu(),
        "cross_pos_z": cross_pos_z.cpu(),
    }


def _summarise_pair(metrics: dict[str, torch.Tensor]) -> dict:
    """Aggregate metrics for one (motion, target) cell."""
    total = int(metrics["goal"].numel())
    goals = int(metrics["goal"].sum().item())
    goal_rate = goals / total * 100.0 if total > 0 else 0.0

    kicked_mask = metrics["kick_speed"] > 0.0
    kicked_count = int(kicked_mask.sum().item())
    kick_rate = kicked_count / total * 100.0 if total > 0 else 0.0
    mean_kick_speed = float(metrics["kick_speed"][kicked_mask].float().mean().item()) if kicked_count > 0 else 0.0

    crossed = int(metrics["cross_recorded"].sum().item())
    cross_rate = crossed / total * 100.0 if total > 0 else 0.0

    finite = torch.isfinite(metrics["target_error"])
    if torch.any(finite):
        target_err_np = metrics["target_error"][finite].float().numpy()
        mean_target_error = float(np.mean(target_err_np))
        hit_rate_0_3m = float((target_err_np <= 0.3).mean()) * 100.0
    else:
        mean_target_error = 0.0
        hit_rate_0_3m = 0.0

    return {
        "total": total,
        "goal_rate": goal_rate,
        "kick_rate": kick_rate,
        "mean_kick_speed": mean_kick_speed,
        "cross_rate": cross_rate,
        "mean_target_error": mean_target_error,
        "hit_rate_0_3m": hit_rate_0_3m,
    }


def _summarise_per_motion(results: dict, motion_idx: int, num_targets: int) -> dict:
    """Average across all targets for one motion."""
    cell_keys = [(motion_idx, t) for t in range(num_targets)]
    cells = [results[k] for k in cell_keys]

    total = sum(c["total"] for c in cells)
    total_goals = sum(int(c["goal_rate"] / 100.0 * c["total"]) for c in cells)
    goal_rate = total_goals / total * 100.0 if total > 0 else 0.0

    kick_speeds = [c["mean_kick_speed"] for c in cells if c["kick_rate"] > 0]
    mean_kick_speed = float(np.mean(kick_speeds)) if kick_speeds else 0.0

    target_errors = [c["mean_target_error"] for c in cells if c["cross_rate"] > 0]
    mean_target_error = float(np.mean(target_errors)) if target_errors else 0.0

    hit_rates = [c["hit_rate_0_3m"] for c in cells if c["cross_rate"] > 0]
    mean_hit_rate = float(np.mean(hit_rates)) if hit_rates else 0.0

    return {
        "total": total,
        "goal_rate": goal_rate,
        "mean_kick_speed": mean_kick_speed,
        "mean_target_error": mean_target_error,
        "hit_rate_0_3m": mean_hit_rate,
        "kick_rate": float(np.mean([c["kick_rate"] for c in cells])),
        "cross_rate": float(np.mean([c["cross_rate"] for c in cells])),
    }


# -- Output formatting ---------------------------------------------------------


def _print_tables(results: dict, motion_info: list[dict], target_xs: np.ndarray,
                  num_targets: int):
    """Print Table 1 (per-motion aggregate) and Tables 2a-2d (per-metric grid)."""
    from prettytable import PrettyTable

    num_motions = len(motion_info)

    # -- Table 1: Per-motion aggregate -----------------------------------------
    print("\n" + "=" * 90)
    print("  Table 1: Per-Motion Aggregate (averaged across all targets)")
    print("=" * 90)

    t1 = PrettyTable()
    t1.field_names = ["Motion", "File", "Kick Leg", "Frames",
                      "Goal Rate", "Kick Rate", "Cross Rate",
                      "Kick Speed", "Target Error", "Hit@0.3m"]
    t1.float_format = ".2"

    for mi in range(num_motions):
        s = _summarise_per_motion(results, mi, num_targets)
        info = motion_info[mi]
        t1.add_row([
            mi, info["name"], info["kick_leg"], info["frames"],
            f"{s['goal_rate']:.1f}%",
            f"{s['kick_rate']:.1f}%",
            f"{s['cross_rate']:.1f}%",
            f"{s['mean_kick_speed']:.2f}",
            f"{s['mean_target_error']:.3f}",
            f"{s['hit_rate_0_3m']:.1f}%",
        ])

    print(t1)
    print()

    # -- Tables 2a-2d: Per-metric grids ----------------------------------------

    metrics_cfg = [
        ("Table 2a: Goal Rate (%)", "goal_rate", "{:.1f}%"),
        ("Table 2b: Mean Kick Speed (m/s)", "mean_kick_speed", "{:.2f}"),
        ("Table 2c: Mean Target Error (m)", "mean_target_error", "{:.3f}"),
        ("Table 2d: Hit Rate @0.3m (%)", "hit_rate_0_3m", "{:.1f}%"),
    ]

    for title, metric_key, fmt in metrics_cfg:
        t = PrettyTable()
        col_labels = [f"x={x:.2f}" for x in target_xs]
        t.field_names = ["Motion", "File"] + col_labels
        t.float_format = ".2"

        for mi in range(num_motions):
            row = [mi, motion_info[mi]["name"]]
            for ti in range(num_targets):
                c = results[(mi, ti)]
                row.append(fmt.format(c[metric_key]))
            t.add_row(row)

        print(f"\n{title}")
        print(t)

    print()


# -- Main ----------------------------------------------------------------------


def main():
    args = parse_args()
    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Motion dir: {args.motion_dir}")
    print(f"[INFO] Targets: {args.num_targets} in [-1.5, 1.5]")
    print(f"[INFO] Trials per pair: {args.trials_per_pair}")
    print(f"[INFO] Parallel envs: {args.num_envs}")

    # Build environment.
    num_envs = min(args.num_envs, args.trials_per_pair)
    if num_envs <= 0:
        num_envs = 1
    env_cfg = load_env_cfg("Eval-Shooter-Stage5", play=False)
    env_cfg.scene.num_envs = num_envs

    motion_cfg = env_cfg.commands.get("motion")
    if motion_cfg is not None:
        motion_cfg.sampling_mode = "start"
        motion_cfg.pose_range = {}
        motion_cfg.velocity_range = {}
        motion_cfg.joint_position_range = (0.0, 0.0)
        motion_cfg.motion_dir = args.motion_dir

    env_base = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env_base, clip_actions=100.0)

    policy = _load_policy(args.checkpoint, env, device)
    command = env_base.command_manager.get_term("motion")

    # Build motion metadata from the command's internal loader.
    motion_info = []
    for mi in range(command.motion.num_files):
        name = command.motion.motion_names[mi]
        kleg = command.motion.kick_leg_labels[mi]
        kleg = kleg if kleg in ("left", "right") else "?"
        nframes = int(command.motion.file_lengths[mi].item())
        motion_info.append({"name": name, "kick_leg": kleg, "frames": nframes})

    num_motions = len(motion_info)
    print(f"[INFO] Loaded {num_motions} motions:")

    target_ys = np.linspace(-1.5, 1.5, args.num_targets)
    total_pairs = num_motions * args.num_targets
    total_trials = total_pairs * args.trials_per_pair
    print(f"[INFO] Total pairs: {total_pairs}")
    print(f"[INFO] Total trials: {total_trials}")
    print()

    actor_terms = list(env_cfg.observations["actor"].terms.keys())
    print(f"[INFO] Actor obs ({len(actor_terms)} terms): {actor_terms}")
    print(f"[INFO] Terminations: {list(env_cfg.terminations.keys())}")
    print(f"[INFO] Episode length: {env_cfg.episode_length_s}s")

    results: dict[tuple[int, int], dict] = {}
    pair_idx = 0

    for mi in range(num_motions):
        info = motion_info[mi]
        print(f"\n  --- Motion {mi}/{num_motions - 1}: {info['name']} "
              f"(kick={info['kick_leg']}, frames={info['frames']}) ---")

        for ti, ty in enumerate(target_ys):
            pair_idx += 1

            all_batch_metrics: dict[str, list[torch.Tensor]] = {}
            remaining = args.trials_per_pair
            trial_offset = 0

            while remaining > 0:
                take = min(num_envs, remaining)
                if num_envs > take:
                    env_base.seed = args.seed + pair_idx * 1000 + trial_offset

                batch_metrics = _run_batch(
                    env, policy, command, mi, ty,
                    max_steps=_MAX_STEPS, num_envs=num_envs, device=device,
                )
                batch_metrics = {k: v[:take] for k, v in batch_metrics.items()}

                for k, v in batch_metrics.items():
                    all_batch_metrics.setdefault(k, []).append(v)

                trial_offset += take
                remaining -= take
                gc.collect()
                torch.cuda.empty_cache()

            merged = {k: torch.cat(v, dim=0) for k, v in all_batch_metrics.items()}
            summary = _summarise_pair(merged)
            results[(mi, ti)] = summary

            # Print progress line.
            print(f"  Target y={ty:+5.2f}: goal={summary['goal_rate']:5.1f}%  "
                  f"kick={summary['kick_rate']:5.1f}%  "
                  f"speed={summary['mean_kick_speed']:.2f}  "
                  f"err={summary['mean_target_error']:.3f}  "
                  f"hit@30cm={summary['hit_rate_0_3m']:.1f}%  "
                  f"(n={summary['total']})")

            gc.collect()
            torch.cuda.empty_cache()

    _print_tables(results, motion_info, target_ys, args.num_targets)
    env.close()


if __name__ == "__main__":
    main()
