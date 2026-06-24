"""Collect MoE goalkeeper teacher rollouts for LSTM student BC.

The teacher rollout path intentionally stays unchanged: checkpoints that are
MoE bundles are loaded through the existing gate + six region experts +
optional prepare/idle expert adapter.  The student frame appends the normalized
predicted landing condition to the normal goalkeeper actor observation.
"""

from __future__ import annotations

import gc
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.soccer.mdp.goalkeeper_ball_reset import (
  launch_staged_ball_after_delay,
  reset_ball_staged_delayed_launch,
)
from src.tasks.soccer.mdp.goalkeeper_student_obs import build_goalkeeper_student_obs, goalkeeper_prediction_condition
from src.tasks.soccer.modules.moe6_goalkeeper_policy import MoE6GoalkeeperPolicy


@dataclass
class CollectConfig:
  checkpoint: str
  """Goalkeeper teacher checkpoint. MoE7 bundles keep their original routing."""

  output_dir: str = "data/goalkeeper_student/teacher_rollouts"
  """Root directory for dataset shards."""

  task_id: str = "Unitree-G1-Goalkeeper-Train"
  device: str | None = None
  seed: int = 2810
  num_episodes: int = 32768
  num_envs: int = 256
  max_steps: int = 300
  run_name: str | None = None
  overwrite: bool = False
  delayed_launch: bool = False
  """Hold the sampled ball stationary first so prepare frames enter the rollout."""

  launch_delay_s: float = 3.0
  """Seconds to wait before launching the staged ball when delayed_launch is enabled."""

  success_only: bool = False
  """Keep only episodes where the ball never enters the goal frame."""

  require_prepare_stable: bool = False
  """Keep only episodes that stay upright during the delayed-launch prepare window."""

  max_attempt_episodes: int = 0
  """Optional cap on attempted episodes when filtering; 0 means no cap."""

  goal_x: float = -0.5
  goal_half_width: float = 1.5
  goal_height: float = 1.8
  fall_limit_angle: float = math.radians(70.0)


def _apply_delayed_launch_sampling(env_cfg, wait_s: float) -> None:
  reset_event = env_cfg.events["reset_ball"]
  ball_pos = (3.0, 0.0, 0.1)
  sampler_params = dict(reset_event.params)
  sampler_params["fixed_start_local"] = ball_pos
  env_cfg.episode_length_s += wait_s
  env_cfg.events.pop("push_robot", None)
  env_cfg.events.pop("perturb_ball_vel", None)
  env_cfg.events["reset_ball"] = EventTermCfg(
    func=reset_ball_staged_delayed_launch,
    mode="reset",
    params={
      "sampler_func": reset_event.func,
      "sampler_params": sampler_params,
      "ball_pos": ball_pos,
      "ball_cfg": SceneEntityCfg("ball"),
    },
  )
  env_cfg.events["launch_delayed_ball"] = EventTermCfg(
    func=launch_staged_ball_after_delay,
    mode="step",
    params={"wait_s": wait_s, "ball_cfg": SceneEntityCfg("ball")},
  )


def _load_teacher_policy(checkpoint_path: str, env, device: str):
  loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
  if not isinstance(loaded, dict):
    raise ValueError(f"{checkpoint_path} is not a supported goalkeeper teacher checkpoint")

  bundle = None
  if "sr" in loaded:
    bundle = loaded
  elif isinstance(loaded.get("actor_state_dict"), dict) and "sr" in loaded["actor_state_dict"]:
    bundle = loaded["actor_state_dict"]

  if bundle is None:
    raise ValueError(
      f"{checkpoint_path} is not a MoE goalkeeper bundle. "
      "Build one with scripts/build_moe7_goalkeeper.py so teacher rollout uses gate + experts."
    )
  return MoE6GoalkeeperPolicy(bundle, env, device)


def _reset_policy(policy) -> None:
  reset = getattr(policy, "reset", None)
  if callable(reset):
    reset()


def _get_actor_obs(obs: Any) -> torch.Tensor:
  if isinstance(obs, (tuple, list)):
    return _get_actor_obs(obs[0])
  if isinstance(obs, dict):
    return obs["actor"]
  if hasattr(obs, "get") and "actor" in obs:
    return obs["actor"]
  raise KeyError("Goalkeeper teacher observations must include an 'actor' group")


def _ball_entered_goal(env: ManagerBasedRlEnv, cfg: CollectConfig) -> torch.Tensor:
  ball = env.scene["ball"]
  pos = ball.data.root_link_pos_w
  rel_x = pos[:, 0] - env.scene.env_origins[:, 0]
  rel_y = pos[:, 1] - env.scene.env_origins[:, 1]
  return (rel_x <= cfg.goal_x) & (rel_y.abs() <= cfg.goal_half_width) & (pos[:, 2] <= cfg.goal_height)


def _bad_orientation(env: ManagerBasedRlEnv, limit_angle: float) -> torch.Tensor:
  robot = env.scene["robot"]
  projected_gravity = robot.data.projected_gravity_b
  return torch.acos(-projected_gravity[:, 2]).abs() > limit_angle


def _prepare_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
  launched = getattr(env, "_gk_delayed_ball_launched", None)
  if launched is not None:
    return ~launched
  return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)


def _collect_batch(env, policy, cfg: CollectConfig) -> dict[str, Any]:
  obs = env.reset()
  if isinstance(obs, tuple):
    obs = obs[0]
  _reset_policy(policy)

  base_env = env.unwrapped
  device = base_env.device
  num_envs = base_env.num_envs
  active = torch.ones(num_envs, dtype=torch.bool, device=device)
  active_steps = torch.full((num_envs,), cfg.max_steps, dtype=torch.long, device=device)
  ball_entered_goal = torch.zeros(num_envs, dtype=torch.bool, device=device)
  prepare_fell = torch.zeros(num_envs, dtype=torch.bool, device=device)

  student_obs_steps: list[torch.Tensor] = []
  teacher_action_steps: list[torch.Tensor] = []
  valid_mask_steps: list[torch.Tensor] = []
  condition_steps: list[torch.Tensor] = []

  for step in range(cfg.max_steps):
    ball_entered_goal |= _ball_entered_goal(base_env, cfg)
    if cfg.require_prepare_stable:
      prepare_fell |= _bad_orientation(base_env, cfg.fall_limit_angle) & _prepare_mask(base_env)

    with torch.inference_mode():
      actor_obs = _get_actor_obs(obs)
      condition = goalkeeper_prediction_condition(base_env)
      student_frame = build_goalkeeper_student_obs(actor_obs, condition)
      action = policy(obs)

    student_obs_steps.append(student_frame.detach().cpu())
    teacher_action_steps.append(action.detach().cpu())
    valid_mask_steps.append(active.detach().cpu())
    condition_steps.append(condition.detach().cpu())

    result = env.step(action)
    obs = result[0]
    ball_entered_goal |= _ball_entered_goal(base_env, cfg)
    raw_term = result[2]
    if isinstance(raw_term, torch.Tensor):
      terminated = raw_term.to(device=device, dtype=torch.bool).view(-1)
    else:
      terminated = torch.as_tensor(raw_term, device=device, dtype=torch.bool).view(-1)

    newly_terminated = active & terminated
    if torch.any(newly_terminated):
      active_steps[newly_terminated] = step + 1
    active = active & ~terminated
    if not torch.any(active):
      break

  student_obs_tensor = torch.stack(student_obs_steps, dim=1)
  teacher_action_tensor = torch.stack(teacher_action_steps, dim=1)
  valid_mask = torch.stack(valid_mask_steps, dim=1)
  prediction_condition = torch.stack(condition_steps, dim=1)
  prepare_stable = ~prepare_fell
  blocked = ~ball_entered_goal
  success = blocked & prepare_stable

  metadata = {
    "success": success.cpu(),
    "blocked": blocked.cpu(),
    "prepare_stable": prepare_stable.cpu(),
    "ball_entered_goal": ball_entered_goal.cpu(),
    "keep_length": active_steps.cpu(),
    "prediction_condition": prediction_condition,
  }
  return {
    "student_obs": student_obs_tensor,
    "teacher_action": teacher_action_tensor,
    "valid_mask": valid_mask,
    "metadata": metadata,
    "summary": {
      "episodes": num_envs,
      "valid_steps": int(valid_mask.sum().item()),
      "mean_keep_length": float(active_steps.float().mean().item()),
      "success": int(success.sum().item()),
      "blocked": int(blocked.sum().item()),
      "prepare_stable": int(prepare_stable.sum().item()),
    },
  }


def _index_batch(batch: dict[str, Any], index: torch.Tensor) -> dict[str, Any]:
  index = index.cpu()
  metadata = {
    key: value[index] if isinstance(value, torch.Tensor) else value
    for key, value in batch["metadata"].items()
  }
  return {
    "student_obs": batch["student_obs"][index],
    "teacher_action": batch["teacher_action"][index],
    "valid_mask": batch["valid_mask"][index],
    "metadata": metadata,
    "summary": {
      "episodes": int(index.numel()),
      "valid_steps": int(batch["valid_mask"][index].sum().item()),
      "mean_keep_length": float(metadata["keep_length"].float().mean().item()),
      "success": int(metadata["success"].sum().item()),
      "blocked": int(metadata["blocked"].sum().item()),
      "prepare_stable": int(metadata["prepare_stable"].sum().item()),
    },
  }


def _slice_batch(batch: dict[str, Any], count: int) -> dict[str, Any]:
  return _index_batch(batch, torch.arange(count))


def _filter_batch(batch: dict[str, Any], cfg: CollectConfig) -> dict[str, Any]:
  keep = torch.ones(batch["student_obs"].shape[0], dtype=torch.bool)
  if cfg.success_only:
    keep &= batch["metadata"]["blocked"].bool()
  if cfg.require_prepare_stable:
    keep &= batch["metadata"]["prepare_stable"].bool()
  return _index_batch(batch, keep.nonzero(as_tuple=False).squeeze(-1))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
  path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_collection(cfg: CollectConfig) -> None:
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  if cfg.delayed_launch:
    _apply_delayed_launch_sampling(env_cfg, wait_s=cfg.launch_delay_s)
  if cfg.success_only and "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None

  run_tag = cfg.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  out_dir = Path(cfg.output_dir).expanduser().resolve() / run_tag
  shard_dir = out_dir / "shards"
  if out_dir.exists() and any(out_dir.iterdir()) and not cfg.overwrite:
    raise FileExistsError(f"Output directory already exists and is not empty: {out_dir}")
  shard_dir.mkdir(parents=True, exist_ok=True)

  print(f"[INFO] Collecting goalkeeper teacher dataset on {device}")
  print(f"[INFO] Output: {out_dir}")
  print(f"[INFO] Episodes to save: {cfg.num_episodes}, envs/batch: {cfg.num_envs}")
  if cfg.success_only or cfg.require_prepare_stable:
    print(
      "[INFO] Filtering:"
      f" success_only={cfg.success_only},"
      f" require_prepare_stable={cfg.require_prepare_stable},"
      f" max_attempt_episodes={cfg.max_attempt_episodes}"
    )

  env_base = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env_base, clip_actions=100.0)
  policy = _load_teacher_policy(cfg.checkpoint, env, device)

  total_episodes = 0
  total_valid_steps = 0
  attempted_episodes = 0
  raw_success = 0
  raw_blocked = 0
  raw_prepare_stable = 0
  try:
    remaining = cfg.num_episodes
    shard_idx = 0
    while remaining > 0:
      if cfg.max_attempt_episodes > 0 and attempted_episodes >= cfg.max_attempt_episodes:
        print(
          f"[WARN] Stopping early after {attempted_episodes} attempted episodes; "
          f"saved {total_episodes}/{cfg.num_episodes} filtered episodes."
        )
        break
      batch = _collect_batch(env, policy, cfg)
      attempted_episodes += int(batch["summary"]["episodes"])
      raw_success += int(batch["summary"]["success"])
      raw_blocked += int(batch["summary"]["blocked"])
      raw_prepare_stable += int(batch["summary"]["prepare_stable"])
      if cfg.success_only or cfg.require_prepare_stable:
        batch = _filter_batch(batch, cfg)
      if batch["summary"]["episodes"] == 0:
        print(
          f"[INFO] attempt_episodes={attempted_episodes} saved={total_episodes}/{cfg.num_episodes} "
          f"raw_success={raw_success} raw_blocked={raw_blocked} raw_prepare_stable={raw_prepare_stable}"
        )
        continue

      take = min(int(batch["summary"]["episodes"]), remaining)
      if take < int(batch["summary"]["episodes"]):
        batch = _slice_batch(batch, take)

      shard_path = shard_dir / f"shard_{shard_idx:06d}.pt"
      torch.save({
        "student_obs": batch["student_obs"],
        "teacher_action": batch["teacher_action"],
        "valid_mask": batch["valid_mask"],
        "metadata": batch["metadata"],
        "config": asdict(cfg),
      }, shard_path)

      total_episodes += int(batch["summary"]["episodes"])
      total_valid_steps += int(batch["summary"]["valid_steps"])
      remaining -= take
      shard_idx += 1
      print(
        f"[INFO] shard={shard_idx:04d} episodes={total_episodes}/{cfg.num_episodes} "
        f"attempted={attempted_episodes} valid_steps={total_valid_steps} "
        f"raw_success={raw_success} raw_blocked={raw_blocked} raw_prepare_stable={raw_prepare_stable}"
      )
      gc.collect()
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
  finally:
    env.close()

  summary = {
    "episodes": total_episodes,
    "attempted_episodes": attempted_episodes,
    "raw_success": raw_success,
    "raw_blocked": raw_blocked,
    "raw_prepare_stable": raw_prepare_stable,
    "yield_rate": total_episodes / max(attempted_episodes, 1),
    "valid_steps": total_valid_steps,
    "mean_valid_steps_per_episode": total_valid_steps / max(total_episodes, 1),
  }
  _save_json(out_dir / "summary.json", summary)
  _save_json(out_dir / "config.json", asdict(cfg))
  print(f"[INFO] Done. Summary: {json.dumps(summary, indent=2)}")


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  cfg = tyro.cli(CollectConfig, prog="collect_goalkeeper_teacher_dataset")
  run_collection(cfg)


if __name__ == "__main__":
  main()
