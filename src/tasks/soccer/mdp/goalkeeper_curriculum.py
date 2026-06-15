"""Curriculum terms for goalkeeper training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import torch

from src.tasks.soccer.mdp.goalkeeper_ball_reset import RegionBallVelCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class GoalkeeperCurriculumStage(TypedDict, total=False):
  """One goalkeeper curriculum stage keyed by environment step count."""

  step: int
  name: str
  regions: list[dict[str, Any]]
  ball_start_x_range: tuple[float, float]
  ball_end_x_range: tuple[float, float]
  t_flight_range: tuple[float, float]
  ball_start_y_range: tuple[float, float]
  ball_start_z_range: tuple[float, float]
  push_vel_xy_range: tuple[float, float]
  push_vel_z_range: tuple[float, float]
  push_ang_vel_range: tuple[float, float]
  push_interval_range_s: tuple[float, float]
  perturb_vel_range: tuple[float, float]
  perturb_interval_range_s: tuple[float, float]
  curriculum_update: int
  curriculumsigma: float
  reward_weights: dict[str, float]


def _pair(value: Any) -> tuple[float, float]:
  return (float(value[0]), float(value[1]))


def _stage_for_step(
  stages: list[GoalkeeperCurriculumStage],
  step: int,
) -> tuple[int, GoalkeeperCurriculumStage]:
  """Return the last stage whose threshold is <= step."""
  if not stages:
    raise ValueError("goalkeeper curriculum requires at least one stage")

  active_index = 0
  for index, stage in enumerate(stages):
    if step >= int(stage.get("step", 0)):
      active_index = index
  return active_index, stages[active_index]


def _get_term_cfg(manager: Any, term_name: str) -> Any | None:
  try:
    return manager.get_term_cfg(term_name)
  except (KeyError, ValueError):
    return None


def _with_stage_ball_ranges(
  base: RegionBallVelCfg,
  stage: GoalkeeperCurriculumStage,
) -> RegionBallVelCfg:
  regions = stage.get("regions", base.regions)
  return RegionBallVelCfg(
    ball_start_x_range=_pair(stage.get("ball_start_x_range", base.ball_start_x_range)),
    ball_end_x_range=_pair(stage.get("ball_end_x_range", base.ball_end_x_range)),
    t_flight_range=_pair(stage.get("t_flight_range", base.t_flight_range)),
    regions=[
      {
        "height": _pair(region["height"]),
        "width": _pair(region["width"]),
        "motion_id": int(region.get("motion_id", index)),
      }
      for index, region in enumerate(regions)
    ],
    ball_start_y_range=_pair(stage.get("ball_start_y_range", base.ball_start_y_range)),
    ball_start_z_range=_pair(stage.get("ball_start_z_range", base.ball_start_z_range)),
    interception_x=float(stage.get("interception_x", base.interception_x)),
    balanced_regions=bool(stage.get("balanced_regions", base.balanced_regions)),
    train_t_flight_range=(
      _pair(stage["train_t_flight_range"])
      if "train_t_flight_range" in stage
      else base.train_t_flight_range
    ),
    play_t_flight_range=(
      _pair(stage["play_t_flight_range"])
      if "play_t_flight_range" in stage
      else base.play_t_flight_range
    ),
  )


def goalkeeper_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  stages: list[GoalkeeperCurriculumStage],
) -> dict[str, torch.Tensor]:
  """Progress goalkeeper from stable standing to full-speed interception.

  The curriculum is evaluated before reset events. It mutates the active reset
  and reward term configs so the next ball launch uses the selected stage.
  """
  del env_ids  # Stage selection is global, keyed by training step.

  stage_index, stage = _stage_for_step(stages, int(env.common_step_counter))
  setattr(env, "_gk_curriculum_stage_name", stage.get("name", f"stage_{stage_index}"))
  setattr(env, "_gk_curriculum_update", int(stage.get("curriculum_update", stage_index)))
  setattr(env, "_gk_curriculumsigma", float(stage.get("curriculumsigma", 5.0)))

  reset_ball_cfg = _get_term_cfg(env.event_manager, "reset_ball")
  if reset_ball_cfg is not None:
    base_vel_cfg = getattr(env, "_gk_base_ball_vel_cfg", None)
    if base_vel_cfg is None:
      base_vel_cfg = reset_ball_cfg.params["vel_cfg"]
      setattr(env, "_gk_base_ball_vel_cfg", base_vel_cfg)
    reset_ball_cfg.params["vel_cfg"] = _with_stage_ball_ranges(base_vel_cfg, stage)

  push_cfg = _get_term_cfg(env.event_manager, "push_robot")
  if push_cfg is not None:
    if "push_vel_xy_range" in stage:
      push_cfg.params["vel_xy_range"] = _pair(stage["push_vel_xy_range"])
    if "push_vel_z_range" in stage:
      push_cfg.params["vel_z_range"] = _pair(stage["push_vel_z_range"])
    if "push_ang_vel_range" in stage:
      push_cfg.params["ang_vel_range"] = _pair(stage["push_ang_vel_range"])
    if "push_interval_range_s" in stage:
      push_cfg.interval_range_s = _pair(stage["push_interval_range_s"])

  perturb_cfg = _get_term_cfg(env.event_manager, "perturb_ball_vel")
  if perturb_cfg is not None:
    if "perturb_vel_range" in stage:
      perturb_cfg.params["vel_range"] = _pair(stage["perturb_vel_range"])
    if "perturb_interval_range_s" in stage:
      perturb_cfg.interval_range_s = _pair(stage["perturb_interval_range_s"])

  for reward_name, weight in stage.get("reward_weights", {}).items():
    reward_cfg = _get_term_cfg(env.reward_manager, reward_name)
    if reward_cfg is None:
      raise ValueError(f"unknown goalkeeper reward term in curriculum: {reward_name}")
    reward_cfg.weight = float(weight)

  current_vel_cfg = reset_ball_cfg.params["vel_cfg"] if reset_ball_cfg is not None else None
  num_regions = current_vel_cfg.num_regions if current_vel_cfg is not None else 0
  t_min, t_max = (
    current_vel_cfg.t_flight_range if current_vel_cfg is not None else (0.0, 0.0)
  )
  perturb_abs = 0.0
  if perturb_cfg is not None:
    perturb_range = perturb_cfg.params.get("vel_range", (0.0, 0.0))
    perturb_abs = max(abs(float(perturb_range[0])), abs(float(perturb_range[1])))

  tensor = lambda value: torch.tensor(float(value), device=env.device)
  return {
    "stage_index": tensor(stage_index),
    "stage_step": tensor(stage.get("step", 0)),
    "num_regions": tensor(num_regions),
    "t_flight_min": tensor(t_min),
    "t_flight_max": tensor(t_max),
    "perturb_abs": tensor(perturb_abs),
  }
