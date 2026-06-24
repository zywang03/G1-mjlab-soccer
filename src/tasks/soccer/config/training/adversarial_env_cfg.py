"""Dual-robot adversarial training environment config helpers."""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg

from src.assets.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from src.assets.robots.unitree_g1.g1_constants import FULL_COLLISION, HOME_KEYFRAME
from src.tasks.soccer.config.soccer_settings import SETTINGS
from src.tasks.soccer.mdp.goalkeeper_obs import _GK_DEFAULT_JOINT_POS, get_gk_robot_cfg
from src.tasks.soccer.mdp.opponent_obs import (
  opponent_joint_state,
  opponent_root_state_in_robot_frame,
)


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
  import math

  half = yaw / 2.0
  return (math.cos(half), 0.0, 0.0, math.sin(half))


def make_shooter_robot() -> object:
  cfg = get_g1_robot_cfg()
  cfg.init_state = replace(
    HOME_KEYFRAME,
    pos=tuple(SETTINGS.scene.shooter_pos),
    rot=_yaw_to_quat(0.0),
  )
  cfg.collisions = (FULL_COLLISION,)
  return cfg


def make_goalkeeper_robot() -> object:
  cfg = get_gk_robot_cfg()
  cfg.init_state = replace(
    cfg.init_state,
    pos=tuple(SETTINGS.scene.goalkeeper_pos),
    rot=_yaw_to_quat(0.0),
    joint_pos=_GK_DEFAULT_JOINT_POS,
  )
  cfg.collisions = (FULL_COLLISION,)
  return cfg


def _add_opponent_observations(cfg: ManagerBasedRlEnvCfg) -> None:
  terms = {
    "opponent_root": ObservationTermCfg(func=opponent_root_state_in_robot_frame),
    "opponent_joints": ObservationTermCfg(func=opponent_joint_state),
  }
  cfg.observations["actor"].terms.update(terms)
  cfg.observations["critic"].terms.update(terms)


def _keep_active_action_observation(cfg: ManagerBasedRlEnvCfg) -> None:
  for group_name in ("actor", "critic"):
    term = cfg.observations[group_name].terms.get("actions")
    if term is None:
      continue
    params = dict(term.params or {})
    params["action_name"] = "joint_pos"
    cfg.observations[group_name].terms["actions"] = replace(term, params=params)


def _add_opponent_action(
  cfg: ManagerBasedRlEnvCfg,
  scale: float,
) -> None:
  cfg.actions["opponent_joint_pos"] = JointPositionActionCfg(
    entity_name="opponent",
    actuator_names=(".*",),
    scale=scale,
    use_default_offset=True,
  )


def make_shooter_adversarial_env_cfg(base_cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Add a real frozen goalkeeper opponent to a shooter training cfg."""
  cfg = base_cfg
  cfg.scene.entities["robot"] = make_shooter_robot()
  cfg.scene.entities["opponent"] = make_goalkeeper_robot()
  _add_opponent_action(cfg, scale=0.25)
  _keep_active_action_observation(cfg)
  _add_opponent_observations(cfg)
  cfg.sim.nconmax = max(cfg.sim.nconmax, 256)
  cfg.sim.njmax = max(cfg.sim.njmax, 3000)
  return cfg


def make_goalkeeper_adversarial_env_cfg(
  base_cfg: ManagerBasedRlEnvCfg,
  *,
  add_opponent_obs: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Add a real frozen shooter opponent to a goalkeeper training cfg."""
  cfg = base_cfg
  cfg.scene.entities["robot"] = make_goalkeeper_robot()
  cfg.scene.entities["opponent"] = make_shooter_robot()
  _add_opponent_action(cfg, scale=G1_ACTION_SCALE)
  _keep_active_action_observation(cfg)
  if add_opponent_obs:
    _add_opponent_observations(cfg)
  cfg.sim.nconmax = max(cfg.sim.nconmax, 256)
  cfg.sim.njmax = max(cfg.sim.njmax, 3000)
  return cfg
