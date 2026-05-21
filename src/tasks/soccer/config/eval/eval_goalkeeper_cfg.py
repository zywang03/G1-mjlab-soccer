"""Goalkeeper evaluation config — matches Humanoid-Goalkeeper paper eval protocol.

Adds ball position and velocity observations with 10-frame history stacking
(matching the paper's HIMPPO actor input). Domain randomization is minimal:
no push, no ball perturb, no joint randomization, no observation noise.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.soccer.config.g1.env_cfgs import unitree_g1_goalkeeper_env_cfg
from src.tasks.soccer.mdp import (
  ball_pos_in_robot_frame,
  ball_vel_in_robot_frame,
)

_BALL_CFG = SceneEntityCfg("ball")


def eval_goalkeeper_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Goalkeeper evaluation config matching Humanoid-Goalkeeper paper.

  Differences from training config:
  - history_length=10 on actor observations (paper's T=10 frame stacking)
  - ball_pos_local: ball position in robot pelvis frame, Oball ∈ R3 (actor + critic)
  - ball_vel_local: ball velocity in robot pelvis frame, vball ∈ R3 (critic only)
  - Observation noise: disabled (matching paper's add_noise=False in play mode)
  - Domain randomization: minimal (no push, no ball perturb, no joint randomize)

  The base goalkeeper config already uses parabolic trajectory ball launching
  with 6 regions, matching the paper's assign_ball_states approach.
  """
  cfg = unitree_g1_goalkeeper_env_cfg(play=play)

  # Disable observation noise for clean eval (matching paper).
  cfg.observations["actor"].enable_corruption = False

  # 10-frame history stacking (paper: num_actor_history=10, 96×10=960D).
  cfg.observations["actor"].history_length = 10

  # Add ball position to actor (Oball in paper Table I).
  actor_terms = dict(cfg.observations["actor"].terms)
  actor_terms["ball_pos_local"] = ObservationTermCfg(
    func=ball_pos_in_robot_frame,
    params={"ball_cfg": _BALL_CFG},
  )
  cfg.observations["actor"].terms = actor_terms

  # Add ball position + velocity to critic.
  critic_terms = dict(cfg.observations["critic"].terms)
  critic_terms["ball_pos_local"] = ObservationTermCfg(
    func=ball_pos_in_robot_frame,
    params={"ball_cfg": _BALL_CFG},
  )
  critic_terms["ball_vel_local"] = ObservationTermCfg(
    func=ball_vel_in_robot_frame,
    params={"ball_cfg": _BALL_CFG},
  )
  cfg.observations["critic"].terms = critic_terms

  # Remove push_robot and perturb_ball_vel if they exist (not in base config).
  cfg.events.pop("push_robot", None)
  cfg.events.pop("perturb_ball_vel", None)

  return cfg
