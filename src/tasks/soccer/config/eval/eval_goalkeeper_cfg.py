"""Goalkeeper evaluation config — matches Humanoid-Goalkeeper paper eval protocol.

Adds ball position and velocity observations with 10-frame history stacking
(matching the paper's HIMPPO actor input). Domain randomization is minimal:
no push, no ball perturb, no joint randomization, no observation noise.

Critical for pretrained checkpoint compatibility:
  - Action scale: uniform 0.25 (reference PD controller scale)
  - Actuator PD gains: matched to reference kp/kd (hip=150/2, knee=300/4, etc.)
  - Default joint position: goalkeeper reference stance (not HOME_KEYFRAME)
  - Observation scaling: matches reference obs_scales
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots.unitree_g1.g1_constants import FULL_COLLISION
from src.tasks.soccer.config.g1.training_env_cfgs import unitree_g1_goalkeeper_training_env_cfg
from src.tasks.soccer.config.soccer_settings import SETTINGS
from src.tasks.soccer.mdp.goalkeeper_obs import (
  _GK_DEFAULT_JOINT_POS,
  get_gk_robot_cfg,
)
from src.tasks.soccer.mdp.goalkeeper_actions import GoalkeeperJointPositionActionCfg

# With reference-matched PD gains, the action scale is uniformly 0.25.
_GK_ACTION = 0.25


def _gk_robot_at(pos: tuple[float, float, float], yaw: float = 0.0):
  """Create G1 robot with GK articulation (ref-matched PD gains) and GK stance."""
  import math

  half = yaw / 2.0
  cfg = get_gk_robot_cfg()
  cfg.init_state = replace(
    cfg.init_state,
    pos=pos,
    rot=(math.cos(half), 0.0, 0.0, math.sin(half)),
    joint_pos=_GK_DEFAULT_JOINT_POS,
  )
  cfg.collisions = (FULL_COLLISION,)
  return cfg


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
  cfg = unitree_g1_goalkeeper_training_env_cfg(play=True)

  # -- Override robot with GK articulation and GK default joint positions ----
  # GK articulation: actuator PD gains match reference kp/kd exactly.
  # GK default joint positions: action offset base (not HOME_KEYFRAME).
  # With matched PD gains, action_scale = 0.25 uniformly.
  s = SETTINGS.scene
  cfg.scene.entities["robot"] = _gk_robot_at(tuple(s.goalkeeper_pos), 0.0)

  # Action scale is uniformly 0.25 (our PD gains now match reference).
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, GoalkeeperJointPositionActionCfg)
  joint_pos_action.scale = _GK_ACTION

  # Disable observation noise for clean eval (matching paper).
  cfg.observations["actor"].enable_corruption = False
  cfg.observations["actor"].history_length = 10

  # Remove push_robot and perturb_ball_vel if they exist (not in base config).
  cfg.events.pop("push_robot", None)
  cfg.events.pop("perturb_ball_vel", None)

  # Increase contact capacity for goalkeeper self-collisions.
  cfg.sim.nconmax = 128

  return cfg
