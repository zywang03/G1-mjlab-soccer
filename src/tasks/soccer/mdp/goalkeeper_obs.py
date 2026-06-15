"""Goalkeeper-specific observation functions.

Includes:
  - Privileged critic observations (ee_positions, reach_distance, end_target, region)
  - Scaled observations matching the reference Humanoid-Goalkeeper paper
  - Reference default joint positions (goalkeeper stance)

The pretrained model expects observations with specific scaling and
centering. Without these, the model receives inputs outside its training
distribution and produces junk actions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.sensor import BuiltinSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.entity import Entity

# -- Reference default joint positions (goalkeeper stance) -------------------
# Matches Humanoid-Goalkeeper repo g1_29_config.py init_state.default_joint_angles.
# Ordered by the 29-DOF joint sequence:
#   left_leg(6) → right_leg(6) → waist(3) → left_arm(7) → right_arm(7)
#
# The model was trained with obs = (joint_pos - default) * 1.0, so we must
# use the same default. Our HOME_KEYFRAME has different shoulder/elbow/hip_roll
# angles, which would shift the observation distribution.

_REF_DEFAULT_DOF_POS: list[float] = [
  # Left leg
  -0.1,  # left_hip_pitch_joint
  0.2,   # left_hip_roll_joint
  0.0,   # left_hip_yaw_joint
  0.3,   # left_knee_joint
  -0.2,  # left_ankle_pitch_joint
  -0.2,  # left_ankle_roll_joint
  # Right leg
  -0.1,  # right_hip_pitch_joint
  -0.2,  # right_hip_roll_joint
  0.0,   # right_hip_yaw_joint
  0.3,   # right_knee_joint
  -0.2,  # right_ankle_pitch_joint
  0.2,   # right_ankle_roll_joint
  # Waist
  0.0,   # waist_yaw_joint
  0.0,   # waist_roll_joint
  0.0,   # waist_pitch_joint
  # Left arm
  0.0,   # left_shoulder_pitch_joint
  0.5,   # left_shoulder_roll_joint
  0.0,   # left_shoulder_yaw_joint
  1.2,   # left_elbow_joint
  0.0,   # left_wrist_roll_joint
  0.0,   # left_wrist_pitch_joint
  0.0,   # left_wrist_yaw_joint
  # Right arm
  0.0,   # right_shoulder_pitch_joint
  -0.5,  # right_shoulder_roll_joint
  0.0,   # right_shoulder_yaw_joint
  1.2,   # right_elbow_joint
  0.0,   # right_wrist_roll_joint
  0.0,   # right_wrist_pitch_joint
  0.0,   # right_wrist_yaw_joint
]

# Joint name order matching the 29-DOF robot sequence.
# Must correspond one-to-one with _REF_DEFAULT_DOF_POS.
_GK_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# GK reference default joint positions as a dict (joint_name → angle).
_GK_DEFAULT_JOINT_POS: dict[str, float] = dict(zip(_GK_JOINT_NAMES, _REF_DEFAULT_DOF_POS))

# Per-joint action scale.
#
# Since the GK-specific articulation sets actuator stiffness/damping to match
# the reference PD gains exactly, the action scale is uniform 0.25 (matching
# the reference's action_scale = 0.25).
_GK_ACTION_SCALE: float = 0.25


def _make_gk_action_scale_dict() -> dict[str, float]:
  """Build per-joint action scale dict matching the reference's 0.25."""
  return {name: _GK_ACTION_SCALE for name in _GK_JOINT_NAMES}


# -- GK-specific actuator configs --------------------------------------------
# Reference PD gains from g1_29_config.py:
#   stiffness(kp): hip_yaw=150, hip_roll=150, hip_pitch=150, knee=300,
#                  ankle=40, shoulder=150, elbow=150, waist=150, wrist=20
#   damping(kd):   hip_yaw=2, hip_roll=2, hip_pitch=2, knee=4,
#                  ankle=2, shoulder=2, elbow=2, waist=2, wrist=0.5
#
# Our armature-based stiffness is 3-9x weaker (e.g. shoulder=17 vs ref=150),
# causing the robot to sag under gravity with insufficient restorative torque.
# Setting stiffness/damping to match reference exactly fixes posture maintenance.

from mjlab.actuator import BuiltinPositionActuatorCfg
from src.assets.robots.unitree_g1.g1_constants import (
  ACTUATOR_5020, ACTUATOR_7520_14, ACTUATOR_7520_22, ACTUATOR_4010,
  EntityArticulationInfoCfg, FULL_COLLISION,
)

_GK_ACTUATOR_HIP_PITCH_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=150.0,
  damping=2.0,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)

_GK_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=150.0,
  damping=2.0,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)

_GK_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=300.0,
  damping=4.0,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)

_GK_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=40.0,
  damping=2.0,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

_GK_ACTUATOR_SHOULDER_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint", ".*_elbow_joint",
  ),
  stiffness=150.0,
  damping=2.0,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)

_GK_ACTUATOR_WRIST_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_roll_joint",),
  stiffness=20.0,
  damping=0.5,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)

_GK_ACTUATOR_WRIST_PITCH_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=20.0,
  damping=0.5,
  effort_limit=ACTUATOR_4010.effort_limit,
  armature=ACTUATOR_4010.reflected_inertia,
)

_GK_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
  stiffness=150.0,
  damping=2.0,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

_GK_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    _GK_ACTUATOR_SHOULDER_ELBOW,
    _GK_ACTUATOR_WRIST_ROLL,
    _GK_ACTUATOR_HIP_PITCH_YAW,
    _GK_ACTUATOR_HIP_ROLL,
    _GK_ACTUATOR_KNEE,
    _GK_ACTUATOR_WRIST_PITCH_YAW,
    _GK_ACTUATOR_WAIST,
    _GK_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_gk_robot_cfg():
  """Create G1 robot config with reference-matched PD gains for goalkeeper.

  Uses actuator stiffness and damping matching the Humanoid-Goalkeeper
  paper's reference PD controller (kp=40-300, kd=0.5-4). This is critical
  because the pretrained policy expects strong position-error restoring
  torque that our armature-based stiffness (3-9x weaker) cannot provide.

  With matched PD gains, the action scale is uniformly 0.25 (matching the
  reference's action_scale).
  """
  from src.assets.robots import get_g1_robot_cfg as _get_base_g1

  cfg = _get_base_g1()
  cfg.articulation = _GK_ARTICULATION
  return cfg


# -- Scaled observation functions --------------------------------------------
# The pretrained model expects observations with specific scaling factors
# matching the reference repo's obs_scales config:
#   dof_vel * 0.05, ang_vel * 0.25, lin_vel * 2.0, ball_vel * 0.2
# Without these, the input distribution shifts and the model produces garbage.

_OBS_SCALE_ANG_VEL = 0.25
_OBS_SCALE_DOF_VEL = 0.05
_OBS_SCALE_LIN_VEL = 2.0
_OBS_SCALE_BALL_VEL = 0.2


def _cached_body_indices(
  env: ManagerBasedRlEnv,
  robot: Entity,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  cache = getattr(env, "_gk_body_index_cache", None)
  if cache is None:
    cache = {}
    setattr(env, "_gk_body_index_cache", cache)
  key = (robot.__class__.__name__, id(robot), tuple(body_names))
  device = robot.data.body_link_pos_w.device
  cached = cache.get(key)
  if cached is None or cached.device != device:
    cached = torch.as_tensor(
      robot.find_bodies(body_names, preserve_order=True)[0],
      device=device,
      dtype=torch.long,
    )
    cache[key] = cached
  return cached


def gk_joint_pos_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Joint position deviation from goalkeeper reference default.

  Matches: (dof_pos - default_dof_pos) * 1.0 from the reference.
  Uses the reference goalkeeper stance as the default, NOT the HOME_KEYFRAME.
  """
  asset: Entity = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  default = torch.tensor(
    _REF_DEFAULT_DOF_POS, device=asset.data.joint_pos.device, dtype=torch.float32
  )[jnt_ids]
  return asset.data.joint_pos[:, jnt_ids] - default


def gk_joint_vel_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Joint velocity scaled by 0.05 (matching reference obs_scales.dof_vel)."""
  asset: Entity = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, jnt_ids] * _OBS_SCALE_DOF_VEL


def gk_ang_vel(
  env: ManagerBasedRlEnv,
  sensor_name: str = "robot/imu_ang_vel",
) -> torch.Tensor:
  """Base angular velocity scaled by 0.25 (matching reference obs_scales.ang_vel)."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return sensor.data * _OBS_SCALE_ANG_VEL


def gk_lin_vel(
  env: ManagerBasedRlEnv,
  sensor_name: str = "robot/imu_lin_vel",
) -> torch.Tensor:
  """Base linear velocity scaled by 2.0 (matching reference obs_scales.lin_vel).

  Only used in critic privileged observations.
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return sensor.data * _OBS_SCALE_LIN_VEL


def gk_ball_pos_local(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  position_noise: float = 0.0,
  dropout_after_s: float = 0.4,
  dropout_prob: float = 0.0,
  stop_speed_threshold: float = 0.1,
  use_official_visibility: bool = False,
) -> torch.Tensor:
  """Ball position in robot pelvis frame with paper-style perception masking.

  Training can add up to 5cm position noise, randomly drop observations after
  0.4s of flight, and zero the observation once the ball is no longer flying.
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  ball_pos_w = ball.data.root_link_pos_w
  robot_pos_w = robot.data.root_link_pos_w
  robot_quat_w = robot.data.root_link_quat_w
  delta_w = ball_pos_w - robot_pos_w
  obs = quat_apply_inverse(robot_quat_w, delta_w)

  if position_noise > 0.0:
    obs = obs + (torch.rand_like(obs) * 2.0 - 1.0) * position_noise

  mask = torch.zeros(env.num_envs, dtype=torch.bool, device=obs.device)

  ball_vel = getattr(ball.data, "root_link_lin_vel_w", None)
  if ball_vel is not None:
    stopped = torch.norm(ball_vel, dim=-1) <= stop_speed_threshold
    mask |= stopped

  if use_official_visibility:
    catchstep = getattr(env, "_gk_catchstep", None)
    startstep = getattr(env, "_gk_startstep", None)
    vanish_step = getattr(env, "_gk_vanish_step", None)
    if catchstep is None or catchstep.shape[0] != env.num_envs:
      catchstep = torch.zeros(env.num_envs, dtype=torch.long, device=obs.device)
    else:
      catchstep = catchstep.to(device=obs.device)
    if startstep is None or startstep.shape[0] != env.num_envs:
      startstep = torch.zeros(env.num_envs, dtype=torch.long, device=obs.device)
    else:
      startstep = startstep.to(device=obs.device)
    if vanish_step is None or vanish_step.shape[0] != env.num_envs:
      vanish_step = torch.zeros(env.num_envs, dtype=torch.long, device=obs.device)
    else:
      vanish_step = vanish_step.to(device=obs.device)
    initial_visible = catchstep < startstep
    random_visible = catchstep > vanish_step
    flying = (
      (obs[:, 0] > 0.05)
      & (obs[:, 0] < 3.4)
      & (obs[:, 1] > -2.0)
      & (obs[:, 1] < 2.0)
      & (obs[:, 2] < 1.8)
      & (catchstep > 0)
    )
    visible = initial_visible & flying & random_visible
    mask |= ~visible

  step_buf = getattr(env, "episode_length_buf", None)
  step_dt = getattr(env, "step_dt", None)
  if step_buf is not None and step_dt is not None and dropout_prob > 0.0:
    after_dropout_time = step_buf.to(obs.device).float() * float(step_dt) >= dropout_after_s
    if dropout_prob >= 1.0:
      dropout = after_dropout_time
    else:
      dropout = after_dropout_time & (torch.rand(env.num_envs, device=obs.device) < dropout_prob)
    mask |= dropout

  return torch.where(mask.unsqueeze(-1), torch.zeros_like(obs), obs)


def gk_ball_vel_local(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Ball velocity in robot pelvis frame, scaled by 0.2 (matching reference).

  Only used in critic privileged observations.
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  ball_vel_w = ball.data.root_link_lin_vel_w
  robot_quat_w = robot.data.root_link_quat_w
  return quat_apply_inverse(robot_quat_w, ball_vel_w) * _OBS_SCALE_BALL_VEL


def goalkeeper_sidestep_command(
  env: ManagerBasedRlEnv,
  position_gain: float = 2.0,
  max_speed: float = 1.2,
  deadzone: float = 0.1,
  approach_distance_threshold: float = 0.8,
  goal_line_x: float = 0.0,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Desired lateral base motion toward the dynamic interception target.

  The actor observes both the signed target-y error and a clipped y-velocity
  command. This makes the goalkeeper task look more like a velocity-commanded
  locomotion problem while still using the paper's landing-point/live-ball
  target switch.
  """
  from src.tasks.soccer.mdp.goalkeeper_rewards import goalkeeper_dynamic_target_pos

  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  target = goalkeeper_dynamic_target_pos(
    env,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )
  y_error = target[:, 1] - robot.data.root_link_pos_w[:, 1]
  cmd_vy = torch.clamp(position_gain * y_error, -max_speed, max_speed)
  active = torch.abs(y_error) > deadzone
  before_goal_line = ball.data.root_link_pos_w[:, 0] > goal_line_x
  cmd_vy = torch.where(active & before_goal_line, cmd_vy, torch.zeros_like(cmd_vy))
  return torch.stack([y_error, cmd_vy], dim=-1)


def gk_last_action(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
) -> torch.Tensor:
  """Raw (unscaled) last action, matching the reference observation.

  The reference stores the model's raw output as the last_action observation.
  mjlab's `env.action_manager.action` returns the *processed* action
  (raw * scale + offset), which would feed scaled values into the history.
  Using raw_action keeps the observation distribution consistent regardless
  of action_scale.
  """
  return env.action_manager.get_term(action_name).raw_action


# -- Privileged critic observations ------------------------------------------


def goalkeeper_ee_positions(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ee_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  ),
) -> torch.Tensor:
  """Hand positions in robot pelvis frame (6D privileged obs).

  Returns both hand positions relative to the pelvis, rotated into the
  pelvis coordinate frame. Flattened to (B, 6) for concatenation.
  """
  robot: Entity = env.scene[robot_cfg.name]
  pelvis_pos = robot.data.root_link_pos_w  # (B, 3)
  pelvis_quat = robot.data.root_link_quat_w  # (B, 4)

  indices = _cached_body_indices(env, robot, ee_body_names)
  hand_pos_w = robot.data.body_link_pos_w[:, indices]  # (B, 2, 3)
  delta = hand_pos_w - pelvis_pos.unsqueeze(1)  # (B, 2, 3)
  hand_pos_local = quat_apply_inverse(
    pelvis_quat.unsqueeze(1).expand(-1, 2, -1), delta
  )  # (B, 2, 3)
  return hand_pos_local.reshape(env.num_envs, -1)  # (B, 6)


def goalkeeper_ball_distance(
  env: ManagerBasedRlEnv,
  approach_distance_threshold: float = 0.8,
  goal_line_x: float = 0.0,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Region-selected end-effector to dynamic target distance (1D privileged obs).

  This matches the paper's privileged reach-distance term. The legacy function
  name is kept so existing critic layouts do not need a semantic rename.
  """
  from src.tasks.soccer.mdp.goalkeeper_rewards import (
    goalkeeper_selected_hand_target_distance,
  )

  return goalkeeper_selected_hand_target_distance(
    env,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )


def goalkeeper_end_target_pos(
  env: ManagerBasedRlEnv,
  approach_distance_threshold: float = 0.8,
  goal_line_x: float = 0.0,
  ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """End target position in robot pelvis frame (3D privileged obs).

  The end target is where the robot should aim its hands to intercept. It uses
  the sampled landing point while far away and switches to the live ball near
  the goalkeeper, matching the paper's dynamic target definition.
  """
  from src.tasks.soccer.mdp.goalkeeper_rewards import goalkeeper_dynamic_target_pos

  robot: Entity = env.scene[robot_cfg.name]
  robot_pos_w = robot.data.root_link_pos_w
  robot_quat_w = robot.data.root_link_quat_w
  target_w = goalkeeper_dynamic_target_pos(
    env,
    approach_distance_threshold=approach_distance_threshold,
    goal_line_x=goal_line_x,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
  )
  delta_w = target_w - robot_pos_w
  return quat_apply_inverse(robot_quat_w, delta_w)  # (B, 3)


def goalkeeper_end_region(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Goal region ID (0-5), normalized by 3.0 to ~[0, 1.67] (1D privileged obs).

  Matches the reference paper's end_regions / 3.0 privileged term.
  Stored by reset_ball_with_parabolic_trajectory on the env object.
  """
  t = getattr(env, "_gk_region", None)
  if t is None:
    t = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    setattr(env, "_gk_region", t)
  val = t / 3.0
  return val.unsqueeze(-1)  # (B, 1)
