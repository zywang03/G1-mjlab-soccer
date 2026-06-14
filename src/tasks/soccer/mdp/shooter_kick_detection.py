"""Kick contact detection and tracking for coordinated soccer rewards.

Port of HumanoidSoccer's kick_detection.py to mjlab, with anti-clamp
improvements: uses per-foot contact sensors to determine real contact
attribution instead of closest-foot heuristic.

Provides KickContactTracker that consolidates ball contact force sensing,
foot identification, and reward window management across multiple reward terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from .shooter_commands import MultiMotionSoccerCommand


@dataclass
class KickContactEvent:
  """Per-step kick contact detection results."""
  new_contact: torch.Tensor      # (num_envs,) bool — first valid contact this episode
  kick_detected: torch.Tensor    # (num_envs,) bool — any ball-robot contact force > threshold
  peak_force: torch.Tensor       # (num_envs,) float
  # Per-foot contact flags.
  left_foot_contact: torch.Tensor   # (num_envs,) bool
  right_foot_contact: torch.Tensor  # (num_envs,) bool
  nonfoot_contact: torch.Tensor     # (num_envs,) bool
  # Valid kick: correct foot only, no other contact.
  valid_kick: torch.Tensor       # (num_envs,) bool — valid kick event this step


@dataclass
class ContactFootInfo:
  """Resolved foot metadata for environments with an active kick contact."""
  env_ids: torch.Tensor         # (K,) long
  body_indices: torch.Tensor    # (K,) long — which foot body made contact
  sides: torch.Tensor           # (K,) int8 — 0=left, 1=right
  expected: torch.Tensor        # (K,) int8 — expected kick leg from motion


class KickContactTracker:
  """Shared kick contact detection reusable across multiple reward terms.

  Uses three sensors:
    - ball_robot_contact: ball vs entire robot (any contact, for forces/metrics)
    - left_foot_ball_contact: ball vs left ankle subtree
    - right_foot_ball_contact: ball vs right ankle subtree
    - nonfoot_ball_contact: ball vs all collision geoms EXCEPT foot geoms

  Nonfoot contact is read directly from the dedicated nonfoot_ball_contact
  sensor, which uses geom-level matching with a negative-lookahead regex.
  This correctly detects body-ball contact even when feet also touch
  simultaneously.

  Valid kick requires:
    - Expected foot contacts ball
    - Other foot does NOT contact ball
    - Non-foot body does NOT contact ball
    - Ball speed exceeds threshold within a short window

  Caches detection results per step so all reward terms share one sensor read.
  """

  def __init__(self, env: ManagerBasedRlEnv, state_prefix: str):
    self._env = env
    self._state_prefix = state_prefix
    self._device = env.device
    self._num_envs = env.num_envs
    self._cache_valid = False
    self._cached_event: KickContactEvent | None = None
    self._any_cache_valid = False
    self._cached_any_event: KickContactEvent | None = None

  def begin_step(self):
    """Reset per-step cache at the beginning of each command update."""
    self._cache_valid = False
    self._cached_event = None
    self._any_cache_valid = False
    self._cached_any_event = None

  def detect(
    self,
    command: MultiMotionSoccerCommand,
    ball_sensor_name: str,
    horizontal_force_threshold: float,
  ) -> KickContactEvent:
    """Detect new kick contacts using per-foot sensors.

    Args:
      command: The motion command (provides kick_leg expected foot).
      ball_sensor_name: Name of the whole-body ball contact sensor.
      horizontal_force_threshold: Force threshold for contact detection.

    Returns:
      KickContactEvent with per-foot contact flags and valid_kick determination.
    """
    if self._cache_valid and self._cached_event is not None:
      return self._cached_event

    device = self._device
    num_envs = self._num_envs

    # Read whole-body ball-robot sensor.
    any_force = self._read_sensor_peak_force(ball_sensor_name)
    # Read per-foot sensors.
    left_force = self._read_sensor_peak_force("left_foot_ball_contact")
    right_force = self._read_sensor_peak_force("right_foot_ball_contact")
    # Read dedicated nonfoot sensor (geom-based, excludes foot geoms).
    nonfoot_force = self._read_sensor_peak_force("nonfoot_ball_contact")

    # Determine per-region contact flags.
    c_any = any_force > horizontal_force_threshold
    c_L = left_force > horizontal_force_threshold
    c_R = right_force > horizontal_force_threshold
    c_N = nonfoot_force > horizontal_force_threshold

    # Determine expected foot from motion data.
    kick_leg = command.kick_leg  # (num_envs,) int8: 0=left, 1=right, -1=unknown
    expected_left = kick_leg == 0
    expected_right = kick_leg == 1

    # Expected foot has contact.
    expected_contact = (expected_left & c_L) | (expected_right & c_R)
    # Other foot does NOT have contact.
    other_foot_clear = ~((expected_left & c_R) | (expected_right & c_L))
    # Non-foot body does NOT have contact.
    nonfoot_clear = ~c_N
    # Kick leg is known.
    leg_known = kick_leg >= 0

    # Valid kick event this step.
    valid_kick_this_step = expected_contact & other_foot_clear & nonfoot_clear & leg_known

    # Track contact_awarded (any contact, for backward compat / proximity logic).
    contact_awarded = self._get_or_init_bool("target_contact_awarded", default=False)
    new_any_contact = (~contact_awarded) & c_any
    if torch.any(new_any_contact):
      contact_awarded[new_any_contact] = True

    # Track valid_kick_awarded (first valid kick this episode).
    valid_kick_awarded = self._get_or_init_bool("valid_kick_awarded", default=False)
    new_valid_kick = (~valid_kick_awarded) & valid_kick_this_step
    if torch.any(new_valid_kick):
      valid_kick_awarded[new_valid_kick] = True

    self._update_detection_state(new_valid_kick)

    event = KickContactEvent(
      new_contact=new_valid_kick,
      kick_detected=c_any,
      peak_force=any_force,
      left_foot_contact=c_L,
      right_foot_contact=c_R,
      nonfoot_contact=c_N,
      valid_kick=valid_kick_this_step,
    )
    self._cached_event = event
    self._cache_valid = True
    return event

  def detect_any_single_foot(
    self,
    ball_sensor_name: str,
    horizontal_force_threshold: float,
  ) -> KickContactEvent:
    """Detect first valid student kick using either foot.

    Student shooter policies do not receive a motion-reference kick-side label,
    so a valid kick is any single-foot ball contact with no simultaneous
    non-foot contact. This intentionally avoids the Stage II expected-leg gate.
    """
    if self._any_cache_valid and self._cached_any_event is not None:
      return self._cached_any_event

    any_force = self._read_sensor_peak_force(ball_sensor_name)
    left_force = self._read_sensor_peak_force("left_foot_ball_contact")
    right_force = self._read_sensor_peak_force("right_foot_ball_contact")
    nonfoot_force = self._read_sensor_peak_force("nonfoot_ball_contact")

    c_any = any_force > horizontal_force_threshold
    c_L = left_force > horizontal_force_threshold
    c_R = right_force > horizontal_force_threshold
    c_N = nonfoot_force > horizontal_force_threshold

    single_foot = c_L ^ c_R
    valid_kick_this_step = c_any & single_foot & (~c_N)

    contact_awarded = self._get_or_init_bool("student_target_contact_awarded", default=False)
    new_any_contact = (~contact_awarded) & c_any
    if torch.any(new_any_contact):
      contact_awarded[new_any_contact] = True

    valid_kick_awarded = self._get_or_init_bool("student_valid_kick_awarded", default=False)
    new_valid_kick = (~valid_kick_awarded) & valid_kick_this_step
    if torch.any(new_valid_kick):
      valid_kick_awarded[new_valid_kick] = True

    event = KickContactEvent(
      new_contact=new_valid_kick,
      kick_detected=c_any,
      peak_force=any_force,
      left_foot_contact=c_L,
      right_foot_contact=c_R,
      nonfoot_contact=c_N,
      valid_kick=valid_kick_this_step,
    )
    self._cached_any_event = event
    self._any_cache_valid = True
    return event

  def get_student_valid_kick_awarded(self) -> torch.Tensor:
    """Return whether a motion-free student valid kick has occurred."""
    return self._get_or_init_bool("student_valid_kick_awarded", default=False)

  def freeze_student_proximity_reward(self, env_ids: torch.Tensor, values: torch.Tensor):
    frozen = self._get_or_init_float("student_frozen_proximity_reward", default=0.0)
    frozen[env_ids] = values
    frozen_flag = self._get_or_init_bool("student_proximity_frozen", default=False)
    frozen_flag[env_ids] = True

  def get_student_frozen_proximity_reward(self) -> torch.Tensor:
    return self._get_or_init_float("student_frozen_proximity_reward", default=0.0)

  def get_student_proximity_frozen(self) -> torch.Tensor:
    return self._get_or_init_bool("student_proximity_frozen", default=False)

  def record_expected_success(self, mask: torch.Tensor, expected_mask: torch.Tensor):
    """Store whether a detected kick matched the expected leg."""
    state = self._get_or_init_bool("expected_kick_success", default=False)
    state[mask] = expected_mask[mask]

  def get_contact_awarded(self) -> torch.Tensor:
    """Return whether any ball-robot contact has occurred this episode."""
    return self._get_or_init_bool("target_contact_awarded", default=False)

  def get_valid_kick_awarded(self) -> torch.Tensor:
    """Return whether a valid kick has occurred this episode."""
    return self._get_or_init_bool("valid_kick_awarded", default=False)

  def freeze_proximity_reward(self, env_ids: torch.Tensor, values: torch.Tensor):
    frozen = self._get_or_init_float("frozen_proximity_reward", default=0.0)
    frozen[env_ids] = values
    frozen_flag = self._get_or_init_bool("proximity_frozen", default=False)
    frozen_flag[env_ids] = True

  def get_frozen_proximity_reward(self) -> torch.Tensor:
    return self._get_or_init_float("frozen_proximity_reward", default=0.0)

  def get_proximity_frozen(self) -> torch.Tensor:
    """Return whether proximity has been frozen (explicit bool, not value-based)."""
    return self._get_or_init_bool("proximity_frozen", default=False)

  def resolve_contact_foot(
    self,
    command: MultiMotionSoccerCommand,
    foot_cfg: SceneEntityCfg,
    mask: torch.Tensor,
  ) -> ContactFootInfo:
    """Determine which foot made contact using per-foot sensors.

    Unlike the old closest-foot heuristic, this uses actual sensor data.
    """
    env_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if env_ids.numel() == 0:
      empty = torch.zeros(0, dtype=torch.long, device=self._device)
      zeros_i8 = torch.zeros(0, dtype=torch.int8, device=self._device)
      return ContactFootInfo(empty, empty, zeros_i8, zeros_i8)

    # Use cached event to determine which foot actually made contact.
    event = self._cached_event
    if event is None:
      empty = torch.zeros(0, dtype=torch.long, device=self._device)
      zeros_i8 = torch.zeros(0, dtype=torch.int8, device=self._device)
      return ContactFootInfo(empty, empty, zeros_i8, zeros_i8)

    # Determine sides from actual sensor contact.
    c_L_sub = event.left_foot_contact[env_ids]
    c_R_sub = event.right_foot_contact[env_ids]

    # If left foot has contact → side=0, if right → side=1.
    # If both or neither, default to left (will fail valid_kick check anyway).
    sides = torch.where(c_R_sub, torch.ones_like(c_L_sub, dtype=torch.int8),
                        torch.zeros_like(c_L_sub, dtype=torch.int8))

    # Get body indices for foot bodies.
    robot = command.robot
    left_idx = robot.body_names.index("left_ankle_roll_link")
    right_idx = robot.body_names.index("right_ankle_roll_link")
    body_indices = torch.where(
      c_R_sub,
      torch.full((env_ids.numel(),), right_idx, dtype=torch.long, device=self._device),
      torch.full((env_ids.numel(),), left_idx, dtype=torch.long, device=self._device),
    )

    expected = command.kick_leg[env_ids].to(torch.int8).clamp(min=0)
    return ContactFootInfo(env_ids, body_indices, sides, expected)

  # -- Internal helpers ---------------------------------------------------------

  def _read_sensor_peak_force(self, sensor_name: str) -> torch.Tensor:
    """Read peak contact force from a named sensor. Returns zeros if unavailable."""
    sensor = self._get_contact_sensor(sensor_name)
    if sensor is None:
      return torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)

    force = sensor.data.force
    if force is None or force.numel() == 0:
      return torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)

    force = force.to(device=self._device)
    # force shape: [B, N_slots, 3] or [B, 3]
    force_norm = torch.linalg.vector_norm(force, dim=-1)  # [B, N] or [B]
    peak_force = force_norm.amax(dim=-1) if force_norm.ndim > 1 else force_norm  # [B]
    return peak_force

  def _get_contact_sensor(self, name: str):
    sensors = self._env.scene.sensors
    if sensors is None:
      return None
    if isinstance(sensors, dict):
      return sensors.get(name)
    try:
      return sensors[name]
    except (KeyError, TypeError):
      return None

  def _tensor_name(self, suffix: str) -> str:
    return f"{self._state_prefix}_{suffix}"

  def _get_or_init_bool(self, suffix: str, default: bool) -> torch.Tensor:
    name = self._tensor_name(suffix)
    t = getattr(self._env, name, None)
    if t is None or t.shape[0] != self._num_envs:
      t = torch.full((self._num_envs,), default, dtype=torch.bool, device=self._device)
      setattr(self._env, name, t)
    return t.to(device=self._device, dtype=torch.bool)

  def _get_or_init_float(self, suffix: str, default: float) -> torch.Tensor:
    name = self._tensor_name(suffix)
    t = getattr(self._env, name, None)
    if t is None or t.shape[0] != self._num_envs:
      t = torch.full((self._num_envs,), default, dtype=torch.float32, device=self._device)
      setattr(self._env, name, t)
    return t.to(device=self._device, dtype=torch.float32)

  def _update_detection_state(self, new_valid_kick: torch.Tensor):
    if not torch.any(new_valid_kick):
      return
    s = self._get_or_init_bool("kick_success", default=False)
    s[new_valid_kick] = True

  def _handle_resample(self, resample_flags: torch.Tensor):
    """Reset kick state for environments that just resampled motions."""
    if not torch.any(resample_flags):
      return
    contact = self._get_or_init_bool("target_contact_awarded", default=False)
    valid_kick = self._get_or_init_bool("valid_kick_awarded", default=False)
    success = self._get_or_init_bool("kick_success", default=False)
    expected = self._get_or_init_bool("expected_kick_success", default=False)
    frozen = self._get_or_init_float("frozen_proximity_reward", default=0.0)
    student_contact = self._get_or_init_bool("student_target_contact_awarded", default=False)
    student_valid_kick = self._get_or_init_bool("student_valid_kick_awarded", default=False)
    student_frozen = self._get_or_init_float("student_frozen_proximity_reward", default=0.0)
    student_frozen_flag = self._get_or_init_bool("student_proximity_frozen", default=False)

    contact[resample_flags] = False
    valid_kick[resample_flags] = False
    success[resample_flags] = False
    expected[resample_flags] = False
    frozen[resample_flags] = 0.0
    student_contact[resample_flags] = False
    student_valid_kick[resample_flags] = False
    student_frozen[resample_flags] = 0.0
    student_frozen_flag[resample_flags] = False

    # Reset proximity frozen flag.
    frozen_flag = self._get_or_init_bool("proximity_frozen", default=False)
    frozen_flag[resample_flags] = False

    # Reset reward timers.
    for suffix in ["dir_align_timer", "speed_timer", "z_speed_timer", "student_dir_align_timer", "student_speed_timer"]:
      name = self._tensor_name(suffix)
      t = getattr(self._env, name, None)
      if t is not None and t.shape[0] == self._num_envs:
        t[resample_flags] = 0

    # -- goal-plane crossing states (Stage 3) --
    for suffix in [
      "goal_cross_processed", "prev_ball_local_valid",
      "goal_cross_cache_crossed", "goal_cross_cache_in_goal",
    ]:
      name = self._tensor_name(suffix)
      t = getattr(self._env, name, None)
      if t is not None and t.shape[0] == self._num_envs:
        t[resample_flags] = False

    for suffix in ["prev_ball_local", "goal_cross_cache_cross_pos"]:
      name = self._tensor_name(suffix)
      t = getattr(self._env, name, None)
      if t is not None and t.shape[0] == self._num_envs:
        t[resample_flags] = 0.0

    for suffix in ["goal_cross_cache_target_error"]:
      name = self._tensor_name(suffix)
      t = getattr(self._env, name, None)
      if t is not None and t.shape[0] == self._num_envs:
        t[resample_flags] = 0.0

    # Invalidate per-step goal-plane cache so reward terms recompute on this step.
    cache_step_name = self._tensor_name("goal_cross_cache_step")
    if hasattr(self._env, cache_step_name):
      setattr(self._env, cache_step_name, -1)
