"""Multi-motion soccer command for shooter training.

Port of HumanoidSoccer's commands_multi_motion_soccer.py to mjlab.
Provides MultiMotionLoader (loads a directory of .npz motion files),
MultiMotionSoccerCommand (mjlab CommandTerm with ball placement), and
MultiMotionSoccerCommandCfg (dataclass configuration).
"""

from __future__ import annotations

import glob
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def _pad_tensor_stack(tensors: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
  """Pad a list of tensors to max_T and stack into (M, max_T, ...)."""
  max_T = max(t.shape[0] for t in tensors)
  padded = []
  for t in tensors:
    if t.shape[0] < max_T:
      pad_shape = (max_T - t.shape[0],) + t.shape[1:]
      pad = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
      t = torch.cat([t, pad], dim=0)
    padded.append(t)
  return torch.stack(padded, dim=0)


# BFS→DFS permutation: Isaac Lab articulation → MJCF joint order.
# Isaac Lab uses breadth-first ordering for articulation DOFs; MJCF uses
# depth-first.  Verified against Isaac Lab robot.find_joints() output
# and confirmed with FK test (max body position error 0.0001m).
_IL_TO_MJCF_JOINT = [
  0,  3,  6,  9, 13, 17,        # left leg chain
  1,  4,  7, 10, 14, 18,        # right leg chain
  2,  5,  8,                     # waist_yaw, waist_roll, waist_pitch
  11, 15, 19, 21, 23, 25, 27,   # left arm chain
  12, 16, 20, 22, 24, 26, 28,   # right arm chain
]

# BFS→DFS permutation: Isaac Lab articulation → MJCF body order.
# Verified against Isaac Lab robot.find_bodies() output and confirmed
# with FK test (max body position error 0.0001m).
_IL_TO_MJCF_BODY = [
  0,  1,  4,  7, 10, 14, 18,   # pelvis, left leg chain
  2,  5,  8, 11, 15, 19,        # right leg chain
  3,  6,  9,                     # waist_yaw, waist_roll, torso
  12, 16, 20, 22, 24, 26, 28,   # left arm chain
  13, 17, 21, 23, 25, 27, 29,   # right arm chain
]


class MultiMotionLoader:
  """Load and index multiple .npz motion files with padding to uniform length.

  Each .npz file contains: joint_pos, joint_vel, body_pos_w, body_quat_w,
  body_lin_vel_w, body_ang_vel_w, fps, and optionally kick_leg.

  The .npz data is in Isaac Lab URDF articulation order. We permute both
  joint and body data to MJCF order on load so all downstream indexing
  uses MJCF indices.
  """

  def __init__(self, motion_dir: str, body_indexes: torch.Tensor, device: str = "cpu",
               motion_glob: str = "*.npz"):
    pattern = os.path.join(motion_dir, motion_glob)
    files = sorted(glob.glob(pattern))
    if not files:
      raise FileNotFoundError(f"No files matching {pattern}")
    self.num_files = len(files)
    self.device = device

    self.motion_names: list[str] = []
    self._file_lengths: list[int] = []
    joint_pos_list: list[torch.Tensor] = []
    joint_vel_list: list[torch.Tensor] = []
    body_pos_w_list: list[torch.Tensor] = []
    body_quat_w_list: list[torch.Tensor] = []
    body_lin_vel_w_list: list[torch.Tensor] = []
    body_ang_vel_w_list: list[torch.Tensor] = []
    kick_leg_labels: list[str | None] = []

    il_to_mjcf_joint = torch.tensor(_IL_TO_MJCF_JOINT, dtype=torch.long, device=device)
    il_to_mjcf_body = torch.tensor(_IL_TO_MJCF_BODY, dtype=torch.long, device=device)

    for f in files:
      data = np.load(f)
      self.motion_names.append(os.path.splitext(os.path.basename(f))[0])
      self._file_lengths.append(data["joint_pos"].shape[0])

      jp = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
      jv = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
      # Permute joints from IL order to MJCF order.
      joint_pos_list.append(jp[:, il_to_mjcf_joint])
      joint_vel_list.append(jv[:, il_to_mjcf_joint])

      bp = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
      bq = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
      blv = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
      bav = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
      # Permute bodies from IL order to MJCF order.
      body_pos_w_list.append(bp[:, il_to_mjcf_body])
      body_quat_w_list.append(bq[:, il_to_mjcf_body])
      body_lin_vel_w_list.append(blv[:, il_to_mjcf_body])
      body_ang_vel_w_list.append(bav[:, il_to_mjcf_body])

      label: str | None = None
      if "kick_leg" in data.files:
        raw = str(data["kick_leg"]).strip().lower()
        if raw in ("left", "right"):
          label = raw
      kick_leg_labels.append(label)

    # Pad and stack.
    self._joint_pos = _pad_tensor_stack(joint_pos_list)        # (M, max_T, 29) MJCF order
    self._joint_vel = _pad_tensor_stack(joint_vel_list)        # (M, max_T, 29)
    self._body_pos_w = _pad_tensor_stack(body_pos_w_list)      # (M, max_T, 30, 3) MJCF order
    self._body_quat_w = _pad_tensor_stack(body_quat_w_list)    # (M, max_T, 30, 4)
    self._body_lin_vel_w = _pad_tensor_stack(body_lin_vel_w_list)  # (M, max_T, 30, 3)
    self._body_ang_vel_w = _pad_tensor_stack(body_ang_vel_w_list)  # (M, max_T, 30, 3)

    self.time_step_total = self._joint_pos.shape[1]  # max_T
    self.file_lengths = torch.tensor(self._file_lengths, dtype=torch.long, device=device)
    self._kick_leg_labels = tuple(kick_leg_labels)

    # Body-indexed views (only the configured subset).
    # body_indexes are MJCF indices — data is now in MJCF order, so this is correct.
    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, :, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, :, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, :, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, :, self._body_indexes]

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._joint_pos

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._joint_vel

  @property
  def kick_leg_labels(self) -> tuple[str | None, ...]:
    return self._kick_leg_labels

  def get_first_frame_anchor_pos(self, motion_idx: int, anchor_body_idx: int) -> torch.Tensor:
    return self._body_pos_w[motion_idx, 0, anchor_body_idx]

  def get_last_frame_anchor_pos(self, motion_idx: int, anchor_body_idx: int, motion_length: int) -> torch.Tensor:
    last = max(0, motion_length - 1)
    return self._body_pos_w[motion_idx, last, anchor_body_idx]


class MultiMotionSoccerCommand(CommandTerm):
  """Command term that plays back multi-motion soccer kick references.

  Extends mjlab's CommandTerm. Exposes the same anchor/body/joint properties
  as the built-in MotionCommand so the existing tracking reward functions work.
  """

  cfg: MultiMotionSoccerCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MultiMotionSoccerCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MultiMotionLoader(cfg.motion_dir, self.body_indexes, device=self.device,
                                      motion_glob=cfg.motion_glob)

    # Per-environment state.
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_length[:] = self.motion.file_lengths[self.motion_idx]

    # Relative transforms (yaw-aligned, updated per step).
    self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
    self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
    self.body_quat_relative_w[:, :, 0] = 1.0

    # Kick leg mapping.
    kick_leg_to_id = {"left": 0, "right": 1}
    self._kick_leg_id_to_name = {0: "left", 1: "right", -1: "unknown"}
    self.motion_kick_leg = torch.full((self.motion.num_files,), -1, dtype=torch.int8, device=self.device)
    for idx, label in enumerate(self.motion.kick_leg_labels):
      normalized = label.lower() if isinstance(label, str) else ""
      if normalized in kick_leg_to_id:
        self.motion_kick_leg[idx] = kick_leg_to_id[normalized]

    # Adaptive sampling state.
    self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(
      (self.motion.num_files, self.bin_count), dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      (self.motion.num_files, self.bin_count), dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    # Soccer-specific state.
    self.target_point_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.soccer_ball_pos = torch.zeros_like(self.target_point_pos)
    self.target_destination_pos = torch.zeros_like(self.target_point_pos)
    self.initial_target_point_pos = torch.zeros_like(self.target_point_pos)

    # Ball entity reference.
    self.soccer_ball: Entity | None = None
    if hasattr(env.scene, "__getitem__"):
      try:
        self.soccer_ball = env.scene[cfg.ball_entity_name]
      except KeyError:
        self.soccer_ball = None

    # Curve offset for ball placement variation.
    self.curve_radius_offset = torch.zeros(self.num_envs, device=self.device)
    cc = cfg.curve_offset_range
    if cc is not None:
      self._radius_min, self._radius_max = cc.radius[0], cc.radius[1]
      self._target_arc_angle = cc.arc_angle
      self._target_height = cc.height
    else:
      self._radius_min, self._radius_max = 0.0, 0.0
      self._target_arc_angle = 0.0
      self._target_height = 0.0

    # Destination rectangle (in local env coords).
    dest = cfg.destination_center
    self.destination_center = torch.tensor(dest, device=self.device)
    self.destination_length = cfg.destination_length
    self.destination_width = cfg.destination_width

    # Ball init velocity flag.
    self._enable_ball_init_vel = cfg.enable_soccer_ball_init_vel
    bvr = cfg.soccer_ball_init_lin_vel_range or {}
    self._ball_init_vel_ranges = torch.tensor(
      [[bvr.get(k, (0.0, 0.0))[0], bvr.get(k, (0.0, 0.0))[1]] for k in ("x", "y", "z")],
      device=self.device,
    )

    # Initialize metrics.
    self._init_metrics()

    # Initialize kick contact tracker for Stage II rewards.
    from .shooter_kick_detection import KickContactTracker
    self.kick_contact_tracker = KickContactTracker(env, "_motion")

    # Resample all envs on first reset (handled by CommandTerm.reset() calling _resample).
    all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    self._sample_soccer_offset(all_env_ids)
    self._compute_soccer_ball_positions(all_env_ids)
    self._update_soccer_ball(all_env_ids)
    self._update_target_points(all_env_ids)
    self._update_destination_points(all_env_ids)

  def _init_metrics(self):
    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_motion"] = torch.zeros(self.num_envs, device=self.device)

  # -- CommandTerm abstract interface -------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  # -- Motion reference properties (match mjlab MotionCommand API) --------------

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.motion_idx, self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.motion_idx, self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self.motion.body_pos_w[self.motion_idx, self.time_steps] + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.motion_idx, self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.motion_idx, self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.motion_idx, self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (self.motion.body_pos_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index]
            + self._env.scene.env_origins)

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index]

  # -- Robot state properties ---------------------------------------------------

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_pelvis_pos_w(self) -> torch.Tensor:
    pelvis_idx = self.robot.body_names.index("pelvis")
    return self.robot.data.body_link_pos_w[:, pelvis_idx]

  @property
  def robot_pelvis_quat_w(self) -> torch.Tensor:
    pelvis_idx = self.robot.body_names.index("pelvis")
    return self.robot.data.body_link_quat_w[:, pelvis_idx]

  # -- Kick leg -----------------------------------------------------------------

  @property
  def kick_leg(self) -> torch.Tensor:
    return self.motion_kick_leg[self.motion_idx]

  @property
  def kick_leg_name(self) -> list[str]:
    ids = self.motion_kick_leg[self.motion_idx].tolist()
    return [self._kick_leg_id_to_name.get(i, "unknown") for i in ids]

  # -- CommandTerm lifecycle ----------------------------------------------------


  def _compute_relative_transforms(self):
    """Compute yaw-aligned relative body transforms (ref → robot anchor frame)."""
    n_bodies = len(self.cfg.body_names)
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, n_bodies, 1)
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, n_bodies, 1)
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, n_bodies, 1)
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, n_bodies, 1)

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

  def _update_metrics(self):
    from mjlab.utils.lab_api.math import quat_error_magnitude
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )
    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _update_command(self):
    """Advance motion frame, resample on completion, update relative transforms."""
    self.kick_contact_tracker.begin_step()
    self.time_steps += 1
    env_ids = torch.where(self.time_steps >= self.motion_length)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    # Update target point from simulation (tracks the ball as it moves).
    self._update_target_points_from_sim()

    # Freeze initial_target_point_pos once a valid kick occurs.
    if hasattr(self, "kick_contact_tracker"):
      valid_kick_awarded = self.kick_contact_tracker.get_valid_kick_awarded()
      no_valid_kick = ~valid_kick_awarded
      if torch.any(no_valid_kick):
        self.initial_target_point_pos[no_valid_kick] = self.target_point_pos[no_valid_kick]

    self._compute_relative_transforms()

    # Update adaptive failure histogram.
    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _resample_command(self, env_ids: torch.Tensor):
    if env_ids.numel() == 0:
      return

    if self.cfg.sampling_mode == "start":
      # Eval mode: only reset time_steps, don't move robot/ball.
      self.time_steps[env_ids] = 0
      return

    self._sample_soccer_offset(env_ids)
    if self.cfg.sampling_mode == "adaptive":
      self._adaptive_sampling(env_ids)
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      self._uniform_sampling(env_ids)

    self._compute_soccer_ball_positions(env_ids)
    self._update_soccer_ball(env_ids)
    self._update_target_points(env_ids)
    self._update_destination_points(env_ids)

    # Write robot state to sim (root pose + joint positions).
    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=self.device)
    rand = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
    root_pos[env_ids] += rand[:, 0:3]
    orient_delta = quat_from_euler_xyz(rand[:, 3], rand[:, 4], rand[:, 5])
    root_ori[env_ids] = quat_mul(orient_delta, root_ori[env_ids])

    # Apply global offset for eval (shift from motion origin to world position).
    gox, goy, goz = self.cfg.motion_origin_offset
    if gox != 0.0 or goy != 0.0 or goz != 0.0:
      root_pos[env_ids, 0] += gox
      root_pos[env_ids, 1] += goy
      root_pos[env_ids, 2] += goz
    if self.cfg.motion_yaw_offset != 0.0:
      yaw_delta = quat_from_euler_xyz(
        torch.zeros(len(env_ids), device=self.device),
        torch.zeros(len(env_ids), device=self.device),
        torch.full((len(env_ids),), self.cfg.motion_yaw_offset, device=self.device),
      )
      root_ori[env_ids] = quat_mul(yaw_delta, root_ori[env_ids])

    range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=self.device)
    rand = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
    root_lin_vel[env_ids] += rand[:, :3]
    root_ang_vel[env_ids] += rand[:, 3:]

    jp = self.joint_pos.clone()
    jv = self.joint_vel.clone()
    jp += sample_uniform(self.cfg.joint_position_range[0], self.cfg.joint_position_range[1],
                          jp.shape, device=jp.device)
    limits = self.robot.data.soft_joint_pos_limits[env_ids]
    jp[env_ids] = torch.clamp(jp[env_ids], limits[:, :, 0], limits[:, :, 1])

    self.robot.write_joint_state_to_sim(jp[env_ids], jv[env_ids], env_ids=env_ids)
    root_state = torch.cat([root_pos[env_ids], root_ori[env_ids],
                            root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.clear_state(env_ids=env_ids)

    # Compute relative transforms NOW so terminations see valid ref_z on step 0.
    # (Normally done in _update_command, but that runs after termination check.)
    self._compute_relative_transforms()

    # Update action offset so zero-action targets the motion reference pose,
    # not the standing pose. Without this, the PD controller pulls the robot
    # back to standing immediately → motion-reference terminations fire.
    action_term = self._env.action_manager._terms.get("joint_pos")
    if action_term is not None and hasattr(action_term, "_offset"):
      action_term._offset[env_ids] = jp[env_ids]

    # Notify kick tracker of resample.
    if hasattr(self, "kick_contact_tracker"):
      flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
      flags[env_ids] = True
      self.kick_contact_tracker._handle_resample(flags)

  # -- Sampling strategies ------------------------------------------------------

  def _uniform_sampling(self, env_ids: torch.Tensor):
    n = len(env_ids)
    self.motion_idx[env_ids] = torch.randint(0, self.motion.num_files, (n,), device=self.device)
    self.motion_length[env_ids] = self.motion.file_lengths[self.motion_idx[env_ids]]
    # Start from frame 0 for uniform sampling (clean start each episode).
    self.time_steps[env_ids] = 0

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    # Collect failure counts from environments that just terminated.
    episode_failed = self._env.termination_manager.terminated[env_ids]
    self._current_bin_failed.zero_()
    if torch.any(episode_failed):
      failed_mask = episode_failed
      failed_motion_idx = self.motion_idx[env_ids][failed_mask]
      failed_lengths = self.motion_length[env_ids][failed_mask].clamp(min=1).float()
      failed_steps = self.time_steps[env_ids][failed_mask].float()
      failed_phase = failed_steps / (failed_lengths - 1.0 + 1e-6)
      failed_bins = torch.clamp((failed_phase * self.bin_count).long(), 0, self.bin_count - 1)
      flat_idx = failed_motion_idx * self.bin_count + failed_bins
      flat_size = int(self.motion.num_files * self.bin_count)
      flat_counts = torch.zeros(flat_size, dtype=torch.float, device=self.device)
      if flat_idx.numel() > 0:
        flat_idx = flat_idx.long()
        ones = torch.ones_like(flat_idx, dtype=torch.float, device=self.device)
        flat_counts.index_add_(0, flat_idx, ones)
      self._current_bin_failed[:] = flat_counts.view(self.motion.num_files, self.bin_count)

    M = max(1, int(self.motion.num_files))
    B = max(1, int(self.bin_count))
    uniform_per_pair = self.cfg.adaptive_uniform_ratio / float(M * B)
    probs = self.bin_failed_count + self._current_bin_failed + uniform_per_pair
    probs = torch.nn.functional.pad(
      probs.unsqueeze(1), (0, self.cfg.adaptive_kernel_size - 1), mode="replicate"
    )
    probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).squeeze(1)
    probs = probs.view(-1)
    probs = probs / (probs.sum() + 1e-12)

    sampled_flat = torch.multinomial(probs, len(env_ids), replacement=True)
    sampled_motion = sampled_flat // self.bin_count
    sampled_bins = sampled_flat % self.bin_count
    self.motion_idx[env_ids] = sampled_motion
    self.motion_length[env_ids] = self.motion.file_lengths[self.motion_idx[env_ids]]
    rand_offset = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
    sampled_phase = (sampled_bins.float() + rand_offset) / float(self.bin_count)
    self.time_steps[env_ids] = (sampled_phase * (self.motion_length[env_ids].float() - 1)).long()

    H = -(probs * (probs + 1e-12).log()).sum()
    denom = math.log(self.bin_count * max(1, int(self.motion.num_files)))
    self.metrics["sampling_entropy"][:] = H / denom if denom > 1e-12 else 0.0
    pmax, imax = probs.max(dim=0)
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = (imax % self.bin_count).float() / self.bin_count
    self.metrics["sampling_top1_motion"][:] = (imax // self.bin_count).float()

  # -- Soccer ball helpers ------------------------------------------------------

  def _sample_soccer_offset(self, env_ids: torch.Tensor):
    if self._radius_min == 0.0 and self._radius_max == 0.0:
      self.curve_radius_offset[env_ids] = 0.0
      return
    rand = torch.rand(len(env_ids), device=self.device)
    span = self._radius_max - self._radius_min
    self.curve_radius_offset[env_ids] = self._radius_min + rand * span

  def _compute_soccer_ball_positions(self, env_ids: torch.Tensor):
    """Vectorized ball placement — batched over all env_ids.

    When cfg.fixed_ball_pos is set (eval mode), skips motion-based computation
    and places the ball at the fixed world position.
    """
    if self.cfg.fixed_ball_pos is not None:
      fp = torch.tensor(self.cfg.fixed_ball_pos, device=self.device)
      self.soccer_ball_pos[env_ids] = fp - self._env.scene.env_origins[env_ids]
      return

    arc_limit = float(self._target_arc_angle)
    base_height = float(self._target_height)
    n = len(env_ids)

    m_idx = self.motion_idx[env_ids]                        # (n,)
    m_len = self.motion_length[env_ids].clamp(min=1)        # (n,)
    last_frame = (m_len - 1).clamp(min=0)                   # (n,)
    anchor_idx = self.motion_anchor_body_index

    # Batched first/last anchor positions from full 30-body motion data.
    first_anchor = self.motion._body_pos_w[m_idx, torch.zeros(n, dtype=torch.long, device=self.device), anchor_idx]  # (n, 3)
    last_anchor = self.motion._body_pos_w[m_idx, last_frame, anchor_idx]  # (n, 3)

    radius_vec = last_anchor[:, :2] - first_anchor[:, :2]   # (n, 2)
    radius_sq = torch.sum(radius_vec * radius_vec, dim=-1)   # (n,)
    radius = torch.sqrt(radius_sq.clamp(min=1e-12))          # (n,)
    radius = radius + self.curve_radius_offset[env_ids]
    radius = torch.clamp(radius, min=0.0)

    # Direction: either base direction (normalized radius_vec) or default (1, 0).
    valid = radius_sq > 1e-12                                # (n,)
    base_dir = torch.where(
      valid.unsqueeze(-1),
      radius_vec / torch.sqrt(radius_sq.clamp(min=1e-12)).unsqueeze(-1),
      torch.tensor([1.0, 0.0], device=self.device).expand(n, -1),
    )                                                        # (n, 2)

    if arc_limit > 0.0:
      base_angle = torch.atan2(radius_vec[:, 1], radius_vec[:, 0])  # (n,)
      angle_offset = sample_uniform(-arc_limit, arc_limit, (n,), device=self.device)  # (n,)
      new_angle = torch.where(valid, base_angle + angle_offset, base_angle)
      direction = torch.stack((torch.cos(new_angle), torch.sin(new_angle)), dim=-1)  # (n, 2)
    else:
      direction = base_dir

    target_xy = first_anchor[:, :2] + radius.unsqueeze(-1) * direction  # (n, 2)
    self.soccer_ball_pos[env_ids, :2] = target_xy
    self.soccer_ball_pos[env_ids, 2] = base_height

  def _update_target_points(self, env_ids: torch.Tensor):
    self.target_point_pos[env_ids] = self.soccer_ball_pos[env_ids]
    self.initial_target_point_pos[env_ids] = self.soccer_ball_pos[env_ids].clone()

  def _update_target_points_from_sim(self):
    if self.soccer_ball is None:
      return
    if self.cfg.fixed_ball_pos is not None:
      return  # Eval mode: target stays at fixed penalty spot.
    env_origins = self._env.scene.env_origins
    ball_world = self.soccer_ball.data.root_link_pos_w
    self.soccer_ball_pos = ball_world - env_origins
    self.target_point_pos = self.soccer_ball_pos.clone()

  def _update_destination_points(self, env_ids: torch.Tensor):
    n = len(env_ids)
    rand_x = (torch.rand(n, device=self.device) - 0.5) * self.destination_length
    rand_y = (torch.rand(n, device=self.device) - 0.5) * self.destination_width
    dest = self.destination_center.expand(n, -1) + torch.stack(
      [rand_x, rand_y, torch.zeros_like(rand_x)], dim=1
    )
    self.target_destination_pos[env_ids] = dest

  def _update_soccer_ball(self, env_ids: torch.Tensor):
    if self.soccer_ball is None or env_ids.numel() == 0:
      return
    ball_pos = self.soccer_ball_pos[env_ids] + self._env.scene.env_origins[env_ids]
    ball_quat = ball_pos.new_zeros((env_ids.numel(), 4))
    ball_quat[:, 0] = 1.0

    if self._enable_ball_init_vel:
      ball_lin_vel = sample_uniform(
        self._ball_init_vel_ranges[:, 0], self._ball_init_vel_ranges[:, 1],
        (env_ids.numel(), 3), device=self.device,
      )
    else:
      ball_lin_vel = ball_pos.new_zeros((env_ids.numel(), 3))
    ball_ang_vel = ball_pos.new_zeros((env_ids.numel(), 3))

    state = torch.cat([ball_pos, ball_quat, ball_lin_vel, ball_ang_vel], dim=-1)
    self.soccer_ball.write_root_state_to_sim(state, env_ids=env_ids)


  def _debug_vis_impl(self, visualizer) -> None:
    """Render ball target (green) and destination (red) spheres."""
    from mjlab.viewer.debug_visualizer import DebugVisualizer
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    ball_radius = 0.10
    for batch in env_indices:
      # Green sphere at ball target position.
      bp = (self.target_point_pos[batch] + self._env.scene.env_origins[batch]).cpu().numpy()
      visualizer.add_sphere(bp, ball_radius, (0.0, 1.0, 0.0, 1.0), label=f"ball_target_{batch}")
      # Red sphere at destination.
      dp = (self.target_destination_pos[batch] + self._env.scene.env_origins[batch]).cpu().numpy()
      visualizer.add_sphere(dp, ball_radius, (1.0, 0.0, 0.0, 1.0), label=f"destination_{batch}")


@dataclass
class CurveOffsetCfg:
  """Ball placement randomization for Stage II perception-guided kicking.

  The ball is offset from the motion's nominal terminal position along a
  curved arc to promote positional generalization.
  """

  radius: tuple[float, float] = (0.0, 0.0)
  """Range for random radius offset (min, max), in meters."""

  arc_angle: float = 0.0
  """Semi-angle of the arc perturbation, in radians."""

  height: float = 0.0
  """Ball height offset, in meters."""


@dataclass(kw_only=True)
class MultiMotionSoccerCommandCfg(CommandTermCfg):
  """Configuration for MultiMotionSoccerCommand."""

  motion_dir: str = ""
  motion_glob: str = "*.npz"
  """Glob pattern for filtering motion files (e.g. \"soccer-standard-*\" for standard motions only)."""
  anchor_body_name: str = "torso_link"
  body_names: tuple[str, ...] = ()
  entity_name: str = "robot"
  ball_entity_name: str = "ball"

  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.1, 0.1)

  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  adaptive_kernel_size: int = 3
  adaptive_lambda: float = 0.1
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.4

  curve_offset_range: CurveOffsetCfg | None = None
  destination_center: tuple[float, float, float] = (0.0, -5.0, 0.11)
  destination_length: float = 1.0
  destination_width: float = 0.5

  enable_soccer_ball_init_vel: bool = False
  soccer_ball_init_lin_vel_range: dict[str, tuple[float, float]] | None = None

  fixed_ball_pos: tuple[float, float, float] | None = None
  """If set, ball is always placed at this fixed world position (eval mode)."""

  motion_origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """XYZ offset added to the motion root position (eval: shift from origin to world)."""

  motion_yaw_offset: float = 0.0
  """Yaw rotation added to the motion root orientation (eval: rotate to face goal)."""

  def build(self, env: ManagerBasedRlEnv) -> MultiMotionSoccerCommand:
    return MultiMotionSoccerCommand(self, env)
