"""Motion-free command/reset generator for shooter student PPO.

Unlike ``MultiMotionSoccerCommand``, this command does not load or expose
motion-reference trajectories. It only resets the robot, samples the ball and
goal-plane destination, and tracks the live ball position for rewards/obs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

from .shooter_kick_detection import KickContactTracker

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def _range_tensor(ranges: dict[str, tuple[float, float]], keys: tuple[str, ...], device: str) -> torch.Tensor:
  return torch.tensor([ranges.get(key, (0.0, 0.0)) for key in keys], device=device)


class StudentShooterCommand(CommandTerm):
  """Command term for motion-free shooter student PPO."""

  cfg: "StudentShooterCommandCfg"

  def __init__(self, cfg: "StudentShooterCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.soccer_ball: Entity = env.scene[cfg.ball_entity_name]

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.target_point_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.initial_target_point_pos = torch.zeros_like(self.target_point_pos)
    self.target_destination_pos = torch.zeros_like(self.target_point_pos)
    self.kick_contact_tracker = KickContactTracker(env, "_student_shooter")

    self.metrics["ball_to_destination_xy"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.target_point_pos, self.target_destination_pos], dim=-1)

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_pelvis_pos_w(self) -> torch.Tensor:
    pelvis_idx = self.robot.body_names.index("pelvis")
    return self.robot.data.body_link_pos_w[:, pelvis_idx]

  @property
  def robot_pelvis_quat_w(self) -> torch.Tensor:
    pelvis_idx = self.robot.body_names.index("pelvis")
    return self.robot.data.body_link_quat_w[:, pelvis_idx]

  def _update_metrics(self) -> None:
    diff = self.target_destination_pos[:, :2] - self.target_point_pos[:, :2]
    self.metrics["ball_to_destination_xy"] = torch.linalg.vector_norm(diff, dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    self.time_steps[env_ids] = 0
    self._reset_robot(env_ids)
    self._sample_ball(env_ids)
    self._sample_destination(env_ids)

    flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    flags[env_ids] = True
    self.kick_contact_tracker._handle_resample(flags)

  def _update_command(self) -> None:
    self.kick_contact_tracker.begin_step()
    self.time_steps += 1
    self._update_target_points_from_sim()

    valid_kick_awarded = self.kick_contact_tracker.get_student_valid_kick_awarded()
    no_valid_kick = ~valid_kick_awarded
    if torch.any(no_valid_kick):
      self.initial_target_point_pos[no_valid_kick] = self.target_point_pos[no_valid_kick]

  def _reset_robot(self, env_ids: torch.Tensor) -> None:
    n = env_ids.numel()
    env_origins = self._env.scene.env_origins[env_ids]
    root_pos = torch.tensor(self.cfg.robot_pos, device=self.device).expand(n, -1).clone()

    pose_ranges = _range_tensor(self.cfg.pose_range, ("x", "y", "z", "roll", "pitch", "yaw"), self.device)
    pose_rand = sample_uniform(pose_ranges[:, 0], pose_ranges[:, 1], (n, 6), device=self.device)
    root_pos += pose_rand[:, :3]
    root_pos += env_origins

    yaw = torch.full((n,), self.cfg.robot_yaw, device=self.device) + pose_rand[:, 5]
    root_quat = quat_from_euler_xyz(pose_rand[:, 3], pose_rand[:, 4], yaw)

    vel_ranges = _range_tensor(self.cfg.velocity_range, ("x", "y", "z", "roll", "pitch", "yaw"), self.device)
    vel_rand = sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (n, 6), device=self.device)
    root_state = torch.cat([root_pos, root_quat, vel_rand[:, :3], vel_rand[:, 3:]], dim=-1)

    default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
    joint_pos = default_joint_pos + sample_uniform(
      self.cfg.joint_position_range[0],
      self.cfg.joint_position_range[1],
      default_joint_pos.shape,
      device=self.device,
    )
    limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clamp(joint_pos, limits[:, :, 0], limits[:, :, 1])

    self.robot.write_joint_state_to_sim(joint_pos, default_joint_vel, env_ids=env_ids)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.clear_state(env_ids=env_ids)

    action_manager = getattr(self._env, "action_manager", None)
    if action_manager is not None:
      action_term = action_manager._terms.get("joint_pos")
      if action_term is not None and hasattr(action_term, "_offset"):
        action_term._offset[env_ids] = joint_pos

  def _sample_ball(self, env_ids: torch.Tensor) -> None:
    n = env_ids.numel()
    ball = torch.tensor(self.cfg.ball_pos, device=self.device).expand(n, -1).clone()
    ball[:, 0] += sample_uniform(self.cfg.ball_x_range[0], self.cfg.ball_x_range[1], (n,), device=self.device)
    ball[:, 1] += sample_uniform(self.cfg.ball_y_range[0], self.cfg.ball_y_range[1], (n,), device=self.device)
    ball[:, 2] += sample_uniform(self.cfg.ball_z_range[0], self.cfg.ball_z_range[1], (n,), device=self.device)
    self.target_point_pos[env_ids] = ball
    self.initial_target_point_pos[env_ids] = ball.clone()

    ball_pos_w = ball + self._env.scene.env_origins[env_ids]
    ball_quat = ball_pos_w.new_zeros((n, 4))
    ball_quat[:, 0] = 1.0
    zeros = ball_pos_w.new_zeros((n, 3))
    self.soccer_ball.write_root_state_to_sim(torch.cat([ball_pos_w, ball_quat, zeros, zeros], dim=-1), env_ids=env_ids)

  def _sample_destination(self, env_ids: torch.Tensor) -> None:
    n = env_ids.numel()
    dest = torch.zeros(n, 3, device=self.device)
    dest[:, 0] = sample_uniform(self.cfg.destination_x_range[0], self.cfg.destination_x_range[1], (n,), device=self.device)
    dest[:, 1] = self.cfg.destination_y
    dest[:, 2] = sample_uniform(self.cfg.destination_z_range[0], self.cfg.destination_z_range[1], (n,), device=self.device)
    self.target_destination_pos[env_ids] = dest

  def _update_target_points_from_sim(self) -> None:
    self.target_point_pos = self.soccer_ball.data.root_link_pos_w - self._env.scene.env_origins

  def _debug_vis_impl(self, visualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    for batch in env_indices:
      origin = self._env.scene.env_origins[batch]
      ball_pos = (self.target_point_pos[batch] + origin).detach().cpu().numpy()
      dest_pos = (self.target_destination_pos[batch] + origin).detach().cpu().numpy()
      visualizer.add_sphere(ball_pos, 0.10, (0.0, 1.0, 0.0, 1.0), label=f"student_ball_{batch}")
      visualizer.add_sphere(dest_pos, 0.08, (1.0, 0.0, 0.0, 1.0), label=f"student_dest_{batch}")


@dataclass(kw_only=True)
class StudentShooterCommandCfg(CommandTermCfg):
  """Configuration for motion-free shooter student command."""

  entity_name: str = "robot"
  ball_entity_name: str = "ball"
  anchor_body_name: str = "torso_link"

  robot_pos: tuple[float, float, float] = (0.0, 0.0, 0.8)
  robot_yaw: float = -1.5707963267948966
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.05, 0.05)

  ball_pos: tuple[float, float, float] = (0.0, -1.5, 0.11)
  ball_x_range: tuple[float, float] = (-0.2, 0.2)
  ball_y_range: tuple[float, float] = (-0.1, 0.1)
  ball_z_range: tuple[float, float] = (0.0, 0.0)

  destination_y: float = -5.0
  destination_x_range: tuple[float, float] = (-1.0, 1.0)
  destination_z_range: tuple[float, float] = (0.5, 1.3)

  def build(self, env: ManagerBasedRlEnv) -> StudentShooterCommand:
    return StudentShooterCommand(self, env)
