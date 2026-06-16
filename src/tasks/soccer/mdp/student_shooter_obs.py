"""Motion-free observation builder for shooter student distillation.

The Stage II teacher observes motion-reference terms.  The student policy must
not depend on those terms, so dataset collection, BC training, PPO fine-tuning,
and API deployment should share this observation definition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from .shooter_commands import MultiMotionSoccerCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_elapsed_phase(env: ManagerBasedRlEnv, command: MultiMotionSoccerCommand) -> torch.Tensor:
  episode_len_s = float(getattr(env.cfg, "episode_length_s", 10.0))
  denom = max(episode_len_s, 1e-6)
  elapsed = command.time_steps.float().unsqueeze(-1) * float(env.step_dt)
  return torch.clamp(elapsed / denom, 0.0, 1.0)


def student_shooter_obs(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  ball_entity_name: str = "ball",
) -> torch.Tensor:
  """Build the motion-free shooter student observation.

  Returns a tensor with terms:
    projected_gravity(3), base_ang_vel_b(3), joint_pos_rel(29),
    joint_vel_rel(29), last_action(29), ball_pos_b(3), ball_vel_b(3),
    destination_pos_b(3), ball_to_destination_b(3), elapsed_phase(1).

  Total dimension: 106 for the current 29-DoF G1 model.
  """
  command: MultiMotionSoccerCommand = env.command_manager.get_term(command_name)
  robot = command.robot
  ball = env.scene[ball_entity_name]

  robot_pos_w = command.robot_pelvis_pos_w
  robot_quat_w = command.robot_pelvis_quat_w
  env_origins = env.scene.env_origins

  projected_gravity = robot.data.projected_gravity_b
  base_ang_vel_b = quat_apply_inverse(robot_quat_w, robot.data.root_link_ang_vel_w)

  default_joint_pos = robot.data.default_joint_pos
  default_joint_vel = robot.data.default_joint_vel
  assert default_joint_pos is not None
  assert default_joint_vel is not None
  joint_pos_rel = robot.data.joint_pos - default_joint_pos
  joint_vel_rel = robot.data.joint_vel - default_joint_vel

  last_action = env.action_manager.action

  ball_pos_w = ball.data.root_link_pos_w
  ball_vel_w = ball.data.root_link_lin_vel_w
  ball_pos_b = quat_apply_inverse(robot_quat_w, ball_pos_w - robot_pos_w)
  ball_vel_b = quat_apply_inverse(robot_quat_w, ball_vel_w)

  destination_w = command.target_destination_pos + env_origins
  destination_pos_b = quat_apply_inverse(robot_quat_w, destination_w - robot_pos_w)
  ball_to_destination_b = quat_apply_inverse(robot_quat_w, destination_w - ball_pos_w)

  elapsed_phase = _get_elapsed_phase(env, command)

  return torch.cat(
    [
      projected_gravity,
      base_ang_vel_b,
      joint_pos_rel,
      joint_vel_rel,
      last_action,
      ball_pos_b,
      ball_vel_b,
      destination_pos_b,
      ball_to_destination_b,
      elapsed_phase,
    ],
    dim=-1,
  )
