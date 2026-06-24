"""Closed-form ball trajectory helpers for goalkeeper routing."""

from __future__ import annotations

from dataclasses import dataclass

import torch


BALL_MODE_AIR = 0
BALL_MODE_GROUND = 1


@dataclass(frozen=True)
class BallPlaneIntersection:
  """Predicted ball intersection with a vertical x-plane."""

  yz: torch.Tensor
  time: torch.Tensor
  valid: torch.Tensor
  idle: torch.Tensor
  mode: torch.Tensor


def predict_ball_plane_intersection(
  ball_pos: torch.Tensor,
  ball_vel: torch.Tensor,
  plane_x: float = 0.0,
  gravity: float | None = None,
  air_gravity: float = 9.81,
  ground_height_threshold: float = 0.2,
  ground_vz_threshold: float = 0.25,
  idle_speed_threshold: float = 1e-3,
  eps: float = 1e-6,
) -> BallPlaneIntersection:
  """Predict where a ballistic ball crosses a vertical x-plane.

  Args:
    ball_pos: Ball position tensor with final dimension ``(x, y, z)``.
    ball_vel: Ball linear velocity tensor with final dimension ``(vx, vy, vz)``.
    plane_x: Target x-plane in the same frame as ``ball_pos``.
    gravity: Downward z acceleration magnitude. ``None`` enables automatic
      air/ground selection; explicit values override automatic selection.
    air_gravity: Gravity used for airborne balls when ``gravity`` is ``None``.
    ground_height_threshold: Maximum z height considered a ground ball.
    ground_vz_threshold: Maximum absolute vertical speed considered rolling.
    idle_speed_threshold: Speeds at or below this value are reported as idle.
    eps: Minimum absolute x velocity used to avoid division by zero.

  Returns:
    ``BallPlaneIntersection`` where ``yz`` is the predicted crossing point,
    ``time`` is seconds until crossing, ``valid`` means the ball reaches the
    plane in forward time, and ``idle`` marks a stationary or near-stationary ball.
    ``mode`` is ``BALL_MODE_AIR`` for air and ``BALL_MODE_GROUND`` for ground.
    Invalid predictions fall back to the current ``(y, z)`` and ``time=0``.
  """
  if ball_pos.shape[-1] != 3 or ball_vel.shape[-1] != 3:
    raise ValueError("ball_pos and ball_vel must have final dimension 3")
  ball_pos, ball_vel = torch.broadcast_tensors(ball_pos, ball_vel)

  speed = torch.linalg.vector_norm(ball_vel, dim=-1)
  idle = speed <= idle_speed_threshold
  vx = ball_vel[..., 0]
  dx = plane_x - ball_pos[..., 0]
  ground = (ball_pos[..., 2] <= ground_height_threshold) & (
    torch.abs(ball_vel[..., 2]) <= ground_vz_threshold
  )
  mode = torch.where(
    ground,
    torch.full_like(vx, BALL_MODE_GROUND, dtype=torch.long),
    torch.full_like(vx, BALL_MODE_AIR, dtype=torch.long),
  )
  if gravity is None:
    effective_gravity = torch.where(
      ground,
      torch.zeros_like(vx),
      torch.full_like(vx, air_gravity),
    )
  else:
    effective_gravity = torch.full_like(vx, gravity)

  has_x_motion = torch.abs(vx) > eps
  t_raw = dx / torch.where(has_x_motion, vx, torch.ones_like(vx))
  valid = has_x_motion & (t_raw >= 0.0) & (~idle)
  t = torch.where(valid, t_raw, torch.zeros_like(t_raw))

  y = ball_pos[..., 1] + ball_vel[..., 1] * t
  z = ball_pos[..., 2] + ball_vel[..., 2] * t - 0.5 * effective_gravity * t * t
  yz = torch.stack([y, z], dim=-1)
  return BallPlaneIntersection(yz=yz, time=t, valid=valid, idle=idle, mode=mode)
