"""Left/right mirror helpers for G1 goalkeeper actor observations/actions."""

from __future__ import annotations

import torch

JOINT_PERM = [
  6, 7, 8, 9, 10, 11,
  0, 1, 2, 3, 4, 5,
  12, 13, 14,
  22, 23, 24, 25, 26, 27, 28,
  15, 16, 17, 18, 19, 20, 21,
]
_FLIP = {1, 2, 5, 7, 8, 11, 12, 13, 16, 17, 19, 21, 23, 24, 26, 28}
JOINT_SIGN = [(-1.0 if i in _FLIP else 1.0) for i in range(29)]

_VEC_BALL = [1.0, -1.0, 1.0]
_VEC_ANGV = [-1.0, 1.0, -1.0]
_VEC_GRAV = [1.0, -1.0, 1.0]

_CACHE: dict[torch.device, tuple[torch.Tensor, ...]] = {}


def _tensors(device: torch.device) -> tuple[torch.Tensor, ...]:
  if device not in _CACHE:
    _CACHE[device] = (
      torch.tensor(JOINT_PERM, device=device),
      torch.tensor(JOINT_SIGN, device=device),
      torch.tensor(_VEC_BALL, device=device),
      torch.tensor(_VEC_ANGV, device=device),
      torch.tensor(_VEC_GRAV, device=device),
    )
  return _CACHE[device]


def mirror_action(action: torch.Tensor) -> torch.Tensor:
  perm, sign, *_ = _tensors(action.device)
  return action[..., perm] * sign


def mirror_obs(obs: torch.Tensor) -> torch.Tensor:
  """Mirror a term-major 960D goalkeeper actor observation."""
  perm, sign, ball_sign, angv_sign, grav_sign = _tensors(obs.device)
  prefix_shape = obs.shape[:-1]
  mirrored = obs.clone()

  def mirror_vec(lo: int, hi: int, vec_sign: torch.Tensor) -> None:
    mirrored[..., lo:hi] = (
      mirrored[..., lo:hi].reshape(*prefix_shape, 10, 3) * vec_sign
    ).reshape(*prefix_shape, hi - lo)

  def mirror_joints(lo: int, hi: int) -> None:
    mirrored[..., lo:hi] = (
      mirrored[..., lo:hi].reshape(*prefix_shape, 10, 29)[..., perm] * sign
    ).reshape(*prefix_shape, hi - lo)

  mirror_vec(0, 30, ball_sign)
  mirror_vec(30, 60, angv_sign)
  mirror_vec(60, 90, grav_sign)
  mirror_joints(90, 380)
  mirror_joints(380, 670)
  mirror_joints(670, 960)
  return mirrored
