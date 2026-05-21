"""Reward functions for the soccer task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def is_terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.termination_manager.terminated.float()
