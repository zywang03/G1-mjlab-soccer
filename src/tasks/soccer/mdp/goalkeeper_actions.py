"""Goalkeeper action terms matching official pre-shot behavior."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg


@dataclass(kw_only=True)
class GoalkeeperJointPositionActionCfg(JointPositionActionCfg):
  """Joint-position action with official pre-shot pose holding."""

  hold_before_start: bool = True

  def build(self, env: ManagerBasedRlEnv) -> "GoalkeeperJointPositionAction":
    return GoalkeeperJointPositionAction(self, env)


class GoalkeeperJointPositionAction(JointPositionAction):
  cfg: GoalkeeperJointPositionActionCfg

  def apply_actions(self) -> None:
    target = self._processed_actions
    if self.cfg.hold_before_start:
      catchstep = getattr(self._env, "_gk_catchstep", None)
      startstep = getattr(self._env, "_gk_startstep", None)
      if (
        isinstance(catchstep, torch.Tensor)
        and isinstance(startstep, torch.Tensor)
        and catchstep.shape[0] == self.num_envs
        and startstep.shape[0] == self.num_envs
      ):
        hold = (catchstep.to(self.device) > startstep.to(self.device)).view(-1, 1)
        offset = self._offset if isinstance(self._offset, torch.Tensor) else torch.full_like(target, float(self._offset))
        target = torch.where(hold, offset, target)
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(target - encoder_bias, joint_ids=self._target_ids)
