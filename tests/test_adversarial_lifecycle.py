"""Lifecycle tests for adversarial frozen-opponent wrappers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from src.tasks.soccer.adversarial import (
  FrozenGoalkeeperPolicy,
  FrozenOpponentVecEnvWrapper,
  FrozenShooterPolicy,
  SingleRobotActionEnvView,
  goalkeeper_checkpoint_kind,
)


class _FakeVecEnv:
  num_envs = 2
  device = torch.device("cpu")
  max_episode_length = 32
  num_actions = 58
  cfg = object()
  render_mode = None
  observation_space = object()
  action_space = object()
  episode_length_buf = torch.zeros(2)

  def __init__(self):
    self.closed = False

  def close(self):
    self.closed = True

  def get_observations(self):
    return {"actor": torch.zeros(2, 3)}

  def reset(self):
    return self.get_observations(), {}

  def step(self, actions):
    return self.get_observations(), torch.zeros(2), torch.zeros(2, dtype=torch.bool), {}

  def seed(self, seed: int = -1) -> int:
    return seed

  @property
  def unwrapped(self):
    return self


class _FakeOpponentPolicy:
  def __init__(self):
    self.closed = False

  def close(self):
    self.closed = True


class _FakeActionTerm:
  def __init__(self, action_dim: int):
    self.action_dim = action_dim
    self._raw_actions = None


class _FakeActionManager:
  def __init__(self):
    self._terms = {
      "joint_pos": _FakeActionTerm(29),
      "opponent_joint_pos": _FakeActionTerm(29),
    }
    self._action = torch.zeros(2, 58)

  def get_term(self, name: str):
    return self._terms[name]


class _IndexHostileScene(dict):
  def __getitem__(self, key):
    if isinstance(key, int):
      raise KeyError(f"bad numeric lookup: {key}")
    return super().__getitem__(key)


class AdversarialLifecycleTest(unittest.TestCase):
  def test_wrapper_closes_frozen_opponent_policy_env(self):
    env = _FakeVecEnv()
    policy = _FakeOpponentPolicy()

    wrapper = FrozenOpponentVecEnvWrapper(env, policy, opponent_role="goalkeeper")
    wrapper.close()

    self.assertTrue(env.closed)
    self.assertTrue(policy.closed)

  def test_real_frozen_policy_classes_expose_close(self):
    self.assertTrue(callable(getattr(FrozenShooterPolicy, "close", None)))
    self.assertTrue(callable(getattr(FrozenGoalkeeperPolicy, "close", None)))

  def test_single_robot_action_env_view_exposes_active_action_dim_only(self):
    env = _FakeVecEnv()
    view = SingleRobotActionEnvView(env, action_dim=29)

    self.assertEqual(view.num_actions, 29)
    self.assertEqual(env.num_actions, 58)
    self.assertIs(view.unwrapped, env)
    self.assertIs(view.cfg, env.cfg)
    self.assertEqual(view.seed(7), 7)

  def test_goalkeeper_checkpoint_kind_detects_actor_state_formats(self):
    self.assertEqual(
      goalkeeper_checkpoint_kind({"actor_state_dict": {"actor_residual.0.weight": torch.zeros(1)}}),
      "adversarial_actor_critic",
    )
    self.assertEqual(
      goalkeeper_checkpoint_kind({"actor_state_dict": {"history_encoder.0.weight": torch.zeros(1)}}),
      "actor_critic",
    )
    self.assertEqual(
      goalkeeper_checkpoint_kind({"actor_state_dict": {"mlp.0.weight": torch.zeros(1)}}),
      "mlp",
    )

  def test_moe6_goalkeeper_checkpoint_uses_legacy_eval_task(self):
    from unittest.mock import patch

    from src.tasks.soccer.adversarial import resolve_goalkeeper_eval_task_id

    requested = "Unitree-G1-Goalkeeper-Adversarial"

    self.assertEqual(resolve_goalkeeper_eval_task_id("moe6", requested), "Eval-Goalkeeper")
    self.assertEqual(resolve_goalkeeper_eval_task_id("actor_critic", requested), "Eval-Goalkeeper")
    self.assertEqual(resolve_goalkeeper_eval_task_id("adversarial_actor_critic", "Eval-Goalkeeper"), requested)

    with patch("src.tasks.soccer.adversarial.torch.load", return_value={"sr": [{} for _ in range(6)]}), \
        patch("src.tasks.soccer.adversarial._make_eval_env") as make_env:
      make_env.side_effect = RuntimeError("stop after task resolution")
      with self.assertRaisesRegex(RuntimeError, "stop after task resolution"):
        FrozenGoalkeeperPolicy("fake_moe6.pt", device="cpu", num_envs=1, task_id=requested)
      self.assertEqual(make_env.call_args.args[0], "Eval-Goalkeeper")

  def test_moe7_with_adversarial_idle_uses_separate_idle_opponent_env(self):
    created = []

    class FakeMoE:
      def __init__(self, bundle, env, device, idle_env=None):
        del bundle, device
        self.env = env
        self.idle_env = idle_env
        created.append(self)

    bundle = {
      "sr": [{} for _ in range(6)],
      "idle": {"actor_state_dict": {"actor_residual.0.weight": torch.zeros(1)}},
    }
    base_env = object()
    idle_env = object()

    with patch("src.tasks.soccer.adversarial.torch.load", return_value=bundle), \
        patch("src.tasks.soccer.adversarial._make_eval_env", side_effect=[base_env, idle_env]) as make_env, \
        patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.MoE6GoalkeeperPolicy", FakeMoE):
      policy = FrozenGoalkeeperPolicy("fake_moe7.pt", device="cpu", num_envs=1, task_id="Unitree-G1-Goalkeeper-Adversarial")

    self.assertIs(policy.env, base_env)
    self.assertIs(created[0].env, base_env)
    self.assertIs(created[0].idle_env.env, idle_env)
    self.assertEqual(make_env.call_args_list[0].args[0], "Eval-Goalkeeper")
    self.assertEqual(make_env.call_args_list[1].args[0], "Unitree-G1-Goalkeeper-Adversarial")

  def test_set_eval_action_term_updates_only_named_action_slice(self):
    from src.tasks.soccer.adversarial import set_eval_action_term

    manager = _FakeActionManager()
    action = torch.ones(2, 29)

    set_eval_action_term(manager, "opponent_joint_pos", action)

    self.assertTrue(torch.equal(manager._action[:, :29], torch.zeros(2, 29)))
    self.assertTrue(torch.equal(manager._action[:, 29:], action))
    self.assertIs(manager.get_term("opponent_joint_pos")._raw_actions, action)

  def test_scene_has_entity_does_not_iterate_scene_by_index(self):
    from src.tasks.soccer.adversarial import scene_has_entity

    scene = _IndexHostileScene({"robot": object(), "ball": object()})

    self.assertTrue(scene_has_entity(scene, "robot"))
    self.assertFalse(scene_has_entity(scene, "opponent"))


if __name__ == "__main__":
  unittest.main()
