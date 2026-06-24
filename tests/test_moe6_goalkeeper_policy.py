"""Tests for the shared MoE6 goalkeeper policy adapter."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch


class _FakeExpert:
  def __init__(self, value: float):
    self.value = value
    self.calls = 0
    self.reset_calls = 0
    self.last_actor_dim = None

  def __call__(self, obs):
    self.calls += 1
    self.last_actor_dim = obs["actor"].shape[-1]
    return torch.full((obs["actor"].shape[0], 3), self.value, dtype=torch.float32)

  def reset(self):
    self.reset_calls += 1


class _FixedGate(torch.nn.Module):
  def __init__(self, target: int, classes: int = 7):
    super().__init__()
    self.target = target
    self.classes = classes

  def forward(self, x):
    logits = torch.zeros(x.shape[0], self.classes, dtype=torch.float32)
    logits[:, self.target] = 1.0
    return logits


class MoE6GoalkeeperPolicyTest(unittest.TestCase):
  def _policy(self, n_envs: int = 2, include_idle: bool = False):
    from src.tasks.soccer.modules.moe6_goalkeeper_policy import MoE6GoalkeeperPolicy

    policy = object.__new__(MoE6GoalkeeperPolicy)
    policy.z_low = 0.85
    policy.z_up = 1.35
    policy.vz_low = -5.0
    policy.latch_hi = 5.0
    policy.dev = torch.device("cpu")
    policy.n_envs = n_envs
    policy.g = 9.81
    policy.latched = torch.full((n_envs,), -1, dtype=torch.long)
    policy.idle_expert_index = 6 if include_idle else None
    policy.idle_speed_threshold = 0.5
    policy.idle_incoming_vx_threshold = -0.5
    policy.learned_gate = None
    policy.experts = [_FakeExpert(float(i + 1)) for i in range(6 + int(include_idle))]
    policy.idle_env = None
    return policy

  def test_raw_state_gate_supports_batched_ball_state(self):
    policy = self._policy(n_envs=2)
    obs = {"actor": torch.zeros(2, 960)}
    raw_state = {
      "ball": {
        "pos": [[3.0, 0.0, 0.1], [0.3, -0.2, 1.0]],
        "vel": [[0.0, 0.0, 0.0], [-4.0, 0.0, 0.0]],
      },
    }

    action = policy(obs, raw_state)

    self.assertEqual(tuple(action.shape), (2, 3))
    self.assertTrue(torch.allclose(action[0], torch.ones(3)))
    self.assertTrue(torch.allclose(action[1], torch.full((3,), 2.0)))
    self.assertEqual(policy.latched.tolist(), [-1, 1])
    self.assertEqual(sum(expert.calls for expert in policy.experts), 6)

  def test_prepare_expert_handles_stationary_ball_when_present(self):
    policy = self._policy(n_envs=2, include_idle=True)
    obs = {"actor": torch.zeros(2, 960)}
    raw_state = {
      "ball": {
        "pos": [[3.0, 0.0, 0.1], [0.3, -0.2, 1.0]],
        "vel": [[0.0, 0.0, 0.0], [-4.0, 0.0, 0.0]],
      },
    }

    action = policy(obs, raw_state)

    self.assertTrue(torch.allclose(action[0], torch.full((3,), 7.0)))
    self.assertTrue(torch.allclose(action[1], torch.full((3,), 2.0)))
    self.assertEqual(policy.latched.tolist(), [-1, 1])
    self.assertEqual(sum(expert.calls for expert in policy.experts), 7)

  def test_only_prepare_expert_receives_opponent_observation(self):
    policy = self._policy(n_envs=2, include_idle=True)

    class FakeIdleEnv:
      def get_observations(self):
        return {"actor": torch.zeros(2, 1111)}

    policy.idle_env = FakeIdleEnv()
    obs = {"actor": torch.zeros(2, 960)}
    raw_state = {
      "ball": {
        "pos": [[3.0, 0.0, 0.1], [3.0, 0.0, 0.1]],
        "vel": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
      },
    }

    action = policy(obs, raw_state)

    self.assertTrue(torch.allclose(action, torch.full((2, 3), 7.0)))
    self.assertEqual([expert.last_actor_dim for expert in policy.experts[:6]], [960] * 6)
    self.assertEqual(policy.experts[6].last_actor_dim, 1111)

  def test_learned_gate_routes_incoming_ball_when_available(self):
    policy = self._policy(n_envs=1, include_idle=True)
    policy.learned_gate = _FixedGate(target=3)
    policy.gate_mean = torch.zeros(6)
    policy.gate_std = torch.ones(6)
    obs = {"actor": torch.zeros(1, 960)}
    raw_state = {
      "ball": {"pos": [[0.3, -0.2, 1.0]], "vel": [[-4.0, 0.0, 0.0]]},
    }

    action = policy(obs, raw_state)

    self.assertTrue(torch.allclose(action[0], torch.full((3,), 4.0)))
    self.assertEqual(policy.latched.tolist(), [3])

  def test_reset_clears_latch_and_resets_experts(self):
    policy = self._policy(n_envs=2, include_idle=True)
    policy.latched = torch.tensor([2, 4], dtype=torch.long)

    policy.reset()

    self.assertEqual(policy.latched.tolist(), [-1, -1])
    self.assertEqual([expert.reset_calls for expert in policy.experts], [1] * 7)

  def test_load_expert_uses_goalkeeper_runner_for_actor_critic_checkpoint(self):
    from src.tasks.soccer.modules.moe6_goalkeeper_policy import _load_expert_policy

    actor_ckpt = {"actor_state_dict": {"history_encoder.0.weight": torch.zeros(1)}}
    env = object()
    fake_runner = MagicMock()
    fake_runner.get_inference_policy.return_value = "policy"
    with patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.GoalkeeperRunner", return_value=fake_runner) as gk_runner, \
         patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.MjlabOnPolicyRunner") as mlp_runner:
      policy = _load_expert_policy(actor_ckpt, env, "cpu")

    self.assertEqual(policy, "policy")
    gk_runner.assert_called_once()
    mlp_runner.assert_not_called()
    fake_runner.load.assert_called_once()

  def test_load_expert_uses_adversarial_runner_for_residual_checkpoint(self):
    from src.tasks.soccer.modules.moe6_goalkeeper_policy import _load_expert_policy

    actor_ckpt = {"actor_state_dict": {"actor_residual.0.weight": torch.zeros(1)}}
    env = object()
    fake_runner = MagicMock()
    fake_runner.get_inference_policy.return_value = "policy"
    with patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.AdversarialGoalkeeperRunner", return_value=fake_runner) as adv_runner, \
         patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.GoalkeeperRunner") as gk_runner, \
         patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.MjlabOnPolicyRunner") as mlp_runner:
      policy = _load_expert_policy(actor_ckpt, env, "cpu")

    self.assertEqual(policy, "policy")
    adv_runner.assert_called_once()
    gk_runner.assert_not_called()
    mlp_runner.assert_not_called()
    fake_runner.load.assert_called_once()

  def test_load_expert_uses_ballistic_residual_runner_for_ballistic_checkpoint(self):
    from src.tasks.soccer.modules.moe6_goalkeeper_policy import _load_expert_policy

    actor_ckpt = {
      "actor_state_dict": {
        "_ballistic_marker": torch.ones(1),
        "base.mlp.0.weight": torch.zeros(1),
        "residual.0.weight": torch.zeros(1),
      },
      "ballistic_residual": {
        "base": "base.pt",
        "base_hidden": (1024, 512, 256),
        "residual_scale": 0.3,
      },
    }
    env = object()
    fake_runner = MagicMock()
    fake_runner.get_inference_policy.return_value = "policy"
    with patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.MjlabOnPolicyRunner", return_value=fake_runner) as mlp_runner, \
         patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.GoalkeeperRunner") as gk_runner, \
         patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.AdversarialGoalkeeperRunner") as adv_runner:
      policy = _load_expert_policy(actor_ckpt, env, "cpu")

    self.assertEqual(policy, "policy")
    mlp_runner.assert_called_once()
    cfg = mlp_runner.call_args.args[1]
    self.assertEqual(
      cfg["actor"]["class_name"],
      "src.tasks.soccer.modules.gk_ballistic_residual.GoalkeeperBallisticResidual",
    )
    gk_runner.assert_not_called()
    adv_runner.assert_not_called()
    fake_runner.load.assert_called_once()

  def test_load_expert_does_not_require_external_base_when_ballistic_checkpoint_embeds_base(self):
    from src.tasks.soccer.modules.moe6_goalkeeper_policy import _load_expert_policy
    import src.tasks.soccer.modules.gk_ballistic_residual as gkbr

    actor_ckpt = {
      "actor_state_dict": {
        "_ballistic_marker": torch.ones(1),
        "base.mlp.0.weight": torch.zeros(1),
        "residual.0.weight": torch.zeros(1),
      },
      "ballistic_residual": {
        "base": "missing_external_base.pt",
        "base_hidden": (1024, 512, 256),
        "residual_scale": 0.3,
      },
    }
    fake_runner = MagicMock()
    fake_runner.get_inference_policy.return_value = "policy"
    with patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.MjlabOnPolicyRunner", return_value=fake_runner):
      _load_expert_policy(actor_ckpt, object(), "cpu")

    self.assertIsNone(gkbr.BASE_CKPT)

  def test_load_expert_uses_native_runner_for_mlp_checkpoint(self):
    from src.tasks.soccer.modules.moe6_goalkeeper_policy import _load_expert_policy

    actor_ckpt = {"actor_state_dict": {"mlp.0.weight": torch.zeros(1)}}
    env = object()
    fake_runner = MagicMock()
    fake_runner.get_inference_policy.return_value = "policy"
    with patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.MjlabOnPolicyRunner", return_value=fake_runner) as mlp_runner, \
         patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.GoalkeeperRunner") as gk_runner:
      policy = _load_expert_policy(actor_ckpt, env, "cpu")

    self.assertEqual(policy, "policy")
    mlp_runner.assert_called_once()
    gk_runner.assert_not_called()
    fake_runner.load.assert_called_once()

  def test_load_expert_wraps_subcheckpoint_with_infos_for_mjlab_runner(self):
    from src.tasks.soccer.modules.moe6_goalkeeper_policy import _load_expert_policy

    seen = {}

    class FakeRunner:
      def __init__(self, env, cfg, device):
        del env, cfg, device

      def load(self, path, load_cfg=None):
        del load_cfg
        seen.update(torch.load(path, map_location="cpu", weights_only=False))
        seen["infos"]

      def get_inference_policy(self, device="cpu"):
        del device
        return "policy"

    actor_ckpt = {"actor_state_dict": {"mlp.0.weight": torch.zeros(1)}}
    with patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.MjlabOnPolicyRunner", FakeRunner):
      policy = _load_expert_policy(actor_ckpt, object(), "cpu")

    self.assertEqual(policy, "policy")
    self.assertIn("infos", seen)
    self.assertEqual(seen["iter"], 0)


if __name__ == "__main__":
  unittest.main()
