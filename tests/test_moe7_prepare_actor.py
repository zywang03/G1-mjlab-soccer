"""Tests for full-MoE7 prepare-only training actor."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict


class MoE7PrepareActorTest(unittest.TestCase):
  def _actor(self):
    from src.tasks.soccer.modules.moe7_prepare_actor import MoE7PrepareGoalkeeperActor

    obs = TensorDict({
      "actor": torch.zeros(2, 1630),
      "critic": torch.zeros(2, 180),
    }, batch_size=[2])
    return MoE7PrepareGoalkeeperActor(obs, {"actor": ["actor"]}, "actor", 29)

  @staticmethod
  def _zero_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
      param.data.zero_()

  def test_only_prepare_expert_is_trainable(self):
    actor = self._actor()

    frozen_names = [
      name
      for name, param in actor.named_parameters()
      if not param.requires_grad and (name.startswith("sr_experts.") or name.startswith("gate."))
    ]
    trainable_names = [name for name, param in actor.named_parameters() if param.requires_grad]

    self.assertTrue(frozen_names)
    self.assertTrue(all(name.startswith("idle_expert.") for name in trainable_names))
    self.assertTrue(any(name.startswith("idle_expert.actor_residual.") for name in trainable_names))
    self.assertNotIn("idle_expert.std", trainable_names)

  def test_prepare_std_is_clamped_and_frozen(self):
    actor = self._actor()
    with torch.no_grad():
      actor.idle_expert.std.fill_(6.0)

    obs = {"actor": torch.zeros(2, 1630)}
    actor(obs)

    self.assertFalse(actor.idle_expert.std.requires_grad)
    self.assertTrue(torch.allclose(actor.output_std, torch.full_like(actor.output_std, 0.15)))

  def test_export_clamps_prepare_std_metadata(self):
    actor = self._actor()
    with torch.no_grad():
      actor.idle_expert.std.fill_(6.0)

    bundle = actor.export_moe_bundle()

    self.assertTrue(bundle["freeze_idle_std"])
    self.assertEqual(bundle["idle_std_min"], 0.15)
    self.assertEqual(bundle["idle_std_max"], 0.15)
    self.assertTrue(torch.allclose(bundle["idle"]["actor_state_dict"]["std"], torch.full((29,), 0.15)))

  def test_load_moe_bundle_keeps_configured_prepare_std(self):
    actor = self._actor()
    bundle = actor.export_moe_bundle()
    bundle["idle_std_min"] = 0.02
    bundle["idle_std_max"] = 0.2
    with torch.no_grad():
      bundle["idle"]["actor_state_dict"]["std"].fill_(6.0)

    loaded = self._actor()
    loaded.load_moe_bundle(bundle)

    self.assertEqual(loaded.idle_std_min, 0.15)
    self.assertEqual(loaded.idle_std_max, 0.15)
    self.assertTrue(torch.allclose(loaded.idle_expert.std, torch.full((29,), 0.15)))

  def test_static_ball_routes_through_prepare_expert(self):
    actor = self._actor()
    self._zero_module(actor)
    with torch.no_grad():
      for expert in actor.sr_experts:
        expert.mlp[-1].bias.fill_(1.0)
      actor.idle_expert.actor[-1].bias.fill_(7.0)

    obs = {"actor": torch.zeros(2, 1630)}
    action = actor(obs)

    self.assertTrue(torch.allclose(action, torch.full((2, 29), 7.0)))

  def test_incoming_ball_routes_through_region_expert(self):
    actor = self._actor()
    self._zero_module(actor)
    with torch.no_grad():
      for expert in actor.sr_experts:
        expert.mlp[-1].bias.fill_(1.0)
      actor.idle_expert.actor[-1].bias.fill_(7.0)

    obs = {"actor": torch.zeros(2, 1630)}
    # Term-major ball history: first 10 x/y/z positions are ball_pos_local.
    # Oldest -> newest, so this gives an incoming ball with vx < 0.
    obs["actor"][:, 0:30] = torch.tensor([
      1.0, 0.0, 0.2,
      0.9, 0.0, 0.2,
      0.8, 0.0, 0.2,
      0.7, 0.0, 0.2,
      0.6, 0.0, 0.2,
      0.5, 0.0, 0.2,
      0.4, 0.0, 0.2,
      0.3, 0.0, 0.2,
      0.2, 0.0, 0.2,
      0.1, 0.0, 0.2,
    ])

    action = actor(obs)

    self.assertTrue(torch.allclose(action, torch.full((2, 29), 1.0)))

  def test_load_moe_bundle_uses_ballistic_region_experts_when_present(self):
    actor = self._actor()
    created = []

    class FakeBallisticExpert(torch.nn.Module):
      def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        created.append(self)
        self.distribution = type("Distribution", (), {"std": torch.ones(29)})()

      def load_state_dict(self, state_dict, strict=True):
        del strict
        self.loaded_keys = sorted(state_dict.keys())
        return None

      def forward(self, obs):
        return torch.zeros(obs["actor"].shape[0], 29)

    bundle = actor.export_moe_bundle()
    bundle["sr"][0] = {
      "actor_state_dict": {
        "_ballistic_marker": torch.ones(1),
        "base.mlp.0.weight": torch.zeros(1),
        "residual.0.weight": torch.zeros(1),
      },
      "ballistic_residual": {
        "base_hidden": (1024, 512, 256),
        "residual_scale": 0.25,
      },
    }

    with patch(
      "src.tasks.soccer.modules.moe7_prepare_actor.GoalkeeperBallisticResidual",
      FakeBallisticExpert,
    ):
      actor.load_moe_bundle(bundle)

    self.assertIs(actor.sr_experts[0], created[0])
    self.assertIn("base.mlp.0.weight", actor.sr_experts[0].loaded_keys)
    self.assertIn("residual.0.weight", actor.sr_experts[0].loaded_keys)


if __name__ == "__main__":
  unittest.main()
