"""Tests for zero-initialized adversarial policy extensions."""

from __future__ import annotations

import unittest

import torch
from tensordict import TensorDict


class AdversarialZeroInitModelsTest(unittest.TestCase):
  def test_adversarial_rnn_matches_base_when_extra_features_are_appended(self):
    from src.tasks.soccer.modules.adversarial_models import AdversarialRNNModel

    torch.manual_seed(7)
    obs_groups = {"actor": ["actor"]}
    base_obs = TensorDict({"actor": torch.randn(2, 160)}, batch_size=[2])
    extra = torch.randn(2, 67)
    adv_obs = TensorDict({
      "actor": torch.cat([base_obs["actor"], extra], dim=-1),
    }, batch_size=[2])
    adv_obs_changed = TensorDict({
      "actor": torch.cat([base_obs["actor"], extra + 10.0], dim=-1),
    }, batch_size=[2])

    base = AdversarialRNNModel(
      base_obs,
      obs_groups,
      "actor",
      output_dim=29,
      hidden_dims=(32,),
      rnn_hidden_dim=16,
      rnn_num_layers=1,
      base_obs_dim=160,
    )
    adv = AdversarialRNNModel(
      adv_obs,
      obs_groups,
      "actor",
      output_dim=29,
      hidden_dims=(32,),
      rnn_hidden_dim=16,
      rnn_num_layers=1,
      base_obs_dim=160,
    )
    adv.load_base_state_dict(base.state_dict())

    base.eval()
    adv.eval()
    with torch.inference_mode():
      self.assertEqual(adv.extra_obs_dim, 67)
      self.assertTrue(torch.allclose(base(base_obs), adv(adv_obs), atol=1e-6))
      base.reset()
      adv.reset()
      self.assertTrue(torch.allclose(base(base_obs), adv(adv_obs_changed), atol=1e-6))

    last_linear = next(module for module in reversed(list(adv.output_residual.modules())) if isinstance(module, torch.nn.Linear))
    with torch.no_grad():
      last_linear.weight.fill_(0.01)
    adv.reset()
    with torch.inference_mode():
      out_a = adv(adv_obs)
      adv.reset()
      out_b = adv(adv_obs_changed)
      self.assertFalse(torch.allclose(out_a, out_b))

  def test_adversarial_rnn_normalizes_only_base_observation_prefix(self):
    from src.tasks.soccer.modules.adversarial_models import AdversarialRNNModel

    torch.manual_seed(13)
    obs_groups = {"critic": ["critic"]}
    base_obs = TensorDict({"critic": torch.randn(4, 298)}, batch_size=[4])
    adv_obs = TensorDict({
      "critic": torch.cat([base_obs["critic"], torch.randn(4, 67)], dim=-1),
    }, batch_size=[4])

    base = AdversarialRNNModel(
      base_obs, obs_groups, "critic", output_dim=1, hidden_dims=(32,),
      rnn_hidden_dim=16, rnn_num_layers=1, base_obs_dim=298, obs_normalization=True,
    )
    adv = AdversarialRNNModel(
      adv_obs, obs_groups, "critic", output_dim=1, hidden_dims=(32,),
      rnn_hidden_dim=16, rnn_num_layers=1, base_obs_dim=298, obs_normalization=True,
    )
    adv.load_base_state_dict(base.state_dict())

    base.eval()
    adv.eval()
    with torch.inference_mode():
      self.assertEqual(tuple(adv.obs_normalizer.mean.shape), (298,))
      self.assertTrue(torch.allclose(base(base_obs), adv(adv_obs), atol=1e-6))

  def test_adversarial_rnn_residual_matches_recurrent_masked_batch_shape(self):
    from src.tasks.soccer.modules.adversarial_models import AdversarialRNNModel

    torch.manual_seed(19)
    obs_groups = {"actor": ["actor"]}
    init_obs = TensorDict({"actor": torch.randn(3, 160 + 67)}, batch_size=[3])
    batch_obs = TensorDict({"actor": torch.randn(5, 3, 160 + 67)}, batch_size=[5, 3])
    masks = torch.tensor([
      [True, True, False],
      [True, True, False],
      [True, True, False],
      [True, True, False],
      [True, True, False],
    ])
    hidden = (
      torch.zeros(1, 3, 16),
      torch.zeros(1, 3, 16),
    )
    model = AdversarialRNNModel(
      init_obs,
      obs_groups,
      "actor",
      output_dim=29,
      hidden_dims=(32,),
      rnn_hidden_dim=16,
      rnn_num_layers=1,
      base_obs_dim=160,
    )

    out = model(batch_obs, masks=masks, hidden_state=hidden)

    self.assertEqual(tuple(out.shape), (5, 2, 29))

  def test_adversarial_goalkeeper_matches_base_with_zero_residual(self):
    from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic
    from src.tasks.soccer.modules.adversarial_models import AdversarialGoalkeeperActorCritic

    torch.manual_seed(11)
    obs_groups = {"actor": ["actor"], "critic": ["critic"]}
    base_obs = TensorDict({
      "actor": torch.randn(3, 960),
      "critic": torch.randn(3, 113),
    }, batch_size=[3])
    adv_obs = TensorDict({
      "actor": torch.cat([base_obs["actor"], torch.randn(3, 67 * 10)], dim=-1),
      "critic": torch.cat([base_obs["critic"], torch.randn(3, 67)], dim=-1),
    }, batch_size=[3])

    base = GoalkeeperActorCritic(base_obs, obs_groups, "actor", 29)
    adv = AdversarialGoalkeeperActorCritic(adv_obs, obs_groups, "actor", 29)
    adv.load_base_state_dict(base.state_dict())

    base.eval()
    adv.eval()
    with torch.inference_mode():
      self.assertTrue(torch.allclose(base(base_obs), adv(adv_obs), atol=1e-6))
      self.assertTrue(torch.allclose(base.evaluate(base_obs), adv.evaluate(adv_obs), atol=1e-6))

  def test_adversarial_goalkeeper_exposes_rsl_rl_training_interface(self):
    from src.tasks.soccer.modules.adversarial_models import AdversarialGoalkeeperActorCritic

    torch.manual_seed(17)
    obs_groups = {"actor": ["actor"], "critic": ["critic"]}
    obs = TensorDict({
      "actor": torch.randn(4, 960 + 67 * 10),
      "critic": torch.randn(4, 113 + 67),
    }, batch_size=[4])
    actor = AdversarialGoalkeeperActorCritic(obs, obs_groups, "actor", 29)
    critic = AdversarialGoalkeeperActorCritic(obs, obs_groups, "critic", 29)

    self.assertIsNone(actor.get_hidden_state())
    self.assertIsNone(critic.get_hidden_state())
    actions = actor(obs, stochastic_output=True)
    values = critic(obs)

    self.assertEqual(tuple(actions.shape), (4, 29))
    self.assertEqual(tuple(values.shape), (4, 1))
    self.assertEqual(tuple(actor.get_output_log_prob(actions).shape), (4,))
    self.assertEqual(len(actor.output_distribution_params), 2)
    actor.detach_hidden_state()
    actor.update_normalization(obs)

  def test_goalkeeper_history_reorder_accepts_empty_minibatch(self):
    from src.tasks.soccer.modules.adversarial_models import AdversarialGoalkeeperActorCritic

    obs_groups = {"actor": ["actor"], "critic": ["critic"]}
    obs = TensorDict({
      "actor": torch.empty(0, 960 + 67 * 10),
      "critic": torch.empty(0, 113 + 67),
    }, batch_size=[0])
    actor = AdversarialGoalkeeperActorCritic(obs, obs_groups, "actor", 29)

    out = actor(obs, stochastic_output=True)

    self.assertEqual(tuple(out.shape), (0, 29))

  def test_goalkeeper_actor_does_not_clobber_torch_normal_api(self):
    from torch.distributions import Normal
    from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic

    obs_groups = {"actor": ["actor"], "critic": ["critic"]}
    obs = TensorDict({
      "actor": torch.randn(1, 960),
      "critic": torch.randn(1, 113),
    }, batch_size=[1])

    before = Normal.set_default_validate_args
    GoalkeeperActorCritic(obs, obs_groups, "actor", 29)

    self.assertIs(Normal.set_default_validate_args, before)
    self.assertTrue(callable(Normal.set_default_validate_args))


if __name__ == "__main__":
  unittest.main()
