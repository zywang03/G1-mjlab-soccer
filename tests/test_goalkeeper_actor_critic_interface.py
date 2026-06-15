import unittest

try:
    import contextlib
    import io

    import torch
except ModuleNotFoundError:  # pragma: no cover - local base env has no torch.
    torch = None


@unittest.skipIf(torch is None, "torch is required for model interface tests")
class GoalkeeperActorCriticInterfaceTest(unittest.TestCase):
    def test_actor_and_critic_match_new_rsl_rl_model_interface(self):
        from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic

        obs = {
            "actor": torch.zeros(2, 960),
            "critic": torch.zeros(2, 113),
        }
        obs_groups = {"actor": ["actor"], "critic": ["critic"]}

        actor = GoalkeeperActorCritic(obs, obs_groups, "actor", 29)
        critic = GoalkeeperActorCritic(obs, obs_groups, "critic", 1)

        self.assertEqual(actor.num_actor_obs, 960)
        self.assertEqual(actor.num_one_step_obs, 96)
        self.assertIn("distribution.std_param", actor.state_dict())
        self.assertNotIn("std", actor.state_dict())
        self.assertEqual(
            next(iter(actor.named_parameters()))[0],
            "distribution.std_param",
        )
        self.assertIsNone(actor.get_hidden_state())
        self.assertIsNone(critic.get_hidden_state())

        actions = actor(obs, stochastic_output=True)
        self.assertEqual(actions.shape, (2, 29))
        self.assertEqual(actor.output_mean.shape, (2, 29))
        self.assertEqual(actor.output_std.shape, (2, 29))
        self.assertEqual(actor.output_entropy.shape, (2,))
        self.assertEqual(actor.get_output_log_prob(actions).shape, (2,))

        old_params = tuple(param.detach().clone() for param in actor.output_distribution_params)
        actor(obs, stochastic_output=True)
        self.assertEqual(
            actor.get_kl_divergence(old_params, actor.output_distribution_params).shape,
            (2,),
        )

        values = critic(obs)
        self.assertEqual(values.shape, (2, 1))

    def test_old_std_checkpoint_key_loads_strictly(self):
        from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic

        obs = {
            "actor": torch.zeros(2, 960),
            "critic": torch.zeros(2, 113),
        }
        obs_groups = {"actor": ["actor"], "critic": ["critic"]}
        actor = GoalkeeperActorCritic(obs, obs_groups, "actor", 29)
        old_style_state = {
            key: value.detach().clone()
            for key, value in actor.state_dict().items()
            if key != "distribution.std_param"
        }
        old_style_state["std"] = torch.full((29,), 0.25)

        restored = GoalkeeperActorCritic(obs, obs_groups, "actor", 29)
        restored.load_state_dict(old_style_state, strict=True)

        self.assertTrue(
            torch.allclose(restored.distribution.std_param, old_style_state["std"])
        )

    def test_runner_rewritten_log_std_checkpoint_key_loads_strictly(self):
        from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic

        obs = {
            "actor": torch.zeros(2, 960),
            "critic": torch.zeros(2, 113),
        }
        obs_groups = {"actor": ["actor"], "critic": ["critic"]}
        actor = GoalkeeperActorCritic(obs, obs_groups, "actor", 29)
        log_std_state = {
            key: value.detach().clone()
            for key, value in actor.state_dict().items()
            if key != "distribution.std_param"
        }
        log_std_state["distribution.log_std_param"] = torch.full((29,), -1.0)

        restored = GoalkeeperActorCritic(obs, obs_groups, "actor", 29)
        restored.load_state_dict(log_std_state, strict=True)

        self.assertTrue(
            torch.allclose(
                restored.distribution.std_param,
                log_std_state["distribution.log_std_param"].exp(),
            )
        )

    def test_distribution_std_is_clamped_when_configured(self):
        from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic

        obs = {
            "actor": torch.zeros(2, 960),
            "critic": torch.zeros(2, 113),
        }
        obs_groups = {"actor": ["actor"], "critic": ["critic"]}
        actor = GoalkeeperActorCritic(
            obs,
            obs_groups,
            "actor",
            29,
            distribution_cfg={"init_std": 0.2, "max_std": 0.3},
        )
        with torch.no_grad():
            actor.distribution.std_param.fill_(1.7)

        actor(obs, stochastic_output=True)

        self.assertLessEqual(float(actor.output_std.max().detach()), 0.300001)

    def test_distribution_std_has_positive_floor_when_configured(self):
        from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic

        obs = {
            "actor": torch.zeros(2, 960),
            "critic": torch.zeros(2, 113),
        }
        obs_groups = {"actor": ["actor"], "critic": ["critic"]}
        actor = GoalkeeperActorCritic(
            obs,
            obs_groups,
            "actor",
            29,
            distribution_cfg={"init_std": 0.2, "min_std": 0.05, "max_std": 0.3},
        )
        with torch.no_grad():
            actor.distribution.std_param.fill_(-1.7)

        actor(obs, stochastic_output=True)

        self.assertGreaterEqual(float(actor.output_std.min().detach()), 0.049999)

    def test_estimator_supervision_loss_trains_ball_and_region_heads(self):
        from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic

        obs = {
            "actor": torch.zeros(2, 960),
            "critic": torch.zeros(2, 113),
        }
        obs_groups = {"actor": ["actor"], "critic": ["critic"]}
        actor = GoalkeeperActorCritic(obs, obs_groups, "actor", 29)
        actor_obs = torch.randn(2, 960)
        ball_target = torch.tensor(
            [
                [1.0, 0.2, 0.3, -2.0, 0.0, 0.5],
                [0.5, -0.4, 0.7, -3.0, 0.2, 0.1],
            ],
            dtype=torch.float32,
        )
        region_target = torch.tensor([0, 5], dtype=torch.long)

        losses = actor.compute_estimator_loss(
            actor_obs,
            ball_target=ball_target,
            region_target=region_target,
        )

        self.assertIn("ball", losses)
        self.assertIn("region", losses)
        self.assertIn("total", losses)
        self.assertGreater(float(losses["total"].detach()), 0.0)

    def test_estimated_region_actor_input_matches_privileged_region_scale(self):
        from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic

        obs = {
            "actor": torch.zeros(2, 960),
            "critic": torch.zeros(2, 113),
        }
        obs_groups = {"actor": ["actor"], "critic": ["critic"]}
        actor = GoalkeeperActorCritic(obs, obs_groups, "actor", 29)
        obs_history = torch.randn(2, 960)
        captured = {}

        class _CaptureActor(torch.nn.Module):
            def forward(self, x):
                captured["input"] = x.detach().clone()
                return torch.zeros(x.shape[0], 29, device=x.device)

        actor.actor = _CaptureActor()
        with torch.no_grad():
            actor.region_estimator[-1].weight.zero_()
            actor.region_estimator[-1].bias[:] = torch.tensor(
                [0.0, 1.0, 2.0, 6.0, 4.0, 5.0],
                dtype=actor.region_estimator[-1].bias.dtype,
            )

        actor(obs_history, stochastic_output=True)

        self.assertTrue(
            torch.allclose(
                captured["input"][:, -1],
                torch.full((2,), 1.0),
            )
        )

    def test_goalkeeper_ppo_update_logs_estimator_losses(self):
        try:
            from tensordict import TensorDict
            from rsl_rl.storage import RolloutStorage
        except ModuleNotFoundError as exc:  # pragma: no cover - optional local deps.
            self.skipTest(f"rsl_rl test dependencies are unavailable: {exc}")

        from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic
        from src.tasks.soccer.modules.goalkeeper_ppo import GoalkeeperPPO

        batch_size = 4
        obs = {
            "actor": torch.zeros(2, 960),
            "critic": torch.zeros(2, 113),
        }
        obs_groups = {"actor": ["actor"], "critic": ["critic"]}
        actor = GoalkeeperActorCritic(obs, obs_groups, "actor", 29)
        critic = GoalkeeperActorCritic(obs, obs_groups, "critic", 1)

        observations = TensorDict(
            {
                "actor": torch.randn(batch_size, 960),
                "critic": torch.zeros(batch_size, 113),
            },
            batch_size=[batch_size],
        )
        observations["critic"][:, 0:3] = torch.tensor([0.5, -0.2, 0.8])
        observations["critic"][:, 102:105] = torch.tensor([-0.4, 0.1, 0.2])
        observations["critic"][:, 99] = torch.tensor([0.0, 1.0 / 3.0, 4.0 / 3.0, 5.0 / 3.0])

        with torch.no_grad():
            actor(observations, stochastic_output=True)
            actions = actor.output_mean.detach().clone()
            old_log_prob = actor.get_output_log_prob(actions).detach().unsqueeze(-1)
            old_distribution_params = tuple(
                param.detach().clone() for param in actor.output_distribution_params
            )

        batch = RolloutStorage.Batch(
            observations=observations,
            actions=actions,
            values=torch.zeros(batch_size, 1),
            advantages=torch.ones(batch_size, 1),
            returns=torch.zeros(batch_size, 1),
            old_actions_log_prob=old_log_prob,
            old_distribution_params=old_distribution_params,
        )

        class _Storage:
            def __init__(self, update_batch):
                self.update_batch = update_batch
                self.cleared = False
                self.generator_args = None

            def mini_batch_generator(self, num_mini_batches, num_epochs):
                self.generator_args = (num_mini_batches, num_epochs)
                yield self.update_batch

            def clear(self):
                self.cleared = True

        storage = _Storage(batch)
        algo = GoalkeeperPPO(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=1,
            num_mini_batches=1,
            desired_kl=None,
            learning_rate=1e-4,
        )

        loss_dict = algo.update()

        self.assertEqual((1, 1), storage.generator_args)
        self.assertTrue(storage.cleared)
        self.assertIn("estimator_ball", loss_dict)
        self.assertIn("estimator_region", loss_dict)
        self.assertIn("estimator", loss_dict)
        self.assertGreater(loss_dict["estimator"], 0.0)

    def test_constructor_is_quiet_by_default(self):
        from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic

        obs = {
            "actor": torch.zeros(2, 960),
            "critic": torch.zeros(2, 113),
        }
        obs_groups = {"actor": ["actor"], "critic": ["critic"]}
        stream = io.StringIO()

        with contextlib.redirect_stdout(stream):
            GoalkeeperActorCritic(obs, obs_groups, "actor", 29)

        self.assertEqual("", stream.getvalue())

    def test_constructor_rejects_history_layout_that_reference_reorder_cannot_handle(self):
        from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic

        obs = {
            "actor": torch.zeros(2, 768),
            "critic": torch.zeros(2, 113),
        }
        obs_groups = {"actor": ["actor"], "critic": ["critic"]}

        with self.assertRaisesRegex(ValueError, "goalkeeper actor history layout"):
            GoalkeeperActorCritic(
                obs,
                obs_groups,
                "actor",
                29,
                num_actor_obs=768,
                actor_history_length=8,
            )


if __name__ == "__main__":
    unittest.main()
