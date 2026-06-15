import unittest
from pathlib import Path


try:
    import torch
    from tensordict import TensorDict
except ModuleNotFoundError as exc:  # pragma: no cover - base env lacks deps.
    torch = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"missing runtime dependency: {_IMPORT_ERROR}")
class GoalkeeperMotionPriorAmpTest(unittest.TestCase):
    def test_motion_prior_loads_six_region_specific_dof_position_transition_buffers(self):
        from src.tasks.soccer.modules.goalkeeper_motion_prior import (
            GoalkeeperMotionPrior,
        )

        prior = GoalkeeperMotionPrior(
            motion_dir="src/assets/soccer/motions/goalkeeper",
            device="cpu",
        )

        self.assertEqual(prior.num_regions, 6)
        self.assertEqual(prior.joint_dim, 21)
        self.assertEqual(prior.frame_dim, 21)
        self.assertEqual(prior.transition_dim, 42)
        self.assertEqual(
            prior.motion_names,
            (
                "lefthand",
                "righthand",
                "leftjump",
                "rightjump",
                "leftstep",
                "rightstep",
            ),
        )
        for region in range(6):
            samples = prior.sample_expert_transitions(
                torch.full((8,), region, dtype=torch.long)
            )
            self.assertEqual(samples.shape, (8, 42))

    def test_actor_critic_obs_are_converted_to_dof_position_amp_transitions(self):
        from src.tasks.soccer.modules.goalkeeper_motion_prior import (
            GoalkeeperMotionPrior,
            build_amp_state_from_observations,
        )

        prior = GoalkeeperMotionPrior(
            motion_dir="src/assets/soccer/motions/goalkeeper",
            device="cpu",
        )
        obs = TensorDict(
            {
                "critic": torch.zeros(3, 113),
                "actor": torch.zeros(3, 960),
            },
            batch_size=[3],
        )
        next_obs = obs.clone()
        obs["critic"][:, 9:38] = 0.1
        next_obs["critic"][:, 9:38] = 0.2

        amp_state = build_amp_state_from_observations(
            obs,
            next_obs,
            joint_indices=prior.full_joint_indices,
            default_joint_pos=prior.full_default_joint_pos,
        )

        self.assertEqual(amp_state.shape, (3, 42))
        self.assertTrue(
            torch.allclose(
                amp_state[:, :21],
                prior.full_default_joint_pos[prior.full_joint_indices].unsqueeze(0) + 0.1,
            )
        )
        self.assertTrue(
            torch.allclose(
                amp_state[:, 21:42],
                prior.full_default_joint_pos[prior.full_joint_indices].unsqueeze(0) + 0.2,
            )
        )

    def test_amp_discriminator_uses_official_lsgan_reward_formula(self):
        from src.tasks.soccer.modules.goalkeeper_amp import GoalkeeperAMP

        amp = GoalkeeperAMP(num_regions=6, state_dim=42, hidden_dims=(32, 16), device="cpu")
        state = torch.zeros(5, 42)
        regions = torch.tensor([0, 1, 2, 3, 4])

        with torch.no_grad():
            for discriminator in amp.discriminators:
                for param in discriminator.parameters():
                    param.zero_()
                discriminator[-1].bias.fill_(1.0)

        reward = amp.predict_reward(state, regions, num_samples=1, sigma=0.0)

        self.assertTrue(torch.allclose(reward, torch.ones(5, 1)))

    def test_amp_discriminator_loss_sums_active_region_means_like_official_code(self):
        from src.tasks.soccer.modules.goalkeeper_amp import GoalkeeperAMP

        amp = GoalkeeperAMP(num_regions=6, state_dim=2, hidden_dims=(), device="cpu")
        policy = torch.zeros(4, 2)
        expert = torch.zeros(4, 2)
        regions = torch.tensor([0, 0, 1, 1])
        with torch.no_grad():
            for discriminator in amp.discriminators:
                for param in discriminator.parameters():
                    param.zero_()
            amp.discriminators[0][-1].bias.fill_(1.0)
            amp.discriminators[1][-1].bias.fill_(0.0)

        losses = amp.compute_loss(policy, expert, regions)

        # Region 0: expert loss 0, policy loss 4. Region 1: expert loss 1,
        # policy loss 1. Per-region official-style sum => 6.
        self.assertAlmostEqual(float(losses["expert"].detach()), 1.0, places=5)
        self.assertAlmostEqual(float(losses["policy"].detach()), 5.0, places=5)
        self.assertAlmostEqual(float(losses["total"].detach()), 6.0, places=5)

    def test_goalkeeper_ppo_mixes_amp_reward_at_official_scale_and_logs_losses(self):
        try:
            from rsl_rl.storage import RolloutStorage
        except ModuleNotFoundError as exc:  # pragma: no cover - optional deps.
            self.skipTest(f"rsl_rl test dependencies are unavailable: {exc}")

        from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic
        from src.tasks.soccer.modules.goalkeeper_ppo import GoalkeeperPPO

        batch_size = 4
        obs = TensorDict(
            {
                "actor": torch.randn(batch_size, 960),
                "critic": torch.zeros(batch_size, 113),
            },
            batch_size=[batch_size],
        )
        next_obs = obs.clone()
        obs["critic"][:, 9:38] = 0.1
        next_obs["critic"][:, 9:38] = 0.2
        obs["critic"][:, 99] = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 5.0 / 3.0])
        next_obs["critic"][:, 99] = obs["critic"][:, 99]

        obs_groups = {"actor": ["actor"], "critic": ["critic"]}
        actor = GoalkeeperActorCritic(obs, obs_groups, "actor", 29)
        critic = GoalkeeperActorCritic(obs, obs_groups, "critic", 1)
        with torch.no_grad():
            actor(obs, stochastic_output=True)
            actions = actor.output_mean.detach().clone()
            old_log_prob = actor.get_output_log_prob(actions).detach().unsqueeze(-1)
            old_distribution_params = tuple(
                param.detach().clone() for param in actor.output_distribution_params
            )

        batch = RolloutStorage.Batch(
            observations=obs,
            actions=actions,
            values=torch.zeros(batch_size, 1),
            advantages=torch.ones(batch_size, 1),
            returns=torch.zeros(batch_size, 1),
            old_actions_log_prob=old_log_prob,
            old_distribution_params=old_distribution_params,
        )

        class _Storage:
            def __init__(self, update_batch):
                self.num_envs = batch_size
                self.num_transitions_per_env = 1
                self.device = "cpu"
                self.observations = obs.unsqueeze(0)
                self.next_observations = next_obs.unsqueeze(0)
                self.actions = update_batch.actions.unsqueeze(0)
                self.values = update_batch.values.unsqueeze(0)
                self.advantages = update_batch.advantages.unsqueeze(0)
                self.returns = update_batch.returns.unsqueeze(0)
                self.actions_log_prob = update_batch.old_actions_log_prob.unsqueeze(0)
                self.distribution_params = tuple(
                    param.unsqueeze(0) for param in update_batch.old_distribution_params
                )
                self.cleared = False

            def mini_batch_generator(self, num_mini_batches, num_epochs):
                yield self.update_batch

            def clear(self):
                self.cleared = True

        algo = GoalkeeperPPO(
            actor=actor,
            critic=critic,
            storage=_Storage(batch),
            num_learning_epochs=1,
            num_mini_batches=1,
            desired_kl=None,
            learning_rate=1e-4,
            amp_cfg={
                "motion_dir": "src/assets/soccer/motions/goalkeeper",
                "reward_mode": "mix",
                "reward_coef": 0.4,
                "reward_dt": 1.0,
                "hidden_dims": (32, 16),
                "num_reward_samples": 1,
                "reward_sigma": 0.0,
            },
        )

        with torch.no_grad():
            for discriminator in algo.amp.discriminators:
                for param in discriminator.parameters():
                    param.zero_()
                discriminator[-1].bias.fill_(1.0)
        rewards = torch.full((batch_size,), 0.2)
        algo._pending_next_obs = next_obs
        algo.transition.observations = obs
        algo.transition.rewards = rewards.clone()
        algo._augment_transition_with_amp_reward()

        self.assertTrue(
            torch.allclose(
                algo.transition.rewards,
                torch.full((batch_size,), 0.32),
                atol=1e-6,
            )
        )

        loss_dict = algo.update()

        self.assertTrue(algo.storage.cleared)
        self.assertIn("amp_discriminator", loss_dict)
        self.assertIn("amp_expert", loss_dict)
        self.assertIn("amp_policy", loss_dict)
        self.assertIn("amp_reward", loss_dict)
        self.assertGreaterEqual(loss_dict["amp_reward"], 0.0)

    def test_amp_minibatches_keep_observation_and_next_observation_pairs(self):
        from rsl_rl.storage import RolloutStorage
        from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic
        from src.tasks.soccer.modules.goalkeeper_ppo import GoalkeeperPPO

        obs = TensorDict(
            {
                "actor": torch.zeros(6, 960),
                "critic": torch.zeros(6, 113),
            },
            batch_size=[6],
        )
        obs_groups = {"actor": ["actor"], "critic": ["critic"]}
        actor = GoalkeeperActorCritic(obs, obs_groups, "actor", 29)
        critic = GoalkeeperActorCritic(obs, obs_groups, "critic", 1)

        class _Storage:
            num_envs = 3
            num_transitions_per_env = 2
            device = "cpu"

            def __init__(self):
                self.observations = TensorDict(
                    {
                        "actor": torch.zeros(2, 3, 960),
                        "critic": torch.zeros(2, 3, 113),
                    },
                    batch_size=[2, 3],
                )
                self.next_observations = self.observations.clone()
                ids = torch.arange(6, dtype=torch.float32).view(2, 3)
                self.observations["critic"][..., 0] = ids
                self.next_observations["critic"][..., 0] = ids + 100.0
                self.actions = torch.zeros(2, 3, 29)
                self.values = torch.zeros(2, 3, 1)
                self.returns = torch.zeros(2, 3, 1)
                self.actions_log_prob = torch.zeros(2, 3, 1)
                self.advantages = torch.ones(2, 3, 1)
                self.distribution_params = (
                    torch.zeros(2, 3, 29),
                    torch.ones(2, 3, 29),
                )

            def clear(self):
                pass

        algo = GoalkeeperPPO(
            actor=actor,
            critic=critic,
            storage=_Storage(),
            num_learning_epochs=1,
            num_mini_batches=2,
            desired_kl=None,
            learning_rate=1e-4,
            amp_cfg={"enabled": False},
        )

        batches = list(algo._mini_batch_generator_with_next_obs(2, 1))

        self.assertEqual(len(batches), 2)
        for batch in batches:
            current_ids = batch.observations["critic"][:, 0]
            next_ids = batch.next_observations["critic"][:, 0]
            self.assertTrue(torch.allclose(next_ids, current_ids + 100.0))

    def test_terminal_transitions_are_excluded_from_amp_reward_and_disc_loss(self):
        from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic
        from src.tasks.soccer.modules.goalkeeper_ppo import GoalkeeperPPO

        batch_size = 3
        obs = TensorDict(
            {
                "actor": torch.zeros(batch_size, 960),
                "critic": torch.zeros(batch_size, 113),
            },
            batch_size=[batch_size],
        )
        next_obs = obs.clone()
        obs["critic"][:, 9:38] = 0.1
        next_obs["critic"][:, 9:38] = 0.2
        obs_groups = {"actor": ["actor"], "critic": ["critic"]}
        actor = GoalkeeperActorCritic(obs, obs_groups, "actor", 29)
        critic = GoalkeeperActorCritic(obs, obs_groups, "critic", 1)

        class _Storage:
            def __init__(self):
                self.num_envs = batch_size
                self.num_transitions_per_env = 1
                self.device = "cpu"
                self.observations = obs.unsqueeze(0)
                self.next_observations = next_obs.unsqueeze(0)
                self.amp_not_done = torch.tensor([[[1.0], [0.0], [1.0]]])
                self.actions = torch.zeros(1, batch_size, 29)
                self.values = torch.zeros(1, batch_size, 1)
                self.returns = torch.zeros(1, batch_size, 1)
                self.actions_log_prob = torch.zeros(1, batch_size, 1)
                self.advantages = torch.ones(1, batch_size, 1)
                self.distribution_params = (
                    torch.zeros(1, batch_size, 29),
                    torch.ones(1, batch_size, 29),
                )

            def clear(self):
                pass

        algo = GoalkeeperPPO(
            actor=actor,
            critic=critic,
            storage=_Storage(),
            num_learning_epochs=1,
            num_mini_batches=1,
            desired_kl=None,
            learning_rate=1e-4,
            amp_cfg={
                "motion_dir": "src/assets/soccer/motions/goalkeeper",
                "reward_mode": "mix",
                "reward_coef": 0.4,
                "reward_dt": 1.0,
                "hidden_dims": (32, 16),
                "num_reward_samples": 1,
                "reward_sigma": 0.0,
            },
        )
        with torch.no_grad():
            for discriminator in algo.amp.discriminators:
                for param in discriminator.parameters():
                    param.zero_()
                discriminator[-1].bias.fill_(1.0)

        rewards = torch.tensor([0.2, -10.0, 0.2])
        algo._pending_next_obs = next_obs
        algo._pending_amp_not_done = torch.tensor([1.0, 0.0, 1.0])
        algo.transition.observations = obs
        algo.transition.rewards = rewards.clone()
        algo._augment_transition_with_amp_reward()

        self.assertAlmostEqual(float(algo.transition.rewards[0]), 0.32, places=6)
        self.assertEqual(float(algo.transition.rewards[1]), -10.0)
        self.assertAlmostEqual(float(algo.transition.rewards[2]), 0.32, places=6)

        batches = list(algo._mini_batch_generator_with_next_obs(1, 1))
        self.assertEqual(int(batches[0].amp_not_done.sum().item()), 2)

    def test_goalkeeper_ppo_computes_official_policy_value_smoothness_loss(self):
        from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic
        from src.tasks.soccer.modules.goalkeeper_ppo import GoalkeeperPPO

        batch_size = 4
        obs = TensorDict(
            {
                "actor": torch.randn(batch_size, 960),
                "critic": torch.randn(batch_size, 113),
            },
            batch_size=[batch_size],
        )
        next_obs = TensorDict(
            {
                "actor": obs["actor"] + 0.25,
                "critic": obs["critic"] + 0.25,
            },
            batch_size=[batch_size],
        )
        obs_groups = {"actor": ["actor"], "critic": ["critic"]}
        actor = GoalkeeperActorCritic(obs, obs_groups, "actor", 29)
        critic = GoalkeeperActorCritic(obs, obs_groups, "critic", 1)

        class _Storage:
            def clear(self):
                pass

        algo = GoalkeeperPPO(
            actor=actor,
            critic=critic,
            storage=_Storage(),
            num_learning_epochs=1,
            num_mini_batches=1,
            desired_kl=None,
            learning_rate=1e-4,
            smoothness_lower_bound=0.1,
            smoothness_upper_bound=1.0,
            value_smoothness_coef=0.1,
            amp_cfg={"enabled": False},
        )
        actor(obs, stochastic_output=True)
        current_mean = actor.output_mean
        current_values = critic(obs)

        losses = algo._smoothness_loss(obs, next_obs, current_mean, current_values)

        self.assertGreater(float(losses["policy"].detach()), 0.0)
        self.assertGreaterEqual(float(losses["value"].detach()), 0.0)
        self.assertGreater(float(losses["total"].detach()), 0.0)

    def test_goalkeeper_training_config_exposes_amp_motion_prior(self):
        from dataclasses import asdict
        from src.tasks.soccer.config.g1.rl_cfg import (
            unitree_g1_goalkeeper_ppo_runner_cfg,
        )

        cfg = asdict(unitree_g1_goalkeeper_ppo_runner_cfg())
        amp_cfg = cfg["algorithm"]["amp_cfg"]

        self.assertTrue(amp_cfg["enabled"])
        self.assertEqual(
            Path(amp_cfg["motion_dir"]).as_posix(),
            "src/assets/soccer/motions/goalkeeper",
        )
        self.assertEqual(amp_cfg["reward_mode"], "mix")
        self.assertAlmostEqual(amp_cfg["reward_coef"], 0.4)
        self.assertAlmostEqual(amp_cfg["reward_dt"], 1.0)
        self.assertAlmostEqual(amp_cfg["reward_scale"], 0.5)
        self.assertEqual(amp_cfg["num_reward_samples"], 4)
        self.assertAlmostEqual(cfg["algorithm"]["value_smoothness_coef"], 0.1)
        self.assertAlmostEqual(cfg["algorithm"]["smoothness_upper_bound"], 1.0)
        self.assertAlmostEqual(cfg["algorithm"]["smoothness_lower_bound"], 0.1)


if __name__ == "__main__":
    unittest.main()
