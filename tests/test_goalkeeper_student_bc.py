"""Tests for position-conditioned goalkeeper student distillation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401


class GoalkeeperStudentBcTest(unittest.TestCase):
  def test_student_obs_appends_normalized_prediction_condition_to_teacher_actor_obs(self):
    from src.tasks.soccer.mdp.goalkeeper_student_obs import (
      GOALKEEPER_STUDENT_OBS_DIM,
      GOALKEEPER_TEACHER_ACTOR_OBS_DIM,
      build_goalkeeper_student_obs,
    )

    teacher_obs = torch.arange(2 * GOALKEEPER_TEACHER_ACTOR_OBS_DIM, dtype=torch.float32).view(
      2, GOALKEEPER_TEACHER_ACTOR_OBS_DIM
    )
    condition = torch.tensor([[0.25, 0.5, 0.75, 0.0], [-0.5, -0.2, 1.0, 1.0]])

    student_obs = build_goalkeeper_student_obs(teacher_obs, condition)

    self.assertEqual(student_obs.shape, (2, GOALKEEPER_STUDENT_OBS_DIM))
    self.assertTrue(torch.equal(student_obs[:, :GOALKEEPER_TEACHER_ACTOR_OBS_DIM], teacher_obs))
    self.assertTrue(torch.equal(student_obs[:, -4:], condition))

  def test_prediction_condition_normalizes_units_before_condition_encoder(self):
    from src.tasks.soccer.mdp.goalkeeper_student_obs import normalize_goalkeeper_prediction

    condition = normalize_goalkeeper_prediction(
      yz=torch.tensor([[1.5, 1.8], [-0.75, 0.45]]),
      time=torch.tensor([3.0, 0.75]),
      idle=torch.tensor([True, False]),
    )

    expected = torch.tensor([
      [1.0, 1.0, 1.0, 1.0],
      [-0.5, -0.5, 0.5, 0.0],
    ])
    self.assertTrue(torch.allclose(condition, expected))

  def test_prediction_condition_treats_prelaunch_ball_as_idle(self):
    from src.tasks.soccer.mdp.goalkeeper_student_obs import goalkeeper_prediction_condition

    class _BallData:
      root_link_pos_w = torch.tensor([[3.0, 0.0, 0.1]])
      root_link_lin_vel_w = torch.tensor([[-0.01, 0.0, 0.02]])
      root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    class _RobotData:
      root_link_pos_w = torch.tensor([[0.0, 0.0, 0.0]])
      root_link_lin_vel_w = torch.zeros(1, 3)
      root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    class _Ball:
      data = _BallData()

    class _Robot:
      data = _RobotData()

    env = type(
      "Env",
      (),
      {
        "device": "cpu",
        "scene": {"ball": _Ball(), "robot": _Robot()},
      },
    )()

    condition = goalkeeper_prediction_condition(env)

    expected = torch.tensor([[0.0, (0.1 - 0.9) / 0.9, 0.0, 1.0]])
    self.assertTrue(torch.allclose(condition, expected, atol=1e-6))

  def test_prediction_condition_uses_delayed_launch_flag_for_idle_label(self):
    from src.tasks.soccer.mdp.goalkeeper_student_obs import goalkeeper_prediction_condition

    class _BallData:
      root_link_pos_w = torch.tensor([[3.0, 0.0, 0.1], [3.0, 0.0, 0.1]])
      root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
      root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    class _RobotData:
      root_link_pos_w = torch.zeros(2, 3)
      root_link_lin_vel_w = torch.zeros(2, 3)
      root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    class _Ball:
      data = _BallData()

    class _Robot:
      data = _RobotData()

    env = type(
      "Env",
      (),
      {
        "device": "cpu",
        "scene": {"ball": _Ball(), "robot": _Robot()},
        "_gk_delayed_ball_launched": torch.tensor([False, True]),
      },
    )()

    condition = goalkeeper_prediction_condition(env)

    self.assertEqual(condition[0, -1].item(), 1.0)
    self.assertEqual(condition[1, -1].item(), 0.0)

  def test_goalkeeper_actor_history_matches_observation_manager_term_major_layout(self):
    from src.tasks.soccer.mdp.goalkeeper_student_obs import (
      GOALKEEPER_ACTOR_FRAME_DIM,
      GOALKEEPER_ACTOR_HISTORY,
      _flatten_goalkeeper_actor_history,
    )

    history = torch.arange(
      GOALKEEPER_ACTOR_HISTORY * GOALKEEPER_ACTOR_FRAME_DIM,
      dtype=torch.float32,
    ).view(1, GOALKEEPER_ACTOR_HISTORY, GOALKEEPER_ACTOR_FRAME_DIM)

    flattened = _flatten_goalkeeper_actor_history(history)

    expected = torch.cat([
      history[:, :, 0:3].reshape(1, -1),
      history[:, :, 3:6].reshape(1, -1),
      history[:, :, 6:9].reshape(1, -1),
      history[:, :, 9:38].reshape(1, -1),
      history[:, :, 38:67].reshape(1, -1),
      history[:, :, 67:96].reshape(1, -1),
    ], dim=-1)
    self.assertTrue(torch.equal(flattened, expected))

  def test_goalkeeper_actor_history_resets_new_episodes_to_current_frame(self):
    from unittest.mock import patch
    from src.tasks.soccer.mdp import goalkeeper_student_obs as gk_obs
    from src.tasks.soccer.mdp.goalkeeper_student_obs import (
      GOALKEEPER_ACTOR_FRAME_DIM,
      GOALKEEPER_ACTOR_HISTORY,
      _flatten_goalkeeper_actor_history,
      _goalkeeper_actor_obs_history,
    )

    env = type("Env", (), {})()
    env.num_envs = 2
    env.device = "cpu"
    env.episode_length_buf = torch.tensor([0, 7])
    env._gk_student_actor_history = torch.ones(
      2,
      GOALKEEPER_ACTOR_HISTORY,
      GOALKEEPER_ACTOR_FRAME_DIM,
    )
    frame = torch.stack([
      torch.full((GOALKEEPER_ACTOR_FRAME_DIM,), 10.0),
      torch.full((GOALKEEPER_ACTOR_FRAME_DIM,), 20.0),
    ])

    with patch.object(gk_obs, "_goalkeeper_actor_obs_frame", return_value=frame):
      obs = _goalkeeper_actor_obs_history(env)

    expected_history = torch.ones_like(env._gk_student_actor_history)
    expected_history[0] = frame[0].view(1, -1).repeat(GOALKEEPER_ACTOR_HISTORY, 1)
    expected_history[1] = torch.cat([
      torch.ones(GOALKEEPER_ACTOR_HISTORY - 1, GOALKEEPER_ACTOR_FRAME_DIM),
      frame[1].view(1, -1),
    ], dim=0)
    self.assertTrue(torch.equal(env._gk_student_actor_history, expected_history))
    self.assertTrue(torch.equal(obs, _flatten_goalkeeper_actor_history(expected_history)))

  def test_lstm_student_actor_uses_film_conditioning_after_960d_lstm(self):
    from scripts.train_goalkeeper_student_bc import BcConfig, make_goalkeeper_student_actor
    from src.tasks.soccer.mdp.goalkeeper_student_obs import (
      GOALKEEPER_STUDENT_OBS_DIM,
      GOALKEEPER_TEACHER_ACTOR_OBS_DIM,
    )

    cfg = BcConfig(dataset_dir="unused", obs_normalization=False, batch_size=2)
    actor = make_goalkeeper_student_actor(
      obs_dim=GOALKEEPER_STUDENT_OBS_DIM,
      action_dim=29,
      cfg=cfg,
      device="cpu",
    )

    self.assertTrue(actor.is_recurrent)
    self.assertEqual(actor.rnn.rnn.input_size, GOALKEEPER_TEACHER_ACTOR_OBS_DIM)
    self.assertTrue(hasattr(actor, "condition_encoder"))
    self.assertTrue(hasattr(actor, "film"))
    self.assertTrue(hasattr(actor, "condition_aux_head"))
    self.assertTrue(hasattr(actor, "region_estimator"))
    self.assertTrue(hasattr(actor, "ball_estimator"))
    self.assertTrue(hasattr(actor, "region_film"))
    obs = TensorDict(
      {"student": torch.zeros(2, GOALKEEPER_STUDENT_OBS_DIM)},
      batch_size=[2],
    )
    action = actor(obs)

    self.assertEqual(action.shape, (2, 29))
    aux = actor.predict_condition_aux(obs)
    self.assertEqual(aux["landing_yz"].shape, (2, 2))
    self.assertEqual(aux["region_logits"].shape, (2, 6))
    region_aux = actor.region_condition_output
    self.assertIsNotNone(region_aux)
    assert region_aux is not None
    self.assertEqual(region_aux["region_logits"].shape, (2, 7))
    self.assertEqual(region_aux["ball_latent"].shape, (2, actor.ball_latent_dim))

  def test_goalkeeper_student_region_condition_changes_active_action_not_prepare_action(self):
    from scripts.train_goalkeeper_student_bc import BcConfig, make_goalkeeper_student_actor
    from src.tasks.soccer.mdp.goalkeeper_student_obs import GOALKEEPER_STUDENT_OBS_DIM

    cfg = BcConfig(dataset_dir="unused", obs_normalization=False, batch_size=2)
    actor = make_goalkeeper_student_actor(GOALKEEPER_STUDENT_OBS_DIM, 29, cfg, "cpu")
    actor.eval()

    obs = torch.zeros(2, GOALKEEPER_STUDENT_OBS_DIM)
    obs[:, -1] = torch.tensor([0.0, 1.0])
    with torch.no_grad():
      for param in actor.condition_encoder.parameters():
        param.zero_()
      actor.film.weight.zero_()
      actor.film.bias.zero_()
      actor.region_film.weight.zero_()
      actor.region_film.bias.zero_()
      actor.region_film.bias[: actor.rnn.rnn.hidden_size].fill_(0.25)
      actor.prepare_region_film.weight.zero_()
      actor.prepare_region_film.bias.zero_()

    td = TensorDict({"student": obs}, batch_size=[2])
    actor.reset()
    active_latent = actor.rnn(obs[:, :960]).squeeze(0)[0]
    actor.reset()
    conditioned = actor.get_latent(td)

    self.assertTrue(torch.allclose(conditioned[0], active_latent * 1.25, atol=1e-5))
    self.assertFalse(torch.allclose(conditioned[1], active_latent * 1.25, atol=1e-5))

  def test_goalkeeper_student_actor_clamps_negative_std_before_sampling(self):
    from scripts.train_goalkeeper_student_bc import BcConfig, make_goalkeeper_student_actor
    from src.tasks.soccer.mdp.goalkeeper_student_obs import GOALKEEPER_STUDENT_OBS_DIM

    cfg = BcConfig(dataset_dir="unused", obs_normalization=False, batch_size=2)
    actor = make_goalkeeper_student_actor(GOALKEEPER_STUDENT_OBS_DIM, 29, cfg, "cpu")
    actor.min_sample_std = 0.01
    with torch.no_grad():
      actor.distribution.std_param.fill_(-0.5)
    obs = TensorDict({"student": torch.zeros(2, GOALKEEPER_STUDENT_OBS_DIM)}, batch_size=[2])

    action = actor(obs, stochastic_output=True)

    self.assertEqual(action.shape, (2, 29))
    self.assertTrue(torch.all(actor.distribution.std_param >= 0.01))
    self.assertTrue(torch.all(torch.isfinite(actor.distribution.std_param)))

  def test_bc_chunk_forward_uses_goalkeeper_film_path(self):
    from scripts.train_goalkeeper_student_bc import BcConfig, make_goalkeeper_student_actor
    from scripts.train_shooter_bc import _forward_rnn_chunk
    from src.tasks.soccer.mdp.goalkeeper_student_obs import GOALKEEPER_STUDENT_OBS_DIM

    cfg = BcConfig(dataset_dir="unused", obs_normalization=False, batch_size=2)
    actor = make_goalkeeper_student_actor(GOALKEEPER_STUDENT_OBS_DIM, 29, cfg, "cpu")

    obs = torch.zeros(3, 2, GOALKEEPER_STUDENT_OBS_DIM)
    masks = torch.ones(3, 2, dtype=torch.bool)
    pred, hidden_state = _forward_rnn_chunk(actor, obs, masks)

    self.assertEqual(pred.shape, (6, 29))
    self.assertIsNotNone(hidden_state)

  def test_goalkeeper_bc_region_aux_loss_supervises_prepare_and_active_routes(self):
    from scripts.train_goalkeeper_student_bc import (
      BcConfig,
      _goalkeeper_region_aux_loss,
      _goalkeeper_route_targets_from_obs,
      make_goalkeeper_student_actor,
    )
    from scripts.train_shooter_bc import _forward_rnn_chunk
    from src.tasks.soccer.mdp.goalkeeper_student_obs import GOALKEEPER_STUDENT_OBS_DIM

    cfg = BcConfig(
      dataset_dir="unused",
      obs_normalization=False,
      batch_size=2,
      region_aux_active_only=False,
    )
    actor = make_goalkeeper_student_actor(GOALKEEPER_STUDENT_OBS_DIM, 29, cfg, "cpu")
    obs = torch.zeros(1, 2, GOALKEEPER_STUDENT_OBS_DIM)
    masks = torch.ones(1, 2, dtype=torch.bool)

    active_history = obs[0, 0, :30].view(10, 3)
    active_history[:, :] = torch.tensor([0.44, 0.2, 1.1])
    active_history[-1] = torch.tensor([0.40, 0.2, 1.1])
    obs[0, 0, -1] = 0.0
    obs[0, 1, -1] = 1.0

    _forward_rnn_chunk(actor, obs, masks)
    targets = _goalkeeper_route_targets_from_obs(obs[masks], route_gate=None)
    loss, samples = _goalkeeper_region_aux_loss(
      actor,
      obs,
      masks,
      cfg,
      route_gate=None,
      reduction="mean",
    )

    self.assertEqual(targets.tolist(), [0, 6])
    self.assertEqual(samples, 2)
    self.assertTrue(loss.requires_grad)
    self.assertEqual(actor.region_condition_output["region_logits"].shape, (2, 7))

  def test_rnn_bc_minibatches_stream_shards_before_first_yield(self):
    from scripts.train_goalkeeper_student_bc import BcConfig, _iter_streaming_rnn_minibatches

    shards = [Path(f"shard_{idx}.pt") for idx in range(4)]
    loaded: list[str] = []

    def _fake_load(path: Path, success_only: bool, action_clip: float | None):
      loaded.append(path.name)
      obs = torch.zeros(2, 3)
      act = torch.zeros(2, 1)
      return [(obs, act)]

    cfg = BcConfig(
      dataset_dir="unused",
      batch_size=2,
      success_only=False,
      max_val_batches=None,
    )
    with patch("scripts.train_shooter_bc._load_episode_sequences", side_effect=_fake_load):
      iterator = _iter_streaming_rnn_minibatches(
        shards, cfg, "cpu", torch.Generator().manual_seed(0), shuffle=False
      )
      obs_padded, act_padded, masks = next(iterator)

    self.assertEqual(loaded, ["shard_0.pt", "shard_1.pt"])
    self.assertEqual(obs_padded.shape, (2, 2, 3))
    self.assertEqual(act_padded.shape, (2, 2, 1))
    self.assertTrue(torch.all(masks))

  def test_goalkeeper_bc_normalizer_warmup_samples_shards(self):
    from scripts.train_goalkeeper_student_bc import BcConfig, _update_goalkeeper_obs_normalizer

    class _Actor:
      def __init__(self):
        self.updates = 0

      def train(self):
        pass

      def update_normalization(self, obs):
        self.updates += 1

    shards = [Path(f"shard_{idx}.pt") for idx in range(5)]
    loaded: list[str] = []

    def _fake_load(path: Path, success_only: bool, action_clip: float | None):
      loaded.append(path.name)
      return torch.zeros(3, 4), torch.zeros(3, 1)

    cfg = BcConfig(
      dataset_dir="unused",
      batch_size=2,
      normalizer_max_shards=2,
      obs_normalization=True,
    )
    actor = _Actor()
    with patch("scripts.train_shooter_bc._load_flat_samples", side_effect=_fake_load):
      samples = _update_goalkeeper_obs_normalizer(actor, shards, cfg, "cpu")

    self.assertEqual(len(loaded), 2)
    self.assertEqual(samples, 6)
    self.assertEqual(actor.updates, 4)

  def test_teacher_loader_uses_moe_bundle_adapter_for_gate_plus_seven_experts(self):
    from scripts.collect_goalkeeper_teacher_dataset import _load_teacher_policy

    bundle = {
      "sr": [{"actor_state_dict": {"w": torch.tensor([i])}} for i in range(6)],
      "idle": {"actor_state_dict": {"w": torch.tensor([6])}},
      "gate": {"state": {}, "mean": torch.zeros(6), "std": torch.ones(6), "num_classes": 7},
    }
    sentinel_policy = object()
    with tempfile.TemporaryDirectory() as tmp:
      ckpt = Path(tmp) / "moe7.pt"
      torch.save(bundle, ckpt)

      with patch(
        "scripts.collect_goalkeeper_teacher_dataset.MoE6GoalkeeperPolicy",
        return_value=sentinel_policy,
      ) as policy_cls:
        policy = _load_teacher_policy(str(ckpt), env=object(), device="cpu")

    self.assertIs(policy, sentinel_policy)
    policy_cls.assert_called_once()
    called_bundle = policy_cls.call_args.args[0]
    self.assertEqual(len(called_bundle["sr"]), 6)
    self.assertIn("idle", called_bundle)
    self.assertEqual(called_bundle["gate"]["num_classes"], 7)

  def test_collection_delayed_launch_keeps_prepare_phase_in_rollout(self):
    from mjlab.tasks.registry import load_env_cfg
    from scripts.collect_goalkeeper_teacher_dataset import _apply_delayed_launch_sampling

    cfg = load_env_cfg("Unitree-G1-Goalkeeper-Train", play=False)
    old_episode_length_s = cfg.episode_length_s

    _apply_delayed_launch_sampling(cfg, wait_s=3.0)

    reset_event = cfg.events["reset_ball"]
    launch_event = cfg.events["launch_delayed_ball"]
    self.assertEqual(reset_event.func.__name__, "reset_ball_staged_delayed_launch")
    self.assertNotIn("perturb_ball_vel", cfg.events)
    self.assertEqual(launch_event.func.__name__, "launch_staged_ball_after_delay")
    self.assertEqual(launch_event.mode, "step")
    self.assertEqual(launch_event.params["wait_s"], 3.0)
    self.assertEqual(reset_event.params["ball_pos"], (3.0, 0.0, 0.1))
    self.assertEqual(reset_event.params["sampler_params"]["fixed_start_local"], (3.0, 0.0, 0.1))
    self.assertAlmostEqual(cfg.episode_length_s, old_episode_length_s + 3.0)

  def test_student_ppo_task_uses_conditioned_actor_and_recurrent_privileged_critic(self):
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

    task_id = "Unitree-G1-Goalkeeper-Student-PPO"
    self.assertIn(task_id, list_tasks())
    env_cfg = load_env_cfg(task_id)
    rl_cfg = load_rl_cfg(task_id)

    self.assertEqual(list(env_cfg.observations["actor"].terms.keys()), ["student"])
    self.assertEqual(env_cfg.observations["actor"].terms["student"].func.__name__, "goalkeeper_student_obs")
    self.assertEqual(env_cfg.observations["actor"].history_length, 1)
    self.assertEqual(env_cfg.observations["critic"].history_length, 1)
    self.assertIn("goal_conceded", env_cfg.rewards)
    self.assertLess(env_cfg.rewards["goal_conceded"].weight, 0.0)
    self.assertEqual(env_cfg.rewards["goal_conceded"].func.__name__, "goalkeeper_active_goal_conceded")
    self.assertIn("intercept", env_cfg.rewards)
    self.assertEqual(env_cfg.rewards["intercept"].func.__name__, "goalkeeper_active_intercept_point")
    self.assertIn("condition_target_reach", env_cfg.rewards)
    self.assertEqual(env_cfg.rewards["condition_target_reach"].func.__name__, "goalkeeper_active_condition_target_reach")
    self.assertGreater(env_cfg.rewards["condition_target_reach"].weight, 0.0)
    self.assertIn("body", env_cfg.rewards)
    self.assertEqual(env_cfg.rewards["body"].func.__name__, "goalkeeper_active_body_intercept")
    self.assertIn("stop_ball", env_cfg.rewards)
    self.assertEqual(env_cfg.rewards["stop_ball"].func.__name__, "goalkeeper_active_stop_ball")
    self.assertIn("idle_fall_penalty", env_cfg.rewards)
    self.assertLessEqual(env_cfg.rewards["idle_fall_penalty"].weight, -1000.0)
    self.assertIn("idle_alive", env_cfg.rewards)
    self.assertGreater(env_cfg.rewards["idle_alive"].weight, 0.0)
    self.assertIn("idle_action_rate", env_cfg.rewards)
    self.assertEqual(env_cfg.rewards["idle_action_rate"].func.__name__, "goalkeeper_idle_action_rate")
    self.assertLess(env_cfg.rewards["idle_action_rate"].weight, 0.0)
    self.assertIn("idle_low_base_height", env_cfg.rewards)
    self.assertIn("idle_base_height_band", env_cfg.rewards)
    self.assertIn("idle_leg_ready_pose", env_cfg.rewards)
    self.assertEqual(rl_cfg.actor.class_name, "GoalkeeperStudentFiLMActor")
    self.assertEqual(rl_cfg.critic.class_name, "RNNModel")
    self.assertEqual(rl_cfg.actor.hidden_dims, (256, 128, 64))
    self.assertEqual(rl_cfg.algorithm.class_name, "src.tasks.soccer.modules.goalkeeper_student_ppo:GoalkeeperStudentPPO")
    self.assertGreater(rl_cfg.algorithm.distill_coef, 0.0)
    self.assertGreater(rl_cfg.algorithm.condition_aux_coef, 0.0)
    self.assertGreater(rl_cfg.algorithm.offline_bc_active_fraction, 0.0)
    self.assertEqual(load_runner_cls(task_id).__name__, "GoalkeeperStudentRunner")

  def test_student_ppo_actor_is_about_one_million_parameters(self):
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from tensordict import TensorDict
    from src.tasks.soccer.config.g1.rl_cfg import GoalkeeperStudentRunner
    from src.tasks.soccer.modules.goalkeeper_student_actor import GoalkeeperStudentFiLMActor
    from src.tasks.soccer.mdp.goalkeeper_student_obs import GOALKEEPER_STUDENT_OBS_DIM

    env_cfg = load_env_cfg("Unitree-G1-Goalkeeper-Student-PPO")
    rl_cfg = load_rl_cfg("Unitree-G1-Goalkeeper-Student-PPO")
    train_cfg = {
      "actor": {
        "class_name": rl_cfg.actor.class_name,
        "hidden_dims": rl_cfg.actor.hidden_dims,
        "activation": rl_cfg.actor.activation,
        "obs_normalization": rl_cfg.actor.obs_normalization,
        "distribution_cfg": rl_cfg.actor.distribution_cfg,
      },
      "critic": {
        "class_name": rl_cfg.critic.class_name,
        "hidden_dims": rl_cfg.critic.hidden_dims,
        "activation": rl_cfg.critic.activation,
        "obs_normalization": rl_cfg.critic.obs_normalization,
      },
    }
    GoalkeeperStudentRunner._inject_recurrent_defaults(train_cfg)
    dummy_obs = TensorDict({"student": torch.zeros(1, GOALKEEPER_STUDENT_OBS_DIM)}, batch_size=[1])
    actor = GoalkeeperStudentFiLMActor(
      dummy_obs,
      {"actor": ["student"]},
      "actor",
      output_dim=29,
      hidden_dims=tuple(train_cfg["actor"]["hidden_dims"]),
      activation=train_cfg["actor"]["activation"],
      obs_normalization=train_cfg["actor"]["obs_normalization"],
      distribution_cfg=train_cfg["actor"]["distribution_cfg"],
      rnn_type=train_cfg["actor"]["rnn_type"],
      rnn_hidden_dim=train_cfg["actor"]["rnn_hidden_dim"],
      rnn_num_layers=train_cfg["actor"]["rnn_num_layers"],
      condition_hidden_dim=train_cfg["actor"]["condition_hidden_dim"],
    )

    param_count = sum(param.numel() for param in actor.parameters())

    self.assertGreaterEqual(param_count, 900_000)
    self.assertLessEqual(param_count, 1_200_000)
    self.assertEqual(actor.rnn.rnn.hidden_size, 160)

  def test_student_ppo_script_defaults_to_scratch_hard3_teacher_task(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(num_envs=64, max_iterations=7)
    train_cfg = build_train_config(cfg)

    self.assertEqual(cfg.task_id, "Unitree-G1-Goalkeeper-Student-PPO")
    self.assertIsNone(train_cfg.load_checkpoint_path)
    self.assertFalse(train_cfg.load_actor_only)
    self.assertEqual(train_cfg.env.scene.num_envs, 64)
    self.assertEqual(train_cfg.agent.max_iterations, 7)
    self.assertEqual(train_cfg.agent.run_name, cfg.run_name)
    self.assertEqual(
      train_cfg.agent.algorithm.teacher_checkpoint_path,
      "/data/Courses/[CS2810]EmbodiedAI/humanoid_soccer_proj/G1-mjlab-soccer/logs/repairs/goalkeeper_moe7_hard3_default_idle.pt",
    )
    self.assertGreater(train_cfg.agent.algorithm.distill_coef, 0.0)
    self.assertNotIn("push_robot", train_cfg.env.events)
    self.assertIn("idle_low_base_height", train_cfg.env.rewards)
    self.assertIn("idle_base_height_band", train_cfg.env.rewards)
    self.assertIn("idle_leg_ready_pose", train_cfg.env.rewards)
    self.assertEqual(train_cfg.env.rewards["idle_low_base_height"].weight, -20.0)
    self.assertEqual(train_cfg.env.rewards["idle_base_height_band"].weight, 5.0)
    self.assertEqual(train_cfg.env.rewards["idle_leg_ready_pose"].weight, -5.0)
    self.assertIn("condition_target_reach", train_cfg.env.rewards)

  def test_student_ppo_bc_regularized_profile_simplifies_prepare_rewards(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(
      init="checkpoints/model_best.pt",
      num_envs=64,
      max_iterations=7,
      profile="bc_regularized_finetune",
      offline_bc_dataset="data/goalkeeper_student/teacher_rollouts/merged",
      offline_bc_coef=0.05,
      offline_idle_bc_coef=1.0,
      offline_bc_batch_size=16,
      offline_bc_seq_len=24,
      offline_bc_every_n_updates=4,
      num_learning_epochs=3,
      learning_rate=1.0e-4,
    )
    train_cfg = build_train_config(cfg)

    self.assertEqual(train_cfg.load_checkpoint_path, "checkpoints/model_best.pt")
    self.assertTrue(train_cfg.load_actor_only)
    self.assertFalse(train_cfg.init_at_random_ep_len)
    self.assertIn("idle_fall_penalty", train_cfg.env.rewards)
    self.assertAlmostEqual(train_cfg.env.rewards["idle_fall_penalty"].weight, -1000.0)
    self.assertIn("action_rate", train_cfg.env.rewards)
    self.assertLess(train_cfg.env.rewards["action_rate"].weight, 0.0)
    self.assertNotIn("posture", train_cfg.env.rewards)
    self.assertIn("idle_low_base_height", train_cfg.env.rewards)
    self.assertIn("idle_base_height_band", train_cfg.env.rewards)
    self.assertIn("idle_leg_ready_pose", train_cfg.env.rewards)
    self.assertIn("active_upright", train_cfg.env.rewards)
    self.assertGreater(train_cfg.env.rewards["active_upright"].weight, 0.0)
    self.assertIn("active_fall_penalty", train_cfg.env.rewards)
    self.assertLess(train_cfg.env.rewards["active_fall_penalty"].weight, 0.0)
    self.assertIn("idle_alive", train_cfg.env.rewards)
    self.assertGreater(train_cfg.env.rewards["idle_alive"].weight, 0.0)
    self.assertIn("idle_action_rate", train_cfg.env.rewards)
    self.assertAlmostEqual(train_cfg.env.rewards["idle_action_rate"].weight, -0.05)
    self.assertIn("idle_upright", train_cfg.env.rewards)
    self.assertGreater(train_cfg.env.rewards["idle_upright"].weight, 0.0)
    self.assertIn("idle_base_still", train_cfg.env.rewards)
    self.assertLess(train_cfg.env.rewards["idle_base_still"].weight, 0.0)
    self.assertEqual(
      train_cfg.agent.algorithm.offline_bc_dataset,
      "data/goalkeeper_student/teacher_rollouts/merged",
    )
    self.assertEqual(train_cfg.agent.algorithm.offline_bc_coef, 0.05)
    self.assertEqual(train_cfg.agent.algorithm.offline_idle_bc_coef, 1.0)
    self.assertEqual(train_cfg.agent.algorithm.offline_bc_batch_size, 16)
    self.assertEqual(train_cfg.agent.algorithm.offline_bc_seq_len, 24)
    self.assertEqual(train_cfg.agent.algorithm.offline_bc_every_n_updates, 4)
    self.assertEqual(train_cfg.agent.algorithm.num_learning_epochs, 3)
    self.assertEqual(train_cfg.agent.algorithm.learning_rate, 1.0e-4)

  def test_student_ppo_polish_profile_matches_train_polish_safety_knobs(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(
      init="checkpoints/model_best.pt",
      num_envs=64,
      max_iterations=7,
      profile="student_polish",
      offline_bc_dataset="data/goalkeeper_student/teacher_rollouts/merged",
    )
    train_cfg = build_train_config(cfg)
    alg = train_cfg.agent.algorithm

    self.assertEqual(train_cfg.actor_std_override, 0.06)
    self.assertAlmostEqual(alg.max_action_std, 0.06)
    self.assertAlmostEqual(alg.min_action_std, 0.01)
    self.assertAlmostEqual(alg.learning_rate, 1.0e-4)
    self.assertAlmostEqual(alg.clip_param, 0.1)
    self.assertAlmostEqual(alg.desired_kl, 0.005)
    self.assertAlmostEqual(alg.entropy_coef, 0.0)
    self.assertAlmostEqual(alg.offline_bc_coef, 0.5)
    self.assertAlmostEqual(alg.offline_idle_bc_coef, 0.5)
    self.assertEqual(alg.offline_bc_every_n_updates, 1)
    self.assertEqual(alg.critic_warmup_iterations, 50)
    self.assertTrue(alg.mask_idle_actor_loss)
    self.assertEqual(cfg.eval_interval, 20)
    self.assertGreater(cfg.rollback_drop, 0.0)

  def test_student_ppo_script_can_still_load_actor_checkpoint_when_init_is_given(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(init="student_bc.pt", num_envs=64, max_iterations=7)
    train_cfg = build_train_config(cfg)

    self.assertEqual(train_cfg.load_checkpoint_path, "student_bc.pt")
    self.assertTrue(train_cfg.load_actor_only)

  def test_student_ppo_script_can_resume_full_checkpoint_when_requested(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(init="model_6700.pt", resume_full=True, num_envs=64, max_iterations=7)
    train_cfg = build_train_config(cfg)

    self.assertEqual(train_cfg.load_checkpoint_path, "model_6700.pt")
    self.assertFalse(train_cfg.load_actor_only)
    self.assertFalse(train_cfg.load_model_only)

  def test_student_ppo_script_allows_distillation_knob_overrides(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(
      init="student_bc.pt",
      teacher="/tmp/teacher.pt",
      distill_coef=0.25,
      teacher_every_n_steps=3,
    )
    train_cfg = build_train_config(cfg)

    self.assertEqual(train_cfg.agent.algorithm.teacher_checkpoint_path, "/tmp/teacher.pt")
    self.assertEqual(train_cfg.agent.algorithm.distill_coef, 0.25)
    self.assertEqual(train_cfg.agent.algorithm.teacher_every_n_steps, 3)

  def test_student_ppo_bc_finetune_freezes_actor_obs_normalizer_updates(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(
      init="checkpoints/model_best.pt",
      profile="bc_regularized_finetune",
    )
    train_cfg = build_train_config(cfg)

    self.assertTrue(train_cfg.agent.algorithm.freeze_actor_obs_normalization)

  def test_student_ppo_script_can_override_loaded_actor_std_and_entropy(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(
      init="student_bc.pt",
      init_action_std=0.03,
      entropy_coef=0.0,
    )
    train_cfg = build_train_config(cfg)

    self.assertEqual(train_cfg.actor_std_override, 0.03)
    self.assertEqual(train_cfg.agent.algorithm.entropy_coef, 0.0)

  def test_student_ppo_script_wires_condition_strength_knobs(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(
      init="student_bc.pt",
      distill_coef=0.05,
      distill_final_coef=0.0,
      distill_anneal_updates=100,
      offline_bc_coef=0.5,
      offline_bc_final_coef=0.05,
      offline_bc_anneal_updates=200,
      offline_bc_active_fraction=0.75,
      condition_aux_coef=0.2,
      condition_aux_final_coef=0.05,
      condition_aux_anneal_updates=300,
    )
    train_cfg = build_train_config(cfg)
    alg = train_cfg.agent.algorithm

    self.assertEqual(alg.distill_coef, 0.05)
    self.assertEqual(alg.distill_final_coef, 0.0)
    self.assertEqual(alg.distill_anneal_updates, 100)
    self.assertEqual(alg.offline_bc_coef, 0.5)
    self.assertEqual(alg.offline_bc_final_coef, 0.05)
    self.assertEqual(alg.offline_bc_anneal_updates, 200)
    self.assertEqual(alg.offline_bc_active_fraction, 0.75)
    self.assertEqual(alg.condition_aux_coef, 0.2)
    self.assertEqual(alg.condition_aux_final_coef, 0.05)
    self.assertEqual(alg.condition_aux_anneal_updates, 300)

  def test_student_ppo_script_can_override_min_action_std_floor(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(
      init="student_bc.pt",
      min_action_std=0.01,
    )
    train_cfg = build_train_config(cfg)

    self.assertEqual(train_cfg.agent.algorithm.min_action_std, 0.01)

  def test_student_ppo_script_can_clip_actor_mean_for_recovery_finetune(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(
      init="model_4999.pt",
      actor_mean_clip=20.0,
    )
    train_cfg = build_train_config(cfg)

    self.assertEqual(train_cfg.agent.algorithm.actor_mean_clip, 20.0)

  def test_student_ppo_script_wires_recovery_stability_knobs(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(
      init="model_4999.pt",
      ppo_log_ratio_clip=4.0,
      idle_deterministic_actions=False,
      normalize_advantage_per_mini_batch=True,
    )
    train_cfg = build_train_config(cfg)
    alg = train_cfg.agent.algorithm

    self.assertEqual(alg.ppo_log_ratio_clip, 4.0)
    self.assertFalse(alg.idle_deterministic_actions)
    self.assertTrue(alg.normalize_advantage_per_mini_batch)

  def test_train_helper_overrides_loaded_actor_std_param(self):
    from scripts.train import _override_actor_std

    std_param = torch.full((29,), 0.2)
    runner = type(
      "Runner",
      (),
      {
        "alg": type(
          "Alg",
          (),
          {
            "actor": type(
              "Actor",
              (),
              {
                "distribution": type("Distribution", (), {"std_param": std_param})(),
              },
            )(),
          },
        )(),
      },
    )()

    _override_actor_std(runner, 0.03)

    self.assertTrue(torch.allclose(std_param, torch.full((29,), 0.03)))

  def test_student_ppo_clamps_actor_std_min_without_capping_exploration(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    std_param = torch.tensor([-0.5, 0.2, 1.5, float("nan"), float("inf"), float("-inf")])
    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.min_action_std = 1.0e-4
    algo.max_action_std = None
    algo.actor = type(
      "Actor",
      (),
      {
        "distribution": type("Distribution", (), {"std_param": std_param})(),
      },
    )()

    algo._clamp_actor_std()

    self.assertTrue(torch.allclose(std_param, torch.tensor([1.0e-4, 0.2, 1.5, 1.0e-4, 1.0e-4, 1.0e-4])))

  def test_student_ppo_can_cap_actor_std_for_polish_finetune(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    std_param = torch.tensor([-0.5, 0.2, 1.5, float("nan"), float("inf"), float("-inf")])
    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.min_action_std = 0.01
    algo.max_action_std = 0.06
    algo.actor = type(
      "Actor",
      (),
      {
        "distribution": type("Distribution", (), {"std_param": std_param})(),
      },
    )()

    algo._clamp_actor_std()

    self.assertTrue(torch.allclose(std_param, torch.tensor([0.01, 0.06, 0.06, 0.01, 0.06, 0.01])))

  def test_student_ppo_can_freeze_actor_during_critic_warmup(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.actor = torch.nn.Linear(2, 2)
    algo.critic = torch.nn.Linear(2, 1)

    previous = algo.set_actor_trainable(False)

    self.assertEqual(previous, [True, True])
    self.assertFalse(any(param.requires_grad for param in algo.actor.parameters()))
    self.assertTrue(all(param.requires_grad for param in algo.critic.parameters()))

    algo.restore_actor_trainable(previous)

    self.assertTrue(all(param.requires_grad for param in algo.actor.parameters()))

  def test_student_ppo_masks_idle_samples_from_actor_policy_loss(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.mask_idle_actor_loss = True
    algo.clip_param = 0.1
    algo.actor = type("Actor", (), {"_raw_student_obs": lambda _self, obs: obs["student"]})()
    observations = {"student": torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.2, 0.1, 0.3, 0.0]])}
    advantages = torch.tensor([100.0, 1.0])
    ratio = torch.ones(2)
    entropy = torch.tensor([1000.0, 2.0])

    surrogate_loss, entropy_loss = algo._masked_actor_losses(observations, advantages, ratio, entropy)

    self.assertTrue(torch.allclose(surrogate_loss, torch.tensor(-1.0)))
    self.assertTrue(torch.allclose(entropy_loss, torch.tensor(2.0)))

  def test_student_ppo_clips_actor_mean_when_recovery_clip_is_set(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    mean = torch.tensor([[-40.0, 3.0, 50.0]])
    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.actor_mean_clip = 20.0
    algo.actor = type(
      "Actor",
      (),
      {
        "output_mean": mean,
      },
    )()

    clipped = algo._actor_output_mean()

    self.assertTrue(torch.equal(clipped, torch.tensor([[-20.0, 3.0, 20.0]])))

  def test_student_ppo_clips_rollout_actions_when_recovery_clip_is_set(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.actor_mean_clip = 20.0

    actions = torch.tensor([[-40.0, 3.0, 50.0, float("nan"), float("inf"), float("-inf")]])

    expected = torch.tensor([[-20.0, 3.0, 20.0, 0.0, 20.0, -20.0]])
    self.assertTrue(torch.equal(algo._clip_actor_actions(actions), expected))

  def test_student_ppo_clamps_extreme_log_ratio_before_exp(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.ppo_log_ratio_clip = 8.0

    ratio = algo._ppo_ratio(
      torch.tensor([1000.0, -1000.0, float("nan"), float("inf"), float("-inf")]),
      torch.zeros(5),
    )

    expected = torch.exp(torch.tensor([8.0, -8.0, 0.0, 8.0, -8.0]))
    self.assertTrue(torch.allclose(ratio, expected))

  def test_student_ppo_linear_loss_schedule_anneals_to_final_value(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo._ppo_update_count = 0

    self.assertEqual(algo._scheduled_coef(1.0, 0.1, 10), 1.0)
    algo._ppo_update_count = 5
    self.assertAlmostEqual(algo._scheduled_coef(1.0, 0.1, 10), 0.55)
    algo._ppo_update_count = 15
    self.assertAlmostEqual(algo._scheduled_coef(1.0, 0.1, 10), 0.1)

  def test_student_ppo_process_env_step_can_skip_actor_normalizer_update(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    class _Transition:
      def clear(self):
        self.cleared = True

    class _Model:
      def __init__(self):
        self.update_calls = 0
        self.reset_dones = None

      def update_normalization(self, obs):
        self.update_calls += 1

      def reset(self, dones):
        self.reset_dones = dones.clone()

    class _Storage:
      def __init__(self):
        self.added = 0

      def add_transition(self, transition):
        self.added += 1

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.actor = _Model()
    algo.critic = _Model()
    algo.rnd = None
    algo.transition = _Transition()
    algo.storage = _Storage()
    algo.gamma = 0.99
    algo.device = "cpu"
    algo.freeze_actor_obs_normalization = True

    obs = TensorDict({"actor": torch.zeros(2, 4)}, batch_size=[2])
    rewards = torch.ones(2)
    dones = torch.tensor([False, True])
    algo.process_env_step(obs, rewards, dones, extras={})

    self.assertEqual(algo.actor.update_calls, 0)
    self.assertEqual(algo.critic.update_calls, 1)
    self.assertEqual(algo.storage.added, 1)
    self.assertTrue(torch.equal(algo.actor.reset_dones, dones))
    self.assertTrue(torch.equal(algo.critic.reset_dones, dones))
    self.assertTrue(algo.transition.cleared)

  def test_student_ppo_act_is_deterministic_only_for_idle_condition_by_default(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    class _Transition:
      pass

    class _Actor:
      def __init__(self):
        self.mean = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        self.sample = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
        self.logged_actions = None

      def get_hidden_state(self):
        return "actor_h"

      def __call__(self, obs, stochastic_output=False):
        self.output_distribution_params = (self.mean, torch.ones_like(self.mean) * 0.2)
        return self.sample if stochastic_output else self.mean

      @property
      def output_mean(self):
        return self.mean

      def get_output_log_prob(self, actions):
        self.logged_actions = actions.clone()
        return actions.sum(dim=-1)

    class _Critic:
      def get_hidden_state(self):
        return "critic_h"

      def __call__(self, obs):
        return torch.zeros(2, 1)

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.actor = _Actor()
    algo.critic = _Critic()
    algo.transition = _Transition()
    algo.teacher_policy = None
    algo.teacher_every_n_steps = 1
    algo._act_step = 0
    algo.idle_deterministic_actions = True

    obs = TensorDict(
      {"student": torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])},
      batch_size=[2],
    )

    actions = algo.act(obs)

    expected = torch.tensor([[1.0, 1.0], [20.0, 20.0]])
    self.assertTrue(torch.equal(actions, expected))
    self.assertTrue(torch.equal(algo.transition.actions, expected))
    self.assertTrue(torch.equal(algo.actor.logged_actions, expected))
    self.assertTrue(torch.equal(algo.transition.privileged_actions, algo.actor.output_mean))

  def test_student_ppo_can_sample_idle_actions_for_prepare_recovery(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    class _Transition:
      pass

    class _Actor:
      output_distribution_params = (torch.zeros(2, 2), torch.ones(2, 2))

      def get_hidden_state(self):
        return "actor_h"

      def __call__(self, obs, stochastic_output=False):
        del obs
        self.mean = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        return torch.tensor([[10.0, 10.0], [20.0, 20.0]]) if stochastic_output else self.mean

      @property
      def output_mean(self):
        return self.mean

      def get_output_log_prob(self, actions):
        return actions.sum(dim=-1)

    class _Critic:
      def get_hidden_state(self):
        return "critic_h"

      def __call__(self, obs):
        return torch.zeros(2, 1)

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.actor = _Actor()
    algo.critic = _Critic()
    algo.transition = _Transition()
    algo.teacher_policy = None
    algo.teacher_every_n_steps = 1
    algo._act_step = 0
    algo.idle_deterministic_actions = False
    algo.actor_mean_clip = None
    algo.min_action_std = 1.0e-4

    obs = TensorDict(
      {"student": torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])},
      batch_size=[2],
    )

    actions = algo.act(obs)

    expected = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
    self.assertTrue(torch.equal(actions, expected))

  def test_student_ppo_algorithm_loads_offline_bc_regularizer_dataset(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    with tempfile.TemporaryDirectory() as tmp:
      dataset = Path(tmp) / "dataset"
      shards = dataset / "shards"
      shards.mkdir(parents=True)
      torch.save(
        {
          "student_obs": torch.zeros(2, 5, 964),
          "teacher_action": torch.ones(2, 5, 29),
          "valid_mask": torch.ones(2, 5, dtype=torch.bool),
        },
        shards / "shard_000000.pt",
      )

      algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
      algo.device = "cpu"
      algo.offline_bc_coef = 0.1
      algo.offline_bc_batch_size = 3
      algo.offline_bc_seq_len = 4
      algo.offline_bc_cache_shards = 8
      algo._init_offline_bc_dataset(str(dataset))

      obs, actions, masks = algo._sample_offline_bc_batch()

    self.assertEqual(obs.shape, (4, 3, 964))
    self.assertEqual(actions.shape, (4, 3, 29))
    self.assertEqual(masks.shape, (4, 3))
    self.assertTrue(torch.all(masks))

  def test_student_ppo_offline_bc_prefers_active_segments_when_requested(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    with tempfile.TemporaryDirectory() as tmp:
      dataset = Path(tmp) / "dataset"
      shards = dataset / "shards"
      shards.mkdir(parents=True)
      student_obs = torch.zeros(1, 8, 964)
      student_obs[0, :4, -1] = 1.0
      student_obs[0, 4:, -1] = 0.0
      torch.save(
        {
          "student_obs": student_obs,
          "teacher_action": torch.ones(1, 8, 29),
          "valid_mask": torch.ones(1, 8, dtype=torch.bool),
        },
        shards / "shard_000000.pt",
      )

      algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
      algo.device = "cpu"
      algo.offline_bc_coef = 0.1
      algo.offline_idle_bc_coef = 0.0
      algo.offline_bc_batch_size = 3
      algo.offline_bc_seq_len = 2
      algo.offline_bc_cache_shards = 1
      algo.offline_bc_active_fraction = 1.0
      algo._init_offline_bc_dataset(str(dataset))

      obs, _, masks = algo._sample_offline_bc_batch()

    self.assertEqual(obs.shape, (2, 3, 964))
    self.assertTrue(torch.all(obs[masks, -1] == 0.0))

  def test_student_ppo_offline_idle_bc_samples_only_prepare_condition(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    with tempfile.TemporaryDirectory() as tmp:
      dataset = Path(tmp) / "dataset"
      shards = dataset / "shards"
      shards.mkdir(parents=True)
      student_obs = torch.zeros(1, 6, 964)
      student_obs[0, :3, -1] = 1.0
      student_obs[0, 3:, -1] = 0.0
      torch.save(
        {
          "student_obs": student_obs,
          "teacher_action": torch.ones(1, 6, 29),
          "valid_mask": torch.ones(1, 6, dtype=torch.bool),
        },
        shards / "shard_000000.pt",
      )

      algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
      algo.device = "cpu"
      algo.offline_bc_coef = 0.1
      algo.offline_bc_batch_size = 2
      algo.offline_bc_seq_len = 2
      algo.offline_bc_cache_shards = 1
      algo._init_offline_bc_dataset(str(dataset))

      obs, _, masks = algo._sample_offline_bc_batch(idle_only=True)

    self.assertEqual(obs.shape, (2, 2, 964))
    self.assertTrue(torch.all(obs[masks, -1] == 1.0))

  def test_student_ppo_offline_idle_bc_uses_only_initial_idle_prefix(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    with tempfile.TemporaryDirectory() as tmp:
      dataset = Path(tmp) / "dataset"
      shards = dataset / "shards"
      shards.mkdir(parents=True)
      student_obs = torch.zeros(1, 6, 964)
      student_obs[0, :, 0] = torch.arange(6, dtype=torch.float32)
      student_obs[0, [0, 1, 4, 5], -1] = 1.0
      torch.save(
        {
          "student_obs": student_obs,
          "teacher_action": torch.ones(1, 6, 29),
          "valid_mask": torch.ones(1, 6, dtype=torch.bool),
        },
        shards / "shard_000000.pt",
      )

      algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
      algo.device = "cpu"
      algo.offline_bc_coef = 0.1
      algo.offline_bc_batch_size = 1
      algo.offline_bc_seq_len = 4
      algo.offline_bc_cache_shards = 1
      algo._init_offline_bc_dataset(str(dataset))

      obs, _, masks = algo._sample_offline_bc_batch(idle_only=True)

    self.assertEqual(obs.shape, (4, 1, 964))
    self.assertEqual(masks[:, 0].tolist(), [True, True, False, False])
    self.assertTrue(torch.equal(obs[masks, 0], torch.tensor([0.0, 1.0])))

  def test_student_ppo_route_targets_match_moe7_heuristic_and_prepare_class(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    obs = torch.zeros(4, 964)
    obs[:, 27:30] = torch.tensor([
      [2.0, 0.6, 1.8],
      [2.0, -0.6, 2.2],
      [2.0, 0.3, 0.8],
      [3.0, 0.0, 0.1],
    ])
    obs[:, 24:27] = obs[:, 27:30] - torch.tensor([
      [-0.10, 0.00, 0.00],
      [-0.10, 0.00, 0.00],
      [-0.10, 0.00, 0.00],
      [0.00, 0.00, 0.00],
    ])
    obs[:, -1] = torch.tensor([0.0, 0.0, 0.0, 1.0])

    target = GoalkeeperStudentPPO._route_targets_from_student_obs(obs)

    self.assertEqual(target.tolist(), [0, 3, 4, 6])

  def test_student_ppo_route_targets_prefer_learned_gate_when_available(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    class _Gate(torch.nn.Module):
      def forward(self, features):
        logits = torch.zeros(features.shape[0], 7, device=features.device)
        logits[:, 2] = 10.0
        return logits

    obs = torch.zeros(2, 964)
    obs[:, 27:30] = torch.tensor([[2.0, 0.6, 0.8], [3.0, 0.0, 0.1]])
    obs[:, 24:27] = obs[:, 27:30] - torch.tensor([[-0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    obs[:, -1] = torch.tensor([0.0, 1.0])

    target = GoalkeeperStudentPPO._route_targets_from_student_obs(
      obs,
      gate=_Gate(),
      gate_mean=torch.zeros(6),
      gate_std=torch.ones(6),
    )

    self.assertEqual(target.tolist(), [2, 6])

  def test_student_ppo_condition_aux_uses_region_head_with_seven_classes(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    class _Actor:
      condition_aux_output = {
        "region_logits": torch.tensor([
          [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
        ]),
        "ball_latent": torch.zeros(2, 6),
      }

      def _raw_student_obs(self, observations):
        return observations["student"]

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.actor = _Actor()
    algo.device = "cpu"
    algo.condition_aux_active_only = False
    batch = type(
      "Batch",
      (),
      {
        "observations": TensorDict(
          {"student": torch.zeros(2, 964)},
          batch_size=[2],
        ),
        "masks": None,
      },
    )()
    batch.observations["student"][0, 27:30] = torch.tensor([2.0, 0.6, 1.8])
    batch.observations["student"][0, 24:27] = torch.tensor([2.1, 0.6, 1.8])
    batch.observations["student"][1, -1] = 1.0

    loss = algo._condition_aux_loss(batch)

    self.assertLess(loss.item(), 0.1)

  def test_student_ppo_condition_aux_aligns_recurrent_padded_masks(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO
    from rsl_rl.utils import split_and_pad_trajectories

    rollout_obs = TensorDict(
      {"student": torch.zeros(2, 2, 964)},
      batch_size=[2, 2],
    )
    rollout_obs["student"][0, 0, 27:30] = torch.tensor([2.0, 0.6, 1.8])
    rollout_obs["student"][0, 0, 24:27] = torch.tensor([2.1, 0.6, 1.8])
    rollout_obs["student"][1, 0, 27:30] = torch.tensor([2.0, 0.6, 1.8])
    rollout_obs["student"][1, 0, 24:27] = torch.tensor([2.1, 0.6, 1.8])
    rollout_obs["student"][:, 1, -1] = 1.0
    dones = torch.zeros(2, 2, dtype=torch.bool)
    padded_obs, masks = split_and_pad_trajectories(rollout_obs, dones)

    class _Actor:
      condition_aux_output = {
        "region_logits": torch.tensor([
          [[5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]],
          [[5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]],
        ]),
        "ball_latent": torch.zeros(2, 2, 6),
      }

      def _raw_student_obs(self, observations):
        return observations["student"]

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.actor = _Actor()
    algo.device = "cpu"
    algo.condition_aux_active_only = False
    batch = type(
      "Batch",
      (),
      {
        "observations": padded_obs,
        "masks": masks,
      },
    )()

    loss = algo._condition_aux_loss(batch)

    self.assertLess(loss.item(), 0.1)

  def test_student_ppo_offline_bc_runs_only_every_n_updates(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.device = "cpu"
    algo.offline_bc_coef = 0.1
    algo.offline_bc_every_n_updates = 4
    algo._offline_bc_shards = [Path("dummy.pt")]

    calls = 0

    def fake_loss():
      nonlocal calls
      calls += 1
      return torch.tensor(3.0)

    algo._offline_bc_loss = fake_loss

    losses = [algo._scheduled_offline_bc_loss(i).item() for i in range(8)]

    self.assertEqual(losses, [3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    self.assertEqual(calls, 2)

  def test_student_ppo_prefetches_offline_bc_shards_before_sampling(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    with tempfile.TemporaryDirectory() as tmp:
      dataset = Path(tmp) / "dataset"
      shards = dataset / "shards"
      shards.mkdir(parents=True)
      for index in range(3):
        torch.save(
          {
            "student_obs": torch.full((2, 5, 964), float(index)),
            "teacher_action": torch.full((2, 5, 29), float(index)),
            "valid_mask": torch.ones(2, 5, dtype=torch.bool),
          },
          shards / f"shard_{index:06d}.pt",
        )

      algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
      algo.device = "cpu"
      algo.offline_bc_coef = 0.1
      algo.offline_bc_batch_size = 3
      algo.offline_bc_seq_len = 4
      algo.offline_bc_cache_shards = 2
      algo._init_offline_bc_dataset(str(dataset))
      algo._prefetch_offline_bc_shards()

      self.assertEqual(len(algo._offline_bc_active_shards), 2)
      self.assertEqual(len(algo._offline_bc_cache), 2)
      with patch("torch.load", side_effect=AssertionError("sampling should use the prefetched shard cache")):
        obs, actions, masks = algo._sample_offline_bc_batch()

    self.assertEqual(obs.shape, (4, 3, 964))
    self.assertEqual(actions.shape, (4, 3, 29))
    self.assertTrue(torch.all(masks))


if __name__ == "__main__":
  unittest.main()
