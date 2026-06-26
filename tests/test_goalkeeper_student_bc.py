"""Tests for position-conditioned goalkeeper student distillation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tensordict import TensorDict
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401


class GoalkeeperStudentBcTest(unittest.TestCase):
  def test_student_obs_appends_normalized_prediction_condition_to_teacher_actor_obs(self):
    from src.tasks.soccer.mdp.goalkeeper_student_obs import (
      GOALKEEPER_TASK_CONTEXT_DIM,
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
    self.assertEqual(student_obs[:, GOALKEEPER_TEACHER_ACTOR_OBS_DIM:].shape[-1], GOALKEEPER_TASK_CONTEXT_DIM)
    self.assertTrue(torch.equal(student_obs[:, GOALKEEPER_TEACHER_ACTOR_OBS_DIM], 1.0 - condition[:, 3]))
    self.assertTrue(torch.equal(student_obs[:, GOALKEEPER_TEACHER_ACTOR_OBS_DIM + 1], condition[:, 3]))
    self.assertTrue(torch.equal(student_obs[:, GOALKEEPER_TEACHER_ACTOR_OBS_DIM + 2], 1.0 - condition[:, 3]))
    self.assertTrue(torch.equal(student_obs[:, GOALKEEPER_TEACHER_ACTOR_OBS_DIM + 10:GOALKEEPER_TEACHER_ACTOR_OBS_DIM + 12], condition[:, :2]))
    self.assertTrue(torch.equal(student_obs[:, GOALKEEPER_TEACHER_ACTOR_OBS_DIM + 3], condition[:, 2]))

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

  def test_lstm_student_actor_uses_shared_recurrent_trunk_with_phase_heads(self):
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
    self.assertFalse(hasattr(actor, "prepare_rnn"))
    self.assertFalse(hasattr(actor, "active_rnn"))
    self.assertTrue(hasattr(actor, "phase_encoder"))
    self.assertTrue(hasattr(actor, "ball_encoder"))
    self.assertTrue(hasattr(actor, "target_encoder"))
    self.assertTrue(hasattr(actor, "context_fusion"))
    self.assertTrue(hasattr(actor, "region_estimator"))
    self.assertTrue(hasattr(actor, "ball_estimator"))
    self.assertTrue(hasattr(actor, "prepare_head"))
    self.assertTrue(hasattr(actor, "active_head"))
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

  def test_goalkeeper_bc_collate_can_prefix_active_episode_with_prepare_sequence(self):
    from scripts.train_goalkeeper_student_bc import _collate_padded
    from src.tasks.soccer.mdp.goalkeeper_student_obs import (
      GOALKEEPER_PHASE_ACTIVE_INDEX,
      GOALKEEPER_PHASE_IDLE_INDEX,
      GOALKEEPER_STUDENT_OBS_DIM,
      GOALKEEPER_TEACHER_ACTOR_OBS_DIM,
    )

    prefix_obs = torch.zeros(4, GOALKEEPER_STUDENT_OBS_DIM)
    prefix_obs[:, :GOALKEEPER_TEACHER_ACTOR_OBS_DIM] = 3.0
    prefix_obs[:, GOALKEEPER_PHASE_IDLE_INDEX] = 1.0
    prefix_act = torch.full((4, 2), 0.25)

    active_obs = torch.zeros(3, GOALKEEPER_STUDENT_OBS_DIM)
    active_obs[:, :GOALKEEPER_TEACHER_ACTOR_OBS_DIM] = -1.0
    active_obs[:, GOALKEEPER_PHASE_ACTIVE_INDEX] = 1.0
    active_act = torch.full((3, 2), -0.5)

    obs, actions, masks = _collate_padded(
      [(active_obs, active_act, prefix_obs, prefix_act)]
    )

    self.assertEqual(obs.shape, (7, 1, GOALKEEPER_STUDENT_OBS_DIM))
    self.assertTrue(torch.all(masks))
    self.assertTrue(torch.all(obs[:4, 0, GOALKEEPER_PHASE_IDLE_INDEX] == 1.0))
    self.assertTrue(torch.all(obs[4:, 0, GOALKEEPER_PHASE_ACTIVE_INDEX] == 1.0))
    self.assertTrue(torch.all(actions[:4, 0] == 0.25))
    self.assertTrue(torch.all(actions[4:, 0] == -0.5))

  def test_goalkeeper_bc_dataset_samples_prepare_prefix_for_active_episode(self):
    from scripts.train_goalkeeper_student_bc import _ShardIterableDataset
    from src.tasks.soccer.mdp.goalkeeper_student_obs import (
      GOALKEEPER_PHASE_ACTIVE_INDEX,
      GOALKEEPER_PHASE_IDLE_INDEX,
      GOALKEEPER_STUDENT_OBS_DIM,
      GOALKEEPER_TEACHER_ACTOR_OBS_DIM,
    )

    with tempfile.TemporaryDirectory() as tmp:
      shard = Path(tmp) / "shard_000000.pt"
      obs = torch.zeros(2, 5, GOALKEEPER_STUDENT_OBS_DIM)
      obs[0, :, :GOALKEEPER_TEACHER_ACTOR_OBS_DIM] = -1.0
      obs[0, :, GOALKEEPER_PHASE_ACTIVE_INDEX] = 1.0
      obs[1, :, :GOALKEEPER_TEACHER_ACTOR_OBS_DIM] = 3.0
      obs[1, :, GOALKEEPER_PHASE_IDLE_INDEX] = 1.0
      actions = torch.stack([
        torch.full((5, 2), -0.5),
        torch.full((5, 2), 0.25),
      ])
      torch.save(
        {
          "student_obs": obs,
          "teacher_action": actions,
          "valid_mask": torch.ones(2, 5, dtype=torch.bool),
          "metadata": {"success": torch.ones(2, dtype=torch.bool)},
        },
        shard,
      )

      dataset = _ShardIterableDataset(
        [shard],
        success_only=False,
        action_clip=None,
        seed=1,
        transition_prefix_prob=1.0,
        transition_prefix_min_steps=3,
        transition_prefix_max_steps=3,
      )
      samples = list(dataset)

    prefixed = [sample for sample in samples if len(sample) == 4]
    self.assertTrue(prefixed)
    active_obs, active_act, prefix_obs, prefix_act = prefixed[0]
    self.assertEqual(prefix_obs.shape[0], 3)
    self.assertTrue(torch.all(prefix_obs[:, GOALKEEPER_PHASE_IDLE_INDEX] == 1.0))
    self.assertTrue(torch.all(active_obs[:, GOALKEEPER_PHASE_ACTIVE_INDEX] == 1.0))
    self.assertTrue(torch.all(prefix_act == 0.25))
    self.assertTrue(torch.all(active_act == -0.5))

  def test_goalkeeper_student_task_context_routes_active_and_prepare_heads(self):
    from scripts.train_goalkeeper_student_bc import BcConfig, make_goalkeeper_student_actor
    from src.tasks.soccer.mdp.goalkeeper_student_obs import (
      GOALKEEPER_STUDENT_OBS_DIM,
      GOALKEEPER_TEACHER_ACTOR_OBS_DIM,
    )

    cfg = BcConfig(dataset_dir="unused", obs_normalization=False, batch_size=2)
    actor = make_goalkeeper_student_actor(GOALKEEPER_STUDENT_OBS_DIM, 29, cfg, "cpu")
    actor.eval()

    obs = torch.zeros(2, GOALKEEPER_STUDENT_OBS_DIM)
    obs[:, GOALKEEPER_TEACHER_ACTOR_OBS_DIM] = torch.tensor([0.0, 1.0])
    obs[:, GOALKEEPER_TEACHER_ACTOR_OBS_DIM + 1] = torch.tensor([1.0, 0.0])
    obs[:, GOALKEEPER_TEACHER_ACTOR_OBS_DIM + 2] = torch.tensor([0.0, 1.0])
    obs[:, GOALKEEPER_TEACHER_ACTOR_OBS_DIM + 3] = torch.tensor([0.0, 1.0])
    with torch.no_grad():
      for param in actor.prepare_head.parameters():
        param.zero_()
      for param in actor.active_head.parameters():
        param.zero_()
      actor.prepare_head[-1].bias.fill_(1.5)
      actor.active_head[-1].bias.fill_(-2.0)

    td = TensorDict({"student": obs}, batch_size=[2])
    action = actor(td)

    self.assertTrue(torch.allclose(action[0], torch.full_like(action[0], 1.5), atol=1e-5))
    self.assertTrue(torch.allclose(action[1], torch.full_like(action[1], -2.0), atol=1e-5))

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

  def test_goalkeeper_bc_validation_schedule_runs_interval_and_final_epoch(self):
    from scripts.train_goalkeeper_student_bc import BcConfig, _should_run_validation

    cfg = BcConfig(dataset_dir="unused", epochs=25, val_interval=10)

    self.assertFalse(_should_run_validation(1, cfg))
    self.assertTrue(_should_run_validation(10, cfg))
    self.assertTrue(_should_run_validation(20, cfg))
    self.assertTrue(_should_run_validation(25, cfg))

  def test_goalkeeper_bc_epoch_shard_sampling_uses_small_epoch_subset(self):
    from scripts.train_goalkeeper_student_bc import BcConfig, _select_epoch_shards

    shards = [Path(f"shard_{idx}.pt") for idx in range(10)]
    cfg = BcConfig(dataset_dir="unused", train_shards_per_epoch=3)

    first = _select_epoch_shards(shards, cfg.train_shards_per_epoch, cfg.seed, epoch=1, shuffle=True)
    second = _select_epoch_shards(shards, cfg.train_shards_per_epoch, cfg.seed, epoch=2, shuffle=True)

    self.assertEqual(len(first), 3)
    self.assertEqual(len(second), 3)
    self.assertNotEqual(first, shards)
    self.assertNotEqual(first, second)
    self.assertTrue(set(first).issubset(set(shards)))

  def test_goalkeeper_bc_validation_honors_max_val_batches(self):
    from scripts.train_goalkeeper_student_bc import BcConfig, _evaluate_goalkeeper_bc

    class _Actor:
      region_condition_output = None

      def eval(self):
        pass

    def _fake_forward(actor, obs_chunk, mask_chunk, hidden_state=None):
      return torch.zeros(int(mask_chunk.sum().item()), 2), hidden_state

    batches = [
      (torch.zeros(1, 1, 964), torch.ones(1, 1, 2), torch.ones(1, 1, dtype=torch.bool))
      for _ in range(3)
    ]
    cfg = BcConfig(dataset_dir="unused", max_val_batches=1, obs_normalization=False)
    with (
      patch("scripts.train_goalkeeper_student_bc._make_dataloader", return_value=batches),
      patch("scripts.train_shooter_bc._forward_rnn_chunk", side_effect=_fake_forward),
    ):
      val = _evaluate_goalkeeper_bc(_Actor(), [Path("val.pt")], cfg, "cpu", route_gate=None)

    self.assertEqual(val["batches"], 1)
    self.assertEqual(val["samples"], 2.0)

  def test_goalkeeper_bc_weighted_action_loss_prioritizes_active_frames(self):
    from scripts.train_goalkeeper_student_bc import BcConfig, _weighted_goalkeeper_action_loss

    pred = torch.zeros(2, 1)
    target = torch.ones(2, 1)
    obs = torch.zeros(2, 964)
    obs[0, -1] = 1.0
    obs[1, -1] = 0.0
    cfg = BcConfig(
      dataset_dir="unused",
      loss="mse",
      active_action_loss_weight=3.0,
    )

    loss, stats = _weighted_goalkeeper_action_loss(pred, target, obs, cfg)

    self.assertAlmostEqual(float(loss.item()), 1.0)
    self.assertEqual(stats["active_action_values"], 1.0)
    self.assertEqual(stats["idle_action_values"], 1.0)
    self.assertEqual(stats["active_action_loss_sum"], 1.0)
    self.assertEqual(stats["idle_action_loss_sum"], 1.0)

  def test_goalkeeper_bc_validation_reports_active_and_idle_losses(self):
    from scripts.train_goalkeeper_student_bc import BcConfig, _evaluate_goalkeeper_bc

    class _Actor:
      region_condition_output = None

      def eval(self):
        pass

    def _fake_forward(actor, obs_chunk, mask_chunk, hidden_state=None):
      del actor, obs_chunk, hidden_state
      return torch.zeros(int(mask_chunk.sum().item()), 1), None

    obs = torch.zeros(2, 1, 964)
    obs[0, 0, -1] = 1.0
    obs[1, 0, -1] = 0.0
    actions = torch.ones(2, 1, 1)
    masks = torch.ones(2, 1, dtype=torch.bool)
    cfg = BcConfig(dataset_dir="unused", loss="mse", max_val_batches=1, obs_normalization=False)
    with (
      patch("scripts.train_goalkeeper_student_bc._make_dataloader", return_value=[(obs, actions, masks)]),
      patch("scripts.train_shooter_bc._forward_rnn_chunk", side_effect=_fake_forward),
    ):
      val = _evaluate_goalkeeper_bc(_Actor(), [Path("val.pt")], cfg, "cpu", route_gate=None)

    self.assertEqual(val["active_action_values"], 1.0)
    self.assertEqual(val["idle_action_values"], 1.0)
    self.assertAlmostEqual(val["active_action_loss"], 1.0)
    self.assertAlmostEqual(val["idle_action_loss"], 1.0)

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

  def test_student_ppo_actor_is_about_one_million_parameters_with_shared_rnn(self):
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
    self.assertLessEqual(param_count, 1_300_000)
    self.assertEqual(actor.rnn.rnn.hidden_size, 160)

  def test_student_ppo_reference_kl_phase_weights_idle_and_active_separately(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    class _Actor:
      def __init__(self, mean, std):
        self.output_distribution_params = (torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32))

      def get_kl_divergence(self, old_params, new_params):
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = torch.distributions.Normal(old_mean, old_std)
        new_dist = torch.distributions.Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

    class _ReferenceActor:
      def __call__(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del obs, masks, hidden_state, stochastic_output
        self.output_distribution_params = (
          torch.zeros(2, 2),
          torch.ones(2, 2),
        )

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.actor = _Actor([[1.0, 0.0], [1.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]])
    algo.reference_actor = _ReferenceActor()
    algo.device = "cpu"
    algo.reference_kl_coef = 0.1
    algo.reference_kl_idle_coef = 0.1
    algo.reference_kl_active_coef = 0.03
    algo.reference_kl_std_floor = 0.1
    batch = type(
      "Batch",
      (),
      {
        "observations": TensorDict({"student": torch.zeros(2, 4)}, batch_size=[2]),
        "masks": None,
        "hidden_states": (None, None),
        "actor_loss_mask": torch.tensor([0.0, 1.0]),
      },
    )()

    raw = algo._reference_kl_loss(batch)
    weighted = algo._reference_kl_loss(batch, phase_weighted=True)

    self.assertAlmostEqual(raw.item(), 0.5)
    self.assertAlmostEqual(weighted.item(), 0.0325)

  def test_goalkeeper_prior_dataset_uses_success_rollout_frames_without_idle_filter(self):
    from src.tasks.soccer.modules.goalkeeper_prior_discriminator import GoalkeeperPriorDataset

    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      shard_dir = root / "shards"
      shard_dir.mkdir()
      obs = torch.zeros(2, 3, 964)
      obs[0, :, -1] = torch.tensor([1.0, 0.0, 1.0])
      obs[1, :, -1] = torch.tensor([0.0, 0.0, 0.0])
      actions = torch.arange(2 * 3 * 29, dtype=torch.float32).reshape(2, 3, 29)
      torch.save(
        {
          "student_obs": obs,
          "teacher_action": actions,
          "valid_mask": torch.ones(2, 3, dtype=torch.bool),
          "metadata": {"success": torch.tensor([True, False])},
        },
        shard_dir / "shard_000000.pt",
      )

      dataset = GoalkeeperPriorDataset(str(root), device="cpu", max_samples=8)
      samples = dataset.sample(8)

    self.assertEqual(dataset.num_samples, 3)
    self.assertEqual(samples.shape[1], 964 + 29)
    self.assertEqual(samples.shape[0], 8)
    # The successful episode has two idle frames and one active frame; all are kept.
    self.assertEqual(int((dataset.features[:, 963] > 0.5).sum().item()), 2)

  def test_goalkeeper_prior_discriminator_reward_is_nonnegative_and_clipped(self):
    from src.tasks.soccer.modules.goalkeeper_prior_discriminator import GoalkeeperPriorDiscriminator

    disc = GoalkeeperPriorDiscriminator(input_dim=3, hidden_dims=(4,), reward_clip=0.7)
    with torch.no_grad():
      for param in disc.parameters():
        param.zero_()

    reward = disc.reward(torch.ones(5, 3))

    self.assertTrue(torch.all(reward >= 0.0))
    self.assertTrue(torch.all(reward <= 0.7))
    self.assertEqual(reward.shape, (5,))

  def test_student_ppo_prior_discriminator_config_is_forwarded_to_algorithm(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(
      prior_disc_dataset_dir="data/prior",
      prior_disc_reward_coef=0.25,
      prior_disc_updates=2,
      prior_disc_batch_size=128,
    )
    train_cfg = build_train_config(cfg)
    alg = train_cfg.agent.algorithm

    self.assertEqual(alg.prior_disc_dataset_dir, "data/prior")
    self.assertEqual(alg.prior_disc_reward_coef, 0.25)
    self.assertEqual(alg.prior_disc_updates, 2)
    self.assertEqual(alg.prior_disc_batch_size, 128)

  def test_student_ppo_active_bc_config_is_forwarded_to_algorithm(self):
    from dataclasses import asdict

    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(
      active_bc_dataset_dir="data/active_bc",
      active_bc_coef=0.2,
      active_bc_final_coef=0.05,
      active_bc_anneal_updates=2000,
      active_bc_batch_size=64,
    )
    train_cfg = build_train_config(cfg)
    alg = train_cfg.agent.algorithm

    self.assertEqual(alg.active_bc_dataset_dir, "data/active_bc")
    self.assertEqual(alg.active_bc_coef, 0.2)
    self.assertEqual(alg.active_bc_final_coef, 0.05)
    self.assertEqual(alg.active_bc_anneal_updates, 2000)
    self.assertEqual(alg.active_bc_batch_size, 64)
    self.assertEqual(asdict(train_cfg.agent)["algorithm"]["active_bc_dataset_dir"], "data/active_bc")

  def test_student_ppo_motion_prior_reward_does_not_enable_rl_bc_loss(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(
      motion_prior_weight=0.35,
      motion_prior_dir="src/assets/soccer/motions/goalkeeper",
      motion_prior_names=("leftjump.pt", "rightjump.pt"),
      motion_prior_route_mode="region",
      motion_prior_std=0.4,
    )
    train_cfg = build_train_config(cfg)

    reward = train_cfg.env.rewards["motion_prior_joint_pose"]
    self.assertEqual(reward.weight, 0.35)
    self.assertEqual(reward.params["motion_dir"], "src/assets/soccer/motions/goalkeeper")
    self.assertEqual(reward.params["motion_names"], ("leftjump.pt", "rightjump.pt"))
    self.assertEqual(reward.params["route_mode"], "region")
    self.assertEqual(reward.params["std"], 0.4)
    self.assertEqual(train_cfg.agent.algorithm.active_bc_coef, 0.0)
    self.assertIsNone(train_cfg.agent.algorithm.active_bc_dataset_dir)

  def test_student_ppo_idle_action_rate_weight_can_be_overridden(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(profile="teacher_kl", idle_action_rate_weight=-0.1)

    train_cfg = build_train_config(cfg)

    self.assertEqual(train_cfg.env.rewards["idle_action_rate"].weight, -0.1)

  def test_student_ppo_idle_joint_pose_weight_can_be_overridden(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(profile="teacher_kl", idle_joint_pose_weight=-0.05)

    train_cfg = build_train_config(cfg)

    self.assertEqual(train_cfg.env.rewards["idle_joint_pose"].weight, -0.05)

  def test_student_ppo_prior_discriminator_reward_applies_only_active_samples(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    class _Norm:
      def update_normalization(self, obs):
        del obs

      def reset(self, dones):
        del dones

    class _Storage:
      def add_transition(self, transition):
        self.rewards = transition.rewards.clone()

    class _Prior:
      def reward(self, features):
        return torch.ones(features.shape[0]) * 2.0

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.freeze_actor_obs_normalization = True
    algo.actor = _Norm()
    algo.critic = _Norm()
    algo.rnd = None
    algo.transition = type("Transition", (), {"values": torch.zeros(2, 1), "clear": lambda self: None})()
    algo.storage = _Storage()
    algo.device = "cpu"
    algo.gamma = 0.99
    algo.prior_disc_reward_coef = 0.5
    algo.prior_discriminator = _Prior()
    algo.env = None
    obs = TensorDict({"student": torch.zeros(2, 964)}, batch_size=[2])
    obs["student"][:, -1] = torch.tensor([1.0, 0.0])
    algo.transition.actions = torch.zeros(2, 29)

    algo.process_env_step(obs, torch.zeros(2), torch.zeros(2, dtype=torch.bool), {})

    self.assertTrue(torch.allclose(algo.storage.rewards, torch.tensor([0.0, 1.0])))

  def test_student_ppo_prior_discriminator_update_separates_expert_and_policy_samples(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO
    from src.tasks.soccer.modules.goalkeeper_prior_discriminator import GoalkeeperPriorDiscriminator

    class _Dataset:
      def sample(self, batch_size):
        return torch.ones(batch_size, 3)

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.prior_discriminator = GoalkeeperPriorDiscriminator(input_dim=3, hidden_dims=(4,))
    algo.prior_dataset = _Dataset()
    algo.prior_disc_optimizer = torch.optim.Adam(algo.prior_discriminator.parameters(), lr=1.0e-3)
    algo.prior_disc_batch_size = 4
    algo.prior_disc_updates = 2
    algo.device = "cpu"
    policy_features = torch.zeros(8, 3)

    metrics = algo._update_prior_discriminator(policy_features)

    self.assertIn("prior_disc", metrics)
    self.assertIn("prior_disc_expert_acc", metrics)
    self.assertIn("prior_disc_policy_acc", metrics)
    self.assertTrue(torch.isfinite(torch.tensor(metrics["prior_disc"])))

  def test_student_ppo_active_bc_loss_supervises_actor_mean_with_expert_actions(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    class _Actor(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.obs_groups = ("actor",)
        self.bias = torch.nn.Parameter(torch.zeros(2))
        self.output_distribution_params = (torch.zeros(3, 2), torch.ones(3, 2))

      def forward(self, obs, stochastic_output=False):
        del stochastic_output
        mean = obs["actor"][:, :2] + self.bias
        self.output_distribution_params = (mean, torch.ones_like(mean) * 0.1)
        return mean

    class _Dataset:
      input_dim = 4

      def sample(self, batch_size):
        del batch_size
        obs = torch.tensor(
          [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
          ]
        )
        actions = torch.tensor(
          [
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
          ]
        )
        return torch.cat([obs, actions], dim=-1)

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.actor = _Actor()
    algo.active_bc_dataset = _Dataset()
    algo.active_bc_coef = 1.0
    algo.active_bc_batch_size = 3
    algo.active_bc_obs_dim = 2
    algo.device = "cpu"
    optimizer = torch.optim.SGD(algo.actor.parameters(), lr=0.5)

    initial_loss = algo._active_bc_loss()
    optimizer.zero_grad()
    initial_loss.backward()
    optimizer.step()
    updated_loss = algo._active_bc_loss()

    self.assertGreater(initial_loss.item(), 0.0)
    self.assertLess(updated_loss.item(), initial_loss.item())

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

  def test_student_ppo_bc_finetune_freezes_actor_obs_normalizer_updates(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(
      init="checkpoints/model_best.pt",
      profile="bc_regularized_finetune",
    )
    train_cfg = build_train_config(cfg)

    self.assertTrue(train_cfg.agent.algorithm.freeze_actor_obs_normalization)

  def test_student_ppo_scratch_defaults_do_not_freeze_actor_obs_normalizer(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(init=None, profile="default")
    train_cfg = build_train_config(cfg)

    self.assertFalse(train_cfg.agent.algorithm.freeze_actor_obs_normalization)

  def test_student_ppo_teacher_kl_defaults_are_looser_for_policy_updates(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(profile="teacher_kl")
    train_cfg = build_train_config(cfg)
    alg = train_cfg.agent.algorithm

    self.assertEqual(alg.teacher_kl_coef, 0.003)
    self.assertEqual(alg.min_action_std, 0.03)
    self.assertEqual(alg.max_action_std, 0.15)
    self.assertFalse(alg.idle_deterministic_actions)
    self.assertEqual(alg.condition_aux_coef, 0.01)
    self.assertEqual(alg.condition_aux_final_coef, 0.01)
    self.assertTrue(alg.condition_aux_active_only)

  def test_student_ppo_teacher_kl_allows_explicit_zero_coef_override(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(profile="teacher_kl", teacher_kl_coef=0.0)
    train_cfg = build_train_config(cfg)

    self.assertEqual(train_cfg.agent.algorithm.teacher_kl_coef, 0.0)

  def test_student_ppo_teacher_kl_profile_relaxes_prepare_motion_penalties(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(profile="teacher_kl")
    train_cfg = build_train_config(cfg)
    rewards = train_cfg.env.rewards

    self.assertEqual(rewards["idle_base_still"].weight, -0.05)
    self.assertEqual(rewards["idle_leg_ready_pose"].weight, 0.0)
    self.assertEqual(rewards["idle_joint_pose"].weight, 0.0)
    self.assertEqual(rewards["idle_action_rate"].weight, -0.005)

  def test_student_ppo_teacher_kl_profile_prioritizes_block_outcomes_over_idle_shaping(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(profile="teacher_kl")
    train_cfg = build_train_config(cfg)
    rewards = train_cfg.env.rewards

    self.assertEqual(rewards["goal_conceded"].weight, -20.0)
    self.assertEqual(rewards["intercept"].weight, 1.0)
    self.assertEqual(rewards["condition_target_reach"].weight, 0.5)
    self.assertEqual(rewards["body"].weight, 1.0)
    self.assertEqual(rewards["stop_ball"].weight, 3.0)
    self.assertEqual(rewards["idle_base_height_band"].weight, 5.0)
    self.assertEqual(rewards["idle_upright"].weight, 2.0)
    self.assertEqual(rewards["idle_alive"].weight, 0.1)
    self.assertEqual(rewards["active_upright"].weight, 2.0)

  def test_student_ppo_teacher_kl_eval_interval_saves_best_block_rate_checkpoint(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, TrainConfig, _run_teacher_kl_training

    @dataclass
    class _FakeEnvCfg:
      seed: int = 0

    @dataclass
    class _FakeAgentCfg:
      seed: int = 0
      max_iterations: int = 5
      experiment_name: str = "tmp_goalkeeper_tests"
      run_name: str = "unit_test_teacher_kl_eval"
      clip_actions: float | None = None

    class _FakeAlg:
      def __init__(self):
        self.actor = torch.nn.Linear(1, 1)

      def save(self):
        return {"actor_state_dict": self.actor.state_dict()}

      def _clamp_actor_std(self):
        return None

    class _FakeRunner:
      def __init__(self, env, agent_cfg, log_dir, device):
        self.env = env
        self.log_dir = log_dir
        self.alg = _FakeAlg()
        self.learn_calls = []
        self.policy = SimpleNamespace(
          training=False,
          eval=lambda: None,
          train=lambda: None,
          reset=lambda: None,
        )

      def add_git_repo_to_log(self, _path):
        return None

      def get_inference_policy(self, device=None):
        return self.policy

      def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self.learn_calls.append(num_learning_iterations)

    fake_env = SimpleNamespace(close=lambda: None)
    cfg = FineTuneConfig(
      profile="teacher_kl",
      init="model_0.pt",
      run_name="unit_test_teacher_kl_eval",
      max_iterations=5,
      eval_interval=2,
      eval_steps=10,
      eval_resets=1,
      device_ids=[0],
    )
    train_cfg = TrainConfig(
      env=_FakeEnvCfg(),
      agent=_FakeAgentCfg(),
      load_checkpoint_path="model_0.pt",
      load_actor_only=True,
      load_model_only=False,
      actor_std_override=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir, \
      patch("scripts.train_goalkeeper_student_ppo.configure_torch_backends"), \
      patch("scripts.train_goalkeeper_student_ppo.torch.cuda.is_available", return_value=False), \
      patch("scripts.train_goalkeeper_student_ppo.Path", wraps=Path), \
      patch("scripts.train_goalkeeper_student_ppo.ManagerBasedRlEnv", return_value=fake_env), \
      patch("scripts.train_goalkeeper_student_ppo.RslRlVecEnvWrapper", side_effect=lambda env, clip_actions=None: env), \
      patch("scripts.train_goalkeeper_student_ppo.load_runner_cls", return_value=_FakeRunner), \
      patch("scripts.train_goalkeeper_student_ppo._load_initial_checkpoint"), \
      patch("scripts.train_goalkeeper_student_ppo.dump_yaml"), \
      patch("scripts.train_goalkeeper_student_ppo._eval_goalkeeper_diagnostics", side_effect=[
        {"block_rate": 0.10, "prelaunch_fall_rate": 0.70, "active_fall_rate": 0.80},
        {"block_rate": 0.30, "prelaunch_fall_rate": 0.50, "active_fall_rate": 0.60},
        {"block_rate": 0.20, "prelaunch_fall_rate": 0.40, "active_fall_rate": 0.50},
        {"block_rate": 0.35, "prelaunch_fall_rate": 0.20, "active_fall_rate": 0.30},
      ]), \
      patch("scripts.train_goalkeeper_student_ppo._save_runner_checkpoint") as save_ckpt, \
      patch("scripts.train_goalkeeper_student_ppo.datetime") as fake_datetime:
      fake_datetime.now.return_value.strftime.return_value = "2026-06-25_23-59-59"
      cwd = Path.cwd()
      try:
        import os
        os.chdir(tmpdir)
        _run_teacher_kl_training(cfg.task_id, cfg, train_cfg)
      finally:
        os.chdir(cwd)

    saved_paths = [call.args[1].name for call in save_ckpt.call_args_list]
    self.assertIn("model_best.pt", saved_paths)
    self.assertGreaterEqual(saved_paths.count("model_best.pt"), 3)

  def test_student_ppo_teacher_kl_resume_wandb_writes_metadata_file(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, TrainConfig, _run_teacher_kl_training

    @dataclass
    class _FakeEnvCfg:
      seed: int = 0

    @dataclass
    class _FakeAgentCfg:
      seed: int = 0
      max_iterations: int = 0
      experiment_name: str = "tmp_goalkeeper_tests"
      run_name: str = "unit_test_teacher_kl_wandb_resume"
      clip_actions: float | None = None

    class _FakeAlg:
      def __init__(self):
        self.actor = torch.nn.Linear(1, 1)

      def save(self):
        return {"actor_state_dict": self.actor.state_dict()}

      def _clamp_actor_std(self):
        return None

    class _FakeRunner:
      def __init__(self, env, agent_cfg, log_dir, device):
        self.alg = _FakeAlg()

      def add_git_repo_to_log(self, _path):
        return None

      def get_inference_policy(self, device=None):
        return SimpleNamespace(training=False, eval=lambda: None, train=lambda: None, reset=lambda: None)

      def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
      prev_run = Path(tmpdir) / "logs" / "rsl_rl" / "g1_goalkeeper_student" / "old_run"
      (prev_run / "params").mkdir(parents=True)
      (prev_run / "params" / "wandb.yaml").write_text("id: resume123\nresume: must\n")
      ckpt = prev_run / "model_600.pt"
      ckpt.write_bytes(b"")
      cfg = FineTuneConfig(
        profile="teacher_kl",
        init=str(ckpt),
        run_name="unit_test_teacher_kl_wandb_resume",
        max_iterations=0,
        eval_interval=0,
        resume_wandb=True,
        device_ids=[0],
      )
      train_cfg = TrainConfig(
        env=_FakeEnvCfg(),
        agent=_FakeAgentCfg(),
        load_checkpoint_path=str(ckpt),
        load_actor_only=True,
        load_model_only=False,
        actor_std_override=None,
      )

      with patch("scripts.train_goalkeeper_student_ppo.configure_torch_backends"), \
        patch("scripts.train_goalkeeper_student_ppo.torch.cuda.is_available", return_value=False), \
        patch("scripts.train_goalkeeper_student_ppo.ManagerBasedRlEnv", return_value=SimpleNamespace(close=lambda: None)), \
        patch("scripts.train_goalkeeper_student_ppo.RslRlVecEnvWrapper", side_effect=lambda env, clip_actions=None: env), \
        patch("scripts.train_goalkeeper_student_ppo.load_runner_cls", return_value=_FakeRunner), \
        patch("scripts.train_goalkeeper_student_ppo._load_initial_checkpoint"), \
        patch.dict("os.environ", {}, clear=True):
        cwd = Path.cwd()
        try:
          import os
          os.chdir(tmpdir)
          _run_teacher_kl_training(cfg.task_id, cfg, train_cfg)
        finally:
          os.chdir(cwd)

      new_runs = sorted((Path(tmpdir) / "logs" / "rsl_rl" / "tmp_goalkeeper_tests").glob("*_unit_test_teacher_kl_wandb_resume"))
      self.assertEqual(len(new_runs), 1)
      wandb_meta = new_runs[0] / "params" / "wandb.yaml"
      self.assertTrue(wandb_meta.exists())
      text = wandb_meta.read_text()
      self.assertIn("id: resume123", text)

  def test_student_ppo_script_can_resume_wandb_from_explicit_id(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, _resolve_wandb_resume

    cfg = FineTuneConfig(
      init="logs/rsl_rl/g1_goalkeeper_student/2026-06-25_23-35-33_run/model_600.pt",
      wandb_resume_id="abc12345",
    )

    with patch.dict("os.environ", {}, clear=True):
      meta = _resolve_wandb_resume(cfg)

    self.assertEqual(meta["id"], "abc12345")
    self.assertEqual(meta["resume"], "must")
    self.assertEqual(meta["source"], "explicit")
    self.assertEqual(meta["dir"], "")

  def test_student_ppo_script_can_resume_wandb_from_init_run_metadata(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, _resolve_wandb_resume

    with tempfile.TemporaryDirectory() as tmpdir:
      run_dir = Path(tmpdir) / "logs" / "rsl_rl" / "g1_goalkeeper_student" / "2026-06-25_23-35-33_run"
      params_dir = run_dir / "params"
      params_dir.mkdir(parents=True)
      (run_dir / "model_600.pt").write_bytes(b"")
      (params_dir / "wandb.yaml").write_text("id: oldrun42\nresume: must\n")
      cfg = FineTuneConfig(
        init=str(run_dir / "model_600.pt"),
        resume_wandb=True,
      )

      with patch.dict("os.environ", {}, clear=True):
        meta = _resolve_wandb_resume(cfg)

    self.assertEqual(meta["id"], "oldrun42")
    self.assertEqual(meta["resume"], "must")
    self.assertEqual(meta["source"], "init")
    self.assertEqual(meta["dir"], str(run_dir))

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
    algo.idle_actor_loss_weight = 0.0
    algo.clip_param = 0.1
    algo.actor = type("Actor", (), {"_raw_student_obs": lambda _self, obs: obs["student"]})()
    observations = {"student": torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.2, 0.1, 0.3, 0.0]])}
    advantages = torch.tensor([100.0, 1.0])
    ratio = torch.ones(2)
    entropy = torch.tensor([1000.0, 2.0])

    surrogate_loss, entropy_loss = algo._masked_actor_losses(observations, advantages, ratio, entropy)

    self.assertTrue(torch.allclose(surrogate_loss, torch.tensor(-1.0)))
    self.assertTrue(torch.allclose(entropy_loss, torch.tensor(2.0)))

  def test_student_ppo_can_keep_small_prepare_policy_loss_weight(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.mask_idle_actor_loss = True
    algo.idle_actor_loss_weight = 0.2
    algo.clip_param = 0.1
    batch = type("Batch", (), {})()
    batch.actor_loss_mask = torch.tensor([0.0, 1.0])
    advantages = torch.tensor([10.0, 2.0])
    ratio = torch.ones(2)
    entropy = torch.tensor([5.0, 1.0])

    surrogate_loss, entropy_loss = algo._masked_actor_losses(batch, advantages, ratio, entropy)

    self.assertTrue(torch.allclose(surrogate_loss, torch.tensor(-3.3333333)))
    self.assertTrue(torch.allclose(entropy_loss, torch.tensor(1.6666666)))

  def test_student_ppo_idle_action_mask_uses_delayed_launch_state_before_obs_condition(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.device = "cpu"
    algo.env = type("Env", (), {"num_envs": 2, "_gk_delayed_ball_launched": torch.tensor([False, True])})()
    algo.actor = type("Actor", (), {"_raw_student_obs": lambda _self, obs: obs["student"]})()
    observations = {"student": torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.2, 0.1, 0.3, 1.0]])}

    idle_mask = algo._idle_action_mask(observations)

    self.assertTrue(torch.equal(idle_mask, torch.tensor([True, False])))

  def test_student_ppo_idle_action_mask_uses_unwrapped_delayed_launch_state(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.device = "cpu"
    base_env = type("BaseEnv", (), {"num_envs": 2, "_gk_delayed_ball_launched": torch.tensor([False, True])})()
    algo.env = type("WrapperEnv", (), {"num_envs": 2, "unwrapped": base_env})()
    algo.actor = type("Actor", (), {"_raw_student_obs": lambda _self, obs: obs["student"]})()
    observations = {"student": torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.2, 0.1, 0.3, 1.0]])}

    idle_mask = algo._idle_action_mask(observations)

    self.assertTrue(torch.equal(idle_mask, torch.tensor([True, False])))

  def test_student_ppo_uses_rollout_actor_loss_mask_for_recurrent_batches(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.mask_idle_actor_loss = True
    algo.idle_actor_loss_weight = 0.0
    algo.clip_param = 0.1
    batch = type("Batch", (), {})()
    batch.actor_loss_mask = torch.tensor([[[0.0], [1.0]], [[1.0], [0.0]]])
    advantages = torch.tensor([[100.0, 1.0], [2.0, 100.0]])
    ratio = torch.ones(2, 2)
    entropy = torch.tensor([[1000.0, 3.0], [4.0, 1000.0]])

    surrogate_loss, entropy_loss = algo._masked_actor_losses(batch, advantages, ratio, entropy)

    self.assertTrue(torch.allclose(surrogate_loss, torch.tensor(-1.5)))
    self.assertTrue(torch.allclose(entropy_loss, torch.tensor(3.5)))

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

  def test_student_ppo_active_bc_weight_uses_linear_schedule(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.active_bc_coef = 0.8
    algo.active_bc_final_coef = 0.05
    algo.active_bc_anneal_updates = 2000
    algo._ppo_update_count = 0

    self.assertAlmostEqual(algo._active_bc_weight(), 0.8)
    algo._ppo_update_count = 1000
    self.assertAlmostEqual(algo._active_bc_weight(), 0.425)
    algo._ppo_update_count = 2500
    self.assertAlmostEqual(algo._active_bc_weight(), 0.05)

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

  def test_student_ppo_reference_kl_penalizes_drift_from_start_policy(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    class _Actor:
      def __init__(self, mean):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.ones_like(self.mean) * 0.5
        self.output_distribution_params = (self.mean, self.std)

      def __call__(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del obs, masks, hidden_state, stochastic_output
        self.output_distribution_params = (self.mean, self.std)
        return self.mean

      def get_kl_divergence(self, old_params, new_params):
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = torch.distributions.Normal(old_mean, old_std)
        new_dist = torch.distributions.Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.actor = _Actor([[0.0, 0.0], [0.0, 0.0]])
    algo.reference_actor = _Actor([[0.0, 0.0], [0.0, 0.0]])
    algo.device = "cpu"
    algo.reference_kl_std_floor = 0.1
    batch = type(
      "Batch",
      (),
      {
        "observations": TensorDict({"student": torch.zeros(2, 4)}, batch_size=[2]),
        "masks": None,
        "hidden_states": (None, None),
      },
    )()

    self.assertAlmostEqual(algo._reference_kl_loss(batch).item(), 0.0)

    algo.actor = _Actor([[1.0, 0.0], [0.0, 0.0]])

    self.assertGreater(algo._reference_kl_loss(batch).item(), 0.0)

  def test_student_ppo_reference_kl_uses_std_floor_for_low_exploration_policy(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    class _Actor:
      def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.ones_like(self.mean) * std
        self.output_distribution_params = (self.mean, self.std)

      def __call__(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del obs, masks, hidden_state, stochastic_output
        self.output_distribution_params = (self.mean, self.std)
        return self.mean

      def get_kl_divergence(self, old_params, new_params):
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = torch.distributions.Normal(old_mean, old_std)
        new_dist = torch.distributions.Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.actor = _Actor([[1.0, 0.0], [0.0, 0.0]], std=0.002)
    algo.reference_actor = _Actor([[0.0, 0.0], [0.0, 0.0]], std=0.002)
    algo.device = "cpu"
    algo.reference_kl_std_floor = 0.1
    batch = type(
      "Batch",
      (),
      {
        "observations": TensorDict({"student": torch.zeros(2, 4)}, batch_size=[2]),
        "masks": None,
        "hidden_states": (None, None),
      },
    )()

    loss = algo._reference_kl_loss(batch)

    self.assertTrue(torch.isfinite(loss))
    self.assertLess(loss.item(), 100.0)

  def test_student_ppo_reference_kl_clips_distribution_means(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.reference_kl_std_floor = 0.1
    algo.actor_mean_clip = 6.0

    mean, std = algo._reference_kl_params((torch.tensor([[100.0, -100.0]]), torch.ones(1, 2) * 0.01))

    self.assertTrue(torch.equal(mean, torch.tensor([[6.0, -6.0]])))
    self.assertTrue(torch.equal(std, torch.ones(1, 2) * 0.1))

  def test_student_ppo_reference_kl_backward_accepts_reference_tensors(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    class _CurrentActor:
      def __init__(self):
        self.mean = torch.zeros(2, 2, requires_grad=True)
        self.std = torch.full((2, 2), 0.1, requires_grad=True)
        self.output_distribution_params = (self.mean, self.std)

      def get_kl_divergence(self, old_params, new_params):
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = torch.distributions.Normal(old_mean, old_std)
        new_dist = torch.distributions.Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

    class _ReferenceActor:
      def __call__(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del masks, hidden_state, stochastic_output
        mean = obs["student"][:, :2] + 0.1
        std = torch.ones_like(mean) * 0.1
        self.output_distribution_params = (mean, std)
        return mean

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.actor = _CurrentActor()
    algo.reference_actor = _ReferenceActor()
    algo.device = "cpu"
    algo.reference_kl_std_floor = 0.1
    batch = type(
      "Batch",
      (),
      {
        "observations": TensorDict({"student": torch.zeros(2, 4)}, batch_size=[2]),
        "masks": None,
        "hidden_states": (None, None),
      },
    )()

    loss = algo._reference_kl_loss(batch)
    loss.backward()

    self.assertIsNotNone(algo.actor.mean.grad)
    self.assertIsNotNone(algo.actor.std.grad)
    self.assertTrue(torch.isfinite(algo.actor.mean.grad).all())
    self.assertTrue(torch.isfinite(algo.actor.std.grad).all())

  def test_student_ppo_clips_actor_distribution_mean_during_update(self):
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    class _Distribution:
      def __init__(self):
        self.mean = torch.tensor([[10.0, -9.0, 2.0]])
        self.updated = None

      def update(self, mean):
        self.updated = mean
        self.mean = mean

    algo = GoalkeeperStudentPPO.__new__(GoalkeeperStudentPPO)
    algo.actor = type("Actor", (), {"distribution": _Distribution()})()
    algo.actor_mean_clip = 6.0

    algo._clip_actor_distribution_mean()

    self.assertTrue(torch.equal(algo.actor.distribution.updated, torch.tensor([[6.0, -6.0, 2.0]])))

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

  def test_student_ppo_active_reward_weights_can_be_overridden(self):
    from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config

    cfg = FineTuneConfig(
      profile="teacher_kl",
      action_rate_weight=0.0,
      active_upright_weight=0.0,
      active_fall_penalty_weight=-10.0,
      intercept_weight=5.0,
      stop_ball_weight=10.0,
    )

    train_cfg = build_train_config(cfg)
    rewards = train_cfg.env.rewards

    self.assertEqual(rewards["action_rate"].weight, 0.0)
    self.assertEqual(rewards["active_upright"].weight, 0.0)
    self.assertEqual(rewards["active_fall_penalty"].weight, -10.0)
    self.assertEqual(rewards["intercept"].weight, 5.0)
    self.assertEqual(rewards["stop_ball"].weight, 10.0)


if __name__ == "__main__":
  unittest.main()
