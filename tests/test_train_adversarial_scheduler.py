"""Tests for the adversarial training scheduler shell."""

from __future__ import annotations

import importlib.util
import io
import os
import random
from pathlib import Path
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from contextlib import redirect_stdout
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_train_adversarial():
  path = REPO_ROOT / "scripts" / "train_adversarial.py"
  spec = importlib.util.spec_from_file_location("train_adversarial", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


class TrainAdversarialSchedulerTest(unittest.TestCase):
  def test_random_target_schedule_is_reproducible_and_in_goal_frame(self):
    adv = _load_train_adversarial()

    a = adv.sample_target_schedule(rounds=3, targets_per_round=4, seed=7)
    b = adv.sample_target_schedule(rounds=3, targets_per_round=4, seed=7)
    c = adv.sample_target_schedule(rounds=3, targets_per_round=4, seed=8)

    self.assertEqual(a, b)
    self.assertNotEqual(a, c)
    self.assertEqual(len(a), 3)
    self.assertEqual(len(a[0]), 4)
    for round_targets in a:
      for target in round_targets:
        self.assertGreaterEqual(target["x"], -1.25)
        self.assertLessEqual(target["x"], 1.25)
        self.assertEqual(target["y"], -5.0)
        self.assertGreaterEqual(target["z"], 0.05)
        self.assertLessEqual(target["z"], 1.5)

  def test_opponent_sampling_uses_current_history_initial_mix(self):
    adv = _load_train_adversarial()

    rng = random.Random(9)
    counts = {"current": 0, "history": 0, "initial": 0}
    history_hits = set()
    for _ in range(1000):
      sample = adv.sample_opponent_ckpt(
        rng,
        initial_ckpt="initial.pt",
        current_ckpt="current.pt",
        history_ckpts=["initial.pt", "old_a.pt", "old_b.pt", "current.pt"],
      )
      counts[sample["bucket"]] += 1
      if sample["bucket"] == "history":
        history_hits.add(sample["ckpt"])
        self.assertNotIn(sample["ckpt"], {"initial.pt", "current.pt"})

    self.assertGreater(counts["current"], counts["history"])
    self.assertGreater(counts["history"], counts["initial"])
    self.assertEqual(history_hits, {"old_a.pt", "old_b.pt"})

  def test_gpu_id_parser_and_latest_model_lookup(self):
    adv = _load_train_adversarial()

    self.assertIsNone(adv.parse_gpu_ids("cpu"))
    self.assertEqual(adv.parse_gpu_ids("all"), "all")
    self.assertEqual(adv.parse_gpu_ids("0, 2"), [0, 2])

    with tempfile.TemporaryDirectory() as tmp:
      log_root = Path(tmp)
      before = set()
      run_dir = log_root / "2026-06-17_10-00-00_adv_r001_shooter"
      run_dir.mkdir()
      (run_dir / "model_100.pt").write_text("old\n", encoding="utf-8")
      (run_dir / "model_300.pt").write_text("new\n", encoding="utf-8")
      self.assertEqual(adv.latest_model_for_run(log_root, "adv_r001_shooter", before), run_dir / "model_300.pt")

  def test_real_training_configs_are_isolated_from_parent_process_by_default(self):
    adv = _load_train_adversarial()

    RealTrainConfig = type("TrainConfig", (), {"__module__": "scripts.train"})
    FakeTrainConfig = type("TrainConfig", (), {})

    cfg = adv.AdversarialConfig()
    self.assertTrue(adv.should_isolate_training_process(RealTrainConfig(), cfg))
    self.assertFalse(adv.should_isolate_training_process(FakeTrainConfig(), cfg))

    cfg.isolated_training_process = False
    self.assertFalse(adv.should_isolate_training_process(RealTrainConfig(), cfg))

  def test_dry_run_alternate_writes_manifest_and_ckpt_chain(self):
    adv = _load_train_adversarial()

    with tempfile.TemporaryDirectory() as tmp:
      cfg = adv.AdversarialConfig(
        mode="alternate",
        rounds=2,
        dry_run=True,
        out_dir=tmp,
        shooter_init="checkpoints/stage2/model_100000.pt",
        keeper_init="src/assets/soccer/weight/goalkeeper_moe6.pt",
        shooter_targets_per_round=2,
        seed=123,
      )
      manifest = adv.run(cfg)

      self.assertEqual([p["role"] for p in manifest["phases"]], [
        "shooter", "keeper", "shooter", "keeper",
      ])
      self.assertEqual(
        manifest["phases"][0]["opponent_ckpt"],
        "src/assets/soccer/weight/goalkeeper_moe6.pt",
      )
      self.assertIn(
        manifest["phases"][1]["opponent_ckpt"],
        {
          "checkpoints/stage2/model_100000.pt",
          str(Path(tmp) / "round_001" / "shooter_best.pt"),
        },
      )
      self.assertEqual(manifest["opponent_sampling"]["current"], 0.5)
      self.assertEqual(manifest["opponent_sampling"]["history"], 0.4)
      self.assertEqual(manifest["opponent_sampling"]["initial"], 0.1)
      self.assertIn("opponent_sample", manifest["phases"][0])
      self.assertIn(manifest["phases"][0]["opponent_sample"]["bucket"], {"current", "initial"})
      self.assertEqual(manifest["phases"][0]["backend"], "scripts.train.launch_training")
      self.assertEqual(manifest["phases"][0]["task_id"], "Unitree-G1-Shooter-Adversarial")
      self.assertEqual(manifest["phases"][1]["backend"], "scripts.train.launch_training")
      self.assertEqual(manifest["phases"][1]["task_id"], "Unitree-G1-Goalkeeper-Adversarial")
      self.assertIn("source_output_ckpt", manifest["phases"][1])
      self.assertNotIn("command", manifest["phases"][0])
      self.assertTrue(manifest["phases"][0]["opponent_consumed_by_backend"])
      self.assertTrue(manifest["phases"][1]["opponent_consumed_by_backend"])
      self.assertEqual(
        manifest["final"]["shooter_ckpt"],
        str(Path(tmp) / "round_002" / "shooter_best.pt"),
      )
      self.assertEqual(
        manifest["final"]["keeper_ckpt"],
        str(Path(tmp) / "round_002" / "keeper_best.pt"),
      )

      manifest_path = Path(tmp) / "manifest.json"
      self.assertTrue(manifest_path.exists())
      self.assertTrue((Path(tmp) / "round_001" / "targets.json").exists())
      self.assertTrue((Path(tmp) / "round_002" / "targets.json").exists())
      shooter_note = Path(tmp) / "round_001" / "shooter_best.note.txt"
      keeper_note = Path(tmp) / "round_001" / "keeper_best.note.txt"
      self.assertTrue(shooter_note.exists())
      self.assertTrue(keeper_note.exists())
      shooter_note_text = shooter_note.read_text(encoding="utf-8")
      self.assertIn("role: shooter", shooter_note_text)
      self.assertIn("source_kind: phase_input", shooter_note_text)
      self.assertIn("source_step: null", shooter_note_text)
      self.assertIn("input_ckpt: checkpoints/stage2/model_100000.pt", shooter_note_text)
      self.assertEqual(manifest["phases"][0]["best_note"], str(shooter_note))

  def test_keeper_only_cli_alias_skips_shooter_training(self):
    adv = _load_train_adversarial()

    cfg = adv.parse_config([
      "--keeper-only",
      "--rounds", "2",
      "--dry-run",
      "--shooter-init", "fixed_shooter.pt",
      "--keeper-init", "keeper_init.pt",
    ])
    self.assertEqual(cfg.mode, "train-keeper")

    with tempfile.TemporaryDirectory() as tmp:
      cfg.out_dir = tmp
      cfg.shooter_targets_per_round = 1
      manifest = adv.run(cfg)

      self.assertEqual([p["role"] for p in manifest["phases"]], ["keeper", "keeper"])
      self.assertEqual(manifest["final"]["shooter_ckpt"], "fixed_shooter.pt")
      self.assertEqual(
        manifest["final"]["keeper_ckpt"],
        str(Path(tmp) / "round_002" / "keeper_best.pt"),
      )
      self.assertFalse((Path(tmp) / "round_001" / "shooter_best.pt").exists())
      self.assertFalse((Path(tmp) / "round_002" / "shooter_best.pt").exists())

  def test_keeper_idle_cli_alias_trains_idle_expert_against_frozen_shooter(self):
    adv = _load_train_adversarial()

    cfg = adv.parse_config([
      "--keeper-idle-only",
      "--rounds", "2",
      "--dry-run",
      "--shooter-init", "fixed_shooter.pt",
      "--keeper-idle-init", "idle_init.pt",
      "--keeper-init", "block_keeper.pt",
    ])
    self.assertEqual(cfg.mode, "train-keeper-idle")

    with tempfile.TemporaryDirectory() as tmp:
      cfg.out_dir = tmp
      cfg.shooter_targets_per_round = 1
      manifest = adv.run(cfg)

      self.assertEqual([p["role"] for p in manifest["phases"]], ["keeper_idle", "keeper_idle"])
      self.assertEqual([p["task_id"] for p in manifest["phases"]], [
        "Unitree-G1-Goalkeeper-MoE7-Prepare-Adversarial",
        "Unitree-G1-Goalkeeper-MoE7-Prepare-Adversarial",
      ])
      self.assertEqual(manifest["phases"][0]["opponent_ckpt"], "fixed_shooter.pt")
      self.assertEqual(manifest["final"]["shooter_ckpt"], "fixed_shooter.pt")
      self.assertEqual(manifest["final"]["keeper_ckpt"], "block_keeper.pt")
      self.assertEqual(
        manifest["final"]["keeper_idle_ckpt"],
        str(Path(tmp) / "round_002" / "keeper_idle_best.pt"),
      )
      self.assertFalse((Path(tmp) / "round_001" / "keeper_best.pt").exists())
      self.assertFalse((Path(tmp) / "round_001" / "shooter_best.pt").exists())

  def test_keeper_idle_shell_script_is_dedicated_idle_entrypoint(self):
    script = REPO_ROOT / "scripts" / "train_keeper_idle_adversarial_from_init.sh"
    text = script.read_text(encoding="utf-8")

    self.assertIn("--keeper-idle-only", text)
    self.assertNotIn("--keeper-moe7-only", text)
    self.assertNotIn('MODE="train-keeper"', text)
    self.assertNotIn("--mode", text)
    self.assertNotIn("--keeper-expert-iters-per-round", text)
    self.assertIn("--keeper-idle-init", text)

  def test_keeper_moe7_cli_alias_plans_all_seven_experts_against_frozen_shooter(self):
    adv = _load_train_adversarial()

    cfg = adv.parse_config([
      "--keeper-moe7-only",
      "--rounds", "1",
      "--dry-run",
      "--shooter-init", "fixed_shooter.pt",
      "--keeper-init", "keeper_moe7.pt",
      "--keeper-expert-iters-per-round", "7",
      "--keeper-idle-iters-per-round", "8",
    ])
    self.assertEqual(cfg.mode, "train-keeper-moe7")

    with tempfile.TemporaryDirectory() as tmp:
      cfg.out_dir = tmp
      cfg.shooter_targets_per_round = 1
      manifest = adv.run(cfg)

      self.assertEqual([p["role"] for p in manifest["phases"]], ["keeper_moe7"])
      phase = manifest["phases"][0]
      self.assertEqual(len(phase["expert_runs"]), 7)
      self.assertEqual([run["expert"] for run in phase["expert_runs"]], [
        "sr0", "sr1", "sr2", "sr3", "sr4", "sr5", "idle",
      ])
      self.assertEqual({run["opponent_ckpt"] for run in phase["expert_runs"]}, {"fixed_shooter.pt"})
      self.assertEqual([run["task_id"] for run in phase["expert_runs"][:6]], [
        adv.KEEPER_EXPERT_TASK_ID,
      ] * 6)
      self.assertEqual(phase["expert_runs"][6]["task_id"], adv.KEEPER_IDLE_TASK_ID)
      self.assertEqual(manifest["final"]["shooter_ckpt"], "fixed_shooter.pt")
      self.assertEqual(
        manifest["final"]["keeper_ckpt"],
        str(Path(tmp) / "round_001" / "keeper_moe7_best.pt"),
      )

  def test_keeper_moe7_non_dry_run_trains_and_rebuilds_all_experts(self):
    adv = _load_train_adversarial()

    @dataclass(frozen=True)
    class FakeTrainConfig:
      env: object = field(default_factory=lambda: SimpleNamespace(scene=SimpleNamespace(num_envs=1), commands={}))
      agent: object = field(default_factory=lambda: SimpleNamespace(experiment_name="g1_soccer", max_iterations=0, run_name="", save_interval=0))
      motion_dir: str | None = None
      load_actor_only: bool = False
      load_checkpoint_path: str | None = None
      frozen_opponent_checkpoint_path: str | None = None
      frozen_opponent_role: str | None = None
      frozen_opponent_task_id: str | None = None
      gpu_ids: object = None

      @staticmethod
      def from_task(_task_id: str) -> "FakeTrainConfig":
        return FakeTrainConfig()

    calls = []

    def fake_launch_training(task_id: str, cfg: FakeTrainConfig) -> None:
      calls.append((task_id, cfg))
      root = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
      run_dir = root / f"2026-06-19_00-10-{len(calls):02d}_{cfg.agent.run_name}"
      run_dir.mkdir(parents=True)
      if task_id == adv.KEEPER_IDLE_TASK_ID:
        state = {"actor_state_dict": {"actor_residual.0.weight": torch.full((1,), 70.0)}}
      else:
        idx = int(cfg.agent.run_name.rsplit("expert", maxsplit=1)[1])
        state = {"actor_state_dict": {f"mlp.{idx}.weight": torch.full((1,), float(idx + 10))}}
      torch.save(state, run_dir / f"model_{cfg.agent.max_iterations}.pt")

    fake_train = types.ModuleType("scripts.train")
    fake_train.TrainConfig = FakeTrainConfig
    fake_train.launch_training = fake_launch_training
    fake_mjlab = types.ModuleType("mjlab")
    fake_mjlab.tasks = types.ModuleType("mjlab.tasks")
    fake_src = types.ModuleType("src")
    fake_src.tasks = types.ModuleType("src.tasks")

    with tempfile.TemporaryDirectory() as tmp:
      old_cwd = os.getcwd()
      os.chdir(tmp)
      try:
        moe7 = {
          "sr": [{"actor_state_dict": {f"mlp.{i}.weight": torch.full((1,), float(i))}} for i in range(6)],
          "idle": {"actor_state_dict": {"history_encoder.0.weight": torch.zeros(1)}},
          "gate": {"num_classes": 7},
        }
        torch.save(moe7, "keeper_moe7.pt")
        Path("shooter.pt").write_text("shooter", encoding="utf-8")
        with patch.dict(sys.modules, {
          "mjlab": fake_mjlab,
          "mjlab.tasks": fake_mjlab.tasks,
          "src": fake_src,
          "src.tasks": fake_src.tasks,
          "scripts.train": fake_train,
        }):
          manifest = adv.run(adv.AdversarialConfig(
            mode="train-keeper-moe7",
            rounds=1,
            dry_run=False,
            out_dir="adv_out",
            shooter_init="shooter.pt",
            keeper_init="keeper_moe7.pt",
            keeper_idle_init="keeper_moe7.pt",
            keeper_log_root="logs/rsl_rl/adversarial",
            num_envs=1,
            gpu_ids="cpu",
            keeper_expert_iters_per_round=3,
            keeper_idle_iters_per_round=4,
            shooter_targets_per_round=1,
            promotion_trials=0,
          ))
      finally:
        os.chdir(old_cwd)

      self.assertEqual([task_id for task_id, _ in calls], [adv.KEEPER_EXPERT_TASK_ID] * 6 + [adv.KEEPER_IDLE_TASK_ID])
      self.assertEqual([cfg.agent.max_iterations for _, cfg in calls], [3, 3, 3, 3, 3, 3, 4])
      for _, cfg in calls:
        self.assertTrue(cfg.load_actor_only)
        self.assertEqual(cfg.frozen_opponent_checkpoint_path, "shooter.pt")
        self.assertEqual(cfg.frozen_opponent_role, "shooter")
        self.assertEqual(cfg.frozen_opponent_task_id, adv.SHOOTER_TASK_ID)

      for idx, (_, cfg) in enumerate(calls[:6]):
        loaded_train_init = torch.load(Path(tmp, cfg.load_checkpoint_path), weights_only=False)
        self.assertEqual(loaded_train_init["actor_state_dict"][f"mlp.{idx}.weight"].item(), float(idx))
        self.assertNotIn("sr", loaded_train_init)

      out_bundle = torch.load(Path(tmp, manifest["final"]["keeper_ckpt"]), weights_only=False)
      self.assertEqual(out_bundle["gate"], {"num_classes": 7})
      self.assertEqual([out_bundle["sr"][i]["actor_state_dict"][f"mlp.{i}.weight"].item() for i in range(6)], [
        10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
      ])
      self.assertIn("actor_residual.0.weight", out_bundle["idle"]["actor_state_dict"])
      self.assertEqual(out_bundle["idle"]["actor_state_dict"]["actor_residual.0.weight"].item(), 70.0)

  def test_keeper_idle_non_dry_run_extracts_and_rebuilds_moe7_idle_only(self):
    adv = _load_train_adversarial()

    @dataclass(frozen=True)
    class FakeTrainConfig:
      env: object = field(default_factory=lambda: SimpleNamespace(scene=SimpleNamespace(num_envs=1), commands={}))
      agent: object = field(default_factory=lambda: SimpleNamespace(experiment_name="g1_soccer", max_iterations=0, run_name="", save_interval=0))
      motion_dir: str | None = None
      load_actor_only: bool = False
      load_checkpoint_path: str | None = None
      frozen_opponent_checkpoint_path: str | None = None
      frozen_opponent_role: str | None = None
      frozen_opponent_task_id: str | None = None
      gpu_ids: object = None

      @staticmethod
      def from_task(_task_id: str) -> "FakeTrainConfig":
        return FakeTrainConfig()

    calls = []

    def fake_launch_training(_task_id: str, cfg: FakeTrainConfig) -> None:
      calls.append(cfg)
      root = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
      run_dir = root / f"2026-06-19_00-00-01_{cfg.agent.run_name}"
      run_dir.mkdir(parents=True)
      torch.save({
        "actor_state_dict": {
          "sr": [{"actor_state_dict": {f"mlp.{i}.weight": torch.full((1,), float(i))}} for i in range(6)],
          "idle": {"actor_state_dict": {"actor_residual.0.weight": torch.ones(1)}},
          "gate": {"num_classes": 7},
        },
      }, run_dir / "model_4.pt")

    fake_train = types.ModuleType("scripts.train")
    fake_train.TrainConfig = FakeTrainConfig
    fake_train.launch_training = fake_launch_training
    fake_mjlab = types.ModuleType("mjlab")
    fake_mjlab.tasks = types.ModuleType("mjlab.tasks")
    fake_src = types.ModuleType("src")
    fake_src.tasks = types.ModuleType("src.tasks")

    with tempfile.TemporaryDirectory() as tmp:
      old_cwd = os.getcwd()
      os.chdir(tmp)
      try:
        moe7 = {
          "sr": [{"actor_state_dict": {f"mlp.{i}.weight": torch.full((1,), float(i))}} for i in range(6)],
          "idle": {"actor_state_dict": {"history_encoder.0.weight": torch.zeros(1)}},
          "gate": {"num_classes": 7},
        }
        torch.save(moe7, "keeper_moe7.pt")
        Path("shooter.pt").write_text("shooter", encoding="utf-8")
        with patch.dict(sys.modules, {
          "mjlab": fake_mjlab,
          "mjlab.tasks": fake_mjlab.tasks,
          "src": fake_src,
          "src.tasks": fake_src.tasks,
          "scripts.train": fake_train,
        }):
          manifest = adv.run(adv.AdversarialConfig(
            mode="train-keeper-idle",
            rounds=1,
            dry_run=False,
            out_dir="adv_out",
            shooter_init="shooter.pt",
            keeper_init="keeper_moe7.pt",
            keeper_idle_init="keeper_moe7.pt",
            keeper_log_root="logs/rsl_rl/adversarial",
            num_envs=1,
            gpu_ids="cpu",
            keeper_idle_iters_per_round=4,
            shooter_targets_per_round=1,
            promotion_trials=0,
          ))
      finally:
        os.chdir(old_cwd)

      self.assertEqual(len(calls), 1)
      self.assertTrue(calls[0].load_actor_only)
      load_path = Path(tmp, calls[0].load_checkpoint_path)
      loaded_train_init = torch.load(load_path, weights_only=False)
      self.assertIn("sr", loaded_train_init)

      out_bundle = torch.load(Path(tmp, manifest["final"]["keeper_idle_ckpt"]), weights_only=False)
      self.assertEqual(len(out_bundle["sr"]), 6)
      self.assertEqual(out_bundle["gate"], {"num_classes": 7})
      self.assertIn("actor_residual.0.weight", out_bundle["idle"]["actor_state_dict"])
      self.assertEqual(out_bundle["sr"][0]["actor_state_dict"]["mlp.0.weight"].item(), 0.0)

  def test_mock_non_dry_run_completes_all_rounds_with_configured_log_root(self):
    adv = _load_train_adversarial()

    @dataclass(frozen=True)
    class FakeTrainConfig:
      env: object = field(default_factory=lambda: SimpleNamespace(
        scene=SimpleNamespace(num_envs=1),
        commands={},
      ))
      agent: object = field(default_factory=lambda: SimpleNamespace(
        experiment_name="g1_soccer",
        max_iterations=0,
        run_name="",
        save_interval=0,
      ))
      motion_dir: str | None = None
      load_checkpoint_path: str | None = None
      frozen_opponent_checkpoint_path: str | None = None
      frozen_opponent_role: str | None = None
      frozen_opponent_task_id: str | None = None
      gpu_ids: object = None

      @staticmethod
      def from_task(_task_id: str) -> "FakeTrainConfig":
        return FakeTrainConfig()

    calls = []

    def fake_launch_training(_task_id: str, cfg: FakeTrainConfig) -> None:
      calls.append(cfg)
      root = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
      run_dir = root / f"2026-06-17_00-00-{len(calls):02d}_{cfg.agent.run_name}"
      run_dir.mkdir(parents=True)
      (run_dir / f"model_{cfg.agent.max_iterations}.pt").write_text(
        f"{cfg.agent.run_name}\n", encoding="utf-8",
      )

    fake_train = types.ModuleType("scripts.train")
    fake_train.TrainConfig = FakeTrainConfig
    fake_train.launch_training = fake_launch_training

    fake_mjlab = types.ModuleType("mjlab")
    fake_mjlab.tasks = types.ModuleType("mjlab.tasks")
    fake_src = types.ModuleType("src")
    fake_src.tasks = types.ModuleType("src.tasks")

    with tempfile.TemporaryDirectory() as tmp:
      old_cwd = os.getcwd()
      os.chdir(tmp)
      try:
        with patch.dict(sys.modules, {
          "mjlab": fake_mjlab,
          "mjlab.tasks": fake_mjlab.tasks,
          "src": fake_src,
          "src.tasks": fake_src.tasks,
          "scripts.train": fake_train,
        }):
          manifest = adv.run(adv.AdversarialConfig(
            mode="alternate",
            rounds=2,
            dry_run=False,
            out_dir="adv_out",
            shooter_init="shooter_init.pt",
            keeper_init="keeper_init.pt",
            motion_dir="motions",
            shooter_log_root="logs/rsl_rl/adversarial",
            keeper_log_root="logs/rsl_rl/adversarial",
            num_envs=2,
            gpu_ids="cpu",
            shooter_iters_per_round=3,
            keeper_blocks_per_round=1,
            keeper_block_iters=2,
            shooter_targets_per_round=1,
            promotion_trials=0,
          ))
      finally:
        os.chdir(old_cwd)

      self.assertEqual([p["role"] for p in manifest["phases"]], [
        "shooter", "keeper", "shooter", "keeper",
      ])
      self.assertEqual([call.agent.max_iterations for call in calls], [3, 2, 3, 2])
      self.assertEqual([call.agent.experiment_name for call in calls], [
        "adversarial", "adversarial", "adversarial", "adversarial",
      ])
      self.assertEqual(calls[0].frozen_opponent_role, "goalkeeper")
      self.assertEqual(calls[0].frozen_opponent_task_id, adv.KEEPER_TASK_ID)
      self.assertEqual(calls[1].frozen_opponent_role, "shooter")
      self.assertEqual(calls[1].frozen_opponent_task_id, adv.SHOOTER_TASK_ID)
      for phase in manifest["phases"]:
        self.assertTrue(Path(tmp, phase["output_ckpt"]).exists())
        self.assertTrue(Path(tmp, phase["best_note"]).exists())
        self.assertIn("logs/rsl_rl/adversarial", phase["source_output_ckpt"])

      shooter_note = Path(tmp) / "adv_out" / "round_001" / "shooter_best.note.txt"
      keeper_note = Path(tmp) / "adv_out" / "round_001" / "keeper_best.note.txt"
      shooter_note_text = shooter_note.read_text(encoding="utf-8")
      keeper_note_text = keeper_note.read_text(encoding="utf-8")
      self.assertIn("source_kind: trained_model", shooter_note_text)
      self.assertIn("source_step: 3", shooter_note_text)
      self.assertIn("source_kind: trained_model", keeper_note_text)
      self.assertIn("source_step: 2", keeper_note_text)

      self.assertEqual(
        manifest["final"]["shooter_ckpt"],
        str(Path("adv_out") / "round_002" / "shooter_best.pt"),
      )
      self.assertEqual(
        manifest["final"]["keeper_ckpt"],
        str(Path("adv_out") / "round_002" / "keeper_best.pt"),
      )

  def test_promotion_eval_keeps_previous_best_when_candidate_loses(self):
    adv = _load_train_adversarial()

    @dataclass(frozen=True)
    class FakeTrainConfig:
      env: object = field(default_factory=lambda: SimpleNamespace(scene=SimpleNamespace(num_envs=1), commands={}))
      agent: object = field(default_factory=lambda: SimpleNamespace(experiment_name="g1_soccer", max_iterations=0, run_name="", save_interval=0))
      motion_dir: str | None = None
      load_checkpoint_path: str | None = None
      frozen_opponent_checkpoint_path: str | None = None
      frozen_opponent_role: str | None = None
      frozen_opponent_task_id: str | None = None
      gpu_ids: object = None

      @staticmethod
      def from_task(_task_id: str) -> "FakeTrainConfig":
        return FakeTrainConfig()

    calls = []

    def fake_launch_training(_task_id: str, cfg: FakeTrainConfig) -> None:
      calls.append(cfg)
      root = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
      run_dir = root / f"2026-06-17_00-01-{len(calls):02d}_{cfg.agent.run_name}"
      run_dir.mkdir(parents=True)
      (run_dir / f"model_{cfg.agent.max_iterations}.pt").write_text(cfg.agent.run_name, encoding="utf-8")

    fake_train = types.ModuleType("scripts.train")
    fake_train.TrainConfig = FakeTrainConfig
    fake_train.launch_training = fake_launch_training

    def fake_compare(role, previous_ckpt, candidate_ckpt, opponent_ckpt, trials, device, seed):
      return {
        "role": role,
        "winner": "previous",
        "previous_ckpt": previous_ckpt,
        "candidate_ckpt": candidate_ckpt,
        "opponent_ckpt": opponent_ckpt,
        "trials": trials,
        "device": device,
        "seed": seed,
        "previous_score": 2,
        "candidate_score": 1,
      }

    fake_mjlab = types.ModuleType("mjlab")
    fake_mjlab.tasks = types.ModuleType("mjlab.tasks")
    fake_src = types.ModuleType("src")
    fake_src.tasks = types.ModuleType("src.tasks")

    with tempfile.TemporaryDirectory() as tmp:
      old_cwd = os.getcwd()
      os.chdir(tmp)
      try:
        with patch.dict(sys.modules, {
          "mjlab": fake_mjlab,
          "mjlab.tasks": fake_mjlab.tasks,
          "src": fake_src,
          "src.tasks": fake_src.tasks,
          "scripts.train": fake_train,
        }), patch.object(adv, "compare_candidate_to_previous", side_effect=fake_compare):
          manifest = adv.run(adv.AdversarialConfig(
            mode="alternate",
            rounds=1,
            dry_run=False,
            out_dir="adv_out",
            shooter_init="shooter_init.pt",
            keeper_init="keeper_init.pt",
            shooter_log_root="logs/rsl_rl/adversarial",
            keeper_log_root="logs/rsl_rl/adversarial",
            num_envs=1,
            gpu_ids="cpu",
            shooter_iters_per_round=3,
            keeper_blocks_per_round=1,
            keeper_block_iters=2,
            promotion_trials=5,
            promotion_device="cpu",
          ))
      finally:
        os.chdir(old_cwd)

      shooter_phase, keeper_phase = manifest["phases"]
      self.assertEqual(shooter_phase["candidate_ckpt"], str(Path("adv_out") / "round_001" / "shooter_candidate.pt"))
      self.assertEqual(shooter_phase["output_ckpt"], str(Path("adv_out") / "round_001" / "shooter_best.pt"))
      self.assertEqual(shooter_phase["promotion_eval"]["winner"], "previous")
      self.assertEqual(shooter_phase["promoted"], False)
      self.assertEqual(Path(tmp, shooter_phase["output_ckpt"]).read_text(encoding="utf-8"), "previous best placeholder: shooter_init.pt\n")
      shooter_note = Path(tmp, shooter_phase["best_note"]).read_text(encoding="utf-8")
      self.assertIn("promotion_winner: previous", shooter_note)
      self.assertIn("best_source_kind: previous_best", shooter_note)
      self.assertEqual(keeper_phase["promotion_eval"]["winner"], "previous")
      self.assertEqual(manifest["final"]["shooter_ckpt"], str(Path("adv_out") / "round_001" / "shooter_best.pt"))
      self.assertEqual(manifest["final"]["keeper_ckpt"], str(Path("adv_out") / "round_001" / "keeper_best.pt"))

  def test_compare_candidate_to_previous_logs_progress(self):
    adv = _load_train_adversarial()

    class FakeEnv:
      num_envs = 1
      device = "cpu"

      def close(self):
        pass

    class FakeWrapped:
      def __init__(self, env, clip_actions):
        self.env = env
        self.clip_actions = clip_actions

    class FakePolicy:
      def __init__(self, role, ckpt, device):
        self.role = role
        self.ckpt = ckpt

      def bind_env(self, env):
        return self

      def close(self):
        pass

    calls = []

    def fake_run_trial(_wrapped, _env, shooter, _keeper):
      calls.append(shooter.ckpt)
      goal = shooter.ckpt == "cand.pt"
      return {
        "goal_scored": goal,
        "blocked": not goal,
        "ball_final_x": -0.7 if goal else 0.2,
      }

    fake_mjlab = types.ModuleType("mjlab")
    fake_mjlab.tasks = types.ModuleType("mjlab.tasks")
    fake_src = types.ModuleType("src")
    fake_src.tasks = types.ModuleType("src.tasks")
    fake_envs = types.ModuleType("mjlab.envs")
    fake_envs.ManagerBasedRlEnv = lambda cfg, device: FakeEnv()
    fake_rl = types.ModuleType("mjlab.rl")
    fake_rl.RslRlVecEnvWrapper = FakeWrapped
    fake_utils = types.ModuleType("mjlab.utils.torch")
    fake_utils.configure_torch_backends = lambda: None
    fake_compete = types.ModuleType("scripts.compete")
    fake_compete.make_compete_env_cfg = lambda: SimpleNamespace(scene=SimpleNamespace(num_envs=0))
    fake_compete.run_trial = fake_run_trial

    stdout = io.StringIO()
    with patch.dict(sys.modules, {
      "mjlab": fake_mjlab,
      "mjlab.tasks": fake_mjlab.tasks,
      "mjlab.envs": fake_envs,
      "mjlab.rl": fake_rl,
      "mjlab.utils.torch": fake_utils,
      "src": fake_src,
      "src.tasks": fake_src.tasks,
      "scripts.compete": fake_compete,
    }), patch.object(adv, "LocalCheckpointPolicy", FakePolicy), redirect_stdout(stdout):
      result = adv.compare_candidate_to_previous(
        "shooter", "prev.pt", "cand.pt", "keeper.pt", trials=2, device="cpu", seed=5,
      )

    text = stdout.getvalue()
    self.assertIn("[PROMOTE] start role=shooter trials=2 metric=goals", text)
    self.assertIn("[PROMOTE] scoring previous active_ckpt=prev.pt", text)
    self.assertIn("[PROMOTE] previous trial=1/2 goal=0 block=1 final_x=0.200 cum_goals=0 cum_blocks=1 cum_crossed=0 score=0", text)
    self.assertIn("[PROMOTE] scoring candidate active_ckpt=cand.pt", text)
    self.assertIn("[PROMOTE] candidate trial=2/2 goal=1 block=0 final_x=-0.700 cum_goals=2 cum_blocks=0 cum_crossed=2 score=2", text)
    self.assertIn("[PROMOTE] result role=shooter winner=candidate previous_score=0 candidate_score=2", text)
    self.assertEqual(result["winner"], "candidate")

  def test_local_checkpoint_policy_reset_handles_inference_tensor_action_cache(self):
    adv = _load_train_adversarial()
    import torch

    class FakePolicy:
      def __call__(self, _dual_env, _last_action):
        return torch.ones(1, 29)

    wrapper = object.__new__(adv.LocalCheckpointPolicy)
    wrapper.role = "shooter"
    wrapper.device = torch.device("cpu")
    wrapper.policy = FakePolicy()
    wrapper.dual_env = object()
    wrapper._last_action = torch.zeros(1, 29)

    with torch.inference_mode():
      wrapper({})

    wrapper.reset()
    self.assertEqual(float(wrapper._last_action.sum()), 0.0)

  def test_local_checkpoint_policy_uses_adversarial_eval_tasks(self):
    adv = _load_train_adversarial()
    import torch

    calls = []

    class FakeFrozenPolicy:
      def __init__(self, checkpoint, device, num_envs, task_id):
        calls.append((checkpoint, device, num_envs, task_id, self.__class__.__name__))

      def __call__(self, _dual_env, _last_action):
        return torch.zeros(1, 29)

    class FakeShooter(FakeFrozenPolicy):
      pass

    class FakeKeeper(FakeFrozenPolicy):
      pass

    fake_module = types.ModuleType("src.tasks.soccer.adversarial")
    fake_module.FrozenShooterPolicy = FakeShooter
    fake_module.FrozenGoalkeeperPolicy = FakeKeeper

    with patch.dict(sys.modules, {"src.tasks.soccer.adversarial": fake_module}):
      adv.LocalCheckpointPolicy("shooter", "s.pt", "cpu")
      adv.LocalCheckpointPolicy("keeper", "k.pt", "cpu")

    self.assertEqual(calls[0][3], adv.SHOOTER_TASK_ID)
    self.assertEqual(calls[1][3], adv.KEEPER_TASK_ID)


if __name__ == "__main__":
  unittest.main()
