"""Tests for original goalkeeper runner checkpoint compatibility."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch


class _Alg:
  def __init__(self):
    self.actor = torch.nn.Linear(1, 1)
    self.critic = torch.nn.Linear(1, 1)


class GoalkeeperRunnerLoadTest(unittest.TestCase):
  def test_keeper_idle_train_script_uses_random_initialization(self):
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_keeper_idle_from_init.sh"
    text = script.read_text()

    self.assertNotIn("--load-checkpoint-path", text)
    self.assertNotIn("--load-actor-only", text)
    self.assertIn("--agent.actor.distribution-cfg.init-std", text)
    self.assertIn("--agent.algorithm.entropy-coef", text)
    self.assertIn("--agent.num-steps-per-env", text)

  def test_goalkeeper_runner_loads_actor_from_moe_bundle(self):
    from src.tasks.soccer.config.g1.rl_cfg import GoalkeeperRunner

    runner = object.__new__(GoalkeeperRunner)
    runner.device = "cpu"
    runner.alg = _Alg()

    actor_sd = {k: torch.full_like(v, 3.0) for k, v in runner.alg.actor.state_dict().items()}
    critic_before = {k: v.clone() for k, v in runner.alg.critic.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmp:
      ckpt = Path(tmp) / "moe6.pt"
      torch.save({"sr": [{"actor_state_dict": actor_sd, "infos": {"x": 1}} for _ in range(6)]}, ckpt)

      info = runner.load(str(ckpt), load_cfg={"actor": True, "critic": False})

    self.assertEqual(info, {"x": 1})
    for value in runner.alg.actor.state_dict().values():
      self.assertTrue(torch.allclose(value, torch.full_like(value, 3.0)))
    for key, value in runner.alg.critic.state_dict().items():
      self.assertTrue(torch.allclose(value, critic_before[key]))

  def test_goalkeeper_runner_rejects_checkpoint_with_no_compatible_actor_keys(self):
    from src.tasks.soccer.config.g1.rl_cfg import GoalkeeperRunner

    runner = object.__new__(GoalkeeperRunner)
    runner.device = "cpu"
    runner.alg = _Alg()

    actor_sd = {
      "mlp.0.weight": torch.ones(4, 4),
      "distribution.std_param": torch.ones(29),
    }
    with tempfile.TemporaryDirectory() as tmp:
      ckpt = Path(tmp) / "mlp_moe.pt"
      torch.save({"sr": [{"actor_state_dict": actor_sd} for _ in range(6)]}, ckpt)

      with self.assertRaisesRegex(ValueError, "No compatible actor weights"):
        runner.load(str(ckpt), load_cfg={"actor": True, "critic": False})


if __name__ == "__main__":
  unittest.main()
