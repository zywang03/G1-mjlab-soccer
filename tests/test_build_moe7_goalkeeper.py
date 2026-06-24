"""Tests for building a MoE goalkeeper bundle with a prepare expert."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch


class BuildMoe7GoalkeeperTest(unittest.TestCase):
  def test_builds_bundle_with_idle_expert_and_gate_metadata(self):
    from scripts import build_moe7_goalkeeper

    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      moe6 = tmp_path / "moe6.pt"
      idle = tmp_path / "idle.pt"
      gate = tmp_path / "gate.pt"
      out = tmp_path / "moe7.pt"
      torch.save({"sr": [{"actor_state_dict": {"w": torch.tensor([i])}} for i in range(6)]}, moe6)
      torch.save({"actor_state_dict": {"idle": torch.tensor([1.0])}}, idle)
      torch.save({"num_classes": 7, "idle_class": 6}, gate)

      build_moe7_goalkeeper.main(build_moe7_goalkeeper.Cfg(
        moe6=str(moe6),
        idle=str(idle),
        gate=str(gate),
        out=str(out),
      ))

      bundle = torch.load(out, map_location="cpu", weights_only=False)
      self.assertEqual(len(bundle["sr"]), 6)
      self.assertIn("idle", bundle)
      self.assertEqual(bundle["gate"]["num_classes"], 7)
      self.assertEqual(bundle["idle_speed_threshold"], 0.5)


if __name__ == "__main__":
  unittest.main()
