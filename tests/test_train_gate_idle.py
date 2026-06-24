"""Tests for the optional prepare/idle class in the learned MoE gate."""

from __future__ import annotations

import unittest

import torch


class TrainGateIdleTest(unittest.TestCase):
  def test_gate_network_can_output_prepare_class(self):
    from scripts.train_gate import make_gate_net

    net = make_gate_net(num_classes=7, device=torch.device("cpu"))

    self.assertEqual(net[-1].out_features, 7)

  def test_make_idle_samples_labels_zero_speed_as_prepare(self):
    from scripts.train_gate import make_idle_samples

    x, y = make_idle_samples(
      count=4,
      idle_class=6,
      ball_pos=(3.0, 0.0, 0.1),
      y_range=(-0.2, 0.2),
      z_range=(0.1, 0.1),
      device=torch.device("cpu"),
    )

    self.assertEqual(tuple(x.shape), (4, 6))
    self.assertTrue(torch.allclose(x[:, 3:], torch.zeros(4, 3)))
    self.assertEqual(y.tolist(), [6, 6, 6, 6])


if __name__ == "__main__":
  unittest.main()
