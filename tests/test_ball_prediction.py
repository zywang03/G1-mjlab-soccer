"""Tests for closed-form soccer ball plane prediction."""

from __future__ import annotations

import unittest

import torch


class BallPredictionTest(unittest.TestCase):
  def test_predicts_parabolic_intersection_with_keeper_plane(self):
    from src.tasks.soccer.ball_prediction import (
      BALL_MODE_AIR,
      predict_ball_plane_intersection,
    )

    pos = torch.tensor([[2.0, 0.5, 1.0]])
    vel = torch.tensor([[-4.0, 0.4, 2.0]])

    pred = predict_ball_plane_intersection(pos, vel, plane_x=0.0, gravity=10.0)

    self.assertTrue(torch.allclose(pred.time, torch.tensor([0.5])))
    self.assertTrue(torch.allclose(pred.yz, torch.tensor([[0.7, 0.75]])))
    self.assertEqual(pred.valid.tolist(), [True])
    self.assertEqual(pred.idle.tolist(), [False])
    self.assertEqual(pred.mode.tolist(), [BALL_MODE_AIR])

  def test_stationary_ball_returns_current_yz_with_idle_flag(self):
    from src.tasks.soccer.ball_prediction import (
      BALL_MODE_GROUND,
      predict_ball_plane_intersection,
    )

    pos = torch.tensor([[3.0, -0.2, 0.1]])
    vel = torch.zeros(1, 3)

    pred = predict_ball_plane_intersection(pos, vel, plane_x=0.0)

    self.assertTrue(torch.allclose(pred.time, torch.tensor([0.0])))
    self.assertTrue(torch.allclose(pred.yz, torch.tensor([[-0.2, 0.1]])))
    self.assertEqual(pred.valid.tolist(), [False])
    self.assertEqual(pred.idle.tolist(), [True])
    self.assertEqual(pred.mode.tolist(), [BALL_MODE_GROUND])

  def test_ball_moving_away_from_plane_is_not_valid_but_not_idle(self):
    from src.tasks.soccer.ball_prediction import predict_ball_plane_intersection

    pos = torch.tensor([[1.0, 0.25, 0.4]])
    vel = torch.tensor([[2.0, 0.0, 0.0]])

    pred = predict_ball_plane_intersection(pos, vel, plane_x=0.0)

    self.assertTrue(torch.allclose(pred.time, torch.tensor([0.0])))
    self.assertTrue(torch.allclose(pred.yz, torch.tensor([[0.25, 0.4]])))
    self.assertEqual(pred.valid.tolist(), [False])
    self.assertEqual(pred.idle.tolist(), [False])

  def test_default_mode_treats_low_horizontal_ball_as_ground_ball(self):
    from src.tasks.soccer.ball_prediction import (
      BALL_MODE_GROUND,
      predict_ball_plane_intersection,
    )

    pos = torch.tensor([[4.0, 0.3, 0.1]])
    vel = torch.tensor([[-5.0, -0.2, 0.0]])

    pred = predict_ball_plane_intersection(pos, vel, plane_x=0.0)

    self.assertTrue(torch.allclose(pred.time, torch.tensor([0.8])))
    self.assertTrue(torch.allclose(pred.yz, torch.tensor([[0.14, 0.1]])))
    self.assertEqual(pred.valid.tolist(), [True])
    self.assertEqual(pred.idle.tolist(), [False])
    self.assertEqual(pred.mode.tolist(), [BALL_MODE_GROUND])


if __name__ == "__main__":
  unittest.main()
