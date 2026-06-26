"""Tests for prepare-only goalkeeper waiting rewards."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from src.tasks.soccer.mdp.goalkeeper_obs import _GK_JOINT_NAMES
from src.tasks.soccer.mdp.goalkeeper_rewards import (
  goalkeeper_active_action_rate,
  goalkeeper_active_body_intercept,
  goalkeeper_active_condition_target_reach,
  goalkeeper_active_fall_penalty,
  goalkeeper_active_intercept_point,
  goalkeeper_active_motion_prior_joint_pose,
  goalkeeper_active_upright,
  goalkeeper_idle_action_rate,
  goalkeeper_idle_alive,
  goalkeeper_idle_base_still,
  goalkeeper_idle_base_height_band,
  goalkeeper_idle_leg_ready_pose,
  goalkeeper_idle_low_base_height,
  goalkeeper_idle_upright,
)


class _FakeScene(dict):
  def __init__(self, *, robot, ball, env_origins):
    super().__init__({"robot": robot, "ball": ball})
    self.env_origins = env_origins


class _FakeRobot:
  def __init__(self, *, root_z, joint_pos, body_link_pos_w=None, projected_gravity_b=None):
    if body_link_pos_w is None:
      body_link_pos_w = [
        [[0.0, 0.0, z], [0.0, 0.2, z]]
        for z in root_z
      ]
    if projected_gravity_b is None:
      projected_gravity_b = [[0.0, 0.0, -1.0] for _ in root_z]
    self.data = SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, z] for z in root_z], dtype=torch.float32),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0] for _ in root_z], dtype=torch.float32),
      root_link_lin_vel_w=torch.zeros(len(root_z), 3),
      root_link_ang_vel_w=torch.zeros(len(root_z), 3),
      body_link_pos_w=torch.tensor(body_link_pos_w, dtype=torch.float32),
      projected_gravity_b=torch.tensor(projected_gravity_b, dtype=torch.float32),
      joint_pos=joint_pos,
    )

  def find_joints(self, joint_names, preserve_order=True):
    if not preserve_order:
      raise AssertionError("test expects ordered joint lookup")
    return [[_GK_JOINT_NAMES.index(name) for name in joint_names], list(joint_names)]

  def find_bodies(self, body_names, preserve_order=True):
    if not preserve_order:
      raise AssertionError("test expects ordered body lookup")
    return [list(range(len(body_names))), list(body_names)]


class _FakeBall:
  def __init__(self, velocities, positions):
    self.data = SimpleNamespace(
      root_link_lin_vel_w=torch.tensor(velocities, dtype=torch.float32),
      root_link_pos_w=torch.tensor(positions, dtype=torch.float32),
    )


def _make_env(
  *,
  root_z,
  joint_pos,
  ball_velocities,
  ball_positions=None,
  body_link_pos_w=None,
  projected_gravity_b=None,
  delayed_ball_launched=None,
  terminated=None,
):
  if ball_positions is None:
    ball_positions = [[0.0, 0.0, z] for z in root_z]
  env = SimpleNamespace()
  env.num_envs = len(root_z)
  env.device = torch.device("cpu")
  env.scene = _FakeScene(
    robot=_FakeRobot(
      root_z=root_z,
      joint_pos=joint_pos,
      body_link_pos_w=body_link_pos_w,
      projected_gravity_b=projected_gravity_b,
    ),
    ball=_FakeBall(ball_velocities, ball_positions),
    env_origins=torch.zeros(len(root_z), 3),
  )
  env.action_manager = SimpleNamespace(
    action=torch.ones(len(root_z), 2),
    prev_action=torch.zeros(len(root_z), 2),
  )
  env.termination_manager = SimpleNamespace(
    terminated=torch.tensor(terminated or [False for _ in root_z], dtype=torch.bool)
  )
  if delayed_ball_launched is not None:
    env._gk_delayed_ball_launched = torch.tensor(delayed_ball_launched, dtype=torch.bool)
  return env


class GoalkeeperPrepareRewardsTest(unittest.TestCase):
  def test_active_motion_prior_joint_pose_tracks_prior_only_after_launch(self):
    with tempfile.TemporaryDirectory() as tmp:
      motion_dir = Path(tmp)
      prior_names = _GK_JOINT_NAMES[:3]
      (motion_dir / "joint_id.txt").write_text(
        "\n".join(f"{i} {name}" for i, name in enumerate(prior_names)),
        encoding="utf-8",
      )
      torch.save(
        {"joint_position": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])},
        motion_dir / "leftjump.pt",
      )

      joint_pos = torch.zeros(3, len(_GK_JOINT_NAMES))
      joint_pos[1, :3] = torch.tensor([0.1, 0.2, 0.3])
      joint_pos[2, :3] = torch.tensor([2.0, 2.0, 2.0])
      env = _make_env(
        root_z=[0.80, 0.80, 0.80],
        joint_pos=joint_pos,
        ball_velocities=[
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
        ],
        delayed_ball_launched=[False, True, True],
      )
      env.episode_length_buf = torch.zeros(3, dtype=torch.long)

      reward = goalkeeper_active_motion_prior_joint_pose(
        env,
        motion_dir=str(motion_dir),
        std=0.2,
        launch_delay_s=0.0,
        dt=1.0,
      )

      self.assertEqual(reward[0].item(), 0.0)
      self.assertGreater(reward[1].item(), 0.95)
      self.assertLess(reward[2].item(), 0.01)

  def test_active_motion_prior_joint_pose_can_be_restricted_to_named_motions(self):
    with tempfile.TemporaryDirectory() as tmp:
      motion_dir = Path(tmp)
      prior_names = _GK_JOINT_NAMES[:3]
      (motion_dir / "joint_id.txt").write_text(
        "\n".join(f"{i} {name}" for i, name in enumerate(prior_names)),
        encoding="utf-8",
      )
      torch.save(
        {"joint_position": torch.tensor([[0.0, 0.0, 0.0]])},
        motion_dir / "lefthand.pt",
      )
      torch.save(
        {"joint_position": torch.tensor([[1.0, 1.0, 1.0]])},
        motion_dir / "leftjump.pt",
      )

      joint_pos = torch.zeros(1, len(_GK_JOINT_NAMES))
      env = _make_env(
        root_z=[0.80],
        joint_pos=joint_pos,
        ball_velocities=[[0.0, 0.0, 0.0]],
        delayed_ball_launched=[True],
      )
      env.episode_length_buf = torch.zeros(1, dtype=torch.long)

      unrestricted = goalkeeper_active_motion_prior_joint_pose(
        env,
        motion_dir=str(motion_dir),
        std=0.2,
        launch_delay_s=0.0,
        dt=1.0,
      )
      jump_only = goalkeeper_active_motion_prior_joint_pose(
        env,
        motion_dir=str(motion_dir),
        motion_names=("leftjump.pt",),
        std=0.2,
        launch_delay_s=0.0,
        dt=1.0,
      )

      self.assertGreater(unrestricted[0].item(), 0.95)
      self.assertLess(jump_only[0].item(), 0.01)

  def test_active_motion_prior_joint_pose_routes_by_goalkeeper_region(self):
    with tempfile.TemporaryDirectory() as tmp:
      motion_dir = Path(tmp)
      prior_names = _GK_JOINT_NAMES[:3]
      (motion_dir / "joint_id.txt").write_text(
        "\n".join(f"{i} {name}" for i, name in enumerate(prior_names)),
        encoding="utf-8",
      )
      region_motions = (
        "righthand.pt",
        "lefthand.pt",
        "rightjump.pt",
        "leftjump.pt",
        "rightstep.pt",
        "leftstep.pt",
      )
      for idx, name in enumerate(region_motions):
        torch.save(
          {"joint_position": torch.tensor([[float(idx), 0.0, 0.0]])},
          motion_dir / name,
        )

      joint_pos = torch.zeros(6, len(_GK_JOINT_NAMES))
      joint_pos[:, 0] = torch.arange(6, dtype=torch.float32)
      env = _make_env(
        root_z=[0.80] * 6,
        joint_pos=joint_pos,
        ball_velocities=[[0.0, 0.0, 0.0] for _ in range(6)],
        delayed_ball_launched=[True] * 6,
      )
      env.episode_length_buf = torch.zeros(6, dtype=torch.long)
      env._gk_region = torch.arange(6, dtype=torch.long)

      reward = goalkeeper_active_motion_prior_joint_pose(
        env,
        motion_dir=str(motion_dir),
        route_mode="region",
        std=0.2,
        launch_delay_s=0.0,
        dt=1.0,
      )

      self.assertTrue(torch.all(reward > 0.95))

  def test_active_motion_prior_region_route_does_not_take_best_of_all_motions(self):
    with tempfile.TemporaryDirectory() as tmp:
      motion_dir = Path(tmp)
      prior_names = _GK_JOINT_NAMES[:3]
      (motion_dir / "joint_id.txt").write_text(
        "\n".join(f"{i} {name}" for i, name in enumerate(prior_names)),
        encoding="utf-8",
      )
      region_motions = (
        "righthand.pt",
        "lefthand.pt",
        "rightjump.pt",
        "leftjump.pt",
        "rightstep.pt",
        "leftstep.pt",
      )
      for idx, name in enumerate(region_motions):
        torch.save(
          {"joint_position": torch.tensor([[float(idx), 0.0, 0.0]])},
          motion_dir / name,
        )

      joint_pos = torch.zeros(2, len(_GK_JOINT_NAMES))
      env = _make_env(
        root_z=[0.80, 0.80],
        joint_pos=joint_pos,
        ball_velocities=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        delayed_ball_launched=[True, True],
      )
      env.episode_length_buf = torch.zeros(2, dtype=torch.long)
      env._gk_region = torch.tensor([0, 1], dtype=torch.long)

      reward = goalkeeper_active_motion_prior_joint_pose(
        env,
        motion_dir=str(motion_dir),
        route_mode="region",
        std=0.2,
        launch_delay_s=0.0,
        dt=1.0,
      )

      self.assertGreater(reward[0].item(), 0.95)
      self.assertLess(reward[1].item(), 0.01)

  def test_low_base_height_penalizes_only_idle_waiting_above_target(self):
    joint_pos = torch.zeros(3, len(_GK_JOINT_NAMES))
    env = _make_env(
      root_z=[0.80, 0.72, 0.80],
      joint_pos=joint_pos,
      ball_velocities=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
      ],
    )

    penalty = goalkeeper_idle_low_base_height(env, target_z=0.73)

    self.assertGreater(penalty[0].item(), 0.0)
    self.assertEqual(penalty[1].item(), 0.0)
    self.assertEqual(penalty[2].item(), 0.0)

  def test_leg_ready_pose_uses_only_key_leg_joints_during_idle_wait(self):
    joint_pos = torch.zeros(3, len(_GK_JOINT_NAMES))
    target = {
      "left_hip_pitch_joint": -0.18,
      "left_knee_joint": 0.45,
      "left_ankle_pitch_joint": -0.27,
      "right_hip_pitch_joint": -0.18,
      "right_knee_joint": 0.45,
      "right_ankle_pitch_joint": -0.27,
    }
    for name, value in target.items():
      joint_pos[:, _GK_JOINT_NAMES.index(name)] = value
    joint_pos[:, _GK_JOINT_NAMES.index("left_shoulder_roll_joint")] = 99.0
    joint_pos[1, _GK_JOINT_NAMES.index("left_knee_joint")] += 0.20

    env = _make_env(
      root_z=[0.8, 0.8, 0.8],
      joint_pos=joint_pos,
      ball_velocities=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
      ],
    )

    penalty = goalkeeper_idle_leg_ready_pose(env, target_joint_pos=target)

    self.assertEqual(penalty[0].item(), 0.0)
    self.assertGreater(penalty[1].item(), 0.0)
    self.assertEqual(penalty[2].item(), 0.0)

  def test_default_leg_ready_pose_is_mild_not_deep_crouch(self):
    joint_pos = torch.zeros(1, len(_GK_JOINT_NAMES))
    mild = {
      "left_hip_pitch_joint": -0.18,
      "left_knee_joint": 0.45,
      "left_ankle_pitch_joint": -0.27,
      "right_hip_pitch_joint": -0.18,
      "right_knee_joint": 0.45,
      "right_ankle_pitch_joint": -0.27,
    }
    for name, value in mild.items():
      joint_pos[:, _GK_JOINT_NAMES.index(name)] = value
    env = _make_env(
      root_z=[0.8],
      joint_pos=joint_pos,
      ball_velocities=[[0.0, 0.0, 0.0]],
    )

    penalty = goalkeeper_idle_leg_ready_pose(env)

    self.assertEqual(penalty[0].item(), 0.0)

  def test_base_height_band_rewards_target_and_penalizes_too_high_or_too_low(self):
    joint_pos = torch.zeros(4, len(_GK_JOINT_NAMES))
    env = _make_env(
      root_z=[0.72, 0.80, 0.60, 0.72],
      joint_pos=joint_pos,
      ball_velocities=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
      ],
    )

    reward = goalkeeper_idle_base_height_band(env, target_z=0.72, tolerance=0.05)

    self.assertGreater(reward[0].item(), 0.9)
    self.assertLess(reward[1].item(), 0.0)
    self.assertLess(reward[2].item(), 0.0)
    self.assertEqual(reward[3].item(), 0.0)

  def test_active_stage_rewards_are_zero_during_idle_wait(self):
    joint_pos = torch.zeros(2, len(_GK_JOINT_NAMES))
    env = _make_env(
      root_z=[0.5, 0.5],
      joint_pos=joint_pos,
      ball_velocities=[
        [0.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
      ],
      ball_positions=[
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.5],
      ],
      body_link_pos_w=[
        [[0.0, 0.0, 0.5], [0.0, 0.2, 0.5]],
        [[0.0, 0.0, 0.5], [0.0, 0.2, 0.5]],
      ],
    )

    body = goalkeeper_active_body_intercept(env)
    intercept = goalkeeper_active_intercept_point(env)
    action_rate = goalkeeper_active_action_rate(env)

    self.assertEqual(body[0].item(), 0.0)
    self.assertEqual(intercept[0].item(), 0.0)
    self.assertEqual(action_rate[0].item(), 0.0)
    self.assertGreater(body[1].item(), 0.0)
    self.assertGreater(intercept[1].item(), 0.0)
    self.assertGreater(action_rate[1].item(), 0.0)

  def test_delayed_launch_flag_defines_idle_and_active_stage(self):
    joint_pos = torch.zeros(2, len(_GK_JOINT_NAMES))
    env = _make_env(
      root_z=[0.80, 0.80],
      joint_pos=joint_pos,
      ball_velocities=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
      ],
      ball_positions=[
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.5],
      ],
      body_link_pos_w=[
        [[0.0, 0.0, 0.5], [0.0, 0.2, 0.5]],
        [[0.0, 0.0, 0.5], [0.0, 0.2, 0.5]],
      ],
      delayed_ball_launched=[False, True],
    )

    idle_height = goalkeeper_idle_low_base_height(env, target_z=0.73)
    idle_alive = goalkeeper_idle_alive(env)
    active_body = goalkeeper_active_body_intercept(env)

    self.assertGreater(idle_height[0].item(), 0.0)
    self.assertEqual(idle_alive[0].item(), 1.0)
    self.assertEqual(active_body[0].item(), 0.0)
    self.assertEqual(idle_height[1].item(), 0.0)
    self.assertEqual(idle_alive[1].item(), 0.0)
    self.assertGreater(active_body[1].item(), 0.0)

  def test_idle_action_rate_penalizes_only_prepare_stage(self):
    joint_pos = torch.zeros(2, len(_GK_JOINT_NAMES))
    env = _make_env(
      root_z=[0.80, 0.80],
      joint_pos=joint_pos,
      ball_velocities=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
      ],
      delayed_ball_launched=[False, True],
    )

    reward = goalkeeper_idle_action_rate(env)

    self.assertGreater(reward[0].item(), 0.0)
    self.assertEqual(reward[1].item(), 0.0)

  def test_idle_upright_and_base_still_apply_only_prepare_stage(self):
    joint_pos = torch.zeros(3, len(_GK_JOINT_NAMES))
    env = _make_env(
      root_z=[0.80, 0.80, 0.80],
      joint_pos=joint_pos,
      ball_velocities=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
      ],
      projected_gravity_b=[
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -0.2],
        [0.0, 0.0, -1.0],
      ],
      delayed_ball_launched=[False, False, True],
    )
    robot = env.scene["robot"]
    robot.data.root_link_lin_vel_w[0, 0] = 2.0
    robot.data.root_link_ang_vel_w[0, 1] = 3.0

    upright = goalkeeper_idle_upright(env)
    still = goalkeeper_idle_base_still(env)

    self.assertGreater(upright[0].item(), 0.95)
    self.assertLess(upright[1].item(), 0.25)
    self.assertEqual(upright[2].item(), 0.0)
    self.assertGreater(still[0].item(), 0.0)
    self.assertEqual(still[1].item(), 0.0)
    self.assertEqual(still[2].item(), 0.0)

  def test_active_fall_guardrails_apply_only_after_launch(self):
    joint_pos = torch.zeros(3, len(_GK_JOINT_NAMES))
    env = _make_env(
      root_z=[0.80, 0.80, 0.80],
      joint_pos=joint_pos,
      ball_velocities=[
        [0.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
      ],
      projected_gravity_b=[
        [0.0, 0.0, -0.1],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -0.1],
      ],
      delayed_ball_launched=[False, True, True],
    )

    upright = goalkeeper_active_upright(env)
    fall = goalkeeper_active_fall_penalty(env)

    self.assertEqual(upright[0].item(), 0.0)
    self.assertEqual(fall[0].item(), 0.0)
    self.assertGreater(upright[1].item(), 0.95)
    self.assertEqual(fall[1].item(), 0.0)
    self.assertLess(upright[2].item(), 0.2)
    self.assertEqual(fall[2].item(), 1.0)

  def test_condition_target_reach_uses_predicted_landing_only_when_active(self):
    joint_pos = torch.zeros(3, len(_GK_JOINT_NAMES))
    env = _make_env(
      root_z=[0.80, 0.80, 0.80],
      joint_pos=joint_pos,
      ball_velocities=[
        [-2.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
      ],
      ball_positions=[
        [3.0, 0.0, 0.5],
        [3.0, 0.0, 0.5],
        [3.0, 0.0, 0.5],
      ],
      body_link_pos_w=[
        [[0.0, 0.55, 2.00], [0.0, -0.40, 1.40], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [[0.0, -0.40, 1.40], [0.0, 0.55, 2.00], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.55, 2.00], [0.0, -0.40, 1.40], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
      ],
      delayed_ball_launched=[True, True, False],
    )
    env._gk_predicted_ball_condition = torch.tensor([
      [0.55, 1.20, 0.40, 0.0],
      [0.55, 1.20, 0.40, 0.0],
      [0.55, 1.20, 0.40, 1.0],
    ])

    reward = goalkeeper_active_condition_target_reach(env, std=0.2)

    self.assertGreater(reward[0].item(), 0.95)
    self.assertLess(reward[1].item(), 0.01)
    self.assertEqual(reward[2].item(), 0.0)


if __name__ == "__main__":
  unittest.main()
