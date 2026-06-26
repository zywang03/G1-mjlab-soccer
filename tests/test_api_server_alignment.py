"""Tests for API-server compete/eval observation alignment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_api_server():
  path = REPO_ROOT / "scripts" / "api_server.py"
  spec = importlib.util.spec_from_file_location("api_server", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


class _FakeEntity:
  def __init__(self):
    self.root_state = None
    self.joint_pos = None
    self.joint_vel = None
    self.cleared = False

  def write_root_state_to_sim(self, root_state, env_ids=None):
    del env_ids
    self.root_state = root_state.clone()

  def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids=None):
    del env_ids
    self.joint_pos = joint_pos.clone()
    self.joint_vel = joint_vel.clone()

  def clear_state(self, env_ids=None):
    del env_ids
    self.cleared = True


class _FakeScene(dict):
  def __init__(self, command):
    super().__init__(robot=_FakeEntity(), ball=_FakeEntity())
    self.env_origins = torch.zeros(1, 3)
    self.command = command
    self.wrote = False

  def write_data_to_sim(self):
    self.wrote = True


class _FakeCommand:
  def __init__(self):
    self.target_point_pos = torch.zeros(1, 3)
    self.target_destination_pos = torch.zeros(1, 3)
    self.relative_updated = False
    self.advanced = False
    self.scene = None

  def _update_target_points_from_sim(self):
    self.target_point_pos[:] = self.scene["ball"].root_state[:, :3]

  def _compute_relative_transforms(self):
    self.relative_updated = True

  def _update_command(self):
    self.advanced = True


class _FakeCommandManager:
  def __init__(self, command):
    self.command = command

  def get_term(self, name):
    assert name == "motion"
    return self.command


class _FakeActionTerm:
  def __init__(self):
    self._offset = torch.tensor([[1.0, 2.0, 3.0]])
    self._scale = torch.tensor([[0.5, 0.25, 1.0]])
    self._raw_actions = torch.zeros(1, 3)

  @property
  def offset(self):
    return self._offset

  @property
  def scale(self):
    return self._scale


class _FakeActionManager:
  def __init__(self):
    self._action = torch.zeros(1, 3)
    self.term = _FakeActionTerm()

  def get_term(self, name):
    assert name == "joint_pos"
    return self.term


class _FakeObservationManager:
  def __init__(self):
    self.called = None
    self.obs_dim = 160

  def compute_group(self, group_name, update_history=False):
    self.called = (group_name, update_history)
    return torch.arange(self.obs_dim, dtype=torch.float32).view(1, self.obs_dim)


class _FakeSim:
  def __init__(self):
    self.forward_called = False
    self.sense_called = False

  def forward(self):
    self.forward_called = True

  def sense(self):
    self.sense_called = True


class _FakeBaseEnv:
  def __init__(self):
    self.command = _FakeCommand()
    self.scene = _FakeScene(self.command)
    self.command.scene = self.scene
    self.command_manager = _FakeCommandManager(self.command)
    self.action_manager = _FakeActionManager()
    self.observation_manager = _FakeObservationManager()
    self.sim = _FakeSim()
    self.reset_called = False

  def reset(self):
    self.reset_called = True
    return {"actor": torch.zeros(1, 160)}, {}


class _FakeEnv:
  def __init__(self):
    self.unwrapped = _FakeBaseEnv()

  def reset(self):
    return self.unwrapped.reset()


class _FakeCfgObservations:
  history_length = 1

  def __init__(self):
    self.terms = {"base": object(), "opponent_root": object(), "opponent_joints": object()}


class _FakeEnvCfg:
  def __init__(self):
    self.scene = type("SceneCfg", (), {"num_envs": 1})()
    self.observations = {"actor": _FakeCfgObservations()}


class _FakeDualActionTerm:
  def __init__(self, action_dim):
    self.action_dim = action_dim
    self._raw_actions = torch.zeros(1, action_dim)


class _FakeDualActionManager:
  def __init__(self):
    self._terms = {
      "joint_pos": _FakeDualActionTerm(29),
      "opponent_joint_pos": _FakeDualActionTerm(29),
    }
    self._action = torch.zeros(1, 58)

  def get_term(self, name):
    return self._terms[name]


class _FakeDualBaseEnv:
  def __init__(self):
    self.action_manager = _FakeDualActionManager()


class _FakeDualWrappedEnv:
  num_actions = 58

  def __init__(self, base_env, clip_actions=None):
    del clip_actions
    self.unwrapped = base_env

  def close(self):
    pass


class _FakeRunner:
  calls = []

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    self.calls.append({
      "env_num_actions": env.num_actions,
      "actor_class_name": train_cfg["actor"].get("class_name"),
      "log_dir": log_dir,
      "device": device,
    })

  def load(self, checkpoint, load_cfg=None, map_location=None):
    self.calls[-1]["checkpoint"] = checkpoint
    self.calls[-1]["load_cfg"] = load_cfg
    self.calls[-1]["map_location"] = map_location

  def get_inference_policy(self, device="cpu"):
    self.calls[-1]["policy_device"] = device
    return lambda obs: torch.zeros(1, 29)


class _FakeRobotData:
  default_joint_pos = torch.zeros(1, 29)


class _FakeRobot:
  data = _FakeRobotData()


class _FakeAppBaseEnv:
  def __init__(self):
    self.scene = {"robot": _FakeRobot()}


class _FakeAppEnv:
  num_actions = 58

  def __init__(self):
    self.unwrapped = _FakeAppBaseEnv()

  def close(self):
    pass


class _FakeExpert:
  def __init__(self, value):
    self.value = value
    self.calls = 0
    self.reset_calls = 0

  def __call__(self, obs):
    del obs
    self.calls += 1
    return torch.full((1, 3), self.value, dtype=torch.float32)

  def reset(self):
    self.reset_calls += 1


class ApiServerAlignmentTest(unittest.TestCase):
  def test_goalkeeper_adapter_uses_eval_obs_manager_history_layout(self):
    api = _load_api_server()
    env = _FakeEnv()
    env.unwrapped.observation_manager.obs_dim = 960
    adapter = api.GoalkeeperEvalObsAdapter(env, device="cpu")
    raw_state = {
      "shooter": {},
      "goalkeeper": {
        "root_pos": [0.0, 0.0, 0.8],
        "root_quat": [1.0, 0.0, 0.0, 0.0],
        "root_lin_vel": [0.0, 0.0, 0.0],
        "root_ang_vel": [0.1, 0.2, 0.3],
        "joint_pos": [0.1] * 29,
        "joint_vel": [0.2] * 29,
        "last_action": [0.3] * 3,
      },
      "ball": {"pos": [2.0, 0.4, 0.6], "vel": [-5.0, 0.1, 1.2]},
    }

    obs = adapter.compute_obs(raw_state)

    self.assertEqual(tuple(obs.shape), (1, 960))
    self.assertEqual(env.unwrapped.observation_manager.called, ("actor", True))
    self.assertTrue(env.unwrapped.scene.wrote)
    self.assertTrue(env.unwrapped.sim.forward_called)
    self.assertTrue(env.unwrapped.sim.sense_called)
    self.assertTrue(torch.allclose(
      env.unwrapped.scene["robot"].root_state[:, :3],
      torch.tensor([[0.0, 0.0, 0.8]]),
    ))
    self.assertTrue(torch.allclose(
      env.unwrapped.scene["ball"].root_state[:, :3],
      torch.tensor([[2.0, 0.4, 0.6]]),
    ))
    self.assertTrue(torch.allclose(
      env.unwrapped.action_manager._action,
      torch.full((1, 3), 0.3),
    ))
    self.assertTrue(torch.allclose(
      env.unwrapped.action_manager.get_term("joint_pos")._raw_actions,
      torch.full((1, 3), 0.3),
    ))

  def test_goalkeeper_adapter_syncs_shooter_into_adversarial_opponent_entity(self):
    api = _load_api_server()
    env = _FakeEnv()
    env.unwrapped.scene["opponent"] = _FakeEntity()
    env.unwrapped.observation_manager.obs_dim = 1630
    adapter = api.GoalkeeperEvalObsAdapter(env, device="cpu")
    raw_state = {
      "shooter": {
        "root_pos": [3.5, -0.1, 0.85],
        "root_quat": [1.0, 0.0, 0.0, 0.0],
        "root_lin_vel": [0.1, 0.2, 0.3],
        "root_ang_vel": [0.4, 0.5, 0.6],
        "joint_pos": [0.7, 0.8, 0.9],
        "joint_vel": [1.0, 1.1, 1.2],
        "last_action": [0.0, 0.0, 0.0],
      },
      "goalkeeper": {
        "root_pos": [0.0, 0.0, 0.8],
        "root_quat": [1.0, 0.0, 0.0, 0.0],
        "root_lin_vel": [0.0, 0.0, 0.0],
        "root_ang_vel": [0.1, 0.2, 0.3],
        "joint_pos": [0.1] * 3,
        "joint_vel": [0.2] * 3,
        "last_action": [0.3] * 3,
      },
      "ball": {"pos": [2.0, 0.4, 0.6], "vel": [-5.0, 0.1, 1.2]},
    }

    adapter.compute_obs(raw_state)

    opponent = env.unwrapped.scene["opponent"]
    self.assertTrue(torch.allclose(opponent.root_state[:, :3], torch.tensor([[3.5, -0.1, 0.85]])))
    self.assertTrue(torch.allclose(opponent.joint_pos, torch.tensor([[0.7, 0.8, 0.9]])))

  def test_adversarial_goalkeeper_checkpoint_uses_adversarial_runner(self):
    api = _load_api_server()
    task_ids = []
    _FakeRunner.calls.clear()

    def fake_load_env_cfg(task_id, play=False):
      del play
      task_ids.append(task_id)
      return _FakeEnvCfg()

    ckpt = {"actor_state_dict": {"actor_residual.0.weight": torch.zeros(1)}}
    with patch.object(api, "load_env_cfg", side_effect=fake_load_env_cfg), \
        patch.object(api, "ManagerBasedRlEnv", side_effect=lambda cfg, device: _FakeDualBaseEnv()), \
        patch.object(api, "RslRlVecEnvWrapper", side_effect=lambda env, clip_actions: _FakeDualWrappedEnv(env, clip_actions)), \
        patch.object(api.torch, "load", return_value=ckpt), \
        patch("src.tasks.soccer.config.g1.rl_cfg.AdversarialGoalkeeperRunner", _FakeRunner):
      policy, env, resolved_task_id = api._load_policy("keeper_candidate.pt", "Eval-Goalkeeper", "cpu")

    self.assertTrue(callable(policy))
    self.assertEqual(env.num_actions, 58)
    self.assertEqual(resolved_task_id, "Unitree-G1-Goalkeeper-Adversarial")
    self.assertEqual(task_ids, ["Unitree-G1-Goalkeeper-Adversarial"])
    self.assertEqual(_FakeRunner.calls[0]["env_num_actions"], 29)
    self.assertEqual(_FakeRunner.calls[0]["actor_class_name"], "AdversarialGoalkeeperActorCritic")

  def test_goalkeeper_training_checkpoint_with_moe_actor_state_loads_moe_policy(self):
    api = _load_api_server()
    captured = {}

    class FakeMoEPolicy:
      def __init__(self, bundle, env, device, idle_env=None):
        captured["bundle"] = bundle
        captured["env"] = env
        captured["device"] = device
        captured["idle_env"] = idle_env

      def reset(self):
        pass

    bundle = {
      "sr": [{"actor_state_dict": {"mlp.0.weight": torch.zeros(1)}} for _ in range(6)],
      "idle": {"actor_state_dict": {"actor_residual.0.weight": torch.zeros(1)}},
      "gate": {"state": {}, "mean": torch.zeros(6), "std": torch.ones(6), "num_classes": 7},
    }
    ckpt = {"actor_state_dict": bundle}

    with patch.object(api, "load_env_cfg", return_value=_FakeEnvCfg()), \
        patch.object(api, "ManagerBasedRlEnv", side_effect=lambda cfg, device: _FakeDualBaseEnv()), \
        patch.object(api, "RslRlVecEnvWrapper", side_effect=lambda env, clip_actions: _FakeDualWrappedEnv(env, clip_actions)), \
        patch.object(api.torch, "load", return_value=ckpt), \
        patch.object(api, "ApiMoE6Policy", FakeMoEPolicy), \
        patch("mjlab.rl.MjlabOnPolicyRunner", side_effect=AssertionError("MoE7 must not load as native MLP")):
      policy, env, resolved_task_id = api._load_policy("model_1100.pt", "Eval-Goalkeeper", "cpu")

    self.assertIsInstance(policy, FakeMoEPolicy)
    self.assertIs(captured["bundle"], bundle)
    self.assertEqual(resolved_task_id, "Eval-Goalkeeper")
    self.assertIsNotNone(captured["idle_env"])
    self.assertEqual(env.num_actions, 58)

  def test_goalkeeper_residual_moe_bundle_loads_moe_policy(self):
    api = _load_api_server()
    captured = {}

    class FakeMoEPolicy:
      def __init__(self, bundle, env, device, idle_env=None):
        captured["bundle"] = bundle
        captured["env"] = env
        captured["device"] = device
        captured["idle_env"] = idle_env

      def reset(self):
        pass

    base_bundle = {
      "sr": [{"actor_state_dict": {"mlp.0.weight": torch.zeros(1)}} for _ in range(6)],
      "z_low": 0.85,
      "z_up": 1.35,
      "vz_low": -5.0,
      "latch_hi": 3.5,
    }
    ckpt = {
      "moe6_residual": True,
      "base_moe6": base_bundle,
      "policy_state_dict": {"residual.0.weight": torch.zeros(1)},
      "idle": {"actor_state_dict": {"actor_residual.0.weight": torch.zeros(1)}},
      "gate": {"state": {}, "mean": torch.zeros(6), "std": torch.ones(6), "num_classes": 7},
      "idle_speed_threshold": 0.5,
      "idle_incoming_vx_threshold": -0.5,
    }

    with patch.object(api, "load_env_cfg", return_value=_FakeEnvCfg()), \
        patch.object(api, "ManagerBasedRlEnv", side_effect=lambda cfg, device: _FakeDualBaseEnv()), \
        patch.object(api, "RslRlVecEnvWrapper", side_effect=lambda env, clip_actions: _FakeDualWrappedEnv(env, clip_actions)), \
        patch.object(api.torch, "load", return_value=ckpt), \
        patch.object(api, "ApiMoE6Policy", FakeMoEPolicy), \
        patch("mjlab.rl.MjlabOnPolicyRunner", side_effect=AssertionError("Residual MoE bundle must not load as native MLP")):
      policy, env, resolved_task_id = api._load_policy("targeted_moe6_residual.pt", "Eval-Goalkeeper", "cpu")

    self.assertIsInstance(policy, FakeMoEPolicy)
    self.assertEqual(resolved_task_id, "Eval-Goalkeeper")
    self.assertIsNot(captured["bundle"], base_bundle)
    self.assertIs(captured["bundle"]["sr"], base_bundle["sr"])
    self.assertEqual(captured["bundle"]["idle_speed_threshold"], 0.5)
    self.assertEqual(captured["bundle"]["idle_incoming_vx_threshold"], -0.5)
    self.assertIn("gate", captured["bundle"])
    self.assertIsNotNone(captured["idle_env"])
    self.assertEqual(env.num_actions, 58)

  def test_create_app_treats_adversarial_goalkeeper_as_goalkeeper(self):
    api = _load_api_server()
    captured = {}

    class CaptureGoalkeeperAdapter:
      def __init__(self, env, device):
        del env, device
        captured["adapter"] = "goalkeeper"

      def reset(self):
        pass

    policy = type("Policy", (), {"reset": lambda self: None})()
    with patch.object(api, "_load_policy", return_value=(policy, _FakeAppEnv(), api._KEEPER_ADV_TASK_ID)), \
        patch.object(api, "GoalkeeperEvalObsAdapter", CaptureGoalkeeperAdapter), \
        patch.object(api, "_load_motions", side_effect=AssertionError("goalkeeper app should not load shooter motions")):
      api.create_app("keeper_candidate.pt", "Eval-Goalkeeper", "cpu")

    self.assertEqual(captured["adapter"], "goalkeeper")

  def test_moe6_policy_uses_default_expert_before_latch(self):
    api = _load_api_server()
    policy = object.__new__(api.ApiMoE6Policy)
    policy.z_low = 0.85
    policy.z_up = 1.35
    policy.vz_low = -5.0
    policy.latch_hi = 5.0
    policy.dev = torch.device("cpu")
    policy.n_envs = 1
    policy.g = 9.81
    policy.latched = torch.full((1,), -1, dtype=torch.long)
    policy.experts = [_FakeExpert(float(i + 1)) for i in range(6)]

    obs = {"actor": torch.zeros(1, 960)}
    stationary_raw = {
      "ball": {"pos": [3.0, 0.0, 0.1], "vel": [0.0, 0.0, 0.0]},
    }
    action = policy(obs, stationary_raw)

    self.assertTrue(torch.allclose(action, torch.ones(1, 3)))
    self.assertEqual(policy.latched.item(), -1)
    self.assertEqual(sum(expert.calls for expert in policy.experts), 6)

    incoming_raw = {
      "ball": {"pos": [3.0, 0.0, 1.0], "vel": [-4.0, 0.0, 0.0]},
    }
    action = policy(obs, incoming_raw)

    self.assertFalse(torch.allclose(action, torch.zeros(1, 3)))
    self.assertGreaterEqual(policy.latched.item(), 0)
    self.assertGreater(sum(expert.calls for expert in policy.experts), 0)

  def test_moe6_policy_uses_prepare_expert_for_stationary_ball_when_available(self):
    api = _load_api_server()
    policy = object.__new__(api.ApiMoE6Policy)
    policy.z_low = 0.85
    policy.z_up = 1.35
    policy.vz_low = -5.0
    policy.latch_hi = 5.0
    policy.dev = torch.device("cpu")
    policy.n_envs = 1
    policy.g = 9.81
    policy.latched = torch.full((1,), -1, dtype=torch.long)
    policy.idle_expert_index = 6
    policy.idle_speed_threshold = 0.5
    policy.idle_incoming_vx_threshold = -0.5
    policy.experts = [_FakeExpert(float(i + 1)) for i in range(7)]

    obs = {"actor": torch.zeros(1, 960)}
    stationary_raw = {
      "ball": {"pos": [3.0, 0.0, 0.1], "vel": [0.0, 0.0, 0.0]},
    }
    incoming_raw = {
      "ball": {"pos": [3.0, 0.0, 1.0], "vel": [-4.0, 0.0, 0.0]},
    }

    stationary_action = policy(obs, stationary_raw)
    incoming_action = policy(obs, incoming_raw)

    self.assertTrue(torch.allclose(stationary_action, torch.full((1, 3), 7.0)))
    self.assertFalse(torch.allclose(incoming_action, torch.full((1, 3), 7.0)))
    self.assertGreaterEqual(policy.latched.item(), 0)

  def test_moe6_reset_clears_latch_and_resets_experts(self):
    api = _load_api_server()
    policy = object.__new__(api.ApiMoE6Policy)
    policy.dev = torch.device("cpu")
    policy.n_envs = 1
    policy.latched = torch.tensor([3], dtype=torch.long)
    policy.experts = [_FakeExpert(float(i + 1)) for i in range(6)]

    policy.reset()

    self.assertEqual(policy.latched.item(), -1)
    self.assertEqual([expert.reset_calls for expert in policy.experts], [1] * 6)


if __name__ == "__main__":
  unittest.main()
