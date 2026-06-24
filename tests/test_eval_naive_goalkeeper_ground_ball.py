"""Tests for eval_naive_goalkeeper ground-ball sampling mode."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeData:
  default_root_state = torch.tensor([[0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]])
  root_link_vel_w = torch.zeros(1, 6)


class _FakeBall:
  data = _FakeData()

  def write_root_link_pose_to_sim(self, pose, env_ids=None):
    del env_ids
    self.pose = pose.clone()

  def write_root_link_velocity_to_sim(self, vel, env_ids=None):
    del env_ids
    self.velocity = vel.clone()


class _FakeScene(dict):
  def __init__(self):
    super().__init__(ball=_FakeBall())
    self.env_origins = torch.zeros(1, 3)


class _FakeEnv:
  num_envs = 1
  device = "cpu"

  def __init__(self):
    self.scene = _FakeScene()


class _FakeWrappedEnv:
  class _Unwrapped:
    num_envs = 1
    device = "cpu"

    def __init__(self):
      self.scene = _FakeScene()

  def __init__(self):
    self.unwrapped = self._Unwrapped()


def _load_eval_script():
  path = REPO_ROOT / "scripts" / "eval_naive_goalkeeper.py"
  spec = importlib.util.spec_from_file_location("eval_naive_goalkeeper", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


class EvalNaiveGoalkeeperGroundBallTest(unittest.TestCase):
  def test_ground_ball_mode_replaces_only_eval_reset_sampler(self):
    script = _load_eval_script()
    cfg = load_env_cfg("Eval-Goalkeeper", play=False)

    self.assertEqual(cfg.events["reset_ball"].func.__name__, "reset_ball_with_parabolic_trajectory")

    script._apply_ground_ball_sampling(cfg, speed_min=3.0, speed_max=4.0)

    event = cfg.events["reset_ball"]
    self.assertEqual(event.func.__name__, "reset_ball_with_ground_trajectory")
    self.assertEqual(event.params["vel_cfg"].speed_range, (3.0, 4.0))
    self.assertEqual(event.params["vel_cfg"].ball_start_z, 0.1)

  def test_delayed_launch_mode_stages_compete_ball_then_launches_later(self):
    script = _load_eval_script()
    cfg = load_env_cfg("Eval-Goalkeeper", play=False)
    old_episode_length_s = cfg.episode_length_s

    script._apply_delayed_launch_sampling(cfg, wait_s=3.0)

    reset_event = cfg.events["reset_ball"]
    launch_event = cfg.events["launch_delayed_ball"]
    self.assertEqual(reset_event.func.__name__, "reset_ball_staged_delayed_launch")
    self.assertEqual(launch_event.func.__name__, "launch_staged_ball_after_delay")
    self.assertEqual(launch_event.mode, "step")
    self.assertEqual(launch_event.params["wait_s"], 3.0)
    self.assertEqual(reset_event.params["ball_pos"], (3.0, 0.0, 0.1))
    self.assertEqual(reset_event.params["sampler_params"]["fixed_start_local"], (3.0, 0.0, 0.1))
    self.assertAlmostEqual(cfg.episode_length_s, old_episode_length_s + 3.0)

  def test_delayed_launch_uses_explicit_sampled_velocity_cache(self):
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from src.tasks.soccer.mdp.goalkeeper_ball_reset import reset_ball_staged_delayed_launch

    def fake_sampler(env, env_ids, **_params):
      vel = torch.tensor([[-5.0, 0.25, 1.0, 0.0, 0.0, 0.0]])
      env._gk_sampled_ball_velocity = vel.clone()

    env = _FakeEnv()
    reset_ball_staged_delayed_launch(
      env,
      torch.tensor([0]),
      sampler_func=fake_sampler,
      sampler_params={},
      ball_cfg=SceneEntityCfg("ball"),
    )

    self.assertTrue(torch.allclose(env.scene["ball"].pose[0, :3], torch.tensor([3.0, 0.0, 0.1])))
    self.assertTrue(torch.allclose(env.scene["ball"].velocity[0, :3], torch.zeros(3)))
    self.assertTrue(torch.allclose(env._gk_delayed_ball_velocity[0, :3], torch.tensor([-5.0, 0.25, 1.0])))

  def test_fixed_start_parabolic_resamples_velocity_to_sampled_target(self):
    from src.tasks.soccer.mdp.goalkeeper_ball_reset import (
      RegionBallVelCfg,
      reset_ball_with_parabolic_trajectory,
    )

    env = _FakeEnv()
    vel_cfg = RegionBallVelCfg(
      ball_end_x_range=(0.5, 0.5),
      t_flight_range=(1.0, 1.0),
      regions=[{"height": (1.0, 1.0), "width": (1.0, 1.0)}],
    )
    reset_ball_with_parabolic_trajectory(
      env,
      torch.tensor([0]),
      vel_cfg=vel_cfg,
      fixed_start_local=(3.0, 0.0, 0.1),
    )

    self.assertTrue(torch.allclose(env.scene["ball"].pose[0, :3], torch.tensor([3.0, 0.0, 0.1])))
    self.assertTrue(torch.allclose(env._gk_sampled_ball_velocity[0, :3], torch.tensor([-3.5, 1.0, 5.805]), atol=1e-5))
    self.assertTrue(torch.allclose(env._gk_ball_end_pos[0], torch.tensor([-0.5, 1.0, 1.0])))

  def test_fixed_start_ground_ball_resamples_velocity_to_sampled_target(self):
    from src.tasks.soccer.mdp.goalkeeper_ball_reset import (
      GroundBallVelCfg,
      reset_ball_with_ground_trajectory,
    )

    env = _FakeEnv()
    reset_ball_with_ground_trajectory(
      env,
      torch.tensor([0]),
      vel_cfg=GroundBallVelCfg(
        ball_end_x_range=(0.5, 0.5),
        speed_range=(2.0, 2.0),
        y_range=(1.0, 1.0),
      ),
      fixed_start_local=(3.0, 0.0, 0.1),
    )

    self.assertTrue(torch.allclose(env.scene["ball"].pose[0, :3], torch.tensor([3.0, 0.0, 0.1])))
    self.assertGreater(env._gk_sampled_ball_velocity[0, 1].item(), 0.0)

  def test_actor_critic_checkpoint_loads_with_goalkeeper_runner(self):
    script = _load_eval_script()
    calls = []

    class FakeRunner:
      def __init__(self, env, cfg, device):
        del env, cfg, device
        calls.append("goalkeeper")

      def load(self, path, load_cfg=None):
        calls.append((path, load_cfg))

      def get_inference_policy(self, device):
        del device
        return lambda obs: torch.zeros(1, 29)

    ckpt = {"actor_state_dict": {"history_encoder.0.weight": torch.zeros(1)}}
    with patch.object(script, "GoalkeeperRunner", FakeRunner), \
        patch.object(script.torch, "load", return_value=ckpt):
      policy = script._load_policy("keeper_idle.pt", env=object(), device="cpu")

    self.assertTrue(callable(policy))
    self.assertEqual(calls, ["goalkeeper", ("keeper_idle.pt", {"actor": True})])

  def test_moe_bundle_loads_actor_critic_idle_expert_with_goalkeeper_runner(self):
    script = _load_eval_script()
    calls = []

    class FakeMlpRunner:
      def __init__(self, env, cfg, device):
        del env, cfg, device
        calls.append("mlp")

      def load(self, path, load_cfg=None):
        del path
        calls.append(("mlp_load", load_cfg))

      def get_inference_policy(self, device):
        del device
        return lambda obs: torch.zeros(obs["actor"].shape[0], 29)

    class FakeGoalkeeperRunner(FakeMlpRunner):
      def __init__(self, env, cfg, device):
        del env, cfg, device
        calls.append("goalkeeper")

      def load(self, path, load_cfg=None):
        del path
        calls.append(("goalkeeper_load", load_cfg))

    bundle = {
      "sr": [{"actor_state_dict": {"mlp.0.weight": torch.zeros(1)}} for _ in range(6)],
      "idle": {"actor_state_dict": {"history_encoder.0.weight": torch.zeros(1)}},
    }
    with patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.MjlabOnPolicyRunner", FakeMlpRunner), \
        patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.GoalkeeperRunner", FakeGoalkeeperRunner), \
        patch.object(script.torch, "load", return_value=bundle):
      policy = script._load_policy("moe7.pt", env=_FakeWrappedEnv(), device="cpu")

    self.assertTrue(callable(policy))
    self.assertEqual(calls.count("mlp"), 6)
    self.assertEqual(calls.count("goalkeeper"), 1)

  def test_moe_bundle_loads_with_shared_moe6_policy_adapter(self):
    script = _load_eval_script()
    bundle = {
      "moe6": True,
      "sr": [
        {
          "actor_state_dict": {
            "_ballistic_marker": torch.ones(1),
            "base.mlp.0.weight": torch.zeros(1),
            "residual.0.weight": torch.zeros(1),
          },
          "ballistic_residual": {"residual_scale": 0.3},
        }
        for _ in range(6)
      ],
    }

    class FakeMoE6Policy:
      def __init__(self, loaded_bundle, env, device):
        self.loaded_bundle = loaded_bundle
        self.env = env
        self.device = device

      def __call__(self, obs):
        return torch.zeros(obs["actor"].shape[0], 29)

    with patch.object(script.torch, "load", return_value=bundle), \
        patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.MoE6GoalkeeperPolicy", FakeMoE6Policy):
      policy = script._load_policy("hard3.pt", env=_FakeWrappedEnv(), device="cpu")

    self.assertIsInstance(policy, FakeMoE6Policy)
    self.assertIs(policy.loaded_bundle, bundle)
    self.assertEqual(policy.device, "cpu")

  def test_training_checkpoint_with_moe_actor_state_loads_as_moe_bundle(self):
    script = _load_eval_script()
    calls = []

    class FakeMlpRunner:
      def __init__(self, env, cfg, device):
        del env, cfg, device
        calls.append("mlp")

      def load(self, path, load_cfg=None):
        del path
        calls.append(("mlp_load", load_cfg))

      def get_inference_policy(self, device):
        del device
        return lambda obs: torch.zeros(obs["actor"].shape[0], 29)

    class FakeGoalkeeperRunner(FakeMlpRunner):
      def __init__(self, env, cfg, device):
        del env, cfg, device
        calls.append("goalkeeper")

      def load(self, path, load_cfg=None):
        del path
        calls.append(("goalkeeper_load", load_cfg))

    bundle = {
      "sr": [{"actor_state_dict": {"mlp.0.weight": torch.zeros(1)}} for _ in range(6)],
      "idle": {"actor_state_dict": {"history_encoder.0.weight": torch.zeros(1)}},
      "gate": {"state": {}, "mean": torch.zeros(6), "std": torch.ones(6), "num_classes": 7},
    }
    ckpt = {"actor_state_dict": bundle, "critic_state_dict": {}, "iter": 1100}
    with patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.MjlabOnPolicyRunner", FakeMlpRunner), \
        patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.GoalkeeperRunner", FakeGoalkeeperRunner), \
        patch.object(script.torch, "load", return_value=ckpt):
      policy = script._load_policy("model_1100.pt", env=_FakeWrappedEnv(), device="cpu")

    self.assertTrue(callable(policy))
    self.assertEqual(calls.count("mlp"), 6)
    self.assertEqual(calls.count("goalkeeper"), 1)

  def test_moe_actor_subcheckpoints_without_infos_are_wrapped_for_runner_load(self):
    script = _load_eval_script()
    calls = []
    real_torch_load = torch.load

    class FakeMlpRunner:
      def __init__(self, env, cfg, device):
        del env, cfg, device
        calls.append("mlp")

      def load(self, path, load_cfg=None):
        loaded = real_torch_load(path, map_location="cpu", weights_only=False)
        loaded["infos"]
        calls.append(("mlp_load", load_cfg))

      def get_inference_policy(self, device):
        del device
        return lambda obs: torch.zeros(obs["actor"].shape[0], 29)

    class FakeGoalkeeperRunner(FakeMlpRunner):
      def __init__(self, env, cfg, device):
        del env, cfg, device
        calls.append("goalkeeper")

      def load(self, path, load_cfg=None):
        loaded = real_torch_load(path, map_location="cpu", weights_only=False)
        loaded["infos"]
        calls.append(("goalkeeper_load", load_cfg))

    bundle = {
      "sr": [{"actor_state_dict": {"mlp.0.weight": torch.zeros(1)}} for _ in range(6)],
      "idle": {"actor_state_dict": {"history_encoder.0.weight": torch.zeros(1)}},
    }
    ckpt = {"actor_state_dict": bundle}

    def fake_load(path, *args, **kwargs):
      return ckpt if path == "model_1100.pt" else real_torch_load(path, *args, **kwargs)

    with patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.MjlabOnPolicyRunner", FakeMlpRunner), \
        patch("src.tasks.soccer.modules.moe6_goalkeeper_policy.GoalkeeperRunner", FakeGoalkeeperRunner), \
        patch.object(script.torch, "load", side_effect=fake_load):
      policy = script._load_policy("model_1100.pt", env=_FakeWrappedEnv(), device="cpu")

    self.assertTrue(callable(policy))
    self.assertEqual(calls.count("mlp"), 6)
    self.assertEqual(calls.count("goalkeeper"), 1)

  def test_goalkeeper_student_checkpoint_loads_position_conditioned_wrapper(self):
    script = _load_eval_script()
    calls = []

    class FakeActor:
      def __init__(self, *args, **kwargs):
        calls.append(("actor_init", kwargs["output_dim"]))

      def load_state_dict(self, state_dict):
        calls.append(("load_state_dict", sorted(state_dict.keys())))

      def eval(self):
        calls.append("eval")
        return self

      def to(self, device):
        calls.append(("to", device))
        return self

    ckpt = {
      "actor_state_dict": {
        "rnn.rnn.weight_ih_l0": torch.zeros(1),
        "condition_encoder.0.weight": torch.zeros(1),
        "film.weight": torch.zeros(1),
      },
      "config": {"hidden_dims": [128, 64, 32]},
    }
    with patch.object(script, "GoalkeeperStudentFiLMActor", FakeActor), \
        patch.object(script.torch, "load", return_value=ckpt):
      policy = script._load_policy("student.pt", env=_FakeWrappedEnv(), device="cpu")

    self.assertIsInstance(policy, script.GoalkeeperStudentPolicy)
    self.assertEqual(calls[0], ("actor_init", 29))
    self.assertTrue(any(call[0] == "load_state_dict" for call in calls if isinstance(call, tuple)))
    self.assertIn("eval", calls)

  def test_goalkeeper_student_checkpoint_infers_large_architecture_without_config(self):
    script = _load_eval_script()
    calls = []

    class FakeActor:
      def __init__(self, *args, **kwargs):
        calls.append(("actor_init", kwargs))

      def load_state_dict(self, state_dict):
        calls.append(("load_state_dict", sorted(state_dict.keys())))

      def eval(self):
        return self

      def to(self, device):
        return self

    actor_state = {
      "rnn.rnn.weight_ih_l0": torch.zeros(4 * 640, 960),
      "rnn.rnn.weight_hh_l0": torch.zeros(4 * 640, 640),
      "rnn.rnn.weight_ih_l1": torch.zeros(4 * 640, 640),
      "condition_encoder.0.weight": torch.zeros(128, 4),
      "film.weight": torch.zeros(2 * 640, 128),
      "mlp.0.weight": torch.zeros(512, 640),
      "mlp.2.weight": torch.zeros(256, 512),
      "mlp.4.weight": torch.zeros(128, 256),
      "mlp.6.weight": torch.zeros(29, 128),
    }
    ckpt = {"actor_state_dict": actor_state}

    with patch.object(script, "GoalkeeperStudentFiLMActor", FakeActor), \
        patch.object(script.torch, "load", return_value=ckpt):
      script._load_policy("student_large.pt", env=_FakeWrappedEnv(), device="cpu")

    kwargs = calls[0][1]
    self.assertEqual(kwargs["rnn_hidden_dim"], 640)
    self.assertEqual(kwargs["rnn_num_layers"], 2)
    self.assertEqual(kwargs["condition_hidden_dim"], 128)
    self.assertEqual(kwargs["hidden_dims"], (512, 256, 128))

  def test_policy_reset_hook_resets_policy_on_env_reset_and_done_step(self):
    script = _load_eval_script()

    class FakeEnv:
      def __init__(self):
        self.reset_calls = 0
        self.step_calls = 0

      def reset(self):
        self.reset_calls += 1
        return "obs"

      def step(self, action):
        del action
        self.step_calls += 1
        done = torch.tensor([self.step_calls == 2])
        return "obs", torch.zeros(1), done, {}

    class FakePolicy:
      def __init__(self):
        self.reset_calls = 0

      def reset(self):
        self.reset_calls += 1

    env = FakeEnv()
    policy = FakePolicy()

    script._attach_policy_reset_to_env(env, policy)

    self.assertEqual(env.reset(), "obs")
    self.assertEqual(policy.reset_calls, 1)
    env.step(torch.zeros(1))
    self.assertEqual(policy.reset_calls, 1)
    env.step(torch.zeros(1))
    self.assertEqual(policy.reset_calls, 2)


if __name__ == "__main__":
  unittest.main()
