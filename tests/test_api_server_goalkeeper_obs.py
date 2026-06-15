import ast
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local base env has no torch.
    torch = None


@unittest.skipIf(torch is None, "torch is required for API obs tests")
class ApiServerGoalkeeperObsTest(unittest.TestCase):
    _SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "api_server.py"

    def test_goalkeeper_raw_state_builds_96d_float_obs_with_official_term_order(self):
        from scripts.api_server import compute_goalkeeper_obs

        raw_state = {
            "goalkeeper": {
                "root_quat": [1, 0, 0, 0],
                "root_ang_vel": [0, 0, 0],
                "joint_pos": [0] * 29,
                "joint_vel": [0] * 29,
                "root_pos": [0, 0, 0],
                "last_action": [0] * 29,
            },
            "ball": {"pos": [2, 1, 1]},
        }

        obs = compute_goalkeeper_obs(raw_state)

        self.assertEqual(obs.shape, (1, 96))
        self.assertEqual(obs.dtype, torch.float32)
        self.assertTrue(torch.allclose(obs[:, :3], torch.tensor([[2.0, 1.0, 1.0]])))
        self.assertTrue(torch.allclose(obs[:, 3:6], torch.zeros(1, 3)))

    def test_goalkeeper_history_stack_matches_training_term_major_layout(self):
        from scripts.api_server import stack_goalkeeper_history
        from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic

        frames = []
        for i in range(10):
            frame = torch.zeros(1, 96)
            frame[:, 0:3] = torch.tensor([[10.0 + i, 20.0 + i, 30.0 + i]])
            frame[:, 3:6] = 100.0 + i
            frame[:, 6:9] = 200.0 + i
            frame[:, 9:38] = 300.0 + i
            frame[:, 38:67] = 400.0 + i
            frame[:, 67:96] = 500.0 + i
            frames.append(frame)

        stacked = stack_goalkeeper_history(frames)
        model = GoalkeeperActorCritic(
            {"actor": torch.zeros(1, 960), "critic": torch.zeros(1, 113)},
            {"actor": ["actor"], "critic": ["critic"]},
            "actor",
            29,
        )
        reordered = model._reorder_obs_history(stacked)

        self.assertEqual(stacked.shape, (1, 960))
        self.assertTrue(torch.allclose(stacked[:, :3], frames[0][:, 0:3]))
        self.assertTrue(torch.allclose(stacked[:, 27:30], frames[-1][:, 0:3]))
        self.assertTrue(torch.allclose(stacked[:, 30:33], frames[0][:, 3:6]))
        self.assertTrue(torch.allclose(stacked[:, 57:60], frames[-1][:, 3:6]))
        self.assertTrue(torch.allclose(reordered[:, -96:-93], frames[-1][:, 0:3]))

    def test_goalkeeper_api_runner_fallback_maps_checkpoint_to_requested_device(self):
        tree = ast.parse(self._SCRIPT.read_text())
        runner_load_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "runner"
        ]

        self.assertTrue(
            any(
                keyword.arg == "map_location"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "device"
                for call in runner_load_calls
                for keyword in call.keywords
            )
        )

    def test_api_moves_stacked_observation_to_policy_device_before_inference(self):
        tree = ast.parse(self._SCRIPT.read_text())
        to_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "stacked"
        ]

        self.assertTrue(
            any(
                any(keyword.arg == "device" for keyword in call.keywords)
                and any(keyword.arg == "dtype" for keyword in call.keywords)
                for call in to_calls
            )
        )


if __name__ == "__main__":
    unittest.main()
