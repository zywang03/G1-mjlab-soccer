import importlib.util
import ast
import unittest
from pathlib import Path


try:
    import torch

    _SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_naive_goalkeeper.py"
    _SPEC = importlib.util.spec_from_file_location("eval_naive_goalkeeper", _SCRIPT)
    assert _SPEC is not None and _SPEC.loader is not None
    _MODULE = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_MODULE)
    _ball_entered_goal = _MODULE._ball_entered_goal
    _successful_block = _MODULE._successful_block
except ModuleNotFoundError as exc:  # pragma: no cover - base env lacks mjlab.
    if exc.name not in {"mjlab", "torch"}:
        raise
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"missing runtime dependency: {_IMPORT_ERROR}")
class GoalkeeperEvalMetricsTest(unittest.TestCase):
    def test_no_goal_counts_as_successful_block(self):
        self.assertTrue(_successful_block(ball_entered_goal=False))
        self.assertFalse(_successful_block(ball_entered_goal=True))

    def test_goal_plane_segment_crossing_detects_fast_ball_between_steps(self):
        prev_pos = torch.tensor([-0.45, 0.0, 0.8])
        curr_pos = torch.tensor([-0.55, 0.0, 0.8])

        self.assertTrue(_ball_entered_goal(curr_pos, prev_pos))

    def test_goal_plane_segment_crossing_rejects_outside_goal_frame(self):
        prev_pos = torch.tensor([-0.45, 2.0, 0.8])
        curr_pos = torch.tensor([-0.55, 2.0, 0.8])

        self.assertFalse(_ball_entered_goal(curr_pos, prev_pos))


class GoalkeeperEvalCheckpointLoadTest(unittest.TestCase):
    def test_runner_checkpoint_fallback_maps_checkpoint_to_requested_device(self):
        tree = ast.parse(_SCRIPT.read_text())
        runner_load_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "runner"
        ]

        self.assertTrue(runner_load_calls)
        self.assertTrue(
            any(
                keyword.arg == "map_location"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "device"
                for call in runner_load_calls
                for keyword in call.keywords
            )
        )


if __name__ == "__main__":
    unittest.main()
