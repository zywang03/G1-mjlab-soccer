import ast
import inspect
import unittest
from dataclasses import asdict
from pathlib import Path


REGISTRATION_FILE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tasks"
    / "soccer"
    / "config"
    / "g1"
    / "__init__.py"
)
SHOOTER_COMMANDS_FILE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tasks"
    / "soccer"
    / "mdp"
    / "shooter_commands.py"
)
EVAL_REGISTRATION_FILE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tasks"
    / "soccer"
    / "config"
    / "eval"
    / "__init__.py"
)


def _registered_tasks() -> dict[str, dict[str, str]]:
    tree = ast.parse(REGISTRATION_FILE.read_text())
    return _registrations_from_tree(tree)


def _registered_eval_tasks() -> dict[str, dict[str, str]]:
    tree = ast.parse(EVAL_REGISTRATION_FILE.read_text())
    return _registrations_from_tree(tree)


def _registrations_from_tree(tree: ast.AST) -> dict[str, dict[str, str]]:
    registrations: dict[str, dict[str, str]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "register_mjlab_task":
            continue

        fields: dict[str, str] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            if isinstance(keyword.value, ast.Constant):
                fields[keyword.arg] = str(keyword.value.value)
            elif isinstance(keyword.value, ast.Call):
                if isinstance(keyword.value.func, ast.Name):
                    fields[keyword.arg] = keyword.value.func.id
                elif isinstance(keyword.value.func, ast.Attribute):
                    fields[keyword.arg] = keyword.value.func.attr
            elif isinstance(keyword.value, ast.Name):
                fields[keyword.arg] = keyword.value.id

        task_id = fields.get("task_id")
        if task_id is not None:
            registrations[task_id] = fields

    return registrations


class SoccerTrainingTaskRegistrationTest(unittest.TestCase):
    def test_keeps_existing_playable_tasks_registered(self):
        tasks = _registered_tasks()

        self.assertEqual(
            tasks["Unitree-G1-Shooter"]["env_cfg"],
            "unitree_g1_shooter_env_cfg",
        )
        self.assertEqual(
            tasks["Unitree-G1-Shooter"]["play_env_cfg"],
            "unitree_g1_shooter_env_cfg",
        )
        self.assertEqual(
            tasks["Unitree-G1-Shooter"]["rl_cfg"],
            "unitree_g1_soccer_ppo_runner_cfg",
        )
        self.assertEqual(tasks["Unitree-G1-Shooter"]["runner_cls"], "None")

        self.assertEqual(
            tasks["Unitree-G1-Goalkeeper"]["env_cfg"],
            "unitree_g1_goalkeeper_env_cfg",
        )
        self.assertEqual(
            tasks["Unitree-G1-Goalkeeper"]["play_env_cfg"],
            "unitree_g1_goalkeeper_env_cfg",
        )
        self.assertEqual(
            tasks["Unitree-G1-Goalkeeper"]["rl_cfg"],
            "unitree_g1_soccer_ppo_runner_cfg",
        )
        self.assertEqual(tasks["Unitree-G1-Goalkeeper"]["runner_cls"], "None")

    def test_registers_shooter_stage_training_tasks(self):
        tasks = _registered_tasks()

        self.assertEqual(
            tasks["Unitree-G1-Shooter-Stage1"]["env_cfg"],
            "unitree_g1_stage1_env_cfg",
        )
        self.assertEqual(
            tasks["Unitree-G1-Shooter-Stage1"]["play_env_cfg"],
            "unitree_g1_stage1_env_cfg",
        )
        self.assertEqual(
            tasks["Unitree-G1-Shooter-Stage1"]["rl_cfg"],
            "unitree_g1_soccer_ppo_runner_cfg",
        )
        self.assertEqual(
            tasks["Unitree-G1-Shooter-Stage1"]["runner_cls"],
            "None",
        )
        self.assertEqual(
            tasks["Unitree-G1-Shooter-Stage2"]["env_cfg"],
            "unitree_g1_stage2_env_cfg",
        )
        self.assertEqual(
            tasks["Unitree-G1-Shooter-Stage2"]["play_env_cfg"],
            "unitree_g1_stage2_env_cfg",
        )
        self.assertEqual(
            tasks["Unitree-G1-Shooter-Stage2"]["rl_cfg"],
            "unitree_g1_soccer_recurrent_runner_cfg",
        )
        self.assertEqual(
            tasks["Unitree-G1-Shooter-Stage2"]["runner_cls"],
            "SoccerRecurrentRunner",
        )

    def test_registers_goalkeeper_training_task_with_custom_runner(self):
        tasks = _registered_tasks()

        self.assertEqual(
            tasks["Unitree-G1-Goalkeeper-Train"]["env_cfg"],
            "unitree_g1_goalkeeper_training_env_cfg",
        )
        self.assertEqual(
            tasks["Unitree-G1-Goalkeeper-Train"]["play_env_cfg"],
            "unitree_g1_goalkeeper_training_env_cfg",
        )
        self.assertEqual(
            tasks["Unitree-G1-Goalkeeper-Train"]["rl_cfg"],
            "unitree_g1_goalkeeper_ppo_runner_cfg",
        )
        self.assertEqual(
            tasks["Unitree-G1-Goalkeeper-Train"]["runner_cls"],
            "GoalkeeperRunner",
        )

    def test_registers_goalkeeper_eval_task_with_goalkeeper_runner(self):
        tasks = _registered_eval_tasks()

        self.assertEqual(
            tasks["Eval-Goalkeeper"]["rl_cfg"],
            "unitree_g1_goalkeeper_ppo_runner_cfg",
        )
        self.assertEqual(
            tasks["Eval-Goalkeeper"]["runner_cls"],
            "GoalkeeperRunner",
        )

    def test_goalkeeper_runner_uses_low_exploration_noise(self):
        from src.tasks.soccer.config.g1.rl_cfg import (
            unitree_g1_goalkeeper_ppo_runner_cfg,
        )

        cfg = unitree_g1_goalkeeper_ppo_runner_cfg()

        self.assertLessEqual(cfg.actor.distribution_cfg["init_std"], 0.25)
        self.assertGreater(cfg.actor.distribution_cfg["min_std"], 0.0)
        self.assertLessEqual(
            cfg.actor.distribution_cfg["min_std"],
            cfg.actor.distribution_cfg["init_std"],
        )
        self.assertGreater(cfg.algorithm.entropy_coef, 0.0)
        self.assertLessEqual(cfg.algorithm.entropy_coef, 0.003)
        self.assertLessEqual(cfg.clip_actions, 1.2)
        self.assertGreaterEqual(cfg.algorithm.estimator_region_loss_coef, 2.0)
        self.assertEqual(
            cfg.algorithm.class_name,
            "src.tasks.soccer.modules.goalkeeper_ppo:GoalkeeperPPO",
        )

    def test_goalkeeper_runner_algorithm_kwargs_match_custom_ppo_signature(self):
        from src.tasks.soccer.config.g1.rl_cfg import (
            unitree_g1_goalkeeper_ppo_runner_cfg,
        )
        from src.tasks.soccer.modules.goalkeeper_ppo import GoalkeeperPPO

        algorithm_cfg = asdict(unitree_g1_goalkeeper_ppo_runner_cfg())["algorithm"]
        signature = inspect.signature(GoalkeeperPPO.__init__)

        self.assertIn("estimator_ball_loss_coef", algorithm_cfg)
        self.assertIn("estimator_region_loss_coef", algorithm_cfg)
        self.assertIn("amp_cfg", algorithm_cfg)
        self.assertIn("estimator_ball_loss_coef", signature.parameters)
        self.assertIn("estimator_region_loss_coef", signature.parameters)
        self.assertIn("amp_cfg", signature.parameters)
        self.assertEqual(algorithm_cfg["amp_cfg"]["reward_mode"], "mix")
        self.assertEqual(algorithm_cfg["amp_cfg"]["reward_dt"], 1.0)


class SoccerCliCompatibilityTest(unittest.TestCase):
    def test_mixed_dict_command_fields_are_hidden_from_tyro_cli(self):
        tree = ast.parse(SHOOTER_COMMANDS_FILE.read_text())
        annotations: dict[str, ast.AST] = {}

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name != "MultiMotionSoccerCommandCfg":
                continue
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    annotations[item.target.id] = item.annotation

        self.assertIn("curve_offset_range", annotations)
        self.assertEqual(
            ast.unparse(annotations["curve_offset_range"]),
            "tyro.conf.Suppress[dict | None]",
        )


if __name__ == "__main__":
    unittest.main()
