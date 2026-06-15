import unittest
from dataclasses import asdict
from types import SimpleNamespace


try:
    import torch

    from src.tasks.soccer.config.g1.rl_cfg import unitree_g1_goalkeeper_ppo_runner_cfg
    from src.tasks.soccer.config.g1.training_env_cfgs import (
        unitree_g1_goalkeeper_training_env_cfg,
    )
    from src.tasks.soccer.mdp.goalkeeper_ball_reset import RegionBallVelCfg
    from src.tasks.soccer.mdp.goalkeeper_curriculum import goalkeeper_curriculum
except ModuleNotFoundError as exc:  # pragma: no cover - base env lacks mjlab.
    if exc.name not in {"mjlab", "torch"}:
        raise
    torch = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class _FakeManager:
    def __init__(self, terms):
        self._terms = terms

    def get_term_cfg(self, term_name):
        if term_name not in self._terms:
            raise ValueError(term_name)
        return self._terms[term_name]


@unittest.skipIf(_IMPORT_ERROR is not None, f"missing runtime dependency: {_IMPORT_ERROR}")
class GoalkeeperCurriculumTest(unittest.TestCase):
    def test_training_config_uses_official_observation_reward_event_set(self):
        cfg = unitree_g1_goalkeeper_training_env_cfg()

        self.assertEqual(
            list(cfg.observations["actor"].terms),
            ["ball_pos_local", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"],
        )
        self.assertEqual(
            list(cfg.observations["critic"].terms),
            [
                "ball_pos_local",
                "base_ang_vel",
                "projected_gravity",
                "joint_pos",
                "joint_vel",
                "actions",
                "base_lin_vel",
                "end_region",
                "end_target_pos",
                "ball_vel_local",
                "ee_positions",
                "ball_distance",
            ],
        )
        self.assertEqual(
            set(cfg.rewards),
            {
                "ee_reach",
                "success",
                "stop_ball",
                "high_ball_foot_block",
                "stay_on_line",
                "no_retreat",
                "success_land",
                "feet_orientation",
                "penalize_sharp_contact",
                "penalize_knee_height",
                "feet_slippage",
                "post_orientation",
                "post_ang_vel",
                "post_lin_vel",
                "post_upper_dof_pos",
                "post_waist_dof_pos",
                "ang_vel_xy",
                "dof_acc",
                "action_smoothness",
                "torques",
                "dof_vel",
                "joint_limit",
                "dof_vel_limits",
                "torque_limits",
                "deviation_waist_pitch_joint",
            },
        )
        self.assertEqual(
            list(cfg.events),
            [
                "decrement_catchstep",
                "reset_ball",
                "push_robot",
                "perturb_ball_vel",
                "reset_robot_base",
                "reset_robot_joints",
                "reset_gk_state",
            ],
        )
        self.assertTrue(cfg.scale_rewards_by_dt)

    def test_curriculum_starts_with_all_six_regions_and_official_weights(self):
        cfg = unitree_g1_goalkeeper_training_env_cfg()
        stages = cfg.curriculum["goalkeeper_difficulty"].params["stages"]
        stage0 = stages[0]

        self.assertEqual([region["motion_id"] for region in stage0["regions"]], [0, 1, 2, 3, 4, 5])
        self.assertEqual(stage0["t_flight_range"], (0.4, 1.0))
        self.assertEqual(stage0["push_interval_range_s"], (15.0, 15.0))
        self.assertEqual(stage0["perturb_interval_range_s"], (0.5, 0.5))
        self.assertEqual(stage0["reward_weights"]["ee_reach"], 10.0)
        self.assertEqual(stage0["reward_weights"]["success"], 5.0)
        self.assertEqual(stage0["reward_weights"]["stop_ball"], 60.0)
        self.assertEqual(stage0["reward_weights"]["high_ball_foot_block"], -80.0)
        self.assertEqual(stage0["reward_weights"]["feet_slippage"], 3.0)
        self.assertLess(stage0["reward_weights"]["action_smoothness"], 0.0)

    def test_curriculum_mutates_ball_ranges_rewards_and_internal_sigma(self):
        vel_cfg = RegionBallVelCfg(
            ball_start_x_range=(3.0, 5.0),
            ball_end_x_range=(0.1, 0.6),
            t_flight_range=(0.4, 1.0),
            regions=[{"height": (0.4, 1.2), "width": (0.2, 1.2), "motion_id": 0}],
        )
        event_terms = {
            "reset_ball": SimpleNamespace(params={"vel_cfg": vel_cfg}),
            "push_robot": SimpleNamespace(params={}, interval_range_s=(15.0, 15.0)),
            "perturb_ball_vel": SimpleNamespace(params={}, interval_range_s=(0.5, 0.5)),
        }
        reward_terms = {
            name: SimpleNamespace(weight=0.0)
            for name in ("ee_reach", "success", "stop_ball", "high_ball_foot_block", "joint_limit", "torque_limits")
        }
        env = SimpleNamespace(
            common_step_counter=60_000,
            device="cpu",
            event_manager=_FakeManager(event_terms),
            reward_manager=_FakeManager(reward_terms),
        )

        result = goalkeeper_curriculum(
            env,
            env_ids=torch.tensor([0]),
            stages=[
                {
                    "step": 0,
                    "name": "initial",
                    "regions": [{"height": (0.4, 1.2), "width": (0.2, 1.2), "motion_id": 0}],
                    "reward_weights": {"ee_reach": 10.0, "success": 5.0, "stop_ball": 60.0},
                },
                {
                    "step": 50_000,
                    "name": "expanded",
                    "regions": [{"height": (0.1, 0.3), "width": (-1.2, -0.2), "motion_id": 5}],
                    "t_flight_range": (0.4, 1.0),
                    "push_vel_xy_range": (-0.5, 0.5),
                    "push_vel_z_range": (0.0, 0.0),
                    "push_ang_vel_range": (0.0, 0.0),
                    "perturb_vel_range": (-0.25, 0.25),
                    "curriculum_update": 1,
                    "curriculumsigma": 5.0,
                    "reward_weights": {
                        "ee_reach": 15.0,
                        "success": 7.5,
                        "stop_ball": 100.0,
                        "high_ball_foot_block": -80.0,
                        "joint_limit": -6.0,
                        "torque_limits": -9.0,
                    },
                },
            ],
        )

        active = env.event_manager.get_term_cfg("reset_ball").params["vel_cfg"]
        self.assertEqual([region["motion_id"] for region in active.regions], [5])
        self.assertEqual(env.event_manager.get_term_cfg("push_robot").params["vel_xy_range"], (-0.5, 0.5))
        self.assertEqual(env.event_manager.get_term_cfg("perturb_ball_vel").params["vel_range"], (-0.25, 0.25))
        self.assertEqual(env.reward_manager.get_term_cfg("stop_ball").weight, 100.0)
        self.assertEqual(env.reward_manager.get_term_cfg("high_ball_foot_block").weight, -80.0)
        self.assertEqual(getattr(env, "_gk_curriculum_update"), 1)
        self.assertEqual(getattr(env, "_gk_curriculumsigma"), 5.0)
        self.assertEqual(int(result["stage_index"].item()), 1)

    def test_goalkeeper_runner_cfg_matches_official_hyperparameters_and_amp(self):
        cfg = asdict(unitree_g1_goalkeeper_ppo_runner_cfg())
        amp_cfg = cfg["algorithm"]["amp_cfg"]

        self.assertEqual(cfg["num_steps_per_env"], 100)
        self.assertEqual(cfg["max_iterations"], 100000)
        self.assertEqual(cfg["algorithm"]["entropy_coef"], 0.01)
        self.assertEqual(cfg["actor"]["distribution_cfg"]["init_std"], 1.0)
        self.assertTrue(amp_cfg["enabled"])
        self.assertEqual(amp_cfg["reward_coef"], 0.4)
        self.assertEqual(amp_cfg["reward_scale"], 0.5)
        self.assertEqual(amp_cfg["grad_penalty_coef"], 0.1)
        self.assertEqual(amp_cfg["num_reward_samples"], 4)


if __name__ == "__main__":
    unittest.main()
