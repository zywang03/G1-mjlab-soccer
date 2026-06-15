import unittest
from types import SimpleNamespace


try:
    import torch

    from src.tasks.soccer.mdp.goalkeeper_ball_reset import (
        RegionBallVelCfg,
        reset_ball_with_parabolic_trajectory,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - base env lacks mjlab.
    if exc.name not in {"mjlab", "torch"}:
        raise
    torch = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class _FakeBall:
    def __init__(self, num_envs):
        self.data = SimpleNamespace(default_root_state=torch.zeros(num_envs, 13))
        self.data.default_root_state[:, 3] = 1.0
        self.written_pose = None
        self.written_velocity = None

    def write_root_link_pose_to_sim(self, pose, env_ids=None):
        del env_ids
        self.written_pose = pose.clone()

    def write_root_link_velocity_to_sim(self, velocity, env_ids=None):
        del env_ids
        self.written_velocity = velocity.clone()


class _Scene(dict):
    def __init__(self, *args, env_origins, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_origins = env_origins


@unittest.skipIf(_IMPORT_ERROR is not None, f"missing runtime dependency: {_IMPORT_ERROR}")
class GoalkeeperBallResetTest(unittest.TestCase):
    def test_parabolic_reset_stores_front_interception_target_not_behind_goal_endpoint(self):
        num_envs = 8
        ball = _FakeBall(num_envs)
        env = SimpleNamespace(
            num_envs=num_envs,
            device="cpu",
            scene=_Scene(
                {"ball": ball},
                env_origins=torch.zeros(num_envs, 3),
            ),
        )
        vel_cfg = RegionBallVelCfg(
            ball_start_x_range=(3.0, 3.0),
            ball_end_x_range=(0.5, 0.5),
            t_flight_range=(1.0, 1.0),
            regions=[{"height": (0.5, 0.5), "width": (1.0, 1.0), "motion_id": 0}],
            ball_start_y_range=(0.0, 0.0),
            ball_start_z_range=(0.5, 0.5),
            interception_x=0.1,
        )

        reset_ball_with_parabolic_trajectory(env, None, vel_cfg)

        self.assertTrue(torch.allclose(env._gk_ball_end_pos[:, 0], torch.full((num_envs,), 0.1)))
        self.assertTrue(torch.all(env._gk_ball_end_pos[:, 0] > 0.0))
        self.assertTrue(torch.allclose(env._gk_region, torch.zeros(num_envs)))
        self.assertIsNotNone(ball.written_pose)
        self.assertIsNotNone(ball.written_velocity)


if __name__ == "__main__":
    unittest.main()
