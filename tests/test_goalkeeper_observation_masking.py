import unittest
from types import SimpleNamespace


try:
    import torch

    from src.tasks.soccer.mdp.goalkeeper_obs import (
        gk_ball_pos_local,
        goalkeeper_ball_distance,
        goalkeeper_sidestep_command,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - base env lacks mjlab.
    if exc.name not in {"mjlab", "torch"}:
        raise
    torch = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class _Scene(dict):
    pass


class _FakeRobot:
    def __init__(self, data, body_names):
        self.data = data
        self._body_names = body_names

    def find_bodies(self, body_names, preserve_order=False):
        del preserve_order
        return [[self._body_names[name] for name in body_names]]


@unittest.skipIf(_IMPORT_ERROR is not None, f"missing runtime dependency: {_IMPORT_ERROR}")
class GoalkeeperObservationMaskingTest(unittest.TestCase):
    def _env(self, ball_pos, ball_vel, step_buf):
        robot_data = SimpleNamespace(
            root_link_pos_w=torch.zeros(len(ball_pos), 3),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * len(ball_pos)),
        )
        ball_data = SimpleNamespace(
            root_link_pos_w=torch.as_tensor(ball_pos, dtype=torch.float32),
            root_link_lin_vel_w=torch.as_tensor(ball_vel, dtype=torch.float32),
        )
        return SimpleNamespace(
            num_envs=len(ball_pos),
            device="cpu",
            step_dt=0.02,
            episode_length_buf=torch.as_tensor(step_buf, dtype=torch.long),
            scene=_Scene(
                robot=SimpleNamespace(data=robot_data),
                ball=SimpleNamespace(data=ball_data),
            ),
        )

    def test_ball_observation_is_zeroed_after_ball_stops_or_dropout_triggers(self):
        env = self._env(
            ball_pos=[
                [1.0, 0.1, 0.5],
                [1.0, 0.2, 0.6],
                [1.0, 0.3, 0.7],
            ],
            ball_vel=[
                [-3.0, 0.0, 0.0],  # visible
                [0.02, 0.0, 0.0],  # stopped
                [-3.0, 0.0, 0.0],  # dropout after 0.4s
            ],
            step_buf=[5, 5, 25],
        )

        obs = gk_ball_pos_local(
            env,
            dropout_after_s=0.4,
            dropout_prob=1.0,
            stop_speed_threshold=0.1,
        )

        self.assertTrue(torch.allclose(obs[0], torch.tensor([1.0, 0.1, 0.5])))
        self.assertTrue(torch.allclose(obs[1], torch.zeros(3)))
        self.assertTrue(torch.allclose(obs[2], torch.zeros(3)))

    def test_critic_ball_distance_is_selected_hand_to_dynamic_target_distance(self):
        robot_data = SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.8]], dtype=torch.float32),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            body_link_pos_w=torch.tensor(
                [
                    [
                        [-0.2, 0.7, 1.0],
                        [0.0, -1.0, 1.0],
                    ],
                ],
                dtype=torch.float32,
            ),
        )
        ball_data = SimpleNamespace(
            root_link_pos_w=torch.tensor([[1.0, 0.0, 0.8]], dtype=torch.float32),
            root_link_lin_vel_w=torch.tensor([[-3.0, 0.0, 0.0]], dtype=torch.float32),
        )
        env = SimpleNamespace(
            num_envs=1,
            device="cpu",
            scene=_Scene(
                robot=_FakeRobot(
                    robot_data,
                    {
                        "left_wrist_yaw_link": 0,
                        "right_wrist_yaw_link": 1,
                    },
                ),
                ball=SimpleNamespace(data=ball_data),
            ),
            _gk_region=torch.tensor([0.0]),
            _gk_ball_end_pos=torch.tensor([[-0.2, 1.0, 1.4]], dtype=torch.float32),
        )

        distance = goalkeeper_ball_distance(env)

        self.assertTrue(torch.allclose(distance, torch.tensor([[0.5]])))

    def test_sidestep_command_exposes_target_error_and_clipped_lateral_velocity(self):
        env = self._env(
            ball_pos=[
                [1.0, 0.0, 0.8],
                [1.0, 0.0, 0.8],
                [0.2, 0.2, 0.8],
            ],
            ball_vel=[
                [-3.0, 0.0, 0.0],
                [-3.0, 0.0, 0.0],
                [-3.0, 0.0, 0.0],
            ],
            step_buf=[0, 0, 0],
        )
        env.scene["robot"].data.root_link_pos_w = torch.tensor(
            [
                [0.0, 0.0, 0.8],
                [0.0, 0.5, 0.8],
                [0.0, 0.0, 0.8],
            ],
            dtype=torch.float32,
        )
        env._gk_ball_end_pos = torch.tensor(
            [
                [0.1, 1.0, 0.8],
                [0.1, -1.0, 0.8],
                [0.1, 0.8, 0.8],
            ],
            dtype=torch.float32,
        )

        command = goalkeeper_sidestep_command(
            env,
            position_gain=2.0,
            max_speed=1.2,
            deadzone=0.1,
            approach_distance_threshold=0.8,
        )

        expected = torch.tensor(
            [
                [1.0, 1.2],
                [-1.5, -1.2],
                [0.2, 0.4],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(command, expected))


if __name__ == "__main__":
    unittest.main()
