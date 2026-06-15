import unittest
from types import SimpleNamespace


try:
    import torch

    from src.tasks.soccer.mdp.goalkeeper_rewards import (
        goalkeeper_dynamic_target_pos,
        goalkeeper_ee_reach,
        goalkeeper_feet_slippage,
        goalkeeper_penalize_high_ball_foot_block,
        goalkeeper_region_motion_modulation,
        goalkeeper_selected_hand_target_distance,
        goalkeeper_stop_ball,
        goalkeeper_success,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - base env lacks mjlab.
    if exc.name not in {"mjlab", "torch"}:
        raise
    torch = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class _Scene(dict):
    def __init__(self, *args, sensor_map=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensors = sensor_map or {}


class _FakeRobot:
    def __init__(self, data, body_names):
        self.data = data
        self._body_names = body_names

    def find_bodies(self, body_names, preserve_order=False):
        del preserve_order
        return [[self._body_names[name] for name in body_names]]


def _env(
    robot_pos=None,
    ball_pos=None,
    ball_vel=None,
    root_lin_vel=None,
    projected_gravity=None,
    body_pos=None,
    body_vel=None,
    foot_contact_found=None,
):
    for candidate in (robot_pos, ball_pos, ball_vel, root_lin_vel, projected_gravity, body_pos, body_vel, foot_contact_found):
        if candidate is not None:
            num_envs = torch.as_tensor(candidate).shape[0]
            break
    else:
        num_envs = 1
    robot_pos_t = torch.as_tensor(robot_pos if robot_pos is not None else [[0.0, 0.0, 0.8]] * num_envs, dtype=torch.float32)
    body_names = {
        "left_wrist_yaw_link": 0,
        "right_wrist_yaw_link": 1,
        "left_ankle_roll_link": 2,
        "right_ankle_roll_link": 3,
    }
    robot_data = SimpleNamespace(
        root_link_pos_w=robot_pos_t,
        root_link_lin_vel_w=torch.as_tensor(
            root_lin_vel if root_lin_vel is not None else [[0.0, 0.0, 0.0]] * num_envs,
            dtype=torch.float32,
        ),
        projected_gravity_b=torch.as_tensor(
            projected_gravity if projected_gravity is not None else [[0.0, 0.0, -1.0]] * num_envs,
            dtype=torch.float32,
        ),
    )
    if body_pos is not None:
        robot_data.body_link_pos_w = torch.as_tensor(body_pos, dtype=torch.float32)
    if body_vel is not None:
        robot_data.body_link_lin_vel_w = torch.as_tensor(body_vel, dtype=torch.float32)
    ball_data = SimpleNamespace(
        root_link_pos_w=torch.as_tensor(
            ball_pos if ball_pos is not None else [[1.0, 0.0, 0.5]] * num_envs,
            dtype=torch.float32,
        ),
        root_link_lin_vel_w=torch.as_tensor(
            ball_vel if ball_vel is not None else [[-4.0, 0.0, 0.0]] * num_envs,
            dtype=torch.float32,
        ),
    )
    sensors = {}
    if foot_contact_found is not None:
        sensors["feet_ground_contact"] = SimpleNamespace(
            data=SimpleNamespace(found=torch.as_tensor(foot_contact_found, dtype=torch.float32))
        )
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=_Scene(
            robot=_FakeRobot(robot_data, body_names),
            ball=SimpleNamespace(data=ball_data),
            sensor_map=sensors,
        ),
    )


@unittest.skipIf(_IMPORT_ERROR is not None, f"missing runtime dependency: {_IMPORT_ERROR}")
class GoalkeeperOfficialRewardSemanticsTest(unittest.TestCase):
    def test_dynamic_target_switches_to_live_ball_only_in_official_close_window(self):
        env = _env(
            ball_pos=[[1.0, 0.2, 0.5], [0.3, -0.4, 0.8], [0.3, 0.5, 0.8]],
            ball_vel=[[-4.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        env._gk_ball_end_pos = torch.tensor(
            [[0.1, 1.0, 1.0], [0.1, 0.9, 1.1], [0.1, -0.8, 0.6]],
            dtype=torch.float32,
        )

        target = goalkeeper_dynamic_target_pos(env)

        self.assertTrue(torch.allclose(target[0], torch.tensor([0.1, 1.0, 1.0])))
        self.assertTrue(torch.allclose(target[1], torch.tensor([0.3, -0.4, 0.8])))
        self.assertTrue(torch.allclose(target[2], torch.tensor([0.1, -0.8, 0.6])))

    def test_region_distance_uses_hands_for_all_six_regions(self):
        env = _env(
            body_pos=[[
                [0.1, 1.0, 0.2],
                [0.1, -1.0, 0.2],
                [0.1, 1.0, 0.05],
                [0.1, -1.0, 0.05],
            ]],
        )
        env._gk_region = torch.tensor([4.0])
        env._gk_ball_end_pos = torch.tensor([[0.1, 1.0, 0.2]], dtype=torch.float32)

        distance = goalkeeper_selected_hand_target_distance(
            env,
            ee_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"),
        )

        self.assertTrue(torch.allclose(distance, torch.zeros(1, 1)))

    def test_official_motion_modulation_is_region_conditioned_multiplier(self):
        env = _env(
            root_lin_vel=[[0.0, 0.5, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.5]],
        )
        env._gk_region = torch.tensor([0.0, 1.0, 2.0])
        env._gk_ball_end_pos = torch.tensor(
            [[0.1, 1.0, 1.0], [0.1, -1.0, 1.0], [0.1, 0.5, 1.4]],
            dtype=torch.float32,
        )

        modulation = goalkeeper_region_motion_modulation(env)

        self.assertTrue(torch.allclose(modulation, torch.tensor([2.5, 2.5, 2.5])))

    def test_stop_ball_is_official_one_shot_speed_drop_in_front(self):
        env = _env(
            ball_pos=[[0.4, 0.0, 0.8], [-0.1, 0.0, 0.8], [0.4, 0.0, 0.8]],
            ball_vel=[[-4.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [-1.5, 0.0, 0.0]],
        )
        env._gk_initial_ball_vx = torch.tensor([-1.5, -1.5, -4.0])

        reward = goalkeeper_stop_ball(env, velocity_drop_threshold=2.0)

        self.assertTrue(torch.allclose(reward, torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(env._gk_success_flag, torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(env._gk_stop_flag, torch.tensor([1.0, 1.0, 0.0])))

        second = goalkeeper_stop_ball(env, velocity_drop_threshold=2.0)
        self.assertTrue(torch.allclose(second, torch.zeros(3)))

    def test_high_ball_stop_requires_selected_hand_proximity(self):
        env = _env(
            ball_pos=[[0.4, 0.8, 1.35], [0.4, -0.8, 1.35]],
            ball_vel=[[-4.0, 0.0, 0.0], [-4.0, 0.0, 0.0]],
            body_pos=[
                [
                    [0.4, 0.82, 1.34],
                    [0.4, -1.4, 0.6],
                    [0.4, 0.8, 1.30],
                    [0.4, -0.8, 1.30],
                ],
                [
                    [0.4, 1.4, 0.6],
                    [0.4, -1.4, 0.6],
                    [0.4, 0.8, 1.30],
                    [0.4, -0.8, 1.30],
                ],
            ],
        )
        env._gk_region = torch.tensor([2.0, 3.0])
        env._gk_initial_ball_vx = torch.tensor([-1.5, -1.5])

        reward = goalkeeper_stop_ball(
            env,
            velocity_drop_threshold=2.0,
            high_ball_hand_gate_radius=0.30,
        )

        self.assertTrue(torch.allclose(reward, torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.allclose(env._gk_success_flag, torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.allclose(env._gk_stop_flag, torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.allclose(env._gk_invalid_high_ball_block, torch.tensor([0.0, 1.0])))

    def test_high_ball_invalid_stop_penalty_is_one_shot(self):
        env = _env(
            ball_pos=[[0.4, 0.8, 1.35]],
            ball_vel=[[-4.0, 0.0, 0.0]],
            body_pos=[[
                [0.4, 1.4, 0.6],
                [0.4, -1.4, 0.6],
                [0.4, 0.8, 1.30],
                [0.4, -0.8, 1.30],
            ]],
        )
        env._gk_region = torch.tensor([2.0])
        env._gk_initial_ball_vx = torch.tensor([-1.5])

        goalkeeper_stop_ball(env, velocity_drop_threshold=2.0, high_ball_hand_gate_radius=0.30)

        first = goalkeeper_penalize_high_ball_foot_block(env)
        second = goalkeeper_penalize_high_ball_foot_block(env)
        self.assertTrue(torch.allclose(first, torch.ones(1)))
        self.assertTrue(torch.allclose(second, torch.zeros(1)))

    def test_success_reward_uses_success_flag_and_strict_hand_distance(self):
        env = _env(
            body_pos=[[
                [0.1, 0.0, 0.5],
                [0.1, -1.0, 0.5],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]],
        )
        env._gk_region = torch.tensor([0.0])
        env._gk_ball_end_pos = torch.tensor([[0.1, 0.0, 0.5]], dtype=torch.float32)
        env._gk_success_flag = torch.tensor([1.0])

        reward = goalkeeper_success(env, strict_distance=0.15)

        self.assertTrue(torch.allclose(reward, torch.tensor([2.0])))

    def test_ee_reach_uses_official_sigmoid_and_upright_gate(self):
        env = _env(
            body_pos=[[
                [0.1, 0.0, 0.5],
                [0.1, -1.0, 0.5],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]],
            ball_pos=[[0.4, 0.0, 0.5]],
        )
        env._gk_region = torch.tensor([0.0])
        env._gk_ball_end_pos = torch.tensor([[0.1, 0.0, 0.5]], dtype=torch.float32)
        env._gk_curriculumsigma = 5.0

        reward = goalkeeper_ee_reach(env, catch_distance=0.2)

        expected = 1.0 - 1.0 / (1.0 + torch.exp(torch.tensor([-5.0 * (0.3 - 0.2)])))
        self.assertTrue(torch.allclose(reward, expected))

    def test_feet_slippage_is_positive_official_exp_reward(self):
        env = _env(
            body_vel=[[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.3, 0.0, 0.0],
            ]],
            foot_contact_found=[[1.0, 0.0]],
        )

        reward = goalkeeper_feet_slippage(env)

        self.assertTrue(torch.allclose(reward, torch.exp(torch.tensor([-1.0]))))


if __name__ == "__main__":
    unittest.main()
