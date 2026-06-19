"""Reference policy server for CS2810 Phase 2.

Students run one server per policy.  The tournament runner sends raw MuJoCo
state to ``POST /act`` and expects a 29-D action in response.

Usage:
  python phase2/api_server.py --checkpoint shooter.pt --port 8000 --task shooter
  python phase2/api_server.py --checkpoint goalkeeper.pt --port 8001 --task goalkeeper
"""

from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
import tyro
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv


_SHOOTER_DEFAULT_JOINT_POS = torch.tensor([
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
])

_GK_DEFAULT_JOINT_POS = torch.tensor([
    0.0, 0.0, 0.0,
    -0.35, 0.7, -0.35, -0.25, 0.3, -0.1,
    -0.35, 0.7, -0.35, -0.25, 0.3, -0.1,
    0.0, 0.3, 0.0,
    0.8, 0.0, -1.6, 0.0, 0.5, 0.0, 0.0,
    0.8, 0.0, -1.6, 0.0, 0.5, 0.0, 0.0,
])


def compute_shooter_obs(raw_state: dict) -> torch.Tensor:
    """Compute a default shooter observation from raw state.

    Teams should customize this function to match their own training setup.
    """
    s = raw_state["shooter"]
    ball = raw_state["ball"]

    root_quat = torch.tensor(s["root_quat"])
    root_ang_vel = torch.tensor(s["root_ang_vel"])
    joint_pos = torch.tensor(s["joint_pos"])
    joint_vel = torch.tensor(s["joint_vel"])
    ball_pos = torch.tensor(ball["pos"])
    root_pos = torch.tensor(s["root_pos"])
    last_action = torch.tensor(s["last_action"])

    gravity_w = torch.tensor([0.0, 0.0, -1.0])
    projected_gravity = quat_apply(quat_inv(root_quat), gravity_w)
    base_ang_vel = quat_apply(quat_inv(root_quat), root_ang_vel)
    joint_pos_rel = joint_pos - _SHOOTER_DEFAULT_JOINT_POS
    ball_pos_local = quat_apply(quat_inv(root_quat), ball_pos - root_pos)

    obs = torch.cat([
        base_ang_vel,
        projected_gravity,
        joint_pos_rel,
        joint_vel,
        last_action,
        ball_pos_local,
    ])
    return obs.unsqueeze(0)


def compute_goalkeeper_obs(raw_state: dict) -> torch.Tensor:
    """Compute a default goalkeeper observation from raw state.

    Teams should customize this function to match their own training setup.
    """
    s = raw_state["goalkeeper"]
    ball = raw_state["ball"]

    root_quat = torch.tensor(s["root_quat"])
    root_ang_vel = torch.tensor(s["root_ang_vel"])
    joint_pos = torch.tensor(s["joint_pos"])
    joint_vel = torch.tensor(s["joint_vel"])
    ball_pos = torch.tensor(ball["pos"])
    root_pos = torch.tensor(s["root_pos"])
    last_action = torch.tensor(s["last_action"])

    gravity_w = torch.tensor([0.0, 0.0, -1.0])
    projected_gravity = quat_apply(quat_inv(root_quat), gravity_w)
    base_ang_vel = quat_apply(quat_inv(root_quat), root_ang_vel) * 0.25
    joint_pos_rel = joint_pos - _GK_DEFAULT_JOINT_POS
    joint_vel_scaled = joint_vel * 0.05
    ball_pos_local = quat_apply(quat_inv(root_quat), ball_pos - root_pos)

    obs = torch.cat([
        ball_pos_local,
        base_ang_vel,
        projected_gravity,
        joint_pos_rel,
        joint_vel_scaled,
        last_action,
    ])
    return obs.unsqueeze(0)


class ActResponse(BaseModel):
    action: list[list[float]]


def _load_policy(checkpoint_path: str, task_id: str, device: str) -> Any:
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    env_cfg = load_env_cfg(task_id, play=False)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    actor_terms = list(env_cfg.observations["actor"].terms.keys())
    history_len = env_cfg.observations["actor"].history_length
    print(f"[INFO] Task: {task_id}")
    print(f"[INFO] Actor obs ({len(actor_terms)} terms x {history_len} history): {actor_terms}")
    print(f"[INFO] Action dim: {env.num_actions}")

    if task_id == "Eval-Goalkeeper":
        from src.tasks.soccer.config.g1.rl_cfg import (
            GoalkeeperRunner,
            unitree_g1_goalkeeper_ppo_runner_cfg,
        )

        loaded = torch.load(checkpoint_path, map_location=device)
        agent_cfg = unitree_g1_goalkeeper_ppo_runner_cfg()
        runner = GoalkeeperRunner(env, asdict(agent_cfg), device=device)
        if "model_state_dict" in loaded and hasattr(runner.alg.actor, "history_encoder"):
            print("[INFO] Detected HIMPPO ActorCritic checkpoint.")
            actor_state = {
                k: v
                for k, v in loaded["model_state_dict"].items()
                if not k.startswith("critic.")
            }
            runner.alg.actor.load_state_dict(actor_state, strict=False)
        else:
            runner.load(checkpoint_path, load_cfg={"actor": True})
    else:
        from src.tasks.soccer.config.g1.rl_cfg import (
            SoccerRecurrentRunner,
            unitree_g1_soccer_recurrent_runner_cfg,
        )

        agent_cfg = unitree_g1_soccer_recurrent_runner_cfg()
        runner = SoccerRecurrentRunner(env, asdict(agent_cfg), log_dir=None, device=device)
        runner.load(checkpoint_path)

    policy = runner.get_inference_policy(device=device)
    print(f"[INFO] Policy loaded from: {checkpoint_path}")
    return policy, env


def create_app(checkpoint_path: str, task_id: str, device: str) -> FastAPI:
    policy, env = _load_policy(checkpoint_path, task_id, device)
    is_gk = task_id == "Eval-Goalkeeper"
    history_len = 10 if is_gk else 1
    history: deque[torch.Tensor] = deque(maxlen=history_len)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"[INFO] Server ready: {task_id} policy on {device}")
        yield
        env.close()
        print("[INFO] Server shutting down.")

    app = FastAPI(title=f"CS2810 Phase 2 - {task_id}", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/act", response_model=ActResponse)
    async def act(req: dict):
        frame = compute_goalkeeper_obs(req) if is_gk else compute_shooter_obs(req)
        if len(history) == 0:
            for _ in range(history_len):
                history.append(frame.clone())
        history.append(frame)
        stacked = torch.cat(list(history), dim=-1)

        with torch.inference_mode():
            action = policy({"actor": stacked})
        return ActResponse(action=action.cpu().tolist())

    @app.post("/reset")
    async def reset():
        policy.reset()
        history.clear()
        return {"status": "ok"}

    return app


@dataclass
class ServerConfig:
    checkpoint: str
    port: int = 8000
    task: str = "shooter"
    host: str = "0.0.0.0"
    device: str | None = None


def main() -> None:
    import src.tasks  # noqa: F401

    args = tyro.cli(ServerConfig, prog="phase2-api-server")
    if args.task not in {"shooter", "goalkeeper"}:
        raise ValueError("--task must be either 'shooter' or 'goalkeeper'")
    task_id = "Eval-Shooter" if args.task == "shooter" else "Eval-Goalkeeper"
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    app = create_app(args.checkpoint, task_id, device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
