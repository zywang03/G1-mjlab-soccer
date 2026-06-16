"""Policy server for Phase 2 tournament — shooter teacher (160D LSTM).

Receives raw MuJoCo state from ``compete.py`` and computes the full 160D
training observation, filling motion-reference terms from real .npz data.

  POST /act    - receive raw state, return action
  POST /reset  - reset policy + motion + destination state

Usage:
  # Shooter server
  python scripts/api_server.py \\
      --checkpoint checkpoints/stage4/model_138985.pt \\
      --port 8000 --task shooter

  # With custom motion directory
  python scripts/api_server.py \\
      --checkpoint model.pt --port 8000 --task shooter \\
      --motion-dir thirdparty/G1-mjlab-soccer/src/assets/soccer/motions/shooter

Observation — 160D (9 terms, concatenated):
  command(ref_joint_pos+ref_joint_vel) 58 + projected_gravity 3 +
  motion_ref_ang_vel 3 + base_ang_vel 3 + joint_pos_rel 29 +
  joint_vel_rel 29 + last_action 29 + target_point_pos 3 +
  target_destination_pos 3                = 160
"""

from __future__ import annotations

import glob
import os
import random
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import tyro
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from src.tasks.soccer.mdp.shooter_commands import _IL_TO_MJCF_JOINT, _IL_TO_MJCF_BODY

# ---------------------------------------------------------------------------
# HOME_KEYFRAME default joint positions (29D, MJCF order)
# ---------------------------------------------------------------------------

_HOME_KEYFRAME_JOINT_POS = torch.tensor([
    -0.1,  0.0,  0.0,  0.3, -0.2,  0.0,   # left  leg (6)
    -0.1,  0.0,  0.0,  0.3, -0.2,  0.0,   # right leg (6)
     0.0,  0.0,  0.0,                       # waist    (3)
     0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0, # left  arm (7)
     0.35,-0.18, 0.0, 0.87, 0.0, 0.0, 0.0, # right arm (7)
])

_GK_DEFAULT_JOINT_POS = torch.tensor([
    0.0, 0.0, 0.0,
    -0.35, 0.7, -0.35, -0.25, 0.3, -0.1,
    -0.35, 0.7, -0.35, -0.25, 0.3, -0.1,
    0.0, 0.3, 0.0,
    0.8, 0.0, -1.6, 0.0, 0.5, 0.0, 0.0,
    0.8, 0.0, -1.6, 0.0, 0.5, 0.0, 0.0,
])

# ---------------------------------------------------------------------------
# Motion data
# ---------------------------------------------------------------------------

TORSO_MJCF_IDX = 15  # torso_link in MJCF body order


@dataclass
class MotionData:
    joint_pos: torch.Tensor       # (T, 29) MJCF order
    joint_vel: torch.Tensor       # (T, 29) MJCF order
    anchor_ang_vel: torch.Tensor  # (T, 3)   torso ang vel
    name: str


def _load_motions(motion_dir: str, device: str) -> list[MotionData]:
    """Load all soccer-standard-*.npz, apply IL→MJCF permute, return list."""
    pattern = os.path.join(motion_dir, "soccer-standard-*.npz")
    files = sorted(glob.glob(pattern))
    motions = []
    for f in files:
        data = np.load(f)
        jp = torch.tensor(data["joint_pos"][:, _IL_TO_MJCF_JOINT],
                          dtype=torch.float32, device=device)
        jv = torch.tensor(data["joint_vel"][:, _IL_TO_MJCF_JOINT],
                          dtype=torch.float32, device=device)
        bav = torch.tensor(data["body_ang_vel_w"][:, _IL_TO_MJCF_BODY, :],
                           dtype=torch.float32, device=device)
        anchor_av = bav[:, TORSO_MJCF_IDX, :]
        motions.append(MotionData(jp, jv, anchor_av, os.path.basename(f)))
    if not motions:
        raise FileNotFoundError(f"No soccer-standard-*.npz found under {motion_dir}")
    print(f"[INFO] Loaded {len(motions)} motion files from {motion_dir}")
    return motions


# ---------------------------------------------------------------------------
# Observation computation  (CUSTOMIZE: match your training observation space)
# ---------------------------------------------------------------------------

def compute_shooter_obs(
    raw_state: dict,
    motion: MotionData,
    time_step: int,
    destination_world: tuple[float, float, float],
) -> torch.Tensor:
    """Compute 160D shooter observation from raw state + motion data.

    Concatenation order matches ``stage1_env_cfg.py`` actor_terms (9 terms).
    """
    s = raw_state["shooter"]
    ball = raw_state["ball"]

    root_quat = torch.tensor(s["root_quat"])
    root_pos = torch.tensor(s["root_pos"])
    root_ang_vel = torch.tensor(s["root_ang_vel"])
    joint_pos = torch.tensor(s["joint_pos"])
    joint_vel = torch.tensor(s["joint_vel"])
    ball_pos = torch.tensor(ball["pos"])
    last_action = torch.tensor(s["last_action"])

    # Freeze at last frame when motion ends.
    T = motion.joint_pos.shape[0]
    t = min(time_step, T - 1)

    # 1. command — ref joint_pos + joint_vel (58D)
    command = torch.cat([motion.joint_pos[t], motion.joint_vel[t]])

    # 2. projected_gravity (3D)
    gravity_w = torch.tensor([0.0, 0.0, -1.0])
    projected_gravity = quat_apply(quat_inv(root_quat), gravity_w)

    # 3. motion_ref_ang_vel — reference anchor angular velocity (3D)
    motion_ref_ang_vel = motion.anchor_ang_vel[t]

    # 4. base_ang_vel — actual base angular velocity in body frame (3D)
    base_ang_vel = quat_apply(quat_inv(root_quat), root_ang_vel)

    # 5. joint_pos_rel — relative to HOME_KEYFRAME (29D)
    joint_pos_rel = joint_pos - _HOME_KEYFRAME_JOINT_POS.to(joint_pos.device)

    # 6. joint_vel_rel (29D, default_vel = 0)
    joint_vel_rel = joint_vel

    # 7. last_action (29D) — already have it

    # 8. target_point_pos — ball in robot pelvis frame (3D)
    ball_body = quat_apply(quat_inv(root_quat), ball_pos - root_pos)

    # 9. target_destination_pos — destination in robot pelvis frame (3D)
    dest_w = torch.tensor(destination_world)
    dest_body = quat_apply(quat_inv(root_quat), dest_w - root_pos)

    obs = torch.cat([
        command,             # 58
        projected_gravity,   #  3
        motion_ref_ang_vel,  #  3
        base_ang_vel,        #  3
        joint_pos_rel,       # 29
        joint_vel_rel,       # 29
        last_action,         # 29
        ball_body,           #  3
        dest_body,           #  3
    ])  # → 160

    return obs.unsqueeze(0)  # (1, 160)


def compute_goalkeeper_obs(raw_state: dict) -> torch.Tensor:
    """Compute goalkeeper observation tensor from raw state (single frame).

    Default: matches ``eval_goalkeeper_cfg`` per-frame terms (96-D).
    Replace with your own obs terms, scaling, and concatenation order.
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
    joint_pos_rel = (joint_pos - _GK_DEFAULT_JOINT_POS) * 1.0
    joint_vel_scaled = joint_vel * 0.05
    ball_pos_local = quat_apply(quat_inv(root_quat), ball_pos - root_pos)

    obs = torch.cat([
        ball_pos_local,         # 3
        base_ang_vel,           # 3
        projected_gravity,      # 3
        joint_pos_rel,          # 29
        joint_vel_scaled,       # 29
        last_action,            # 29
    ])
    return obs.unsqueeze(0)  # (1, 96)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ActResponse(BaseModel):
    action: list[list[float]]


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def _load_policy(checkpoint_path: str, task_id: str, device: str) -> Any:
    """Build env from task config, load checkpoint, return inference policy."""
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()

    env_cfg = load_env_cfg(task_id, play=False)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    actor_terms = list(env_cfg.observations["actor"].terms.keys())
    history_len = env_cfg.observations["actor"].history_length
    print(f"[INFO] Task: {task_id}")
    print(f"[INFO] Actor obs  ({len(actor_terms)} terms × {history_len} history): {actor_terms}")
    print(f"[INFO] Action dim: {env.action_manager.action_dim}")

    if task_id == "Eval-Goalkeeper":
        from src.tasks.soccer.config.g1.rl_cfg import (
            GoalkeeperRunner,
            unitree_g1_goalkeeper_ppo_runner_cfg,
        )
        loaded = torch.load(checkpoint_path, map_location=device)
        agent_cfg = unitree_g1_goalkeeper_ppo_runner_cfg()
        runner = GoalkeeperRunner(env, asdict(agent_cfg), device=device)

        if "model_state_dict" in loaded and hasattr(runner.alg.actor, "history_encoder"):
            print("[INFO] Detected HIMPPO ActorCritic checkpoint — loading directly.")
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
        runner = SoccerRecurrentRunner(
            env, asdict(agent_cfg), log_dir=None, device=device,
        )
        runner.load(checkpoint_path)

    policy = runner.get_inference_policy(device=device)
    print(f"[INFO] Policy loaded from: {checkpoint_path}")
    return policy, env


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    checkpoint_path: str,
    task_id: str,
    device: str,
    motion_dir: str,
) -> FastAPI:
    """Build the FastAPI app with a loaded policy and motion data."""

    policy, env = _load_policy(checkpoint_path, task_id, device)
    is_gk = task_id == "Eval-Goalkeeper"
    history_len = 10 if is_gk else 1

    # History buffer for observation stack.
    history: deque[torch.Tensor] = deque(maxlen=history_len)

    # Motion + destination state (re-sampled per episode).
    motions: list[MotionData] = []
    motion_ctx: dict[str, Any] = {
        "motion": None,
        "time_step": 0,
        "destination": (0.0, 0.0, 0.0),
    }

    if not is_gk:
        motions = _load_motions(motion_dir, device)
        motion_ctx["motion"] = motions[0]  # placeholder, overwritten on /reset

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"[INFO] Server ready — {task_id} policy on {device}")
        yield
        env.close()
        print("[INFO] Server shutting down.")

    app = FastAPI(title=f"CS2810 Phase 2 — {task_id}", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/act", response_model=ActResponse)
    async def act(req: dict):
        raw_state = req
        if is_gk:
            frame = compute_goalkeeper_obs(raw_state)
        else:
            m = motion_ctx["motion"]
            frame = compute_shooter_obs(raw_state, m, motion_ctx["time_step"], motion_ctx["destination"])
            motion_ctx["time_step"] += 1

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
        if is_gk:
            policy.reset()
        else:
            motion_ctx["motion"] = random.choice(motions)
            motion_ctx["time_step"] = 0
            dest_y = random.uniform(-1.1, 1.1)
            motion_ctx["destination"] = (-0.5, dest_y, 0.11)
            policy.reset()
        history.clear()
        return {"status": "ok"}

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    checkpoint: str
    """Path to the policy checkpoint (.pt)."""
    port: int = 8000
    """Port to listen on."""
    task: str = "shooter"
    """Task type: 'shooter' or 'goalkeeper'."""
    host: str = "0.0.0.0"
    """Host to bind to."""
    device: str | None = None
    """Torch device (auto-detected if omitted)."""
    motion_dir: str = "thirdparty/G1-mjlab-soccer/src/assets/soccer/motions/shooter"
    """Directory containing soccer-standard-*.npz motion files."""


def main():
    import src.tasks  # noqa: F401  — register eval tasks

    args = tyro.cli(ServerConfig, prog="api_server")

    task_id = "Eval-Shooter" if args.task == "shooter" else "Eval-Goalkeeper"
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    app = create_app(args.checkpoint, task_id, device, args.motion_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
