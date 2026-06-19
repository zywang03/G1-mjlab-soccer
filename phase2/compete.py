"""Run one CS2810 Phase 2 match with Viser visualization and JSON results."""

from __future__ import annotations

import json
import math
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import torch
import tyro
import yaml

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer, ViewerConfig

from src.assets.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from src.assets.robots.unitree_g1.g1_constants import FULL_COLLISION, HOME_KEYFRAME
from src.tasks.soccer import mdp
from src.tasks.soccer.ball import get_ball_cfg
from src.tasks.soccer.goal import get_goal_cfg
from src.tasks.soccer.ground import get_ground_cfg
from src.tasks.soccer.mdp.goalkeeper_obs import _GK_DEFAULT_JOINT_POS, get_gk_robot_cfg
from src.tasks.soccer.soccer_env_cfg import _add_soccer_scene_postproc


PHASE2_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PHASE2_DIR / "phase2_config.yaml"
DEFAULT_RESULTS_DIR = PHASE2_DIR / "results"

_SHOOTER_CFG = SceneEntityCfg("shooter")
_GK_CFG = SceneEntityCfg("goalkeeper")
_BALL_CFG = SceneEntityCfg("ball")


def _load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "team"


def _utc_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = yaw / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _vec3(values: list[float] | tuple[float, float, float]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _build_raw_state(
    env_base: ManagerBasedRlEnv,
    prev_action_shooter: torch.Tensor,
    prev_action_gk: torch.Tensor,
) -> dict[str, Any]:
    scene = env_base.scene
    shooter = scene["shooter"]
    goalkeeper = scene["goalkeeper"]
    ball = scene["ball"]

    def _robot_state(robot) -> dict[str, Any]:
        return {
            "root_pos": robot.data.root_link_pos_w[0].cpu().tolist(),
            "root_quat": robot.data.root_link_quat_w[0].cpu().tolist(),
            "root_lin_vel": robot.data.root_link_lin_vel_w[0].cpu().tolist(),
            "root_ang_vel": robot.data.root_link_ang_vel_w[0].cpu().tolist(),
            "joint_pos": robot.data.joint_pos[0].cpu().tolist(),
            "joint_vel": robot.data.joint_vel[0].cpu().tolist(),
        }

    return {
        "shooter": {
            **_robot_state(shooter),
            "last_action": prev_action_shooter[0].cpu().tolist(),
        },
        "goalkeeper": {
            **_robot_state(goalkeeper),
            "last_action": prev_action_gk[0].cpu().tolist(),
        },
        "ball": {
            "pos": ball.data.root_link_pos_w[0].cpu().tolist(),
            "vel": ball.data.root_link_vel_w[0, :3].cpu().tolist(),
        },
    }


def _make_shooter_robot(config: dict[str, Any]) -> Any:
    cfg = get_g1_robot_cfg()
    scene = config["scene"]
    cfg.init_state = replace(
        HOME_KEYFRAME,
        pos=_vec3(scene["shooter_pos"]),
        rot=_yaw_to_quat(float(scene["shooter_yaw"])),
    )
    cfg.collisions = (FULL_COLLISION,)
    return cfg


def _make_goalkeeper_robot(config: dict[str, Any]) -> Any:
    cfg = get_gk_robot_cfg()
    scene = config["scene"]
    cfg.init_state = replace(
        cfg.init_state,
        pos=_vec3(scene["goalkeeper_pos"]),
        rot=_yaw_to_quat(float(scene["goalkeeper_yaw"])),
        joint_pos=_GK_DEFAULT_JOINT_POS,
    )
    cfg.collisions = (FULL_COLLISION,)
    return cfg


def make_compete_env_cfg(config: dict[str, Any]) -> ManagerBasedRlEnvCfg:
    scene_cfg = config["scene"]
    sim_cfg = config["sim"]
    entities: dict[str, Any] = {
        "ground": get_ground_cfg(),
        "ball": get_ball_cfg(pos=_vec3(scene_cfg["ball_pos"])),
        "goal": get_goal_cfg(pos=_vec3(config["goal"]["pos"])),
        "shooter": _make_shooter_robot(config),
        "goalkeeper": _make_goalkeeper_robot(config),
    }

    actions: dict[str, ActionTermCfg] = {
        "shooter_joint_pos": JointPositionActionCfg(
            entity_name="shooter",
            actuator_names=(".*",),
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
        ),
        "goalkeeper_joint_pos": JointPositionActionCfg(
            entity_name="goalkeeper",
            actuator_names=(".*",),
            scale=float(config["actions"]["goalkeeper_scale"]),
            use_default_offset=True,
        ),
    }

    events: dict[str, EventTermCfg] = {
        "reset_shooter_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}, "asset_cfg": _SHOOTER_CFG},
        ),
        "reset_shooter_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.0, 0.0),
                "velocity_range": (-0.0, 0.0),
                "asset_cfg": SceneEntityCfg("shooter", joint_names=(".*",)),
            },
        ),
        "reset_goalkeeper_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}, "asset_cfg": _GK_CFG},
        ),
        "reset_goalkeeper_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.0, 0.0),
                "velocity_range": (-0.0, 0.0),
                "asset_cfg": SceneEntityCfg("goalkeeper", joint_names=(".*",)),
            },
        ),
        "reset_ball": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}, "asset_cfg": _BALL_CFG},
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities=entities,
            num_envs=1,
            spec_fn=_add_soccer_scene_postproc,
        ),
        observations={
            "shooter_actor": ObservationGroupCfg(
                terms={
                    "dummy": ObservationTermCfg(
                        func=mdp.builtin_sensor,
                        params={"sensor_name": "shooter/imu_ang_vel"},
                    ),
                },
                concatenate_terms=True,
                enable_corruption=False,
                history_length=1,
            ),
            "goalkeeper_actor": ObservationGroupCfg(
                terms={
                    "dummy": ObservationTermCfg(
                        func=mdp.builtin_sensor,
                        params={"sensor_name": "goalkeeper/imu_ang_vel"},
                    ),
                },
                concatenate_terms=True,
                enable_corruption=False,
                history_length=1,
            ),
        },
        actions=actions,
        commands={},
        events=events,
        rewards={
            "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
        },
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        },
        viewer=ViewerConfig(
            lookat=(2.0, 0.0, 1.0),
            distance=6.0,
            elevation=-15.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=int(sim_cfg["nconmax"]),
            njmax=int(sim_cfg["njmax"]),
            contact_sensor_maxmatch=int(sim_cfg["contact_sensor_maxmatch"]),
            mujoco=MujocoCfg(
                timestep=float(sim_cfg["timestep"]),
                iterations=int(sim_cfg["iterations"]),
                ls_iterations=int(sim_cfg["ls_iterations"]),
                ccd_iterations=int(sim_cfg["ccd_iterations"]),
            ),
        ),
        decimation=int(sim_cfg["decimation"]),
        episode_length_s=float(config["episode_length_s"]),
    )


class ZeroPolicy:
    def __init__(self, action_dim: int, device: str):
        self._zero = torch.zeros(1, action_dim, device=device)

    def __call__(self, _input: Any) -> torch.Tensor:
        return self._zero

    def reset(self) -> None:
        pass


class ApiPolicy:
    def __init__(self, url: str, action_dim: int, device: str, timeout: float = 2.0):
        self._url = url.rstrip("/")
        self._action_dim = action_dim
        self._device = device
        self._timeout = timeout
        resp = requests.post(f"{self._url}/reset", json={}, timeout=self._timeout)
        resp.raise_for_status()
        print(f"[INFO] API connected: {self._url} (act_dim={action_dim})", flush=True)

    def __call__(self, raw_state: dict[str, Any]) -> torch.Tensor:
        resp = requests.post(f"{self._url}/act", json=raw_state, timeout=self._timeout)
        resp.raise_for_status()
        payload = resp.json()
        action = torch.tensor(payload["action"], device=self._device, dtype=torch.float32)
        if action.shape != (1, self._action_dim):
            raise RuntimeError(
                f"{self._url}/act returned shape {tuple(action.shape)}, "
                f"expected (1, {self._action_dim})"
            )
        return action

    def reset(self) -> None:
        try:
            requests.post(f"{self._url}/reset", json={}, timeout=self._timeout)
        except requests.RequestException:
            pass


class CombinedPolicy:
    def __init__(
        self,
        shooter_policy: Any,
        goalkeeper_policy: Any,
        env_base: ManagerBasedRlEnv,
        device: str,
    ):
        self._shooter = shooter_policy
        self._goalkeeper = goalkeeper_policy
        self._env_base = env_base
        self._prev_action_s = torch.zeros(1, 29, device=device)
        self._prev_action_g = torch.zeros(1, 29, device=device)

    def __call__(self, _obs: dict[str, Any]) -> torch.Tensor:
        raw = _build_raw_state(self._env_base, self._prev_action_s, self._prev_action_g)
        s_act = self._shooter(raw)
        g_act = self._goalkeeper(raw)
        self._prev_action_s = s_act.detach().clone()
        self._prev_action_g = g_act.detach().clone()
        return torch.cat([s_act, g_act], dim=-1)

    def reset(self) -> None:
        self._shooter.reset()
        self._goalkeeper.reset()
        self._prev_action_s.zero_()
        self._prev_action_g.zero_()


class PassiveViserViewer(ViserPlayViewer):
    """Viser viewer that renders the environment without stepping physics."""

    def __init__(
        self,
        *args: Any,
        scoreboard: "ScoreboardState | None" = None,
        start_event: threading.Event | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._scoreboard = scoreboard
        self._start_event = start_event
        self._scoreboard_html = None
        self._start_button = None

    def setup(self) -> None:
        super().setup()
        if self._scoreboard is not None:
            import viser

            with self._server.gui.add_folder("Match Scoreboard"):
                if self._start_event is not None:
                    self._start_button = self._server.gui.add_button(
                        "Start Trials",
                        icon=viser.Icon.PLAYER_PLAY,
                        color="green",
                    )

                    @self._start_button.on_click
                    def _(_) -> None:
                        if self._start_event is None or self._start_event.is_set():
                            return
                        self._start_event.set()
                        self._scoreboard.set_phase("starting")
                        self._start_button.label = "Trials Started"
                        self._start_button.disabled = True

                self._scoreboard_html = self._server.gui.add_html("")
            self._update_scoreboard_display()

    def _step_physics(self, dt: float) -> None:
        del dt
        return

    def reset_environment(self) -> None:
        return

    def _update_scoreboard_display(self) -> None:
        if self._scoreboard_html is None or self._scoreboard is None:
            return
        self._scoreboard_html.content = self._scoreboard.to_html()

    def tick(self) -> bool:
        rendered = super().tick()
        self._update_scoreboard_display()
        return rendered


@dataclass
class ScoreboardState:
    shooter_team: str
    goalkeeper_team: str
    total_trials: int
    current_trial: int = 0
    phase: str = "initializing"
    results: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set_phase(self, phase: str, current_trial: int | None = None) -> None:
        with self.lock:
            self.phase = phase
            if current_trial is not None:
                self.current_trial = current_trial

    def record(self, winner: str) -> None:
        with self.lock:
            self.results.append(winner)

    def snapshot(self) -> tuple[int, str, list[str]]:
        with self.lock:
            return self.current_trial, self.phase, list(self.results)

    def to_html(self) -> str:
        current_trial, phase, results = self.snapshot()
        shooter_wins = sum(1 for r in results if r == "shooter")
        gk_wins = sum(1 for r in results if r == "goalkeeper")
        symbols = []
        for result in results:
            if result == "shooter":
                color, label = "#16a34a", "S"
            elif result == "goalkeeper":
                color, label = "#2563eb", "G"
            else:
                color, label = "#dc2626", "!"
            symbols.append(
                "<span style='display:inline-flex;align-items:center;justify-content:center;"
                "width:22px;height:22px;border-radius:50%;margin:2px;color:white;"
                f"font-weight:700;background:{color};'>{label}</span>"
            )
        for _ in range(max(0, self.total_trials - len(results))):
            symbols.append(
                "<span style='display:inline-flex;align-items:center;justify-content:center;"
                "width:22px;height:22px;border-radius:50%;margin:2px;color:#6b7280;"
                "font-weight:700;background:#e5e7eb;'>-</span>"
            )
        current = current_trial if current_trial > 0 else "-"
        return f"""
        <div style="font-size:0.9em;line-height:1.35;padding:0 1em 0.6em 1em;">
          <strong>{self.shooter_team}</strong> vs <strong>{self.goalkeeper_team}</strong><br/>
          <strong>Current trial:</strong> {current}/{self.total_trials}<br/>
          <strong>Phase:</strong> {phase}<br/>
          <strong>Score:</strong> Shooter {shooter_wins} - {gk_wins} Goalkeeper<br/>
          <div style="margin-top:6px;">{''.join(symbols)}</div>
          <div style="color:#6b7280;margin-top:4px;">S = shooter goal, G = goalkeeper save</div>
        </div>
        """


def _ball_entered_goal(ball_pos: torch.Tensor, config: dict[str, Any]) -> bool:
    goal = config["goal"]
    x, y, z = ball_pos[0].item(), ball_pos[1].item(), ball_pos[2].item()
    return (
        x <= float(goal["plane_x"])
        and abs(y) <= float(goal["half_width"])
        and z <= float(goal["height"])
    )


def _minimal_config_audit(config: dict[str, Any], max_steps: int) -> dict[str, Any]:
    return {
        "episode_length_s": float(config["episode_length_s"]),
        "max_steps": max_steps,
        "robot": {
            "joint_order": config["robot"]["joint_order"],
            "shooter_initial_joints": config["robot"]["shooter_initial_joints"],
            "goalkeeper_initial_joints": config["robot"]["goalkeeper_initial_joints"],
            "shooter_yaw": float(config["scene"]["shooter_yaw"]),
            "goalkeeper_yaw": float(config["scene"]["goalkeeper_yaw"]),
        },
        "ball": {
            "initial_pos": config["scene"]["ball_pos"],
        },
        "sim": {
            "timestep": float(config["sim"]["timestep"]),
            "decimation": int(config["sim"]["decimation"]),
            "step_dt": float(config["sim"]["timestep"]) * int(config["sim"]["decimation"]),
        },
        "ground_contact": config["ground_contact"],
    }


def run_trial(
    trial_index: int,
    env: RslRlVecEnvWrapper,
    env_base: ManagerBasedRlEnv,
    shooter_policy: Any,
    goalkeeper_policy: Any,
    config: dict[str, Any],
    max_steps: int,
    step_dt: float,
    realtime: bool,
    scoreboard: ScoreboardState | None = None,
) -> dict[str, Any]:
    if scoreboard is not None:
        scoreboard.set_phase("running", trial_index)
    env.reset()
    shooter_policy.reset()
    goalkeeper_policy.reset()

    device = env.unwrapped.device
    prev_action_s = torch.zeros(1, 29, device=device)
    prev_action_g = torch.zeros(1, 29, device=device)
    ball = env.unwrapped.scene["ball"]
    goal_scored = False
    error: str | None = None
    steps = 0
    start_time = time.perf_counter()

    for _ in range(max_steps):
        try:
            with torch.inference_mode():
                raw = _build_raw_state(env_base, prev_action_s, prev_action_g)
                s_act = shooter_policy(raw)
                g_act = goalkeeper_policy(raw)
        except Exception as exc:
            error = str(exc)
            break

        result = env.step(torch.cat([s_act, g_act], dim=-1))
        steps += 1
        prev_action_s = s_act.detach().clone()
        prev_action_g = g_act.detach().clone()

        ball_pos = ball.data.root_link_pos_w[0].cpu()
        if _ball_entered_goal(ball_pos, config):
            goal_scored = True
            break

        terminated = result[2]
        terminated = bool(terminated.item()) if hasattr(terminated, "item") else bool(terminated)
        if terminated:
            break

        if realtime:
            target_time = start_time + steps * step_dt
            sleep_time = target_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    winner = "shooter" if goal_scored else "goalkeeper"
    if error is not None:
        winner = "error"
    if scoreboard is not None:
        scoreboard.record(winner)
    ball_pos = ball.data.root_link_pos_w[0].cpu()
    return {
        "trial": trial_index,
        "winner": winner,
        "goal_scored": goal_scored,
        "steps": steps,
        "ball_final_pos": ball_pos.tolist(),
        "error": error,
    }


def run_match(
    cfg: "CompeteConfig",
    config: dict[str, Any],
    env: RslRlVecEnvWrapper,
    env_base: ManagerBasedRlEnv,
    shooter_policy: Any,
    goalkeeper_policy: Any,
    max_steps: int,
    step_dt: float,
    scoreboard: ScoreboardState | None = None,
) -> dict[str, Any]:
    print(f"[INFO] Running {cfg.num_trials} trials, max_steps={max_steps}", flush=True)
    trials: list[dict[str, Any]] = []
    goals = 0
    errors = 0
    for index in range(1, cfg.num_trials + 1):
        stats = run_trial(
            index,
            env,
            env_base,
            shooter_policy,
            goalkeeper_policy,
            config,
            max_steps,
            step_dt,
            cfg.realtime,
            scoreboard,
        )
        trials.append(stats)
        if stats["goal_scored"]:
            goals += 1
        if stats["error"]:
            errors += 1
        print(
            f"[TRIAL {index}/{cfg.num_trials}] winner={stats['winner']} "
            f"goal={stats['goal_scored']} steps={stats['steps']}",
            flush=True,
        )
        if stats["error"]:
            print(f"[ERROR] {stats['error']}", flush=True)

    goalkeeper_wins = cfg.num_trials - goals - errors
    summary = {
        "num_trials": cfg.num_trials,
        "goals": goals,
        "goalkeeper_wins": goalkeeper_wins,
        "errors": errors,
        "winner_decision": "shooter" if goals > cfg.num_trials / 2 else "goalkeeper",
    }
    print(f"[SUMMARY] {summary}", flush=True)
    if scoreboard is not None:
        scoreboard.set_phase("finished")
    return {"summary": summary, "trials": trials}


def wait_for_start(
    start_event: threading.Event,
    scoreboard: ScoreboardState | None = None,
    auto_start: bool = False,
) -> None:
    if auto_start:
        start_event.set()
    if scoreboard is not None:
        scoreboard.set_phase("waiting for Start Trials", 0)
    print("[INFO] Ready. Waiting for Start Trials.", flush=True)
    start_event.wait()
    if scoreboard is not None:
        scoreboard.set_phase("starting", 0)
    print("[INFO] Start Trials pressed; beginning official trials.", flush=True)


def _default_results_path(cfg: "CompeteConfig", timestamp: str) -> Path:
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    match_name = cfg.match_id or (
        f"{_sanitize(cfg.shooter_team)}_shooter_vs_"
        f"{_sanitize(cfg.goalkeeper_team)}_goalkeeper"
    )
    return DEFAULT_RESULTS_DIR / f"{timestamp}_{_sanitize(match_name)}.json"


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[INFO] Result JSON: {path}", flush=True)


@dataclass
class CompeteConfig:
    shooter_api: str | None = None
    goalkeeper_api: str | None = None
    shooter_team: str = "ShooterTeam"
    goalkeeper_team: str = "GoalkeeperTeam"
    match_id: str | None = None
    num_trials: int = 10
    config_path: str = str(DEFAULT_CONFIG_PATH)
    results_json: str | None = None
    viser_host: str = "0.0.0.0"
    viser_port: int = 7000
    no_viewer: bool = False
    request_timeout: float = 2.0
    realtime: bool = True
    device: str | None = None
    seed: int = 2810


def run_compete(cfg: CompeteConfig) -> dict[str, Any]:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    configure_torch_backends()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    config = _load_config(cfg.config_path)
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    step_dt = float(config["sim"]["timestep"]) * int(config["sim"]["decimation"])
    max_steps = int(round(float(config["episode_length_s"]) / step_dt))
    timestamp = _utc_timestamp()
    result_path = Path(cfg.results_json) if cfg.results_json else _default_results_path(cfg, timestamp)

    print(f"[INFO] Match id: {cfg.match_id or result_path.stem}", flush=True)
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Viser: http://{cfg.viser_host}:{cfg.viser_port}", flush=True)

    env_cfg = make_compete_env_cfg(config)
    env_base = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    act_dim_shooter = env_base.action_manager.get_term("shooter_joint_pos").action_dim
    act_dim_goalkeeper = env_base.action_manager.get_term("goalkeeper_joint_pos").action_dim

    shooter_policy: Any
    goalkeeper_policy: Any
    try:
        shooter_policy = (
            ApiPolicy(cfg.shooter_api, act_dim_shooter, device, cfg.request_timeout)
            if cfg.shooter_api
            else ZeroPolicy(act_dim_shooter, device)
        )
        goalkeeper_policy = (
            ApiPolicy(cfg.goalkeeper_api, act_dim_goalkeeper, device, cfg.request_timeout)
            if cfg.goalkeeper_api
            else ZeroPolicy(act_dim_goalkeeper, device)
        )
        env = RslRlVecEnvWrapper(env_base, clip_actions=100.0)
        result_holder: dict[str, Any] = {}
        done_event = threading.Event()
        start_event = threading.Event()
        scoreboard = ScoreboardState(
            shooter_team=cfg.shooter_team,
            goalkeeper_team=cfg.goalkeeper_team,
            total_trials=cfg.num_trials,
        )

        def _worker() -> None:
            try:
                wait_for_start(
                    start_event,
                    scoreboard,
                    auto_start=cfg.no_viewer,
                )
                result_holder.update(
                    run_match(
                        cfg,
                        config,
                        env,
                        env_base,
                        shooter_policy,
                        goalkeeper_policy,
                        max_steps,
                        step_dt,
                        scoreboard,
                    )
                )
            except Exception as exc:
                result_holder["summary"] = {
                    "num_trials": cfg.num_trials,
                    "goals": 0,
                    "goalkeeper_wins": 0,
                    "errors": cfg.num_trials,
                    "winner_decision": "error",
                }
                result_holder["trials"] = []
                result_holder["fatal_error"] = str(exc)
                print(f"[FATAL] {exc}", flush=True)
            finally:
                if "summary" not in result_holder:
                    scoreboard.set_phase("failed")
                done_event.set()

        worker = threading.Thread(target=_worker, name="phase2-match", daemon=True)
        worker.start()

        if not cfg.no_viewer:
            try:
                import viser

                server = viser.ViserServer(host=cfg.viser_host, port=cfg.viser_port, label="phase2")
                combined = CombinedPolicy(shooter_policy, goalkeeper_policy, env_base, device)
                viewer = PassiveViserViewer(
                    env,
                    combined,
                    viser_server=server,
                    scoreboard=scoreboard,
                    start_event=start_event,
                )
                viewer.setup()
                try:
                    while viewer.is_running() and not done_event.is_set():
                        if not viewer.tick():
                            time.sleep(0.001)
                        viewer._update_stats()
                finally:
                    viewer.close()
            except TypeError:
                print("[WARN] ViserServer host/port signature mismatch; running without viewer.", flush=True)
                start_event.set()
                done_event.wait()
        else:
            done_event.wait()

        worker.join(timeout=5.0)
        payload = {
            "timestamp": timestamp,
            "match_id": cfg.match_id or result_path.stem,
            "teams": {
                "shooter": cfg.shooter_team,
                "goalkeeper": cfg.goalkeeper_team,
            },
            "apis": {
                "shooter": cfg.shooter_api,
                "goalkeeper": cfg.goalkeeper_api,
            },
            "minimal_config_audit": _minimal_config_audit(config, max_steps),
            **result_holder,
        }
        _write_result(result_path, payload)
        return payload
    except Exception as exc:
        payload = {
            "timestamp": timestamp,
            "match_id": cfg.match_id or result_path.stem,
            "teams": {"shooter": cfg.shooter_team, "goalkeeper": cfg.goalkeeper_team},
            "apis": {"shooter": cfg.shooter_api, "goalkeeper": cfg.goalkeeper_api},
            "minimal_config_audit": _minimal_config_audit(config, max_steps),
            "summary": {
                "num_trials": cfg.num_trials,
                "goals": 0,
                "goalkeeper_wins": 0,
                "errors": cfg.num_trials,
                "winner_decision": "error",
            },
            "trials": [],
            "fatal_error": str(exc),
        }
        _write_result(result_path, payload)
        raise
    finally:
        try:
            env_base.close()
        except Exception:
            pass


def main() -> None:
    args = tyro.cli(CompeteConfig, prog="phase2-compete")
    run_compete(args)


if __name__ == "__main__":
    main()
