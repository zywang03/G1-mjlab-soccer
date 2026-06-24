"""Utilities for frozen-opponent adversarial training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper


OpponentRole = Literal["shooter", "goalkeeper"]
_DEFAULT_TASK_BY_ROLE: dict[OpponentRole, str] = {
  "shooter": "Eval-Shooter-Stage3",
  "goalkeeper": "Eval-Goalkeeper",
}

GoalkeeperCheckpointKind = Literal[
  "moe6",
  "reference_actor_critic",
  "adversarial_actor_critic",
  "actor_critic",
  "mlp",
  "unknown",
]


@dataclass(frozen=True)
class FrozenOpponentSpec:
  role: OpponentRole
  checkpoint: str
  task_id: str


class ZeroPolicy:
  def __init__(self, num_envs: int, action_dim: int, device: torch.device):
    self._action = torch.zeros(num_envs, action_dim, device=device)

  def __call__(self, *_args, **_kwargs) -> torch.Tensor:
    return self._action

  def reset(self) -> None:
    self._action.zero_()


class FrozenOpponentVecEnvWrapper:
  """Expose only active actions while injecting a frozen opponent action."""

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    opponent_policy: Any,
    opponent_role: OpponentRole,
    active_action_dim: int = 29,
    opponent_action_dim: int = 29,
  ):
    self.env = env
    self.opponent_policy = opponent_policy
    self.opponent_role = opponent_role
    self.active_action_dim = active_action_dim
    self.opponent_action_dim = opponent_action_dim
    self.num_envs = env.num_envs
    self.device = env.device
    self.max_episode_length = env.max_episode_length
    self.num_actions = active_action_dim
    self._prev_active_action = torch.zeros(self.num_envs, active_action_dim, device=self.device)
    self._prev_opponent_action = torch.zeros(self.num_envs, opponent_action_dim, device=self.device)

  @property
  def cfg(self):
    return self.env.cfg

  @property
  def render_mode(self):
    return self.env.render_mode

  @property
  def observation_space(self):
    return self.env.observation_space

  @property
  def action_space(self):
    return self.env.action_space

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.env.unwrapped

  @property
  def episode_length_buf(self):
    return self.env.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value):
    self.env.episode_length_buf = value

  def seed(self, seed: int = -1) -> int:
    return self.env.seed(seed)

  def get_observations(self) -> TensorDict:
    return self.env.get_observations()

  def reset(self):
    obs, extras = self.env.reset()
    self._prev_active_action.zero_()
    self._prev_opponent_action.zero_()
    _reset(self.opponent_policy)
    return obs, extras

  def step(self, actions: torch.Tensor):
    opponent_action = self._compute_opponent_action()
    full_action = torch.cat((actions, opponent_action), dim=-1)
    obs, rew, dones, extras = self.env.step(full_action)
    self._prev_active_action = actions.detach().clone()
    self._prev_opponent_action = opponent_action.detach().clone()
    return obs, rew, dones, extras

  def close(self) -> None:
    try:
      self.env.close()
    finally:
      close_fn = getattr(self.opponent_policy, "close", None)
      if close_fn is not None:
        close_fn()

  def _compute_opponent_action(self) -> torch.Tensor:
    with torch.inference_mode():
      try:
        action = self.opponent_policy(self.unwrapped, self._prev_opponent_action)
      except TypeError:
        raw = build_raw_state_from_dual_env(
          self.unwrapped,
          self._prev_active_action,
          self._prev_opponent_action,
          self.opponent_role,
        )
        action = self.opponent_policy(raw)
    return action.to(device=self.device, dtype=torch.float32).view(self.num_envs, self.opponent_action_dim)


class SingleRobotActionEnvView:
  """Proxy a dual-action eval env as a single-robot policy env."""

  def __init__(self, env: RslRlVecEnvWrapper, action_dim: int = 29):
    self.env = env
    self.num_actions = action_dim

  def __getattr__(self, name: str):
    return getattr(self.env, name)

  @property
  def unwrapped(self):
    return self.env.unwrapped


def _reset(module: Any) -> None:
  reset_fn = getattr(module, "reset", None)
  if reset_fn is not None:
    reset_fn()


def _robot_root_state(robot, device: torch.device) -> torch.Tensor:
  return torch.cat((
    robot.data.root_link_pos_w,
    robot.data.root_link_quat_w,
    robot.data.root_link_lin_vel_w,
    robot.data.root_link_ang_vel_w,
  ), dim=-1).to(device)


def _ball_root_state(ball, device: torch.device) -> torch.Tensor:
  quat = torch.zeros(ball.data.root_link_pos_w.shape[0], 4, device=ball.data.root_link_pos_w.device)
  quat[:, 0] = 1.0
  return torch.cat((
    ball.data.root_link_pos_w,
    quat,
    ball.data.root_link_vel_w[:, :3],
    ball.data.root_link_vel_w[:, 3:],
  ), dim=-1).to(device)


def _make_eval_env(task_id: str, num_envs: int, device: str):
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg

  env_cfg = load_env_cfg(task_id, play=False)
  env_cfg.scene.num_envs = num_envs
  return RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=100.0)


def goalkeeper_checkpoint_kind(loaded: dict[str, Any]) -> GoalkeeperCheckpointKind:
  if "sr" in loaded:
    return "moe6"
  if "model_state_dict" in loaded:
    return "reference_actor_critic"
  actor_state = loaded.get("actor_state_dict")
  if not isinstance(actor_state, dict):
    return "unknown"
  keys = set(actor_state)
  if any(key.startswith("actor_residual.") or key.startswith("critic_residual.") for key in keys):
    return "adversarial_actor_critic"
  if any(key.startswith("history_encoder.") or key.startswith("ball_estimator.") for key in keys):
    return "actor_critic"
  if any(key.startswith("mlp.") for key in keys):
    return "mlp"
  return "unknown"


def moe_bundle_has_adversarial_idle(loaded: dict[str, Any]) -> bool:
  idle = next((loaded[k] for k in ("idle", "prepare", "idle_expert") if k in loaded), None)
  if not isinstance(idle, dict):
    return False
  actor_state = idle.get("actor_state_dict")
  if not isinstance(actor_state, dict):
    return False
  return any(key.startswith(("actor_residual.", "critic_residual.")) for key in actor_state)


def resolve_goalkeeper_eval_task_id(kind: GoalkeeperCheckpointKind, requested_task_id: str) -> str:
  if kind == "adversarial_actor_critic":
    return "Unitree-G1-Goalkeeper-Adversarial"
  return "Eval-Goalkeeper"


def set_eval_action_term(action_manager: Any, term_name: str, action: torch.Tensor) -> None:
  offset = 0
  for name, term in action_manager._terms.items():
    next_offset = offset + term.action_dim
    if name == term_name:
      action_manager._action[:, offset:next_offset] = action.to(action_manager._action.device)
      if hasattr(term, "_raw_actions"):
        term._raw_actions = action
      return
    offset = next_offset
  raise KeyError(f"Action term not found: {term_name}")


def scene_has_entity(scene: Any, name: str) -> bool:
  if hasattr(scene, "entities"):
    return name in scene.entities
  try:
    scene[name]
  except KeyError:
    return False
  return True


def _sync_eval_env(
  eval_env: RslRlVecEnvWrapper,
  action_term_name: str,
  dual_env: ManagerBasedRlEnv,
  last_action: torch.Tensor,
  device: torch.device,
) -> None:
  src_robot = dual_env.scene["opponent"]
  src_opponent = dual_env.scene["robot"]
  src_ball = dual_env.scene["ball"]
  eval_base = eval_env.unwrapped
  dst_robot = eval_base.scene["robot"]
  dst_ball = eval_base.scene["ball"]

  dst_robot.write_root_state_to_sim(_robot_root_state(src_robot, device))
  dst_robot.write_joint_state_to_sim(src_robot.data.joint_pos.to(device), src_robot.data.joint_vel.to(device))
  dst_robot.clear_state()
  if scene_has_entity(eval_base.scene, "opponent"):
    dst_opponent = eval_base.scene["opponent"]
    dst_opponent.write_root_state_to_sim(_robot_root_state(src_opponent, device))
    dst_opponent.write_joint_state_to_sim(src_opponent.data.joint_pos.to(device), src_opponent.data.joint_vel.to(device))
    dst_opponent.clear_state()
  dst_ball.write_root_state_to_sim(_ball_root_state(src_ball, device))
  dst_ball.clear_state()

  eval_base.scene.write_data_to_sim()
  eval_base.sim.forward()
  eval_base.sim.sense()
  set_eval_action_term(eval_base.action_manager, action_term_name, last_action.to(device))


def build_raw_state_from_dual_env(
  env: ManagerBasedRlEnv,
  prev_active_action: torch.Tensor,
  prev_opponent_action: torch.Tensor,
  opponent_role: OpponentRole,
) -> dict[str, Any]:
  """Read a compete-style raw state dict from a dual-robot adversarial env."""
  scene = env.scene
  active = scene["robot"]
  opponent = scene["opponent"]
  ball = scene["ball"]
  active_role = "goalkeeper" if opponent_role == "shooter" else "shooter"

  def _robot_state(robot, last_action: torch.Tensor) -> dict[str, Any]:
    return {
      "root_pos": robot.data.root_link_pos_w.detach().cpu().tolist(),
      "root_quat": robot.data.root_link_quat_w.detach().cpu().tolist(),
      "root_lin_vel": robot.data.root_link_lin_vel_w.detach().cpu().tolist(),
      "root_ang_vel": robot.data.root_link_ang_vel_w.detach().cpu().tolist(),
      "joint_pos": robot.data.joint_pos.detach().cpu().tolist(),
      "joint_vel": robot.data.joint_vel.detach().cpu().tolist(),
      "last_action": last_action.detach().cpu().tolist(),
    }

  raw = {
    active_role: _robot_state(active, prev_active_action),
    opponent_role: _robot_state(opponent, prev_opponent_action),
    "ball": {
      "pos": ball.data.root_link_pos_w.detach().cpu().tolist(),
      "vel": ball.data.root_link_vel_w[:, :3].detach().cpu().tolist(),
    },
  }
  return _squeeze_raw_state(raw) if env.num_envs == 1 else raw


def _squeeze_raw_state(raw: dict[str, Any]) -> dict[str, Any]:
  squeezed: dict[str, Any] = {"ball": {}}
  for role in ("shooter", "goalkeeper"):
    squeezed[role] = {
      key: value[0] if isinstance(value, list) and value and isinstance(value[0], list) else value
      for key, value in raw[role].items()
    }
  squeezed["ball"] = {
    key: value[0] if isinstance(value, list) and value and isinstance(value[0], list) else value
    for key, value in raw["ball"].items()
  }
  return squeezed


def build_frozen_opponent_policy(
  spec: FrozenOpponentSpec,
  device: str,
  num_envs: int = 1,
):
  """Load a frozen opponent policy using the same adapters as competition API."""
  if not spec.checkpoint:
    return ZeroPolicy(num_envs, 29, torch.device(device))
  path = Path(spec.checkpoint).expanduser()
  if not path.exists():
    raise FileNotFoundError(f"Frozen opponent checkpoint not found: {path}")

  task_id = spec.task_id or _DEFAULT_TASK_BY_ROLE[spec.role]
  if spec.role == "shooter":
    return FrozenShooterPolicy(str(path), device=device, num_envs=num_envs, task_id=task_id)
  return FrozenGoalkeeperPolicy(str(path), device=device, num_envs=num_envs, task_id=task_id)


class FrozenShooterPolicy:
  """Frozen shooter policy driven by the current dual-robot env state."""

  def __init__(self, checkpoint: str, device: str, num_envs: int, task_id: str = "Eval-Shooter-Stage3"):
    from dataclasses import asdict
    from src.tasks.soccer.config.g1.rl_cfg import (
      AdversarialSoccerRecurrentRunner,
      SoccerRecurrentRunner,
      unitree_g1_soccer_adversarial_recurrent_runner_cfg,
      unitree_g1_soccer_recurrent_runner_cfg,
    )

    self.device = torch.device(device)
    self.env = _make_eval_env(task_id, num_envs, device)
    runner_env = SingleRobotActionEnvView(self.env)
    is_adversarial = "Adversarial" in task_id
    runner_cls = AdversarialSoccerRecurrentRunner if is_adversarial else SoccerRecurrentRunner
    runner_cfg = (
      unitree_g1_soccer_adversarial_recurrent_runner_cfg()
      if is_adversarial
      else unitree_g1_soccer_recurrent_runner_cfg()
    )
    runner = runner_cls(
      runner_env,
      asdict(runner_cfg),
      log_dir=None,
      device=device,
    )
    runner.load(
      checkpoint,
      load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
      map_location=device,
    )
    self.policy = runner.get_inference_policy(device=device)
    self.action_term_name = "joint_pos"

  def reset(self) -> None:
    self.env.reset()
    _reset(self.policy)

  def close(self) -> None:
    self.env.close()

  def __call__(self, dual_env: ManagerBasedRlEnv, last_action: torch.Tensor) -> torch.Tensor:
    _sync_eval_env(self.env, self.action_term_name, dual_env, last_action, self.device)
    return self.policy(self.env.get_observations())


class FrozenGoalkeeperPolicy:
  """Frozen goalkeeper policy driven by the current dual-robot env state."""

  def __init__(self, checkpoint: str, device: str, num_envs: int, task_id: str = "Eval-Goalkeeper"):
    from dataclasses import asdict
    from mjlab.rl import MjlabOnPolicyRunner
    from src.tasks.soccer.config.g1.gk_train_cfg import goalkeeper_train_runner_cfg
    from src.tasks.soccer.config.g1.rl_cfg import (
      AdversarialGoalkeeperRunner,
      GoalkeeperRunner,
      unitree_g1_goalkeeper_adversarial_ppo_runner_cfg,
      unitree_g1_goalkeeper_ppo_runner_cfg,
    )
    from src.tasks.soccer.modules.moe6_goalkeeper_policy import MoE6GoalkeeperPolicy

    self.device = torch.device(device)
    loaded = torch.load(checkpoint, map_location=device, weights_only=False)
    kind = goalkeeper_checkpoint_kind(loaded) if isinstance(loaded, dict) else "unknown"
    task_id = resolve_goalkeeper_eval_task_id(kind, task_id)
    self.env = _make_eval_env(task_id, num_envs, device)
    self.action_term_name = "joint_pos"

    if kind == "moe6":
      idle_env = None
      if moe_bundle_has_adversarial_idle(loaded):
        idle_env = SingleRobotActionEnvView(
          _make_eval_env("Unitree-G1-Goalkeeper-Adversarial", num_envs, device)
        )
      self.policy = MoE6GoalkeeperPolicy(loaded, self.env, device, idle_env=idle_env)
    elif kind == "reference_actor_critic":
      runner = GoalkeeperRunner(
        SingleRobotActionEnvView(self.env),
        asdict(unitree_g1_goalkeeper_ppo_runner_cfg()),
        device=device,
      )
      runner.alg.actor.load_state_dict({
        key: value for key, value in loaded["model_state_dict"].items()
        if not key.startswith("critic.")
      }, strict=False)
      self.policy = runner.get_inference_policy(device=device)
    elif kind == "adversarial_actor_critic":
      runner = AdversarialGoalkeeperRunner(
        SingleRobotActionEnvView(self.env),
        asdict(unitree_g1_goalkeeper_adversarial_ppo_runner_cfg()),
        device=device,
      )
      runner.load(
        checkpoint,
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
        map_location=device,
      )
      self.policy = runner.get_inference_policy(device=device)
    elif kind == "actor_critic":
      runner = GoalkeeperRunner(
        SingleRobotActionEnvView(self.env),
        asdict(unitree_g1_goalkeeper_ppo_runner_cfg()),
        device=device,
      )
      runner.load(
        checkpoint,
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
        map_location=device,
      )
      self.policy = runner.get_inference_policy(device=device)
    else:
      runner = MjlabOnPolicyRunner(SingleRobotActionEnvView(self.env), asdict(goalkeeper_train_runner_cfg()), device=device)
      runner.load(checkpoint, load_cfg={"actor": True}, map_location=device)
      self.policy = runner.get_inference_policy(device=device)

  def reset(self) -> None:
    self.env.reset()
    _reset(self.policy)

  def close(self) -> None:
    try:
      close_fn = getattr(self.policy, "close", None)
      if close_fn is not None:
        close_fn()
    finally:
      self.env.close()

  def __call__(self, dual_env: ManagerBasedRlEnv, last_action: torch.Tensor) -> torch.Tensor:
    _sync_eval_env(self.env, self.action_term_name, dual_env, last_action, self.device)
    idle_env = getattr(self.policy, "idle_env", None)
    if idle_env is not None:
      _sync_eval_env(idle_env, self.action_term_name, dual_env, last_action, self.device)
    obs = self.env.get_observations()
    return self.policy({"actor": obs["actor"]})
