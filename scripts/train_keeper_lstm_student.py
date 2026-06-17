"""Distill a teacher keeper into a single recurrent student policy."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from src.tasks.soccer.config.g1.gk_train_cfg import (
  goalkeeper_ballistic_residual_runner_cfg,
  goalkeeper_lstm_student_runner_cfg,
  goalkeeper_train_runner_cfg,
)


@dataclass
class Cfg:
  teacher: str = "src/assets/soccer/weight/goalkeeper_moe6_lyk.pt"
  out: str = "logs/lyk/goalkeeper_lstm_student.pt"
  num_envs: int = 1024
  collect_steps: int = 149
  dagger_iters: int = 12
  bc_epochs: int = 4
  batch_envs: int = 128
  lr: float = 3.0e-4
  lr_final: float = 5.0e-5
  beta_decay: float = 0.6
  rnn_type: str = "lstm"
  rnn_hidden_dim: int = 256
  rnn_num_layers: int = 1
  seed: int = 2810
  device: str = "cuda:0"


def _load_teacher(env, checkpoint: str, device: str):
  loaded = torch.load(checkpoint, map_location=device, weights_only=False)
  if isinstance(loaded, dict) and loaded.get("moe6"):
    from src.tasks.soccer.modules.gk_moe6 import GoalkeeperMoE6Policy

    return GoalkeeperMoE6Policy(loaded, env, device)
  if loaded.get("ballistic_residual"):
    import src.tasks.soccer.modules.gk_ballistic_residual as gkbr

    meta = loaded["ballistic_residual"]
    gkbr.BASE_CKPT = meta.get("base")
    gkbr.BASE_HIDDEN = tuple(meta.get("base_hidden", (1024, 512, 256)))
    gkbr.RESIDUAL_SCALE = float(meta.get("residual_scale", 0.25))
    agent_cfg = goalkeeper_ballistic_residual_runner_cfg()
  else:
    agent_cfg = goalkeeper_train_runner_cfg()
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
  runner.load(checkpoint, load_cfg={"actor": True})
  return runner.get_inference_policy(device=device)


def _student_runner_dict(cfg: Cfg) -> dict:
  agent_cfg = asdict(goalkeeper_lstm_student_runner_cfg())
  agent_cfg["actor"]["rnn_type"] = cfg.rnn_type
  agent_cfg["actor"]["rnn_hidden_dim"] = cfg.rnn_hidden_dim
  agent_cfg["actor"]["rnn_num_layers"] = cfg.rnn_num_layers
  return agent_cfg


def main(cfg: Cfg) -> None:
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  device = cfg.device

  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  if "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device),
    clip_actions=100.0,
  )

  teacher = _load_teacher(env, cfg.teacher, device)
  runner = MjlabOnPolicyRunner(env, _student_runner_dict(cfg), device=device)
  student = runner.alg.actor
  opt = torch.optim.Adam(student.parameters(), lr=cfg.lr)

  for it in range(cfg.dagger_iters):
    obs = env.reset()
    if isinstance(obs, tuple):
      obs = obs[0]
    reset = getattr(teacher, "reset", None)
    if reset is not None:
      reset()
    student.reset()
    xs, ys = [], []
    beta = cfg.beta_decay ** it
    frac = it / max(1, cfg.dagger_iters - 1)
    lr = cfg.lr + (cfg.lr_final - cfg.lr) * frac
    for group in opt.param_groups:
      group["lr"] = lr

    for _ in range(cfg.collect_steps):
      actor_obs = obs["actor"].detach()
      with torch.inference_mode():
        teacher_action = teacher(obs)
        if torch.rand(()) < beta:
          rollout_action = teacher_action
        else:
          rollout_action = student({"actor": actor_obs})
      xs.append(actor_obs.cpu())
      ys.append(teacher_action.detach().cpu())
      obs = env.step(rollout_action)[0]

    x = torch.stack(xs)  # (T, B, O)
    y = torch.stack(ys)  # (T, B, A)
    n_env = x.shape[1]
    losses = []
    for _ in range(cfg.bc_epochs):
      perm = torch.randperm(n_env)
      for start in range(0, n_env, cfg.batch_envs):
        idx = perm[start : start + cfg.batch_envs]
        xb = x[:, idx].to(device, non_blocking=True)
        yb = y[:, idx].to(device, non_blocking=True)
        pred = student.sequence_mean(xb)
        loss = torch.nn.functional.smooth_l1_loss(pred, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    print(
      f"[BC] iter {it + 1}/{cfg.dagger_iters} beta={beta:.3f} "
      f"lr={lr:.1e} loss={sum(losses) / max(1, len(losses)):.5f}",
      flush=True,
    )

  os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
  saved = runner.alg.save()
  saved["iter"] = 0
  saved["infos"] = {"env_state": {"common_step_counter": 0}}
  saved["goalkeeper_lstm_student"] = {
    "teacher": cfg.teacher,
    "rnn_type": cfg.rnn_type,
    "rnn_hidden_dim": cfg.rnn_hidden_dim,
    "rnn_num_layers": cfg.rnn_num_layers,
  }
  torch.save(saved, cfg.out)
  print(f"[INFO] saved recurrent keeper student to {cfg.out}", flush=True)
  env.close()


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  import src.tasks.soccer.config.eval  # noqa: F401

  main(tyro.cli(Cfg, prog="train_keeper_lstm_student"))
