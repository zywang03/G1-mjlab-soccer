"""Runner / network config for the distilled goalkeeper policy.

Our Phase-1 goalkeeper is a native rsl_rl MLP policy distilled (DAgger) from the
reference Humanoid-Goalkeeper checkpoint. This module defines the actor-critic
network used both to build the student during distillation
(``scripts/distill_goalkeeper.py``) and to load it at evaluation
(``scripts/eval_naive_goalkeeper.py`` / ``scripts/render_cases.py``).

obs_normalization is OFF: the goalkeeper observations are already manually scaled
(ang_vel*0.25, dof_vel*0.05, ...) to match the reference's input distribution, so
an extra running normalizer would double-normalize.
"""

from __future__ import annotations

from mjlab.rl import (
  MjlabOnPolicyRunner,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

# MLP hidden dims of the distilled goalkeeper actor/critic. Must match the saved
# checkpoint (src/assets/soccer/weight/goalkeeper_distilled_v3.pt).
_GK_MLP_HIDDEN = (1024, 512, 256)


def goalkeeper_train_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """rsl_rl runner config for the native MLP goalkeeper policy."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=_GK_MLP_HIDDEN,
      activation="elu",
      obs_normalization=False,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(
      hidden_dims=_GK_MLP_HIDDEN,
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
      entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
      learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
      desired_kl=0.01, max_grad_norm=1.0,
    ),
    experiment_name="g1_goalkeeper",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def goalkeeper_ballistic_residual_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Runner config for the frozen-base ballistic residual goalkeeper.

  The actor keeps the same 960D observation contract as the distilled MLP, but
  uses ``GoalkeeperBallisticResidual`` internally: old checkpoint behavior is the
  frozen base, and PPO only updates a small residual head.
  """
  cfg = goalkeeper_train_runner_cfg()
  cfg.actor.class_name = "src.tasks.soccer.modules.gk_ballistic_residual.GoalkeeperBallisticResidual"
  cfg.actor.hidden_dims = (512, 256, 128)
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.08,
    "std_type": "scalar",
  }
  cfg.algorithm.entropy_coef = 0.0
  cfg.algorithm.clip_param = 0.1
  cfg.algorithm.learning_rate = 1.0e-4
  cfg.algorithm.desired_kl = 0.005
  cfg.experiment_name = "g1_goalkeeper_ballistic_residual"
  cfg.save_interval = 50
  return cfg


def goalkeeper_lstm_student_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Runner config for a single recurrent goalkeeper student policy."""
  cfg = goalkeeper_train_runner_cfg()
  cfg.actor.class_name = "src.tasks.soccer.modules.gk_lstm_student.GoalkeeperLSTMStudent"
  cfg.actor.hidden_dims = (256, 128)
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.05,
    "std_type": "scalar",
  }
  cfg.critic.hidden_dims = (512, 256, 256)
  cfg.algorithm.entropy_coef = 0.0
  cfg.algorithm.clip_param = 0.08
  cfg.algorithm.learning_rate = 3.0e-4
  cfg.algorithm.desired_kl = 0.005
  cfg.experiment_name = "g1_goalkeeper_lstm_student"
  cfg.save_interval = 50
  return cfg


def goalkeeper_lstm_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Pure PPO LSTM goalkeeper config, with no teacher/MoE dependency.

  This mirrors the shooter recurrent PPO setup, but keeps goalkeeper
  observation normalization off because keeper observations are already scaled
  in the observation functions.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.8,
        "std_type": "scalar",
      },
      class_name="RNNModel",
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=False,
      class_name="RNNModel",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=5.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_goalkeeper_lstm_ppo",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=12000,
  )


class GoalkeeperRecurrentRunner(MjlabOnPolicyRunner):
  """Runner that injects recurrent params into RSL-RL's RNNModel config."""

  def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
    for key in ("actor", "critic"):
      if train_cfg[key].get("class_name") == "RNNModel":
        train_cfg[key].setdefault("rnn_type", "lstm")
        train_cfg[key].setdefault("rnn_hidden_dim", 256)
        train_cfg[key].setdefault("rnn_num_layers", 1)
    super().__init__(env, train_cfg, log_dir, device, **kwargs)
