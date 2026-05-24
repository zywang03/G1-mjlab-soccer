"""RL configuration for Unitree G1 soccer task.

Two configs:
  - MLP:  [512, 256, 128] ELU  (eval / baseline)
  - LSTM: [128, 64, 32] ELU + LSTM(2×128)  (training, matches reference)
"""

from mjlab.rl import (
  MjlabOnPolicyRunner,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


_BASE_MLP_HIDDEN = (512, 256, 128)
_RNN_HIDDEN = (128, 64, 32)
_RNN_TYPE = "lstm"
_RNN_HIDDEN_DIM = 128
_RNN_NUM_LAYERS = 2


def unitree_g1_soccer_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """MLP runner config (512-256-128 ELU) — eval / baseline."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=_BASE_MLP_HIDDEN,
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(
      hidden_dims=_BASE_MLP_HIDDEN,
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
      entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
      learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
      desired_kl=0.01, max_grad_norm=1.0,
    ),
    experiment_name="g1_soccer",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def unitree_g1_soccer_recurrent_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """LSTM runner config — matches HumanoidSoccer G1FlatRecurrentPPORunnerCfg.

  MLP [128,64,32] ELU + LSTM(2 layers, 128 hidden).
  Actor and critic share the same architecture.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=_RNN_HIDDEN,
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
      class_name="RNNModel",
    ),
    critic=RslRlModelCfg(
      hidden_dims=_RNN_HIDDEN,
      activation="elu",
      obs_normalization=True,
      class_name="RNNModel",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
      entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
      learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
      desired_kl=0.01, max_grad_norm=1.0,
    ),
    experiment_name="g1_soccer",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


class SoccerRecurrentRunner(MjlabOnPolicyRunner):
  """Runner that injects LSTM params into the RSL-RL config dict.

  RslRlModelCfg doesn't have rnn_* fields, so we add them after asdict().
  log_dir is optional (not needed for play/inference).
  """

  def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
    for key in ("actor", "critic"):
      train_cfg[key].setdefault("rnn_type", _RNN_TYPE)
      train_cfg[key].setdefault("rnn_hidden_dim", _RNN_HIDDEN_DIM)
      train_cfg[key].setdefault("rnn_num_layers", _RNN_NUM_LAYERS)
    super().__init__(env, train_cfg, log_dir, device, **kwargs)
