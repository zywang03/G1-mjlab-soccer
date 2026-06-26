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

from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPpoAlgorithmCfg


_BASE_MLP_HIDDEN = (512, 256, 128)
_RNN_HIDDEN = (128, 64, 32)
_RNN_TYPE = "lstm"
_RNN_HIDDEN_DIM = 128
_RNN_NUM_LAYERS = 2
_GK_STUDENT_HIDDEN = (256, 128, 64)
_GK_STUDENT_RNN_HIDDEN_DIM = 160
_GK_STUDENT_RNN_NUM_LAYERS = 2
_GK_STUDENT_CONDITION_HIDDEN_DIM = 64
_GK_STUDENT_BALL_LATENT_DIM = 6


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


def unitree_g1_soccer_adversarial_recurrent_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """LSTM PPO config for shooter adversarial training with appended opponent obs."""
  cfg = unitree_g1_soccer_recurrent_runner_cfg()
  cfg.actor.class_name = "AdversarialRNNModel"
  cfg.critic.class_name = "AdversarialRNNModel"
  return cfg


def unitree_g1_student_shooter_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """LSTM PPO config for motion-free shooter student.

  Uses the same RNN architecture as Stage II (128-64-32 MLP + LSTM 2×128)
  so BC actor checkpoints can initialize PPO with ``--load-actor-only``.
  RNN hidden/type/layers are injected by ``SoccerRecurrentRunner``.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=_RNN_HIDDEN,
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
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
      learning_rate=3.0e-4, schedule="adaptive", gamma=0.99, lam=0.95,
      desired_kl=0.01, max_grad_norm=1.0,
    ),
    experiment_name="g1_soccer_student",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=20000,
  )


def unitree_g1_goalkeeper_student_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO config for the position-conditioned FiLM LSTM goalkeeper student.

  Actor consumes the 964D student observation. Critic is also recurrent, but it
  still consumes the existing privileged goalkeeper critic state.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=_GK_STUDENT_HIDDEN,
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.2, "std_type": "scalar"},
      class_name="GoalkeeperStudentFiLMActor",
    ),
    critic=RslRlModelCfg(
      hidden_dims=_RNN_HIDDEN,
      activation="elu",
      obs_normalization=True,
      class_name="RNNModel",
    ),
    algorithm=GoalkeeperStudentPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
      entropy_coef=0.001, num_learning_epochs=5, num_mini_batches=4,
      learning_rate=3.0e-4, schedule="adaptive", gamma=0.99, lam=0.95,
      desired_kl=0.01, max_grad_norm=1.0,
    ),
    experiment_name="g1_goalkeeper_student",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10000,
  )


class SoccerRecurrentRunner(MjlabOnPolicyRunner):
  """Runner that injects LSTM params into the RSL-RL config dict.

  RslRlModelCfg doesn't have rnn_* fields, so we add them after asdict().
  log_dir is optional (not needed for play/inference).
  """

  def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
    for key in ("actor", "critic"):
      if train_cfg[key].get("class_name") in ("RNNModel", "AdversarialRNNModel"):
        train_cfg[key].setdefault("rnn_type", _RNN_TYPE)
        train_cfg[key].setdefault("rnn_hidden_dim", _RNN_HIDDEN_DIM)
        train_cfg[key].setdefault("rnn_num_layers", _RNN_NUM_LAYERS)
    super().__init__(env, train_cfg, log_dir, device, **kwargs)


# -- Goalkeeper: HIMPPO-inspired ActorCritic ----------------------------------

_GK_ACTOR_HIDDEN = (512, 256, 256)
_GK_CRITIC_HIDDEN = (512, 256, 256)


def unitree_g1_goalkeeper_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Goalkeeper PPO config using the custom GoalkeeperActorCritic model.

  Actor uses history encoder (960→16D) + ball/region estimators → MLP [512,256,256].
  Critic uses MLP [512,256,256] on 113D privileged obs.

  obs_normalization is DISABLED because the reference model was trained with
  manual observation scaling (obs_scales: ang_vel*0.25, dof_vel*0.05, etc.)
  applied in compute_observations, NOT with RSL-RL's running-mean normalizer.
  Double-normalizing would produce wrong observation values.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=_GK_ACTOR_HIDDEN,
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      class_name="GoalkeeperActorCritic",
    ),
    critic=RslRlModelCfg(
      hidden_dims=_GK_CRITIC_HIDDEN,
      activation="elu",
      obs_normalization=False,
      class_name="GoalkeeperActorCritic",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_soccer",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def unitree_g1_goalkeeper_adversarial_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Goalkeeper PPO config with zero-initialized opponent residuals."""
  cfg = unitree_g1_goalkeeper_ppo_runner_cfg()
  cfg.actor.class_name = "AdversarialGoalkeeperActorCritic"
  cfg.critic.class_name = "AdversarialGoalkeeperActorCritic"
  return cfg


def unitree_g1_goalkeeper_moe7_prepare_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Goalkeeper PPO config that runs full MoE7 but trains prepare only."""
  cfg = unitree_g1_goalkeeper_adversarial_ppo_runner_cfg()
  cfg.actor.class_name = "MoE7PrepareGoalkeeperActor"
  cfg.critic.class_name = "AdversarialGoalkeeperActorCritic"
  cfg.actor.distribution_cfg = dict(cfg.actor.distribution_cfg)
  cfg.actor.distribution_cfg.update({
    "freeze_idle_std": True,
    "idle_std_min": 0.15,
    "idle_std_max": 0.15,
  })
  cfg.algorithm.entropy_coef = 0.0
  return cfg


class GoalkeeperRunner(MjlabOnPolicyRunner):
  """Runner that injects GoalkeeperActorCritic into rsl_rl.models.

  RSL-RL uses eval(class_name) to resolve model classes. We monkey-patch
  our custom model into rsl_rl.models so that eval("GoalkeeperActorCritic")
  resolves correctly.
  """

  def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
    import rsl_rl.models
    from src.tasks.soccer.modules.gk_actor_critic import GoalkeeperActorCritic

    rsl_rl.models.GoalkeeperActorCritic = GoalkeeperActorCritic
    super().__init__(env, train_cfg, log_dir, device, **kwargs)

  def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
    loaded = _load_goalkeeper_compatible_checkpoint(self, path, load_cfg, map_location)
    if loaded is not None:
      return loaded
    return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)


class GoalkeeperStudentRunner(MjlabOnPolicyRunner):
  """Runner that registers the FiLM student actor and hybrid PPO algorithm."""

  @staticmethod
  def _inject_recurrent_defaults(train_cfg: dict) -> None:
    train_cfg["actor"].setdefault("rnn_type", _RNN_TYPE)
    train_cfg["actor"].setdefault("rnn_hidden_dim", _GK_STUDENT_RNN_HIDDEN_DIM)
    train_cfg["actor"].setdefault("rnn_num_layers", _GK_STUDENT_RNN_NUM_LAYERS)
    train_cfg["actor"].setdefault("condition_hidden_dim", _GK_STUDENT_CONDITION_HIDDEN_DIM)
    train_cfg["actor"].setdefault("ball_latent_dim", _GK_STUDENT_BALL_LATENT_DIM)
    train_cfg["critic"].setdefault("rnn_type", _RNN_TYPE)
    train_cfg["critic"].setdefault("rnn_hidden_dim", _GK_STUDENT_RNN_HIDDEN_DIM)
    train_cfg["critic"].setdefault("rnn_num_layers", _RNN_NUM_LAYERS)

  def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
    import rsl_rl.algorithms
    import rsl_rl.models
    from src.tasks.soccer.modules.goalkeeper_student_actor import GoalkeeperStudentFiLMActor
    from src.tasks.soccer.modules.goalkeeper_student_ppo import GoalkeeperStudentPPO

    rsl_rl.algorithms.GoalkeeperStudentPPO = GoalkeeperStudentPPO
    rsl_rl.models.GoalkeeperStudentFiLMActor = GoalkeeperStudentFiLMActor
    self._inject_recurrent_defaults(train_cfg)
    super().__init__(env, train_cfg, log_dir, device, **kwargs)

  def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
    loaded = _load_goalkeeper_compatible_checkpoint(self, path, load_cfg, map_location)
    if loaded is not None:
      return loaded
    return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)


def _filter_compatible_state_dict(model, state_dict):
  own = model.state_dict()
  return {
    key: value
    for key, value in state_dict.items()
    if key in own and own[key].shape == value.shape
  }


def _compatible_state_dict_or_raise(model, state_dict, path: str, label: str):
  compatible = _filter_compatible_state_dict(model, state_dict)
  if not compatible:
    raise ValueError(
      f"No compatible {label} weights found in checkpoint {path}; "
      f"checkpoint keys look incompatible with {type(model).__name__}."
    )
  return compatible


def _load_goalkeeper_compatible_checkpoint(
  runner,
  path: str,
  load_cfg: dict | None = None,
  map_location: str | None = None,
):
  import torch

  loaded = torch.load(path, map_location=map_location or runner.device, weights_only=False)
  if isinstance(loaded, dict) and "sr" in loaded:
    loaded = loaded["sr"][0]

  if isinstance(loaded, dict) and "model_state_dict" in loaded:
    actor_sd = {
      key: value
      for key, value in loaded["model_state_dict"].items()
      if not key.startswith("critic.")
    }
    actor_sd = _compatible_state_dict_or_raise(runner.alg.actor, actor_sd, path, "actor")
    runner.alg.actor.load_state_dict(actor_sd, strict=False)
    return loaded.get("infos", {})

  if isinstance(loaded, dict) and "actor_state_dict" in loaded:
    load_actor = load_cfg is None or load_cfg.get("actor", True)
    load_critic = load_cfg is None or load_cfg.get("critic", True)
    if load_actor:
      actor_sd = _compatible_state_dict_or_raise(
        runner.alg.actor, loaded["actor_state_dict"], path, "actor"
      )
      runner.alg.actor.load_state_dict(actor_sd, strict=False)
    if load_critic and "critic_state_dict" in loaded:
      runner.alg.critic.load_state_dict(
        _filter_compatible_state_dict(runner.alg.critic, loaded["critic_state_dict"]),
        strict=False,
      )
    return loaded.get("infos", {})

  return None


class _AdversarialLoadMixin:
  """Load old checkpoints into widened adversarial models when possible."""

  def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
    loaded = _load_goalkeeper_compatible_checkpoint(self, path, load_cfg, map_location)
    if loaded is not None:
      return loaded
    return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)


class AdversarialSoccerRecurrentRunner(_AdversarialLoadMixin, SoccerRecurrentRunner):
  """Shooter recurrent runner with adversarial model registration."""

  def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
    import rsl_rl.models
    from src.tasks.soccer.modules.adversarial_models import AdversarialRNNModel

    rsl_rl.models.AdversarialRNNModel = AdversarialRNNModel
    train_cfg["actor"].setdefault("base_obs_dim", 160)
    train_cfg["critic"].setdefault("base_obs_dim", 298)
    super().__init__(env, train_cfg, log_dir, device, **kwargs)


class AdversarialGoalkeeperRunner(_AdversarialLoadMixin, GoalkeeperRunner):
  """Goalkeeper runner with adversarial model registration."""

  def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
    import rsl_rl.models
    from src.tasks.soccer.modules.adversarial_models import AdversarialGoalkeeperActorCritic

    rsl_rl.models.AdversarialGoalkeeperActorCritic = AdversarialGoalkeeperActorCritic
    super().__init__(env, train_cfg, log_dir, device, **kwargs)


class MoE7PrepareGoalkeeperRunner(AdversarialGoalkeeperRunner):
  """Runner that loads/saves a deployable MoE7 actor while training prepare only."""

  def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
    import rsl_rl.models
    from src.tasks.soccer.modules.moe7_prepare_actor import MoE7PrepareGoalkeeperActor

    rsl_rl.models.MoE7PrepareGoalkeeperActor = MoE7PrepareGoalkeeperActor
    super().__init__(env, train_cfg, log_dir, device, **kwargs)

  def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
    import torch

    loaded = torch.load(path, map_location=map_location or self.device, weights_only=False)
    actor_bundle = None
    if isinstance(loaded, dict) and "sr" in loaded:
      actor_bundle = loaded
    elif isinstance(loaded, dict) and isinstance(loaded.get("actor_state_dict"), dict):
      actor_state = loaded["actor_state_dict"]
      if "sr" in actor_state:
        actor_bundle = actor_state

    if actor_bundle is not None:
      load_actor = load_cfg is None or load_cfg.get("actor", True)
      load_critic = load_cfg is None or load_cfg.get("critic", True)
      if load_actor:
        self.alg.actor.load_moe_bundle(actor_bundle)
      if load_critic and isinstance(loaded, dict) and "critic_state_dict" in loaded:
        self.alg.critic.load_state_dict(
          _filter_compatible_state_dict(self.alg.critic, loaded["critic_state_dict"]),
          strict=False,
        )
      return loaded.get("infos", {}) if isinstance(loaded, dict) else {}

    return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
