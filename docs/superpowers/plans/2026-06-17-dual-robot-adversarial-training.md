# Dual Robot Adversarial Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the existing PPO training scheme while running a real frozen opponent robot in the same MuJoCo scene and exposing opponent state to the trainable policy.

**Architecture:** Add adversarial task configs that create two robot entities and two low-level action terms. Wrap the RSL-RL vec env so the runner still emits only the active robot action while the wrapper injects frozen opponent actions from a checkpoint-backed policy. Use zero-initialized adapters/residuals so old checkpoints start with unchanged behavior when opponent observations are appended.

**Tech Stack:** mjlab ManagerBasedRlEnv, RSL-RL PPO runner/model interfaces, Unitree G1 soccer task registry, Python unittest.

---

### Task 1: Tests First

**Files:**
- Modify: `tests/test_train_adversarial_scheduler.py`
- Create: `tests/test_adversarial_dual_robot_cfg.py`
- Create: `tests/test_adversarial_zero_init_models.py`

- [ ] Add scheduler assertions that dry-run phases mark `opponent_consumed_by_backend=True` and use adversarial task ids.
- [ ] Add config assertions that shooter and keeper adversarial envs contain `robot`, `opponent`, `ball`, and active/opponent action terms.
- [ ] Add model assertions that zero-initialized adversarial models match their base outputs when opponent features are present.

### Task 2: Dual Robot Configs

**Files:**
- Create: `src/tasks/soccer/mdp/opponent_obs.py`
- Create: `src/tasks/soccer/config/training/adversarial_env_cfg.py`
- Modify: `src/tasks/soccer/config/g1/training_env_cfgs.py`
- Modify: `src/tasks/soccer/config/g1/__init__.py`

- [ ] Add reusable opponent observation terms in the active robot frame.
- [ ] Add shooter adversarial env factory based on Stage III with a real goalkeeper opponent.
- [ ] Add goalkeeper adversarial env factory based on goalkeeper training with a real shooter opponent.
- [ ] Register `Unitree-G1-Shooter-Adversarial` and `Unitree-G1-Goalkeeper-Adversarial`.

### Task 3: Frozen Opponent Action Injection

**Files:**
- Create: `src/tasks/soccer/adversarial.py`
- Modify: `scripts/train.py`
- Modify: `scripts/train_adversarial.py`

- [ ] Add a vec-env wrapper that exposes only active action dim and concatenates frozen opponent actions before stepping the dual-action env.
- [ ] Add checkpoint path fields to `TrainConfig` and wrap only adversarial envs.
- [ ] Update the alternating scheduler to pass sampled opponent checkpoints into the training backend.

### Task 4: Zero-Init Compatibility

**Files:**
- Create: `src/tasks/soccer/modules/adversarial_models.py`
- Modify: `src/tasks/soccer/config/g1/rl_cfg.py`

- [ ] Add an RNN model variant that ignores appended opponent dims at initialization and can expand old recurrent checkpoint input weights.
- [ ] Add a goalkeeper adversarial actor-critic that runs the old 960D/113D base path and adds zero-initialized opponent residual outputs.
- [ ] Add runner load migration for old checkpoints into the adversarial model shapes.

### Task 5: Verification

**Commands:**
- `/home/wangzhy/software/miniconda3/envs/unitree_rl_mjlab/bin/python -m unittest tests.test_train_adversarial_scheduler tests.test_adversarial_dual_robot_cfg tests.test_adversarial_zero_init_models`
- `/home/wangzhy/software/miniconda3/envs/unitree_rl_mjlab/bin/python -m compileall -q scripts src/tasks/soccer tests`
- `/home/wangzhy/software/miniconda3/envs/unitree_rl_mjlab/bin/python scripts/train.py Unitree-G1-Shooter-Adversarial --help`
- `/home/wangzhy/software/miniconda3/envs/unitree_rl_mjlab/bin/python scripts/train.py Unitree-G1-Goalkeeper-Adversarial --help`
