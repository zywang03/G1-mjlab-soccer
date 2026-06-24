#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="/home/wangzhy/software/miniconda3/envs/unitree_rl_mjlab/bin/python"

NUM_ENVS="1024"
MAX_ITERATIONS="10000"
NUM_STEPS_PER_ENV="24"
INIT_STD="0.10"
ENTROPY_COEF="0.0"
LEARNING_RATE="3.0e-4"
DESIRED_KL="0.005"
CLIP_PARAM="0.1"
GPU_IDS="[0]"
RUN_NAME="keeper_idle_train_$(date +%Y%m%d_%H%M%S)"

echo "[INFO] Starting random-init keeper idle/prepare training"
echo "[INFO] task=Unitree-G1-Goalkeeper-Idle-Train"
echo "[INFO] num_envs=${NUM_ENVS}, max_iterations=${MAX_ITERATIONS}, gpu_ids=${GPU_IDS}"
echo "[INFO] num_steps=${NUM_STEPS_PER_ENV}, init_std=${INIT_STD}, entropy=${ENTROPY_COEF}"

exec "${PYTHON}" scripts/train.py Unitree-G1-Goalkeeper-Idle-Train \
  --env.scene.num-envs "${NUM_ENVS}" \
  --agent.num-steps-per-env "${NUM_STEPS_PER_ENV}" \
  --agent.max-iterations "${MAX_ITERATIONS}" \
  --agent.actor.distribution-cfg.init-std "${INIT_STD}" \
  --agent.algorithm.entropy-coef "${ENTROPY_COEF}" \
  --agent.algorithm.learning-rate "${LEARNING_RATE}" \
  --agent.algorithm.desired-kl "${DESIRED_KL}" \
  --agent.algorithm.clip-param "${CLIP_PARAM}" \
  --agent.run-name "${RUN_NAME}" \
  --gpu-ids "${GPU_IDS}" \
  "$@"
