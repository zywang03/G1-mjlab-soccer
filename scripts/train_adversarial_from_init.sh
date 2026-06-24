#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="/home/wangzhy/software/miniconda3/envs/unitree_rl_mjlab/bin/python"

SHOOTER_INIT="/data/Courses/[CS2810]EmbodiedAI/humanoid_soccer_proj/G1-mjlab-soccer/checkpoints/stage4/model_143984.pt"
KEEPER_INIT="src/assets/soccer/weight/goalkeeper_moe6.pt"
KEEPER_IDLE_INIT="src/assets/soccer/weight/goalkeeper_moe6.pt"
MOTION_DIR="src/assets/soccer/motions/shooter"

ROUNDS="1"
MODE="train-keeper"
NUM_ENVS="512"
GPU_IDS="0"
SEED="2810"
RUN_TAG="init_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="logs/adversarial/${RUN_TAG}"

SHOOTER_ITERS_PER_ROUND="3000"
KEEPER_BLOCKS_PER_ROUND="200"
KEEPER_BLOCK_ITERS="20"
KEEPER_IDLE_ITERS_PER_ROUND="1000"
SHOOTER_TARGETS_PER_ROUND="64"
PROMOTION_TRIALS="64"

SHOOTER_LOG_ROOT="logs/rsl_rl/adversarial"
KEEPER_LOG_ROOT="logs/rsl_rl/adversarial"
KEEPER_DEVICE="cuda:0"
PROMOTION_DEVICE="cuda:0"

for path in "${SHOOTER_INIT}" "${KEEPER_INIT}" "${KEEPER_IDLE_INIT}" "${MOTION_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required path: ${path}" >&2
    exit 1
  fi
done

echo "[INFO] Starting adversarial training from init checkpoints"
echo "[INFO] shooter=${SHOOTER_INIT}"
echo "[INFO] keeper=${KEEPER_INIT}"
echo "[INFO] out_dir=${OUT_DIR}"
echo "[INFO] mode=${MODE}, rounds=${ROUNDS}, num_envs=${NUM_ENVS}, gpu_ids=${GPU_IDS}"

exec "${PYTHON}" scripts/train_adversarial.py \
  --mode "${MODE}" \
  --rounds "${ROUNDS}" \
  --no-dry-run \
  --out-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --shooter-init "${SHOOTER_INIT}" \
  --keeper-init "${KEEPER_INIT}" \
  --keeper-idle-init "${KEEPER_IDLE_INIT}" \
  --motion-dir "${MOTION_DIR}" \
  --shooter-log-root "${SHOOTER_LOG_ROOT}" \
  --keeper-log-root "${KEEPER_LOG_ROOT}" \
  --num-envs "${NUM_ENVS}" \
  --gpu-ids "${GPU_IDS}" \
  --keeper-device "${KEEPER_DEVICE}" \
  --shooter-iters-per-round "${SHOOTER_ITERS_PER_ROUND}" \
  --keeper-blocks-per-round "${KEEPER_BLOCKS_PER_ROUND}" \
  --keeper-block-iters "${KEEPER_BLOCK_ITERS}" \
  --keeper-idle-iters-per-round "${KEEPER_IDLE_ITERS_PER_ROUND}" \
  --promotion-trials "${PROMOTION_TRIALS}" \
  --promotion-device "${PROMOTION_DEVICE}" \
  --shooter-targets-per-round "${SHOOTER_TARGETS_PER_ROUND}" \
  "$@"
