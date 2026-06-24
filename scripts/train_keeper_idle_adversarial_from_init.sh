#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="/home/wangzhy/software/miniconda3/envs/unitree_rl_mjlab/bin/python"

SHOOTER_INIT="checkpoints/stage4/model_143984.pt"
KEEPER_INIT="logs/repairs/goalkeeper_moe7_idle.pt"
KEEPER_IDLE_INIT="logs/repairs/goalkeeper_moe7_idle.pt"

ROUNDS="1"
NUM_ENVS="512"
GPU_IDS="0"
SEED="2810"
RUN_TAG="keeper_idle_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="logs/adversarial/${RUN_TAG}"

KEEPER_IDLE_ITERS_PER_ROUND="10000"
PROMOTION_TRIALS="0"
KEEPER_LOG_ROOT="logs/rsl_rl/adversarial"

for path in "${SHOOTER_INIT}" "${KEEPER_INIT}" "${KEEPER_IDLE_INIT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required path: ${path}" >&2
    exit 1
  fi
done

echo "[INFO] Starting keeper prepare/idle adversarial training"
echo "[INFO] shooter=${SHOOTER_INIT}"
echo "[INFO] keeper=${KEEPER_INIT}"
echo "[INFO] keeper_idle=${KEEPER_IDLE_INIT}"
echo "[INFO] out_dir=${OUT_DIR}"
echo "[INFO] rounds=${ROUNDS}, num_envs=${NUM_ENVS}, gpu_ids=${GPU_IDS}"

exec "${PYTHON}" scripts/train_adversarial.py \
  --keeper-idle-only \
  --rounds "${ROUNDS}" \
  --no-dry-run \
  --out-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --shooter-init "${SHOOTER_INIT}" \
  --keeper-init "${KEEPER_INIT}" \
  --keeper-idle-init "${KEEPER_IDLE_INIT}" \
  --keeper-log-root "${KEEPER_LOG_ROOT}" \
  --num-envs "${NUM_ENVS}" \
  --gpu-ids "${GPU_IDS}" \
  --keeper-idle-iters-per-round "${KEEPER_IDLE_ITERS_PER_ROUND}" \
  --promotion-trials "${PROMOTION_TRIALS}" \
  "$@"
