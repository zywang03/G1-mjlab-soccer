#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPUS="4,5,6,7"
BASE_CKPT="src/assets/soccer/weight/goalkeeper_moe6_hard3_default.pt"
INIT_CKPT="src/assets/soccer/weight/model_repaired_lyk.pt"
BASE_EXPERT_DIR="logs/lyk/fuse_8gpu/hard3"
OUT_ROOT="logs/lyk/failure_replay"
FINAL_CKPT="src/assets/soccer/weight/goalkeeper_moe6_failure_replay.pt"
NUM_ENVS=4096
COLLECT_ENVS=256
COLLECT_BATCHES=64
EVAL_BATCHES=64
BLOCKS=120
OFFICIAL_TRIALS=500
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/lyk/failure_replay_${STAMP}.log"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_failure_replay_pipeline.sh [options]

Options:
  --gpus LIST             Comma-separated physical GPU ids (default: 4,5,6,7)
  --base-ckpt PATH        Checkpoint to mine failures from
  --init PATH             Frozen-base/init checkpoint for residual PPO
  --base-expert-dir DIR   Six-expert directory used as fusion baseline
  --out-root DIR          Output workspace
  --final PATH            Output bundled checkpoint
  --num-envs N            PPO envs per training process (default: 4096)
  --collect-envs N        Failure collection env count (default: 256)
  --collect-batches N     Failure collection batches (default: 64)
  --eval-batches N        Candidate eval batches (default: 64)
  --blocks N              PPO blocks per specialist (default: 120)
  --official-trials N     Final official eval trials (default: 500)
  --log PATH              Combined log file
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --base-ckpt) BASE_CKPT="$2"; shift 2 ;;
    --init) INIT_CKPT="$2"; shift 2 ;;
    --base-expert-dir) BASE_EXPERT_DIR="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --final) FINAL_CKPT="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --collect-envs) COLLECT_ENVS="$2"; shift 2 ;;
    --collect-batches) COLLECT_BATCHES="$2"; shift 2 ;;
    --eval-batches) EVAL_BATCHES="$2"; shift 2 ;;
    --blocks) BLOCKS="$2"; shift 2 ;;
    --official-trials) OFFICIAL_TRIALS="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a GPU_ARR <<<"$GPUS"
if [[ ${#GPU_ARR[@]} -lt 4 ]]; then
  echo "[ERROR] need four GPUs for regions 0/1/2/3" >&2
  exit 1
fi

FAIL_CSV="$OUT_ROOT/failures.csv"
EXPERT_DIR="$OUT_ROOT/experts"
FUSE_DIR="$OUT_ROOT/fuse"
EVAL_DIR="$OUT_ROOT/eval"
RUN_LOG_DIR="$OUT_ROOT/train_logs"

mkdir -p "$(dirname "$LOG")" "$OUT_ROOT" "$EXPERT_DIR" "$FUSE_DIR" "$EVAL_DIR" "$RUN_LOG_DIR" "$(dirname "$FINAL_CKPT")"
exec > >(tee -a "$LOG") 2>&1

echo "===== Failure replay pipeline started: $(date) ====="
echo "[INFO] repo=$ROOT"
echo "[INFO] gpus=$GPUS num_envs=$NUM_ENVS blocks=$BLOCKS"
echo "[INFO] base_ckpt=$BASE_CKPT"
echo "[INFO] base_expert_dir=$BASE_EXPERT_DIR"
echo "[INFO] final_ckpt=$FINAL_CKPT"
echo "[INFO] log=$LOG"
git --no-pager log --oneline -3 || true
git --no-pager status --short || true

run_cuda() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="$gpu" \
  MUJOCO_EGL_DEVICE_ID=0 \
  PYOPENGL_PLATFORM=egl \
  MUJOCO_GL=egl \
  WANDB_MODE=disabled \
  "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] missing required file: $1" >&2
    exit 1
  fi
}

require_expert_dir() {
  local dir="$1"
  for idx in 0 1 2 3 4 5; do
    require_file "$dir/stable_sr${idx}.pt"
  done
}

if [[ ! -f "$BASE_CKPT" ]]; then
  for fallback in \
    src/assets/soccer/weight/goalkeeper_moe6_lyk_selected.pt \
    src/assets/soccer/weight/goalkeeper_moe6_8gpu_hard3.pt \
    src/assets/soccer/weight/goalkeeper_moe6_lyk_v2.pt \
    src/assets/soccer/weight/goalkeeper_moe6_8gpu_base.pt; do
    if [[ -f "$fallback" ]]; then
      echo "[WARN] base ckpt not found, using fallback: $fallback"
      BASE_CKPT="$fallback"
      break
    fi
  done
fi
require_file "$BASE_CKPT"
require_file "$INIT_CKPT"
require_expert_dir "$BASE_EXPERT_DIR"

echo
echo "===== Collect current failures ====="
run_cuda "${GPU_ARR[0]}" python scripts/collect_moe6_failures.py \
  --checkpoint "$BASE_CKPT" \
  --out-csv "$FAIL_CSV" \
  --num-envs "$COLLECT_ENVS" \
  --batches "$COLLECT_BATCHES" \
  --device cuda:0

FAIL_COUNT="$(python - "$FAIL_CSV" <<'PY'
import csv, sys
with open(sys.argv[1], newline="") as f:
  print(sum(1 for _ in csv.DictReader(f)))
PY
)"
echo "[INFO] failure rows=$FAIL_COUNT"
if [[ "$FAIL_COUNT" -lt 100 ]]; then
  echo "[WARN] few failures collected; replay signal may be noisy"
fi

train_region() {
  local region="$1"
  local name="$2"
  local gpu="$3"
  local weights="$4"
  local ratio="$5"
  local std="$6"
  local scale="$7"
  local seed="$8"
  local out="$EXPERT_DIR/stable_sr${region}_${name}.pt"
  local log="$RUN_LOG_DIR/sr${region}_${name}_gpu${gpu}.log"
  echo "[LAUNCH] region=$region name=$name gpu=$gpu out=$out"
  run_cuda "$gpu" python scripts/train_ballistic_residual.py \
    --init "$INIT_CKPT" \
    --out "$out" \
    --region-weights $weights \
    --failure-csv "$FAIL_CSV" \
    --failure-regions "$region" \
    --failure-replay-ratio "$ratio" \
    --failure-pos-jitter 0.025 \
    --failure-vel-jitter 0.06 \
    --num-envs "$NUM_ENVS" \
    --warmup 30 \
    --block-iters 20 \
    --blocks "$BLOCKS" \
    --eval-resets 4 \
    --lr 3e-05 \
    --std "$std" \
    --residual-scale "$scale" \
    --w-conceded 24.0 \
    --w-intercept 4.0 \
    --w-body 1.8 \
    --w-stop 1.2 \
    --w-posture 0.8 \
    --w-recovery 2.5 \
    --w-post-save-ang-vel 0.03 \
    --post-save-action-rate 0.03 \
    --w-feet-slip 0.04 \
    --w-ang-vel 0.02 \
    --action-rate 0.04 \
    --clip-param 0.07 \
    --desired-kl 0.004 \
    --stable-save-weight 0.25 \
    --rollback-drop 0.004 \
    --seed "$seed" \
    --device cuda:0 > "$log" 2>&1 &
  TRAIN_PIDS+=("$!")
  TRAIN_LOGS+=("$log")
  TRAIN_NAMES+=("sr${region}_${name}")
}

echo
echo "===== Train failure-replay specialists ====="
TRAIN_PIDS=()
TRAIN_LOGS=()
TRAIN_NAMES=()
train_region 0 right_mid "${GPU_ARR[0]}" "4.0 1.5 1.5 1.5 0.5 0.5" 0.25 0.045 0.42 7310
train_region 1 left_mid "${GPU_ARR[1]}" "1.0 4.0 1.5 1.5 0.5 0.5" 0.30 0.045 0.42 7311
train_region 2 right_up "${GPU_ARR[2]}" "1.0 1.5 4.0 1.5 0.5 0.5" 0.35 0.045 0.42 7312
train_region 3 left_up "${GPU_ARR[3]}" "1.0 1.5 1.5 4.0 0.5 0.5" 0.35 0.045 0.42 7313
monitor_training() {
  while :; do
    sleep 120
    local running=0
    for pid in "${TRAIN_PIDS[@]}"; do
      if ps -p "$pid" >/dev/null 2>&1; then
        running=$((running + 1))
      fi
    done
    [[ "$running" -eq 0 ]] && break
    echo "[WAIT] $running specialist training jobs still running at $(date)"
    for i in "${!TRAIN_LOGS[@]}"; do
      echo "--- ${TRAIN_NAMES[$i]} tail ---"
      grep -E "\[INFO\] failure replay|\[EVAL\].*(best|rollback|init)|saved ballistic|Traceback|RuntimeError" \
        "${TRAIN_LOGS[$i]}" | tail -8 || true
    done
  done
}

monitor_training &
MONITOR_PID="$!"
failed=0
for i in "${!TRAIN_PIDS[@]}"; do
  pid="${TRAIN_PIDS[$i]}"
  if ! wait "$pid"; then
    echo "[FAIL] ${TRAIN_NAMES[$i]} pid=$pid"
    tail -80 "${TRAIN_LOGS[$i]}" || true
    failed=1
  fi
done
kill "$MONITOR_PID" >/dev/null 2>&1 || true
wait "$MONITOR_PID" >/dev/null 2>&1 || true
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

echo
echo "===== Training summaries ====="
grep -hE "\[EVAL\].*(best|rollback)|saved ballistic|Traceback|RuntimeError|FAIL" "$RUN_LOG_DIR"/*.log || true

require_file "$EXPERT_DIR/stable_sr0_right_mid.pt"
require_file "$EXPERT_DIR/stable_sr1_left_mid.pt"
require_file "$EXPERT_DIR/stable_sr2_right_up.pt"
require_file "$EXPERT_DIR/stable_sr3_left_up.pt"

make_candidate() {
  local name="$1"
  mkdir -p "$FUSE_DIR/$name"
  cp "$BASE_EXPERT_DIR"/stable_sr*.pt "$FUSE_DIR/$name"/
}

echo
echo "===== Build replay fuse candidates ====="
make_candidate base
make_candidate r0
cp "$EXPERT_DIR/stable_sr0_right_mid.pt" "$FUSE_DIR/r0/stable_sr0.pt"
make_candidate r1
cp "$EXPERT_DIR/stable_sr1_left_mid.pt" "$FUSE_DIR/r1/stable_sr1.pt"
make_candidate r2
cp "$EXPERT_DIR/stable_sr2_right_up.pt" "$FUSE_DIR/r2/stable_sr2.pt"
make_candidate r3
cp "$EXPERT_DIR/stable_sr3_left_up.pt" "$FUSE_DIR/r3/stable_sr3.pt"
make_candidate r123
cp "$EXPERT_DIR/stable_sr1_left_mid.pt" "$FUSE_DIR/r123/stable_sr1.pt"
cp "$EXPERT_DIR/stable_sr2_right_up.pt" "$FUSE_DIR/r123/stable_sr2.pt"
cp "$EXPERT_DIR/stable_sr3_left_up.pt" "$FUSE_DIR/r123/stable_sr3.pt"
make_candidate r0123
cp "$EXPERT_DIR/stable_sr0_right_mid.pt" "$FUSE_DIR/r0123/stable_sr0.pt"
cp "$EXPERT_DIR/stable_sr1_left_mid.pt" "$FUSE_DIR/r0123/stable_sr1.pt"
cp "$EXPERT_DIR/stable_sr2_right_up.pt" "$FUSE_DIR/r0123/stable_sr2.pt"
cp "$EXPERT_DIR/stable_sr3_left_up.pt" "$FUSE_DIR/r0123/stable_sr3.pt"
make_candidate r1_r2mirror
cp "$EXPERT_DIR/stable_sr1_left_mid.pt" "$FUSE_DIR/r1_r2mirror/stable_sr1.pt"
cp "$EXPERT_DIR/stable_sr2_right_up.pt" "$FUSE_DIR/r1_r2mirror/stable_sr2.pt"
make_candidate r0mirror_r2mirror
cp "$EXPERT_DIR/stable_sr0_right_mid.pt" "$FUSE_DIR/r0mirror_r2mirror/stable_sr0.pt"
cp "$EXPERT_DIR/stable_sr2_right_up.pt" "$FUSE_DIR/r0mirror_r2mirror/stable_sr2.pt"

BEST_NAME=""
BEST_MIRROR=""
BEST_RATE="-1"

eval_candidate() {
  local name="$1"
  local mirror="$2"
  local out="$EVAL_DIR/${name}.log"
  echo
  echo "===== Eval candidate: $name mirror='$mirror' ====="
  run_cuda "${GPU_ARR[0]}" python scripts/eval_stable_moe6.py \
    --expert-dir "$FUSE_DIR/$name" \
    --prefix stable_sr \
    --mirror-map "$mirror" \
    --num-envs 256 \
    --batches "$EVAL_BATCHES" \
    --steps 149 \
    --latch-hi 5.0 \
    --z-low 0.85 \
    --z-up 1.35 \
    --device cuda:0 > "$out" 2>&1
  cat "$out"
  local rate
  rate="$(awk '/MoE6 block:/ {gsub("%", "", $NF); print $NF}' "$out" | tail -1)"
  echo "[RESULT] $name block=${rate}%"
  if awk -v a="$rate" -v b="$BEST_RATE" 'BEGIN { exit !(a > b) }'; then
    BEST_NAME="$name"
    BEST_MIRROR="$mirror"
    BEST_RATE="$rate"
  fi
}

eval_candidate base "1:0"
eval_candidate r0 "1:0,3:2"
eval_candidate r1 "3:2"
eval_candidate r2 "1:0,3:2"
eval_candidate r3 "1:0"
eval_candidate r123 ""
eval_candidate r0123 ""
eval_candidate r1_r2mirror "3:2"
eval_candidate r0mirror_r2mirror "1:0,3:2"

echo
echo "===== Candidate summary ====="
grep -hE "MoE6 block|Right-|Left-|Gate accuracy" "$EVAL_DIR"/*.log || true
echo "[BEST_CANDIDATE] name=$BEST_NAME mirror='$BEST_MIRROR' block=${BEST_RATE}%"

echo
echo "===== Bundle final checkpoint ====="
python scripts/bundle_moe6.py \
  --expert-dir "$FUSE_DIR/$BEST_NAME" \
  --prefix stable_sr \
  --out "$FINAL_CKPT" \
  --mirror-map "$BEST_MIRROR" \
  --z-low 0.85 \
  --z-up 1.35 \
  --latch-hi 5.0

echo
echo "===== Official eval ====="
run_cuda "${GPU_ARR[0]}" python scripts/eval_naive_goalkeeper.py \
  --headless \
  --num-trials "$OFFICIAL_TRIALS" \
  --checkpoint "$FINAL_CKPT" \
  --device cuda:0

echo
echo "===== Failure replay pipeline finished: $(date) ====="
echo "[INFO] final_ckpt=$FINAL_CKPT"
echo "[INFO] log=$LOG"
