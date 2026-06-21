#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU=7
NUM_ENVS=256
BATCHES=64
OFFICIAL_TRIALS=500
FINAL_CKPT="src/assets/soccer/weight/goalkeeper_moe6_lyk_selected.pt"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/lyk/moe6_select_${STAMP}.log"

usage() {
  cat <<'EOF'
Usage: bash scripts/select_keeper_moe6.sh [options]

Options:
  --gpu ID              Physical GPU id to use after CUDA_VISIBLE_DEVICES (default: 7)
  --num-envs N          Eval env count (default: 256)
  --batches N           Eval batches per candidate (default: 64)
  --official-trials N   Final official eval trials (default: 500)
  --final PATH          Output bundled checkpoint
  --log PATH            One combined log file
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --batches) BATCHES="$2"; shift 2 ;;
    --official-trials) OFFICIAL_TRIALS="$2"; shift 2 ;;
    --final) FINAL_CKPT="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$LOG")" logs/lyk/fuse_select logs/lyk/select_eval logs/lyk/gate_logs "$(dirname "$FINAL_CKPT")"
exec > >(tee -a "$LOG") 2>&1

echo "===== MoE6 selection started: $(date) ====="
echo "[INFO] repo=$ROOT"
echo "[INFO] gpu=$GPU num_envs=$NUM_ENVS batches=$BATCHES official_trials=$OFFICIAL_TRIALS"
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

BASE_DIR="logs/lyk/fuse_8gpu/base"
HARD3_DIR="logs/lyk/fuse_8gpu/hard3"
NEXT2_DIR="logs/lyk/experts_next2"
PROBE_DIR="logs/lyk/experts_probe"
SELECT_DIR="logs/lyk/fuse_select"

echo
echo "===== Input check ====="
require_expert_dir "$BASE_DIR"
if [[ -d "$HARD3_DIR" ]]; then
  require_expert_dir "$HARD3_DIR"
fi
ls -lh "$BASE_DIR"/stable_sr*.pt
ls -lh "$HARD3_DIR"/stable_sr*.pt 2>/dev/null || true
ls -lh "$NEXT2_DIR"/*.pt 2>/dev/null || true
ls -lh "$PROBE_DIR"/*.pt 2>/dev/null || true

make_from_base() {
  local name="$1"
  local dir="$SELECT_DIR/$name"
  mkdir -p "$dir"
  cp "$BASE_DIR"/stable_sr*.pt "$dir"/
}

make_from_dir() {
  local name="$1"
  local src="$2"
  local dir="$SELECT_DIR/$name"
  mkdir -p "$dir"
  cp "$src"/stable_sr*.pt "$dir"/
}

replace_if_exists() {
  local candidate="$1"
  local region="$2"
  local src="$3"
  if [[ -f "$src" ]]; then
    cp "$src" "$SELECT_DIR/$candidate/stable_sr${region}.pt"
    echo "[INFO] $candidate: stable_sr${region} <- $src"
    return 0
  fi
  echo "[WARN] $candidate: missing $src; candidate will be skipped"
  return 1
}

valid_candidate() {
  local name="$1"
  local dir="$SELECT_DIR/$name"
  for idx in 0 1 2 3 4 5; do
    [[ -f "$dir/stable_sr${idx}.pt" ]] || return 1
  done
  return 0
}

CANDIDATES=()

add_candidate() {
  local name="$1"
  local mirror="$2"
  if valid_candidate "$name"; then
    CANDIDATES+=("${name}|${mirror}")
  else
    echo "[WARN] skip invalid candidate: $name"
  fi
}

echo
echo "===== Build fuse candidates ====="
make_from_base base
add_candidate base "1:0,3:2"

if [[ -d "$HARD3_DIR" ]]; then
  make_from_dir hard3 "$HARD3_DIR"
  add_candidate hard3 "1:0"
fi

make_from_base sr1_cons
if replace_if_exists sr1_cons 1 "$NEXT2_DIR/stable_sr1_cons.pt"; then
  add_candidate sr1_cons "3:2"
fi

make_from_base sr1_aggr
if replace_if_exists sr1_aggr 1 "$NEXT2_DIR/stable_sr1_aggr.pt"; then
  add_candidate sr1_aggr "3:2"
fi

make_from_base sr2_up
if replace_if_exists sr2_up 2 "$NEXT2_DIR/stable_sr2_up.pt"; then
  add_candidate sr2_up "1:0,3:2"
fi

make_from_base sr3_up
if replace_if_exists sr3_up 3 "$NEXT2_DIR/stable_sr3_up.pt"; then
  add_candidate sr3_up "1:0"
fi

make_from_base sr1_block
if replace_if_exists sr1_block 1 "$PROBE_DIR/stable_sr1_blockfirst.pt"; then
  add_candidate sr1_block "3:2"
fi

make_from_base sr2_block
if replace_if_exists sr2_block 2 "$PROBE_DIR/stable_sr2_blockfirst.pt"; then
  add_candidate sr2_block "1:0,3:2"
fi

make_from_base sr1block_sr2block
if replace_if_exists sr1block_sr2block 1 "$PROBE_DIR/stable_sr1_blockfirst.pt" \
  && replace_if_exists sr1block_sr2block 2 "$PROBE_DIR/stable_sr2_blockfirst.pt"; then
  add_candidate sr1block_sr2block "3:2"
fi

make_from_base sr1cons_sr2up_sr3up
if replace_if_exists sr1cons_sr2up_sr3up 1 "$NEXT2_DIR/stable_sr1_cons.pt" \
  && replace_if_exists sr1cons_sr2up_sr3up 2 "$NEXT2_DIR/stable_sr2_up.pt" \
  && replace_if_exists sr1cons_sr2up_sr3up 3 "$NEXT2_DIR/stable_sr3_up.pt"; then
  add_candidate sr1cons_sr2up_sr3up ""
fi

make_from_base sr1aggr_sr2up_sr3up
if replace_if_exists sr1aggr_sr2up_sr3up 1 "$NEXT2_DIR/stable_sr1_aggr.pt" \
  && replace_if_exists sr1aggr_sr2up_sr3up 2 "$NEXT2_DIR/stable_sr2_up.pt" \
  && replace_if_exists sr1aggr_sr2up_sr3up 3 "$NEXT2_DIR/stable_sr3_up.pt"; then
  add_candidate sr1aggr_sr2up_sr3up ""
fi

make_from_base sr1block_sr2block_sr3up
if replace_if_exists sr1block_sr2block_sr3up 1 "$PROBE_DIR/stable_sr1_blockfirst.pt" \
  && replace_if_exists sr1block_sr2block_sr3up 2 "$PROBE_DIR/stable_sr2_blockfirst.pt" \
  && replace_if_exists sr1block_sr2block_sr3up 3 "$NEXT2_DIR/stable_sr3_up.pt"; then
  add_candidate sr1block_sr2block_sr3up ""
fi

echo "[INFO] candidates:"
printf '  %s\n' "${CANDIDATES[@]}"

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "[ERROR] no valid candidates"
  exit 1
fi

BEST_NAME=""
BEST_MIRROR=""
BEST_RATE="-1"

eval_candidate() {
  local name="$1"
  local mirror="$2"
  local out="logs/lyk/select_eval/${name}.log"
  echo
  echo "===== Eval candidate: $name mirror='$mirror' ====="
  if ! run_cuda "$GPU" python scripts/eval_stable_moe6.py \
    --expert-dir "$SELECT_DIR/$name" \
    --prefix stable_sr \
    --mirror-map "$mirror" \
    --num-envs "$NUM_ENVS" \
    --batches "$BATCHES" \
    --steps 149 \
    --latch-hi 5.0 \
    --z-low 0.85 \
    --z-up 1.35 \
    --device cuda:0 > "$out" 2>&1; then
    cat "$out"
    echo "[WARN] eval failed for $name; skipping"
    return 0
  fi
  cat "$out"

  local rate
  rate="$(awk '/MoE6 block:/ {gsub("%", "", $NF); print $NF}' "$out" | tail -1)"
  if [[ -z "$rate" ]]; then
    echo "[WARN] could not parse block rate for $name"
    return 0
  fi
  echo "[RESULT] $name block=${rate}%"
  if awk -v a="$rate" -v b="$BEST_RATE" 'BEGIN { exit !(a > b) }'; then
    BEST_NAME="$name"
    BEST_MIRROR="$mirror"
    BEST_RATE="$rate"
  fi
}

for item in "${CANDIDATES[@]}"; do
  eval_candidate "${item%%|*}" "${item#*|}"
done

echo
echo "===== Candidate summary ====="
grep -hE "MoE6 block|Right-|Left-|Gate accuracy" logs/lyk/select_eval/*.log || true
echo "[BEST_CANDIDATE] name=$BEST_NAME mirror='$BEST_MIRROR' block=${BEST_RATE}%"

if [[ -z "$BEST_NAME" ]]; then
  echo "[ERROR] no candidate produced a parseable result"
  exit 1
fi

SWEEP_LOG="logs/lyk/gate_logs/${BEST_NAME}_select_sweep.log"
SWEEP_CSV="logs/lyk/gate_sweep_${BEST_NAME}_select.csv"

echo
echo "===== Gate sweep best candidate: $BEST_NAME ====="
run_cuda "$GPU" python scripts/sweep_moe6_gate.py \
  --expert-dir "$SELECT_DIR/$BEST_NAME" \
  --prefix stable_sr \
  --mirror-map "$BEST_MIRROR" \
  --out-csv "$SWEEP_CSV" \
  --num-envs "$NUM_ENVS" \
  --batches "$BATCHES" \
  --z-lows 0.60 0.70 0.80 0.85 \
  --z-ups 1.25 1.35 1.45 \
  --latch-his 4.0 5.0 6.0 \
  --device cuda:0 > "$SWEEP_LOG" 2>&1
cat "$SWEEP_LOG"

BEST_LINE="$(grep '\[BEST\]' "$SWEEP_LOG" | tail -1)"
if [[ -z "$BEST_LINE" ]]; then
  echo "[ERROR] gate sweep did not print a [BEST] line"
  exit 1
fi

Z_LOW="$(sed -n 's/.*z_low=\([^ ]*\).*/\1/p' <<<"$BEST_LINE")"
Z_UP="$(sed -n 's/.*z_up=\([^ ]*\).*/\1/p' <<<"$BEST_LINE")"
LATCH_HI="$(sed -n 's/.*latch_hi=\([^ ]*\).*/\1/p' <<<"$BEST_LINE")"
VZ_LOW="$(sed -n 's/.*vz_low=\([^ ]*\).*/\1/p' <<<"$BEST_LINE")"

echo
echo "===== Bundle final checkpoint ====="
echo "[FINAL_PARAMS] candidate=$BEST_NAME mirror='$BEST_MIRROR' z_low=$Z_LOW z_up=$Z_UP latch_hi=$LATCH_HI vz_low=$VZ_LOW"
python scripts/bundle_moe6.py \
  --expert-dir "$SELECT_DIR/$BEST_NAME" \
  --prefix stable_sr \
  --out "$FINAL_CKPT" \
  --mirror-map "$BEST_MIRROR" \
  --z-low "$Z_LOW" \
  --z-up "$Z_UP" \
  --latch-hi "$LATCH_HI" \
  --vz-low "$VZ_LOW"

echo
echo "===== Official eval: $FINAL_CKPT ====="
run_cuda "$GPU" python scripts/eval_naive_goalkeeper.py \
  --headless \
  --num-trials "$OFFICIAL_TRIALS" \
  --checkpoint "$FINAL_CKPT" \
  --device cuda:0

echo
echo "===== MoE6 selection finished: $(date) ====="
echo "[INFO] final_ckpt=$FINAL_CKPT"
echo "[INFO] log=$LOG"
