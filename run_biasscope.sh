#!/usr/bin/env bash
# BiasScope driver (repo root). For each analysis dataset × judge model:
#   Stage 1: attack_judge_and_analysis.py
#   Stage 2: synthesis_bias_verification.py
#
#   bash run_biasscope.sh
#
# Override without editing: CUDA_VISIBLE_DEVICES, PYTHON, TEACHER_MODEL_PATH, DRY_RUN=1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- User config ---

# GPU ids visible to this process (e.g. 0 or 0,1).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Interpreter (e.g. conda env python).
PYTHON="${PYTHON:-python3}"

# vLLM: max sequences per generate batch (tune for VRAM).
BATCH_SIZE=64

# vLLM tensor parallel for teacher and judge (--self-defined-tp-size).
TP_SIZE=2

# Seed bias definitions JSON.
BIAS_JSON="${REPO_ROOT}/data/bias/basic_biases.json"

# Stage 1 input(s): preference parquet(s), paths relative to repo root or absolute.
ANALYSIS_DATASETS=(
  "data/rewardbench/rewardbench_filtered.parquet"
)

# Stage 2 verification parquet (test / held-out).
TEST_DATA="data/judgeBench/judge_bench.parquet"

# Teacher model directory (HF id or local path). Can set via env instead.
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/Qwen/Qwen2.5-72B-Instruct}"

# Judge model(s) to evaluate; one full pipeline per entry.
JUDGE_MODELS=(
  "/Qwen/Qwen2.5-7B-Instruct"
)

# Repeat the inner model loop (same data); use 1 for normal runs.
REPEAT=2

# Print commands only, do not run Python.
DRY_RUN="${DRY_RUN:-0}"

# Optional extra flags for both scripts, e.g. API teacher:
# EXTRA_ARGS=( --teacher-backend api --api-key "$OPENAI_API_KEY" --teacher-model gpt-4o --base-url "https://api.openai.com/v1" )
EXTRA_ARGS=()

# --- Run ---

run_py() {
  local script=$1
  shift
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[dry-run] %q %q' "$PYTHON" "$script"
    printf ' %q' "$@"
    echo
    return 0
  fi
  "$PYTHON" "$script" "$@"
}

for analysis_data in "${ANALYSIS_DATASETS[@]}"; do
  echo "=== Dataset: $analysis_data ==="
  for ((r = 1; r <= REPEAT; r++)); do
    [[ "$REPEAT" -gt 1 ]] && echo "=== Repeat $r / $REPEAT ==="
    for model_path in "${JUDGE_MODELS[@]}"; do
      echo "=== Judge: $model_path ==="

      run_py attack_judge_and_analysis.py \
        --bias-json "$BIAS_JSON" \
        --analysis-data-path "$analysis_data" \
        --model-path "$model_path" \
        --teacher-model-path "$TEACHER_MODEL_PATH" \
        --batch-size "$BATCH_SIZE" \
        --self-defined-tp-size "$TP_SIZE" \
        "${EXTRA_ARGS[@]}"

      run_py synthesis_bias_verification.py \
        --bias-json "$BIAS_JSON" \
        --analysis-data-path "$analysis_data" \
        --model-path "$model_path" \
        --teacher-model-path "$TEACHER_MODEL_PATH" \
        --test-data-path "$TEST_DATA" \
        --batch-size "$BATCH_SIZE" \
        --self-defined-tp-size "$TP_SIZE" \
        "${EXTRA_ARGS[@]}"
    done
  done
done

echo "Done."
