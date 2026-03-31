#!/usr/bin/env bash
set -euo pipefail

# readonly BATCH_SIZES=(1 2 4 8 16 32)
readonly BATCH_SIZES=(1 2)
readonly MAX_TOKENS="${MAX_TOKENS:-10}"
readonly WARMUP_RUNS="${WARMUP_RUNS:-1}"
readonly METRICS_FREQ="${METRICS_FREQ:-10000}"
readonly TRACE_TYPES="${TRACE_TYPES:-cuda,nvtx,osrt,cublas}"
readonly OUTPUT_PREFIX="${OUTPUT_PREFIX:-nsys-reps/nemotron_h_batch}"
readonly PROFILE_START_STEP="${PROFILE_START_STEP:-0}"
readonly PROFILE_END_STEP="${PROFILE_END_STEP:-10}"

mkdir -p logs nsys-reps

for BS in "${BATCH_SIZES[@]}"; do
  echo "=== Profiling batch_size=${BS} (max_tokens=${MAX_TOKENS}, warmup_runs=${WARMUP_RUNS}) ==="
  nsys_args=(
    profile
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --cuda-graph-trace=node
    --trace-fork-before-exec=true
    -t "${TRACE_TYPES}"
    -o "${OUTPUT_PREFIX}_${BS}"
    -f true
  )

  bench_args=(
    --batch-size "${BS}"
    --max-tokens "${MAX_TOKENS}"
    --warmup-runs "${WARMUP_RUNS}"
  )
  if [[ -n "${PROFILE_START_STEP}" ]]; then
    bench_args+=(--profile-start-step "${PROFILE_START_STEP}")
  fi
  if [[ -n "${PROFILE_END_STEP}" ]]; then
    bench_args+=(--profile-end-step "${PROFILE_END_STEP}")
  fi

  timeout 7200s nsys "${nsys_args[@]}" \
    uv run python bench.py "${bench_args[@]}"
done
