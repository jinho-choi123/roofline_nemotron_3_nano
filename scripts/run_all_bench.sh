#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/nsys-reps"
LOG_DIR="${PROJECT_ROOT}/logs"

timestamp="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/run_all_bench_${timestamp}.log"

BATCH_SIZES=(1 2 4 8)
PROMPT_LENGTHS=(1 16 256 1024 2048 4096 8000)
WARMUP_ITERATIONS=2

TOTAL_RUNS=$((${#BATCH_SIZES[@]} * ${#PROMPT_LENGTHS[@]}))
CURRENT_RUN=0

FAILED_RUNS=()

mkdir -p "${REPORT_DIR}"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to: ${LOG_FILE}"

if ! command -v nsys >/dev/null 2>&1; then
	echo "ERROR: nsys command not found. Please install Nsight Systems first." >&2
	exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
	echo "ERROR: uv command not found. Please install uv first." >&2
	exit 1
fi

echo "Starting benchmark sweep: ${TOTAL_RUNS} runs"

for batch_size in "${BATCH_SIZES[@]}"; do
	for prompt_length in "${PROMPT_LENGTHS[@]}"; do
		CURRENT_RUN=$((CURRENT_RUN + 1))
		max_seq_length=$((prompt_length + 2))

		report_prefix="${REPORT_DIR}/benchmark_bs${batch_size}_pl${prompt_length}_ws${WARMUP_ITERATIONS}"

		env_vars="BENCHMARK_BATCH_SIZE=${batch_size},BENCHMARK_MAX_SEQ_LENGTH=${max_seq_length},BENCHMARK_WARMUP_ITERATIONS=${WARMUP_ITERATIONS},BENCHMARK_PROMPT_LENGTH=${prompt_length}"

		echo "[${CURRENT_RUN}/${TOTAL_RUNS}] Running batch_size=${batch_size}, prompt_length=${prompt_length}, max_seq_length=${max_seq_length}"

		if ! nsys profile \
			--trace=cuda,nvtx,osrt,cublas \
			-o "${report_prefix}" \
			-f true \
			--capture-range=cudaProfilerApi \
			--capture-range-end=stop \
			-e "${env_vars}" \
			uv run run_bench.py; then
			FAILED_RUNS+=("batch_size=${batch_size},prompt_length=${prompt_length},max_seq_length=${max_seq_length}")
			echo "FAILED: batch_size=${batch_size}, prompt_length=${prompt_length}" >&2
			continue
		fi

		echo "DONE: ${report_prefix}.nsys-rep"
	done
done

echo
echo "Sweep completed."

if ((${#FAILED_RUNS[@]} > 0)); then
	echo "Failed runs: ${#FAILED_RUNS[@]}"
	for failed in "${FAILED_RUNS[@]}"; do
		echo "  - ${failed}"
	done
	exit 1
fi

echo "All runs succeeded: ${TOTAL_RUNS}/${TOTAL_RUNS}"
exit 0
