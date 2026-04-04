#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/ncu-reps"
LOG_DIR="${PROJECT_ROOT}/logs"

timestamp="$(date +"%Y%m%d_%H%M%S")"

BATCH_SIZES=(1 2)
PROMPT_LENGTHS=(1 16 256 1024 2048 4096 8192 16384 32768 65536)
WARMUP_ITERATIONS=2

# Default to the top-level range that wraps benchmark generation.
NVTX_INCLUDE="${NVTX_INCLUDE:-llm_generation}"
NCU_SET="${NCU_SET:-full}"
NCU_REPLAY_MODE="${NCU_REPLAY_MODE:-kernel}"

TOTAL_RUNS=$((${#BATCH_SIZES[@]} * ${#PROMPT_LENGTHS[@]}))
CURRENT_RUN=0

FAILED_RUNS=()

if ! command -v ncu >/dev/null 2>&1; then
	echo "ERROR: ncu command not found. Please install Nsight Compute first." >&2
	exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
	echo "ERROR: uv command not found. Please install uv first." >&2
	exit 1
fi

nvtx_tag="$(echo "${NVTX_INCLUDE}" | tr -c '[:alnum:]' '_' | sed 's/_\+/_/g; s/^_//; s/_$//')"
if [[ -z "${nvtx_tag}" ]]; then
	nvtx_tag="nvtx"
fi

LOG_FILE="${LOG_DIR}/run_all_bench_ncu_${nvtx_tag}_${timestamp}.log"
RUN_ID="$(basename "${LOG_FILE}" .log)"
RUN_REPORT_DIR="${REPORT_DIR}/${RUN_ID}"

mkdir -p "${REPORT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${RUN_REPORT_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to: ${LOG_FILE}"
echo "NCU set: ${NCU_SET}"
echo "NCU replay mode: ${NCU_REPLAY_MODE}"
echo "NVTX include range: ${NVTX_INCLUDE}"
echo "Starting benchmark sweep: ${TOTAL_RUNS} runs"

sleep 5

for batch_size in "${BATCH_SIZES[@]}"; do
	for prompt_length in "${PROMPT_LENGTHS[@]}"; do
		CURRENT_RUN=$((CURRENT_RUN + 1))
		max_seq_length=$((prompt_length + 2))

		report_prefix="${RUN_REPORT_DIR}/benchmark_bs${batch_size}_pl${prompt_length}_ws${WARMUP_ITERATIONS}_${nvtx_tag}"

		echo "[${CURRENT_RUN}/${TOTAL_RUNS}] Running batch_size=${batch_size}, prompt_length=${prompt_length}, max_seq_length=${max_seq_length}, nvtx_include=${NVTX_INCLUDE}"

		ncu_cmd=(
			ncu
			-o "${report_prefix}"
			-f
			--set "${NCU_SET}"
			--replay-mode "${NCU_REPLAY_MODE}"
			--nvtx 
			--nvtx-include "llm_generation/"
			env
			"BENCHMARK_BATCH_SIZE=${batch_size}"
			"BENCHMARK_MAX_SEQ_LENGTH=${max_seq_length}"
			"BENCHMARK_WARMUP_ITERATIONS=${WARMUP_ITERATIONS}"
			"BENCHMARK_PROMPT_LENGTH=${prompt_length}"
			uv run run_bench.py
		)

		if ! "${ncu_cmd[@]}"; then
			FAILED_RUNS+=("batch_size=${batch_size},prompt_length=${prompt_length},max_seq_length=${max_seq_length}")
			echo "FAILED: batch_size=${batch_size}, prompt_length=${prompt_length}" >&2
			continue
		fi

		echo "DONE: ${report_prefix}.ncu-rep"
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
