#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/nsys-reps"
LOG_DIR="${PROJECT_ROOT}/logs"

timestamp="$(date +"%Y%m%d_%H%M%S")"
PROFILE_MODE="gm_none"
GPU_METRICS_FLAG=()
gpu_metrics_help_status=1
gpu_metrics_help_output=""

BATCH_SIZES=(1 2)
PROMPT_LENGTHS=(1 16 256 1024 2048 4096 8192 16384 32768 65536)
WARMUP_ITERATIONS=2

TOTAL_RUNS=$((${#BATCH_SIZES[@]} * ${#PROMPT_LENGTHS[@]}))
CURRENT_RUN=0

FAILED_RUNS=()

if ! command -v nsys >/dev/null 2>&1; then
	echo "ERROR: nsys command not found. Please install Nsight Systems first." >&2
	exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
	echo "ERROR: uv command not found. Please install uv first." >&2
	exit 1
fi

set +e
gpu_metrics_help_output="$(nsys profile --gpu-metrics-devices=help 2>&1)"
gpu_metrics_help_status=$?
set -e

if [[ ${gpu_metrics_help_status} -eq 0 ]] \
	&& grep -qi "gpu-metrics-devices" <<< "${gpu_metrics_help_output}" \
	&& grep -qiE '(^|[[:space:]])all([[:space:]:]|$)' <<< "${gpu_metrics_help_output}"; then
	PROFILE_MODE="gm_all"
	GPU_METRICS_FLAG=(--gpu-metrics-devices=all)
fi

LOG_FILE="${LOG_DIR}/run_all_bench_${PROFILE_MODE}_${timestamp}.log"
RUN_ID="$(basename "${LOG_FILE}" .log)"
RUN_REPORT_DIR="${REPORT_DIR}/${RUN_ID}"

mkdir -p "${REPORT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${RUN_REPORT_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to: ${LOG_FILE}"
echo "GPU metrics mode: ${PROFILE_MODE}"
if [[ "${PROFILE_MODE}" == "gm_all" ]]; then
	echo "Using extra nsys flag: ${GPU_METRICS_FLAG[*]}"
else
	if [[ ${gpu_metrics_help_status} -ne 0 ]]; then
		echo "WARN: Failed to query GPU metrics support (status=${gpu_metrics_help_status}). Running without --gpu-metrics-devices=all."
	else
		echo "GPU metrics option '--gpu-metrics-devices=all' not supported by this nsys environment."
	fi
fi

sleep 5 # Give user a moment to read the above info before starting the runs

echo "Starting benchmark sweep: ${TOTAL_RUNS} runs"

for batch_size in "${BATCH_SIZES[@]}"; do
	for prompt_length in "${PROMPT_LENGTHS[@]}"; do
		CURRENT_RUN=$((CURRENT_RUN + 1))
		max_seq_length=$((prompt_length + 2))

		report_prefix="${RUN_REPORT_DIR}/benchmark_bs${batch_size}_pl${prompt_length}_ws${WARMUP_ITERATIONS}_${PROFILE_MODE}"

		env_vars="BENCHMARK_BATCH_SIZE=${batch_size},BENCHMARK_MAX_SEQ_LENGTH=${max_seq_length},BENCHMARK_WARMUP_ITERATIONS=${WARMUP_ITERATIONS},BENCHMARK_PROMPT_LENGTH=${prompt_length}"

		echo "[${CURRENT_RUN}/${TOTAL_RUNS}] Running batch_size=${batch_size}, prompt_length=${prompt_length}, max_seq_length=${max_seq_length}, mode=${PROFILE_MODE}"

		nsys_cmd=(
			nsys profile
			--trace=cuda,nvtx,osrt,cublas
			-o "${report_prefix}"
			-f true
			--capture-range=cudaProfilerApi
			--capture-range-end=stop
		)

		if [[ "${PROFILE_MODE}" == "gm_all" ]]; then
			nsys_cmd+=("${GPU_METRICS_FLAG[@]}")
		fi

		nsys_cmd+=(
			-e "${env_vars}"
			uv run run_bench.py
		)

		if ! "${nsys_cmd[@]}"; then
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
