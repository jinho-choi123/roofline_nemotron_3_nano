#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/ncu-reps"
LOG_DIR="${PROJECT_ROOT}/logs"

timestamp="$(date +"%Y%m%d_%H%M%S")"

BATCH_SIZES=(1)
PROMPT_LENGTHS=(8 16 256 1024 2048 4096 8192 16384 32768 65536 131072)
WARMUP_ITERATIONS=2

NCU_SET="${NCU_SET:-full}"
NCU_REPLAY_MODE="${NCU_REPLAY_MODE:-kernel}"

NCU_PHASE_MODES=("prefill" "decode")
TARGET_LAYER_NVTX_MARKER="layer=17_module=Attention"
TARGET_NAME="attention"

for phase_mode in "${NCU_PHASE_MODES[@]}"; do
	if [[ "${phase_mode}" != "prefill" && "${phase_mode}" != "decode" ]]; then
		echo "ERROR: NCU_PHASE_MODES can only include: prefill, decode" >&2
		exit 1
	fi
done

TOTAL_RUNS_PER_PHASE=$((${#BATCH_SIZES[@]} * ${#PROMPT_LENGTHS[@]}))
TOTAL_RUNS=$((TOTAL_RUNS_PER_PHASE * ${#NCU_PHASE_MODES[@]}))
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

LOG_FILE="${LOG_DIR}/run_all_bench_ncu_${TARGET_NAME}_prefill_decode_${timestamp}_attention.log"
RUN_ID="$(basename "${LOG_FILE}" .log)"
RUN_REPORT_DIR="${REPORT_DIR}/${RUN_ID}"

mkdir -p "${REPORT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${RUN_REPORT_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to: ${LOG_FILE}"
echo "Target marker: ${TARGET_LAYER_NVTX_MARKER}"
echo "NCU set: ${NCU_SET}"
echo "NCU replay mode: ${NCU_REPLAY_MODE}"
echo "NCU phase modes: ${NCU_PHASE_MODES[*]}"
echo "NVTX include ranges by phase:"
for phase_mode in "${NCU_PHASE_MODES[@]}"; do
	echo "  - llm_generation/${phase_mode}_phase/model_forward/${TARGET_LAYER_NVTX_MARKER}"
done
echo "Starting benchmark sweep: ${TOTAL_RUNS} runs (${TOTAL_RUNS_PER_PHASE} per phase)"

sleep 5

for phase_mode in "${NCU_PHASE_MODES[@]}"; do
	echo
	echo "==== Phase sweep start: ${phase_mode} (${TOTAL_RUNS_PER_PHASE} runs) ===="

	for batch_size in "${BATCH_SIZES[@]}"; do
		for prompt_length in "${PROMPT_LENGTHS[@]}"; do
			CURRENT_RUN=$((CURRENT_RUN + 1))
			max_seq_length=$((prompt_length + 2))
			include="llm_generation/${phase_mode}_phase/model_forward/${TARGET_LAYER_NVTX_MARKER}"

			report_prefix="${RUN_REPORT_DIR}/benchmark_${TARGET_NAME}_bs${batch_size}_pl${prompt_length}_ws${WARMUP_ITERATIONS}_${phase_mode}_phase_attention"

			echo "[${CURRENT_RUN}/${TOTAL_RUNS}] Running target=${TARGET_NAME}, batch_size=${batch_size}, prompt_length=${prompt_length}, max_seq_length=${max_seq_length}, phase_mode=${phase_mode}"

			ncu_cmd=(
				ncu
				-o "${report_prefix}"
				-f
				--set "${NCU_SET}"
				--replay-mode "${NCU_REPLAY_MODE}"
				--nvtx
				--nvtx-include "${include}"
				env
				"BENCHMARK_BATCH_SIZE=${batch_size}"
				"BENCHMARK_MAX_SEQ_LENGTH=${max_seq_length}"
				"BENCHMARK_WARMUP_ITERATIONS=${WARMUP_ITERATIONS}"
				"BENCHMARK_PROMPT_LENGTH=${prompt_length}"
				uv run run_bench.py
			)

			if ! "${ncu_cmd[@]}"; then
				FAILED_RUNS+=("phase=${phase_mode},batch_size=${batch_size},prompt_length=${prompt_length},max_seq_length=${max_seq_length}")
				echo "FAILED: phase=${phase_mode}, batch_size=${batch_size}, prompt_length=${prompt_length}" >&2
				continue
			fi

			echo "DONE: ${report_prefix}.ncu-rep"
		done
	done

	echo "==== Phase sweep end: ${phase_mode} ===="
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
