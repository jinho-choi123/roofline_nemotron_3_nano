# hybrid_model_benchmark

Benchmark GPU behavior of `nvidia/NVIDIA-Nemotron-Nano-9B-v2` with Nsight Systems.
The benchmark adds NVTX ranges for:
- module boundaries (`Mamba`, `MLP`, `Attention`)
- inference call scope (`Inference/batch_N`)

This makes it possible to inspect layer-level timing and GPU activity by batch size while
keeping Nsight reports small by default.

**IMPORTANT: This benchmark only profiles the last decode step.**

## Model Overview

Nemotron-Nano-9B-v2 is a hybrid stack with 52 layers and pattern:

```text
M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-
```

- `M`: Mamba (SSM) decoder layer
- `-`: MLP/FFN decoder layer
- `*`: Attention decoder layer

## Setup GPU Container(Optional)

It's recommended to use a GPU container for consistent environment and easy Nsight Systems use. 

**IMPORTANT: The following steps assume you have sudo privileges in Bare Metal GPU Server.**

1. Build image:

```bash
docker build -t hybrid-model-benchmark:cuda12.4 .docker/
```

2. Run container with a single selected GPU, `SYS_ADMIN` capability, and project mount to `/workspace` inside the container:

```bash
scripts/generate_gpu_container.sh 0
```

- The argument (`0`) is the GPU index to expose to the container.
- The project directory is mounted to `/workspace`.
- The run command includes `--cap-add=SYS_ADMIN` for GPU performance counter/profiling use cases.

## Getting Started - Run Full Benchmark Sweep
1. Install dependencies:

```bash
uv sync
```

2. Install nsys and ncu cli tool:
```bash
./scripts/install_nsight_systems.sh
./scripts/install_nsight_compute.sh
```

3. Run all the benchmarks with Nsight Systems and Nsight Compute:

```bash
./scripts/run_all_bench_nsys.sh ; ./scripts/run_all_bench_ncu.sh ; ./scripts/zip_bench_results.sh
```

The script runs the following search space:

- `batch_size`: `1` # Only single batch size
- `prompt_length`: `4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072`
- `warmup_iterations`: always `2`
- `max_seq_length`: always `prompt_length + 2`

Total runs: `1 x 16 = 16`

## Run Single Benchmark Configuration
If you want to run a single benchmark configuration(instead of running a full sweep), set the environment variables and run the script directly:
**Change <benchmark_report_name> with the desired report name**

### Nsight Systems profiling
```bash
# profile with GPU metrics (if supported)
nsys profile \
--trace=cuda,nvtx,osrt,cublas \
-o nsys-reps/<benchmark_report_name> \
--gpu-metrics-devices=all \
-f true \
--capture-range=cudaProfilerApi \
--capture-range-end=stop \
-e BENCHMARK_BATCH_SIZE=4,BENCHMARK_MAX_SEQ_LENGTH=2048,BENCHMARK_WARMUP_ITERATIONS=1,BENCHMARK_PROMPT_LENGTH=2046 \
uv run run_bench.py

# profile without GPU metrics
nsys profile \
--trace=cuda,nvtx,osrt,cublas \
-o nsys-reps/<benchmark_report_name> \
-f true \
--capture-range=cudaProfilerApi \
--capture-range-end=stop \
-e BENCHMARK_BATCH_SIZE=4,BENCHMARK_MAX_SEQ_LENGTH=2048,BENCHMARK_WARMUP_ITERATIONS=1,BENCHMARK_PROMPT_LENGTH=2046 \
uv run run_bench.py
```

### Nsight Compute profiling

```bash
# 1) Attention prefill (first matched prefill range only)
ncu \
-o ncu-reps/<benchmark_report_name>_attention_prefill \
-f \
--set full \
--replay-mode kernel \
--nvtx \
--nvtx-include "llm_generation/prefill_phase/model_forward/layer=17_module=Attention" \
--range-filter ":1:1" \
env BENCHMARK_BATCH_SIZE=1 BENCHMARK_MAX_SEQ_LENGTH=2048 BENCHMARK_WARMUP_ITERATIONS=2 BENCHMARK_PROMPT_LENGTH=2046 \
uv run run_bench.py

# 2) Attention decode
ncu \
-o ncu-reps/<benchmark_report_name>_attention_decode \
-f \
--set full \
--replay-mode kernel \
--nvtx \
--nvtx-include "llm_generation/decode_phase/model_forward/layer=17_module=Attention" \
env BENCHMARK_BATCH_SIZE=1 BENCHMARK_MAX_SEQ_LENGTH=2048 BENCHMARK_WARMUP_ITERATIONS=2 BENCHMARK_PROMPT_LENGTH=2046 \
uv run run_bench.py

# 3) MLP prefill (first matched prefill range only)
ncu \
-o ncu-reps/<benchmark_report_name>_mlp_prefill \
-f \
--set full \
--replay-mode kernel \
--nvtx \
--nvtx-include "llm_generation/prefill_phase/model_forward/layer=18_module=MLP" \
--range-filter ":1:1" \
env BENCHMARK_BATCH_SIZE=1 BENCHMARK_MAX_SEQ_LENGTH=2048 BENCHMARK_WARMUP_ITERATIONS=2 BENCHMARK_PROMPT_LENGTH=2046 \
uv run run_bench.py

# 4) MLP decode
ncu \
-o ncu-reps/<benchmark_report_name>_mlp_decode \
-f \
--set full \
--replay-mode kernel \
--nvtx \
--nvtx-include "llm_generation/decode_phase/model_forward/layer=18_module=MLP" \
env BENCHMARK_BATCH_SIZE=1 BENCHMARK_MAX_SEQ_LENGTH=2048 BENCHMARK_WARMUP_ITERATIONS=2 BENCHMARK_PROMPT_LENGTH=2046 \
uv run run_bench.py

# 5) Mamba prefill (first matched prefill range only)
ncu \
-o ncu-reps/<benchmark_report_name>_mamba_prefill \
-f \
--set full \
--replay-mode kernel \
--nvtx \
--nvtx-include "llm_generation/prefill_phase/model_forward/layer=16_module=Mamba" \
--range-filter ":1:1" \
env BENCHMARK_BATCH_SIZE=1 BENCHMARK_MAX_SEQ_LENGTH=2048 BENCHMARK_WARMUP_ITERATIONS=2 BENCHMARK_PROMPT_LENGTH=2046 \
uv run run_bench.py

# 6) Mamba decode
ncu \
-o ncu-reps/<benchmark_report_name>_mamba_decode \
-f \
--set full \
--replay-mode kernel \
--nvtx \
--nvtx-include "llm_generation/decode_phase/model_forward/layer=16_module=Mamba" \
env BENCHMARK_BATCH_SIZE=1 BENCHMARK_MAX_SEQ_LENGTH=2048 BENCHMARK_WARMUP_ITERATIONS=2 BENCHMARK_PROMPT_LENGTH=2046 \
uv run run_bench.py
```

- For `prefill`, `--range-filter ":1:1"` keeps only the first matched prefill NVTX range.
- For `decode`, do not pass `--range-filter`.
- Keep `BENCHMARK_MAX_SEQ_LENGTH = BENCHMARK_PROMPT_LENGTH + 2`.


## NVTX Hierarchy

During profiled execution, ranges are nested as:

```text
llm_generation
  prefill_phase | decode_phase
    model_forward
      layer=i_module=Mamba
      layer=i_module=MLP
      layer=i_module=Attention
```

This structure helps isolate:
- per-module behavior in the model forward path
- phase-specific behavior (`prefill` vs `decode`)

Notes:
- In long-context prefill, the same `layer=i_module=*` marker can appear multiple times.
- Nsight Compute filtering in this repo targets: `llm_generation/<phase>_phase/model_forward/layer=...`.

## Benchmark Result Analysis
- [Result analysis in 3090 GPU](./docs/3090_RESULT_ANALYSIS.md)