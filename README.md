# hybrid_model_benchmark

Benchmark GPU behavior of `nvidia/Nemotron-H-8B-Base-8K` with Nsight Systems.
The benchmark adds NVTX ranges for:
- module boundaries (`Mamba`, `MLP`, `Attention`)
- inference call scope (`Inference/batch_N`)

This makes it possible to inspect layer-level timing and GPU activity by batch size while
keeping Nsight reports small by default.

**IMPORTANT: This benchmark only profiles the last decode step.**

## Model Overview

Nemotron-H is a hybrid stack with 52 layers and pattern:

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

Inside the container:

```bash
uv sync
```

## Getting Started - Run the Benchmark
1. Install dependencies:

```bash
uv sync
```

2. Run the benchmark with Nsight Systems:

```bash
nsys profile \
--trace=cuda,nvtx,osrt,cublas \
-o nsys-reps/benchmark_report \
-f true \
--capture-range=cudaProfilerApi \
--capture-range-end=stop \
-e BENCHMARK_BATCH_SIZE=4,BENCHMARK_MAX_SEQ_LENGTH=2048,BENCHMARK_WARMUP_ITERATIONS=1,BENCHMARK_PROMPT_LENGTH=2046 \
uv run run_bench.py
```

## Run Full Benchmark Sweep

Use the helper script to run all benchmark combinations in one shot.

```bash
chmod +x scripts/run_all_bench.sh
./scripts/run_all_bench.sh
```

The script runs the following search space:

- `batch_size`: `1, 2, 4, 8`
- `prompt_length`: `1, 16, 256, 1024, 2048, 4096, 8000`
- `warmup_iterations`: always `2`
- `max_seq_length`: always `prompt_length + 2`

Total runs: `4 x 7 = 28`

For each run, it executes `uv run run_bench.py` with:

- `BENCHMARK_BATCH_SIZE`
- `BENCHMARK_PROMPT_LENGTH`
- `BENCHMARK_WARMUP_ITERATIONS=2`
- `BENCHMARK_MAX_SEQ_LENGTH=prompt_length+2`

Each Nsight report is written to `nsys-reps/` with this naming pattern:

```text
benchmark_bs{batch_size}_pl{prompt_length}_ws2.nsys-rep
```

Failure behavior:

- If one run fails, the script continues with the remaining runs.
- At the end, it prints a failed-run summary.
- Exit code is `1` when there is any failure, otherwise `0`.

## NVTX Hierarchy

During profiled execution, ranges are nested as:

```text
llm_generation
  layer=i_module=Mamba
  layer=i_module=MLP
  layer=i_module=Attention
```

This structure helps isolate:
- per-module behavior in the model forward path
- batch-size effects across runs

## Benchmark Result Analysis
- [Result analysis in 3090 GPU](./docs/3090_RESULT_ANALYSIS.md)