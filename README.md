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

## Prerequisites

- NVIDIA GPU with CUDA runtime available
- Nsight Systems (`nsys`) installed
- `uv` installed
- Python dependencies installed via project lock

Reference installation guides:
- Nsight Systems: [NVIDIA docs](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html#package-manager-installation)
- uv: [Astral docs](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Quick Start

Run a quick sanity execution (no Nsight collection):

```bash
uv run python bench.py --batch-size 1 --max-tokens 8 --warmup-runs 0
```

Generated outputs:
- Application logs: `logs/benchmark_*.log`

## `bench.py` CLI

```bash
uv run python bench.py [options]
```

Options:
- `--batch-size` (`int`, default: `4`)
- `--max-tokens` (`int`, default: `100`)
- `--warmup-runs` (`int`, default: `1`)

Examples:

```bash
# Profile one run with Nsight Systems.
# Capture starts automatically at the last decode step (step=max_tokens).
mkdir -p nsys-reps
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  -t cuda,nvtx,osrt,cublas -o nsys-reps/nemotron_h_batch4_max512 -f true \
  uv run python bench.py --batch-size 4 --max-tokens 512 --warmup-runs 1
```

Long-context example:

```bash
mkdir -p nsys-reps
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  -t cuda,nvtx,osrt,cublas -o nsys-reps/nemotron_h_batch4_max2048 -f true \
  uv run python bench.py --batch-size 4 --max-tokens 2048 --warmup-runs 1
```

## Profiling Behavior

- `bench.py` enables `cudaProfilerStart()` only at the last decode step.
- Last decode step target is `max(1, max_tokens - 1)`.
  - vLLM emits the first generated token in prefill, then decode runs for remaining tokens.
- Warmup runs do not trigger profiling.
- `cudaProfilerStop()` is called after `llm.generate()` returns.

This usually reduces `.nsys-rep` size significantly compared with full-generation capture.

## NVTX Hierarchy

During profiled execution, ranges are nested as:

```text
Inference/batch_N
  module=Mamba/layer=i
  module=MLP/layer=j
  module=Attention/layer=k
```

This structure helps isolate:
- per-module behavior in the model forward path
- batch-size effects across runs

## Verify Capture Window

```bash
nsys stats --report nvtx_pushpop_sum nsys-reps/nemotron_h_batch4_max2048.nsys-rep
```

Check that:
- `module=Mamba/layer=*`, `module=MLP/layer=*`, `module=Attention/layer=*` appear
- timeline duration is concentrated near the final decode step

