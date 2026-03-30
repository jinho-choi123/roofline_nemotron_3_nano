# hybrid_model_benchmark

Benchmark GPU behavior of `nvidia/Nemotron-H-8B-Base-8K` with Nsight Systems.
The benchmark adds NVTX ranges for:
- layer type (`Mamba`, `MLP`, `Attention`)
- phase (`Prefill`, `Decode`)
- inference call scope (`Inference/batch_N`)

This makes it possible to inspect layer-level timing and phase behavior by batch size.

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

Run the batch benchmark script:

```bash
./run_bench.sh
```

Generated outputs:
- Nsight reports: `nsys-reps/*.nsys-rep`
- Application logs: `logs/benchmark_*.log`

## `bench.py` CLI

```bash
uv run python bench.py [options]
```

Options:
- `--batch-size` (`int`, default: `4`)
- `--max-tokens` (`int`, default: `100`)
- `--warmup-runs` (`int`, default: `1`)
- `--profile-start-step` (`int`, default: unset)
- `--profile-end-step` (`int`, default: unset)

### Step-range profiling window

`--profile-start-step` and `--profile-end-step` limit profiling to a decode-step window:
- decode step counting starts at `1` (first decode token)
- `0` for start means "start from prefill"
- end step is exclusive in practice: profiler stop is triggered when decode step reaches `--profile-end-step`

Examples:

```bash
# Profile the entire generate() call
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  -t cuda,nvtx,osrt,cublas -o nsys-reps/full -f true \
  uv run python bench.py --batch-size 4 --max-tokens 512

# Long-context run, profile only decode steps 900..999
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  -t cuda,nvtx,osrt,cublas -o nsys-reps/long_ctx_window -f true \
  uv run python bench.py --batch-size 4 --max-tokens 8000 \
    --profile-start-step 900 --profile-end-step 1000
```

## `run_bench.sh` environment variables

`run_bench.sh` reads configuration from env vars:

- `MAX_TOKENS` (default in script)
- `WARMUP_RUNS` (default in script)
- `PROFILE_START_STEP` (optional)
- `PROFILE_END_STEP` (optional)
- `METRICS_FREQ` (kept for compatibility with optional GPU metrics variants)
- `TRACE_TYPES` (default: `cuda,nvtx,osrt,cublas`)
- `OUTPUT_PREFIX` (default: `nsys-reps/nemotron_h_batch`)

Examples:

```bash
# Basic run
./run_bench.sh

# Long context with small profiling window (smaller .nsys-rep)
MAX_TOKENS=8000 PROFILE_START_STEP=900 PROFILE_END_STEP=1000 ./run_bench.sh
```

## NVTX Hierarchy

During profiled execution, ranges are nested as:

```text
Inference/batch_N[/profile_window_start_end]
  Prefill/num_tokens=T
    Mamba/layer_i
    MLP/layer_j
    Attention/layer_k
  Decode/num_tokens=N
    Mamba/layer_i
    MLP/layer_j
    Attention/layer_k
```

This structure helps isolate:
- prefill cost vs decode cost
- per-layer behavior inside each phase
- batch-size effects across runs

