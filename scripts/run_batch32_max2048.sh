#!/bin/bash

nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
-t cuda,nvtx,osrt,cublas -o nsys-reps/nemotron_h_batch32_max2048 -f true \
uv run python bench.py \
--batch-size 32 --max-tokens 2048 --warmup-runs 3
