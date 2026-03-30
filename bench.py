"""Benchmark Nemotron-H layer-level profiling with NVTX ranges."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

import torch
import torch.cuda.nvtx as nvtx
from loguru import logger
from vllm import LLM, SamplingParams

MODEL_NAME = "nvidia/Nemotron-H-8B-Base-8K"
LOG_DIR = Path("logs")
LAYER_TYPE_NAMES = {"M": "Mamba", "-": "MLP", "*": "Attention", "E": "MoE"}


def parse_args() -> argparse.Namespace:
    """Parse benchmark CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Nemotron-H Nsight Systems benchmark")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of prompts per generation call. Defaults to 4.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum generated tokens per prompt. Defaults to 100.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup generate calls before profiling. Defaults to 1.",
    )
    parser.add_argument(
        "--profile-start-step",
        type=int,
        default=None,
        help=(
            "Decode step where profiling starts (1-indexed decode steps; 0 means prefill). "
            "Defaults to None."
        ),
    )
    parser.add_argument(
        "--profile-end-step",
        type=int,
        default=None,
        help=(
            "Decode step where profiling stops (exclusive). "
            "Defaults to None."
        ),
    )
    return parser.parse_args()


def build_prompts(batch_size: int) -> list[str]:
    """Build synthetic prompts for a target batch size.

    Args:
        batch_size (int): Number of prompts to generate. Defaults to caller-provided value.

    Returns:
        list[str]: Prompt strings with deterministic content.

    Raises:
        ValueError: If `batch_size` is lower than 1.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [
        (
            f"Count fron 1 to 100000: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 "
        )
        for index in range(batch_size)
    ]


def _patch_nemotron_layers_on_worker(module: Any) -> dict[str, Any]:
    """Patch Nemotron-H layers with NVTX ranges on worker process.

    Args:
        module (Any): Worker-side root model module. Defaults to no default value.

    Returns:
        dict[str, Any]: Worker patch metadata including path, pattern and counts.

    Raises:
        RuntimeError: If Nemotron-H layers cannot be found.
        ValueError: If layer pattern and layer count do not match.
    """
    nemotron_model = module.model if hasattr(module, "model") else module
    if not (hasattr(nemotron_model, "layers") and hasattr(nemotron_model, "config")):
        raise RuntimeError("Worker module does not expose Nemotron layers/config.")

    pattern: str = nemotron_model.config.hybrid_override_pattern
    layers = nemotron_model.layers
    if len(pattern) < len(layers):
        raise ValueError("hybrid_override_pattern is shorter than available layers.")

    import torch.cuda.nvtx as worker_nvtx

    try:
        phase_batch_size = max(1, int(os.environ.get("BENCH_BATCH_SIZE", "1")))
    except ValueError:
        phase_batch_size = 1
    try:
        profile_start_step = int(os.environ["BENCH_PROFILE_START_STEP"])
    except (KeyError, ValueError):
        profile_start_step = None
    try:
        profile_end_step = int(os.environ["BENCH_PROFILE_END_STEP"])
    except (KeyError, ValueError):
        profile_end_step = None

    profile_range_enabled = profile_start_step is not None or profile_end_step is not None
    effective_start_step = profile_start_step
    if profile_range_enabled and effective_start_step is None:
        effective_start_step = 0
    decode_state: dict[str, Any] = {
        "decode_step": 0,
        "profiler_started": False,
        "profiler_stopped": False,
    }

    patched_counts = {"Mamba": 0, "MLP": 0, "Attention": 0, "MoE": 0, "Unknown": 0}
    for index, layer in enumerate(layers):
        if not hasattr(layer, "forward"):
            continue
        layer_token = pattern[index]
        layer_type = LAYER_TYPE_NAMES.get(layer_token, "Unknown")
        range_name = f"{layer_type}/layer_{index}"
        original_forward = layer.forward

        if getattr(original_forward, "_nvtx_wrapped", False):
            continue

        @wraps(original_forward)
        def wrapped_forward(
            *args: Any,
            __orig_forward: Callable[..., Any] = original_forward,
            __range_name: str = range_name,
            **kwargs: Any,
        ) -> Any:
            worker_nvtx.range_push(__range_name)
            try:
                return __orig_forward(*args, **kwargs)
            finally:
                worker_nvtx.range_pop()

        setattr(wrapped_forward, "_nvtx_wrapped", True)
        layer.forward = wrapped_forward
        patched_counts[layer_type] = patched_counts.get(layer_type, 0) + 1

    original_model_forward = nemotron_model.forward
    if not getattr(original_model_forward, "_nvtx_phase_wrapped", False):

        @wraps(original_model_forward)
        def model_forward_with_phase(*args: Any, **kwargs: Any) -> Any:
            positions = kwargs.get("positions")
            if positions is None and len(args) > 1:
                positions = args[1]
            if positions is not None and hasattr(positions, "shape") and len(positions.shape) > 0:
                num_tokens = int(positions.shape[0])
            else:
                num_tokens = 0
            phase = "Decode" if num_tokens <= phase_batch_size else "Prefill"

            if (
                profile_range_enabled
                and effective_start_step == 0
                and not decode_state["profiler_started"]
            ):
                status = torch.cuda.cudart().cudaProfilerStart()
                if status != 0:
                    raise RuntimeError(
                        f"cudaProfilerStart failed with status={status} at prefill"
                    )
                decode_state["profiler_started"] = True

            if phase == "Decode":
                decode_state["decode_step"] += 1
                decode_step = decode_state["decode_step"]

                if (
                    profile_range_enabled
                    and effective_start_step is not None
                    and effective_start_step > 0
                    and not decode_state["profiler_started"]
                    and decode_step == effective_start_step
                ):
                    status = torch.cuda.cudart().cudaProfilerStart()
                    if status != 0:
                        raise RuntimeError(
                            f"cudaProfilerStart failed with status={status} at decode_step={decode_step}"
                        )
                    decode_state["profiler_started"] = True

                if (
                    profile_range_enabled
                    and profile_end_step is not None
                    and decode_state["profiler_started"]
                    and not decode_state["profiler_stopped"]
                    and decode_step == profile_end_step
                ):
                    status = torch.cuda.cudart().cudaProfilerStop()
                    if status != 0:
                        raise RuntimeError(
                            f"cudaProfilerStop failed with status={status} at decode_step={decode_step}"
                        )
                    decode_state["profiler_stopped"] = True

            worker_nvtx.range_push(f"{phase}/num_tokens={num_tokens}")
            try:
                return original_model_forward(*args, **kwargs)
            finally:
                worker_nvtx.range_pop()

        setattr(model_forward_with_phase, "_nvtx_phase_wrapped", True)
        nemotron_model.forward = model_forward_with_phase

    return {
        "access_path": "module.model.layers",
        "module_type": type(module).__name__,
        "nemotron_model_type": type(nemotron_model).__name__,
        "pattern": pattern,
        "layer_count": len(layers),
        "patched_counts": patched_counts,
        "phase_batch_size": phase_batch_size,
        "model_forward_patched": True,
        "effective_start_step": effective_start_step,
        "profile_start_step": profile_start_step,
        "profile_end_step": profile_end_step,
        "profile_range_enabled": profile_range_enabled,
    }


def inject_nvtx_annotations(llm: LLM) -> dict[str, int]:
    """Inject per-layer NVTX ranges into Nemotron-H decoder layers.

    Args:
        llm (LLM): Initialized vLLM LLM instance. Defaults to no default value.

    Returns:
        dict[str, int]: Count of patched layers grouped by logical layer type.

    Raises:
        RuntimeError: If model internals cannot be traversed.
        ValueError: If layer pattern and layer list lengths mismatch.
    """
    worker_results: list[dict[str, Any]] = llm.llm_engine.apply_model(
        _patch_nemotron_layers_on_worker
    )
    if not worker_results:
        raise RuntimeError("No worker result returned from NVTX patch injection.")

    merged_counts = {"Mamba": 0, "MLP": 0, "Attention": 0, "MoE": 0, "Unknown": 0}
    for worker_idx, result in enumerate(worker_results):
        logger.info(
            (
                "Worker {} NVTX patch metadata: path={}, module_type={}, "
                "nemotron_model_type={}, layer_count={}, model_forward_patched={}, "
                "phase_batch_size={}, profile_range_enabled={}, effective_start_step={}, "
                "profile_start_step={}, profile_end_step={}"
            ),
            worker_idx,
            result["access_path"],
            result["module_type"],
            result["nemotron_model_type"],
            result["layer_count"],
            result["model_forward_patched"],
            result["phase_batch_size"],
            result["profile_range_enabled"],
            result["effective_start_step"],
            result["profile_start_step"],
            result["profile_end_step"],
        )
        logger.info("Worker {} hybrid pattern: {}", worker_idx, result["pattern"])
        worker_counts = result["patched_counts"]
        for key, value in worker_counts.items():
            merged_counts[key] = merged_counts.get(key, 0) + int(value)
    return merged_counts


def _cuda_profiler_start() -> None:
    """Start CUDA profiler range capture.

    Returns:
        None: No return value.

    Raises:
        RuntimeError: If the CUDA profiler start call fails.
    """
    status = torch.cuda.cudart().cudaProfilerStart()
    if status != 0:
        raise RuntimeError(f"cudaProfilerStart failed with status={status}")


def _cuda_profiler_stop() -> None:
    """Stop CUDA profiler range capture.

    Returns:
        None: No return value.

    Raises:
        RuntimeError: If the CUDA profiler stop call fails.
    """
    status = torch.cuda.cudart().cudaProfilerStop()
    if status != 0:
        raise RuntimeError(f"cudaProfilerStop failed with status={status}")


def bench(
    batch_size: int,
    max_tokens: int,
    warmup_runs: int,
    profile_start_step: int | None,
    profile_end_step: int | None,
) -> Path:
    """Run Nemotron-H benchmark with NVTX + cudaProfilerApi control.

    Args:
        batch_size (int): Number of prompts per generation call. Defaults to 4.
        max_tokens (int): Maximum generated tokens per prompt. Defaults to 100.
        warmup_runs (int): Number of warmup generation calls. Defaults to 1.
        profile_start_step (int | None): Decode step at which profiling starts.
            Defaults to None.
        profile_end_step (int | None): Decode step at which profiling stops
            (exclusive). Defaults to None.

    Returns:
        Path: Output log file path.

    Raises:
        ValueError: If one or more numeric arguments are invalid.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")
    if profile_start_step is not None and profile_start_step < 0:
        raise ValueError("profile_start_step must be >= 0")
    if profile_end_step is not None and profile_end_step < 0:
        raise ValueError("profile_end_step must be >= 0")
    if (
        profile_start_step is not None
        and profile_end_step is not None
        and profile_end_step <= profile_start_step
    ):
        raise ValueError("profile_end_step must be greater than profile_start_step")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_file)
    logger.info("Starting benchmark for batch_size={}, max_tokens={}", batch_size, max_tokens)

    prompts = build_prompts(batch_size=batch_size)

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens, ignore_eos=True)

    # Create an LLM.
    logger.info("Creating LLM for model={}", MODEL_NAME)
    # Required for callable serialization when using llm_engine.apply_model.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    # Keep EngineCore in-process so nsys can see layer-level NVTX ranges.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["BENCH_BATCH_SIZE"] = str(batch_size)
    if profile_start_step is not None:
        os.environ["BENCH_PROFILE_START_STEP"] = str(profile_start_step)
    else:
        os.environ.pop("BENCH_PROFILE_START_STEP", None)
    if profile_end_step is not None:
        os.environ["BENCH_PROFILE_END_STEP"] = str(profile_end_step)
    else:
        os.environ.pop("BENCH_PROFILE_END_STEP", None)
    llm = LLM(model=MODEL_NAME, trust_remote_code=True, enforce_eager=True)
    patch_counts = inject_nvtx_annotations(llm=llm)
    logger.info("NVTX layer patch counts: {}", patch_counts)

    if warmup_runs > 0:
        logger.info("Running {} warmup run(s)...", warmup_runs)
        warmup_params = SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=min(8, max_tokens)
        )
        for run_idx in range(warmup_runs):
            logger.info("Warmup run {}/{}", run_idx + 1, warmup_runs)
            llm.generate(prompts, warmup_params)

    use_profile_window = profile_start_step is not None or profile_end_step is not None
    logger.info(
        "Starting generation with profile window enabled={} start={} end={}",
        use_profile_window,
        profile_start_step,
        profile_end_step,
    )
    if not use_profile_window:
        _cuda_profiler_start()
    try:
        inference_range = (
            f"Inference/batch_{batch_size}/profile_window_{profile_start_step}_{profile_end_step}"
            if use_profile_window
            else f"Inference/batch_{batch_size}"
        )
        nvtx.range_push(inference_range)
        try:
            outputs = llm.generate(prompts, sampling_params)
        finally:
            nvtx.range_pop()
    finally:
        if not use_profile_window:
            _cuda_profiler_stop()

    logger.info("Outputs:")
    for output in outputs:
        logger.info("Prompt: {}", output.prompt)
        text = output.outputs[0].text if output.outputs else ""
        logger.info("Output: {}", text)
        logger.info("-" * 50)

    return log_file


def main() -> None:
    """Program entry point.

    Returns:
        None: No return value.
    """
    args = parse_args()
    log_file = bench(
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        warmup_runs=args.warmup_runs,
        profile_start_step=args.profile_start_step,
        profile_end_step=args.profile_end_step,
    )
    logger.info("Benchmark log saved to {}", log_file)


if __name__ == "__main__":
    main()
