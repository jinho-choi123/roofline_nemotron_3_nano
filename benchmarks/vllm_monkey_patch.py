"""Monkey patch the vllm Worker to inject NVTX range and enable cudaProfiling."""

from benchmarks.config import BenchmarkConfig

from functools import wraps
import threading

from benchmarks.utils import (
    get_layer_type,
    summarize_scheduler_output,
    cuda_profiler_start,
    cuda_profiler_stop,
)

from typing import Any

from vllm import LLM
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import SchedulerOutput
from loguru import logger
import torch


_thread_state = threading.local()


_PROFILED_PHASE_MARKERS = {"prefill_phase", "decode_phase"}
_state_lock = threading.Lock()
_state = {
    "phase_marker": "unknown_phase",
    "profiling_enabled": False,
}


def _infer_phase_marker(scheduler_output: SchedulerOutput) -> str:
    """Infer phase marker name from scheduler output for single-request benchmarking."""
    token_counts = list(scheduler_output.num_scheduled_tokens.values())
    if not token_counts:
        return "idle_phase"

    # This benchmark is constrained to a single active request.
    if len(token_counts) > 1:
        raise RuntimeError(
            f"Multiple active requests found in scheduler output: {token_counts}. This benchmark is designed for single-request scenarios."
        )

    return "decode_phase" if token_counts[0] == 1 else "prefill_phase"


def _get_current_phase_marker() -> str:
    with _state_lock:
        return _state["phase_marker"]


def _set_current_phase_marker(phase_marker: str) -> None:
    with _state_lock:
        _state["phase_marker"] = phase_marker


def set_profiling_enabled(enabled: bool) -> None:
    """Enable or disable CUDA profiler API calls inside layer wrappers."""
    with _state_lock:
        _state["profiling_enabled"] = enabled


def _is_profiling_enabled() -> bool:
    with _state_lock:
        return _state["profiling_enabled"]


def monkey_patch_scheduler():
    original_schedule = Scheduler.schedule
    if getattr(Scheduler, "_logger_injected", False):
        # The schedule method is already patched, skip patching to avoid double counting.
        return

    @wraps(original_schedule)
    def logger_injected_schedule(self):
        scheduler_output: SchedulerOutput = original_schedule(self)
        phase_marker = _infer_phase_marker(scheduler_output)
        _set_current_phase_marker(phase_marker)

        summary = summarize_scheduler_output(scheduler_output)
        logger.info("-" * 50)
        logger.info("SchedulerOutput summary:")

        # This benchmark is designed to have only one request in-flight at a time, so there should be only one req_id in the summary.
        assert len(summary.keys()) <= 1, (
            "This benchmark is designed to have only one request in-flight at a time. Multiple req_id found in SchedulerOutput summary."
        )

        logger.info(f"phase_marker={phase_marker}")

        for req_id, log in summary.items():
            logger.info(f"req_id={req_id} {log}")
        logger.info("-" * 50)

        return scheduler_output

    setattr(Scheduler, "schedule", logger_injected_schedule)
    setattr(Scheduler, "_logger_injected", True)


def monkey_patch_llm_engine(llm: LLM, bench_config: BenchmarkConfig):
    """Inject per-layer NVTX ranges into the model forward pass.

    Args:
        llm (LLM): Initialized vLLM LLM instance.
        bench_config (BenchmarkConfig): Benchmark configuration instance.

    """

    def _patch_worker_module(worker_module):
        """Patch the forward method of the model to include NVTX ranges."""

        model = (
            worker_module.model if hasattr(worker_module, "model") else worker_module
        )

        if not (hasattr(model, "layers") and hasattr(model, "config")):
            raise RuntimeError(
                "Worker module does not expose layers/config attributes."
            )

        hybrid_layer_pattern: str = model.config.hybrid_override_pattern
        layers = model.layers

        if len(hybrid_layer_pattern) != len(layers):
            raise RuntimeError(
                f"Length of hybrid_layer_pattern ({len(hybrid_layer_pattern)}) does not match number of layers ({len(layers)})."
            )

        # import torch's nvtx module here
        import torch.cuda.nvtx as worker_nvtx

        original_model_forward = model.forward
        if not getattr(original_model_forward, "_phase_nvtx_injected", False):

            @wraps(original_model_forward)
            def phase_wrapped_model_forward(
                *args: Any,
                _original_model_forward=original_model_forward,
                **kwargs: Any,
            ) -> Any:
                phase_marker = _get_current_phase_marker()
                worker_nvtx.range_push(phase_marker)
                try:
                    return _original_model_forward(*args, **kwargs)
                finally:
                    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before popping the NVTX range to get accurate timing.
                    worker_nvtx.range_pop()

            setattr(phase_wrapped_model_forward, "_phase_nvtx_injected", True)
            model.forward = phase_wrapped_model_forward

        # Iterate through the layers and patch their forward methods to include NVTX markers
        for index, layer in enumerate(layers):
            if not hasattr(layer, "forward"):
                raise RuntimeError(f"Layer {index} does not have a forward method.")

            layer_type = get_layer_type(hybrid_layer_pattern[index])

            is_profile_target_layer = (
                True if index in bench_config.profile_target_layer_ids else False
            )

            # If the layer is not in the profile_target_layer_ids, we will not inject NVTX markers for it to save profiling overhead, and log this decision.
            if not is_profile_target_layer:
                logger.info(
                    f"Layer {index} ({layer_type}) | NOT TARGET LAYER | no NVTX markers will be injected for this layer to save profiling overhead."
                )
                continue
            else:
                logger.info(
                    f"Layer {index} ({layer_type}) | TARGET LAYER | it will be profiled with NVTX markers injected."
                )

            nvtx_marker_name = f"layer={index}_module={layer_type}"

            original_forward = layer.forward

            # check if the layer is already patched
            if getattr(original_forward, "_nvtx_injected", False):
                # the original_forward already has "_nvtx_injected" attribute set to True.
                # which means the forward method has already been patched. skip patching to avoid double counting.
                continue

            @wraps(original_forward)
            def nvtx_injected_forward(
                *args: Any,
                _original_forward=original_forward,
                _nvtx_marker_name=nvtx_marker_name,
                **kwargs: Any,
            ) -> Any:
                """A wrapper around the original forward method that includes NVTX markers."""
                phase_marker = _get_current_phase_marker()
                should_capture = _is_profiling_enabled() and (
                    phase_marker in _PROFILED_PHASE_MARKERS
                )
                profiler_started = False
                try:
                    if should_capture:
                        logger.debug(
                            f"Starting CUDA profiler for {_nvtx_marker_name} during {phase_marker}."
                        )  # log when the profiler starts for better visibility in logs
                        cuda_profiler_start()
                        profiler_started = True
                    worker_nvtx.range_push(_nvtx_marker_name)

                    output = _original_forward(*args, **kwargs)
                finally:
                    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before popping the NVTX range to get accurate timing.
                    worker_nvtx.range_pop()
                    if profiler_started:
                        cuda_profiler_stop()
                        logger.debug(
                            f"Stopped CUDA profiler for {_nvtx_marker_name} during {phase_marker}."
                        )  # log when the profiler stops for better visibility in logs
                return output

            setattr(nvtx_injected_forward, "_nvtx_injected", True)
            layer.forward = nvtx_injected_forward

        return {
            "module_type": type(worker_module).__name__,
            "model_type": type(model).__name__,
            "pattern": hybrid_layer_pattern,
            "num_layers": len(layers),
        }

    worker_module_patch_result = llm.llm_engine.apply_model(_patch_worker_module)

    if not worker_module_patch_result:
        raise RuntimeError("Failed to patch worker module.")
