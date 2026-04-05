"""Monkey patch the vllm Worker to inject NVTX range and enable cudaProfiling."""

from functools import wraps
import threading

from benchmarks.utils import get_layer_type, summarize_scheduler_output

from typing import Any

from vllm import LLM
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import SchedulerOutput
from loguru import logger


_thread_state = threading.local()


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
    return getattr(_thread_state, "phase_marker", "unknown_phase")


def monkey_patch_scheduler():
    original_schedule = Scheduler.schedule
    if getattr(Scheduler, "_logger_injected", False):
        # The schedule method is already patched, skip patching to avoid double counting.
        return

    @wraps(original_schedule)
    def logger_injected_schedule(self):
        scheduler_output: SchedulerOutput = original_schedule(self)
        phase_marker = _infer_phase_marker(scheduler_output)
        _thread_state.phase_marker = phase_marker

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


def monkey_patch_llm_engine(llm: LLM):
    """Inject per-layer NVTX ranges into the model forward pass.

    Args:
        llm (LLM): Initialized vLLM LLM instance.

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
                worker_nvtx.range_push("model_forward")
                try:
                    return _original_model_forward(*args, **kwargs)
                finally:
                    worker_nvtx.range_pop()
                    worker_nvtx.range_pop()

            setattr(phase_wrapped_model_forward, "_phase_nvtx_injected", True)
            model.forward = phase_wrapped_model_forward

        # Iterate through the layers and patch their forward methods to include NVTX markers
        for index, layer in enumerate(layers):
            if not hasattr(layer, "forward"):
                raise RuntimeError(f"Layer {index} does not have a forward method.")

            layer_type = get_layer_type(hybrid_layer_pattern[index])

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
                worker_nvtx.range_push(_nvtx_marker_name)
                try:
                    output = _original_forward(*args, **kwargs)
                finally:
                    worker_nvtx.range_pop()
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
