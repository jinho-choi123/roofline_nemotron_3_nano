"""Monkey patch the vllm Worker to inject NVTX range and enable cudaProfiling."""

from functools import wraps

from benchmarks.utils import get_layer_type

from typing import Any

from vllm import LLM, SamplingParams
import torch.nn as nn
from vllm.v1.core.sched.scheduler import Scheduler
from loguru import logger


def monkey_patch_scheduler():
    original_schedule = Scheduler.schedule
    if getattr(Scheduler, "_logger_injected", False):
        # The schedule method is already patched, skip patching to avoid double counting.
        return

    @wraps(original_schedule)
    def logger_injected_schedule(*args, **kwargs):
        # Add logging or other functionality here

        scheduler_output = original_schedule(*args, **kwargs)
        # Add logging or other functionality here
        logger.info(f"Scheduler called with args: {args}, kwargs: {kwargs}")
        return scheduler_output

    Scheduler.schedule = logger_injected_schedule
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
