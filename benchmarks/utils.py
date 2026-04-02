"""Utility functions for benchmarks."""

from typing import Any, Dict, List

from benchmarks.config import BenchmarkConfig

from transformers import AutoTokenizer
from vllm.v1.core.sched.output import SchedulerOutput
import torch


def build_prompt(config: BenchmarkConfig) -> List[str]:
    """Build a prompt of the specified length."""
    prompt_token_ids = [123] * config.prompt_length

    # Convert token IDs to tokens and join them into a string
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    prompt_string = tokenizer.decode(prompt_token_ids)

    return [prompt_string] * config.batch_size


def get_layer_type(layer_token: str) -> str:
    """Extract the layer type from the hybrid layer token.

    Args:
        layer_token (str): The token representing the layer type in the hybrid pattern.
    Returns:
        str: The human-readable layer type corresponding to the token. "Unknown" if the layer is not recognized.
    """

    layer_type_names = {"M": "Mamba", "-": "MLP", "*": "Attention", "E": "MoE"}

    if layer_token not in layer_type_names.keys():
        return "Unknown"

    return layer_type_names[layer_token]


def cuda_profiler_start() -> None:
    """Start CUDA profiler range capture.

    Returns:
        None: No return value.

    Raises:
        RuntimeError: If the CUDA profiler start call fails.
    """
    status = torch.cuda.cudart().cudaProfilerStart()
    if status != 0:
        raise RuntimeError(f"cudaProfilerStart failed with status={status}")


def cuda_profiler_stop() -> None:
    """Stop CUDA profiler range capture.

    Returns:
        None: No return value.

    Raises:
        RuntimeError: If the CUDA profiler stop call fails.
    """
    status = torch.cuda.cudart().cudaProfilerStop()
    if status != 0:
        raise RuntimeError(f"cudaProfilerStop failed with status={status}")


def summarize_scheduler_output(
    scheduler_output: SchedulerOutput,
) -> Dict[str, Any]:
    """Summarize SchedulerOutput in a version-tolerant way for logging.

    Args:
        scheduler_output (Any): Output object returned by vLLM scheduler.

    Returns:
        Dict[str, Any]: Structured summary fields for logger consumption.
    """
    summary = {}
    for req_id, num_scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
        if num_scheduled_tokens == 1:
            # decode
            summary[req_id] = "decode 1 token"

        else:
            # prefill
            summary[req_id] = f"prefill {num_scheduled_tokens} tokens"

    return summary
