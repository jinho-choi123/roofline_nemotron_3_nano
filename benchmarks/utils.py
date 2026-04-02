"""Utility functions for benchmarks."""

from typing import List

from benchmarks.config import BenchmarkConfig

from transformers import AutoTokenizer


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
