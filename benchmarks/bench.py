"""Run benchmark."""

from typing import List

from benchmarks.config import BenchmarkConfig
from benchmarks.utils import build_prompt
from benchmarks.vllm_monkey_patch import monkey_patch_llm_engine
from vllm import LLM, SamplingParams
from loguru import logger


def _setup_benchmark(config: BenchmarkConfig) -> tuple[List[str], LLM, SamplingParams]:
    """Set up the benchmark by building the prompt, initializing the LLM, and setting up sampling parameters."""
    logger.info(f"Setting up benchmark with configuration: {config.__dict__}")

    # Build the prompt
    prompt = build_prompt(config)

    # Initialize the vLLM LLM instance
    llm = LLM(model=config.model_name, trust_remote_code=True)

    # Monkey patch the LLM engine to include NVTX markers
    monkey_patch_llm_engine(llm)

    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=config.max_seq_length,
        ignore_eos=True,  # Ensure decoding continues until max length is reached
    )

    return prompt, llm, sampling_params


def run_benchmark(config: BenchmarkConfig):
    """Run the benchmark with the specified configuration."""
    logger.info(f"Running benchmark with configuration: {config.__dict__}")

    prompt, llm, sampling_params = _setup_benchmark(config)

    # Warmup iterations
    for _ in range(config.warmup_iterations):
        llm.generate(prompt, sampling_params)

    llm.generate(prompt, sampling_params)
