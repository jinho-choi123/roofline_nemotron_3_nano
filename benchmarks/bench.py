"""Run benchmark."""

from typing import List
from benchmarks.config import BenchmarkConfig
from benchmarks.utils import build_prompt, cuda_profiler_start, cuda_profiler_stop
from benchmarks.vllm_monkey_patch import monkey_patch_llm_engine
from vllm import LLM, SamplingParams
from loguru import logger
import os
import torch.cuda.nvtx as nvtx


def _setup_benchmark(config: BenchmarkConfig) -> tuple[List[str], LLM, SamplingParams]:
    """Set up the benchmark by building the prompt, initializing the LLM, and setting up sampling parameters."""
    logger.info(f"Setting up benchmark with configuration: {config.__dict__}")

    # Required for callable serialization when using llm_engine.apply_model.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # Keep EngineCore in-process so nsys can see layer-level NVTX ranges.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Build the prompt
    prompt = build_prompt(config)

    # Initialize the vLLM LLM instance
    llm = LLM(model=config.model_name, trust_remote_code=True, enforce_eager=True)

    # Monkey patch the LLM engine to include NVTX markers
    monkey_patch_llm_engine(llm)

    # Warmup the llm engine
    logger.info("Warming up the LLM engine...")
    llm.generate(prompt, SamplingParams(max_tokens=1))

    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=config.max_seq_length - config.prompt_length,
        ignore_eos=True,  # Ensure decoding continues until max length is reached
    )

    return prompt, llm, sampling_params


def run_benchmark(config: BenchmarkConfig):
    """Run the benchmark with the specified configuration."""
    logger.info(f"Running benchmark with configuration: {config.__dict__}")

    prompt, llm, sampling_params = _setup_benchmark(config)

    # Generate output from the LLM using the prompt and sampling parameters
    logger.info("Generating output from the LLM...")

    try:
        cuda_profiler_start()
        nvtx.range_push("llm_generation")

        llm.generate(prompt, sampling_params)
    except Exception as e:
        logger.error(f"An error occurred during benchmark execution: {e}")
        raise
    finally:
        nvtx.range_pop()
        cuda_profiler_stop()
