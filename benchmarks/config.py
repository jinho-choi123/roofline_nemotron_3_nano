"""Benchmark Configuration"""

from loguru import logger
import os


class BenchmarkConfig:
    """Configuration for the benchmark.
    If environment variables are set, then they have the highest priority.
    """

    # model_name for huggingface models, e.g., "nvidia/Nemotron-H-8B-Base-8K"
    model_name: str

    # batch size for the benchmark
    batch_size: int

    # maximum sequence length for the benchmark
    # Decoding will be performed until the max sequence length is reached by setting ignore_eos=True in sampling params.
    max_seq_length: int

    # number of warmup iterations before measuring latency
    warmup_iterations: int

    # number of tokens in the prompt for the benchmark
    prompt_length: int

    # Environment variable names for optional runtime overrides.
    ENV_MODEL_NAME = "BENCHMARK_MODEL_NAME"
    ENV_BATCH_SIZE = "BENCHMARK_BATCH_SIZE"
    ENV_MAX_SEQ_LENGTH = "BENCHMARK_MAX_SEQ_LENGTH"
    ENV_WARMUP_ITERATIONS = "BENCHMARK_WARMUP_ITERATIONS"
    ENV_PROMPT_LENGTH = "BENCHMARK_PROMPT_LENGTH"

    def __init__(
        self,
        model_name: str = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        batch_size: int = 4,
        max_seq_length: int = 513,
        warmup_iterations: int = 3,
        prompt_length: int = 512,
    ):

        env_model_name = os.getenv(self.ENV_MODEL_NAME)
        env_batch_size = os.getenv(self.ENV_BATCH_SIZE)
        env_max_seq_length = os.getenv(self.ENV_MAX_SEQ_LENGTH)
        env_warmup_iterations = os.getenv(self.ENV_WARMUP_ITERATIONS)
        env_prompt_length = os.getenv(self.ENV_PROMPT_LENGTH)

        self.model_name = env_model_name if env_model_name is not None else model_name
        self.batch_size = (
            int(env_batch_size) if env_batch_size is not None else batch_size
        )
        self.max_seq_length = (
            int(env_max_seq_length)
            if env_max_seq_length is not None
            else max_seq_length
        )
        self.warmup_iterations = (
            int(env_warmup_iterations)
            if env_warmup_iterations is not None
            else warmup_iterations
        )
        self.prompt_length = (
            int(env_prompt_length) if env_prompt_length is not None else prompt_length
        )

        assert self.max_seq_length > self.prompt_length, (
            "max_seq_length must be greater than prompt_length to allow for generation."
        )

        logger.info(f"BenchmarkConfig initialized with: {self.__dict__}")
