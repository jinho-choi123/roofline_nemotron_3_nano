"""Benchmark Configuration"""

from loguru import logger


class BenchmarkConfig:
    """Configuration for the benchmark."""

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

    def __init__(
        self,
        model_name: str = "nvidia/Nemotron-H-8B-Base-8K",
        batch_size: int = 4,
        max_seq_length: int = 2048,
        warmup_iterations: int = 3,
        prompt_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.warmup_iterations = warmup_iterations
        self.prompt_length = prompt_length

        logger.info(f"BenchmarkConfig initialized with: {self.__dict__}")
