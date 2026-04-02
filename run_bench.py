"""Run benchmark."""

from benchmarks.bench import run_benchmark
from benchmarks.config import BenchmarkConfig


def main():

    config = BenchmarkConfig()
    run_benchmark(config)


if __name__ == "__main__":
    main()
