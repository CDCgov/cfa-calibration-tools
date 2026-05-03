"""Benchmark serial and parallel example-model calibration execution."""

from __future__ import annotations

import argparse
import json
import timeit
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from calibrationtools.sampler import ABCSampler
from calibrationtools.variance_adapter import AdaptMultivariateNormalVariance

from .calibrate import (
    CALIBRATION_SPEC,
    DEFAULT_INPUTS,
    PRIORS,
    build_perturbation_kernel,
    outputs_to_distance,
)
from .direct_runner import ExampleModelDirectRunner

DEFAULT_BENCHMARK_DIR = Path("benchmarks")
DEFAULT_BENCHMARK_OUTPUT = DEFAULT_BENCHMARK_DIR / "parallelization_check.json"
DEFAULT_GENERATION_PARTICLE_COUNT = 100
MIN_GENERATION_PARTICLE_COUNT = 3
DEFAULT_TOLERANCE_VALUES = [5.0, 2.0]
DEFAULT_WORKER_COUNTS = (8, 2, 1)


class _BenchmarkRunResult(Protocol):
    smc_step_attempts: Any


class _BenchmarkSampler(Protocol):
    def run_serial(self) -> _BenchmarkRunResult: ...

    def run_parallel(self, *, max_workers: int) -> _BenchmarkRunResult: ...


@dataclass(frozen=True)
class BenchmarkResult:
    """One benchmark timing result."""

    time: float
    attempts: Any
    max_workers: int | None


def build_sampler(
    *,
    generation_particle_count: int = DEFAULT_GENERATION_PARTICLE_COUNT,
    tolerance_values: list[float] | None = None,
) -> ABCSampler:
    """Build the sampler used by the benchmark command."""
    return ABCSampler(
        generation_particle_count=generation_particle_count,
        tolerance_values=(
            tolerance_values
            if tolerance_values is not None
            else DEFAULT_TOLERANCE_VALUES
        ),
        priors=PRIORS,
        perturbation_kernel=build_perturbation_kernel(),
        variance_adapter=AdaptMultivariateNormalVariance(),
        default_parameters=DEFAULT_INPUTS,
        outputs_to_distance=outputs_to_distance,
        target_data=CALIBRATION_SPEC.target_data,
        model_runner=ExampleModelDirectRunner(),
        entropy=CALIBRATION_SPEC.entropy,
        verbose=False,
    )


def run_benchmark(
    *,
    sampler: _BenchmarkSampler | None = None,
    worker_counts: Sequence[int] = DEFAULT_WORKER_COUNTS,
    generation_particle_count: int = DEFAULT_GENERATION_PARTICLE_COUNT,
) -> list[BenchmarkResult]:
    """Run serial and parallel timing checks and return structured results."""
    benchmark_sampler = sampler or build_sampler(
        generation_particle_count=generation_particle_count
    )
    results: list[BenchmarkResult] = []

    start = timeit.default_timer()
    calibration_results = benchmark_sampler.run_serial()
    elapsed = timeit.default_timer() - start
    print(f"workers: serial, time: {elapsed}")
    results.append(
        BenchmarkResult(
            time=elapsed,
            attempts=calibration_results.smc_step_attempts,
            max_workers=None,
        )
    )

    for max_workers in worker_counts:
        start = timeit.default_timer()
        calibration_results = benchmark_sampler.run_parallel(
            max_workers=max_workers
        )
        elapsed = timeit.default_timer() - start
        print(f"workers: {max_workers}, time: {elapsed}")
        results.append(
            BenchmarkResult(
                time=elapsed,
                attempts=calibration_results.smc_step_attempts,
                max_workers=max_workers,
            )
        )

    return results


def write_benchmark_results(
    results: Sequence[BenchmarkResult],
    output_path: Path = DEFAULT_BENCHMARK_OUTPUT,
) -> None:
    """Write benchmark results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(result) for result in results]) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark command-line parser."""
    parser = argparse.ArgumentParser(
        description="Compare serial and parallel example calibration."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BENCHMARK_OUTPUT,
        help="JSON output path for benchmark results.",
    )
    parser.add_argument(
        "--generation-particle-count",
        type=int,
        default=DEFAULT_GENERATION_PARTICLE_COUNT,
        help="Accepted particles per SMC generation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=list(DEFAULT_WORKER_COUNTS),
        help="Parallel worker counts to benchmark.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the example benchmark command."""
    args = build_parser().parse_args(argv)
    if args.generation_particle_count < MIN_GENERATION_PARTICLE_COUNT:
        raise ValueError(
            "--generation-particle-count must be at least "
            f"{MIN_GENERATION_PARTICLE_COUNT}"
        )
    results = run_benchmark(
        worker_counts=args.workers,
        generation_particle_count=args.generation_particle_count,
    )
    write_benchmark_results(results, args.output)


if __name__ == "__main__":
    main()
