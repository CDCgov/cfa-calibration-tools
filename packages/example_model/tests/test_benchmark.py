from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from example_model import benchmark


class FakeSampler:
    """Small sampler double for benchmark command tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def run_serial(self):
        self.calls.append(("serial", None))
        return SimpleNamespace(smc_step_attempts=[1, 2])

    def run_parallel(self, *, max_workers: int):
        self.calls.append(("parallel", max_workers))
        return SimpleNamespace(smc_step_attempts=[max_workers])


def test_run_benchmark_invokes_serial_and_configured_parallel_workers():
    sampler = FakeSampler()

    results = benchmark.run_benchmark(
        sampler=sampler,
        worker_counts=(3, 1),
    )

    assert sampler.calls == [
        ("serial", None),
        ("parallel", 3),
        ("parallel", 1),
    ]
    assert [result.max_workers for result in results] == [None, 3, 1]
    assert [result.attempts for result in results] == [[1, 2], [3], [1]]


def test_write_benchmark_results_writes_json(tmp_path: Path):
    output_path = tmp_path / "benchmarks" / "parallelization_check.json"

    benchmark.write_benchmark_results(
        [
            benchmark.BenchmarkResult(
                time=1.25,
                attempts=[1, 2],
                max_workers=None,
            ),
            benchmark.BenchmarkResult(
                time=0.75,
                attempts=[3],
                max_workers=3,
            ),
        ],
        output_path,
    )

    assert output_path.read_text(encoding="utf-8") == (
        '[{"time": 1.25, "attempts": [1, 2], "max_workers": null}, '
        '{"time": 0.75, "attempts": [3], "max_workers": 3}]\n'
    )


def test_main_rejects_single_particle_benchmark():
    with pytest.raises(ValueError, match="at least 3"):
        benchmark.main(["--generation-particle-count", "1"])
