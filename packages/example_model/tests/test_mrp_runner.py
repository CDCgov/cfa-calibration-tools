from pathlib import Path

import numpy as np
import pytest
from example_model.mrp_runner import ExampleModelMRPRunner
from mrp.runtime import RunResult


def test_mrp_runner_parses_population(monkeypatch):
    def fake_mrp_run(config_path, overrides):
        assert config_path == Path("/tmp/example_model.mrp.toml")
        assert overrides["input"] == {"seed": 123}
        assert overrides["output"] == {"spec": "stdout"}
        return RunResult(
            exit_code=0,
            stdout=b"generation,population\r\n0,1\r\n1,2\r\n",
            stderr=b"",
        )

    monkeypatch.setattr("example_model.mrp_runner.mrp_run", fake_mrp_run)

    runner = ExampleModelMRPRunner("/tmp/example_model.mrp.toml")

    assert runner.simulate({"seed": 123}) == [1, 2]


def test_mrp_runner_raises_on_failed_run(monkeypatch):
    def fake_mrp_run(config_path, overrides):
        return RunResult(
            exit_code=1,
            stdout=b"",
            stderr=b"model failed",
        )

    monkeypatch.setattr("example_model.mrp_runner.mrp_run", fake_mrp_run)

    runner = ExampleModelMRPRunner("/tmp/example_model.mrp.toml")

    with pytest.raises(RuntimeError, match="model failed"):
        runner.simulate({"seed": 123})


def test_mrp_runner_requires_population_column(monkeypatch):
    def fake_mrp_run(config_path, overrides):
        return RunResult(
            exit_code=0,
            stdout=b"generation,size\r\n0,1\r\n",
            stderr=b"",
        )

    monkeypatch.setattr("example_model.mrp_runner.mrp_run", fake_mrp_run)

    runner = ExampleModelMRPRunner("/tmp/example_model.mrp.toml")

    with pytest.raises(
        ValueError,
        match="MRP model output did not include a 'population' column",
    ):
        runner.simulate({"seed": 123})


def test_mrp_runner_converts_numpy_scalars(monkeypatch):
    def fake_mrp_run(config_path, overrides):
        assert overrides["input"] == {"seed": 123, "p": 0.5}
        assert isinstance(overrides["input"]["seed"], int)
        assert isinstance(overrides["input"]["p"], float)
        return RunResult(
            exit_code=0,
            stdout=b"generation,population\r\n0,1\r\n",
            stderr=b"",
        )

    monkeypatch.setattr("example_model.mrp_runner.mrp_run", fake_mrp_run)

    runner = ExampleModelMRPRunner("/tmp/example_model.mrp.toml")

    assert runner.simulate({"seed": np.int64(123), "p": np.float64(0.5)}) == [1]


def test_mrp_runner_ignores_stdout_preamble(monkeypatch):
    def fake_mrp_run(config_path, overrides):
        return RunResult(
            exit_code=0,
            stdout=(
                b"Running Binomial Branching Process Model...\n"
                b"generation,population\r\n0,1\r\n1,2\r\n"
            ),
            stderr=b"",
        )

    monkeypatch.setattr("example_model.mrp_runner.mrp_run", fake_mrp_run)

    runner = ExampleModelMRPRunner("/tmp/example_model.mrp.toml")

    assert runner.simulate({"seed": 123}) == [1, 2]
