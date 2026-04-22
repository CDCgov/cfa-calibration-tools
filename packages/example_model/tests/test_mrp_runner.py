from pathlib import Path

import example_model.mrp_runner as mrp_runner
import numpy as np
import pytest
from example_model.mrp_runner import ExampleModelMRPRunner
from mrp.runtime import RunResult

from calibrationtools.mrp_csv_runner import CSVOutputMRPRunner


def test_example_model_mrp_runner_is_shared_csv_runner():
    runner = ExampleModelMRPRunner("/tmp/example_model.mrp.toml")

    assert isinstance(runner, CSVOutputMRPRunner)


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

    assert runner.simulate({"seed": np.int64(123), "p": np.float64(0.5)}) == [
        1
    ]


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


def test_mrp_runner_uses_staged_input_and_output_dirs(monkeypatch, tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text('{"seed": 123, "run_id": "gen-1_particle-1"}')
    run_output_dir = tmp_path / "output"

    def fake_mrp_run(config_path, overrides, output_dir=None):
        assert config_path == Path("/tmp/example_model.mrp.toml")
        assert overrides["input"] == str(input_path)
        assert output_dir == str(run_output_dir)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        (run_output_dir / "output.csv").write_text(
            "generation,population\n0,1\n1,2\n"
        )
        return RunResult(exit_code=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("example_model.mrp_runner.mrp_run", fake_mrp_run)

    runner = ExampleModelMRPRunner("/tmp/example_model.mrp.toml")

    assert runner.simulate(
        {"seed": 123},
        input_path=input_path,
        output_dir=run_output_dir,
        run_id="gen-1_particle-1",
    ) == [1, 2]


def test_default_config_path_falls_back_to_materialized_assets(monkeypatch):
    monkeypatch.setattr(mrp_runner, "_BUNDLED_CONFIG_DIR", None)
    monkeypatch.setattr(
        mrp_runner,
        "_repo_default_config_path",
        lambda filename: None,
    )

    config_path = mrp_runner._default_config_path(
        "example_model.mrp.cloud.toml"
    )

    assert config_path.name == "example_model.mrp.cloud.toml"
    assert config_path.exists()
    assert (config_path.parent / "defaults.json").exists()
    assert 'input = "' in config_path.read_text()
