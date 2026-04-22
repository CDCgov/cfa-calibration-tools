from pathlib import Path

import numpy as np
import pytest
from mrp.runtime import RunResult

from calibrationtools.mrp_csv_runner import (
    CSVOutputMRPRunner,
    make_csv_output_dir_reader,
)


def _runner(mrp_run_func):
    return CSVOutputMRPRunner(
        "/tmp/example_model.mrp.toml",
        output_filename="output.csv",
        value_column="population",
        value_parser=int,
        header_fields=("generation", "population"),
        mrp_run_func=mrp_run_func,
    )


def test_csv_output_mrp_runner_parses_population():
    def fake_mrp_run(config_path, overrides):
        assert config_path == Path("/tmp/example_model.mrp.toml")
        assert overrides["input"] == {"seed": 123}
        assert overrides["output"] == {"spec": "stdout"}
        return RunResult(
            exit_code=0,
            stdout=b"generation,population\r\n0,1\r\n1,2\r\n",
            stderr=b"",
        )

    runner = _runner(fake_mrp_run)

    assert runner.simulate({"seed": 123}) == [1, 2]


def test_csv_output_mrp_runner_raises_on_failed_run():
    def fake_mrp_run(config_path, overrides):
        return RunResult(
            exit_code=1,
            stdout=b"",
            stderr=b"model failed",
        )

    runner = _runner(fake_mrp_run)

    with pytest.raises(RuntimeError, match="model failed"):
        runner.simulate({"seed": 123})


def test_csv_output_mrp_runner_requires_population_column():
    def fake_mrp_run(config_path, overrides):
        return RunResult(
            exit_code=0,
            stdout=b"generation,size\r\n0,1\r\n",
            stderr=b"",
        )

    runner = _runner(fake_mrp_run)

    with pytest.raises(
        ValueError,
        match="MRP model output did not include a 'population' column",
    ):
        runner.simulate({"seed": 123})


def test_csv_output_mrp_runner_converts_numpy_scalars():
    def fake_mrp_run(config_path, overrides):
        assert overrides["input"] == {"seed": 123, "p": 0.5}
        assert isinstance(overrides["input"]["seed"], int)
        assert isinstance(overrides["input"]["p"], float)
        return RunResult(
            exit_code=0,
            stdout=b"generation,population\r\n0,1\r\n",
            stderr=b"",
        )

    runner = _runner(fake_mrp_run)

    assert runner.simulate({"seed": np.int64(123), "p": np.float64(0.5)}) == [
        1
    ]


def test_csv_output_mrp_runner_ignores_stdout_preamble():
    def fake_mrp_run(config_path, overrides):
        return RunResult(
            exit_code=0,
            stdout=(
                b"Running Binomial Branching Process Model...\n"
                b"generation,population\r\n0,1\r\n1,2\r\n"
            ),
            stderr=b"",
        )

    runner = _runner(fake_mrp_run)

    assert runner.simulate({"seed": 123}) == [1, 2]


def test_csv_output_mrp_runner_uses_staged_input_and_output_dirs(tmp_path):
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

    runner = _runner(fake_mrp_run)

    assert runner.simulate(
        {"seed": 123},
        input_path=input_path,
        output_dir=run_output_dir,
        run_id="gen-1_particle-1",
    ) == [1, 2]


def test_make_csv_output_dir_reader_reads_expected_column(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "output.csv").write_text("generation,population\n0,1\n1,2\n")

    reader = make_csv_output_dir_reader(
        output_filename="output.csv",
        value_column="population",
        value_parser=int,
    )

    assert reader(output_dir) == [1, 2]
