import tomllib
from pathlib import Path

import example_model.mrp_runner as mrp_runner
import numpy as np
import pytest
from example_model.example_model import run_inline
from example_model.mrp_runner import ExampleModelMRPRunner
from mrp import run as mrp_run
from mrp.runtime import RunResult

from calibrationtools.mrp_csv_runner import CSVOutputMRPRunner

REPO_ROOT = Path(__file__).resolve().parents[3]
INLINE_MODEL_CALLABLE = "example_model.example_model:run_inline"
INLINE_CLOUD_CALLABLE = "example_model.cloud_mrp_executor:execute_cloud_run"


def _load_root_config(filename: str) -> dict:
    with (REPO_ROOT / filename).open("rb") as f:
        return tomllib.load(f)


@pytest.mark.parametrize(
    ("filename", "callable_path"),
    [
        ("example_model.mrp.toml", INLINE_MODEL_CALLABLE),
        ("example_model.mrp.task.toml", INLINE_MODEL_CALLABLE),
        ("example_model.mrp.cloud.toml", INLINE_CLOUD_CALLABLE),
    ],
)
def test_non_docker_mrp_configs_use_inline_runtime(
    filename: str, callable_path: str
):
    config = _load_root_config(filename)

    assert config["runtime"]["spec"] == "inline"
    assert config["runtime"]["callable"] == callable_path
    assert "command" not in config["runtime"]
    assert "args" not in config["runtime"]
    assert "env" not in config["runtime"]


def test_docker_mrp_config_remains_process_backed():
    config = _load_root_config("example_model.mrp.docker.toml")

    assert config["runtime"].get("spec", "process") != "inline"
    assert config["runtime"]["command"] == "sh"
    assert "docker run" in " ".join(config["runtime"]["args"])
    assert "callable" not in config["runtime"]


def test_run_inline_writes_csv_to_stdout(capsys):
    run_inline(
        {
            "input": {
                "seed": 123,
                "max_gen": 3,
                "n": 3,
                "p": 0.5,
                "max_infect": 500,
            },
            "output": {"spec": "stdout"},
        }
    )

    captured = capsys.readouterr()
    assert "Running Binomial Branching Process Model..." in captured.out
    assert "generation,population" in captured.out
    assert "0,1" in captured.out


def test_root_mrp_config_runs_inline_and_emits_csv():
    result = mrp_run(
        REPO_ROOT / "example_model.mrp.toml",
        {
            "input": {
                "seed": 123,
                "max_gen": 3,
                "n": 3,
                "p": 0.5,
                "max_infect": 500,
            },
            "output": {"spec": "stdout"},
        },
    )

    assert result.ok, result.stderr.decode()
    output = result.stdout.decode()
    assert "generation,population" in output
    assert "0,1" in output


@pytest.mark.parametrize(
    ("filename", "callable_path"),
    [
        ("example_model.mrp.toml", INLINE_MODEL_CALLABLE),
        ("example_model.mrp.cloud.toml", INLINE_CLOUD_CALLABLE),
    ],
)
def test_bundled_non_docker_config_text_uses_inline_runtime(
    filename: str, callable_path: str, tmp_path: Path
):
    text = mrp_runner._bundled_config_text(
        filename,
        defaults_path=tmp_path / "defaults.json",
    )

    assert 'spec = "inline"' in text
    assert f'callable = "{callable_path}"' in text
    assert "command = " not in text
    assert "args = [" not in text


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
