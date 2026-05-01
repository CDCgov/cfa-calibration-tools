from __future__ import annotations

import tomllib
from pathlib import Path

from example_model.example_model import run_inline
from mrp import run as mrp_run

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MODEL_ROOT = (
    REPO_ROOT / "packages" / "example_model" / "src" / "example_model"
)
INLINE_MODEL_CALLABLE = "example_model.example_model:run_inline"
INLINE_CLOUD_CALLABLE = "calibrationtools.cloud.executor:execute_cloud_run"


def _load_example_config(filename: str) -> dict:
    """Load one packaged example-model MRP TOML config."""
    with (EXAMPLE_MODEL_ROOT / filename).open("rb") as f:
        return tomllib.load(f)


def test_non_docker_mrp_configs_use_inline_runtime():
    """Validate local, task, and legacy cloud MRP configs use inline runtime."""
    expected_callables = {
        "example_model.mrp.toml": INLINE_MODEL_CALLABLE,
        "example_model.mrp.cloud.toml": INLINE_CLOUD_CALLABLE,
    }

    for filename, callable_path in expected_callables.items():
        config = _load_example_config(filename)
        assert config["runtime"]["spec"] == "inline"
        assert config["runtime"]["callable"] == callable_path
        assert "command" not in config["runtime"]
        assert "args" not in config["runtime"]
        assert "env" not in config["runtime"]


def test_docker_mrp_config_remains_process_backed():
    """Validate the Docker MRP config runs through a process runtime."""
    config = _load_example_config("example_model.mrp.docker.toml")

    assert config["runtime"].get("spec", "process") != "inline"
    assert config["runtime"]["command"] == "sh"
    assert "docker run" in " ".join(config["runtime"]["args"])
    assert "callable" not in config["runtime"]


def test_docker_config_uses_calling_user_identity():
    """Validate local Docker runs bind output as the calling user."""
    config = _load_example_config("example_model.mrp.docker.toml")

    command = " ".join(config["runtime"]["args"])
    assert '--user "$(id -u):$(id -g)"' in command


def test_run_inline_writes_csv_to_stdout(capsys):
    """Validate the example inline MRP callable emits CSV output."""
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


def test_packaged_mrp_config_runs_inline_and_emits_csv():
    """Run the packaged local MRP config as an integration smoke test."""
    result = mrp_run(
        EXAMPLE_MODEL_ROOT / "example_model.mrp.toml",
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
