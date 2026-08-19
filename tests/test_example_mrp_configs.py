from __future__ import annotations

from pathlib import Path

from example_model.example_model import run_inline
from mrp import run as mrp_run

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MODEL_ROOT = (
    REPO_ROOT / "packages" / "example_model" / "src" / "example_model"
)


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
