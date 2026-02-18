import json
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def input_file():
    """Path to the test input JSON file."""
    return Path(__file__).parent / "model_input.json"


@pytest.fixture
def expected_output():
    """Load expected output from fixture file."""
    expected_path = Path(__file__).parent / "expected_output.json"
    with open(expected_path, "r") as f:
        return json.load(f)


def test_model_produces_expected_output(input_file, expected_output):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_output:
        output_path = Path(tmp_output.name)

    try:
        subprocess.run(
            [
                "uv",
                "sync",
                "--all-packages",
            ],
        )
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "example_model.example_model",
                str(input_file),
                "-o",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert result.returncode == 0, f"Command failed with: {result.stderr}"

        with open(output_path, "r") as f:
            output_data = json.load(f)

        assert output_data == expected_output, (
            f"Output does not match expected. "
            f"Got: {output_data}, Expected: {expected_output}"
        )

    finally:
        if output_path.exists():
            output_path.unlink()
