import json
from pathlib import Path

import pytest
from example_model.example_model import Binom_BP_Model


@pytest.fixture
def input_data():
    """Load test input from fixture file."""
    input_path = Path(__file__).parent / "model_input.json"
    with open(input_path, "r") as f:
        return json.load(f)


@pytest.fixture
def expected_output():
    """Load expected output from fixture file."""
    expected_path = Path(__file__).parent / "expected_output.json"
    with open(expected_path, "r") as f:
        return json.load(f)


def test_model_produces_expected_output(input_data, expected_output):
    results = Binom_BP_Model.simulate(input_data)
    assert results == expected_output
