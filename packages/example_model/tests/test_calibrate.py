from __future__ import annotations

import pytest
from example_model.calibrate import (
    CALIBRATION_SPEC,
    DEFAULT_CLOUD_CONFIG_PATH,
    DEFAULT_DOCKER_MRP_CONFIG_PATH,
    DEFAULT_INPUTS,
    DEFAULT_MRP_CONFIG_PATH,
    PRIORS,
    TOLERANCE_VALUES,
    build_perturbation_kernel,
    outputs_to_distance,
)


def test_calibration_spec_carries_example_model_choices():
    """Validate model-owned calibration choices stay in the example package."""
    assert CALIBRATION_SPEC.default_inputs == DEFAULT_INPUTS
    assert CALIBRATION_SPEC.priors == PRIORS
    assert CALIBRATION_SPEC.tolerance_values == TOLERANCE_VALUES
    assert CALIBRATION_SPEC.target_data == 5
    assert CALIBRATION_SPEC.output_contract.filename == "output.csv"
    assert CALIBRATION_SPEC.output_contract.value_column == "population"


def test_default_configs_are_packaged_with_example_model():
    """Validate calibration defaults use package-local config files."""
    assert DEFAULT_MRP_CONFIG_PATH.name == "example_model.mrp.toml"
    assert DEFAULT_DOCKER_MRP_CONFIG_PATH.name == (
        "example_model.mrp.docker.toml"
    )
    assert DEFAULT_CLOUD_CONFIG_PATH.name == "example_model.cloud_config.toml"
    assert DEFAULT_MRP_CONFIG_PATH.parent.name == "example_model"
    assert DEFAULT_MRP_CONFIG_PATH.is_file()
    assert DEFAULT_DOCKER_MRP_CONFIG_PATH.is_file()
    assert DEFAULT_CLOUD_CONFIG_PATH.is_file()


def test_outputs_to_distance_scores_total_population():
    """Validate the example distance function."""
    assert outputs_to_distance([1, 2, 3], 5) == 1.0


def test_build_perturbation_kernel_uses_example_parameters():
    """Validate the example kernel factory remains callable."""
    kernel = build_perturbation_kernel()

    assert len(kernel.kernels) == 2


def test_reporter_is_configured():
    """Validate the example spec has a model-specific output reporter."""
    assert CALIBRATION_SPEC.output_reporter is not None


def test_priors_include_expected_model_parameters():
    """Validate priors expose only the calibrated model parameters."""
    assert set(PRIORS["priors"]) == {"p", "n"}
    with pytest.raises(KeyError):
        _ = PRIORS["priors"]["seed"]
