"""Calibrate the example branching process."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from calibrationtools.calibration_app import (
    CalibrationAppSpec,
    CSVOutputContract,
    run_calibration_app,
)
from calibrationtools.perturbation_kernel import (
    IndependentKernels,
    MultivariateNormalKernel,
    SeedKernel,
)
from calibrationtools.variance_adapter import AdaptMultivariateNormalVariance

from .direct_runner import ExampleModelDirectRunner

DEFAULT_INPUTS = {
    "seed": 123,
    "max_gen": 15,
    "n": 3,
    "p": 0.5,
    "max_infect": 500,
}

PRIORS = {
    "priors": {
        "p": {
            "distribution": "uniform",
            "parameters": {"min": 0.0, "max": 1.0},
        },
        "n": {
            "distribution": "uniform",
            "parameters": {"min": 0.0, "max": 5.0},
        },
    }
}
TOLERANCE_VALUES = [5.0, 1.0]
DEFAULT_MAX_CONCURRENT_SIMULATIONS = 10
DEFAULT_CLOUD_MAX_CONCURRENT_SIMULATIONS = 50
DEFAULT_ARTIFACTS_DIR = Path("artifacts")
_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MRP_CONFIG_PATH = _PACKAGE_DIR / "example_model.mrp.toml"
DEFAULT_DOCKER_MRP_CONFIG_PATH = _PACKAGE_DIR / "example_model.mrp.docker.toml"
DEFAULT_CLOUD_CONFIG_PATH = _PACKAGE_DIR / "example_model.cloud_config.toml"


def outputs_to_distance(model_output: list[int], target_data: int) -> float:
    """Return absolute distance between simulated and target population."""
    return float(abs(np.sum(model_output) - target_data))


def build_perturbation_kernel() -> IndependentKernels:
    """Build the example model's ABC-SMC perturbation kernel."""
    return IndependentKernels(
        [
            MultivariateNormalKernel(
                [parameter for parameter in PRIORS["priors"]]
            ),
            SeedKernel("seed"),
        ]
    )


def report_results(results: Any) -> None:
    """Print the example model's calibration summary."""
    print(results)
    posterior_particles = results.posterior_particles
    p_values = [particle["p"] for particle in posterior_particles.particles]
    n_values = [particle["n"] for particle in posterior_particles.particles]

    print(
        f"param p(25-75):{np.percentile(p_values, 25)} - {np.percentile(p_values, 75)}"
    )
    print(
        f"param n(25-75):{np.percentile(n_values, 25)} - {np.percentile(n_values, 75)}"
    )


CALIBRATION_SPEC = CalibrationAppSpec(
    default_inputs=DEFAULT_INPUTS,
    priors=PRIORS,
    tolerance_values=TOLERANCE_VALUES,
    target_data=5,
    outputs_to_distance=outputs_to_distance,
    direct_runner_factory=ExampleModelDirectRunner,
    output_contract=CSVOutputContract(
        filename="output.csv",
        value_column="population",
        value_parser=int,
        header_fields=("generation", "population"),
    ),
    perturbation_kernel_factory=build_perturbation_kernel,
    variance_adapter_factory=AdaptMultivariateNormalVariance,
    output_reporter=report_results,
    default_mrp_config_path=DEFAULT_MRP_CONFIG_PATH,
    default_docker_mrp_config_path=DEFAULT_DOCKER_MRP_CONFIG_PATH,
    default_cloud_config_path=DEFAULT_CLOUD_CONFIG_PATH,
    generation_particle_count=500,
    cloud_default_concurrency=DEFAULT_CLOUD_MAX_CONCURRENT_SIMULATIONS,
    local_default_concurrency=DEFAULT_MAX_CONCURRENT_SIMULATIONS,
    default_artifacts_dir=DEFAULT_ARTIFACTS_DIR,
    entropy=123,
)


def main(argv: list[str] | None = None) -> None:
    """Run the example-model calibration CLI."""
    run_calibration_app(argv, CALIBRATION_SPEC)


if __name__ == "__main__":
    main()
