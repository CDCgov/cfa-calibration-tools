"""Calibrate the example branching process."""

import argparse
from pathlib import Path

import numpy as np
from mrp.api import apply_dict_overrides

from calibrationtools.perturbation_kernel import (
    IndependentKernels,
    MultivariateNormalKernel,
    SeedKernel,
)
from calibrationtools.sampler import ABCSampler
from calibrationtools.variance_adapter import AdaptMultivariateNormalVariance
from example_model import (
    DEFAULT_DOCKER_MRP_CONFIG_PATH,
    Binom_BP_Model,
    ExampleModelMRPRunner,
)

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

def particles_to_params(particle, **kwargs):
    base_inputs = kwargs.get("base_inputs")
    return apply_dict_overrides(base_inputs, particle)


def outputs_to_distance(model_output, target_data):
    return abs(np.sum(model_output) - target_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ABC-SMC calibration for the example model."
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run each simulation through the Docker-backed MRP config.",
    )
    parser.add_argument(
        "--mrp-config",
        type=Path,
        help="Run simulations through the given MRP config path.",
    )
    parser.add_argument(
        "--max-concurrent-simulations",
        type=int,
        default=10,
        help="Maximum number of simulations to evaluate at once.",
    )
    return parser.parse_args()


def resolve_model_runner(args: argparse.Namespace):
    if args.mrp_config is not None:
        return ExampleModelMRPRunner(args.mrp_config)
    if args.docker:
        return ExampleModelMRPRunner(DEFAULT_DOCKER_MRP_CONFIG_PATH)
    return Binom_BP_Model


def run_calibration(
    *,
    model_runner,
    max_concurrent_simulations: int = 10,
):
    kernel = IndependentKernels(
        [
            MultivariateNormalKernel(
                [parameter for parameter in PRIORS["priors"]]
            ),
            SeedKernel("seed"),
        ]
    )
    variance_adapter = AdaptMultivariateNormalVariance()
    sampler = ABCSampler(
        generation_particle_count=500,
        tolerance_values=[5.0, 1.0],
        priors=PRIORS,
        perturbation_kernel=kernel,
        variance_adapter=variance_adapter,
        particles_to_params=particles_to_params,
        outputs_to_distance=outputs_to_distance,
        target_data=5,
        model_runner=model_runner,
        max_concurrent_simulations=max_concurrent_simulations,
        seed=123,  # Propagation of seed must be SeedSequence not int for proper pseudorandom draws
    )
    sampler.run(base_inputs=DEFAULT_INPUTS)

    posterior_particles = sampler.get_posterior_particles()
    p_values = [particle["p"] for particle in posterior_particles.particles]
    n_values = [particle["n"] for particle in posterior_particles.particles]

    print(
        f"param p(25-75):{np.percentile(p_values, 25)} - {np.percentile(p_values, 75)}"
    )
    print(
        f"param n(25-75):{np.percentile(n_values, 25)} - {np.percentile(n_values, 75)}"
    )


def main():
    args = parse_args()
    run_calibration(
        model_runner=resolve_model_runner(args),
        max_concurrent_simulations=args.max_concurrent_simulations,
    )


if __name__ == "__main__":
    main()
