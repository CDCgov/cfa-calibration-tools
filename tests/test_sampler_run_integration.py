"""Run each calibration method for the example branching process."""

import numpy as np
import pytest
from example_model import Binom_BP_Model
from mrp import Environment
from mrp.api import apply_dict_overrides

from calibrationtools.perturbation_kernel import (
    IndependentKernels,
    MultivariateNormalKernel,
    SeedKernel,
)
from calibrationtools.sampler import ABCSampler
from calibrationtools.variance_adapter import AdaptMultivariateNormalVariance


@pytest.fixture()
def example_model_defaults() -> dict:
    return {
        "seed": 123,
        "max_gen": 15,
        "n": 3,
        "p": 0.5,
        "max_infect": 500,
    }


@pytest.fixture()
def example_model_sampler() -> ABCSampler:
    ##===================================#
    ## Define model
    ##===================================#
    env = Environment(
        {
            "input": {
                "seed": 123,
                "max_gen": 15,
                "n": 3,
                "p": 0.5,
                "max_infect": 500,
            },
            "output": {"spec": "filesystem", "dir": "./output"},
        }
    )
    model = Binom_BP_Model(env=env)

    ##===================================#
    ## Define priors
    ##===================================#
    P = {
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

    K = IndependentKernels(
        [
            MultivariateNormalKernel(
                [p for p in P["priors"].keys()],
            ),
            SeedKernel("seed"),
        ]
    )

    V = AdaptMultivariateNormalVariance()

    ##===================================#
    ## Run ABC-SMC
    ##===================================#
    def particles_to_params(particle, **kwargs):
        base_inputs = kwargs.get("base_inputs")
        model_params = apply_dict_overrides(base_inputs, particle)
        return model_params

    def outputs_to_distance(model_output, target_data):
        return abs(np.sum(model_output) - target_data)

    sampler = ABCSampler(
        generation_particle_count=15,
        tolerance_values=[50.0, 10.0],
        priors=P,
        perturbation_kernel=K,
        variance_adapter=V,
        particles_to_params=particles_to_params,
        outputs_to_distance=outputs_to_distance,
        target_data=5,
        model_runner=model,
        entropy=0x60636577C7AD93BBE463F30A6241FDE4,
    )
    return sampler


def test_sampler_run_integration(
    example_model_sampler, example_model_defaults
):
    results_serial = example_model_sampler.run(
        execution="serial", base_inputs=example_model_defaults
    )
    results_parallel = example_model_sampler.run(
        execution="parallel", base_inputs=example_model_defaults
    )

    assert (
        results_serial.posterior_particles.ess
        == results_parallel.posterior_particles.ess
    )


def test_sampler_run_parallel_batches_integration(
    example_model_sampler, example_model_defaults
):
    results_parallel_batches = example_model_sampler.run_parallel_batches(
        base_inputs=example_model_defaults
    )
    assert results_parallel_batches.posterior_particles.ess > 0
