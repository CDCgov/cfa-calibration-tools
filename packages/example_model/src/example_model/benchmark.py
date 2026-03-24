"""Calibrate the example branching process."""

import json
import timeit

import numpy as np
from mrp import Environment
from mrp.api import apply_dict_overrides

from calibrationtools.perturbation_kernel import (
    IndependentKernels,
    MultivariateNormalKernel,
    SeedKernel,
)
from calibrationtools.sampler import ABCSampler
from calibrationtools.variance_adapter import AdaptMultivariateNormalVariance
from example_model import Binom_BP_Model

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
default_inputs = {
    "seed": 123,
    "max_gen": 15,
    "n": 3,
    "p": 0.5,
    "max_infect": 500,
}
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
    generation_particle_count=500,
    tolerance_values=[5.0, 4.0, 3.0],
    priors=P,
    perturbation_kernel=K,
    variance_adapter=V,
    particles_to_params=particles_to_params,
    outputs_to_distance=outputs_to_distance,
    target_data=5,
    model_runner=model,
    seed=123,  # Propagation of seed must be SeedSequence not int for proper pseudorandom draws
)

benchmark_results = []

start = timeit.default_timer()
results = sampler.run(execution="serial", base_inputs=default_inputs)
end = timeit.default_timer()
print(f"Execution time: {end - start} seconds")
benchmark_results.append(
    {
        "time": end - start,
        "attempts": results.smc_step_attempts,
        "max_workers": None,
        "chunksize": None,
    }
)

for max_workers in [8, 2, 1]:
    # for chunksize in [8, 128]:
    start = timeit.default_timer()
    results = sampler.run(
        execution="parallel",
        base_inputs=default_inputs,
        # chunksize=1,
        max_workers=max_workers,
    )
    end = timeit.default_timer()
    print(f"Execution time: {end - start} seconds")

    benchmark_results.append(
        {
            "time": end - start,
            "attempts": results.smc_step_attempts,
            # "chunksize": chunksize,
            "max_workers": max_workers,
        }
    )


# Defualt printed output is the CalibrationResults object, which includes ESS, acceptance rates, and parameter details
for result in benchmark_results:
    print(f"workers: {result['max_workers']}, time: {result['time']}")

with open("./benchmarks/parallelization_check.json", "w") as fp:
    json.dump(benchmark_results, fp)
