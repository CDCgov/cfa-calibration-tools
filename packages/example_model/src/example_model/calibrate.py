"""Calibrate the example branching process."""

import numpy as np
from mrp import Environment

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
def outputs_to_distance(model_output, target_data):
    return abs(np.sum(model_output) - target_data)


sampler = ABCSampler(
    generation_particle_count=500,
    tolerance_values=[5.0, 1.0],
    priors=P,
    perturbation_kernel=K,
    variance_adapter=V,
    outputs_to_distance=outputs_to_distance,
    target_data=5,
    model_runner=model,
    seed=123,  # Propagation of seed must be SeedSequence not int for proper pseudorandom draws
)

sampler.run(default_params=default_inputs)

##===================================#
## Get results
##===================================#
# Print IQR of param1 in the posterior particles
posterior_particles = sampler.get_posterior_particles()
p_values = [p["p"] for p in posterior_particles.particles]
n_values = [p["n"] for p in posterior_particles.particles]

print(
    f"param p(25-75):{np.percentile(p_values, 25)} - {np.percentile(p_values, 75)}"
)
print(
    f"param n(25-75):{np.percentile(n_values, 25)} - {np.percentile(n_values, 75)}"
)
