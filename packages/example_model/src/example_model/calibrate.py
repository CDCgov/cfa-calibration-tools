"""Calibrate the example branching process."""

import numpy as np
from mrp import Environment
from mrp.api import apply_dict_overrides

from calibrationtools.perturbation_kernel import (
    IndependentKernels,
    MultivariateNormalKernel,
    SeedKernel,
)
from calibrationtools.prior_distribution import (
    IndependentPriors,
    SeedPrior,
    UniformPrior,
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
P = IndependentPriors(
    [
        UniformPrior("n", 0, 5),
        UniformPrior("p", 0, 1),
        SeedPrior("seed"),
    ]
)

K = IndependentKernels(
    [
        MultivariateNormalKernel(
            [p.params[0] for p in P.priors if not isinstance(p, SeedPrior)],
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
    tolerance_values=[5.0, 1.0],
    priors=P,
    perturbation_kernel=K,
    variance_adapter=V,
    particles_to_params=particles_to_params,
    outputs_to_distance=outputs_to_distance,
    target_data=5,
    model_runner=model,
    seed=123,  # Propagation of seed must be SeedSequence not int for proper pseudorandom draws
)

sampler.run(base_inputs=default_inputs)

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
