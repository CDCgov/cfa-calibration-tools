import numpy as np

from calibrationtools.perturbation_kernel import (
    IndependentKernels,
    MultivariateNormalKernel,
    SeedKernel,
)
from calibrationtools.prior_distribution import (
    ExponentialPrior,
    IndependentPriors,
    LogNormalPrior,
    NormalPrior,
    SeedPrior,
    UniformPrior,
)
from calibrationtools.sampler import ABCSampler
from calibrationtools.variance_adapter import AdaptMultivariateNormalVariance

P = IndependentPriors(
    [
        UniformPrior("param1", 0, 1),
        NormalPrior("param2", 0, 1),
        LogNormalPrior("param3", 0, 1),
        ExponentialPrior("param4", 1),
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


class DummyModelRunner:
    def simulate(self, params):
        return (
            params["param1"]
            + params["param2"]
            + params["param3"]
            + params["param4"]
        )


def particles_to_params(particle):
    return particle.state


def outputs_to_distance(model_output, target_data):
    return abs(model_output - target_data)


sampler = ABCSampler(
    generation_particle_count=500,
    tolerance_values=[5.0, 0.5, 0.1],
    priors=P,
    perturbation_kernel=K,
    variance_adapter=V,
    particles_to_params=particles_to_params,
    outputs_to_distance=outputs_to_distance,
    target_data=0.5,
    model_runner=DummyModelRunner(),
    seed=123,
)

sampler.run()

# Print IQR of param1 in the posterior particles
posterior_particles = sampler.get_posterior_particles()
param1_values = [p.state["param1"] for p in posterior_particles.all_particles]
print(posterior_particles)
print(np.percentile(param1_values, 25), np.percentile(param1_values, 75))
