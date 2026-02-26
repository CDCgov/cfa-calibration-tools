import pytest
from calibrationtools.perturbation_kernel import (
    IndependentKernels,
    MultivariateNormalKernel,
    SeedKernel,
    NormalKernel,
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

class DummyModelRunner:
    def simulate(self, params):
        return 0.5 + params['p']

def particles_to_params(particle):
    return particle

def outputs_to_distance(model_output, target_data):
    return abs(model_output - target_data)

def test_abc_sampler_run(K, P, Vnorm):
    original_std_dev = K.kernels[0].std_dev
    original_seed_kernel = K.kernels[1]
    sampler = ABCSampler(
        generation_particle_count=5,
        tolerance_values=[0.5, 0.1],
        priors=P,
        perturbation_kernel=K,
        variance_adapter=Vnorm,
        particles_to_params=particles_to_params,
        outputs_to_distance=outputs_to_distance,
        target_data=0.75,
        model_runner=DummyModelRunner(),
        seed=123,
    )
    sampler.run()
    posterior_particles = sampler.get_posterior_particles()

    # Assess population handling and updating
    assert(len(sampler.previous_population_archive) == len(sampler.tolerance_values))
    for pop in sampler.previous_population_archive.values():
        assert len(pop.particles) == sampler.generation_particle_count
        assert pop.total_weight == pytest.approx(1.0)
        assert all(p not in posterior_particles.particles for p in pop.particles)

    assert sampler.particle_population == posterior_particles
    assert len(posterior_particles.particles) == sampler.generation_particle_count

    # Test that the perturbation kernel has been updated by adapter Vnorm
    current_perturbation_kernels = sampler._updater.perturbation_kernel.kernels
    assert isinstance(current_perturbation_kernels[0], NormalKernel)
    assert current_perturbation_kernels[0].std_dev < original_std_dev

    assert isinstance(current_perturbation_kernels[1], SeedKernel)
    assert current_perturbation_kernels[1] == original_seed_kernel

def test_abc_sampler_sample_from_priors(K, P, Vnorm):
    def get_sampler() -> ABCSampler:
        return ABCSampler(
            generation_particle_count=5,
            tolerance_values=[0.5, 0.1],
            priors=P,
            perturbation_kernel=K,
            variance_adapter=Vnorm,
            particles_to_params=particles_to_params,
            outputs_to_distance=outputs_to_distance,
            target_data=0.75,
            model_runner=DummyModelRunner(),
            seed=123,
        )
    sampler = get_sampler()
    # Test that sampling from priors works before any population is set
    pop = sampler.sample_particles_from_priors()
    assert pop.size == sampler.generation_particle_count

    # Assert that priors continue to sample from seed sequence for new variants
    pop_small = sampler.sample_particles_from_priors(2)
    assert len(pop_small.particles) == 2
    assert all(p not in pop.particles for p in pop_small.particles)

    # Initiate new sampler to reset seed sequence and assert consistency across runs
    sampler = get_sampler()
    pop_small = sampler.sample_particles_from_priors(2)
    assert all(p in pop.particles for p in pop_small.particles)
    