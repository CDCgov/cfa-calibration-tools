from copy import deepcopy

import pytest

from calibrationtools.calibration_results import CalibrationResults
from calibrationtools.perturbation_kernel import (
    NormalKernel,
    SeedKernel,
)
from calibrationtools.sampler import ABCSampler


class DummyModelRunner:
    def simulate(self, params):
        return 0.5 + params["p"]


def particles_to_params(particle):
    return particle


def outputs_to_distance(model_output, target_data):
    return abs(model_output - target_data)


@pytest.fixture()
def sampler(K, P, Vnorm) -> ABCSampler:
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


def test_abc_sampler_run(K, sampler):
    original_std_dev = K.kernels[0].std_dev
    results = sampler.run()
    assert isinstance(results, CalibrationResults)
    posterior_particles = results.posterior.particle_population

    # Assert success condition after run
    assert all(
        [
            count == sampler.generation_particle_count
            for count in results.smc_step_successes
        ]
    )

    # Assess population handling and updating
    assert (
        len(results.population_archive) == len(sampler.tolerance_values)
    ) - 1
    for pop in results.population_archive.values():
        assert len(pop.particles) == sampler.generation_particle_count
        assert pop.total_weight == pytest.approx(1.0)
        assert all(
            p not in posterior_particles.particles for p in pop.particles
        )

    assert (
        len(posterior_particles.particles) == sampler.generation_particle_count
    )

    # Test that the perturbation kernel has been updated by adapter Vnorm
    current_perturbation_kernels = sampler._updater.perturbation_kernel.kernels
    posterior_perturbation_kernels = (
        results._updater.perturbation_kernel.kernels
    )
    assert isinstance(current_perturbation_kernels[0], NormalKernel)
    assert current_perturbation_kernels[0].std_dev == original_std_dev
    assert posterior_perturbation_kernels[0].std_dev < original_std_dev

    assert isinstance(current_perturbation_kernels[1], SeedKernel)
    assert isinstance(posterior_perturbation_kernels[1], SeedKernel)


def test_sampler_run_repeatable(sampler):
    # Sampler produces same results when seed is set
    results1 = sampler.run()
    results2 = sampler.run()

    assert results1.point_estimates == results2.point_estimates
    assert results1.ess == results2.ess
    assert results1.acceptance_rates == results2.acceptance_rates


def test_sample_from_priors(sampler):
    # Test that sampling from priors works before any population is set
    states = sampler.sample_priors(5)
    assert len(states) == 5

    # Assert that priors continue to sample from seed sequence for new variants
    pop_small = sampler.sample_priors(2)
    assert len(pop_small) == 2
    assert all(p not in states for p in pop_small)


def test_sample_from_priors_repeatable(sampler):
    # Sampler reproduces same samples from priors when seed is set
    def get_sampler():
        return deepcopy(sampler)

    sampler1 = get_sampler()
    sampler2 = get_sampler()

    states1 = sampler1.sample_priors(5)
    states2 = sampler2.sample_priors(5)

    assert states1 == states2
