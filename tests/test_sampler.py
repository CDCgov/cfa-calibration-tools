from copy import deepcopy

import pytest

from calibrationtools.calibration_results import CalibrationResults
from calibrationtools.perturbation_kernel import (
    IndependentKernels,
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


def test_abc_sampler_run(K, sampler: ABCSampler):
    original_std_dev = K.kernels[0].std_dev
    results = sampler.run_serial()
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
    reset_perturbation = sampler._updater.perturbation_kernel
    assert isinstance(reset_perturbation, IndependentKernels)
    reset_perturbation_kernels = reset_perturbation.kernels

    posterior_perturbation = results._updater.perturbation_kernel
    assert isinstance(posterior_perturbation, IndependentKernels)
    posterior_perturbation_kernels = posterior_perturbation.kernels

    assert isinstance(reset_perturbation_kernels[0], NormalKernel)
    assert isinstance(posterior_perturbation_kernels[0], NormalKernel)
    assert reset_perturbation_kernels[0].std_dev == original_std_dev
    assert posterior_perturbation_kernels[0].std_dev < original_std_dev

    assert isinstance(reset_perturbation_kernels[1], SeedKernel)
    assert isinstance(posterior_perturbation_kernels[1], SeedKernel)


def test_sampler_run_repeatable(sampler):
    # Sampler produces same results when seed is set
    results1 = sampler.run_serial()
    results2 = sampler.run_serial()

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


def test_sampler_run_parallel_equal(sampler: ABCSampler):
    # Test that parallel and serial runs produce similar results
    results_serial = sampler.run_serial()
    results_parallel = sampler.run_parallel()

    assert results_serial.point_estimates == results_parallel.point_estimates
    assert results_serial.ess == results_parallel.ess
    assert results_serial.acceptance_rates == results_parallel.acceptance_rates
    assert (
        results_serial.posterior.particle_population.particles
        == results_parallel.posterior.particle_population.particles
    )
    # Assert that the particle generator key history is the same for both runs
    for generation, generator_list in results_serial.generator_history.items():
        parallel_generator_list = results_parallel.generator_history[
            generation
        ]
        for gen_serial, gen_parallel in zip(
            generator_list, parallel_generator_list
        ):
            assert gen_serial["id"] == gen_parallel["id"]
            assert (
                gen_serial["seed_sequence"].entropy
                == gen_parallel["seed_sequence"].entropy
            )
            assert (
                gen_serial["seed_sequence"].spawn_key
                == gen_parallel["seed_sequence"].spawn_key
            )
