import threading
import time
from copy import deepcopy

import pytest

import calibrationtools.sampler as sampler_module
from calibrationtools.calibration_results import CalibrationResults
from calibrationtools.particle import Particle
from calibrationtools.perturbation_kernel import (
    IndependentKernels,
    NormalKernel,
    SeedKernel,
)
from calibrationtools.sampler import ABCSampler


class DummyModelRunner:
    def simulate(self, params):
        return 0.5 + params["p"]


class UnpickleableModelRunner:
    def __init__(self):
        self.bad = lambda x: x

    def simulate(self, params):
        return 0.5 + params["p"]


class NonThreadSafeModelRunner:
    def __init__(self):
        self._lock = threading.Lock()

    def simulate(self, params):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent simulate on shared runner")
        try:
            time.sleep(0.01)
            return 0.5 + params["p"]
        finally:
            self._lock.release()


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


@pytest.fixture()
def sampler_with_archive(K, P, Vnorm) -> ABCSampler:
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
        keep_previous_population_data=True,
    )


def test_abc_sampler_run(K, sampler_with_archive: ABCSampler):
    original_std_dev = K.kernels[0].std_dev
    results = sampler_with_archive.run_serial()
    assert isinstance(results, CalibrationResults)
    posterior_particles = results.posterior.particle_population

    # Assert success condition after run
    assert all(
        [
            count == sampler_with_archive.generation_particle_count
            for count in results.smc_step_successes
        ]
    )

    # Assess population handling and updating
    assert (
        len(results.population_archive)
        == len(sampler_with_archive.tolerance_values) - 1
    )
    for pop in results.population_archive.values():
        assert (
            len(pop.particles)
            == sampler_with_archive.generation_particle_count
        )
        assert pop.total_weight == pytest.approx(1.0)
        assert all(
            p not in posterior_particles.particles for p in pop.particles
        )

    assert (
        len(posterior_particles.particles)
        == sampler_with_archive.generation_particle_count
    )

    # Test that the perturbation kernel has been updated by adapter Vnorm
    reset_perturbation = sampler_with_archive._updater.perturbation_kernel
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


def test_sampler_run_does_not_archive_previous_population_by_default(
    sampler: ABCSampler,
):
    results = sampler.run_serial()

    assert results.population_archive == {}


def test_sampler_run_repeatable(sampler):
    # Sampler produces same results when seed is set
    results1 = sampler.run_serial()
    results2 = sampler.run_serial()

    assert results1.point_estimates == results2.point_estimates
    assert results1.ess == results2.ess
    assert results1.acceptance_rates == results2.acceptance_rates


def test_sampler_particle_to_distance_delegates_to_evaluator(sampler):
    class RecordingEvaluator:
        def __init__(self):
            self.calls = []

        def distance(self, particle, **kwargs):
            self.calls.append((particle, kwargs))
            return 1.23

    recording_evaluator = RecordingEvaluator()
    sampler._particle_evaluator = recording_evaluator
    particle = Particle({"p": 0.1, "seed": 0})

    distance = sampler.particle_to_distance(particle, scale=2.0)

    assert distance == 1.23
    assert recording_evaluator.calls == [(particle, {"scale": 2.0})]


def test_sampler_run_resets_internal_run_state(sampler_with_archive):
    results1 = sampler_with_archive.run_serial()
    results2 = sampler_with_archive.run_serial()

    expected_archive_size = len(sampler_with_archive.tolerance_values) - 1

    assert sampler_with_archive.step_successes == [0] * len(
        sampler_with_archive.tolerance_values
    )
    assert sampler_with_archive.step_attempts == [0] * len(
        sampler_with_archive.tolerance_values
    )
    assert sampler_with_archive.generator_history == {}
    assert sampler_with_archive.population_archive == {}
    assert len(results1.population_archive) == expected_archive_size
    assert len(results2.population_archive) == expected_archive_size


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
            assert gen_serial.id == gen_parallel.id
            assert (
                gen_serial.seed_sequence.entropy
                == gen_parallel.seed_sequence.entropy
            )
            assert (
                gen_serial.seed_sequence.spawn_key
                == gen_parallel.seed_sequence.spawn_key
            )


def test_sampler_run_parallel_with_unpickleable_runner(K, P, Vnorm):
    sampler = ABCSampler(
        generation_particle_count=5,
        tolerance_values=[0.5, 0.1],
        priors=P,
        perturbation_kernel=K,
        variance_adapter=Vnorm,
        particles_to_params=particles_to_params,
        outputs_to_distance=outputs_to_distance,
        target_data=0.75,
        model_runner=UnpickleableModelRunner(),
        seed=123,
    )

    results = sampler.run_parallel(max_workers=2)

    assert isinstance(results, CalibrationResults)


def test_sampler_run_parallel_batches_repeatable(sampler):
    results1 = sampler.run_parallel_batches(
        max_workers=2, chunksize=2, batchsize=4
    )
    results2 = sampler.run_parallel_batches(
        max_workers=1, chunksize=1, batchsize=4
    )

    assert results1.point_estimates == results2.point_estimates
    assert results1.ess == results2.ess
    assert results1.acceptance_rates == results2.acceptance_rates
    assert (
        results1.posterior.particle_population.particles
        == results2.posterior.particle_population.particles
    )


def test_sampler_run_parallel_batches_with_unpickleable_runner(K, P, Vnorm):
    sampler = ABCSampler(
        generation_particle_count=5,
        tolerance_values=[0.5, 0.1],
        priors=P,
        perturbation_kernel=K,
        variance_adapter=Vnorm,
        particles_to_params=particles_to_params,
        outputs_to_distance=outputs_to_distance,
        target_data=0.75,
        model_runner=UnpickleableModelRunner(),
        seed=123,
    )

    results = sampler.run_parallel_batches(
        max_workers=2, chunksize=2, batchsize=4
    )

    assert isinstance(results, CalibrationResults)


def test_sampler_parallel_worker_count_default_is_configured(
    K, P, Vnorm, monkeypatch
):
    recorded = {}
    real_executor = sampler_module.ThreadPoolExecutor

    class RecordingExecutor(real_executor):
        def __init__(self, *args, **kwargs):
            recorded["max_workers"] = kwargs.get("max_workers")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        sampler_module, "ThreadPoolExecutor", RecordingExecutor
    )

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
        parallel_worker_count=3,
        seed=123,
    )

    sampler.run_parallel()

    assert recorded["max_workers"] == 3


def test_sampler_parallel_worker_failure_does_not_leak_future_errors(
    K, P, Vnorm, capfd
):
    sampler = ABCSampler(
        generation_particle_count=5,
        tolerance_values=[0.5],
        priors=P,
        perturbation_kernel=K,
        variance_adapter=Vnorm,
        particles_to_params=particles_to_params,
        outputs_to_distance=outputs_to_distance,
        target_data=0.75,
        model_runner=NonThreadSafeModelRunner(),
        seed=123,
        verbose=False,
    )

    with pytest.raises(
        RuntimeError, match="concurrent simulate on shared runner"
    ):
        sampler.run_parallel(max_workers=2)

    captured = capfd.readouterr()
    assert "Future exception was never retrieved" not in captured.err


def test_sampler_verbose_false_suppresses_output(K, P, Vnorm, capfd):
    sampler = ABCSampler(
        generation_particle_count=5,
        tolerance_values=[0.5],
        priors=P,
        perturbation_kernel=K,
        variance_adapter=Vnorm,
        particles_to_params=particles_to_params,
        outputs_to_distance=outputs_to_distance,
        target_data=0.75,
        model_runner=DummyModelRunner(),
        seed=123,
        verbose=False,
    )

    sampler.run_serial()

    captured = capfd.readouterr()
    assert captured.out == ""
