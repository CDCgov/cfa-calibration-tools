import json
import threading
import time
from copy import deepcopy
from pathlib import Path

import pytest
from numpy.random import SeedSequence

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


class FileAwareModelRunner:
    def __init__(self):
        self.calls: list[tuple[str, Path, Path]] = []

    def simulate(
        self,
        params,
        *,
        input_path=None,
        output_dir=None,
        run_id=None,
    ):
        assert input_path is not None
        assert output_dir is not None
        assert run_id is not None

        payload = json.loads(Path(input_path).read_text())
        assert payload["run_id"] == run_id
        assert payload == params

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "runner.json").write_text(
            json.dumps({"run_id": run_id})
        )
        self.calls.append((run_id, Path(input_path), Path(output_dir)))
        return 0.75


class AsyncPreferredModelRunner:
    prefer_simulate_async = True

    def __init__(self):
        self.calls = 0
        self.active = 0
        self.max_active = 0

    def simulate(self, params, **kwargs):
        raise AssertionError("simulate() should not be used")

    def simulate_from_sync(self, params, **kwargs):
        raise AssertionError("simulate_from_sync() should not be used")

    async def simulate_async(self, params, **kwargs):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await sampler_module.asyncio.sleep(0)
            return 0.75
        finally:
            self.active -= 1


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
        entropy=123,
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
        entropy=123,
        keep_previous_population_data=True,
    )


def test_abc_sampler_run(K, sampler_with_archive: ABCSampler):
    original_std_dev = K.kernels[0].std_dev
    results = sampler_with_archive.run_serial()
    assert isinstance(results, CalibrationResults)
    posterior_particles = results.posterior.particle_population

    assert all(
        [
            count == sampler_with_archive.generation_particle_count
            for count in results.smc_step_successes
        ]
    )

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


def test_sample_from_priors(sampler, seed_sequence):
    states = sampler.sample_priors(5, seed_sequence)
    assert len(states) == 5

    pop_small = sampler.sample_priors(2, seed_sequence)
    assert len(pop_small) == 2
    assert all(p not in states for p in pop_small)


def test_sample_from_priors_repeatable(sampler, seed_sequence):
    def get_sampler():
        return deepcopy(sampler)

    sampler1 = get_sampler()
    sampler2 = get_sampler()
    new_sequence = SeedSequence(seed_sequence.entropy)

    states1 = sampler1.sample_priors(5, seed_sequence)
    states2 = sampler2.sample_priors(5, new_sequence)

    assert states1 == states2


def test_sampler_run_parallel_equal(sampler: ABCSampler):
    results_serial = sampler.run_serial()
    results_parallel = sampler.run_parallel()

    assert results_serial.point_estimates == results_parallel.point_estimates
    assert results_serial.ess == results_parallel.ess
    assert results_serial.acceptance_rates == results_parallel.acceptance_rates
    assert (
        results_serial.posterior.particle_population.particles
        == results_parallel.posterior.particle_population.particles
    )
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
        entropy=123,
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
        entropy=123,
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
        entropy=123,
    )

    sampler.run_parallel()

    assert recorded["max_workers"] == 3


def test_sampler_max_concurrent_simulations_alias_sets_default_worker_count(
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
        max_concurrent_simulations=2,
        entropy=123,
    )

    sampler.run_parallel()

    assert recorded["max_workers"] == 2
    assert sampler.max_concurrent_simulations == 2


def test_sampler_async_preferred_runner_avoids_thread_pool_fanout(
    K, P, Vnorm, monkeypatch
):
    def fail_thread_pool(*args, **kwargs):
        raise AssertionError("ThreadPoolExecutor should not be used")

    monkeypatch.setattr(sampler_module, "ThreadPoolExecutor", fail_thread_pool)

    runner = AsyncPreferredModelRunner()
    sampler = ABCSampler(
        generation_particle_count=5,
        tolerance_values=[0.5],
        priors=P,
        perturbation_kernel=K,
        variance_adapter=Vnorm,
        particles_to_params=particles_to_params,
        outputs_to_distance=outputs_to_distance,
        target_data=0.75,
        model_runner=runner,
        max_concurrent_simulations=2,
        entropy=123,
        verbose=False,
    )

    sampler.run_parallel()

    assert runner.calls == 5
    assert runner.max_active <= 2


def test_sampler_rejects_invalid_max_concurrent_simulations(K, P, Vnorm):
    with pytest.raises(
        ValueError,
        match="max_concurrent_simulations must be at least 1",
    ):
        ABCSampler(
            generation_particle_count=4,
            tolerance_values=[0.5],
            priors=P,
            perturbation_kernel=K,
            variance_adapter=Vnorm,
            particles_to_params=particles_to_params,
            outputs_to_distance=outputs_to_distance,
            target_data=0.75,
            model_runner=DummyModelRunner(),
            max_concurrent_simulations=0,
            entropy=123,
            verbose=False,
        )


def test_sampler_stages_simulation_inputs_and_outputs(tmp_path, K, P, Vnorm):
    runner = FileAwareModelRunner()
    sampler = ABCSampler(
        generation_particle_count=4,
        tolerance_values=[0.1],
        priors=P,
        perturbation_kernel=K,
        variance_adapter=Vnorm,
        particles_to_params=particles_to_params,
        outputs_to_distance=outputs_to_distance,
        target_data=0.75,
        model_runner=runner,
        max_concurrent_simulations=2,
        entropy=123,
        verbose=False,
        artifacts_dir=tmp_path,
    )

    sampler.run_serial()

    assert [call[0] for call in runner.calls] == [
        "gen_0_particle_0_attempt_0",
        "gen_0_particle_1_attempt_0",
        "gen_0_particle_2_attempt_0",
        "gen_0_particle_3_attempt_0",
    ]

    for run_id, input_path, output_dir in runner.calls:
        assert input_path == (
            tmp_path / "input" / "generation-0" / f"{run_id}.json"
        )
        assert output_dir == (tmp_path / "output" / "generation-0" / run_id)
        assert json.loads(input_path.read_text())["run_id"] == run_id
        assert json.loads((output_dir / "result.json").read_text()) == 0.75
        assert json.loads((output_dir / "runner.json").read_text()) == {
            "run_id": run_id
        }


def test_sampler_get_posterior_particles_uses_last_results(sampler):
    results = sampler.run_serial()

    assert sampler.get_posterior_particles() == results.posterior_particles


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
        entropy=123,
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
        entropy=123,
        verbose=False,
    )

    sampler.run_serial()

    captured = capfd.readouterr()
    assert captured.out == ""


def test_results_inherit_entropy(K, P, Vnorm, seed_sequence):
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
        entropy=seed_sequence.entropy,
    )

    results = sampler.run_serial()

    assert results.seed_sequence.entropy == seed_sequence.entropy
    assert results.seed_sequence.spawn_key == sampler._seed_sequence.spawn_key


def test_results_dont_inherit_entropy(K, P, Vnorm, seed_sequence):
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
        entropy=seed_sequence.entropy,
        results_inherit_entropy_only=False,
    )

    results = sampler.run_serial()

    assert results.seed_sequence.entropy == seed_sequence.entropy
    assert results.seed_sequence.spawn_key != sampler._seed_sequence.spawn_key
