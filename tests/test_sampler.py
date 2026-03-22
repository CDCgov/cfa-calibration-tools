from copy import deepcopy
import json
import threading
import time
from pathlib import Path

import pytest

from calibrationtools.perturbation_kernel import (
    NormalKernel,
    SeedKernel,
)
from calibrationtools.sampler import ABCSampler


class DummyModelRunner:
    def simulate(self, params):
        return 0.5 + params["p"]


class TrackingModelRunner:
    def __init__(self, delay: float = 0.01):
        self.delay = delay
        self._active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def simulate(self, params):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(self.delay)
            return 0.5 + params["p"]
        finally:
            with self._lock:
                self._active -= 1


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
    original_seed_kernel = K.kernels[1]
    sampler.run()
    posterior_particles = sampler.get_posterior_particles()

    # Assert success condition after run
    assert all(
        [
            count == sampler.generation_particle_count
            for count in sampler.smc_step_successes
        ]
    )

    # Assess population handling and updating
    assert (
        len(sampler.population_archive) == len(sampler.tolerance_values)
    ) - 1
    for pop in sampler.population_archive.values():
        assert len(pop.particles) == sampler.generation_particle_count
        assert pop.total_weight == pytest.approx(1.0)
        assert all(
            p not in posterior_particles.particles for p in pop.particles
        )

    assert sampler.particle_population == posterior_particles
    assert (
        len(posterior_particles.particles) == sampler.generation_particle_count
    )

    # Test that the perturbation kernel has been updated by adapter Vnorm
    current_perturbation_kernels = sampler._updater.perturbation_kernel.kernels
    assert isinstance(current_perturbation_kernels[0], NormalKernel)
    assert current_perturbation_kernels[0].std_dev < original_std_dev

    assert isinstance(current_perturbation_kernels[1], SeedKernel)
    assert current_perturbation_kernels[1] == original_seed_kernel


def test_get_posterior_particles_before_run(sampler):
    with pytest.raises(
        ValueError,
        match="Posterior population is not fully populated. Please run the sampler to completion before accessing the posterior population.",
    ):
        sampler.get_posterior_particles()


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


def test_abc_sampler_limits_parallel_simulations(K, P, Vnorm):
    tracker = TrackingModelRunner()
    sampler = ABCSampler(
        generation_particle_count=4,
        tolerance_values=[0.5],
        priors=P,
        perturbation_kernel=K,
        variance_adapter=Vnorm,
        particles_to_params=particles_to_params,
        outputs_to_distance=outputs_to_distance,
        target_data=0.75,
        model_runner=tracker,
        max_concurrent_simulations=2,
        seed=123,
        verbose=False,
    )

    sampler.run()

    assert tracker.max_active == 2
    assert sampler.get_posterior_particles().size == 4


def test_abc_sampler_rejects_invalid_parallelism(K, P, Vnorm):
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
            seed=123,
            verbose=False,
        )


def test_abc_sampler_stages_simulation_inputs_and_outputs(
    tmp_path, K, P, Vnorm
):
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
        seed=123,
        verbose=False,
        artifacts_dir=tmp_path,
    )

    sampler.run()

    assert [call[0] for call in runner.calls] == [
        "gen-1_particle-1",
        "gen-1_particle-2",
        "gen-1_particle-3",
        "gen-1_particle-4",
    ]

    for run_id, input_path, output_dir in runner.calls:
        assert input_path == (
            tmp_path / "input" / "generation-1" / f"{run_id}.json"
        )
        assert output_dir == (
            tmp_path / "output" / "generation-1" / run_id
        )
        assert json.loads(input_path.read_text())["run_id"] == run_id
        assert json.loads((output_dir / "result.json").read_text()) == 0.75
        assert json.loads((output_dir / "runner.json").read_text()) == {
            "run_id": run_id
        }
