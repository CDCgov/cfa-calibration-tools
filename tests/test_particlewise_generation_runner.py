import time
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

from numpy.random import SeedSequence
from rich.console import Console

from calibrationtools.particle import Particle
from calibrationtools.particle_population import ParticlePopulation
from calibrationtools.particlewise_generation_runner import (
    ParticlewiseGenerationConfig,
    ParticlewiseGenerationRequest,
    ParticlewiseGenerationRunner,
)
from calibrationtools.sampler_reporting import SamplerReporter
from calibrationtools.sampler_run_state import SamplerRunState
from calibrationtools.sampler_types import GeneratorSlot


def test_particlewise_generation_runner_sample_particles_until_accepted():
    reporter = SamplerReporter(
        verbose=True,
        console=Console(file=StringIO(), force_terminal=True),
    )
    runner = ParticlewiseGenerationRunner(
        config=ParticlewiseGenerationConfig(
            generation_particle_count=1,
            tolerance_values=[0.5],
            seed_sequence=SeedSequence(123),
            max_attempts_per_proposal=5,
            sample_particle_from_priors=lambda _: Particle({"p": 0.2}),
            sample_and_perturb_particle=lambda _: Particle({"p": 0.8}),
            particle_to_distance=lambda particle, **_: abs(
                particle["p"] - 0.2
            ),
            calculate_weight=lambda _: 1.0,
            replace_particle_population=lambda _: None,
            reporter=reporter,
        ),
        run_state=SamplerRunState(1, False),
    )

    accepted_proposal = runner.sample_particles_until_accepted(
        generator=GeneratorSlot(id=7, seed_sequence=SeedSequence(456)),
        tolerance=0.1,
        sample_method=lambda _: Particle({"p": 0.2}),
        evaluation_kwargs={},
    )

    assert accepted_proposal.slot_id == 7
    assert accepted_proposal.particle == Particle({"p": 0.2})
    assert accepted_proposal.attempts == 1


def test_particlewise_generation_runner_run_generation_records_state():
    stored_populations: list[ParticlePopulation] = []
    run_state = SamplerRunState(1, False)
    reporter = SamplerReporter(
        verbose=True,
        console=Console(file=StringIO(), force_terminal=True),
    )
    runner = ParticlewiseGenerationRunner(
        config=ParticlewiseGenerationConfig(
            generation_particle_count=1,
            tolerance_values=[0.5],
            seed_sequence=SeedSequence(123),
            max_attempts_per_proposal=5,
            sample_particle_from_priors=lambda _: Particle({"p": 0.2}),
            sample_and_perturb_particle=lambda _: Particle({"p": 0.8}),
            particle_to_distance=lambda particle, **_: abs(
                particle["p"] - 0.2
            ),
            calculate_weight=lambda _: 1.0,
            replace_particle_population=stored_populations.append,
            reporter=reporter,
        ),
        run_state=run_state,
    )
    generation_start_time = time.time()

    generation_stats = runner.run_generation(
        ParticlewiseGenerationRequest(
            generation=0,
            n_workers=1,
            parallel_executor=None,
            overall_start_time=generation_start_time,
            generation_start_time=generation_start_time,
            particle_kwargs={},
        )
    )

    assert generation_stats.successes == 1
    assert generation_stats.attempts == 1
    assert run_state.step_successes == [1]
    assert run_state.step_attempts == [1]
    assert len(run_state.generator_history[0]) == 1
    assert stored_populations[0].size == 1


def _build_speculative_runner(
    *,
    particles: list[Particle],
    distances: dict[float, float],
    sleeps: dict[float, float],
) -> tuple[ParticlewiseGenerationRunner, list[ParticlePopulation]]:
    stored_populations: list[ParticlePopulation] = []
    reporter = SamplerReporter(
        verbose=True,
        console=Console(file=StringIO(), force_terminal=True),
    )

    def sample_particle(_):
        return particles.pop(0)

    def particle_to_distance(particle, **_):
        value = particle["p"]
        time.sleep(sleeps.get(value, 0.0))
        return distances[value]

    return (
        ParticlewiseGenerationRunner(
            config=ParticlewiseGenerationConfig(
                generation_particle_count=1,
                tolerance_values=[0.5],
                seed_sequence=SeedSequence(123),
                max_attempts_per_proposal=2,
                sample_particle_from_priors=sample_particle,
                sample_and_perturb_particle=sample_particle,
                particle_to_distance=particle_to_distance,
                calculate_weight=lambda _: 1.0,
                replace_particle_population=stored_populations.append,
                reporter=reporter,
                slot_lookahead=2,
            ),
            run_state=SamplerRunState(1, False),
        ),
        stored_populations,
    )


def test_particlewise_generation_runner_speculative_accepts_first_attempt_in_order():
    runner, stored_populations = _build_speculative_runner(
        particles=[Particle({"p": 0.0}), Particle({"p": 1.0})],
        distances={0.0: 0.0, 1.0: 0.0},
        sleeps={0.0: 0.03, 1.0: 0.0},
    )
    generation_start_time = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        generation_stats = runner.run_generation(
            ParticlewiseGenerationRequest(
                generation=0,
                n_workers=2,
                parallel_executor=executor,
                overall_start_time=generation_start_time,
                generation_start_time=generation_start_time,
                particle_kwargs={},
            )
        )

    assert generation_stats.attempts == 1
    assert stored_populations[0].particles == [Particle({"p": 0.0})]


def test_particlewise_generation_runner_speculative_buffers_until_prior_attempt_rejects():
    runner, stored_populations = _build_speculative_runner(
        particles=[Particle({"p": 0.0}), Particle({"p": 1.0})],
        distances={0.0: 1.0, 1.0: 0.0},
        sleeps={0.0: 0.03, 1.0: 0.0},
    )
    generation_start_time = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        generation_stats = runner.run_generation(
            ParticlewiseGenerationRequest(
                generation=0,
                n_workers=2,
                parallel_executor=executor,
                overall_start_time=generation_start_time,
                generation_start_time=generation_start_time,
                particle_kwargs={},
            )
        )

    assert generation_stats.attempts == 2
    assert stored_populations[0].particles == [Particle({"p": 1.0})]
