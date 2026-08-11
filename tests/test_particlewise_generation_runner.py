import time
from io import StringIO

from numpy.random import SeedSequence
from rich.console import Console

from calibrationtools.cloud_executor import (
    CloudAcceptanceResult,
    CloudAcceptanceTask,
    CloudExecutor,
    run_cloud_acceptance_task,
)
from calibrationtools.particle import Particle
from calibrationtools.particle_population import ParticlePopulation
from calibrationtools.particlewise_generation_runner import (
    ParticlewiseGenerationConfig,
    ParticlewiseGenerationRequest,
    ParticlewiseGenerationRunner,
)
from calibrationtools.sampler_reporting import SamplerReporter
from calibrationtools.sampler_run_state import SamplerRunState
from calibrationtools.sampler_types import GeneratorSlot, ProgressEvent


def cloud_sample(_: SeedSequence | None) -> Particle:
    return Particle({"p": 0.2})


def cloud_distance(particle: Particle, **_: object) -> float:
    return abs(particle["p"] - 0.2)


class ReverseCloudExecutor(CloudExecutor):
    async def execute_tasks(
        self,
        tasks: list[CloudAcceptanceTask],
        *,
        progress_callback=None,
    ) -> list[CloudAcceptanceResult]:
        results = [run_cloud_acceptance_task(task) for task in reversed(tasks)]
        if progress_callback is not None:
            progress_callback(
                ProgressEvent(
                    event_type="executor_message",
                    payload={"message": "fake cloud complete"},
                )
            )
        return results


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


def test_particlewise_generation_runner_uses_cloud_results_in_slot_order():
    stored_populations: list[ParticlePopulation] = []
    events: list[ProgressEvent] = []
    run_state = SamplerRunState(1, False)
    reporter = SamplerReporter(
        verbose=True,
        console=Console(file=StringIO(), force_terminal=True),
    )
    runner = ParticlewiseGenerationRunner(
        config=ParticlewiseGenerationConfig(
            generation_particle_count=2,
            tolerance_values=[0.5],
            seed_sequence=SeedSequence(123),
            max_attempts_per_proposal=5,
            sample_particle_from_priors=cloud_sample,
            sample_and_perturb_particle=cloud_sample,
            particle_to_distance=cloud_distance,
            calculate_weight=lambda _: 1.0,
            replace_particle_population=stored_populations.append,
            reporter=reporter,
        ),
        run_state=run_state,
    )
    started = time.time()

    stats = runner.run_generation(
        ParticlewiseGenerationRequest(
            generation=0,
            n_workers=1,
            parallel_executor=None,
            overall_start_time=started,
            generation_start_time=started,
            particle_kwargs={},
            cloud_executor=ReverseCloudExecutor(),
            progress_callback=events.append,
        )
    )

    assert stats.successes == 2
    assert [entry["slot_id"] for entry in run_state.distance_history[0]] == [
        0,
        1,
    ]
    assert [event.event_type for event in events] == [
        "generation_started",
        "executor_message",
        "work_progressed",
        "work_progressed",
    ]


def test_particlewise_generation_runner_rejects_duplicate_cloud_results():
    task = CloudAcceptanceTask(
        slot_id=0,
        seed_sequence=SeedSequence(1),
        tolerance=1.0,
        max_attempts=1,
        sample_method=cloud_sample,
        particle_to_distance=cloud_distance,
    )
    result = CloudAcceptanceResult(
        slot_id=0,
        particle=Particle({"p": 0.2}),
        distance=0.0,
        attempts=1,
        status="accepted",
    )

    try:
        ParticlewiseGenerationRunner._validate_cloud_results(
            [task], [result, result]
        )
    except RuntimeError as exc:
        assert "duplicate slots" in str(exc)
    else:  # pragma: no cover - explicit failure message
        raise AssertionError("duplicate cloud results must be rejected")
