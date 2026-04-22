import time
from io import StringIO

from numpy.random import SeedSequence
from rich.console import Console

from calibrationtools.batch_generation_runner import (
    BatchGenerationConfig,
    BatchGenerationRequest,
    BatchGenerationRunner,
    BatchGenerationState,
)
from calibrationtools.particle import Particle
from calibrationtools.particle_population import ParticlePopulation
from calibrationtools.sampler_reporting import SamplerReporter
from calibrationtools.sampler_run_state import SamplerRunState


def test_batch_generation_runner_accepts_equal_tolerance():
    reporter = SamplerReporter(
        verbose=True,
        console=Console(file=StringIO(), force_terminal=True),
    )
    runner = BatchGenerationRunner(
        config=BatchGenerationConfig(
            generation_particle_count=1,
            tolerance_values=[0.5],
            seed_sequence=SeedSequence(123),
            max_proposals_per_batch=10_000,
            sample_particle_from_priors=lambda _: Particle({"p": 0.25}),
            sample_and_perturb_particle=lambda _: Particle({"p": 0.25}),
            particle_to_distance=lambda particle, **_: abs(
                particle["p"] - 0.25
            ),
            calculate_weight=lambda _: 1.0,
            replace_particle_population=lambda _: None,
            reporter=reporter,
        ),
        run_state=SamplerRunState(1, False),
    )
    proposed_population = ParticlePopulation()

    considered = runner._accept_particle_batch(
        generation=0,
        proposed_population=proposed_population,
        proposed_particles=[Particle({"p": 0.25})],
        errs=[0.5],
    )

    assert considered == 1
    assert proposed_population.size == 1


def test_batch_generation_runner_run_generation_records_state():
    stored_populations: list[ParticlePopulation] = []
    reporter = SamplerReporter(
        verbose=True,
        console=Console(file=StringIO(), force_terminal=True),
    )
    run_state = SamplerRunState(1, False)
    runner = BatchGenerationRunner(
        config=BatchGenerationConfig(
            generation_particle_count=1,
            tolerance_values=[0.5],
            seed_sequence=SeedSequence(123),
            max_proposals_per_batch=10_000,
            sample_particle_from_priors=lambda _: Particle({"p": 0.25}),
            sample_and_perturb_particle=lambda _: Particle({"p": 0.8}),
            particle_to_distance=lambda particle, **_: abs(
                particle["p"] - 0.25
            ),
            calculate_weight=lambda _: 1.0,
            replace_particle_population=stored_populations.append,
            reporter=reporter,
        ),
        run_state=run_state,
    )
    generation_start_time = time.time()

    generation_stats = runner.run_generation(
        BatchGenerationRequest(
            generation=0,
            batchsize=1,
            warmup=False,
            chunksize=1,
            executor=None,
            overall_start_time=generation_start_time,
            generation_start_time=generation_start_time,
            particle_kwargs={},
        )
    )

    assert generation_stats.successes == 1
    assert generation_stats.attempts == 1
    assert run_state.step_successes == [1]
    assert run_state.step_attempts == [1]
    assert stored_populations[0].size == 1


def _make_runner_for_batch_size(
    *,
    generation_particle_count: int,
    max_proposals_per_batch: int,
) -> BatchGenerationRunner:
    reporter = SamplerReporter(
        verbose=False,
        console=Console(file=StringIO(), force_terminal=True),
    )
    return BatchGenerationRunner(
        config=BatchGenerationConfig(
            generation_particle_count=generation_particle_count,
            tolerance_values=[1.0],
            seed_sequence=SeedSequence(0),
            max_proposals_per_batch=max_proposals_per_batch,
            sample_particle_from_priors=lambda _: Particle({"p": 0.0}),
            sample_and_perturb_particle=lambda _: Particle({"p": 0.0}),
            particle_to_distance=lambda _particle, **_: 0.0,
            calculate_weight=lambda _: 1.0,
            replace_particle_population=lambda _: None,
            reporter=reporter,
        ),
        run_state=SamplerRunState(1, False),
    )


def test_get_batch_sample_size_warmup_first_batch_uses_configured_batchsize():
    """Pin the warmup heuristic so future edits do not silently rewire it.

    During warmup the *first* batch (population still empty) intentionally
    uses the caller-supplied ``batchsize`` rather than the warmup ceiling;
    later batches in the same warmup phase scale up to
    ``max_proposals_per_batch``. This test locks both branches in.
    """
    runner = _make_runner_for_batch_size(
        generation_particle_count=100,
        max_proposals_per_batch=64,
    )
    state = BatchGenerationState(proposed_population=ParticlePopulation())

    # First batch: population empty -> use the configured batchsize.
    assert runner._get_batch_sample_size(state, batchsize=8, warmup=True) == 8

    # Second batch in warmup: population has progressed, so the warmup
    # ceiling kicks in.
    state.proposed_population.add_particle(Particle({"p": 0.0}), 1.0)
    state.attempts = 1
    sample = runner._get_batch_sample_size(state, batchsize=8, warmup=True)
    assert sample <= 64
    assert sample >= 1


def test_get_batch_sample_size_non_warmup_uses_batchsize_only():
    runner = _make_runner_for_batch_size(
        generation_particle_count=100,
        max_proposals_per_batch=10_000,
    )
    state = BatchGenerationState(proposed_population=ParticlePopulation())

    # Empty population -> first batch returns the configured batchsize.
    assert runner._get_batch_sample_size(state, batchsize=4, warmup=False) == 4
