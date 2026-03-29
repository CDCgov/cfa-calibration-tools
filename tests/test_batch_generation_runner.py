import time
from io import StringIO

from rich.console import Console

from calibrationtools.batch_generation_runner import (
    BatchGenerationConfig,
    BatchGenerationRunner,
)
from calibrationtools.particle import Particle
from calibrationtools.particle_population import ParticlePopulation
from calibrationtools.sampler_reporting import SamplerReporter
from calibrationtools.sampler_run_state import SamplerRunState
from calibrationtools.sampler_types import BatchGenerationRequest


def test_batch_generation_runner_accepts_equal_tolerance():
    reporter = SamplerReporter(
        verbose=True,
        console=Console(file=StringIO(), force_terminal=True),
    )
    runner = BatchGenerationRunner(
        config=BatchGenerationConfig(
            generation_particle_count=1,
            tolerance_values=[0.5],
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
