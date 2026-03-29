from calibrationtools.particle_population import ParticlePopulation
from calibrationtools.sampler_run_state import SamplerRunState
from calibrationtools.sampler_types import GeneratorSlot


def test_sampler_run_state_archives_previous_population_when_enabled(
    particle_population,
):
    state = SamplerRunState(
        generation_count=2,
        keep_previous_population_data=True,
    )
    state.replace_population(ParticlePopulation())
    assert state.population_archive == {}

    state.replace_population(particle_population)

    assert state.population_archive == {0: particle_population}


def test_sampler_run_state_does_not_archive_previous_population_when_disabled(
    particle_population,
):
    state = SamplerRunState(
        generation_count=2,
        keep_previous_population_data=False,
    )

    state.replace_population(particle_population)

    assert state.population_archive == {}


def test_sampler_run_state_reset_clears_bookkeeping(
    particle_population, seed_sequence
):
    state = SamplerRunState(
        generation_count=2,
        keep_previous_population_data=True,
    )
    generator_slots = [GeneratorSlot(id=0, seed_sequence=seed_sequence)]

    state.record_generation_history(0, generator_slots)
    state.record_attempts(generation=0, attempts=4, successes=1)
    state.replace_population(particle_population)

    state.reset()

    assert state.step_successes == [0, 0]
    assert state.step_attempts == [0, 0]
    assert state.generator_history == {}
    assert state.population_archive == {}
