import pytest

from calibrationtools import Particle, ParticlePopulation


def test_particle_initialization() -> None:
    state = {"x": 1.0, "y": 2.0}
    weight = 0.5
    particle = Particle(state, weight)

    assert particle.state == state
    assert particle.weight == weight


def test_particle_init() -> None:
    state = {"val": 1.0}
    particle = Particle(state, weight=0.5)
    assert particle.state == state
    assert particle.weight == 0.5


def test_particle_repr() -> None:
    state = {"x": 1.0, "y": 2.0}
    weight = 0.5
    particle = Particle(state, weight)

    repr_str = repr(particle)
    assert "Particle" in repr_str
    assert "x" in repr_str or "state" in repr_str


def test_particle_population_initialization() -> None:
    initial_states = [{"x": 1.0}, {"x": 2.0}]
    initial_weights = [0.3, 0.7]
    population = ParticlePopulation(initial_states, initial_weights)

    assert population.size == 2
    assert population.particles[0].state == {"x": 1.0}
    assert population.particles[0].weight == 0.3
    assert population.particles[1].state == {"x": 2.0}
    assert population.particles[1].weight == 0.7


def test_particle_population_initialization_raises_on_weight_mismatch() -> (
    None
):
    initial_states = [{"x": 1.0}, {"x": 2.0}]
    initial_weights = [1.0]

    with pytest.raises(ValueError, match="Length of initial_weights"):
        ParticlePopulation(initial_states, initial_weights)


def test_particle_population_repr() -> None:
    initial_states = [{"x": 1.0}, {"x": 2.0}]
    initial_weights = [0.3, 0.7]
    population = ParticlePopulation(initial_states, initial_weights)

    repr_str = repr(population)
    assert "ParticlePopulation" in repr_str
    assert "size=2" in repr_str
    assert "ESS" in repr_str


def test_particle_population_ess() -> None:
    initial_states = [{"x": 1.0}, {"x": 2.0}]
    initial_weights = [0.5, 0.5]
    population = ParticlePopulation(initial_states, initial_weights)

    assert population.ess == 2.0

    population.particles[0].weight = 0.9
    population.particles[1].weight = 0.1
    population.normalize_weights()

    assert population.ess < 2.0


def test_particle_population_normalize_weights() -> None:
    initial_states = [{"x": 1.0}, {"x": 2.0}]
    initial_weights = [0.9, 0.7]
    population = ParticlePopulation(initial_states, initial_weights)

    population.normalize_weights()

    total_weight = sum([p.weight for p in population.particles])

    population.particles[0].weight = initial_weights[0] / total_weight
    population.particles[1].weight = initial_weights[1] / total_weight
    assert abs(total_weight - 1.0) < 1e-6


def test_empty_particle_population_properties() -> None:
    population = ParticlePopulation()

    assert population.is_empty is True
    assert population.size == 0
    assert population.total_weight == 0.0
    assert population.ess == 0.0


def test_empty_particle_population_normalize_weights_noop() -> None:
    population = ParticlePopulation()

    population.normalize_weights()

    assert population.size == 0
    assert population.total_weight == 0.0
