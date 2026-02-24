import pytest

from calibrationtools import ParticlePopulation


@pytest.fixture
def initial_states(state, state2) -> list[dict[str, any]]:
    return [state, state2]


@pytest.fixture
def initial_weights() -> list[float]:
    return [0.3, 0.7]


@pytest.fixture
def population(initial_states, initial_weights) -> ParticlePopulation:
    return ParticlePopulation(initial_states, initial_weights)


def test_particle_population_initialization(
    initial_states, initial_weights
) -> None:
    population = ParticlePopulation(initial_states, initial_weights)

    assert population.size == 2
    assert population.particles[0] == {"x": 1.0, "y": 2.0}
    assert population.weights[0] == 0.3
    assert population.particles[1] == {"x": 3.0, "y": 4.0}
    assert population.weights[1] == 0.7


def test_particle_population_initialization_raises_on_weight_mismatch(
    initial_states,
) -> None:
    initial_weights = [1.0]

    with pytest.raises(ValueError, match="Length of weights"):
        ParticlePopulation(initial_states, initial_weights)


def test_particle_population_repr(population) -> None:
    repr_str = repr(population)
    assert "ParticlePopulation" in repr_str
    assert "size=2" in repr_str
    assert "ESS" in repr_str


def test_particle_population_ess(population) -> None:
    assert population.ess == pytest.approx(1 / (0.3**2 + 0.7**2))

    population.weights[0] = 0.9
    population.weights[1] = 0.1
    population.normalize_weights()

    assert population.ess == pytest.approx(1 / (0.9**2 + 0.1**2))


def test_particle_population_normalize_weights() -> None:
    initial_states = [{"x": 1.0}, {"x": 2.0}]
    initial_weights = [0.9, 0.7]
    population = ParticlePopulation(initial_states, initial_weights)

    population.normalize_weights()

    total_weight = sum(population.weights)

    population.weights[0] = initial_weights[0] / total_weight
    population.weights[1] = initial_weights[1] / total_weight
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
