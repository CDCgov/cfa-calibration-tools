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
    assert population.all_particles[0].state == {"x": 1.0}
    assert population.all_particles[0].weight == 0.3
    assert population.all_particles[1].state == {"x": 2.0}
    assert population.all_particles[1].weight == 0.7


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

    population.all_particles[0].weight = 0.9
    population.all_particles[1].weight = 0.1
    population.normalize_weights()

    assert population.ess < 2.0


def test_particle_population_normalize_weights() -> None:
    initial_states = [{"x": 1.0}, {"x": 2.0}]
    initial_weights = [0.9, 0.7]
    population = ParticlePopulation(initial_states, initial_weights)

    population.normalize_weights()

    total_weight = sum(p.weight for p in population.all_particles)
    assert abs(total_weight - 1.0) < 1e-6
    assert sum([w for w in population.weights.values()]) == 1.0
