import pytest

from calibrationtools import ParticlePopulation, _ParticleUpdater


@pytest.fixture
def particle_updater(
    seed_sequence, K, P, V, particle_population
) -> _ParticleUpdater:
    updater = _ParticleUpdater(K, P, V, seed_sequence, particle_population)
    return updater


def test_set_particle_population_normalizes_weights(
    particle_updater, particle_population
):
    assert particle_updater.particle_population == particle_population
    # Check that the weights are normalized
    assert particle_updater.particle_population.total_weight == pytest.approx(
        1.0
    )

    particle_population_unnormalized = ParticlePopulation(
        states=particle_population.particles, weights=[0.1, 0.1, 0.1]
    )  # Not normalized
    particle_updater.particle_population = particle_population_unnormalized

    # Check that the weights are normalized
    assert particle_updater.particle_population.total_weight == pytest.approx(
        1.0
    )
    assert particle_updater.particle_population.weights == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3]
    )


def test_sample_particle(particle_updater, particle_population):
    assert particle_updater.particle_population == particle_population
    sampled_particle = particle_updater.sample_particle()
    assert sampled_particle in particle_population.particles


def test_sample_and_perturb_particle(particle_updater, particle_population):
    assert particle_updater.particle_population == particle_population
    perturbed_particle = particle_updater.sample_and_perturb_particle()
    assert (
        perturbed_particle not in particle_population.particles
    )  # Perturbed particle should not be the same as any in the population
    # Assert that the updater is unchanged by sample and perturb
    assert particle_updater.particle_population == particle_population
    assert (
        particle_updater.priors.probability_density(perturbed_particle) > 0
    )  # Perturbed particle should have non-zero prior density


def test_sample_and_perturb_particle_max_attempts(
    particle_updater, particle_population
):
    # Create a perturbation kernel that always produces invalid particles
    class InvalidPerturbationKernel:
        def perturb(self, current_particle, seed_sequence):
            return {
                "p": -1.0,
                "seed": 0,
            }  # Invalid particle outside the prior support

    particle_updater.perturbation_kernel = InvalidPerturbationKernel()
    particle_updater.particle_population = particle_population

    with pytest.raises(RuntimeError):
        particle_updater.sample_and_perturb_particle(max_attempts=5)


def test_calculate_weight(
    particle_updater, particle_population, proposed_particle
):
    assert particle_updater.particle_population == particle_population
    weight = particle_updater.calculate_weight(proposed_particle)
    assert weight >= 0  # Weights should be non-negative

    # Check that weight was calculated correctly for normal perturbation
    states = particle_population.particles
    weights = particle_population.weights
    transition_probs = [
        particle_updater.perturbation_kernel.transition_probability(
            to_particle=proposed_particle, from_particle=p
        )
        for p in states
    ]
    weighted_probs = [w * p for w, p in zip(weights, transition_probs)]

    expected_weight = particle_updater.priors.probability_density(
        proposed_particle
    ) / sum(weighted_probs)
    assert weight == pytest.approx(expected_weight)


def test_calculate_weight_zero_prob_perturbation(
    particle_updater, particle_population
):
    # Create a perturbation kernel that always produces zero transition probability
    class ZeroTransitionPerturbationKernel:
        def transition_probability(self, to_particle, from_particle):
            return 0.0  # Zero transition probability

    particle_updater.perturbation_kernel = ZeroTransitionPerturbationKernel()
    assert particle_updater.particle_population == particle_population

    proposed_particle = {
        "p": 0.5,
        "seed": 1,
    }  # A valid particle with non-zero prior density
    weight = particle_updater.calculate_weight(proposed_particle)
    assert (
        weight == 0.0
    )  # Weight should be zero due to zero transition probabilities
