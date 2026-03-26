import copy

import pytest

from calibrationtools.calibration_results import CalibrationResults
from calibrationtools.particle_population import ParticlePopulation
from calibrationtools.particle_population_metrics import (
    ParticlePopulationMetrics,
)
from calibrationtools.particle_updater import _ParticleUpdater


@pytest.fixture()
def updater(
    K, P, Vnorm, particle_population, seed_sequence
) -> _ParticleUpdater:
    return _ParticleUpdater(
        perturbation_kernel=K,
        priors=P,
        variance_adapter=Vnorm,
        particle_population=particle_population,
        seed_sequence=seed_sequence,
    )


def test_init_calibration_results(updater):
    results = CalibrationResults(
        _updater=updater,
        generator_history={},
        population_archive={},
        success_counts={
            "generation_particle_count": [3, 3],
            "successes": [3, 3],
            "attempts": [6, 12],
        },
        tolerance_values=[0.1, 0.05],
    )
    assert results.fitted_params == ["p"]
    assert isinstance(results.posterior_particles, ParticlePopulation)
    assert results.ess == pytest.approx(1 / (0.2**2 + 0.3**2 + 0.5**2))
    assert results.acceptance_rates == [0.5, 0.25]
    assert results.aggregate_acceptance_rate == 1 / 3
    assert results.credible_intervals == {"p": (0.1, 0.9)}
    assert results.point_estimates == {"p": 0.2 * 0.1 + 0.3 * 0.5 + 0.5 * 0.9}

    assert isinstance(results.posterior, ParticlePopulationMetrics)
    assert results.posterior.particle_population == updater.particle_population


def test_calibration_results_validation(updater):
    # Incorrect number of generations in success counts
    with pytest.raises(ValueError):
        CalibrationResults(
            _updater=updater,
            generator_history={},
            population_archive={},
            success_counts={
                "generation_particle_count": [3],
                "successes": [3],
                "attempts": [6],
            },
            tolerance_values=[0.1, 0.05],
        )

    # Incorrect particle count for successes
    with pytest.raises(ValueError):
        CalibrationResults(
            _updater=updater,
            generator_history={},
            population_archive={},
            success_counts={
                "generation_particle_count": [3, 3],
                "successes": [3, 10],
                "attempts": [6, 12],
            },
            tolerance_values=[0.1, 0.05],
        )

    # Incorrect number of successes in final step
    with pytest.raises(ValueError):
        CalibrationResults(
            _updater=updater,
            generator_history={},
            population_archive={},
            success_counts={
                "generation_particle_count": [3, 3],
                "successes": [3, 2],
                "attempts": [6, 12],
            },
            tolerance_values=[0.1, 0.05],
        )

    # Incorrect total weight in particle population
    with pytest.raises(ValueError):
        updater.particle_population.add_particle(
            {"p": 0.2}, weight=0.1
        )  # Add a particle to make total weight > 1
        CalibrationResults(
            _updater=updater,
            generator_history={},
            population_archive={},
            success_counts={
                "generation_particle_count": [3, 3],
                "successes": [3, 3],
                "attempts": [6, 12],
            },
            tolerance_values=[0.1, 0.05],
        )


def test_sample_posterior_particles(updater):
    results = CalibrationResults(
        _updater=updater,
        generator_history={},
        population_archive={},
        success_counts={
            "generation_particle_count": [3, 3],
            "successes": [3, 3],
            "attempts": [6, 12],
        },
        tolerance_values=[0.1, 0.05],
    )
    samples = results.sample_posterior_particles(n=2, perturb=False)
    assert len(samples) == 2
    assert all(
        [s in results.posterior.particle_population.particles for s in samples]
    )

    perturbed_samples = results.sample_posterior_particles(n=2, perturb=True)
    assert len(perturbed_samples) == 2
    assert all(
        [
            s not in results.posterior.particle_population.particles
            for s in perturbed_samples
        ]
    )


def test_sample_posterior_repeatable(updater):
    results = CalibrationResults(
        _updater=copy.deepcopy(updater),
        generator_history={},
        population_archive={},
        success_counts={
            "generation_particle_count": [3, 3],
            "successes": [3, 3],
            "attempts": [6, 12],
        },
        tolerance_values=[0.1, 0.05],
    )
    samples1 = results.sample_posterior_particles(n=5, perturb=True)

    results2 = CalibrationResults(
        _updater=copy.deepcopy(updater),
        generator_history={},
        population_archive={},
        success_counts={
            "generation_particle_count": [3, 3],
            "successes": [3, 3],
            "attempts": [6, 12],
        },
        tolerance_values=[0.1, 0.05],
    )
    samples2 = results2.sample_posterior_particles(n=5, perturb=True)

    assert samples1 == samples2


def test_get_diagnostics(updater):
    results = CalibrationResults(
        _updater=updater,
        generator_history={},
        population_archive={},
        success_counts={
            "generation_particle_count": [3, 3],
            "successes": [3, 3],
            "attempts": [6, 12],
        },
        tolerance_values=[0.1, 0.05],
    )

    diagnostics = results.get_diagnostics()
    assert set(diagnostics.keys()) == {
        "ess_values",
        "acceptance_rates",
        "credible_intervals",
        "point_estimates",
        "quantiles",
        "posterior_weights",
        "covariance_matrix",
        "correlation_matrix",
    }
    assert diagnostics["correlation_matrix"].shape == (1, 1)
    assert diagnostics["covariance_matrix"].shape == (1, 1)
    assert diagnostics["correlation_matrix"][0, 0] == 1.0

    avg = results.point_estimates["p"]
    assert diagnostics["covariance_matrix"][0, 0] == pytest.approx(
        0.2 * (0.1 - avg) ** 2
        + 0.3 * (0.5 - avg) ** 2
        + 0.5 * (0.9 - avg) ** 2
    )
