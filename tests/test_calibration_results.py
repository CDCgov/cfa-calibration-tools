import pytest

from calibrationtools.calibration_results import CalibrationResults
from calibrationtools.particle_population import ParticlePopulation
from calibrationtools.particle_updater import _ParticleUpdater
from calibrationtools.perturbation_kernel import (
    NormalKernel,
    SeedKernel,
)
from calibrationtools.sampler import ABCSampler

@pytest.fixture()
def updater(K, P, Vnorm, particle_population) -> _ParticleUpdater:
    _ParticleUpdater(
        perturbation_kernel=K,
        priors=P,
        variance_adapter=Vnorm,
        particle_population=particle_population
    )

def test_init_calibration_results(P, updater):
    results = CalibrationResults(
        _updater=updater,
        population_archive={},
        success_counts={"generation_particle_count": [10, 10], "successes": [10, 10], "attempts": [100, 200]},
        tolerance_values=[0.1, 0.05],
        priors=P,
    )
    assert results.fitted_params == ["p"]
    assert isinstance(results.posterior_particles, ParticlePopulation)
    assert results.ess == pytest.approx(1 / (0.2**2 + 0.3**2 + 0.5**2))
    assert results.acceptance_rates == [0.1, 0.05]
    assert results.aggregate_acceptance_rate == 20 / 300
    assert results.credible_intervals == {"p": (0.1, 0.9)}
    assert results.point_estimates == {"p": 0.2 * 0.1 + 0.3 * 0.5 + 0.5 * 0.9}