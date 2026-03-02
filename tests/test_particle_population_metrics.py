import pytest
import numpy as np
from calibrationtools.particle_population_metrics import ParticlePopulationMetrics
from calibrationtools.particle_population import ParticlePopulation

@pytest.fixture
def mock_particle_population():
    particles = [
        {"param1": 1.0, "param2": 2.0, "param3": 3.0},
        {"param1": 1.5, "param2": 2.5, "param3": 3.5},
        {"param1": 2.0, "param2": 3.0, "param3": 4.0},
    ]
    return ParticlePopulation(particles)

@pytest.fixture
def metrics(mock_particle_population):
    return ParticlePopulationMetrics(mock_particle_population)

def test_validate_params(metrics):
    valid_params = metrics._validate_params(["param1", "param2"])
    assert valid_params == ["param1", "param2"]

    all_params = metrics._validate_params(None)
    assert all_params == ["param1", "param2", "param3"]

    with pytest.raises(ValueError):
        metrics._validate_params(["invalid_param"])

def test_get_quantiles(metrics):
    quantiles = metrics.get_quantiles(quantiles=[0.25, 0.5, 0.75], params=["param1", "param2"])
    assert "param1" in quantiles
    assert quantiles["param1"][0.5] == 1.5
    assert(len(quantiles) == 2)
    assert all(len(qs) == 3 for qs in quantiles.values())
    assert all(isinstance(d, dict) for d in quantiles.values())

    with pytest.raises(ValueError):
        metrics.get_quantiles(quantiles=[-0.1, 0.25])
    with pytest.raises(ValueError):
        metrics.get_quantiles(quantiles=[0.25, 1.1])

def test_get_credible_intervals(metrics):
    intervals = metrics.get_credible_intervals(lower_quantile=0.25, upper_quantile=0.75, params=["param1"])
    assert "param1" in intervals
    assert intervals["param1"] == (1.25, 1.75)

    with pytest.raises(ValueError):
        metrics.get_credible_intervals(lower_quantile=1.1, upper_quantile=0.9)

def test_get_point_estimates(metrics):
    estimates = metrics.get_point_estimates(params=["param1", "param2"])
    assert estimates["param1"] == pytest.approx(1.5)
    assert estimates["param2"] == pytest.approx(2.5)

def test_get_covariance_matrix(metrics):
    covariance_matrix = metrics.get_covariance_matrix(params=["param1", "param2"])
    assert covariance_matrix.shape == (2, 2)
    assert covariance_matrix[0, 0] > 0  # Variance of param1
    assert covariance_matrix[1, 1] > 0  # Variance of param2

def test_get_correlation_matrix(metrics):
    correlation_matrix = metrics.get_correlation_matrix(params=["param1", "param2"])
    assert correlation_matrix.shape == (2, 2)
    assert -1 <= correlation_matrix[0, 1] <= 1  # Correlation between param1 and param2
    assert correlation_matrix[0, 0] == 1  # Correlation of param1 with itself
    assert correlation_matrix[1, 1] == 1  # Correlation of param2 with itself