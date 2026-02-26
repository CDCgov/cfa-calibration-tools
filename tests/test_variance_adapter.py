import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from calibrationtools import (
    AdaptIdentityVariance,
    AdaptMultivariateNormalVariance,
    AdaptNormalVariance,
    AdaptUniformVariance,
    IndependentKernels,
    MultivariateNormalKernel,
    NormalKernel,
    ParticlePopulation,
    UniformKernel,
)


def test_adapt_identity_variance():
    adapter = AdaptIdentityVariance()
    population = MagicMock(spec=ParticlePopulation)
    kernel = MagicMock()
    adapter.adapt(population, kernel)
    # No changes expected, just ensure no exceptions are raised


def test_adapt_normal_variance():
    adapter = AdaptNormalVariance()
    population = ParticlePopulation(
        states=[{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    )
    kernel = NormalKernel(param="x", std_dev=1.0)
    adapter.adapt(population, kernel)
    expected_std_dev = math.sqrt(np.var([1.0, 2.0, 3.0]) * 2.0)
    assert kernel.std_dev == pytest.approx(expected_std_dev)


def test_adapt_uniform_variance():
    adapter = AdaptUniformVariance()
    population = ParticlePopulation(
        states=[{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    )
    kernel = UniformKernel(param="x", width=1.0)
    adapter.adapt(population, kernel)
    expected_width = math.sqrt(np.var([1.0, 2.0, 3.0]) * 2.0) * 2.0
    assert kernel.width == pytest.approx(expected_width)


def test_adapt_multivariate_normal_variance():
    adapter = AdaptMultivariateNormalVariance()
    population = ParticlePopulation(
        states=[
            {"x": 1.0, "y": 2.0},
            {"x": 2.0, "y": 3.0},
            {"x": 3.0, "y": 4.0},
        ]
    )
    kernel = MultivariateNormalKernel(params=["x", "y"], cov_matrix=np.eye(2))
    adapter.adapt(population, kernel)
    states_matrix = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    expected_cov_matrix = np.cov(states_matrix.T) * 2.0
    assert np.allclose(kernel.cov_matrix, expected_cov_matrix)


def test_adapt_composite_kernel():
    adapter = AdaptNormalVariance()
    population = ParticlePopulation(
        states=[{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    )
    normal_kernel = NormalKernel(param="x", std_dev=1.0)
    composite_kernel = IndependentKernels(kernels=[normal_kernel])
    adapter.adapt(population, composite_kernel)
    expected_std_dev = math.sqrt(np.var([1.0, 2.0, 3.0]) * 2.0)
    assert normal_kernel.std_dev == pytest.approx(expected_std_dev)
