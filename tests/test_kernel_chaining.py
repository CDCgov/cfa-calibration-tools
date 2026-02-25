"""Tests for kernel chaining behavior in CompositePerturbationKernel.

These tests verify that kernels are applied sequentially (each sees the
result from the previous kernel) rather than all seeing the original particle.
"""

import copy
from typing import Any

import numpy as np
from numpy.random import SeedSequence

from calibrationtools import (
    IndependentKernels,
    MultivariateNormalKernel,
    Particle,
    PerturbationKernel,
    SeedKernel,
)


def test_kernels_chain_sequentially() -> None:
    """Verify that kernels see results from previous kernels in the chain."""
    # Create a kernel that will change 'x' from int to float
    mvn_kernel = MultivariateNormalKernel(["x"], cov_matrix=np.array([[0.1]]))

    # Create a custom kernel that checks the type it receives
    class TypeCheckKernel(PerturbationKernel):
        params = ["y"]

        def __init__(self) -> None:
            self.received_type: Any | None = None

        def perturb(
            self, from_particle: Particle, seed_sequence: SeedSequence
        ) -> Particle:
            self.received_type = type(from_particle.get("x"))
            to_particle = copy.deepcopy(from_particle)
            to_particle["y"] = 999
            return to_particle

        def transition_probability(
            self, to_particle: Particle, from_particle: Particle
        ) -> float:
            return 1.0

    type_checker = TypeCheckKernel()

    # If kernels chain, type_checker should see float from MVN
    # If they don't chain, type_checker should see original int
    kernels = IndependentKernels([mvn_kernel, type_checker])

    initial_particle = Particle({"x": 42, "y": 0})
    seed_seq = SeedSequence(123)

    result = kernels.perturb(initial_particle, seed_seq)

    # MVN should have made x a float
    assert isinstance(result["x"], (float, np.floating)), (
        "MultivariateNormalKernel should produce float"
    )

    # TypeCheckKernel should have seen the float (proves chaining)
    assert type_checker.received_type in (float, np.floating), (
        "Second kernel should see result from first kernel (chaining), "
        f"but received {type_checker.received_type}"
    )


def test_correct_kernel_order_preserves_types() -> None:
    """Verify correct kernel order (MVN excludes seed, SeedKernel last) works."""
    # CORRECT: MVN only has continuous params, SeedKernel comes after
    correct_kernels = IndependentKernels(
        [
            MultivariateNormalKernel(
                ["x", "y"], cov_matrix=np.array([[0.1, 0.0], [0.0, 0.1]])
            ),
            SeedKernel("seed"),
        ]
    )

    initial_particle = Particle({"seed": 12345, "x": 1.0, "y": 2.0})
    seed_seq = SeedSequence(999)

    # Should complete without error
    result = correct_kernels.perturb(initial_particle, seed_seq)

    # Seed should be integer
    assert isinstance(result["seed"], (int, np.integer)), (
        "SeedKernel must produce integer seed"
    )
    # Continuous params should be floats
    assert isinstance(result["x"], (float, np.floating))
    assert isinstance(result["y"], (float, np.floating))


def test_seed_kernel_order_matters_with_chaining() -> None:
    """Demonstrate that kernel order matters when kernels chain sequentially.

    This test shows why SeedKernel should come AFTER continuous parameter
    kernels: if MVN accidentally includes seed and comes after SeedKernel,
    the seed will be corrupted to float.
    """
    # Bad order 1: SeedKernel first, then MVN that includes seed
    bad_order_1 = IndependentKernels(
        [
            SeedKernel("seed"),
            MultivariateNormalKernel(
                ["seed", "x"], cov_matrix=np.array([[0.1, 0.0], [0.0, 0.1]])
            ),
        ]
    )

    initial_particle = Particle({"seed": 12345, "x": 1.0})
    seed_seq = SeedSequence(999)

    result = bad_order_1.perturb(initial_particle, seed_seq)

    # MVN corrupts the integer seed to float
    assert isinstance(result["seed"], (float, np.floating)), (
        "When MVN comes after SeedKernel and includes 'seed', "
        "it corrupts the integer to float"
    )

    # Good order: MVN excludes seed, SeedKernel last ensures integer
    good_order = IndependentKernels(
        [
            MultivariateNormalKernel(["x"], cov_matrix=np.array([[0.1]])),
            SeedKernel("seed"),
        ]
    )

    result = good_order.perturb(initial_particle, seed_seq)

    # SeedKernel ensures integer
    assert isinstance(result["seed"], (int, np.integer)), (
        "SeedKernel last ensures seed stays integer"
    )
