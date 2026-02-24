import py
import pytest
from numpy.random import SeedSequence

from calibrationtools import (
    IndependentKernels,
    NormalKernel,
    Particle,
    SeedKernel,
)
from calibrationtools.prior_distribution import (
    IndependentPriors,
    SeedPrior,
    UniformPrior,
)


@pytest.fixture
def state():
    return {"x": 1.0, "y": 2.0}


@pytest.fixture
def particle(state):
    return Particle(state)


@pytest.fixture
def state2():
    return {"x": 3.0, "y": 4.0}


@pytest.fixture
def seed() -> int:
    return 0x4F266E25423C8DF01D27280A9FF78BBA


@pytest.fixture
def seed_sequence(seed: int) -> SeedSequence:
    return SeedSequence(seed)


@pytest.fixture
def seed_sequence_copy(seed: int) -> SeedSequence:
    return SeedSequence(seed)


@pytest.fixture
def K() -> IndependentKernels:
    K = IndependentKernels(
        [
            NormalKernel("p", 0.25),
            SeedKernel("seed"),
        ]
    )
    return K


@pytest.fixture
def Kc() -> IndependentKernels:
    Kc = IndependentKernels(
        [
            NormalKernel("p", 0.25),
            SeedKernel("seed"),
        ]
    )
    return Kc


@pytest.fixture
def N() -> int:
    return 10


@pytest.fixture
def Nbig() -> int:
    return 500


@pytest.fixture
def eps() -> list[int]:
    return [1000]


@pytest.fixture
def P() -> IndependentPriors:
    return IndependentPriors([UniformPrior("p", 0.0, 1.0), SeedPrior("seed")])


@pytest.fixture
def Pc() -> IndependentPriors:
    return IndependentPriors([UniformPrior("p", 0.0, 1.0), SeedPrior("seed")])
