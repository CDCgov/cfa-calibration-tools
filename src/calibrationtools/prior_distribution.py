from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np
from numpy.random import SeedSequence
from scipy.stats import expon, lognorm, norm

from .spawn_rng import spawn_rng


class PriorDistribution(ABC):
    params: list[str]

    def __init__(self, params: list[str]) -> None:
        self.params = params

        @abstractmethod
        def sample(self, n: int, seed: SeedSequence | None) -> Iterator[dict]:
            raise NotImplementedError("Subclasses must implement this method")

        @abstractmethod
        def probability_density(self, state: dict) -> float:
            raise NotImplementedError("Subclasses must implement this method")


class MultiParameterPriorDistribution(PriorDistribution):
    """Base class for prior distributions that sample multiple parameters."""

    priors: list[PriorDistribution]

    def __init__(self, priors: list[PriorDistribution]) -> None:
        super().__init__([])
        self.priors = priors


class SingleParameterPriorDistribution(PriorDistribution):
    """Base class for prior distributions that sample a single parameter."""

    def __init__(self, param: str) -> None:
        super().__init__([param])
        self.param = param

    @property
    def param(self) -> str:
        return self.params[0]

    @param.setter
    def param(self, value: str):
        self.params[0] = value


## ----------------------------------------------
# Single parameter prior types ------
## ----------------------------------------------
class UniformPrior(SingleParameterPriorDistribution):
    min: float
    max: float

    def __init__(self, param: str, min: float, max: float) -> None:
        super().__init__(param)
        self.min = min
        self.max = max

    def sample(self, n: int, seed: SeedSequence | None) -> Iterator[dict]:
        rng = spawn_rng(seed)
        for _ in range(n):
            yield {self.param: rng.uniform(self.min, self.max)}

    def probability_density(self, state: dict) -> float:
        if self.min <= state[self.params[0]] <= self.max:
            return 1.0 / (self.max - self.min)
        else:
            return 0.0


class NormalPrior(SingleParameterPriorDistribution):
    mean: float
    std_dev: float

    def __init__(self, param: str, mean: float, std_dev: float) -> None:
        super().__init__(param)
        self.mean = mean
        self.std_dev = std_dev

    def sample(self, n: int, seed: int | None) -> Iterator[dict]:
        rng = spawn_rng(seed)
        for _ in range(n):
            yield {self.param: rng.normal(self.mean, self.std_dev)}

    def probability_density(self, state: dict) -> float:
        return norm.pdf(
            state[self.params[0]], loc=self.mean, scale=self.std_dev
        )


class LogNormalPrior(SingleParameterPriorDistribution):
    mean: float
    std_dev: float

    def __init__(self, param: str, mean: float, std_dev: float) -> None:
        super().__init__(param)
        self.mean = mean
        self.std_dev = std_dev

    def sample(self, n: int, seed: SeedSequence | None) -> Iterator[dict]:
        rng = spawn_rng(seed)
        for _ in range(n):
            yield {self.param: rng.lognormal(self.mean, self.std_dev)}

    def probability_density(self, state: dict) -> float:
        return lognorm.pdf(
            state[self.params[0]], s=self.std_dev, scale=np.exp(self.mean)
        )


class ExponentialPrior(SingleParameterPriorDistribution):
    rate: float

    def __init__(self, param: str, rate: float) -> None:
        super().__init__(param)
        self.rate = rate

    def sample(self, n: int, seed: SeedSequence | None) -> Iterator[dict]:
        rng = spawn_rng(seed)
        for _ in range(n):
            yield {self.param: rng.exponential(1 / self.rate)}

    def probability_density(self, state: dict) -> float:
        return float(expon.pdf(state[self.params[0]], scale=1 / self.rate))


class SeedPrior(SingleParameterPriorDistribution):
    def __init__(self, param: str) -> None:
        super().__init__(param)

    def sample(self, n: int, seed: SeedSequence | None) -> Iterator[dict]:
        rng = spawn_rng(seed)
        for _ in range(n):
            yield {self.param: rng.integers(0, 2**32)}

    def probability_density(self, state: dict) -> float:
        if self.param in state:
            return 1.0
        else:
            return 0.0


### ----------------------------------------------
### Multi-parameter prior types ------
### ----------------------------------------------
class IndependentPriors(MultiParameterPriorDistribution):
    """A multi-parameter prior distribution where each parameter is sampled independently."""

    def __init__(self, priors: list[PriorDistribution]) -> None:
        super().__init__(priors)

    def sample(self, n: int, seed: SeedSequence | None) -> Iterator[dict]:
        for _ in range(n):
            sample = {}
            for prior in self.priors:
                sample.update(next(prior.sample(1, seed)))
            yield sample

    def probability_density(self, state: dict) -> float:
        density = 1.0
        for prior in self.priors:
            density *= prior.probability_density(state)
        return density
