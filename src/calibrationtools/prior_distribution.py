from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
from numpy.random import SeedSequence
from scipy.stats import expon, lognorm, norm

from .particle import Particle
from .spawn_rng import spawn_rng


class PriorDistribution(ABC):
    """
    An abstract base class representing a prior distribution for use in calibration tools.

    This class provides a blueprint for defining prior distributions, which are used to
    generate samples and calculate probability densities for given particles. Subclasses
    must implement the `sample` and `probability_density` methods.

    Args:
        params (list[str]): A list of parameter names associated with the prior distribution.

    Methods:
        sample(n: int, seed: SeedSequence | None) -> Sequence[dict[str, Any]]:
            Abstract method to generate `n` samples from the prior distribution.
            Subclasses must implement this method.

        probability_density(particle: Particle) -> float:
            Abstract method to compute the probability density of a given particle.
            Subclasses must implement this method.
    """

    def __init__(self, params: list[str]) -> None:
        self.params = params

    @abstractmethod
    def sample(
        self, n: int, seed: SeedSequence | None
    ) -> Sequence[dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def probability_density(self, particle: Particle) -> float:
        raise NotImplementedError("Subclasses must implement this method")


class CompositePriorDistribution(PriorDistribution):
    """Base class for prior distributions that sample multiple parameters."""

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
## Single parameter prior types
## ----------------------------------------------
class UniformPrior(SingleParameterPriorDistribution):
    """
    Represents a uniform prior distribution for a single parameter.

    This class defines a uniform distribution over a specified range [min, max]
    for a given parameter. It provides methods to sample from the distribution
    and calculate the probability density for a given particle.

    Args:
        param (str): The name of the parameter this prior is associated with.
        min (float): The lower bound of the uniform distribution.
        max (float): The upper bound of the uniform distribution.

    Methods:
        __init__(param: str, min: float, max: float) -> None:
            Initializes the UniformPrior with the parameter name and range.
            Raises a ValueError if `min` is greater than or equal to `max`.

        sample(n: int, seed: SeedSequence | None) -> Sequence[dict[str, Any]]:
            Generates `n` samples from the uniform distribution.
            Optionally accepts a random seed for reproducibility.

        probability_density(particle: Particle) -> float:
            Computes the probability density for a given particle.
            Returns 1.0 / (max - min) if the particle's parameter value is
            within the range [min, max], otherwise returns 0.0.
    Raises:
        ValueError: If `min` is greater than or equal to `max`.
    """

    def __init__(self, param: str, min: float, max: float) -> None:
        super().__init__(param)
        self.min = min
        self.max = max
        if self.min >= self.max:
            raise ValueError(
                "Minimum must be less than maximum for UniformPrior."
            )

    def sample(
        self, n: int, seed: SeedSequence | None
    ) -> Sequence[dict[str, Any]]:
        rng = spawn_rng(seed)
        return [
            {self.param: rng.uniform(self.min, self.max)} for _ in range(n)
        ]

    def probability_density(self, particle: Particle) -> float:
        if self.min <= particle[self.params[0]] <= self.max:
            return 1.0 / (self.max - self.min)
        else:
            return 0.0


class NormalPrior(SingleParameterPriorDistribution):
    """
    Represents a normal (Gaussian) prior distribution for a single parameter.

    Args:
        mean (float): The mean (μ) of the normal distribution.
        std_dev (float): The standard deviation (σ) of the normal distribution.

    Args:
        param (str): The name of the parameter this prior is associated with.
        mean (float): The mean (μ) of the normal distribution.
        std_dev (float): The standard deviation (σ) of the normal distribution.
            Must be positive.

    Raises:
        ValueError: If `std_dev` is not positive.

    Methods:
        sample(n: int, seed: SeedSequence | None) -> Sequence[dict[str, Any]]:
            Generates `n` samples from the normal distribution.

        probability_density(particle: Particle) -> float:
            Computes the probability density of a given particle under the
            normal distribution.
    """

    def __init__(self, param: str, mean: float, std_dev: float) -> None:
        super().__init__(param)
        self.mean = mean
        self.std_dev = std_dev
        if self.std_dev <= 0:
            raise ValueError(
                "Standard deviation must be positive for NormalPrior."
            )

    def sample(
        self, n: int, seed: SeedSequence | None
    ) -> Sequence[dict[str, Any]]:
        rng = spawn_rng(seed)
        return [
            {self.param: rng.normal(self.mean, self.std_dev)} for _ in range(n)
        ]

    def probability_density(self, particle: Particle) -> float:
        return norm.pdf(
            particle[self.params[0]], loc=self.mean, scale=self.std_dev
        )


class LogNormalPrior(SingleParameterPriorDistribution):
    class LogNormalPrior:
        """
        Represents a log-normal prior distribution for a single parameter.

        A log-normal distribution is defined by its mean and standard deviation
        in the logarithmic space. This class allows sampling from the distribution
        and calculating the probability density for a given particle.

        Args:
            mean (float): The mean of the log-normal distribution in logarithmic space.
            std_dev (float): The standard deviation of the log-normal distribution
                in logarithmic space.

        Methods:
            __init__(param: str, mean: float, std_dev: float) -> None:
                Initializes the LogNormalPrior with the parameter name, mean, and
                standard deviation. Raises a ValueError if the standard deviation
                is not positive.

            sample(n: int, seed: SeedSequence | None) -> Sequence[dict[str, Any]]:
                Generates `n` samples from the log-normal distribution. Optionally,
                a random seed can be provided for reproducibility.

            probability_density(particle: Particle) -> float:
                Computes the probability density of the log-normal distribution
                for a given particle.
        """

    def __init__(self, param: str, mean: float, std_dev: float) -> None:
        super().__init__(param)
        self.mean = mean
        self.std_dev = std_dev
        if self.std_dev <= 0:
            raise ValueError(
                "Standard deviation must be positive for LogNormalPrior."
            )

    def sample(
        self, n: int, seed: SeedSequence | None
    ) -> Sequence[dict[str, Any]]:
        rng = spawn_rng(seed)
        return [
            {self.param: rng.lognormal(self.mean, self.std_dev)}
            for _ in range(n)
        ]

    def probability_density(self, particle: Particle) -> float:
        return lognorm.pdf(
            particle[self.params[0]], s=self.std_dev, scale=np.exp(self.mean)
        )


class ExponentialPrior(SingleParameterPriorDistribution):
    """
    Represents an exponential prior distribution for a single parameter.

    The exponential distribution is defined by a rate parameter (λ), which must
    be positive. This class provides methods to sample from the distribution
    and calculate the probability density for a given particle.

    Args:
        param (str): The name of the parameter this prior is associated with.
        rate (float): The rate parameter (λ) of the exponential distribution.
                        Must be positive.

    Methods:
        __init__(param: str, rate: float) -> None:
            Initializes the ExponentialPrior with the given parameter name and
            rate. Raises a ValueError if the rate is not positive.

        sample(n: int, seed: SeedSequence | None) -> Sequence[dict[str, Any]]:
            Generates `n` samples from the exponential distribution using the
            specified random seed. Returns a sequence of dictionaries where
            each dictionary contains the sampled value for the parameter.

        probability_density(particle: Particle) -> float:
            Computes the probability density of the given particle under the
            exponential distribution.

    Raises:
        ValueError: If `rate` is not positive.
    """

    def __init__(self, param: str, rate: float) -> None:
        super().__init__(param)
        self.rate = rate
        if self.rate <= 0:
            raise ValueError("Rate must be positive for ExponentialPrior.")

    def sample(
        self, n: int, seed: SeedSequence | None
    ) -> Sequence[dict[str, Any]]:
        rng = spawn_rng(seed)
        return [{self.param: rng.exponential(1 / self.rate)} for _ in range(n)]

    def probability_density(self, particle: Particle) -> float:
        return float(expon.pdf(particle[self.params[0]], scale=1 / self.rate))


class SeedPrior(SingleParameterPriorDistribution):
    """
    A prior distribution for a single parameter that generates random integer seeds.

    Args:
        param (str): The name of the parameter for which the prior distribution is defined.

    Methods:
        sample(n: int, seed: SeedSequence | None) -> Sequence[dict[str, Any]]:
            Samples `n` random seeds using the provided seed sequence.

        probability_density(particle: Particle) -> float:
            Computes the probability density for the given particle.
    """

    def __init__(self, param: str) -> None:
        super().__init__(param)

    def sample(
        self, n: int, seed: SeedSequence | None
    ) -> Sequence[dict[str, Any]]:
        rng = spawn_rng(seed)
        return [{self.param: rng.integers(0, 2**32)} for _ in range(n)]

    def probability_density(self, particle: Particle) -> float:
        if self.param in particle:
            return 1.0
        else:
            return 0.0


### ----------------------------------------------
### Multi-parameter prior types
### ----------------------------------------------
class IndependentPriors(CompositePriorDistribution):
    """
    A multi-parameter prior distribution list where each parameter is sampled independently.

    Args:
        priors (list[PriorDistribution]): A list of individual prior distributions for each parameter.
    Methods:
        sample(n: int, seed: SeedSequence | None) -> Sequence[dict[str, Any]]:
            Generates `n` samples from the independent prior distributions. Each sample is a
            dictionary that combines the sampled values from all individual priors.
        probability_density(particle: Particle) -> float:
            Computes the joint probability density of a given particle under the assumption that
            the parameters are independent. This is calculated as the product of the probability
            densities from each individual prior distribution for the corresponding parameter
            values in the particle.
    """

    def __init__(self, priors: list[PriorDistribution]) -> None:
        super().__init__(priors)

    def sample(
        self, n: int, seed: SeedSequence | None
    ) -> Sequence[dict[str, Any]]:
        samples = []
        for _ in range(n):
            sample = {}
            for prior in self.priors:
                sample.update(prior.sample(1, seed)[0])
            samples.append(sample)
        return samples

    def probability_density(self, particle: Particle) -> float:
        density = 1.0
        for prior in self.priors:
            density *= prior.probability_density(particle)
        return density
