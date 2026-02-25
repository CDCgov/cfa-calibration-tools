import copy
from abc import ABC, abstractmethod

import numpy as np
from numpy.random import SeedSequence
from scipy.stats import multivariate_normal, norm

from .particle import Particle
from .spawn_rng import spawn_rng


class PerturbationKernel(ABC):
    params: list[str]

    def __init__(self) -> None:
        pass

    def change_particle_values(
        self,
        from_particle: Particle,
        perturbed_values: float | np.ndarray,
        type: type = float,
    ) -> Particle:
        """Change the particle values based on the perturbation kernel."""
        to_particle = copy.deepcopy(from_particle)
        if isinstance(perturbed_values, np.ndarray):
            if len(perturbed_values) == 1:
                to_particle[self.params[0]] = type(perturbed_values[0])
            else:
                for i, param in enumerate(self.params):
                    to_particle[param] = type(perturbed_values[i])
        else:
            to_particle[self.params[0]] = type(perturbed_values)
        return to_particle

    @abstractmethod
    def perturb(
        self, from_particle: Particle, seed_sequence: SeedSequence | None
    ) -> Particle:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def transition_probability(
        self, to_particle: Particle, from_particle: Particle
    ) -> float:
        raise NotImplementedError("Subclasses must implement this method")


class SingleParameterPerturbationKernel(PerturbationKernel, ABC):
    def __init__(self, param: str) -> None:
        super().__init__()
        self.params = [param]


class MultiParameterPerturbationKernel(PerturbationKernel, ABC):
    """Base class for perturbation kernels that perturb multiple parameters."""


class CompositePerturbationKernel(PerturbationKernel, ABC):
    """Composite kernel applying multiple independent kernels.

    Attributes:
        kernels (list[PerturbationKernel]): Component kernels.

    Args:
        kernels (list[PerturbationKernel] | None): Component kernels.
    """

    kernels: list[PerturbationKernel]

    def __init__(
        self, kernels: list[PerturbationKernel] | None = None
    ) -> None:
        super().__init__()
        self.kernels = kernels if kernels is not None else []

    def perturb(
        self, from_particle: Particle, seed_sequence: SeedSequence
    ) -> Particle:
        """Return combined perturbation from all component kernels.

        Kernels are applied sequentially: each kernel sees the result from
        the previous kernel, allowing kernels to build on each other's
        changes while preserving types through the chain.

        Args:
            from_particle (Particle): Original particle.
            seed_sequence (SeedSequence): SeedSequence for RNG spawning.

        Returns:
            Particle: Updated particle.
        """
        to_particle = copy.deepcopy(from_particle)

        for kernel in self.kernels:
            to_particle = kernel.perturb(to_particle, seed_sequence)
        return to_particle

    @abstractmethod
    def transition_probability(
        self, to_particle: Particle, from_particle: Particle
    ) -> float:
        """Return product of individual kernel probabilities.

        Args:
            to_particle (Particle): Destination particle.
            from_particle (Particle): Source particle.

        Returns:
            float: Joint probability.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError(
            "Method 'transition_probability(self, to_particle, from_particle)' must "
            "be implemented by subclasses."
        )


## ---------------------------------------------
## Single parameter kernel types
## ---------------------------------------------


class SeedKernel(SingleParameterPerturbationKernel):
    def perturb(
        self, from_particle: Particle, seed_sequence: SeedSequence | None
    ) -> Particle:
        to_particle = copy.deepcopy(from_particle)
        to_particle[self.params[0]] = spawn_rng(seed_sequence).integers(
            0, 2**32 - 1
        )
        return to_particle

    def transition_probability(
        self, to_particle: Particle, from_particle: Particle
    ) -> float:
        return 1.0


class UniformKernel(SingleParameterPerturbationKernel):
    width: float

    def __init__(self, param: str, width: float) -> None:
        super().__init__(param)
        self.width = width
        if self.width <= 0:
            raise ValueError("Width must be positive for UniformKernel.")

    def perturb(
        self, from_particle: Particle, seed_sequence: SeedSequence | None
    ) -> Particle:
        return self.change_particle_values(
            from_particle,
            spawn_rng(seed_sequence).uniform(
                from_particle[self.params[0]] - 0.5 * self.width,
                from_particle[self.params[0]] + 0.5 * self.width,
            ),
        )

    def transition_probability(
        self, to_particle: Particle, from_particle: Particle
    ) -> float:
        lower_bound = from_particle[self.params[0]] - 0.5 * self.width
        upper_bound = from_particle[self.params[0]] + 0.5 * self.width
        if lower_bound <= to_particle[self.params[0]] <= upper_bound:
            return 1.0 / self.width
        else:
            return 0.0


class NormalKernel(SingleParameterPerturbationKernel):
    std_dev: float

    def __init__(self, param: str, std_dev: float) -> None:
        super().__init__(param)
        self.std_dev = std_dev
        if self.std_dev <= 0:
            raise ValueError(
                "Standard deviation must be positive for NormalKernel."
            )

    def perturb(
        self, from_particle: Particle, seed_sequence: SeedSequence | None
    ) -> Particle:
        return self.change_particle_values(
            from_particle,
            spawn_rng(seed_sequence).normal(
                from_particle[self.params[0]], self.std_dev
            ),
        )

    def transition_probability(
        self, to_particle: Particle, from_particle: Particle
    ) -> float:
        from_value = from_particle[self.params[0]]
        to_value = to_particle[self.params[0]]
        return float(norm.pdf(to_value, loc=from_value, scale=self.std_dev))


## ---------------------------------------------
## Multi parameter kernel types
## ---------------------------------------------


class IndependentKernels(CompositePerturbationKernel):
    def transition_probability(
        self, to_particle: Particle, from_particle: Particle
    ) -> float:
        prob = 1.0
        for kernel in self.kernels:
            prob *= kernel.transition_probability(to_particle, from_particle)
        return prob


class MultivariateNormalKernel(MultiParameterPerturbationKernel):
    params: list[str]
    cov_matrix: np.ndarray | None = None

    def __init__(
        self, params: list[str], cov_matrix: np.ndarray | None = None
    ) -> None:
        super().__init__()
        self.params = params
        self.cov_matrix = cov_matrix

    def perturb(
        self, from_particle: Particle, seed_sequence: SeedSequence | None
    ) -> Particle:
        if self.cov_matrix is None:
            raise ValueError(
                "Covariance matrix must be set prior to calling perturb."
            )
        rng = spawn_rng(seed_sequence)
        from_values = [from_particle[param] for param in self.params]
        return self.change_particle_values(
            from_particle,
            rng.multivariate_normal(mean=from_values, cov=self.cov_matrix),
        )

    def transition_probability(
        self, to_particle: Particle, from_particle: Particle
    ) -> float:
        if self.cov_matrix is None:
            raise ValueError(
                "Covariance matrix must be set prior to calling transition_probability."
            )
        from_values = [from_particle[param] for param in self.params]
        to_values = [to_particle[param] for param in self.params]

        return float(
            multivariate_normal.pdf(
                to_values, mean=from_values, cov=self.cov_matrix
            )
        )
