from abc import ABC, abstractmethod

import numpy as np
from numpy.random import default_rng
from scipy.stats import multivariate_normal, norm


class PerturbationKernel(ABC):
    params: list[str]

    def __init__(self) -> None:
        pass

    def change_state_values(
        self,
        from_state: dict,
        perturbed_values: float | np.ndarray,
        type: type = float,
    ) -> dict:
        """Change the state values based on the perturbation kernel."""
        result = from_state.copy()
        if len(self.params) == 1:
            result[self.params[0]] = type(perturbed_values)
        else:
            for i, param in enumerate(self.params):
                result[param] = type(perturbed_values[i])
        return result

    @abstractmethod
    def perturb(self, state: dict, seed: int | None) -> dict:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def transition_probability(
        self, to_state: dict, from_state: dict
    ) -> float:
        raise NotImplementedError("Subclasses must implement this method")


class SingleParameterPerturbationKernel(PerturbationKernel, ABC):
    def __init__(self, param: str) -> None:
        super().__init__()
        self.params = [param]


class MultiParameterPerturbationKernel(PerturbationKernel, ABC):
    """Base class for perturbation kernels that perturb multiple parameters."""


## ---------------------------------------------
# Single parameter kernel types ------
## ---------------------------------------------


class SeedKernel(SingleParameterPerturbationKernel):
    def perturb(self, from_state: dict, seed: int | None) -> dict:
        results = from_state.copy()
        results[self.params[0]] = default_rng(seed).integers(0, 2**32 - 1)
        return results

    def transition_probability(self, to_state, from_state):
        return 1.0


class UniformKernel(SingleParameterPerturbationKernel):
    width: float

    def __init__(self, param: str, width: float) -> None:
        super().__init__(param)
        self.width = width

    def perturb(self, from_state: dict, seed: int | None) -> dict:
        return self.change_state_values(
            from_state,
            default_rng(seed).uniform(
                from_state[self.params[0]] - self.width,
                from_state[self.params[0]] + self.width,
            ),
        )

    def transition_probability(
        self, to_state: dict, from_state: dict
    ) -> float:
        lower_bound = from_state[self.params[0]] - self.width
        upper_bound = from_state[self.params[0]] + self.width
        if lower_bound <= to_state[self.params[0]] <= upper_bound:
            return 1.0 / (2 * self.width)
        else:
            return 0.0


class NormalKernel(SingleParameterPerturbationKernel):
    std_dev: float

    def __init__(self, param: str, std_dev: float) -> None:
        super().__init__(param)
        self.std_dev = std_dev

    def perturb(self, from_state: dict, seed: int | None) -> dict:
        return self.change_state_values(
            from_state,
            default_rng(seed).normal(from_state[self.params[0]], self.std_dev),
        )

    def transition_probability(
        self, to_state: dict, from_state: dict
    ) -> float:
        from_value = from_state[self.params[0]]
        to_value = to_state[self.params[0]]
        return float(norm.pdf(to_value, loc=from_value, scale=self.std_dev))


## ---------------------------------------------
# Multi parameter kernel types ------
## ---------------------------------------------


class IndependentKernels(MultiParameterPerturbationKernel, ABC):
    """Base class for perturbation kernels that perturb multiple parameters independently."""

    kernels: list[PerturbationKernel]

    def __init__(
        self, kernels: list[PerturbationKernel] | None = None
    ) -> None:
        super().__init__()
        self.kernels = kernels if kernels is not None else []

    def perturb(self, from_state: dict, seed: int | None = None) -> dict:
        result = from_state.copy()
        for kernel in self.kernels:
            perturbed_values = kernel.perturb(result, seed)
            result.update(perturbed_values)
        return result

    def transition_probability(
        self, to_state: dict, from_state: dict
    ) -> float:
        prob = 1.0
        for kernel in self.kernels:
            prob *= kernel.transition_probability(to_state, from_state)
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

    def perturb(self, from_state: dict, seed: int | None) -> dict:
        if self.cov_matrix is None:
            raise ValueError(
                "Covariance matrix must be set prior to calling perturb."
            )
        rng = default_rng(seed)
        from_values = [from_state[param] for param in self.params]
        return self.change_state_values(
            from_state,
            rng.multivariate_normal(mean=from_values, cov=self.cov_matrix),
        )

    def transition_probability(
        self, to_state: dict, from_state: dict
    ) -> float:
        if self.cov_matrix is None:
            raise ValueError(
                "Covariance matrix must be set prior to calling transition_probability."
            )
        from_values = [from_state[param] for param in self.params]
        to_values = [to_state[param] for param in self.params]

        return float(
            multivariate_normal.pdf(
                to_values, mean=from_values, cov=self.cov_matrix
            )
        )
