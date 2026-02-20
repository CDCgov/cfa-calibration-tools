import math
from abc import ABC, abstractmethod

import numpy as np

from .particle import ParticlePopulation
from .perturbation_kernel import (
    MultiParameterPerturbationKernel,
    MultivariateNormalKernel,
    NormalKernel,
)


class VarianceAdapter(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def adapt(
        self,
        population: ParticlePopulation,
        kernel: MultivariateNormalKernel | NormalKernel,
    ) -> None:
        pass


class AdaptNormalVariance(VarianceAdapter):
    def adapt(
        self,
        population: ParticlePopulation,
        kernel: MultiParameterPerturbationKernel,
    ) -> None:
        normal_kernel: NormalKernel | None = None
        for k in kernel.kernels:
            if isinstance(k, NormalKernel):
                normal_kernel = k
                break
        if normal_kernel is None:
            return
        states = [
            particle.state[normal_kernel.params[0]]
            for particle in population.all_particles
        ]
        var = np.var(states)
        normal_kernel.std_dev = math.sqrt(var * 2.0)


class AdaptMultivariateNormalVariance(VarianceAdapter):
    def adapt(
        self,
        population: ParticlePopulation,
        kernel: MultiParameterPerturbationKernel,
    ) -> None:
        mvn_kernel: MultivariateNormalKernel | None = None
        for k in kernel.kernels:
            if isinstance(k, MultivariateNormalKernel):
                mvn_kernel = k
                break
        if mvn_kernel is None:
            return
        states_matrix = np.array(
            [
                [particle.state[param] for param in mvn_kernel.params]
                for particle in population.all_particles
            ]
        )
        cov_matrix = np.cov(states_matrix.T)
        mvn_kernel.cov_matrix = cov_matrix * 2.0
