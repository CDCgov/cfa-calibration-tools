import math
from abc import ABC, abstractmethod

import numpy as np

from .particle_population import ParticlePopulation
from .perturbation_kernel import (
    CompositePerturbationKernel,
    MultivariateNormalKernel,
    NormalKernel,
    PerturbationKernel,
    UniformKernel,
)


class VarianceAdapter(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def adapt(
        self,
        population: ParticlePopulation,
        kernel: PerturbationKernel,
    ) -> None:
        pass


class AdaptIdentityVariance(VarianceAdapter):
    """No adaptation of variance."""

    def adapt(
        self,
        population: ParticlePopulation,
        kernel: PerturbationKernel,
    ) -> None:
        pass


class AdaptNormalVariance(VarianceAdapter):
    def adapt(
        self,
        population: ParticlePopulation,
        kernel: PerturbationKernel,
    ) -> None:
        normal_kernel: NormalKernel | None = None
        if isinstance(kernel, NormalKernel):
            normal_kernel = kernel
        elif isinstance(kernel, CompositePerturbationKernel):
            for k in kernel.kernels:
                if isinstance(k, NormalKernel):
                    normal_kernel = k
                    break
        if normal_kernel is None:
            return

        norm_params = [
            particle[normal_kernel.params[0]]
            for particle in population.particles
        ]
        var = np.var(norm_params)
        normal_kernel.std_dev = math.sqrt(var * 2.0)


class AdaptUniformVariance(VarianceAdapter):
    def adapt(
        self,
        population: ParticlePopulation,
        kernel: PerturbationKernel,
    ) -> None:
        uniform_kernel: UniformKernel | None = None
        if isinstance(kernel, UniformKernel):
            uniform_kernel = kernel
        elif isinstance(kernel, CompositePerturbationKernel):
            for k in kernel.kernels:
                if isinstance(k, UniformKernel):
                    uniform_kernel = k
                    break
        if uniform_kernel is None:
            return

        unif_params = [
            particle[uniform_kernel.params[0]]
            for particle in population.particles
        ]
        var = np.var(unif_params)
        uniform_kernel.width = math.sqrt(var * 2.0) * 2.0


class AdaptMultivariateNormalVariance(VarianceAdapter):
    def adapt(
        self,
        population: ParticlePopulation,
        kernel: PerturbationKernel,
    ) -> None:
        mvn_kernel: MultivariateNormalKernel | None = None
        if isinstance(kernel, MultivariateNormalKernel):
            mvn_kernel = kernel
        elif isinstance(kernel, CompositePerturbationKernel):
            for k in kernel.kernels:
                if isinstance(k, MultivariateNormalKernel):
                    mvn_kernel = k
                    break
        if mvn_kernel is None:
            return
        states_matrix = np.array(
            [
                [particle[param] for param in mvn_kernel.params]
                for particle in population.particles
            ]
        )
        cov_matrix = np.cov(states_matrix.T)
        mvn_kernel.cov_matrix = cov_matrix * 2.0
