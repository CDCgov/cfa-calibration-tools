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
    """
    Abstract base class interface for adapting the variance of supplied
    perturbation kernels based on particle population variance.

    Methods
    -------
    adapt(population: ParticlePopulation, kernel: PerturbationKernel) -> None
        Abstract method to adapt the variance of the given perturbation kernel
        using the specified particle population. This method must be implemented
        by subclasses.
    """

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
    """
    A class that adapts the variance of a normal perturbation kernel based on a given particle population.

    Methods
    -------
    adapt(population: ParticlePopulation, kernel: PerturbationKernel) -> None
        Adjusts the standard deviation of a normal perturbation kernel based on the variance
        of the specified parameter in the particle population. If the kernel is a composite
        kernel, it searches for a normal kernel within the composite. If no normal kernel is
        found, the method exits without making changes.

    Parameters
    ----------
    population : ParticlePopulation
        The population of particles used to calculate the variance of the parameter.
    kernel : PerturbationKernel
        The perturbation kernel whose variance is to be adapted. This can be a normal kernel
        or a composite kernel containing a normal kernel.

    Notes
    -----
    - The method calculates the variance of the parameter specified in the normal kernel
      across all particles in the population.
    - The standard deviation of the normal kernel is updated to the square root of twice
      the calculated population variance.
    - If the kernel is not a normal kernel or does not contain one, no changes are made.
    """

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
    """
    A class that adjusts the variance of a uniform perturbation kernel based on
    the variance of a given particle population.

    Methods
    -------
    adapt(population: ParticlePopulation, kernel: PerturbationKernel) -> None
        Adjusts the width of a UniformKernel or a UniformKernel within a
        CompositePerturbationKernel based on the variance of the specified
        parameter in the particle population. If no UniformKernel is found,
        the method exits without making changes.
    """

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
    """
    A class that adapts the covariance matrix of a multivariate normal perturbation kernel
    based on the particle population.

    Methods:
        adapt(population: ParticlePopulation, kernel: PerturbationKernel) -> None:
            Adjusts the covariance matrix of a `MultivariateNormalKernel` or a
            `CompositePerturbationKernel` containing a `MultivariateNormalKernel`
            using the particle population. If no `MultivariateNormalKernel` is found,
            the method exits without making changes.

    Notes:
        - The covariance matrix is scaled by a factor of 2.0 after being computed
          from the particle states.
        - The method assumes that the `population` contains particles with parameters
          matching those specified in the `MultivariateNormalKernel`.
    """

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
