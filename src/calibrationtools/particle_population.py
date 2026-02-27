from typing import Any, Sequence

from .particle import Particle


class ParticlePopulation:
    """
    ParticlePopulation is a class that represents a collection of particles, each with an associated weight.
    It provides methods for managing the particles, normalizing their weights, and computing properties
    such as the effective sample size (ESS).

    Args:
        states (Sequence[dict[str, Any]] | None): Optional initial states to create as Particle objects.
        weights (Sequence[float] | None): Optional initial weights for the initial states.

    Attributes:
        particles (list[Particle]): A list of Particle objects in the population.
        weights (list[float]): A list of weights corresponding to each particle.
        ess (float): The effective sample size of the particle population.
        size (int): The number of particles in the population.
        total_weight (float): The sum of all particle weights.

    Methods:
        __init__(states, weights):
            Initializes the ParticlePopulation with optional states and weights.
        add_particle(particle, weight):
            Adds a new particle to the population with the specified weight.
        is_empty():
            Checks if the particle population is empty.
        normalize_weights():
            Normalizes the weights of the particles so that they sum to 1.
        __repr__():
            Returns a string representation of the ParticlePopulation instance.

    Errors:
        ValueError: If the length of weights does not match the length of particles on initialization.
    """

    def __init__(
        self,
        states: Sequence[dict[str, Any]] | None = None,
        weights: Sequence[float] | None = None,
    ):
        """
        Initializes a particle population with optional states and weights.

        Args:
            states (Sequence[dict[str, Any]] | None, optional): A sequence of dictionaries
                representing the states of the particles. If None, an empty particle list
                is initialized. Defaults to None.
            weights (Sequence[float] | None, optional): A sequence of weights corresponding
                to the particles. If None, all specified particle states are assigned equal
                weights of 1.0. Defaults to None. Supplied weights are normalized to 1.0 upon
                initialization.

        Raises:
            ValueError: If the length of the weights does not match the length of the particles.
        """
        self._particles: list[Particle] = (
            [] if states is None else [Particle(x) for x in states]
        )

        if weights is None:
            self._weights = [1.0] * len(self._particles)
        else:
            self._weights = list(weights)

        if len(self._weights) > 0 and len(self._weights) != len(
            self._particles
        ):
            raise ValueError(
                "Length of weights must match length of particles"
            )

        self.normalize_weights()

    @property
    def particles(self) -> list[Particle]:
        return self._particles

    @property
    def weights(self) -> list[float]:
        return self._weights

    def add_particle(self, particle: Particle, weight: float):
        self._particles.append(particle)
        self._weights.append(weight)

    @property
    def ess(self) -> float:
        """
        Calculate the Effective Sample Size (ESS) of the particle population.

        The ESS is a measure of the diversity of the particle weights. It is
        calculated as the square of the total weight divided by the sum of the
        squared weights. An ESS closer to the true size indicates a more uniform
        distribution of weights, while a lower ESS indicates that the weights
        are concentrated on fewer particles.

        Returns:
            float: The effective sample size. Returns 0.0 if the total weight
            is zero.
        """
        if self.total_weight == 0:
            return 0.0
        else:
            return (self.total_weight**2) / sum(w**2 for w in self.weights)

    @property
    def size(self) -> int:
        return len(self.particles)

    @property
    def total_weight(self) -> float:
        if self.is_empty():
            return 0.0
        return sum(self.weights)

    def is_empty(self) -> bool:
        return self.size == 0

    def normalize_weights(self):
        """
        Normalize the weights of the particle population.

        This method adjusts the weights of all particles in the population so that
        their total sum equals 1. If the population is empty, the method exits
        without performing any operation.
        """
        if self.is_empty():
            return
        else:
            normalization_factor = 1.0 / self.total_weight
            for i in range(self.size):
                self.weights[i] *= normalization_factor

    def __repr__(self):
        return f"ParticlePopulation(size={self.size}, ESS={self.ess})"
