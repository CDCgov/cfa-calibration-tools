import numpy as np
from numpy.random import SeedSequence

from .particle import Particle
from .particle_population import ParticlePopulation
from .perturbation_kernel import PerturbationKernel
from .prior_distribution import PriorDistribution
from .spawn_rng import spawn_rng
from .variance_adapter import AdaptIdentityVariance, VarianceAdapter


class _ParticleUpdater:
    """
    A class responsible for managing and updating a population of particles
    in an ABC-SMC framework. It provides functionality for sampling,
    perturbing, and calculating weights for proposed particles, as well as
    adapting the variance of the perturbation kernel.

    Attributes:
        perturbation_kernel (PerturbationKernel): The kernel used to perturb particles.
        priors (PriorDistribution): The prior distribution of particle states. This remains fixed regardless of population changes.
        variance_adapter (VarianceAdapter): The adapter used to adjust the variance of the perturbation kernel according to population particle state variance.
        seed_sequence (SeedSequence | None): An optional seed sequence for random number generation.
        particle_population (ParticlePopulation): The current population of particles.

    Methods:
        sample_particle() -> Particle:
            Samples a particle from the current population based on their weights.

        sample_and_perturb_particle(max_attempts: int = 10_000) -> Particle:
            Samples a particle, perturbs it using the perturbation kernel, and returns
            the perturbed particle. Raises a RuntimeError if a valid particle cannot
            be sampled within the maximum number of attempts.

        calculate_weight(particle: Particle) -> float:
            Calculates the weight of a given particle based on the prior distribution
            and the transition probabilities of the perturbation kernel.

        adapt_variance():
            Adapts the variance of the perturbation kernel based on the current particle population.

    Raises:
        ValueError: If the particle population is not set when attempting to sample a particle.
        RuntimeError: If a valid perturbed particle cannot be sampled within the maximum number of attempts.
    """

    def __init__(
        self,
        perturbation_kernel: PerturbationKernel,
        priors: PriorDistribution,
        variance_adapter: VarianceAdapter,
        seed_sequence: SeedSequence | None = None,
        particle_population: ParticlePopulation | None = None,
    ):
        """
        Initializes the ParticleUpdater class.

        Args:
            perturbation_kernel (PerturbationKernel): The initial kernel used to perturb particles during proposals.
            priors (PriorDistribution): The prior distribution used for calculating particle weights.
            variance_adapter (VarianceAdapter): The adapter responsible for adjusting perturbation variance.
            seed_sequence (SeedSequence | None, optional): A sequence of seeds for replicable random number generation. Defaults to None.
            particle_population (ParticlePopulation | None, optional): An initial population of particles. If not provided, a new ParticlePopulation instance is created. Defaults to None.
        """
        self.perturbation_kernel = perturbation_kernel
        self.priors = priors
        self.variance_adapter = variance_adapter
        self.seed_sequence = seed_sequence
        self._particle_population = (
            particle_population
            if particle_population
            else ParticlePopulation()
        )

    @property
    def particle_population(self) -> ParticlePopulation:
        return self._particle_population

    @particle_population.setter
    def particle_population(self, particle_population: ParticlePopulation):
        """
        Updates the particle population and ensures its weights are normalized.

        This method sets the particle population, normalizes its weights if the
        total weight is not equal to 1.0, and adapts the perturbation variance
        according to the new stored particle population.

        Args:
            particle_population (ParticlePopulation): The particle population to update.
        """
        self._particle_population = particle_population
        if self._particle_population.total_weight != 1.0:
            self._particle_population.normalize_weights()
        self.adapt_variance()

    def sample_particle(self) -> Particle:
        """
        Samples a particle from the particle population based on their weights.

        Returns:
            Particle: The sampled particle from the particle population.

        Raises:
            ValueError: If the particle population is not set.
        """
        if self.particle_population.is_empty():
            raise ValueError(
                "Particle population is empty. Please add entries to the particle population before sampling."
            )
        idx = spawn_rng(self.seed_sequence).choice(
            self.particle_population.size,
            p=self.particle_population.weights,
        )
        return self.particle_population.particles[idx]

    def sample_and_perturb_particle(
        self, max_attempts: int = 10_000
    ) -> Particle:
        """
        Samples a particle from the current population and applies a perturbation to it,
        ensuring the perturbed particle satisfies the prior probability density constraints.
        If a perturbed particle fails to meet the prior constraints, a new particle is
        sampled with replacement and perturbed until a valid particle is obtained or the
        maximum number of attempts is reached.

        Args:
            max_attempts (int): The maximum number of attempts to sample and perturb
                a particle before aborting. Defaults to 10,000.

        Returns:
            Particle: A new particle object created from the perturbed particle.

        Raises:
            RuntimeError: If the method fails to sample and perturb a particle
                within the specified maximum number of attempts.
        """
        for _ in range(max_attempts):
            current_particle = self.sample_particle()
            new_particle = self.perturbation_kernel.perturb(
                current_particle, self.seed_sequence
            )
            if self.priors.probability_density(new_particle) > 0:
                return Particle(new_particle)
        raise RuntimeError(
            "Failed to sample perturbed particle after maximum attempts"
        )

    def calculate_weight(self, particle: Particle) -> float:
        """
        Calculate the weight of a proposed particle based on the prior probability
        and the weighted transition probabilities from the particles of the current population.

        Args:
            particle (Particle): The particle for which the weight is to be calculated.

        Returns:
            float: The calculated weight of the particle. Returns 0.0 if the denominator
            (weighted sum of transition probabilities) is zero.
        """
        numerator = self.priors.probability_density(particle)
        transition_probs = np.array(
            [
                self.perturbation_kernel.transition_probability(
                    to_particle=particle, from_particle=p
                )
                for p in self.particle_population.particles
            ]
        )

        denominator = np.sum(
            np.array(self.particle_population.weights) * transition_probs
        )

        if denominator > 0:
            proposed_weight = numerator / denominator
            return proposed_weight
        else:
            return 0.0

    def adapt_variance(self):
        """
        Adjusts the variance of the particle population using the variance adapter.

        This method utilizes the `variance_adapter` to adapt the variance of the
        `perturbation_kernel` based on the current `particle_population`. The
        perturbation kernel parameters are modified during this call but the
        particle population remaions the same.

        Raises:
            ValueError: If `variance_adapter` is not an AdaptIdentityVariance and
            `particle_population` is empty, adapt variance will fail.
        """
        if self.particle_population.is_empty() and not isinstance(
            self.variance_adapter, AdaptIdentityVariance
        ):
            raise ValueError(
                "Particle population is empty and variance adapter depends on population variance. Please add entries to the particle population or use `AdaptIdentityVariance` class."
            )
        self.variance_adapter.adapt(
            self.particle_population, self.perturbation_kernel
        )
