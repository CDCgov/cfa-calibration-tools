import numpy as np
from numpy.random import SeedSequence

from .particle import Particle
from .particle_population import ParticlePopulation
from .perturbation_kernel import PerturbationKernel
from .prior_distribution import PriorDistribution
from .spawn_rng import spawn_rng
from .variance_adapter import VarianceAdapter


class _ParticleUpdater:
    def __init__(
        self,
        perturbation_kernel: PerturbationKernel,
        priors: PriorDistribution,
        variance_adapter: VarianceAdapter,
        seed_sequence: SeedSequence | None = None,
    ):
        self.perturbation_kernel = perturbation_kernel
        self.priors = priors
        self.variance_adapter = variance_adapter
        self.seed_sequence = seed_sequence
        self.calculate_transition_probability = np.vectorize(
            self.perturbation_kernel.transition_probability
        )

    def set_particle_population(self, particle_population: ParticlePopulation):
        self.particle_population = particle_population
        if self.particle_population.total_weight != 1.0:
            self.particle_population.normalize_weights()
        self.adapt_variance()

    def sample_particle(self) -> Particle:
        if not hasattr(self, "particle_population"):
            raise ValueError(
                "Particle population is not set. Please set the particle population before sampling."
            )
        idx = spawn_rng(self.seed_sequence).choice(
            self.particle_population.size,
            p=self.particle_population.weights,
        )
        return self.particle_population.particles[idx]

    def sample_perturbed_particle(
        self, max_attempts: int = 10_000
    ) -> Particle:
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
        numerator = self.priors.probability_density(particle)
        transition_probs = self.calculate_transition_probability(
            particle, self.particle_population.particles
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
        self.variance_adapter.adapt(
            self.particle_population, self.perturbation_kernel
        )
        self.calculate_transition_probability = np.vectorize(
            self.perturbation_kernel.transition_probability
        )
