from numpy.random import SeedSequence

from .particle import Particle, ParticlePopulation
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

    def set_particle_population(self, particle_population: ParticlePopulation):
        self.particle_population = particle_population
        self.adapt_variance()

    def sample_particle(self) -> Particle:
        if not hasattr(self, "particle_population"):
            raise ValueError(
                "Particle population is not set. Please set the particle population before sampling."
            )
        idx = spawn_rng(self.seed_sequence).choice(
            self.particle_population.size,
            p=[w for w in self.particle_population.weights.values()],
        )
        return self.particle_population.all_particles[idx]

    def perturb_particle(
        self, particle: Particle, max_attempts: int = 10_000
    ) -> Particle:
        for _ in range(max_attempts):
            new_particle_state = self.perturbation_kernel.perturb(
                particle.state
            )
            if self.priors.probability_density(new_particle_state) > 0:
                return Particle(
                    state=new_particle_state, weight=particle.weight
                )
        raise RuntimeError("Failed to perturb particle after maximum attempts")

    def calculate_weight(self, particle: Particle) -> float:
        numerator = self.priors.probability_density(particle.state)
        denominator = sum(
            self.particle_population.all_particles[j].weight
            * self.perturbation_kernel.transition_probability(
                particle.state, self.particle_population.all_particles[j].state
            )
            for j in range(self.particle_population.size)
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
