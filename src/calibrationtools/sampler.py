from typing import Any, Callable

from numpy.random import SeedSequence

from .particle import Particle
from .particle_population import ParticlePopulation
from .particle_updater import _ParticleUpdater
from .perturbation_kernel import PerturbationKernel
from .prior_distribution import PriorDistribution
from .variance_adapter import VarianceAdapter


class ABCSampler:
    def __init__(
        self,
        generation_particle_count: int,
        tolerance_values: list[float],
        priors: PriorDistribution,
        particles_to_params: Callable[[Particle], dict],
        outputs_to_distance: Callable[..., float],
        target_data: Any,
        model_runner: Any,
        perturbation_kernel: PerturbationKernel,
        variance_adapter: VarianceAdapter,
        max_attempts_per_proposal: int = 10_000,
        seed: int | None = None,
        verbose: bool = True,
    ):
        self.generation_particle_count = generation_particle_count
        self.max_attempts_per_proposal = max_attempts_per_proposal
        self.tolerance_values = tolerance_values
        self._priors = priors
        self._perturbation_kernel = perturbation_kernel
        self._variance_adapter = variance_adapter
        self.particles_to_params = particles_to_params
        self.outputs_to_distance = outputs_to_distance
        self.target_data = target_data
        self.model_runner = model_runner
        self.seed = seed
        self._seed_sequence = SeedSequence(seed)
        self.previous_population_archive: dict[int, ParticlePopulation] = {}
        self.verbose = verbose

        self._updater = _ParticleUpdater(
            self._perturbation_kernel,
            self._priors,
            self._variance_adapter,
            self._seed_sequence,
        )

    @property
    def particle_population(self) -> ParticlePopulation:
        return self._updater.particle_population

    @particle_population.setter
    def particle_population(self, population: ParticlePopulation):
        self._updater.set_particle_population(population)

    def run(self):
        self.particle_population = self.sample_particles_from_priors()
        proposed_population = ParticlePopulation()

        for generation in range(len(self.tolerance_values)):
            if self.verbose:
                print(
                    f"Running generation {generation + 1} with tolerance {self.tolerance_values[generation]}... Previous population is {self.particle_population}"
                )

            # Rejection sampling algorithm
            attempts = 0
            while proposed_population.size < self.generation_particle_count:
                if self.verbose and attempts > 0 and attempts % 100 == 0:
                    print(
                        f"Attempt {attempts}... current population size is {proposed_population.size}. Acceptance rate is {proposed_population.size / attempts if attempts > 0 else 0:.4f}",
                        end="\r",
                    )
                attempts += 1
                # Create the parameter inputs for the runner by sampling perturbed value from previous population
                proposed_particle = self.sample_proposed_particle()
                params = self.particles_to_params(proposed_particle)

                # Generate the distance metric from model run
                outputs = self.model_runner.simulate(params)
                err = self.outputs_to_distance(outputs, self.target_data)

                # Add the particle to the population if accepted
                if err < self.tolerance_values[generation]:
                    particle_weight = self.calculate_weight(proposed_particle)
                    proposed_population.add_particle(
                        proposed_particle, particle_weight
                    )

            # Archive the previous generation population and make new population for next step
            self.archive_population(generation)
            self.particle_population = proposed_population
            proposed_population = ParticlePopulation()

        # Store posterior particle population
        self.posterior_population = self.particle_population

    def archive_population(self, generation: int):
        self.previous_population_archive.update(
            {generation: self.particle_population}
        )

    def sample_particles_from_priors(self, n=None) -> ParticlePopulation:
        """Return a particle from the prior distribution"""
        if not n:
            n = self.generation_particle_count
        sample_states = self._priors.sample(n, self._seed_sequence)
        population = ParticlePopulation(states=sample_states)
        return population

    def sample_current_particle(self) -> Particle:
        return self._updater.sample_particle()

    def sample_proposed_particle(self) -> Particle:
        return self._updater.sample_perturbed_particle(
            max_attempts=self.max_attempts_per_proposal
        )

    def calculate_weight(self, particle) -> float:
        return self._updater.calculate_weight(particle)

    def get_posterior_particles(self) -> ParticlePopulation:
        return self.posterior_population
