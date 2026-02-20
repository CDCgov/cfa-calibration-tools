from typing import Any, Callable

from .particle import Particle, ParticlePopulation
from .particle_updater import _ParticleUpdater
from .perturbation_kernel import PerturbationKernel
from .prior_distribution import PriorDistribution
from .variance_adapter import VarianceAdapter


class ABCSampler:
    def __init__(
        self,
        generation_particle_count: int,
        tolerance_values: int | float | list[int] | list[float],
        priors: PriorDistribution,
        particles_to_params: Callable[[Particle], dict],
        outputs_to_distance: Callable[..., float],
        target_data: Any,
        model_runner: Any,
        perturbation_kernel: PerturbationKernel | None = None,
        variance_adapter: VarianceAdapter | None = None,
        seed: int | None = None,
    ):
        self.generation_particle_count = generation_particle_count
        self.tolerance_values = tolerance_values
        self.priors = priors
        self.perturbation_kernel = perturbation_kernel
        self.variance_adapter = variance_adapter
        self.particles_to_params = particles_to_params
        self.outputs_to_distance = outputs_to_distance
        self.target_data = target_data
        self.model_runner = model_runner
        self.seed = seed
        self.previous_population_archive = {}

        self._updater = _ParticleUpdater(
            self.perturbation_kernel,
            self.priors,
            self.variance_adapter,
            self.seed,
        )

    def run(self):
        previous_population = self.sample_particles_from_priors()

        for generation in range(len(self.tolerance_values)):
            print(
                f"Running generation {generation + 1} with tolerance {self.tolerance_values[generation]}... previous population size is {previous_population.size}"
            )
            current_population = ParticlePopulation()  # Inits a new population
            self.set_population(
                previous_population
            )  # sets `all_particles` to the previous population

            # Rejection sampling algorithm
            attempts = 0
            while current_population.size < self.generation_particle_count:
                if attempts % 100 == 0:
                    print(
                        f"Attempt {attempts}... current population size is {current_population.size}. Acceptance rate is {current_population.size / attempts if attempts > 0 else 0:.4f}",
                        end="\r",
                    )
                attempts += 1
                # Create the parameter inputs for the runner by sampling perturbed value from previous population
                particle = self.sample_particle()
                perturbed_particle = self.perturb_particle(particle)
                params = self.particles_to_params(perturbed_particle)

                # Generate the distance metric from model run
                outputs = self.model_runner.simulate(params)
                err = self.outputs_to_distance(outputs, self.target_data)

                # Add the particle to the population if accepted
                if err < self.tolerance_values[generation]:
                    perturbed_particle.weight = self.calculate_weight(
                        perturbed_particle
                    )
                    current_population.add(perturbed_particle)

            # Archive the previous generation population and make new population for next step
            self.previous_population_archive.update(
                {generation: previous_population}
            )
            current_population.normalize_weights()
            previous_population = current_population

        # Store posterior particle population
        self.posterior_population = current_population

    def set_population(self, population: ParticlePopulation):
        self._updater.set_particle_population(population)

    def sample_particles_from_priors(self, n=None) -> ParticlePopulation:
        """Return a particle from the prior distribution"""
        if not n:
            n = self.generation_particle_count
        sample_states = self.priors.sample(n, self.seed)
        population = ParticlePopulation(initial_states=sample_states)
        return population

    def perturb_particle(self, particle: Particle) -> Particle:
        return self._updater.perturb_particle(particle)

    def sample_particle(self) -> Particle:
        return self._updater.sample_particle()

    def calculate_weight(self, particle) -> float:
        return self._updater.calculate_weight(particle)

    def get_posterior_particles(self) -> ParticlePopulation:
        return self.posterior_population
