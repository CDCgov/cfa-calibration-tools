from typing import Any, Callable, Sequence

from mrp import MRPModel
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
        model_runner: MRPModel,
        perturbation_kernel: PerturbationKernel,
        variance_adapter: VarianceAdapter,
        max_attempts_per_proposal: int = 10_000,
        seed: int | None = None,
        verbose: bool = True,
        drop_previous_population_data: bool = False,
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
        self.drop_previous_population_data = drop_previous_population_data
        self.population_archive: dict[int, ParticlePopulation] = {}
        self.smc_step_successes = [0] * len(tolerance_values)
        self.verbose = verbose

        self._updater = _ParticleUpdater(
            self._perturbation_kernel,
            self._priors,
            self._variance_adapter,
            self._seed_sequence,
            ParticlePopulation(),
        )

    @property
    def particle_population(self) -> ParticlePopulation:
        return self._updater.particle_population

    @particle_population.setter
    def particle_population(self, population: ParticlePopulation):
        if (
            not self.drop_previous_population_data
            and self._updater.particle_population.size > 0
        ):
            step = (
                max(self.population_archive.keys()) + 1
                if self.population_archive
                else 0
            )
            self.population_archive.update({step: self.particle_population})
        self._updater.particle_population = population

    def get_posterior_particles(self) -> ParticlePopulation:
        if self.smc_step_successes[-1] != self.generation_particle_count:
            raise ValueError(
                "Posterior population is not fully populated. Please run the sampler to completion before accessing the posterior population."
            )
        return self.particle_population

    def run(self, **kwargs: Any):
        for k in kwargs.keys():
            if k in self.__class__.__dict__:
                raise ValueError(
                    f"Keyword argument '{k}' conflicts with existing attribute. Please choose a different name for the argument. Attributes cannot be set from `.run()`"
                )

        proposed_population = ParticlePopulation()

        for generation in range(len(self.tolerance_values)):
            if self.verbose:
                print(
                    f"Running generation {generation + 1} with tolerance {self.tolerance_values[generation]}..."
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
                if generation == 0:
                    proposed_particle = self.sample_particle_from_priors()
                else:
                    proposed_particle = self.sample_and_perturb_particle()
                params = self.particles_to_params(proposed_particle, **kwargs)

                # Generate the distance metric from model run
                outputs = self.model_runner.simulate(params)
                err = self.outputs_to_distance(outputs, self.target_data)

                # Add the particle to the population if accepted
                if err < self.tolerance_values[generation]:
                    if generation == 0:
                        particle_weight = 1.0
                    else:
                        particle_weight = self.calculate_weight(
                            proposed_particle
                        )
                    proposed_population.add_particle(
                        proposed_particle, particle_weight
                    )

            self.smc_step_successes[generation] = proposed_population.size
            self.particle_population = proposed_population
            proposed_population = ParticlePopulation()

    def sample_priors(self, n: int = 1) -> Sequence[dict[str, Any]]:
        """Return a sequence of states sampled from the prior distribution"""
        return self._priors.sample(n, self._seed_sequence)

    def sample_particle_from_priors(self) -> Particle:
        return Particle(self.sample_priors(1)[0])

    def sample_particle(self) -> Particle:
        return self._updater.sample_particle()

    def sample_and_perturb_particle(self) -> Particle:
        return self._updater.sample_and_perturb_particle(
            max_attempts=self.max_attempts_per_proposal
        )

    def calculate_weight(self, particle) -> float:
        return self._updater.calculate_weight(particle)
