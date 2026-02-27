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
    """
    ABCSampler is a class that implements an Approximate Bayesian Computation (ABC)
    Sequential Monte Carlo (SMC) sampler. This sampler is used to estimate posterior
    distributions of parameters for a given model by iteratively sampling and perturbing
    particles, and evaluating their distance from observed data using user-supplied functions.

    Attributes:
        generation_particle_count (int): Number of particles to accept per generation for a complete population.
        max_attempts_per_proposal (int): Maximum number of sample and perturb attempts to propose a particle.
        tolerance_values (list[float]): List of tolerance values for each generation for evaluating acceptance criterion.
        _priors (PriorDistribution): Prior distribution of the parameters being calibrated.
        _perturbation_kernel (PerturbationKernel): Initial kernel used to perturb particles across SMC steps.
        _variance_adapter (VarianceAdapter): Adapter to adjust perturbation variance across SMC steps.
        particles_to_params (Callable[[Particle], dict]): Function to map particles to model parameters.
        outputs_to_distance (Callable[..., float]): Function to compute distance between model outputs and target data.
        target_data (Any): Observed data to compare against.
        model_runner (MRPModel): Model runner to simulate outputs given parameters.
        seed (int | None): Random seed for reproducibility.
        _seed_sequence (SeedSequence): Seed sequence for random number generation throughout the sampler run.
        drop_previous_population_data (bool): Whether to drop previous population data when storing the accepted particles between SMC steps.
        population_archive (dict[int, ParticlePopulation]): Archive of particle populations from previous generations.
        smc_step_successes (list[int]): List of number of accepted particles for each successful SMC step. Initializes to zeroes.
        verbose (bool): Whether to print verbose output during execution.
        _updater (_ParticleUpdater): Internal helper for particle updates.

    Methods:
        particle_population:
            Getter and setter for the current particle population. Automatically archives
            the previous population if `drop_previous_population_data` is False.

        get_posterior_particles() -> ParticlePopulation:
            Returns the posterior particle population after the sampler has run to completion.

        run(**kwargs: Any):
            Executes the ABC-SMC algorithm. Raises an error if any keyword argument conflicts
            with existing attributes.

        sample_priors(n: int = 1) -> Sequence[dict[str, Any]]:
            Samples `n` states from the prior distribution.

        sample_particle_from_priors() -> Particle:
            Samples a single particle from the prior distribution.

        sample_particle() -> Particle:
            Samples a particle from the current population.

        sample_and_perturb_particle() -> Particle:
            Samples and perturbs a particle from the current population.

        calculate_weight(particle) -> float:
            Calculates the weight of a given particle based on its prior and perturbed probabilities.
    """

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
        """
        Updates the particle population for the sampler.

        If `drop_previous_population_data` is set to False and there is existing
        particle population data, the current particle population is archived
        before updating to the new population.

        Args:
            population (ParticlePopulation): The new particle population to set.

        Attributes:
            drop_previous_population_data (bool): Determines whether to discard
                previous population data or archive it.
            _updater.particle_population (ParticlePopulation): The current particle
                population managed by the updater.
            population_archive (dict): A dictionary storing archived particle
                populations, indexed by step.

        Behavior:
            - If `drop_previous_population_data` is False and there is existing
              particle population data, the current population is archived with
              a step index.
            - Updates the `_updater.particle_population` with the new population.
            - Weights of the new population are normalized and the perturbation
              variance is adapted by the particle updater's setter method.
        """
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
        """
        Retrieve the posterior particle population.

        This method returns the current particle population representing the posterior
        distribution after the sampling process has been completed. It ensures
        that the posterior population is fully populated before returning it.

        Returns:
            ParticlePopulation: The particle population representing the posterior
            distribution.

        Raises:
            ValueError: If the posterior population is not fully populated,
            indicating that the sampler has not been run to completion.
        """
        if self.smc_step_successes[-1] != self.generation_particle_count:
            raise ValueError(
                "Posterior population is not fully populated. Please run the sampler to completion before accessing the posterior population."
            )
        return self.particle_population

    def run(self, **kwargs: Any):
        """
        Executes the Sequential Monte Carlo (SMC) sampling process.

        This method performs the SMC algorithm to generate a population of particles
        that approximate the posterior distribution of the model parameters. The process
        involves iteratively sampling and perturbing particles, evaluating their fitness
        using a distance metric, and accepting or rejecting them based on a tolerance value.

        Keyword Arguments:
            **kwargs: Additional keyword arguments that can be passed to the method.
                      These arguments are used to modify the behavior of the particle
                      sampling process. Note that the keyword arguments must not conflict
                      with existing attributes of the class.

        Raises:
            ValueError: If a keyword argument conflicts with an existing attribute of the class.

        Process:
            1. For each generation, particles are sampled either from the prior distribution
               (for the first generation) or by perturbing particles from the previous generation.
            2. The sampled particles are evaluated using the model to compute a distance metric
               relative to the target data.
            3. Particles that meet the tolerance criteria are accepted and added to the population.
            4. The process continues until the desired number of particles is obtained for the generation.

        Attributes Updated:
            - `smc_step_successes`: A dictionary tracking the number of successful particles
              for each generation.
            - `particle_population`: The final population of particles for the current generation.
            - `perturbation_kernel`: The perturbation kernel may be updated by the variance adapter based on the successive particle population.
            - `population_archive`: If `drop_previous_population_data` is False, previous populations are archived before updating to the new population.

        Notes:
            - The method prints progress information if `verbose` is set to True.
            - The acceptance rate is displayed periodically during the sampling process.
        """
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
