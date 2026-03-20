import copy
from pathlib import Path
from typing import Any, Callable, Sequence

import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial


import numpy as np
from mrp import MRPModel
from numpy.random import SeedSequence

from .calibration_results import CalibrationResults
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

    Args:
        generation_particle_count (int): Number of particles to accept per generation for a complete population.
        tolerance_values (list[float]): List of tolerance values for each generation for evaluating acceptance criterion.
        priors (PriorDistribution | dict | Path): Prior distribution of the parameters being calibrated. Can be provided as a PriorDistribution object, a dictionary, or a path to a JSON file containing a valid priors schema.
        particles_to_params (Callable[[Particle], dict]): Function to map particles to model parameters.
        outputs_to_distance (Callable[..., float]): Function to compute distance between model outputs and target data.
        target_data (Any): Observed data to compare against.
        model_runner (MRPModel): Model runner to simulate outputs given parameters.
        perturbation_kernel (PerturbationKernel): Initial kernel used to perturb particles across SMC steps.
        variance_adapter (VarianceAdapter): Adapter to adjust perturbation variance across SMC steps.
        max_attempts_per_proposal (int): Maximum number of sample and perturb attempts to propose a particle.
        seed (int | None): Random seed for reproducibility.
        verbose (bool): Whether to print verbose output during execution.
        drop_previous_population_data (bool): Whether to drop previous population data when storing the accepted particles between SMC steps.
        seed_parameter_name (str | None): The name of the seed parameter to include in the priors if `incl_seed_parameter` is True when loading priors from a dictionary or JSON file.

    Methods:
        particle_population:
            Getter and setter for the current particle population. Automatically archives
            the previous population if `drop_previous_population_data` is False.

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
        priors: PriorDistribution | dict | Path,
        particles_to_params: Callable[[Particle], dict],
        outputs_to_distance: Callable[..., float],
        target_data: Any,
        model_runner: MRPModel,
        perturbation_kernel: PerturbationKernel,
        variance_adapter: VarianceAdapter,
        max_attempts_per_proposal: int = np.iinfo(np.int32).max,
        seed: int | None = None,
        verbose: bool = True,
        drop_previous_population_data: bool = False,
        seed_parameter_name: str | None = "seed",
    ):
        self.generation_particle_count = generation_particle_count
        self.max_attempts_per_proposal = max_attempts_per_proposal
        self.tolerance_values = tolerance_values
        self._perturbation_kernel = perturbation_kernel
        self._variance_adapter = variance_adapter
        self.particles_to_params = particles_to_params
        self.outputs_to_distance = outputs_to_distance
        self.target_data = target_data
        self.model_runner = model_runner
        self.seed = seed
        self.drop_previous_population_data = drop_previous_population_data
        self.population_archive: dict[int, ParticlePopulation] = {}
        self.verbose = verbose

        if isinstance(priors, PriorDistribution):
            self._priors = priors
        elif isinstance(priors, dict):
            from .load_priors import independent_priors_from_dict

            self._priors = independent_priors_from_dict(
                priors,
                incl_seed_parameter=seed_parameter_name is not None,
                seed_parameter_name=seed_parameter_name,
            )
        elif isinstance(priors, Path) or isinstance(priors, str):
            from .load_priors import load_priors_from_json

            self._priors = load_priors_from_json(priors)

        self.init_updater()

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

    def init_updater(self, K: PerturbationKernel | None = None):
        """
        Initializes the particle updater with the current perturbation kernel, priors, variance adapter, seed sequence, and an empty particle population.
        """
        if K is None:
            K = self._perturbation_kernel
        self._seed_sequence = SeedSequence(self.seed)
        self._updater = _ParticleUpdater(
            K,
            self._priors,
            self._variance_adapter,
            self._seed_sequence,
            ParticlePopulation(),
        )

    def run(self, **kwargs: Any) -> CalibrationResults:
        """
        Executes the Sequential Monte Carlo (SMC) sampling process.

        This method performs the SMC algorithm to generate a population of particles
        that approximate the posterior distribution of the model parameters. The process
        involves iteratively sampling and perturbing particles, evaluating their fitness
        using a distance metric, and accepting or rejecting them based on a tolerance value.

        Args:
            **kwargs (Any): Additional keyword arguments that can be passed to the method.
                      These arguments are supplied to the particles_to_params function.
                      Note that the keyword arguments must not conflict with existing
                      attributes of the class.

        Returns:
            CalibrationResults: An object containing the results of the calibration process.

        Raises:
            ValueError: If a keyword argument conflicts with an existing attribute of the class.

        Process:
            1. For each generation, particles are sampled either from the prior distribution
               (for the first generation) or by perturbing particles from the previous generation.
            2. The sampled particles are evaluated using the model to compute a distance metric
               relative to the target data.
            3. Particles that meet the tolerance criteria are accepted and added to the population.
            4. The process continues until the desired number of particles is obtained for the generation.

        Args Updated:
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
                    f"Keyword argument '{k}' conflicts with existing attribute. Please choose a different name for the argument. Args cannot be set from `.run()`"
                )

        proposed_population = ParticlePopulation()
        originator_perturbation_kernel = copy.deepcopy(
            self._updater.perturbation_kernel
        )
        smc_step_successes = [0] * len(self.tolerance_values)
        smc_step_attempts = [0] * len(self.tolerance_values)

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

            smc_step_successes[generation] = proposed_population.size
            smc_step_attempts[generation] = attempts
            self.particle_population = proposed_population
            proposed_population = ParticlePopulation()

        results = CalibrationResults(
            copy.deepcopy(self._updater),
            self.population_archive,
            {
                "generation_particle_count": [self.generation_particle_count]
                * len(self.tolerance_values),
                "successes": smc_step_successes,
                "attempts": smc_step_attempts,
            },
            self.tolerance_values,
        )

        # Reset particle sampler updater to original perturbation kernel for consistent performance on re-run
        self.init_updater(K=originator_perturbation_kernel)
        return results
    
    def run_parallel(self, chunksize: int = 1, batchsize: int | None = None, max_workers: int | None = None, **kwargs: Any) -> CalibrationResults:
        """
        Executes the Sequential Monte Carlo (SMC) sampling process in parallel using a LocalParallelExecutor.

        This method performs the SMC algorithm to generate a population of particles
        that approximate the posterior distribution of the model parameters. The process
        involves iteratively sampling and perturbing particles, evaluating their fitness
        using a distance metric, and accepting or rejecting them based on a tolerance value.
        The execution is parallelized to improve performance.

        Args:
            **kwargs (Any): Additional keyword arguments that can be passed to the method.
                      These arguments are supplied to the particles_to_params function.
                      Note that the keyword arguments must not conflict with existing
                        attributes of the class.
        Returns:
            CalibrationResults: An object containing the results of the calibration process.
        """
        for k in kwargs.keys():
            if k in self.__class__.__dict__:
                raise ValueError(
                    f"Keyword argument '{k}' conflicts with existing attribute. Please choose a different name for the argument. Args cannot be set from `.run()`"
                )

        proposed_population = ParticlePopulation()
        originator_perturbation_kernel = copy.deepcopy(
            self._updater.perturbation_kernel
        )
        smc_step_successes = [0] * len(self.tolerance_values)
        smc_step_attempts = [0] * len(self.tolerance_values)

        actual_workers = (
            min(max_workers, (max(mp.cpu_count(), 1)))
            if max_workers
            else (mp.cpu_count() or 1)
        )
        if not batchsize:
            batchsize = self.generation_particle_count
            warmup = True
        else:            
            warmup = False

        if mp.current_process().name == 'MainProcess':
            with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                for generation in range(len(self.tolerance_values)):
                    if self.verbose:
                        print(
                            f"Running generation {generation + 1} with tolerance {self.tolerance_values[generation]}..."
                        )

                    # Rejection sampling algorithm
                    attempts = 0
                    while proposed_population.size < self.generation_particle_count:
                        if proposed_population.size > 0:
                            if warmup:
                                batchsize = 10_000
                            sample_size = int(min(batchsize, (self.generation_particle_count - proposed_population.size) * attempts / proposed_population.size))
                        else:
                            sample_size = batchsize
                        if generation == 0:
                            proposed_particles = [self.sample_particle_from_priors() for _ in range(sample_size)]
                        else:
                            proposed_particles = [self.sample_and_perturb_particle() for _ in range(sample_size)]
                        if self.verbose and attempts > 0:
                            print(
                                f"Attempt {attempts}... current population size is {proposed_population.size}. Acceptance rate is {proposed_population.size / attempts if attempts > 0 else 0:.4f}",
                                end="\r",
                            )
                        errs = executor.map(
                            partial(self._evaluate_particle, **kwargs), proposed_particles, chunksize=chunksize
                        )
                        for err, proposed_particle in zip(errs, proposed_particles):
                            if err < self.tolerance_values[generation] and proposed_population.size < self.generation_particle_count:
                                if generation == 0:
                                    particle_weight = 1.0
                                else:
                                    particle_weight = self.calculate_weight(
                                        proposed_particle
                                    )
                                proposed_population.add_particle(
                                    proposed_particle, particle_weight
                                )
                        attempts += len(proposed_particles)
                    smc_step_successes[generation] = proposed_population.size
                    smc_step_attempts[generation] = attempts
                    self.particle_population = proposed_population
                    proposed_population = ParticlePopulation()

        results = CalibrationResults(
            copy.deepcopy(self._updater),
            self.population_archive,
            {
                "generation_particle_count": [self.generation_particle_count]
                * len(self.tolerance_values),
                "successes": smc_step_successes,
                "attempts": smc_step_attempts,
            },
            self.tolerance_values,
        )

        # Reset particle sampler updater to original perturbation kernel for consistent performance on re-run
        self.init_updater(K=originator_perturbation_kernel)
        return results
    
    def run_parallel_by_particle(self, max_workers: int | None = None, chunksize: int = 1, max_attempts: int | None = None, **kwargs):
        for k in kwargs.keys():
            if k in self.__class__.__dict__:
                raise ValueError(
                    f"Keyword argument '{k}' conflicts with existing attribute. Please choose a different name for the argument. Args cannot be set from `.run()`"
                )
        # if sys.platform.startswith("linux"):
        #     import multiprocessing

        #     multiprocessing.set_start_method("spawn", force=True)

        proposed_population = ParticlePopulation()
        originator_perturbation_kernel = copy.deepcopy(
            self._updater.perturbation_kernel
        )
        smc_step_successes = [0] * len(self.tolerance_values)
        smc_step_attempts = [0] * len(self.tolerance_values)

        actual_workers = (
            min(max_workers, (max(mp.cpu_count(), 1)))
            if max_workers
            else (mp.cpu_count() or 1)
        )

        if mp.current_process().name == 'MainProcess':
            with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                for generation in range(len(self.tolerance_values)):
                    if self.verbose:
                        print(
                            f"Running generation {generation + 1} with tolerance {self.tolerance_values[generation]}..."
                        )

                    new_particles = executor.map(
                        partial(
                            self.get_accepted_particle, 
                            tolerance=self.tolerance_values[generation],
                            sample_from_priors=(generation == 0), 
                            max_attempts=max_attempts, 
                            **kwargs
                        ), 
                        self._seed_sequence.spawn(self.generation_particle_count), 
                        chunksize=chunksize
                    )

                    total_attempts = 0
                    for particle, attempts in new_particles:
                        if particle:
                            if generation == 0:
                                particle_weight = 1.0
                            else:
                                particle_weight = self.calculate_weight(particle)
                            proposed_population.add_particle(particle, particle_weight)
                        else:
                            raise UserWarning(f"Failed to collect a particle after {attempts} attempts")
                        total_attempts += attempts
                    
                    smc_step_successes[generation] = proposed_population.size
                    smc_step_attempts[generation] = attempts
                    self.particle_population = proposed_population
                    proposed_population = ParticlePopulation()
                    
        results = CalibrationResults(
            copy.deepcopy(self._updater),
            self.population_archive,
            {
                "generation_particle_count": [self.generation_particle_count]
                * len(self.tolerance_values),
                "successes": smc_step_successes,
                "attempts": smc_step_attempts,
            },
            self.tolerance_values,
        )

        # Reset particle sampler updater to original perturbation kernel for consistent performance on re-run
        self.init_updater(K=originator_perturbation_kernel)
        return results
            


    def get_accepted_particle(self, seed_sequence: SeedSequence, tolerance: float, sample_from_priors: bool = False, max_attempts: int | None = None, **kwargs) -> Particle:
        if not max_attempts:
            max_attempts = self.max_attempts_per_proposal
        if not seed_sequence:
            seed_sequence = self._seed_sequence
        for i in range(max_attempts):
            if sample_from_priors:
                proposed_particle = self.sample_particle_from_priors(seed_sequence)
            else:
                proposed_particle = self.sample_and_perturb_particle(seed_sequence)
            err = self._evaluate_particle(proposed_particle, **kwargs)
            if err < tolerance:
                return (proposed_particle, i)
        return (None, max_attempts)


    def sample_priors(self, n: int = 1, seed_sequence: SeedSequence | None = None) -> Sequence[dict[str, Any]]:
        """Return a sequence of states sampled from the prior distribution"""
        if not seed_sequence:
            seed_sequence = self._seed_sequence
        return self._priors.sample(n, seed_sequence)

    def sample_particle_from_priors(self, seed_sequence: SeedSequence | None = None) -> Particle:
        return Particle(self.sample_priors(n=1, seed_sequence=seed_sequence)[0])

    def sample_particle(self, seed_sequence: SeedSequence | None = None) -> Particle:
        return self._updater.sample_particle(seed_sequence)

    def sample_and_perturb_particle(self, seed_sequence: SeedSequence | None = None) -> Particle:
        return self._updater.sample_and_perturb_particle(
            max_attempts=self.max_attempts_per_proposal,
            seed_sequence=seed_sequence,
        )

    def _evaluate_particle(self, particle: Particle, **kwargs) -> float:
        params = self.particles_to_params(particle, **kwargs)
        outputs = self.model_runner.simulate(params)
        err = self.outputs_to_distance(outputs, self.target_data)
        return err

    def calculate_weight(self, particle) -> float:
        return self._updater.calculate_weight(particle)
