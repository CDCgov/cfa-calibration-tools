import copy
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
from mrp import MRPModel
from numpy.random import SeedSequence

from .batch_generation_runner import (
    BatchGenerationConfig,
    BatchGenerationRequest,
    BatchGenerationRunner,
)
from .calibration_results import CalibrationResults
from .particle import Particle
from .particle_evaluator import ParticleEvaluator
from .particle_population import ParticlePopulation
from .particle_updater import _ParticleUpdater
from .particlewise_generation_runner import (
    ParticlewiseGenerationConfig,
    ParticlewiseGenerationRequest,
    ParticlewiseGenerationRunner,
)
from .perturbation_kernel import PerturbationKernel
from .prior_distribution import PriorDistribution
from .sampler_reporting import SamplerReporter
from .sampler_run_state import SamplerRunState
from .sampler_types import GeneratorSlot
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
        particles_to_params (Callable[..., dict]): Function to map particles to model parameters.
        outputs_to_distance (Callable[..., float]): Function to compute distance between model outputs and target data.
        target_data (Any): Observed data to compare against.
        model_runner (MRPModel): Model runner to simulate outputs given parameters.
        perturbation_kernel (PerturbationKernel): Initial kernel used to perturb particles across SMC steps.
        variance_adapter (VarianceAdapter): Adapter to adjust perturbation variance across SMC steps.
        max_attempts_per_proposal (int): Maximum number of sample and perturb attempts to propose a particle.
        max_proposals_per_batch (int): Maximum number of particles to propose in a single batch when running in parallel with batched proposals with automatic batch sizes.
        parallel_worker_count (int): Default number of workers to use for sampler parallel execution when `max_workers` is not supplied.
        entropy (int | None): Entropy to initialize the seed sequence for reproducibility.
        verbose (bool): Whether to print verbose output during execution.
        keep_previous_population_data (bool): Whether to retain previous
            population data in the per-run archive when storing accepted
            particles between SMC steps.
        results_inherit_entropy_only (bool): Whether to initialize the seed sequence for sampling posterior particles in the results with only the sampler entropy or whether to spawn a new seed sequence from the sampler.
        seed_parameter_name (str | None): The name of the seed parameter to include in the priors if `incl_seed_parameter` is True when loading priors from a dictionary or JSON file.

    Raises:
        ValueError: If `parallel_worker_count` is not positive.

    Methods:
        particle_population:
            Getter and setter for the current particle population.

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
        particles_to_params: Callable[..., dict],
        outputs_to_distance: Callable[..., float],
        target_data: Any,
        model_runner: MRPModel,
        perturbation_kernel: PerturbationKernel,
        variance_adapter: VarianceAdapter,
        max_attempts_per_proposal: int = np.iinfo(np.int32).max,
        max_proposals_per_batch: int = 10_000,
        parallel_worker_count: int = 10,
        entropy: int | None = None,
        verbose: bool = True,
        keep_previous_population_data: bool = False,
        results_inherit_entropy_only: bool = True,
        seed_parameter_name: str | None = "seed",
    ):
        if parallel_worker_count <= 0:
            raise ValueError("parallel_worker_count must be positive")
        self.generation_particle_count = generation_particle_count
        self.max_attempts_per_proposal = max_attempts_per_proposal
        self.max_proposals_per_batch = max_proposals_per_batch
        self.parallel_worker_count = parallel_worker_count
        self.tolerance_values = tolerance_values
        self._variance_adapter = variance_adapter
        self.particles_to_params = particles_to_params
        self.outputs_to_distance = outputs_to_distance
        self.target_data = target_data
        self.model_runner = model_runner
        self._particle_evaluator = ParticleEvaluator(
            particles_to_params=particles_to_params,
            outputs_to_distance=outputs_to_distance,
            target_data=target_data,
            model_runner=model_runner,
        )
        self.entropy = entropy
        self.keep_previous_population_data = keep_previous_population_data
        self.results_inherit_entropy_only = results_inherit_entropy_only
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

        self._init_random()
        self.set_updater(perturbation_kernel)
        self._run_state = SamplerRunState(
            generation_count=len(self.tolerance_values),
            keep_previous_population_data=keep_previous_population_data,
        )

    @property
    def step_successes(self) -> list[int]:
        """Return accepted-particle counts for the active run.

        This property exposes the generation-level success counts recorded in
        the sampler run state.

        Returns:
            list[int]: Accepted-particle count for each generation in the
                active run.
        """

        return self._run_state.step_successes

    @property
    def step_attempts(self) -> list[int]:
        """Return proposal-attempt counts for the active run.

        This property exposes the generation-level attempt counts recorded in
        the sampler run state.

        Returns:
            list[int]: Proposal-attempt count for each generation in the active
                run.
        """

        return self._run_state.step_attempts

    @property
    def generator_history(self) -> dict[int, list[GeneratorSlot]]:
        """Return generator slots used for each completed generation.

        This property exposes the deterministic generator slots recorded during
        particlewise execution.

        Returns:
            dict[int, list[GeneratorSlot]]: Generator slots keyed by generation
                index.
        """

        return self._run_state.generator_history

    @property
    def population_archive(self) -> dict[int, ParticlePopulation]:
        """Return archived populations captured during the active run.

        This property exposes the per-run archive of previous populations that
        was recorded while the active run progressed across generations.

        Returns:
            dict[int, ParticlePopulation]: Archived populations keyed by their
                archive step.
        """

        return self._run_state.population_archive

    @property
    def particle_population(self) -> ParticlePopulation:
        """Return the current particle population.

        This property exposes the sampler's active population without changing
        any run bookkeeping.

        Returns:
            ParticlePopulation: Current particle population stored on the
                updater.
        """

        return self._updater.particle_population

    @particle_population.setter
    def particle_population(self, population: ParticlePopulation) -> None:
        """Set the current particle population without altering bookkeeping.

        This setter updates the population stored on the updater while leaving
        archive and generation counters untouched.

        Args:
            population (ParticlePopulation): Population to store as the current
                sampler population.

        Returns:
            None: This setter does not return a value.
        """

        self._updater.particle_population = population

    def _replace_particle_population(
        self, population: ParticlePopulation
    ) -> None:
        """Archive the current population in run state, then store the new one.

        This helper keeps archive bookkeeping explicit by recording the
        outgoing population before updating the current population.

        Args:
            population (ParticlePopulation): New population to store on the
                sampler.

        Returns:
            None: This helper does not return a value.
        """

        self._run_state.archive_population(self.particle_population)
        self.particle_population = population

    @property
    def perturbation_kernel(self) -> PerturbationKernel:
        return self._updater.perturbation_kernel

    def _init_random(self):
        """
        Initializes the random seed sequence for the sampler based on the sampler entropy.
        """
        self._seed_sequence = SeedSequence(self.entropy)

    def set_updater(self, perturbation_kernel: PerturbationKernel):
        """
        Initializes the particle updater with the current perturbation kernel, priors, variance adapter, and an empty particle population.
        """
        self._updater = _ParticleUpdater(
            perturbation_kernel,
            self._priors,
            self._variance_adapter,
            ParticlePopulation(),
        )

    def sample_priors(
        self, n: int, seed_sequence: SeedSequence
    ) -> Sequence[dict[str, Any]]:
        """
        Return a sequence of states sampled from the prior distribution
        Args:
            n (int): The number of samples to draw from the prior distribution. Defaults to 1.
            seed_sequence (SeedSequence): A seed sequence for random number generation.
        Returns:
            Sequence[dict[str, Any]]: A sequence of states sampled from the prior distribution, where each state is represented as a dictionary of parameter values.
        """
        return self._priors.sample(n, seed_sequence)

    def sample_particle_from_priors(
        self, seed_sequence: SeedSequence
    ) -> Particle:
        """
        Return a single particle sampled from the prior distribution
        Args:
            seed_sequence (SeedSequence): A seed sequence for random number generation.
        Returns:
            Particle: A single particle sampled from the prior distribution, represented as a Particle object.
        """
        return Particle(
            self.sample_priors(n=1, seed_sequence=seed_sequence)[0]
        )

    def sample_particle(self, seed_sequence: SeedSequence) -> Particle:
        """
        Return a single particle sampled from the current particle population based on their weights.
        Args:
            seed_sequence (SeedSequence): A seed sequence for random number generation.
        Returns:
            Particle: A single particle sampled from the current particle population, represented as a Particle object.
        """
        return self._updater.sample_particle(seed_sequence)

    def sample_and_perturb_particle(
        self, seed_sequence: SeedSequence
    ) -> Particle:
        """
        Return a single particle sampled from the current population and perturbed based on the perturbation kernel, ensuring that the perturbed particle satisfies the prior probability density constraints. If a perturbed particle fails to meet the prior constraints, a new particle is sampled with replacement and perturbed until a valid particle is obtained or the maximum number of attempts is reached.
        Args:
            seed_sequence (SeedSequence): A seed sequence for random number generation.
        Returns:
            Particle: A single particle sampled from the current population and perturbed based on the perturbation kernel.
        """
        return self._updater.sample_and_perturb_particle(
            max_attempts=self.max_attempts_per_proposal,
            seed_sequence=seed_sequence,
        )

    def particle_to_distance(self, particle: Particle, **kwargs: Any) -> float:
        """Compute the distance for one proposed particle.

        This method keeps `ABCSampler` as the public entry point for particle
        evaluation while delegating the actual model execution and scoring work
        to the extracted `ParticleEvaluator`.

        Args:
            particle (Particle): The particle for which to compute the distance.
            **kwargs (Any): Additional keyword arguments forwarded to
                `particles_to_params`.

        Returns:
            float: Distance between the simulated outputs and the target data.
        """
        return self._particle_evaluator.distance(particle, **kwargs)

    def calculate_weight(self, particle: Particle) -> float:
        """Calculate the importance weight for one accepted particle.

        This method preserves the public sampler API while delegating the
        actual weight calculation to the particle updater.

        Args:
            particle (Particle): The particle for which to calculate the weight.

        Returns:
            float: Importance weight for the particle under the current
                population and perturbation kernel.
        """
        return self._updater.calculate_weight(particle)

    def get_results_and_reset(
        self, perturbation_kernel: PerturbationKernel
    ) -> CalibrationResults:
        """Build calibration results and reset mutable sampler state.

        This method validates that each generation produced a full accepted
        population, creates the immutable `CalibrationResults` snapshot, and
        then resets the sampler so it can be reused for a later run.

        Args:
            perturbation_kernel (PerturbationKernel): Perturbation kernel to
                restore on the sampler after result construction.

        Returns:
            CalibrationResults: Snapshot containing the final posterior,
                generation history, archive data, and success statistics.
        """
        results = self._build_results()
        self._reset_after_run(perturbation_kernel)
        return results

    def run_parallel(
        self, max_workers: int | None = None, **kwargs: Any
    ) -> CalibrationResults:
        """
        Executes the Sequential Monte Carlo (SMC) sampling process in parallel using async orchestration over a thread pool.

        This method performs the SMC algorithm to generate a population of particles
        that approximate the posterior distribution of the model parameters. The process
        involves iteratively sampling and perturbing particles, evaluating their fitness
        using a distance metric, and accepting or rejecting them based on a tolerance value.
        The execution is parallelized to improve performance.

        Args:
            max_workers (int | None): The maximum number of worker threads to use when running in parallel. If None, it defaults to the sampler's configured `parallel_worker_count`.
            **kwargs (Any): Additional keyword arguments that can be passed to the method.
                      These arguments are supplied to the particles_to_params function.
                      Note that the keyword arguments must not conflict with existing
                      attributes of the class.
        Returns:
            CalibrationResults: An object containing the results of the calibration process.
        """
        return self.run(
            execution="parallel", max_workers=max_workers, **kwargs
        )

    def run_serial(self, **kwargs: Any) -> CalibrationResults:
        """
        Executes the Sequential Monte Carlo (SMC) sampling process in serial.

        This method performs the SMC algorithm to generate a population of particles
        that approximate the posterior distribution of the model parameters. The process
        involves iteratively sampling and perturbing particles, evaluating their fitness
        using a distance metric, and accepting or rejecting them based on a tolerance value.
        The execution is performed in serial.

        Args:
            **kwargs (Any): Additional keyword arguments that can be passed to the method.
                      These arguments are supplied to the particles_to_params function.
                      Note that the keyword arguments must not conflict with existing
                      attributes of the class.
        Returns:
            CalibrationResults: An object containing the results of the calibration process.
        """
        return self.run(execution="serial", **kwargs)

    def _resolve_worker_count(self, max_workers: int | None) -> int:
        """Resolve the worker count for a parallel sampler run.

        This helper applies the sampler default when the caller does not supply
        `max_workers` and validates that the resolved value is positive.

        Args:
            max_workers (int | None): Optional worker-count override supplied
                by the caller.

        Returns:
            int: Positive worker count for the run.

        Raises:
            ValueError: Raised when the resolved worker count is not positive.
        """
        worker_count = (
            max_workers
            if max_workers is not None
            else self.parallel_worker_count
        )
        if worker_count <= 0:
            raise ValueError("max_workers must be positive")
        return worker_count

    def _build_reporter(self) -> SamplerReporter:
        """Create the reporter used for one sampler run.

        This helper centralizes reporter construction so the public run
        methods can share the same output behavior and honor the sampler's
        `verbose` flag consistently.

        Returns:
            SamplerReporter: Reporter configured for the current verbosity.
        """

        return SamplerReporter(verbose=self.verbose)

    def _build_particlewise_generation_runner(
        self,
        reporter: SamplerReporter,
    ) -> ParticlewiseGenerationRunner:
        """Create the particlewise execution engine for the active run.

        This helper collects the stable callbacks and configuration needed by
        the extracted particlewise runner while keeping the public sampler
        facade small.

        Args:
            reporter (SamplerReporter): Reporter used for progress and summary
                output during the run.

        Returns:
            ParticlewiseGenerationRunner: Runner configured for the current
                sampler state.
        """

        return ParticlewiseGenerationRunner(
            config=ParticlewiseGenerationConfig(
                generation_particle_count=self.generation_particle_count,
                tolerance_values=self.tolerance_values,
                seed_sequence=self._seed_sequence,
                max_attempts_per_proposal=self.max_attempts_per_proposal,
                sample_particle_from_priors=self.sample_particle_from_priors,
                sample_and_perturb_particle=self.sample_and_perturb_particle,
                particle_to_distance=self.particle_to_distance,
                calculate_weight=self.calculate_weight,
                replace_particle_population=self._replace_particle_population,
                reporter=reporter,
            ),
            run_state=self._run_state,
        )

    def _build_batch_generation_runner(
        self,
        reporter: SamplerReporter,
    ) -> BatchGenerationRunner:
        """Create the batched execution engine for the active run.

        This helper collects the stable callbacks and configuration needed by
        the extracted batch runner while keeping the public sampler facade
        focused on orchestration.

        Args:
            reporter (SamplerReporter): Reporter used for progress and summary
                output during the run.

        Returns:
            BatchGenerationRunner: Runner configured for the current sampler
                state.
        """

        return BatchGenerationRunner(
            config=BatchGenerationConfig(
                generation_particle_count=self.generation_particle_count,
                tolerance_values=self.tolerance_values,
                seed_sequence=self._seed_sequence,
                max_proposals_per_batch=self.max_proposals_per_batch,
                sample_particle_from_priors=self.sample_particle_from_priors,
                sample_and_perturb_particle=self.sample_and_perturb_particle,
                particle_to_distance=self.particle_to_distance,
                calculate_weight=self.calculate_weight,
                replace_particle_population=self._replace_particle_population,
                reporter=reporter,
            ),
            run_state=self._run_state,
        )

    def _validate_run_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Validate keyword arguments forwarded into particle evaluation.

        This helper protects sampler execution from accidental collisions
        between run-time keyword arguments and existing class attributes.

        Args:
            kwargs (dict[str, Any]): Keyword arguments supplied to a run method.

        Returns:
            None: This helper does not return a value.

        Raises:
            ValueError: Raised when a run-time keyword collides with an
                existing class attribute.
        """
        for key in kwargs:
            if key in self.__class__.__dict__:
                raise ValueError(
                    f"Keyword argument '{key}' conflicts with existing attribute. Please choose a different name for the argument. ABCSampler attributes cannot be set from `.run()`"
                )

    def _build_executor(
        self, max_workers: int = 1
    ) -> ThreadPoolExecutor | None:
        """Build the executor for parallel execution with optional throttling.

        This helper constructs a ThreadPoolExecutor for parallel execution when
        `max_workers` is greater than 1

        Args:
            max_workers (int): The maximum number of worker threads to use for parallel execution. Defaults to 1.

        Returns:
            ThreadPoolExecutor | None: A ThreadPoolExecutor configured for the specified number of workers, or None if `max_workers` is 1.
        """
        if max_workers > 1:
            return ThreadPoolExecutor(
                max_workers=max_workers,
            )
        else:
            return None

    def _build_results(self) -> CalibrationResults:
        """Build the immutable results snapshot for the completed run.

        This helper validates that every generation reached the target
        population size and constructs the `CalibrationResults` object from the
        sampler's current state.

        Returns:
            CalibrationResults: Snapshot containing the final posterior,
                generation history, archive data, and success statistics.

        Raises:
            UserWarning: Raised when any generation finished with fewer accepted
                particles than `generation_particle_count`.
        """

        if any(
            count < self.generation_particle_count
            for count in self.step_successes
        ):
            raise UserWarning(
                "The number of successful particles in at least one generation is less than the specified generation_particle_count. This may indicate that the maximum particle proposal attempts are too low or the error tolerance values are too strict for the model and target data."
            )

        if self.results_inherit_entropy_only:
            # Initialize the seed sequence for sampling posterior particles in the results with the same entropy as the sampler for reproducibility across runs
            result_seed_sequence = SeedSequence(self.entropy)
        else:
            # Initialize the seed sequence for sampling posterior particles in the results by spawining from the sampler's seed sequence to ensure different entropy
            result_seed_sequence = self._seed_sequence.spawn(1)[0]
        return CalibrationResults(
            copy.deepcopy(self._updater),
            self.generator_history,
            self.population_archive,
            self._run_state.build_success_counts(
                self.generation_particle_count
            ),
            self._run_state.distance_history,
            self.tolerance_values,
            result_seed_sequence,
        )

    def _reset_after_run(
        self,
        perturbation_kernel: PerturbationKernel,
    ) -> None:
        """Reset mutable sampler state after a completed run.

        This helper restores the original perturbation kernel and clears all
        per-run bookkeeping so the sampler can be reused safely.

        Args:
            perturbation_kernel (PerturbationKernel): Perturbation kernel to
                restore on the sampler.

        Returns:
            None: This helper does not return a value.
        """

        self._init_random()
        self.set_updater(perturbation_kernel)
        self._run_state.reset()

    def run(
        self,
        execution: Literal["serial", "parallel"] = "parallel",
        max_workers: int | None = None,
        **kwargs: Any,
    ) -> CalibrationResults:
        """
        Executes the Sequential Monte Carlo (SMC) sampling process.

        This method performs the SMC algorithm to generate a population of particles
        that approximate the posterior distribution of the model parameters. The process
        involves iteratively sampling and perturbing particles, evaluating their fitness
        using a distance metric, and accepting or rejecting them based on a tolerance value.

        Args:
            execution (Literal['serial', 'parallel']): Determines whether to run the SMC sampling process in serial or parallel. Defaults to 'serial'.
            max_workers (int | None): The maximum number of worker threads to use when running in parallel. If None, it defaults to the sampler's configured `parallel_worker_count`. This argument is ignored when execution is set to 'serial'.
            **kwargs (Any): Additional keyword arguments that can be passed to the method.
                      These arguments are supplied to the particles_to_params function.
                      Note that the keyword arguments must not conflict with existing
                      attributes of the class.
        Returns:
            CalibrationResults: An object containing the results of the calibration process.
        """
        self._validate_run_kwargs(kwargs)
        originator_perturbation_kernel = copy.deepcopy(
            self.perturbation_kernel
        )
        reporter = self._build_reporter()
        overall_start_time = time.time()
        n_workers = (
            self._resolve_worker_count(max_workers)
            if execution == "parallel"
            else 1
        )
        particlewise_runner = self._build_particlewise_generation_runner(
            reporter=reporter
        )
        parallel_executor = self._build_executor(max_workers=n_workers)

        try:
            for generation in range(len(self.tolerance_values)):
                generation_stats = particlewise_runner.run_generation(
                    ParticlewiseGenerationRequest(
                        generation=generation,
                        n_workers=n_workers,
                        parallel_executor=parallel_executor,
                        overall_start_time=overall_start_time,
                        generation_start_time=time.time(),
                        particle_kwargs=dict(kwargs),
                    )
                )
        finally:
            if parallel_executor is not None:
                parallel_executor.shutdown(wait=True)

        reporter.print_run_summary(generation_stats.total_time)
        return self.get_results_and_reset(originator_perturbation_kernel)

    def run_parallel_batches(
        self,
        chunksize: int = 1,
        batchsize: int | None = None,
        max_workers: int | None = None,
        **kwargs: Any,
    ) -> CalibrationResults:
        """
        Executes the Sequential Monte Carlo (SMC) sampling process in parallel using async orchestration over a thread pool.

        This method performs the SMC algorithm to generate a population of particles
        that approximate the posterior distribution of the model parameters. The process
        involves iteratively sampling and perturbing particles, evaluating their fitness
        using a distance metric, and accepting or rejecting them based on a tolerance value.
        The execution is parallelized to improve performance.

        Args:
            chunksize (int): The approximate number of parameter sets to process in serial for each task when evaluating in parallel. Defaults to 1.
            batchsize (int | None): The number of proposed particles to generate in each batch when evaluating in parallel. If None, it defaults to the generation_particle_count. This controls how many particles are proposed at once and submitted to the executor.
            max_workers (int | None): The maximum number of worker threads to use when running in parallel. If None, it defaults to the sampler's configured `parallel_worker_count`.
            **kwargs (Any): Additional keyword arguments that can be passed to the method.
                      These arguments are supplied to the particles_to_params function.
                      Note that the keyword arguments must not conflict with existing
                        attributes of the class.
        Returns:
            CalibrationResults: An object containing the results of the calibration process.
        """
        self._validate_run_kwargs(kwargs)
        actual_workers = self._resolve_worker_count(max_workers)
        reporter = self._build_reporter()
        batch_runner = self._build_batch_generation_runner(reporter=reporter)
        resolved_batchsize, warmup = batch_runner.resolve_settings(
            batchsize=batchsize, chunksize=chunksize
        )
        originator_perturbation_kernel = copy.deepcopy(
            self.perturbation_kernel
        )
        overall_start_time = time.time()
        executor = self._build_executor(max_workers=actual_workers)

        try:
            for generation in range(len(self.tolerance_values)):
                generation_start_time = time.time()
                generation_stats = batch_runner.run_generation(
                    BatchGenerationRequest(
                        generation=generation,
                        batchsize=resolved_batchsize,
                        warmup=warmup,
                        chunksize=chunksize,
                        executor=executor,
                        overall_start_time=overall_start_time,
                        generation_start_time=generation_start_time,
                        particle_kwargs=dict(kwargs),
                    )
                )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        reporter.print_run_summary(generation_stats.total_time)
        return self.get_results_and_reset(originator_perturbation_kernel)
