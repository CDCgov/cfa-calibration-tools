import asyncio
import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
from mrp import MRPModel
from numpy.random import SeedSequence
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from . import formatting
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
        parallel_worker_count (int): Default number of workers to use for sampler parallel execution when `max_workers` is not supplied.
        seed (int | None): Random seed for reproducibility.
        verbose (bool): Whether to print verbose output during execution.
        drop_previous_population_data (bool): Whether to drop previous population data when storing the accepted particles between SMC steps.
        seed_parameter_name (str | None): The name of the seed parameter to include in the priors if `incl_seed_parameter` is True when loading priors from a dictionary or JSON file.

    Raises:
        ValueError: If `parallel_worker_count` is not positive.

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
        parallel_worker_count: int = 10,
        seed: int | None = None,
        verbose: bool = True,
        drop_previous_population_data: bool = False,
        seed_parameter_name: str | None = "seed",
    ):
        if parallel_worker_count <= 0:
            raise ValueError("parallel_worker_count must be positive")
        self.generation_particle_count = generation_particle_count
        self.max_attempts_per_proposal = max_attempts_per_proposal
        self.parallel_worker_count = parallel_worker_count
        self.tolerance_values = tolerance_values
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

        self.init_updater(perturbation_kernel)
        self.step_successes = [0] * len(self.tolerance_values)
        self.step_attempts = [0] * len(self.tolerance_values)
        self.generator_history = {}

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

    @property
    def perturbation_kernel(self) -> PerturbationKernel:
        return self._updater.perturbation_kernel

    def init_updater(self, perturbation_kernel: PerturbationKernel):
        """
        Initializes the particle updater with the current perturbation kernel, priors, variance adapter, seed sequence, and an empty particle population.
        """
        self._seed_sequence = SeedSequence(self.seed)
        self._updater = _ParticleUpdater(
            perturbation_kernel,
            self._priors,
            self._variance_adapter,
            self._seed_sequence,
            ParticlePopulation(),
        )

    def sample_priors(
        self, n: int = 1, seed_sequence: SeedSequence | None = None
    ) -> Sequence[dict[str, Any]]:
        """
        Return a sequence of states sampled from the prior distribution
        Args:
            n (int): The number of samples to draw from the prior distribution. Defaults to 1.
            seed_sequence (SeedSequence | None): An optional SeedSequence to use for sampling. If None, the sampler's internal seed sequence will be used.
        Returns:
            Sequence[dict[str, Any]]: A sequence of states sampled from the prior distribution, where each state is represented as a dictionary of parameter values.
        """
        if not seed_sequence:
            seed_sequence = self._seed_sequence
        return self._priors.sample(n, seed_sequence)

    def sample_particle_from_priors(
        self, seed_sequence: SeedSequence | None = None
    ) -> Particle:
        """
        Return a single particle sampled from the prior distribution
        Args:
            seed_sequence (SeedSequence | None): An optional SeedSequence to use for sampling. If None, the sampler's internal seed sequence will be used.
        Returns:
            Particle: A single particle sampled from the prior distribution, represented as a Particle object.
        """
        return Particle(
            self.sample_priors(n=1, seed_sequence=seed_sequence)[0]
        )

    def sample_particle(
        self, seed_sequence: SeedSequence | None = None
    ) -> Particle:
        """
        Return a single particle sampled from the current particle population based on their weights.
        Args:
            seed_sequence (SeedSequence | None): An optional SeedSequence to use for sampling. If None, the sampler's internal seed sequence will be used.
        Returns:
            Particle: A single particle sampled from the current particle population, represented as a Particle object.
        """
        return self._updater.sample_particle(seed_sequence)

    def sample_and_perturb_particle(
        self, seed_sequence: SeedSequence | None = None
    ) -> Particle:
        """
        Return a single particle sampled from the current population and perturbed based on the perturbation kernel, ensuring that the perturbed particle satisfies the prior probability density constraints. If a perturbed particle fails to meet the prior constraints, a new particle is sampled with replacement and perturbed until a valid particle is obtained or the maximum number of attempts is reached.
        Args:
            seed_sequence (SeedSequence | None): An optional SeedSequence to use for sampling. If None, the sampler's internal seed sequence will be used.
        Returns:
            Particle: A single particle sampled from the current population and perturbed based on the perturbation kernel.
        """
        return self._updater.sample_and_perturb_particle(
            max_attempts=self.max_attempts_per_proposal,
            seed_sequence=seed_sequence,
        )

    def particle_to_distance(self, particle: Particle, **kwargs: Any) -> float:
        """
        Computes the distance between the model output generated from the given particle and the target data using the user-supplied `particles_to_params` and `outputs_to_distance` functions.
        Args:
            particle (Particle): The particle for which to compute the distance.
            **kwargs (Any): Additional keyword arguments that can be passed to the `particles_to_params` function. These arguments are supplied from the `run()` method and can include any user-defined parameters needed for mapping particles to model parameters.
        Returns:
            float: The computed distance between the model output generated from the given particle and the target data, as calculated by the `outputs_to_distance` function.
        """
        params = self.particles_to_params(particle, **kwargs)
        outputs = self.model_runner.simulate(params)
        err = self.outputs_to_distance(outputs, self.target_data)
        return err

    def calculate_weight(self, particle: Particle) -> float:
        """
        Calculates the weight of a given particle based on its prior and perturbed probabilities using the particle updater's calculate_weight method.
        Args:
            particle (Particle): The particle for which to calculate the weight.
        Returns:
            float: The calculated weight of the particle, which is based on the prior probability density and the
        """
        return self._updater.calculate_weight(particle)

    def get_results_and_reset(
        self, perturbation_kernel: PerturbationKernel
    ) -> CalibrationResults:
        """
        Compiles the results of the calibration process into a CalibrationResults object and resets the sampler for potential future runs.
        Args:
            perturbation_kernel (PerturbationKernel): The originator perturbation kernel to reset to after the run.
        Returns:
            CalibrationResults: An object containing the results of the calibration process, including the final particle population
            and the history of successes and attempts for each generation.
        Raises:
            UserWarning: If the number of successful particles in any generation is less than the specified generation
        """
        if any(
            [
                count < self.generation_particle_count
                for count in self.step_successes
            ]
        ):
            raise UserWarning(
                "The number of successful particles in at least one generation is less than the specified generation_particle_count. This may indicate that the maximum particle proposal attempts are too low or the error tolerance values are too strict for the model and target data."
            )
        results = CalibrationResults(
            copy.deepcopy(self._updater),
            self.generator_history,
            self.population_archive,
            {
                "generation_particle_count": [self.generation_particle_count]
                * len(self.tolerance_values),
                "successes": self.step_successes,
                "attempts": self.step_attempts,
            },
            self.tolerance_values,
        )

        # Reset particle sampler and successes
        self.init_updater(perturbation_kernel)
        self.step_successes = [0] * len(self.tolerance_values)
        self.step_attempts = [0] * len(self.tolerance_values)
        self.generator_history = {}
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
        """
        Resolve the number of concurrent workers to use for parallel runs.
        """
        worker_count = (
            max_workers
            if max_workers is not None
            else self.parallel_worker_count
        )
        if worker_count <= 0:
            raise ValueError("max_workers must be positive")
        return worker_count

    def _run_coroutine(self, coroutine_factory: Callable[[], Any]) -> Any:
        """
        Run an async workflow from synchronous code, even if an event loop is already active.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine_factory())

        result: dict[str, Any] = {}
        error: dict[str, BaseException] = {}

        def runner():
            try:
                result["value"] = asyncio.run(coroutine_factory())
            except BaseException as exc:  # pragma: no cover - passthrough
                error["value"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()
        if "value" in error:
            raise error["value"]
        return result["value"]

    async def _collect_accepted_particles_async(
        self,
        generator_list: list[dict[str, Any]],
        tolerance: float,
        sample_method: Callable[[SeedSequence], Particle],
        executor: ThreadPoolExecutor,
        progress: Progress,
        progress_task_id: TaskID,
        generation_start_time: float,
        **kwargs: Any,
    ) -> tuple[list[tuple[int, Particle | None, int]], int]:
        """
        Collect accepted particles concurrently using async orchestration over a thread pool.
        """
        accepted_list: list[tuple[int, Particle | None, int]] = []
        total_attempts = 0
        completed = 0
        loop = asyncio.get_running_loop()
        worker = partial(
            self.sample_particles_until_accepted,
            tolerance=tolerance,
            sample_method=sample_method,
            **kwargs,
        )
        tasks = [
            loop.run_in_executor(executor, worker, generator)
            for generator in generator_list
        ]

        try:
            for task in asyncio.as_completed(tasks):
                completed_generator = await task
                accepted_list.append(completed_generator)
                total_attempts += completed_generator[2]
                completed += 1
                elapsed = time.time() - generation_start_time
                eta = (
                    elapsed
                    * (self.generation_particle_count - completed)
                    / (completed or 1)
                    if elapsed > 0 and completed > 0
                    else 0.0
                )
                acceptance_rate = (
                    100.0 * completed / total_attempts
                    if total_attempts > 0
                    else 0.0
                )
                progress.update(
                    progress_task_id,
                    completed=completed,
                    acceptance=f"{acceptance_rate:.1f}%",
                    eta=formatting._format_time(eta),
                )
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        return accepted_list, total_attempts

    def _validate_run_kwargs(self, kwargs: dict[str, Any]) -> None:
        """
        Validate that run-time kwargs do not shadow sampler attributes.
        """
        for key in kwargs:
            if key in self.__class__.__dict__:
                raise ValueError(
                    f"Keyword argument '{key}' conflicts with existing attribute. Please choose a different name for the argument. ABCSampler attributes cannot be set from `.run()`"
                )

    def _get_sample_method(
        self, generation: int
    ) -> Callable[[SeedSequence | None], Particle]:
        """
        Return the appropriate proposal method for the current generation.
        """
        if generation == 0:
            return self.sample_particle_from_priors
        return self.sample_and_perturb_particle

    def _init_particlewise_generation(
        self, generation: int
    ) -> tuple[
        ParticlePopulation, list[dict[str, Any]], Callable[..., Particle]
    ]:
        """
        Create particlewise per-generation state used by the sampling loop.
        """
        proposed_population = ParticlePopulation()
        generator_list = [
            {"id": i, "seed_sequence": seed_sequence}
            for i, seed_sequence in enumerate(
                self._seed_sequence.spawn(self.generation_particle_count)
            )
        ]
        sample_method = self._get_sample_method(generation)
        return proposed_population, generator_list, sample_method

    def _collect_accepted_particles_serial(
        self,
        generation: int,
        generator_list: list[dict[str, Any]],
        sample_method: Callable[[SeedSequence | None], Particle],
        progress: Progress,
        progress_task_id: TaskID,
        generation_start_time: float,
        **kwargs: Any,
    ) -> tuple[list[tuple[int, Particle | None, int]], int]:
        """
        Collect accepted particles serially while updating progress.
        """
        accepted_list: list[tuple[int, Particle | None, int]] = []
        total_attempts = 0

        for completed, generator in enumerate(generator_list, start=1):
            accepted_list.append(
                self.sample_particles_until_accepted(
                    generator=generator,
                    tolerance=self.tolerance_values[generation],
                    sample_method=sample_method,
                    **kwargs,
                )
            )
            total_attempts += accepted_list[-1][2]
            elapsed = time.time() - generation_start_time
            eta = (
                elapsed
                * (self.generation_particle_count - completed)
                / completed
                if elapsed > 0
                else 0.0
            )
            acceptance_rate = (
                100.0 * completed / total_attempts
                if total_attempts > 0
                else 0.0
            )
            progress.update(
                progress_task_id,
                completed=completed,
                acceptance=f"{acceptance_rate:.1f}%",
                eta=formatting._format_time(eta),
            )

        return accepted_list, total_attempts

    def _collect_accepted_particles(
        self,
        generation: int,
        generator_list: list[dict[str, Any]],
        sample_method: Callable[[SeedSequence | None], Particle],
        n_workers: int,
        parallel_executor: ThreadPoolExecutor | None,
        console: Any,
        overall_start_time: float,
        generation_start_time: float,
        **kwargs: Any,
    ) -> tuple[list[tuple[int, Particle | None, int]], float, float]:
        """
        Collect accepted particles for one generation and emit progress output.
        """
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("acceptance: {task.fields[acceptance]}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TextColumn("ETA: {task.fields[eta]}"),
            console=console,
            transient=True,
        ) as progress:
            progress_task_id = progress.add_task(
                f"Generation {generation + 1} (tolerance {self.tolerance_values[generation]})...",
                total=self.generation_particle_count,
                acceptance="N/A",
                eta="calculating...",
            )
            if n_workers == 1:
                accepted_list, total_attempts = (
                    self._collect_accepted_particles_serial(
                        generation=generation,
                        generator_list=generator_list,
                        sample_method=sample_method,
                        progress=progress,
                        progress_task_id=progress_task_id,
                        generation_start_time=generation_start_time,
                        **kwargs,
                    )
                )
            else:
                assert parallel_executor is not None
                accepted_list, total_attempts = self._run_coroutine(
                    lambda: self._collect_accepted_particles_async(
                        generator_list=generator_list,
                        tolerance=self.tolerance_values[generation],
                        sample_method=sample_method,
                        executor=parallel_executor,
                        progress=progress,
                        progress_task_id=progress_task_id,
                        generation_start_time=generation_start_time,
                        **kwargs,
                    )
                )

            processing_time = time.time() - generation_start_time
            total_time = time.time() - overall_start_time
            acceptance_rate = (
                100.0 * len(accepted_list) / total_attempts
                if total_attempts > 0
                else 0.0
            )
            console.print(
                f"[green]✓[/green] Generation {generation + 1} run complete! "
                f"Tolerance: {self.tolerance_values[generation]}, acceptance rate: {acceptance_rate:.1f}% of {total_attempts} attempts"
            )

        return accepted_list, processing_time, total_time

    def _finalize_generation(
        self,
        generation: int,
        generator_list: list[dict[str, Any]],
        accepted_list: list[tuple[int, Particle | None, int]],
        proposed_population: ParticlePopulation,
        console: Any,
        generation_start_time: float,
        processing_time: float,
        total_time: float,
    ) -> None:
        """
        Convert accepted proposals into the next particle population.
        """
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task(
                "Calculating weights...",
                total=self.generation_particle_count,
            )
            for id, accepted_particle, samples in sorted(
                accepted_list, key=lambda x: x[0]
            ):
                if accepted_particle is None:
                    raise UserWarning(
                        f"Particle proposal attempt {id} used {samples} samples and found no acceptable values."
                    )
                self.step_successes[generation] += 1
                particle_weight = (
                    1.0
                    if generation == 0
                    else self.calculate_weight(accepted_particle)
                )
                proposed_population.add_particle(
                    accepted_particle, particle_weight
                )
                self.step_attempts[generation] += samples
                progress.update(task_id, advance=1)

        self.generator_history.update({generation: generator_list})
        self.particle_population = proposed_population
        weights_time = time.time() - generation_start_time - processing_time
        console.print(
            f"(Run: {formatting._format_time(processing_time)}, Weights calculation: {formatting._format_time(weights_time)}, total time: {formatting._format_time(total_time)})"
        )

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
        console = formatting.get_console()
        overall_start_time = time.time()
        n_workers = (
            self._resolve_worker_count(max_workers)
            if execution == "parallel"
            else 1
        )
        parallel_executor = (
            ThreadPoolExecutor(max_workers=n_workers)
            if execution == "parallel" and n_workers > 1
            else None
        )

        try:
            for generation in range(len(self.tolerance_values)):
                generation_start_time = time.time()
                (
                    proposed_population,
                    generator_list,
                    sample_method,
                ) = self._init_particlewise_generation(generation)
                accepted_list, processing_time, total_time = (
                    self._collect_accepted_particles(
                        generation=generation,
                        generator_list=generator_list,
                        sample_method=sample_method,
                        n_workers=n_workers,
                        parallel_executor=parallel_executor,
                        console=console,
                        overall_start_time=overall_start_time,
                        generation_start_time=generation_start_time,
                        **kwargs,
                    )
                )
                self._finalize_generation(
                    generation=generation,
                    generator_list=generator_list,
                    accepted_list=accepted_list,
                    proposed_population=proposed_population,
                    console=console,
                    generation_start_time=generation_start_time,
                    processing_time=processing_time,
                    total_time=total_time,
                )
        finally:
            if parallel_executor is not None:
                parallel_executor.shutdown(wait=True)

        # Summary with checkmark
        console.print(
            f"[green]✓[/green] Calibration complete! "
            f"(total time: {formatting._format_time(total_time)})"
        )
        return self.get_results_and_reset(originator_perturbation_kernel)

    def sample_particles_until_accepted(
        self,
        generator: dict[str, int | SeedSequence],
        tolerance: float,
        sample_method: Callable[[SeedSequence], Particle],
        max_attempts: int | None = None,
        **kwargs: Any,
    ) -> tuple[int, Particle | None, int]:
        """
        Rejection sampling routine to return a single value

        Args:
            generator (dict[str, int | SeedSequence]): A dictionary containing the particle id and seed sequence generator for the random number generator spawn used in sampling.
            tolerance (float): The tolerance value for accepting a particle based on the error returned from particle_to_distance().
            sample_method (Callable[[SeedSequence], Particle]): The method used to sample particles, which can be either from the priors or by perturbing existing particles when called from the sampler SMC routine. Any method that accepts a seed sequence and returns a particle is valid.
            max_attempts (int | None): The maximum number of attempts to sample and perturb a particle before aborting. If None, it defaults to the sampler's `max_attempts_per_proposal` attribute.
            **kwargs (Any): Additional keyword arguments that can be passed to the `particles_to_params` function. These arguments are supplied from the `run()` method and can include any user-defined parameters needed for mapping particles to model parameters.
        Returns:
            tuple[int, Particle | None, int]: A tuple containing the particle id, the accepted particle (or None if no acceptable particle was found within the maximum attempts), and the number of samples taken to find an acceptable particle below the provided tolerance.
        """
        if not max_attempts:
            max_attempts = self.max_attempts_per_proposal

        for i in range(max_attempts):
            proposed_particle = sample_method(generator["seed_sequence"])
            err = self.particle_to_distance(proposed_particle, **kwargs)
            if err <= tolerance:
                return (generator["id"], proposed_particle, i + 1)
        return (generator["id"], None, max_attempts)

    def _resolve_parallel_batch_settings(
        self, batchsize: int | None, chunksize: int
    ) -> tuple[int, bool]:
        """
        Resolve validated batch execution settings.
        """
        if chunksize <= 0:
            raise ValueError("chunksize must be positive")
        if batchsize is None:
            return self.generation_particle_count, True
        if batchsize <= 0:
            raise ValueError("batchsize must be positive")
        return batchsize, False

    def _get_batch_sample_size(
        self,
        proposed_population: ParticlePopulation,
        attempts: int,
        batchsize: int,
        warmup: bool,
    ) -> int:
        """
        Estimate how many particles to propose for the next batch.
        """
        effective_batchsize = (
            10_000 if warmup and proposed_population.size > 0 else batchsize
        )
        if proposed_population.size == 0:
            return effective_batchsize

        remaining = self.generation_particle_count - proposed_population.size
        sample_size = min(
            effective_batchsize,
            remaining * attempts / proposed_population.size,
        )
        return max(int(sample_size), 1)

    def _sample_generation_particles(
        self, generation: int, sample_size: int
    ) -> list[Particle]:
        """
        Sample a batch of proposed particles for one generation.
        """
        sample_method = self._get_sample_method(generation)
        return [sample_method(None) for _ in range(sample_size)]

    def _evaluate_particle_chunk(
        self, proposed_particles: list[Particle], **kwargs: Any
    ) -> list[float]:
        """
        Evaluate a chunk of proposed particles serially.
        """
        return [
            self.particle_to_distance(proposed_particle, **kwargs)
            for proposed_particle in proposed_particles
        ]

    async def _evaluate_particle_batch_async(
        self,
        proposed_particles: list[Particle],
        executor: ThreadPoolExecutor,
        chunksize: int,
        **kwargs: Any,
    ) -> list[float]:
        """
        Evaluate a batch of proposed particles concurrently using async orchestration over a thread pool.
        """
        loop = asyncio.get_running_loop()
        worker = partial(self._evaluate_particle_chunk, **kwargs)
        particle_chunks = [
            proposed_particles[i : i + chunksize]
            for i in range(0, len(proposed_particles), chunksize)
        ]
        chunk_results = await asyncio.gather(
            *[
                loop.run_in_executor(executor, worker, particle_chunk)
                for particle_chunk in particle_chunks
            ]
        )
        return [err for chunk_result in chunk_results for err in chunk_result]

    def _evaluate_particle_batch(
        self,
        proposed_particles: list[Particle],
        executor: ThreadPoolExecutor | None,
        chunksize: int,
        **kwargs: Any,
    ) -> list[float]:
        """
        Evaluate a batch of proposed particles either serially or with async/threaded execution.
        """
        if executor is None or len(proposed_particles) <= chunksize:
            return self._evaluate_particle_chunk(proposed_particles, **kwargs)

        return self._run_coroutine(
            lambda: self._evaluate_particle_batch_async(
                proposed_particles=proposed_particles,
                executor=executor,
                chunksize=chunksize,
                **kwargs,
            )
        )

    def _accept_particle_batch(
        self,
        generation: int,
        proposed_population: ParticlePopulation,
        proposed_particles: list[Particle],
        errs: list[float],
    ) -> None:
        """
        Accept a batch of evaluated particles into the proposed population.
        """
        for err, proposed_particle in zip(errs, proposed_particles):
            if (
                err < self.tolerance_values[generation]
                and proposed_population.size < self.generation_particle_count
            ):
                particle_weight = (
                    1.0
                    if generation == 0
                    else self.calculate_weight(proposed_particle)
                )
                proposed_population.add_particle(
                    proposed_particle, particle_weight
                )

    def _run_generation_parallel_batches(
        self,
        generation: int,
        batchsize: int,
        warmup: bool,
        chunksize: int,
        executor: ThreadPoolExecutor | None,
        console: Any,
        overall_start_time: float,
        generation_start_time: float,
        **kwargs: Any,
    ) -> tuple[ParticlePopulation, int, float, float]:
        """
        Run one generation of the batched parallel sampler.
        """
        proposed_population = ParticlePopulation()
        attempts = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("acceptance: {task.fields[acceptance]}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TextColumn("ETA: {task.fields[eta]}"),
            console=console,
            transient=True,
        ) as progress:
            progress_task_id = progress.add_task(
                f"Generation {generation + 1} (tolerance {self.tolerance_values[generation]})...",
                total=self.generation_particle_count,
                acceptance="N/A",
                eta="calculating...",
            )

            while proposed_population.size < self.generation_particle_count:
                sample_size = self._get_batch_sample_size(
                    proposed_population=proposed_population,
                    attempts=attempts,
                    batchsize=batchsize,
                    warmup=warmup,
                )
                proposed_particles = self._sample_generation_particles(
                    generation=generation, sample_size=sample_size
                )
                errs = self._evaluate_particle_batch(
                    proposed_particles=proposed_particles,
                    executor=executor,
                    chunksize=chunksize,
                    **kwargs,
                )
                self._accept_particle_batch(
                    generation=generation,
                    proposed_population=proposed_population,
                    proposed_particles=proposed_particles,
                    errs=errs,
                )
                attempts += len(proposed_particles)

                elapsed = time.time() - generation_start_time
                completed = proposed_population.size
                eta = (
                    elapsed
                    * (self.generation_particle_count - completed)
                    / (completed or 1)
                    if elapsed > 0 and completed > 0
                    else 0.0
                )
                acceptance_rate = (
                    100.0 * completed / attempts if attempts > 0 else 0.0
                )
                progress.update(
                    progress_task_id,
                    completed=completed,
                    acceptance=f"{acceptance_rate:.1f}%",
                    eta=formatting._format_time(eta),
                )

            processing_time = time.time() - generation_start_time
            total_time = time.time() - overall_start_time
            acceptance_rate = (
                100.0 * proposed_population.size / attempts
                if attempts > 0
                else 0.0
            )
            console.print(
                f"[green]✓[/green] Generation {generation + 1} run complete! "
                f"Tolerance: {self.tolerance_values[generation]}, acceptance rate: {acceptance_rate:.1f}% of {attempts} attempts"
            )

        return proposed_population, attempts, processing_time, total_time

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
        resolved_batchsize, warmup = self._resolve_parallel_batch_settings(
            batchsize=batchsize, chunksize=chunksize
        )
        originator_perturbation_kernel = copy.deepcopy(
            self.perturbation_kernel
        )
        console = formatting.get_console()
        overall_start_time = time.time()
        executor = (
            ThreadPoolExecutor(max_workers=actual_workers)
            if actual_workers > 1
            else None
        )

        try:
            for generation in range(len(self.tolerance_values)):
                generation_start_time = time.time()
                proposed_population, attempts, processing_time, total_time = (
                    self._run_generation_parallel_batches(
                        generation=generation,
                        batchsize=resolved_batchsize,
                        warmup=warmup,
                        chunksize=chunksize,
                        executor=executor,
                        console=console,
                        overall_start_time=overall_start_time,
                        generation_start_time=generation_start_time,
                        **kwargs,
                    )
                )
                self.step_successes[generation] = proposed_population.size
                self.step_attempts[generation] = attempts
                self.particle_population = proposed_population
                console.print(
                    f"(Run: {formatting._format_time(processing_time)}, total time: {formatting._format_time(total_time)})"
                )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        console.print(
            f"[green]✓[/green] Calibration complete! "
            f"(total time: {formatting._format_time(total_time)})"
        )
        return self.get_results_and_reset(originator_perturbation_kernel)
