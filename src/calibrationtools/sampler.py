import asyncio
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
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
from .particle_reader import ParticleReader
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


def _resolve_sampler_entropy(
    *,
    seed: int | None,
    entropy: int | None,
) -> int | None:
    if seed is None:
        return entropy
    if entropy is not None and entropy != seed:
        raise ValueError("seed and entropy must match when both are set")
    return seed


def _resolve_parallel_worker_count(
    *,
    parallel_worker_count: int,
    max_concurrent_simulations: int | None,
) -> int:
    if max_concurrent_simulations is not None:
        if max_concurrent_simulations < 1:
            raise ValueError("max_concurrent_simulations must be at least 1")
        if (
            parallel_worker_count != 10
            and parallel_worker_count != max_concurrent_simulations
        ):
            raise ValueError(
                "parallel_worker_count and max_concurrent_simulations must match when both are provided"
            )
        parallel_worker_count = max_concurrent_simulations

    if parallel_worker_count <= 0:
        raise ValueError("parallel_worker_count must be positive")
    return parallel_worker_count


def _resolve_population_archive_flag(
    *,
    keep_previous_population_data: bool,
    drop_previous_population_data: bool | None,
) -> bool:
    if drop_previous_population_data is None:
        return keep_previous_population_data
    return not drop_previous_population_data


def _load_prior_distribution(
    priors: PriorDistribution | dict[str, Any] | Path | str,
    *,
    seed_parameter_name: str | None,
) -> PriorDistribution:
    if isinstance(priors, PriorDistribution):
        return priors
    if isinstance(priors, dict):
        from .load_priors import independent_priors_from_dict

        return independent_priors_from_dict(
            priors,
            incl_seed_parameter=seed_parameter_name is not None,
            seed_parameter_name=seed_parameter_name,
        )
    if isinstance(priors, Path) or isinstance(priors, str):
        from .load_priors import load_priors_from_json

        return load_priors_from_json(priors)
    raise TypeError("Unsupported priors type")


def _build_particle_reader(
    priors: PriorDistribution,
    *,
    default_parameters: dict[str, Any] | None,
    particles_to_params: Callable[..., dict] | None,
) -> ParticleReader:
    return ParticleReader(
        particle_param_names=priors.params,
        default_params=default_parameters,
        read_fn=particles_to_params,
    )


def _build_particle_evaluator(
    *,
    particle_reader: ParticleReader,
    outputs_to_distance: Callable[..., float],
    target_data: Any,
    model_runner: object,
    artifacts_dir: Path | None,
) -> ParticleEvaluator:
    return ParticleEvaluator(
        particle_reader=particle_reader,
        outputs_to_distance=outputs_to_distance,
        target_data=target_data,
        model_runner=model_runner,
        artifacts_dir=artifacts_dir,
    )


class ABCSampler:
    """Approximate Bayesian Computation Sequential Monte Carlo sampler.

    `ABCSampler` estimates posterior parameter distributions by iteratively
    proposing particles, running a model for each proposal, scoring simulated
    outputs against observed data, and accepting particles that satisfy the
    generation tolerance.

    Args:
        generation_particle_count (int): Number of particles to accept per
            generation for a complete population.
        tolerance_values (list[float]): Tolerance threshold for each SMC
            generation.
        priors (PriorDistribution | dict | Path): Parameter priors. Priors can
            be supplied as a distribution object, a priors-schema dictionary,
            or a path to a priors JSON file.
        particles_to_params (Callable[..., dict] | None): Optional function
            that maps a `Particle` into model parameters. When omitted,
            particle values are read directly through `ParticleReader`.
        outputs_to_distance (Callable[..., float] | None): Function that
            scores model outputs against `target_data`.
        target_data (Any): Observed data used by `outputs_to_distance`.
        model_runner (object | None): Runner object defining `simulate()`,
            `simulate_async()`, or both.
        perturbation_kernel (PerturbationKernel | None): Initial kernel used
            to perturb accepted particles across SMC generations.
        variance_adapter (VarianceAdapter | None): Adapter that updates the
            perturbation scale from accepted populations.
        default_parameters (dict[str, Any] | None): Optional nested defaults
            merged with particle values before model execution.
        max_attempts_per_proposal (int): Maximum sample-and-perturb attempts
            used to produce one valid proposal.
        max_proposals_per_batch (int): Maximum proposed particles generated in
            one batch when using batched parallel execution.
        parallel_worker_count (int): Default worker count for parallel runs.
        max_concurrent_simulations (int | None): Alias for
            `parallel_worker_count` used by runner-oriented callers.
        entropy (int | None): Entropy used to initialize sampler randomness.
        seed (int | None): Alias for `entropy`; when both are supplied they
            must match.
        verbose (bool): Whether to print run summaries.
        print_generation_progress (bool): Whether to print generation-level
            progress even when `verbose` is false.
        keep_previous_population_data (bool): Whether to retain previous
            populations in the per-run archive.
        drop_previous_population_data (bool | None): Deprecated inverse of
            `keep_previous_population_data`.
        results_inherit_entropy_only (bool): Whether posterior result sampling
            reuses only the sampler entropy or spawns from the sampler seed
            sequence.
        seed_parameter_name (str | None): Name of the seed parameter to add
            when loading priors from dictionaries or JSON. Set to `None` to
            skip adding a seed parameter.
        artifacts_dir (Path | str | None): Optional directory for staged model
            inputs and outputs.

    Raises:
        TypeError: If required callbacks or runner objects are omitted.
        ValueError: If worker-count or seed aliases conflict.
    """

    def __init__(
        self,
        generation_particle_count: int,
        tolerance_values: list[float],
        priors: PriorDistribution | dict | Path,
        particles_to_params: Callable[..., dict] | None = None,
        outputs_to_distance: Callable[..., float] | None = None,
        target_data: Any = None,
        model_runner: object | None = None,
        perturbation_kernel: PerturbationKernel | None = None,
        variance_adapter: VarianceAdapter | None = None,
        default_parameters: dict[str, Any] | None = None,
        max_attempts_per_proposal: int = np.iinfo(np.int32).max,
        max_proposals_per_batch: int = 10_000,
        parallel_worker_count: int = 10,
        max_concurrent_simulations: int | None = None,
        entropy: int | None = None,
        seed: int | None = None,
        verbose: bool = True,
        print_generation_progress: bool = False,
        keep_previous_population_data: bool = False,
        drop_previous_population_data: bool | None = None,
        results_inherit_entropy_only: bool = True,
        seed_parameter_name: str | None = "seed",
        artifacts_dir: Path | str | None = None,
    ):
        if outputs_to_distance is None:
            raise TypeError("outputs_to_distance is required")
        if model_runner is None:
            raise TypeError("model_runner is required")
        if perturbation_kernel is None:
            raise TypeError("perturbation_kernel is required")
        if variance_adapter is None:
            raise TypeError("variance_adapter is required")

        if seed is not None and entropy is not None and entropy != seed:
            raise ValueError("seed and entropy must match when both are set")
        entropy = _resolve_sampler_entropy(seed=seed, entropy=entropy)
        parallel_worker_count = _resolve_parallel_worker_count(
            parallel_worker_count=parallel_worker_count,
            max_concurrent_simulations=max_concurrent_simulations,
        )
        keep_previous_population_data = _resolve_population_archive_flag(
            keep_previous_population_data=keep_previous_population_data,
            drop_previous_population_data=drop_previous_population_data,
        )

        self.generation_particle_count = generation_particle_count
        self.max_attempts_per_proposal = max_attempts_per_proposal
        self.max_proposals_per_batch = max_proposals_per_batch
        self.parallel_worker_count = parallel_worker_count
        self.max_concurrent_simulations = parallel_worker_count
        self.tolerance_values = tolerance_values
        self._variance_adapter = variance_adapter
        self.particles_to_params = particles_to_params
        self.outputs_to_distance = outputs_to_distance
        self.target_data = target_data
        self.model_runner = model_runner
        self.entropy = entropy
        self.seed = entropy
        self.verbose = verbose
        self.print_generation_progress = print_generation_progress
        self.keep_previous_population_data = keep_previous_population_data
        self.results_inherit_entropy_only = results_inherit_entropy_only
        self.artifacts_dir = (
            Path(artifacts_dir) if artifacts_dir is not None else None
        )
        self._last_results: CalibrationResults | None = None

        self._priors = _load_prior_distribution(
            priors,
            seed_parameter_name=seed_parameter_name,
        )
        self.particle_reader = _build_particle_reader(
            self._priors,
            default_parameters=default_parameters,
            particles_to_params=self.particles_to_params,
        )
        self._particle_evaluator = _build_particle_evaluator(
            particle_reader=self.particle_reader,
            outputs_to_distance=outputs_to_distance,
            target_data=target_data,
            model_runner=model_runner,
            artifacts_dir=self.artifacts_dir,
        )
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
        """

        self._updater.particle_population = population

    def _replace_particle_population(
        self, population: ParticlePopulation
    ) -> None:
        """Archive the current population in run state, then store the new one.

        Args:
            population (ParticlePopulation): New population to store on the
                sampler.
        """

        self._run_state.archive_population(self.particle_population)
        self.particle_population = population

    @property
    def perturbation_kernel(self) -> PerturbationKernel:
        return self._updater.perturbation_kernel

    def _init_random(self) -> None:
        """Initialize the sampler seed sequence from sampler entropy."""

        self._seed_sequence = SeedSequence(self.entropy)

    def set_updater(self, perturbation_kernel: PerturbationKernel) -> None:
        """Initialize the particle updater for a new sampler run.

        Args:
            perturbation_kernel (PerturbationKernel): Kernel used to perturb
                particles across generations.
        """

        self._updater = _ParticleUpdater(
            perturbation_kernel,
            self._priors,
            self._variance_adapter,
            ParticlePopulation(),
        )

    def _resolve_seed_sequence(
        self, seed_sequence: SeedSequence | None
    ) -> SeedSequence:
        return self._seed_sequence if seed_sequence is None else seed_sequence

    def sample_priors(
        self,
        n: int = 1,
        seed_sequence: SeedSequence | None = None,
    ) -> Sequence[dict[str, Any]]:
        """Return states sampled from the prior distribution.

        Args:
            n (int): Number of prior samples to draw.
            seed_sequence (SeedSequence | None): Seed sequence for random
                generation. When omitted, the sampler seed sequence is used.

        Returns:
            Sequence[dict[str, Any]]: Sampled prior states.
        """

        return self._priors.sample(
            n, self._resolve_seed_sequence(seed_sequence)
        )

    def sample_particle_from_priors(
        self, seed_sequence: SeedSequence | None = None
    ) -> Particle:
        """Return one particle sampled directly from the prior distribution.

        Args:
            seed_sequence (SeedSequence | None): Seed sequence for random
                generation. When omitted, the sampler seed sequence is used.

        Returns:
            Particle: Particle initialized from one prior sample.
        """

        return Particle(
            self.sample_priors(
                n=1,
                seed_sequence=self._resolve_seed_sequence(seed_sequence),
            )[0]
        )

    def sample_particle(
        self, seed_sequence: SeedSequence | None = None
    ) -> Particle:
        """Return one particle sampled from the current population.

        Args:
            seed_sequence (SeedSequence | None): Seed sequence for weighted
                population sampling. When omitted, the sampler seed sequence is
                used.

        Returns:
            Particle: Particle sampled from the current population.
        """

        return self._updater.sample_particle(
            self._resolve_seed_sequence(seed_sequence)
        )

    def sample_and_perturb_particle(
        self, seed_sequence: SeedSequence | None = None
    ) -> Particle:
        """Sample and perturb one particle from the current population.

        Sampling retries until the perturbed particle satisfies the prior
        density constraints or `max_attempts_per_proposal` is exhausted.

        Args:
            seed_sequence (SeedSequence | None): Seed sequence for sampling and
                perturbation. When omitted, the sampler seed sequence is used.

        Returns:
            Particle: Valid perturbed particle.
        """

        return self._updater.sample_and_perturb_particle(
            max_attempts=self.max_attempts_per_proposal,
            seed_sequence=self._resolve_seed_sequence(seed_sequence),
        )

    def particle_to_distance(self, particle: Particle, **kwargs: Any) -> float:
        """Compute the distance for one proposed particle.

        This keeps `ABCSampler` as the public particle-evaluation entry point
        while delegating parameter conversion, model execution, artifact
        staging, and scoring to `ParticleEvaluator`.

        Args:
            particle (Particle): Particle to evaluate.
            **kwargs (Any): Additional keyword arguments forwarded to the
                particle reader.

        Returns:
            float: Distance between simulated outputs and target data.
        """

        return self._particle_evaluator.distance(particle, **kwargs)

    async def particle_to_distance_async(
        self,
        particle: Particle,
        **kwargs: Any,
    ) -> float:
        """Asynchronously compute the distance for one proposed particle.

        Args:
            particle (Particle): Particle to evaluate.
            **kwargs (Any): Additional keyword arguments forwarded to the
                particle reader.

        Returns:
            float: Distance between simulated outputs and target data.
        """

        return await self._particle_evaluator.distance_async(
            particle, **kwargs
        )

    def calculate_weight(self, particle: Particle) -> float:
        """Calculate the importance weight for one accepted particle.

        Args:
            particle (Particle): Particle for which to calculate a weight.

        Returns:
            float: Importance weight under the current population and
                perturbation kernel.
        """

        return self._updater.calculate_weight(particle)

    def get_posterior_particles(self) -> ParticlePopulation:
        if self._last_results is not None:
            return self._last_results.posterior_particles
        if any(
            count < self.generation_particle_count
            for count in self.step_successes
        ):
            raise ValueError(
                "Posterior population is not fully populated. Please run the sampler to completion before accessing the posterior population."
            )
        return self.particle_population

    def get_results_and_reset(
        self, perturbation_kernel: PerturbationKernel
    ) -> CalibrationResults:
        """Build calibration results and reset mutable sampler state.

        Args:
            perturbation_kernel (PerturbationKernel): Perturbation kernel to
                restore after result construction.

        Returns:
            CalibrationResults: Immutable snapshot for the completed run.
        """

        results = self._build_results()
        self._last_results = results
        self._reset_after_run(perturbation_kernel)
        return results

    def run_parallel(
        self, max_workers: int | None = None, **kwargs: Any
    ) -> CalibrationResults:
        """Run the ABC-SMC sampler with parallel particle evaluation.

        Args:
            max_workers (int | None): Worker-count override. When omitted, the
                sampler's configured `parallel_worker_count` is used.
            **kwargs (Any): Additional keyword arguments forwarded to particle
                evaluation. These must not conflict with sampler attributes.

        Returns:
            CalibrationResults: Results for the completed calibration run.
        """

        return self.run(
            execution="parallel", max_workers=max_workers, **kwargs
        )

    def run_serial(self, **kwargs: Any) -> CalibrationResults:
        """Run the ABC-SMC sampler with serial particle evaluation.

        Args:
            **kwargs (Any): Additional keyword arguments forwarded to particle
                evaluation. These must not conflict with sampler attributes.

        Returns:
            CalibrationResults: Results for the completed calibration run.
        """

        return self.run(execution="serial", **kwargs)

    async def run_async(
        self,
        execution: Literal["serial", "parallel"] = "parallel",
        max_workers: int | None = None,
        **kwargs: Any,
    ) -> CalibrationResults:
        """Run the sampler from async code without blocking the event loop.

        The synchronous sampler run is moved to a worker thread. Model runners
        that prefer native async simulation are still honored inside the run.

        Args:
            execution (Literal["serial", "parallel"]): Execution mode.
            max_workers (int | None): Worker-count override for parallel mode.
            **kwargs (Any): Additional keyword arguments forwarded to particle
                evaluation.

        Returns:
            CalibrationResults: Results for the completed calibration run.
        """

        return await asyncio.to_thread(
            lambda: self.run(
                execution=execution,
                max_workers=max_workers,
                **kwargs,
            )
        )

    def _resolve_worker_count(self, max_workers: int | None) -> int:
        """Resolve and validate the worker count for a parallel run.

        Args:
            max_workers (int | None): Optional worker-count override.

        Returns:
            int: Positive worker count.

        Raises:
            ValueError: If the resolved worker count is not positive.
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

        Returns:
            SamplerReporter: Reporter configured for summary and progress
                output.
        """

        return SamplerReporter(
            verbose=self.verbose or self.print_generation_progress
        )

    def _build_particlewise_generation_runner(
        self,
        reporter: SamplerReporter,
    ) -> ParticlewiseGenerationRunner:
        """Create the particlewise execution engine for the active run.

        Args:
            reporter (SamplerReporter): Reporter used for progress and summary
                output.

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
                particle_to_distance_async=(
                    self.particle_to_distance_async
                    if self._uses_native_async_collection()
                    else None
                ),
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

        Args:
            reporter (SamplerReporter): Reporter used for progress and summary
                output.

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

        Args:
            kwargs (dict[str, Any]): Keyword arguments supplied to a run method.

        Raises:
            ValueError: If a keyword collides with a sampler attribute.
        """

        for key in kwargs:
            if key in self.__class__.__dict__:
                raise ValueError(
                    f"Keyword argument '{key}' conflicts with existing attribute. Please choose a different name for the argument. ABCSampler attributes cannot be set from `.run()`"
                )

    def _uses_native_async_collection(self) -> bool:
        return bool(
            getattr(self.model_runner, "prefer_simulate_async", False)
            and callable(getattr(self.model_runner, "simulate_async", None))
        )

    def _resolve_native_async_worker_count(self, worker_count: int) -> int:
        """Return native-async submission width, including runner buffer."""

        dispatch_buffer_size = getattr(
            self.model_runner,
            "dispatch_buffer_size",
            None,
        )
        if dispatch_buffer_size is None:
            return worker_count
        buffer_size = (
            dispatch_buffer_size()
            if callable(dispatch_buffer_size)
            else dispatch_buffer_size
        )
        if buffer_size is None:
            return worker_count
        buffer_size = int(buffer_size)
        if buffer_size < 0:
            raise ValueError("dispatch_buffer_size must be non-negative")
        return worker_count + buffer_size

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

        Returns:
            CalibrationResults: Snapshot containing the final posterior,
                generation history, archive data, and success statistics.

        Raises:
            UserWarning: If any generation finished with fewer accepted
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
            result_seed_sequence = SeedSequence(self.entropy)
        else:
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

        Args:
            perturbation_kernel (PerturbationKernel): Perturbation kernel to
                restore on the sampler.
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
        """Execute the ABC-SMC sampling process.

        Args:
            execution (Literal["serial", "parallel"]): Whether to evaluate
                particles serially or in parallel.
            max_workers (int | None): Worker-count override for parallel mode.
            **kwargs (Any): Additional keyword arguments forwarded to particle
                evaluation. These must not conflict with sampler attributes.

        Returns:
            CalibrationResults: Results for the completed calibration run.
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
        use_native_async = (
            execution == "parallel" and self._uses_native_async_collection()
        )
        collection_workers = (
            self._resolve_native_async_worker_count(n_workers)
            if use_native_async
            else n_workers
        )
        particlewise_runner = self._build_particlewise_generation_runner(
            reporter=reporter
        )
        parallel_executor = (
            None
            if use_native_async
            else self._build_executor(max_workers=n_workers)
        )

        try:
            for generation in range(len(self.tolerance_values)):
                generation_stats = particlewise_runner.run_generation(
                    ParticlewiseGenerationRequest(
                        generation=generation,
                        n_workers=collection_workers,
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
        """Run the ABC-SMC sampler with batched parallel evaluation.

        Args:
            chunksize (int): Approximate number of parameter sets processed in
                each serial chunk inside a worker.
            batchsize (int | None): Number of proposed particles generated per
                batch. When omitted, the batch runner chooses a default from
                sampler settings.
            max_workers (int | None): Worker-count override. When omitted, the
                sampler's configured `parallel_worker_count` is used.
            **kwargs (Any): Additional keyword arguments forwarded to particle
                evaluation. These must not conflict with sampler attributes.

        Returns:
            CalibrationResults: Results for the completed calibration run.
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
                generation_stats = batch_runner.run_generation(
                    BatchGenerationRequest(
                        generation=generation,
                        batchsize=resolved_batchsize,
                        warmup=warmup,
                        chunksize=chunksize,
                        executor=executor,
                        overall_start_time=overall_start_time,
                        generation_start_time=time.time(),
                        particle_kwargs=dict(kwargs),
                    )
                )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        reporter.print_run_summary(generation_stats.total_time)
        return self.get_results_and_reset(originator_perturbation_kernel)
