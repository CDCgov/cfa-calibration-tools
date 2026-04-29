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
    """Approximate Bayesian Computation Sequential Monte Carlo sampler."""

    def __init__(
        self,
        generation_particle_count: int,
        tolerance_values: list[float],
        priors: PriorDistribution | dict | Path,
        particles_to_params: Callable[..., dict],
        outputs_to_distance: Callable[..., float],
        target_data: Any,
        model_runner: object,
        perturbation_kernel: PerturbationKernel,
        variance_adapter: VarianceAdapter,
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
        if seed is not None:
            if entropy is not None and entropy != seed:
                raise ValueError(
                    "seed and entropy must match when both are set"
                )
            entropy = seed

        if max_concurrent_simulations is not None:
            if max_concurrent_simulations < 1:
                raise ValueError(
                    "max_concurrent_simulations must be at least 1"
                )
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

        if drop_previous_population_data is not None:
            keep_previous_population_data = not drop_previous_population_data

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
        else:  # pragma: no cover - defensive typing
            raise TypeError("Unsupported priors type")

        self._particle_evaluator = ParticleEvaluator(
            particles_to_params=particles_to_params,
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
        return self._run_state.step_successes

    @property
    def step_attempts(self) -> list[int]:
        return self._run_state.step_attempts

    @property
    def generator_history(self) -> dict[int, list[GeneratorSlot]]:
        return self._run_state.generator_history

    @property
    def population_archive(self) -> dict[int, ParticlePopulation]:
        return self._run_state.population_archive

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
        self._updater.particle_population = population

    def _replace_particle_population(
        self, population: ParticlePopulation
    ) -> None:
        self._run_state.archive_population(self.particle_population)
        self.particle_population = population

    @property
    def perturbation_kernel(self) -> PerturbationKernel:
        return self._updater.perturbation_kernel

    def _init_random(self) -> None:
        self._seed_sequence = SeedSequence(self.entropy)

    def set_updater(self, perturbation_kernel: PerturbationKernel) -> None:
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
        return self._priors.sample(
            n, self._resolve_seed_sequence(seed_sequence)
        )

    def sample_particle_from_priors(
        self, seed_sequence: SeedSequence | None = None
    ) -> Particle:
        return Particle(
            self.sample_priors(
                n=1,
                seed_sequence=self._resolve_seed_sequence(seed_sequence),
            )[0]
        )

    def sample_particle(
        self, seed_sequence: SeedSequence | None = None
    ) -> Particle:
        return self._updater.sample_particle(
            self._resolve_seed_sequence(seed_sequence)
        )

    def sample_and_perturb_particle(
        self, seed_sequence: SeedSequence | None = None
    ) -> Particle:
        return self._updater.sample_and_perturb_particle(
            max_attempts=self.max_attempts_per_proposal,
            seed_sequence=self._resolve_seed_sequence(seed_sequence),
        )

    def particle_to_distance(self, particle: Particle, **kwargs: Any) -> float:
        return self._particle_evaluator.distance(particle, **kwargs)

    async def particle_to_distance_async(
        self,
        particle: Particle,
        **kwargs: Any,
    ) -> float:
        return await self._particle_evaluator.distance_async(
            particle, **kwargs
        )

    def calculate_weight(self, particle: Particle) -> float:
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
        results = self._build_results()
        self._last_results = results
        self._reset_after_run(perturbation_kernel)
        return results

    def run_parallel(
        self, max_workers: int | None = None, **kwargs: Any
    ) -> CalibrationResults:
        return self.run(
            execution="parallel", max_workers=max_workers, **kwargs
        )

    def run_serial(self, **kwargs: Any) -> CalibrationResults:
        return self.run(execution="serial", **kwargs)

    async def run_async(
        self,
        execution: Literal["serial", "parallel"] = "parallel",
        max_workers: int | None = None,
        **kwargs: Any,
    ) -> CalibrationResults:
        return await asyncio.to_thread(
            lambda: self.run(
                execution=execution,
                max_workers=max_workers,
                **kwargs,
            )
        )

    def _resolve_worker_count(self, max_workers: int | None) -> int:
        worker_count = (
            max_workers
            if max_workers is not None
            else self.parallel_worker_count
        )
        if worker_count <= 0:
            raise ValueError("max_workers must be positive")
        return worker_count

    def _build_reporter(self) -> SamplerReporter:
        return SamplerReporter(
            verbose=self.verbose or self.print_generation_progress
        )

    def _build_particlewise_generation_runner(
        self,
        reporter: SamplerReporter,
    ) -> ParticlewiseGenerationRunner:
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

    def _build_results(self) -> CalibrationResults:
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
            self.tolerance_values,
            result_seed_sequence,
        )

    def _reset_after_run(
        self,
        perturbation_kernel: PerturbationKernel,
    ) -> None:
        self._init_random()
        self.set_updater(perturbation_kernel)
        self._run_state.reset()

    def run(
        self,
        execution: Literal["serial", "parallel"] = "parallel",
        max_workers: int | None = None,
        **kwargs: Any,
    ) -> CalibrationResults:
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
        particlewise_runner = self._build_particlewise_generation_runner(
            reporter=reporter
        )
        parallel_executor = (
            ThreadPoolExecutor(max_workers=n_workers)
            if execution == "parallel"
            and n_workers > 1
            and not use_native_async
            else None
        )

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
        executor = (
            ThreadPoolExecutor(max_workers=actual_workers)
            if actual_workers > 1
            else None
        )

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
