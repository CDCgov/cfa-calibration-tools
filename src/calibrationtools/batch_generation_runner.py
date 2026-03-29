"""Run batched ABC-SMC generations outside the sampler facade.

This module contains the execution engine for the batched sampling path. It
keeps batch proposal sizing, evaluation, acceptance, and progress reporting
out of `ABCSampler`.
"""

import asyncio
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from .async_runner import run_coroutine_from_sync
from .particle import Particle
from .particle_population import ParticlePopulation
from .sampler_reporting import ProgressHandle, SamplerReporter
from .sampler_run_state import SamplerRunState
from .sampler_types import BatchGenerationRequest, GenerationStats


@dataclass(frozen=True, slots=True)
class BatchGenerationConfig:
    """Store static settings and callbacks for batched generations.

    This configuration object groups the sampler collaborators that remain
    stable across batched generations so the runner constructor stays small
    and execution methods operate on named fields.

    Attributes:
        generation_particle_count (int): Number of accepted particles required
            to complete a generation.
        tolerance_values (list[float]): Acceptance tolerance for each
            generation.
        sample_particle_from_priors (Callable[[Any], Particle]): Proposal
            function for the initial generation.
        sample_and_perturb_particle (Callable[[Any], Particle]): Proposal
            function for later generations.
        particle_to_distance (Callable[..., float]): Function used to evaluate
            the distance of one proposed particle.
        calculate_weight (Callable[[Particle], float]): Function used to weight
            accepted particles after the first generation.
        replace_particle_population (Callable[[ParticlePopulation], None]):
            Callback that stores the finalized population on the sampler.
        reporter (SamplerReporter): Reporter used for progress and summary
            output.
    """

    generation_particle_count: int
    tolerance_values: list[float]
    sample_particle_from_priors: Callable[[Any], Particle]
    sample_and_perturb_particle: Callable[[Any], Particle]
    particle_to_distance: Callable[..., float]
    calculate_weight: Callable[[Particle], float]
    replace_particle_population: Callable[[ParticlePopulation], None]
    reporter: SamplerReporter


@dataclass(slots=True)
class BatchGenerationState:
    """Store mutable state for one batched generation.

    This state groups the accepted population and the running attempt count so
    helper methods can share batch-generation state without long parameter
    lists.

    Attributes:
        proposed_population (ParticlePopulation): Population being filled for
            the active generation.
        attempts (int): Total proposal attempts consumed so far.
    """

    proposed_population: ParticlePopulation
    attempts: int = 0


class BatchGenerationRunner:
    """Run one batched ABC-SMC generation in serial or threaded mode.

    This runner isolates the batched execution engine from `ABCSampler`. It
    handles batch proposal sizing, chunk evaluation, acceptance accounting,
    and generation-level run-state updates.

    Args:
        config (BatchGenerationConfig): Static settings and callbacks used
            across batched generations.
        run_state (SamplerRunState): Mutable bookkeeping for the active sampler
            run.
    """

    def __init__(
        self,
        config: BatchGenerationConfig,
        run_state: SamplerRunState,
    ) -> None:
        self.config = config
        self.run_state = run_state

    def resolve_settings(
        self,
        batchsize: int | None,
        chunksize: int,
    ) -> tuple[int, bool]:
        """Resolve validated batch-execution settings.

        This helper normalizes the requested batch configuration and returns
        both the resolved batch size and whether warmup estimation should be
        used for the first adaptive batch.

        Args:
            batchsize (int | None): Optional batch-size override.
            chunksize (int): Number of particles processed serially inside one
                executor task.

        Returns:
            tuple[int, bool]: Resolved batch size and whether warmup mode is
                enabled.

        Raises:
            ValueError: Raised when `chunksize` or an explicit `batchsize` is
                not positive.
        """

        if chunksize <= 0:
            raise ValueError("chunksize must be positive")
        if batchsize is None:
            return self.config.generation_particle_count, True
        if batchsize <= 0:
            raise ValueError("batchsize must be positive")
        return batchsize, False

    def run_generation(
        self, request: BatchGenerationRequest
    ) -> GenerationStats:
        """Execute one batched generation and store its final population.

        This method coordinates adaptive proposal sizing, batch evaluation,
        acceptance accounting, and final population storage for the batched
        sampler path.

        Args:
            request (BatchGenerationRequest): Runtime inputs for the generation
                being executed.

        Returns:
            GenerationStats: Attempts, successes, and timing metrics recorded
                for the completed generation.
        """

        state = BatchGenerationState(proposed_population=ParticlePopulation())
        description = (
            f"Generation {request.generation + 1} "
            f"(tolerance {self.config.tolerance_values[request.generation]})..."
        )
        with self.config.reporter.create_collection_progress() as progress:
            handle = self.config.reporter.start_collection_task(
                progress=progress,
                description=description,
                total=self.config.generation_particle_count,
            )
            while (
                state.proposed_population.size
                < self.config.generation_particle_count
            ):
                sample_size = self._get_batch_sample_size(
                    state=state,
                    batchsize=request.batchsize,
                    warmup=request.warmup,
                )
                proposed_particles = self._sample_generation_particles(
                    generation=request.generation,
                    sample_size=sample_size,
                )
                state.attempts += self._process_particle_batch(
                    request=request,
                    state=state,
                    proposed_particles=proposed_particles,
                )
                self._update_progress(
                    handle=handle,
                    state=state,
                    generation_start_time=request.generation_start_time,
                )

            generation_stats = self._build_generation_stats(
                request=request,
                state=state,
            )
            self.config.reporter.print_generation_summary(
                generation=request.generation,
                tolerance=self.config.tolerance_values[request.generation],
                generation_stats=generation_stats,
            )

        self.run_state.record_attempts(
            generation=request.generation,
            attempts=generation_stats.attempts,
            successes=generation_stats.successes,
        )
        self.config.replace_particle_population(state.proposed_population)
        self.config.reporter.print_timing_summary(
            processing_time=generation_stats.processing_time,
            total_time=generation_stats.total_time,
        )
        return generation_stats

    def _get_batch_sample_size(
        self,
        state: BatchGenerationState,
        batchsize: int,
        warmup: bool,
    ) -> int:
        """Estimate the proposal count for the next batch.

        This helper increases the early batch size during warmup and then
        adapts future proposal counts using the observed acceptance rate from
        the current generation.

        Args:
            state (BatchGenerationState): Mutable state for the active
                generation.
            batchsize (int): Configured batch size for normal operation.
            warmup (bool): Whether the warmup heuristic should still apply.

        Returns:
            int: Number of particles to propose for the next batch.
        """

        effective_batchsize = (
            10_000
            if warmup and state.proposed_population.size > 0
            else batchsize
        )
        if state.proposed_population.size == 0:
            return effective_batchsize

        remaining = (
            self.config.generation_particle_count
            - state.proposed_population.size
        )
        sample_size = min(
            effective_batchsize,
            remaining * state.attempts / state.proposed_population.size,
        )
        return max(int(sample_size), 1)

    def _sample_generation_particles(
        self,
        generation: int,
        sample_size: int,
    ) -> list[Particle]:
        """Sample a batch of proposed particles for one generation.

        This helper chooses the correct proposal function for the generation
        and returns the requested number of proposed particles.

        Args:
            generation (int): Zero-based generation index being sampled.
            sample_size (int): Number of particles to propose.

        Returns:
            list[Particle]: Proposed particles for the batch.
        """

        sample_method = (
            self.config.sample_particle_from_priors
            if generation == 0
            else self.config.sample_and_perturb_particle
        )
        return [sample_method(None) for _ in range(sample_size)]

    def _evaluate_particle_chunk(
        self,
        proposed_particles: list[Particle],
        particle_kwargs: dict[str, Any],
    ) -> list[float]:
        """Evaluate a chunk of proposed particles serially.

        This helper keeps chunk evaluation reusable between the serial and
        threaded batch-processing paths.

        Args:
            proposed_particles (list[Particle]): Proposed particles to score.
            particle_kwargs (dict[str, Any]): Additional keyword arguments
                forwarded into particle evaluation.

        Returns:
            list[float]: Distances computed for the proposed particles.
        """

        return [
            self.config.particle_to_distance(
                proposed_particle,
                **particle_kwargs,
            )
            for proposed_particle in proposed_particles
        ]

    async def _process_particle_batch_async(
        self,
        request: BatchGenerationRequest,
        state: BatchGenerationState,
        proposed_particles: list[Particle],
    ) -> int:
        """Evaluate and accept one batch of particles concurrently.

        This helper splits proposed particles into chunks, evaluates each chunk
        on the executor, and accepts particles in chunk order until the
        generation population is full or all proposed particles are
        considered.

        Args:
            request (BatchGenerationRequest): Runtime inputs for the generation
                being executed.
            state (BatchGenerationState): Mutable state for the active
                generation.
            proposed_particles (list[Particle]): Proposed particles to
                evaluate.

        Returns:
            int: Number of proposed particles that were considered.
        """

        assert request.executor is not None

        loop = asyncio.get_running_loop()
        worker = partial(
            self._evaluate_particle_chunk,
            particle_kwargs=request.particle_kwargs,
        )
        particle_chunks = [
            proposed_particles[index : index + request.chunksize]
            for index in range(0, len(proposed_particles), request.chunksize)
        ]
        tasks = []
        for chunk in particle_chunks:
            task = loop.run_in_executor(request.executor, worker, chunk)
            tasks.append((task, chunk))

        attempts = 0
        try:
            for task, chunk in tasks:
                chunk_results = await task
                attempts += self._accept_particle_batch(
                    generation=request.generation,
                    proposed_population=state.proposed_population,
                    proposed_particles=chunk,
                    errs=chunk_results,
                )
                if (
                    state.proposed_population.size
                    >= self.config.generation_particle_count
                ):
                    break
        finally:
            for task, _ in tasks:
                task.cancel()

        return attempts

    def _process_particle_batch(
        self,
        request: BatchGenerationRequest,
        state: BatchGenerationState,
        proposed_particles: list[Particle],
    ) -> int:
        """Evaluate and accept one batch of proposed particles.

        This helper dispatches batch evaluation either serially or through the
        executor-backed async path, then returns how many proposed particles
        were considered before the population filled or the batch was
        exhausted.

        Args:
            request (BatchGenerationRequest): Runtime inputs for the generation
                being executed.
            state (BatchGenerationState): Mutable state for the active
                generation.
            proposed_particles (list[Particle]): Proposed particles to
                evaluate.

        Returns:
            int: Number of proposed particles that were considered.
        """

        if (
            request.executor is None
            or len(proposed_particles) <= request.chunksize
        ):
            attempts = 0
            for index in range(0, len(proposed_particles), request.chunksize):
                chunk = proposed_particles[index : index + request.chunksize]
                errs = self._evaluate_particle_chunk(
                    proposed_particles=chunk,
                    particle_kwargs=request.particle_kwargs,
                )
                attempts += self._accept_particle_batch(
                    generation=request.generation,
                    proposed_population=state.proposed_population,
                    proposed_particles=chunk,
                    errs=errs,
                )
                if (
                    state.proposed_population.size
                    >= self.config.generation_particle_count
                ):
                    break
            return attempts

        return run_coroutine_from_sync(
            lambda: self._process_particle_batch_async(
                request=request,
                state=state,
                proposed_particles=proposed_particles,
            )
        )

    def _accept_particle_batch(
        self,
        generation: int,
        proposed_population: ParticlePopulation,
        proposed_particles: list[Particle],
        errs: list[float],
    ) -> int:
        """Accept evaluated particles into the proposed population.

        This helper applies the generation tolerance to evaluated particles,
        computes weights for accepted particles, and stops early once the
        proposed population reaches the target size.

        Args:
            generation (int): Zero-based generation index being executed.
            proposed_population (ParticlePopulation): Population being filled
                for the generation.
            proposed_particles (list[Particle]): Proposed particles that were
                evaluated.
            errs (list[float]): Distances computed for the proposed particles.

        Returns:
            int: Number of evaluated particles that were considered.
        """

        considered = 0
        for err, proposed_particle in zip(errs, proposed_particles):
            if (
                proposed_population.size
                >= self.config.generation_particle_count
            ):
                break

            considered += 1
            if err <= self.config.tolerance_values[generation]:
                particle_weight = (
                    1.0
                    if generation == 0
                    else self.config.calculate_weight(proposed_particle)
                )
                proposed_population.add_particle(
                    proposed_particle,
                    particle_weight,
                )

        return considered

    def _update_progress(
        self,
        handle: ProgressHandle,
        state: BatchGenerationState,
        generation_start_time: float,
    ) -> None:
        """Update generation progress for the batched path.

        This helper computes ETA and acceptance rate for the current batched
        generation state before delegating the actual Rich update to the
        reporter.

        Args:
            handle (ProgressHandle): Handle referencing the active progress
                task.
            state (BatchGenerationState): Mutable state for the active
                generation.
            generation_start_time (float): Timestamp recorded at the start of
                the generation.

        Returns:
            None: This helper does not return a value.
        """

        elapsed = time.time() - generation_start_time
        completed = state.proposed_population.size
        eta = (
            elapsed
            * (self.config.generation_particle_count - completed)
            / completed
            if elapsed > 0 and completed > 0
            else 0.0
        )
        acceptance_rate = (
            100.0 * completed / state.attempts if state.attempts > 0 else 0.0
        )
        self.config.reporter.update_collection_progress(
            handle=handle,
            completed=completed,
            acceptance_rate=acceptance_rate,
            eta_seconds=eta,
        )

    def _build_generation_stats(
        self,
        request: BatchGenerationRequest,
        state: BatchGenerationState,
    ) -> GenerationStats:
        """Build summary metrics for a completed batched generation.

        This helper records attempts, accepted particles, and elapsed timing in
        the shared `GenerationStats` carrier used by sampler execution paths.

        Args:
            request (BatchGenerationRequest): Runtime inputs for the generation
                being executed.
            state (BatchGenerationState): Mutable state for the active
                generation.

        Returns:
            GenerationStats: Summary metrics for the completed generation.
        """

        return GenerationStats(
            attempts=state.attempts,
            successes=state.proposed_population.size,
            processing_time=time.time() - request.generation_start_time,
            total_time=time.time() - request.overall_start_time,
        )
