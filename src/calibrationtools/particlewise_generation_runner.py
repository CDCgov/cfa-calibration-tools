"""Run particlewise ABC-SMC generations outside the sampler facade.

This module contains the dedicated execution engine for the particlewise
sampling path. It keeps generation setup, proposal collection, progress
reporting, and population finalization out of `ABCSampler`.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from numpy.random import SeedSequence

from .async_runner import run_coroutine_from_sync
from .particle import Particle
from .particle_evaluator import build_evaluation_context_kwargs
from .particle_population import ParticlePopulation
from .sampler_reporting import ProgressHandle, SamplerReporter
from .sampler_run_state import SamplerRunState
from .sampler_types import AcceptedProposal, GenerationStats, GeneratorSlot


@dataclass(frozen=True, slots=True)
class ParticlewiseGenerationConfig:
    """Store static settings and callbacks for particlewise generations.

    This configuration object groups the sampler collaborators that remain
    stable across generations so the runner constructor stays small and the
    execution methods can depend on named fields instead of long parameter
    lists.

    Attributes:
        generation_particle_count (int): Number of accepted particles required
            to complete a generation.
        tolerance_values (list[float]): Acceptance tolerance for each
            generation.
        seed_sequence (SeedSequence): Root seed sequence used to spawn
            deterministic generator slots.
        max_attempts_per_proposal (int): Maximum number of proposal attempts
            allowed for one generator slot.
        sample_particle_from_priors (Callable[[SeedSequence | None], Particle]):
            Proposal function for the initial generation.
        sample_and_perturb_particle (Callable[[SeedSequence | None], Particle]):
            Proposal function for later generations.
        particle_to_distance (Callable[..., float]): Function used to evaluate
            the distance of one proposed particle.
        calculate_weight (Callable[[Particle], float]): Function used to weight
            accepted particles after the first generation.
        replace_particle_population (Callable[[ParticlePopulation], None]):
            Callback that stores the finalized population on the sampler.
        reporter (SamplerReporter): Reporter used for progress and summary
            output.
        particle_to_distance_async (Callable[..., Any] | None): Optional async
            distance evaluator used by runners that prefer native async
            simulation collection.
    """

    generation_particle_count: int
    tolerance_values: list[float]
    seed_sequence: SeedSequence
    max_attempts_per_proposal: int
    sample_particle_from_priors: Callable[[SeedSequence | None], Particle]
    sample_and_perturb_particle: Callable[[SeedSequence | None], Particle]
    particle_to_distance: Callable[..., float]
    calculate_weight: Callable[[Particle], float]
    replace_particle_population: Callable[[ParticlePopulation], None]
    reporter: SamplerReporter
    particle_to_distance_async: Callable[..., Any] | None = None


@dataclass(frozen=True, slots=True)
class ParticlewiseGenerationRequest:
    """Describe one particlewise generation execution.

    This request object captures the per-generation runtime inputs that vary
    across sampler iterations, including executor access, timing markers, and
    keyword arguments forwarded into particle evaluation.

    Attributes:
        generation (int): Zero-based generation index to execute.
        n_workers (int): Number of workers available to the generation.
        parallel_executor (ThreadPoolExecutor | None): Executor used for
            threaded proposal collection when parallel execution is active.
        overall_start_time (float): Timestamp recorded at the start of the full
            sampler run.
        generation_start_time (float): Timestamp recorded at the start of the
            generation.
        particle_kwargs (dict[str, Any]): Keyword arguments forwarded into
            particle evaluation.
    """

    generation: int
    n_workers: int
    parallel_executor: ThreadPoolExecutor | None
    overall_start_time: float
    generation_start_time: float
    particle_kwargs: dict[str, Any]


@dataclass(slots=True)
class ParticlewiseGenerationState:
    """Store mutable state for one particlewise generation.

    This state groups the proposed population, deterministic generator slots,
    and generation-specific sample method so helper methods can share that data
    without long positional argument lists.

    Attributes:
        proposed_population (ParticlePopulation): Population being filled for
            the active generation.
        error_distribution (list[dict[str, int | float]]): List of dictionaries
            containing the slot ID and distance for each proposed particle in the generation
        generator_slots (list[GeneratorSlot]): Proposal slots used to preserve
            deterministic ordering across execution modes.
        sample_method (Callable[[SeedSequence | None], Particle]): Proposal
            function for the active generation.
    """

    proposed_population: ParticlePopulation
    error_distribution: list[dict[str, int | float]]
    generator_slots: list[GeneratorSlot]
    sample_method: Callable[[SeedSequence | None], Particle]


class ParticlewiseGenerationRunner:
    """Run one particlewise ABC-SMC generation in serial or threaded mode.

    This runner isolates the particlewise execution engine from `ABCSampler`.
    It handles proposal generation, progress reporting, acceptance accounting,
    and population finalization while writing sampler run bookkeeping through a
    dedicated run-state object.

    Args:
        config (ParticlewiseGenerationConfig): Static settings and callbacks
            used across particlewise generations.
        run_state (SamplerRunState): Mutable bookkeeping for the active sampler
            run.
    """

    def __init__(
        self,
        config: ParticlewiseGenerationConfig,
        run_state: SamplerRunState,
    ) -> None:
        self.config = config
        self.run_state = run_state

    def run_generation(
        self,
        request: ParticlewiseGenerationRequest,
    ) -> GenerationStats:
        """Execute one particlewise generation and store its final population.

        This method coordinates generation setup, proposal collection, and
        population finalization for the particlewise sampler path while keeping
        the caller focused on top-level orchestration.

        Args:
            request (ParticlewiseGenerationRequest): Runtime inputs for the
                generation being executed.

        Returns:
            GenerationStats: Attempts, successes, and timing metrics recorded
                for the completed generation.
        """

        state = self._init_generation(request.generation)
        accepted_list, generation_stats = self._collect_accepted_particles(
            request=request,
            state=state,
        )
        self._finalize_generation(
            request=request,
            state=state,
            accepted_list=accepted_list,
            generation_stats=generation_stats,
        )
        return generation_stats

    def sample_particles_until_accepted(
        self,
        generator: GeneratorSlot,
        tolerance: float,
        sample_method: Callable[[SeedSequence | None], Particle],
        evaluation_kwargs: dict[str, Any],
        max_attempts: int | None = None,
        generation: int = 0,
    ) -> AcceptedProposal:
        """Propose particles until one is accepted or attempts are exhausted.

        This rejection-sampling loop repeatedly generates one particle for the
        provided slot until its distance is at most the generation tolerance or
        the configured attempt limit is reached.

        Args:
            generator (GeneratorSlot): Deterministic generator slot to evaluate.
            tolerance (float): Maximum accepted distance for the proposal.
            sample_method (Callable[[SeedSequence | None], Particle]): Proposal
                function for the active generation.
            evaluation_kwargs (dict[str, Any]): Keyword arguments forwarded into
                particle evaluation.
            max_attempts (int | None): Override for the maximum number of
                proposal attempts allowed for the slot.
            generation (int): Zero-based generation index used for evaluation
                context metadata. Defaults to 0 to preserve the previous public
                call shape for direct callers.

        Returns:
            AcceptedProposal: Accepted particle data for the slot, or a record
                showing that no particle was accepted before attempts were
                exhausted.
        """

        if max_attempts is None:
            max_attempts = self.config.max_attempts_per_proposal

        for attempt in range(max_attempts):
            proposed_particle = sample_method(generator.seed_sequence)
            particle_kwargs = build_evaluation_context_kwargs(
                generation=generation,
                proposal_index=generator.id,
                attempt_index=attempt,
                base_kwargs=evaluation_kwargs,
            )
            err = self.config.particle_to_distance(
                proposed_particle,
                **particle_kwargs,
            )
            if err <= tolerance:
                return AcceptedProposal(
                    slot_id=generator.id,
                    particle=proposed_particle,
                    distance=err,
                    attempts=attempt + 1,
                )
        return AcceptedProposal(
            slot_id=generator.id,
            particle=None,
            distance=None,
            attempts=max_attempts,
        )

    async def sample_particles_until_accepted_async(
        self,
        generator: GeneratorSlot,
        tolerance: float,
        sample_method: Callable[[SeedSequence | None], Particle],
        evaluation_kwargs: dict[str, Any],
        max_attempts: int | None = None,
        generation: int = 0,
    ) -> AcceptedProposal:
        if self.config.particle_to_distance_async is None:
            raise RuntimeError("Async particle evaluation is not configured.")

        if max_attempts is None:
            max_attempts = self.config.max_attempts_per_proposal

        for attempt in range(max_attempts):
            proposed_particle = sample_method(generator.seed_sequence)
            particle_kwargs = build_evaluation_context_kwargs(
                generation=generation,
                proposal_index=generator.id,
                attempt_index=attempt,
                base_kwargs=evaluation_kwargs,
            )
            err = await self.config.particle_to_distance_async(
                proposed_particle,
                **particle_kwargs,
            )
            if err <= tolerance:
                return AcceptedProposal(
                    slot_id=generator.id,
                    particle=proposed_particle,
                    distance=err,
                    attempts=attempt + 1,
                )
        return AcceptedProposal(
            slot_id=generator.id,
            particle=None,
            distance=None,
            attempts=max_attempts,
        )

    def _get_sample_method(
        self,
        generation: int,
    ) -> Callable[[SeedSequence | None], Particle]:
        """Return the proposal function for the requested generation.

        The initial generation samples directly from the priors, while later
        generations sample from the previous population and perturb the chosen
        particle.

        Args:
            generation (int): Zero-based generation index being executed.

        Returns:
            Callable[[SeedSequence | None], Particle]: Proposal function for the
                requested generation.
        """

        if generation == 0:
            return self.config.sample_particle_from_priors
        return self.config.sample_and_perturb_particle

    def _init_generation(
        self,
        generation: int,
    ) -> ParticlewiseGenerationState:
        """Create the mutable state needed to execute one generation.

        This method prepares the next empty population, spawns deterministic
        generator slots, and selects the proposal function for the generation.

        Args:
            generation (int): Zero-based generation index being initialized.

        Returns:
            ParticlewiseGenerationState: Mutable generation state shared across
                the collection and finalization steps.
        """

        generator_slots = [
            GeneratorSlot(id=index, seed_sequence=seed_sequence)
            for index, seed_sequence in enumerate(
                self.config.seed_sequence.spawn(
                    self.config.generation_particle_count
                )
            )
        ]
        return ParticlewiseGenerationState(
            proposed_population=ParticlePopulation(),
            generator_slots=generator_slots,
            error_distribution=[],
            sample_method=self._get_sample_method(generation),
        )

    def _collect_accepted_particles_serial(
        self,
        request: ParticlewiseGenerationRequest,
        state: ParticlewiseGenerationState,
        handle: ProgressHandle,
    ) -> tuple[list[AcceptedProposal], int]:
        """Collect accepted proposals serially for one generation.

        This path evaluates one generator slot at a time while keeping the
        shared progress display up to date using deterministic proposal order.

        Args:
            request (ParticlewiseGenerationRequest): Runtime inputs for the
                generation being executed.
            state (ParticlewiseGenerationState): Mutable state for the active
                generation.
            handle (ProgressHandle): Handle referencing the active progress
                task.

        Returns:
            tuple[list[AcceptedProposal], int]: Accepted proposals for the
                generation and the total number of attempts consumed.
        """

        accepted_list: list[AcceptedProposal] = []
        total_attempts = 0
        for completed, generator in enumerate(state.generator_slots, start=1):
            accepted_proposal = self.sample_particles_until_accepted(
                generator=generator,
                tolerance=self.config.tolerance_values[request.generation],
                sample_method=state.sample_method,
                evaluation_kwargs=request.particle_kwargs,
                generation=request.generation,
            )
            accepted_list.append(accepted_proposal)
            total_attempts += accepted_proposal.attempts
            self._update_progress(
                handle=handle,
                completed=completed,
                total_attempts=total_attempts,
                generation_start_time=request.generation_start_time,
            )

        return accepted_list, total_attempts

    async def _collect_accepted_particles_async(
        self,
        request: ParticlewiseGenerationRequest,
        state: ParticlewiseGenerationState,
        handle: ProgressHandle,
    ) -> tuple[list[AcceptedProposal], int]:
        """Collect accepted proposals concurrently over the executor.

        This path submits one generator slot per executor task, records results
        as tasks complete, and keeps the shared progress display synchronized
        with the aggregate attempt count.

        Args:
            request (ParticlewiseGenerationRequest): Runtime inputs for the
                generation being executed.
            state (ParticlewiseGenerationState): Mutable state for the active
                generation.
            handle (ProgressHandle): Handle referencing the active progress
                task.

        Returns:
            tuple[list[AcceptedProposal], int]: Accepted proposals for the
                generation and the total number of attempts consumed.

        Raises:
            BaseException: Re-raises any exception raised while collecting
                proposals after cancelling the outstanding tasks.
        """

        assert request.parallel_executor is not None

        accepted_list: list[AcceptedProposal] = []
        total_attempts = 0
        completed = 0
        loop = asyncio.get_running_loop()
        worker = partial(
            self.sample_particles_until_accepted,
            tolerance=self.config.tolerance_values[request.generation],
            sample_method=state.sample_method,
            evaluation_kwargs=request.particle_kwargs,
            generation=request.generation,
        )
        tasks = [
            loop.run_in_executor(request.parallel_executor, worker, generator)
            for generator in state.generator_slots
        ]

        try:
            for task in asyncio.as_completed(tasks):
                accepted_proposal = await task
                accepted_list.append(accepted_proposal)
                total_attempts += accepted_proposal.attempts
                completed += 1
                self._update_progress(
                    handle=handle,
                    completed=completed,
                    total_attempts=total_attempts,
                    generation_start_time=request.generation_start_time,
                )
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        return accepted_list, total_attempts

    async def _collect_accepted_particles_native_async(
        self,
        request: ParticlewiseGenerationRequest,
        state: ParticlewiseGenerationState,
        handle: ProgressHandle,
    ) -> tuple[list[AcceptedProposal], int]:
        accepted_list: list[AcceptedProposal] = []
        total_attempts = 0
        completed = 0
        semaphore = asyncio.Semaphore(request.n_workers)

        async def evaluate(generator: GeneratorSlot) -> AcceptedProposal:
            async with semaphore:
                return await self.sample_particles_until_accepted_async(
                    generator=generator,
                    tolerance=self.config.tolerance_values[request.generation],
                    sample_method=state.sample_method,
                    evaluation_kwargs=request.particle_kwargs,
                    generation=request.generation,
                )

        tasks = [
            asyncio.create_task(evaluate(generator))
            for generator in state.generator_slots
        ]

        try:
            for task in asyncio.as_completed(tasks):
                accepted_proposal = await task
                accepted_list.append(accepted_proposal)
                total_attempts += accepted_proposal.attempts
                completed += 1
                self._update_progress(
                    handle=handle,
                    completed=completed,
                    total_attempts=total_attempts,
                    generation_start_time=request.generation_start_time,
                )
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        return accepted_list, total_attempts

    def _collect_accepted_particles(
        self,
        request: ParticlewiseGenerationRequest,
        state: ParticlewiseGenerationState,
    ) -> tuple[list[AcceptedProposal], GenerationStats]:
        """Collect accepted proposals and emit progress output for one generation.

        This method owns the shared progress UI, dispatches to either the
        serial or threaded collection path, and produces the generation summary
        metrics once proposal collection is complete.

        Args:
            request (ParticlewiseGenerationRequest): Runtime inputs for the
                generation being executed.
            state (ParticlewiseGenerationState): Mutable state for the active
                generation.

        Returns:
            tuple[list[AcceptedProposal], GenerationStats]: Accepted proposals
                for the generation and the summary statistics derived from the
                collection phase.
        """

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
            if self.config.particle_to_distance_async is not None:
                accepted_list, total_attempts = run_coroutine_from_sync(
                    lambda: self._collect_accepted_particles_native_async(
                        request=request,
                        state=state,
                        handle=handle,
                    )
                )
            elif request.n_workers == 1:
                accepted_list, total_attempts = (
                    self._collect_accepted_particles_serial(
                        request=request,
                        state=state,
                        handle=handle,
                    )
                )
            else:
                accepted_list, total_attempts = run_coroutine_from_sync(
                    lambda: self._collect_accepted_particles_async(
                        request=request,
                        state=state,
                        handle=handle,
                    )
                )

            generation_stats = self._build_generation_stats(
                request=request,
                total_attempts=total_attempts,
                accepted_count=len(accepted_list),
            )
            self.config.reporter.print_generation_summary(
                generation=request.generation,
                tolerance=self.config.tolerance_values[request.generation],
                generation_stats=generation_stats,
            )

        return accepted_list, generation_stats

    def _update_progress(
        self,
        handle: ProgressHandle,
        completed: int,
        total_attempts: int,
        generation_start_time: float,
    ) -> None:
        """Update generation progress using completed slots and total attempts.

        This helper keeps ETA and acceptance-rate calculation in one place so
        both serial and threaded proposal collection report progress the same
        way.

        Args:
            handle (ProgressHandle): Handle referencing the active progress
                task.
            completed (int): Number of generator slots completed so far.
            total_attempts (int): Total proposal attempts consumed so far.
            generation_start_time (float): Timestamp recorded at the start of
                the generation.

        Returns:
            None: This helper does not return a value.
        """

        elapsed = time.time() - generation_start_time
        eta = (
            elapsed
            * (self.config.generation_particle_count - completed)
            / completed
            if elapsed > 0 and completed > 0
            else 0.0
        )
        acceptance_rate = (
            100.0 * completed / total_attempts if total_attempts > 0 else 0.0
        )
        self.config.reporter.update_collection_progress(
            handle=handle,
            completed=completed,
            acceptance_rate=acceptance_rate,
            eta_seconds=eta,
        )

    def _build_generation_stats(
        self,
        request: ParticlewiseGenerationRequest,
        total_attempts: int,
        accepted_count: int,
    ) -> GenerationStats:
        """Build summary metrics for a completed collection phase.

        This helper records attempts, accepted particles, and elapsed timing in
        the shared `GenerationStats` carrier used by sampler execution paths.

        Args:
            request (ParticlewiseGenerationRequest): Runtime inputs for the
                generation being executed.
            total_attempts (int): Total proposal attempts consumed.
            accepted_count (int): Number of accepted particles collected.

        Returns:
            GenerationStats: Summary metrics for the completed generation.
        """

        return GenerationStats(
            attempts=total_attempts,
            successes=accepted_count,
            processing_time=time.time() - request.generation_start_time,
            total_time=time.time() - request.overall_start_time,
        )

    def _finalize_generation(
        self,
        request: ParticlewiseGenerationRequest,
        state: ParticlewiseGenerationState,
        accepted_list: list[AcceptedProposal],
        generation_stats: GenerationStats,
    ) -> None:
        """Convert accepted proposals into the next weighted population.

        This method sorts accepted proposals back into deterministic slot order,
        computes particle weights, records generation bookkeeping, and stores
        the finalized population on the sampler.

        Args:
            request (ParticlewiseGenerationRequest): Runtime inputs for the
                generation being executed.
            state (ParticlewiseGenerationState): Mutable state for the active
                generation.
            accepted_list (list[AcceptedProposal]): Accepted proposal records
                collected for the generation.
            generation_stats (GenerationStats): Summary metrics for the
                completed generation.

        Returns:
            None: This helper does not return a value.

        Raises:
            UserWarning: Raised when a proposal slot exhausts all attempts
                without producing an accepted particle.
        """

        with self.config.reporter.create_task_progress() as progress:
            handle = self.config.reporter.start_task(
                description="Calculating weights... ",
                progress=progress,
                total=self.config.generation_particle_count,
            )
            for accepted_proposal in sorted(
                accepted_list,
                key=lambda proposal: proposal.slot_id,
            ):
                if accepted_proposal.particle is None:
                    raise UserWarning(
                        "Particle proposal attempt "
                        f"{accepted_proposal.slot_id} used "
                        f"{accepted_proposal.attempts} samples and found no "
                        "acceptable values."
                    )
                else:
                    assert accepted_proposal.distance is not None
                    particle_weight = (
                        1.0
                        if request.generation == 0
                        else self.config.calculate_weight(
                            accepted_proposal.particle
                        )
                    )
                    state.proposed_population.add_particle(
                        accepted_proposal.particle,
                        particle_weight,
                    )
                    state.error_distribution.append(
                        {
                            "slot_id": accepted_proposal.slot_id,
                            "distance": accepted_proposal.distance,
                        }
                    )
                self.config.reporter.advance(handle)

        self.run_state.record_generation_history(
            request.generation,
            state.generator_slots,
        )
        self.run_state.record_attempts(
            generation=request.generation,
            attempts=generation_stats.attempts,
            successes=generation_stats.successes,
        )
        self.run_state.record_distances(state.error_distribution)
        self.config.replace_particle_population(state.proposed_population)
        weights_time = (
            time.time()
            - request.generation_start_time
            - generation_stats.processing_time
        )
        self.config.reporter.print_timing_summary(
            processing_time=generation_stats.processing_time,
            total_time=generation_stats.total_time,
            weights_time=weights_time,
        )
