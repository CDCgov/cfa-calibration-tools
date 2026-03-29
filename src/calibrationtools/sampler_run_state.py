"""Track mutable sampler bookkeeping for one execution.

This module keeps per-run counters, generator history, and population archives
separate from the public sampler facade so state reset behavior stays explicit.
"""

from .particle_population import ParticlePopulation
from .sampler_types import GeneratorSlot


class SamplerRunState:
    """Track mutable per-run bookkeeping for an `ABCSampler` execution.

    This class owns the generation counters and archive data that should be
    reset between sampler runs while allowing `ABCSampler` to stay focused on
    orchestration and public API concerns.

    Args:
        generation_count (int): Number of configured generations in the run.
        keep_previous_population_data (bool): Whether previous populations
            should be archived between generations.
    """

    def __init__(
        self,
        generation_count: int,
        keep_previous_population_data: bool,
    ) -> None:
        self.generation_count = generation_count
        self.keep_previous_population_data = keep_previous_population_data
        self.reset()

    def reset(self) -> None:
        """Reset generation counters and archived run data.

        This method clears all per-run bookkeeping so the sampler can start a
        fresh execution without leaking counters or archived populations from a
        previous run.

        """

        self.step_successes = [0] * self.generation_count
        self.step_attempts = [0] * self.generation_count
        self.generator_history: dict[int, list[GeneratorSlot]] = {}
        self.population_archive: dict[int, ParticlePopulation] = {}

    def record_generation_history(
        self,
        generation: int,
        generator_slots: list[GeneratorSlot],
    ) -> None:
        """Store the generator slots used to propose one generation.

        This method captures the deterministic generator slots for later result
        inspection and for serial-versus-parallel comparisons in tests.

        Args:
            generation (int): Zero-based generation index being recorded.
            generator_slots (list[GeneratorSlot]): Generator slots used for the
                generation.

        """

        self.generator_history[generation] = list(generator_slots)

    def record_attempts(
        self,
        generation: int,
        attempts: int,
        successes: int,
    ) -> None:
        """Store the attempt and success counts for one generation.

        This method records the accepted-particle count and total proposal
        attempts so result construction can report generation-level acceptance
        diagnostics.

        Args:
            generation (int): Zero-based generation index being recorded.
            attempts (int): Total proposal attempts consumed by the generation.
            successes (int): Total accepted particles produced by the
                generation.

        """

        self.step_attempts[generation] = attempts
        self.step_successes[generation] = successes

    def replace_population(
        self,
        previous_population: ParticlePopulation,
    ) -> None:
        """Archive the previous population before a replacement is stored.

        This method records the outgoing population when archive retention is
        enabled and the previous population is not empty.

        Args:
            previous_population (ParticlePopulation): Population currently
                stored on the sampler before replacement.

        """

        if (
            self.keep_previous_population_data
            and not previous_population.is_empty()
        ):
            step = (
                max(self.population_archive.keys()) + 1
                if self.population_archive
                else 0
            )
            self.population_archive[step] = previous_population

    def build_success_counts(
        self, generation_particle_count: int
    ) -> dict[str, list[int]]:
        """Build the success-count payload for `CalibrationResults`.

        This method packages generation size, successes, and attempts into the
        structure expected by the existing `CalibrationResults` API.

        Args:
            generation_particle_count (int): Target accepted-particle count for
                each generation.

        Returns:
            dict[str, list[int]]: Success-count payload used by result
                construction.
        """

        return {
            "generation_particle_count": [generation_particle_count]
            * self.generation_count,
            "successes": list(self.step_successes),
            "attempts": list(self.step_attempts),
        }
