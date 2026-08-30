"""Portable task contracts for cloud-backed particle acceptance.

The cloud boundary transports complete deterministic generator-slot acceptance loops.
Concrete backends only distribute those tasks; sampler state remains owned by the local
generation runner.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

from numpy.random import SeedSequence

from .particle import Particle
from .sampler_types import ProgressCallback


@dataclass(frozen=True, slots=True)
class CloudAcceptanceTask:
    """Describe one deterministic particle-acceptance loop for a cloud worker."""

    slot_id: int
    seed_sequence: SeedSequence
    tolerance: float
    max_attempts: int
    sample_method: Callable[[SeedSequence | None], Particle]
    particle_to_distance: Callable[..., float]
    evaluation_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.slot_id < 0:
            raise ValueError("slot_id must be non-negative")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if not math.isfinite(self.tolerance):
            raise ValueError("tolerance must be finite")
        if not callable(self.sample_method):
            raise TypeError("sample_method must be callable")
        if not callable(self.particle_to_distance):
            raise TypeError("particle_to_distance must be callable")
        object.__setattr__(
            self, "evaluation_kwargs", dict(self.evaluation_kwargs)
        )


@dataclass(frozen=True, slots=True)
class CloudAcceptanceResult:
    """Store the result of one cloud acceptance task."""

    slot_id: int
    particle: Particle | None
    distance: float | None
    attempts: int
    status: Literal["accepted", "exhausted"]

    def __post_init__(self) -> None:
        if self.slot_id < 0:
            raise ValueError("slot_id must be non-negative")
        if self.attempts <= 0:
            raise ValueError("attempts must be positive")
        if self.status == "accepted":
            if self.particle is None or self.distance is None:
                raise ValueError(
                    "accepted results require particle and distance"
                )
        elif self.particle is not None or self.distance is not None:
            raise ValueError(
                "exhausted results cannot include particle or distance"
            )


def run_cloud_acceptance_task(
    task: CloudAcceptanceTask,
) -> CloudAcceptanceResult:
    """Run a complete acceptance loop for one deterministic generator slot."""

    for attempt in range(1, task.max_attempts + 1):
        particle = task.sample_method(task.seed_sequence)
        distance = task.particle_to_distance(
            particle, **task.evaluation_kwargs
        )
        if distance <= task.tolerance:
            return CloudAcceptanceResult(
                slot_id=task.slot_id,
                particle=particle,
                distance=distance,
                attempts=attempt,
                status="accepted",
            )
    return CloudAcceptanceResult(
        slot_id=task.slot_id,
        particle=None,
        distance=None,
        attempts=task.max_attempts,
        status="exhausted",
    )


class CloudExecutor(ABC):
    """Execute complete acceptance tasks without mutating sampler state."""

    async def prepare_study(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Prepare resources shared by all scenarios in one study.

        Backends without study-scoped resources keep the default no-op behavior.

        Args:
            progress_callback (ProgressCallback | None): Optional observer for
                setup-stage events emitted while shared resources come up.
        """

        del progress_callback

    async def cleanup_study(self) -> None:
        """Clean resources owned by one complete study after all scenarios settle.

        Backends without study-scoped resources keep the default no-op behavior.
        """

    @abstractmethod
    async def execute_tasks(
        self,
        tasks: list[CloudAcceptanceTask],
        *,
        progress_callback: ProgressCallback | None = None,
        on_result: Callable[[CloudAcceptanceResult], None] | None = None,
    ) -> list[CloudAcceptanceResult]:
        """Return one result for every submitted task.

        Args:
            tasks (list[CloudAcceptanceTask]): Work to run remotely.
            progress_callback (ProgressCallback | None): Observer for backend
                progress events.
            on_result (Callable[[CloudAcceptanceResult], None] | None): Invoked
                as each result becomes available, so callers can report
                acceptance while the rest of the batch is still running.
                Backends that cannot stream may invoke it once per result
                before returning.

        Returns:
            list[CloudAcceptanceResult]: Results in submission order.
        """

    def clone_for_scenario(self, scenario_name: str) -> "CloudExecutor":
        """Create an isolated backend for a concurrent study scenario."""

        raise NotImplementedError(
            f"{type(self).__name__} cannot be cloned for scenario {scenario_name!r}"
        )
