from __future__ import annotations

import pytest
from numpy.random import SeedSequence

from calibrationtools.cloud_executor import (
    CloudAcceptanceResult,
    CloudAcceptanceTask,
    run_cloud_acceptance_task,
)
from calibrationtools.particle import Particle


def make_particle(_: SeedSequence | None) -> Particle:
    return Particle({"value": 1.0})


def test_cloud_task_retries_until_accepted() -> None:
    distances = iter([2.0, 0.5])
    task = CloudAcceptanceTask(
        slot_id=3,
        seed_sequence=SeedSequence(42),
        tolerance=1.0,
        max_attempts=3,
        sample_method=make_particle,
        particle_to_distance=lambda particle: next(distances),
    )

    result = run_cloud_acceptance_task(task)

    assert result.slot_id == 3
    assert result.status == "accepted"
    assert result.distance == 0.5
    assert result.attempts == 2
    assert result.particle is not None


def test_cloud_task_returns_exhausted_result() -> None:
    task = CloudAcceptanceTask(
        slot_id=0,
        seed_sequence=SeedSequence(1),
        tolerance=0.1,
        max_attempts=2,
        sample_method=make_particle,
        particle_to_distance=lambda particle: 1.0,
    )

    result = run_cloud_acceptance_task(task)

    assert result.status == "exhausted"
    assert result.particle is None
    assert result.distance is None
    assert result.attempts == 2


@pytest.mark.parametrize(
    ("slot_id", "max_attempts", "tolerance", "message"),
    [
        (-1, 1, 1.0, "slot_id"),
        (0, 0, 1.0, "max_attempts"),
        (0, 1, float("inf"), "tolerance"),
    ],
)
def test_cloud_task_validates_inputs(
    slot_id: int, max_attempts: int, tolerance: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CloudAcceptanceTask(
            slot_id=slot_id,
            seed_sequence=SeedSequence(1),
            tolerance=tolerance,
            max_attempts=max_attempts,
            sample_method=make_particle,
            particle_to_distance=lambda particle: 0.0,
        )


def test_result_rejects_inconsistent_status() -> None:
    with pytest.raises(ValueError, match="accepted results"):
        CloudAcceptanceResult(
            slot_id=0,
            particle=None,
            distance=None,
            attempts=1,
            status="accepted",
        )
