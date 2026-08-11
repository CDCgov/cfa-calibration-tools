from __future__ import annotations

from numpy.random import SeedSequence

from calibrationtools.azure_batch_worker import run_tasks
from calibrationtools.cloud_executor import CloudAcceptanceTask
from calibrationtools.particle import Particle


def make_particle(_: SeedSequence | None) -> Particle:
    return Particle({"value": 1.0})


def test_worker_runs_a_task_chunk() -> None:
    results = run_tasks(
        [
            CloudAcceptanceTask(
                slot_id=1,
                seed_sequence=SeedSequence(1),
                tolerance=1.0,
                max_attempts=1,
                sample_method=make_particle,
                particle_to_distance=lambda particle: 0.0,
            )
        ]
    )

    assert results[0].slot_id == 1
    assert results[0].status == "accepted"
