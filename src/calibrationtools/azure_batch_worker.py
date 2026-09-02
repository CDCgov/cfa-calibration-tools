"""Mounted Azure Batch worker for cloud acceptance-task chunks."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from .cloud_executor import (
    CloudAcceptanceResult,
    CloudAcceptanceTask,
    run_cloud_acceptance_task,
)


def run_tasks(tasks: list[CloudAcceptanceTask]) -> list[CloudAcceptanceResult]:
    """Run one mounted task chunk sequentially inside the model image."""

    return [run_cloud_acceptance_task(task) for task in tasks]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run calibration acceptance tasks on Azure"
    )
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--blob-container", required=True)
    parser.add_argument("--mount-path", default="/mnt/batch/tasks/fsmounts")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    mount = Path(args.mount_path) / args.blob_container
    task_path = mount / args.tasks
    output_path = mount / (args.output or f"results-{Path(args.tasks).name}")
    with task_path.open("rb") as file:
        tasks: list[CloudAcceptanceTask] = pickle.load(file)
    results = run_tasks(tasks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    main()
