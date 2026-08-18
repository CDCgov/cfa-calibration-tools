import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

from rich.console import Console
from rich.live import Live

from calibrationtools.sampler_types import ProgressEvent
from calibrationtools.study_progress import (
    ScenarioState,
    StudyProgressReporter,
)


def test_reporter_tracks_progress_and_writes_scenario_detail_log(
    tmp_path: Path,
) -> None:
    reporter = StudyProgressReporter(
        study_name="demo",
        scenario_names=["first", "second"],
        detail_log_dir=tmp_path,
        quiet=True,
    )
    reporter.start()
    reporter.mark_shared_pool_preparing()
    assert (
        reporter.snapshot().scenarios[1].status_message
        == "Preparing shared Azure pool"
    )
    reporter.mark_started("first", parameters={"beta": 0.2})
    reporter.handle_sampler_event(
        "first",
        ProgressEvent(
            event_type="generation_started",
            generation=0,
            payload={"tolerance": 0.5, "generation_total": 2},
        ),
    )
    reporter.handle_sampler_event(
        "first",
        ProgressEvent(
            event_type="work_progressed",
            generation=0,
            payload={
                "completed": 2,
                "total": 4,
                "attempts": 5,
                "acceptance_rate": 40.0,
                "eta_seconds": 3.0,
            },
        ),
    )
    reporter.handle_sampler_event(
        "first",
        ProgressEvent(
            event_type="executor_message",
            payload={"source": "azure_batch", "message": "pool ready"},
        ),
    )

    snapshot = reporter.snapshot()
    first = snapshot.scenarios[0]
    assert snapshot.running_count == 1
    assert snapshot.queued_count == 1
    assert first.state is ScenarioState.RUNNING
    assert first.status_message == "pool ready"
    assert first.generation == 0
    assert first.generation_total == 2
    assert first.completed == 2
    assert first.total == 4
    assert first.attempts == 5
    assert first.acceptance_rate == 40.0
    assert first.detail_log_path is not None
    records = [
        json.loads(line)
        for line in Path(first.detail_log_path).read_text().splitlines()
    ]
    assert records[-1]["payload"]["message"] == "pool ready"
    console = Console(record=True, width=200)
    console.print(reporter._render())
    assert "1/2" in console.export_text()
    reporter.finish(success=True)


def test_reporter_records_failure_and_running_elapsed_time(
    tmp_path: Path,
) -> None:
    reporter = StudyProgressReporter(
        study_name="demo",
        scenario_names=["first"],
        detail_log_dir=tmp_path,
        quiet=True,
    )
    reporter.mark_started("first")
    time.sleep(0.001)
    reporter.mark_failed("first", RuntimeError("worker unavailable"))

    scenario = reporter.snapshot().scenarios[0]
    assert scenario.state is ScenarioState.FAILED
    assert scenario.elapsed_seconds is not None
    assert scenario.failure_summary == "worker unavailable"
    assert scenario.detail_log_path is not None
    assert "worker unavailable" in Path(scenario.detail_log_path).read_text()
    console = Console(record=True, width=200)
    console.print(reporter._render())
    assert "worker unavailable" in console.export_text()


class _RecordingLive:
    """Stand-in for rich's ``Live`` that records lock state on every call."""

    def __init__(self, lock_is_held: Callable[[], bool]) -> None:
        self._lock_is_held = lock_is_held
        self.violations: list[str] = []

    def _record(self, method: str) -> None:
        if self._lock_is_held():
            self.violations.append(method)

    def start(self, refresh: bool = False) -> None:
        self._record("start")

    def stop(self) -> None:
        self._record("stop")

    def refresh(self) -> None:
        self._record("refresh")

    def update(self, renderable: object, refresh: bool = False) -> None:
        self._record("update")


def test_reporter_never_touches_live_display_while_holding_its_lock(
    tmp_path: Path,
) -> None:
    """Guard the lock ordering that keeps concurrent studies from deadlocking.

    ``Live`` calls take rich's internal lock, while rich's refresh thread
    renders the dashboard and takes the reporter lock. Touching the display
    while holding the reporter lock inverts that order and wedges every
    scenario thread, including the one that creates the Azure Batch job.
    """

    reporter = StudyProgressReporter(
        study_name="demo",
        scenario_names=["first", "second", "third"],
        detail_log_dir=tmp_path,
        quiet=True,
    )
    live = _RecordingLive(getattr(reporter._lock, "_is_owned"))
    reporter._live = cast(Live, live)

    reporter.mark_shared_pool_preparing()
    reporter.mark_started("first", parameters={"beta": 0.2})
    reporter.handle_sampler_event(
        "first",
        ProgressEvent(
            event_type="generation_started",
            generation=0,
            payload={"generation_total": 2},
        ),
    )
    reporter.mark_completed("first")
    reporter.mark_started("second")
    reporter.mark_failed("second", RuntimeError("boom"))
    reporter.mark_started("third")
    reporter.mark_cancelled("third")
    reporter.finish(success=False)

    assert live.violations == []
