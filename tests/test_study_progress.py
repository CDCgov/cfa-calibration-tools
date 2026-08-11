import json
import time
from pathlib import Path

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
    reporter.mark_started("first", parameters={"beta": 0.2})
    reporter.handle_sampler_event(
        "first",
        ProgressEvent(
            event_type="generation_started",
            generation=0,
            payload={"tolerance": 0.5},
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
    assert first.generation == 0
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
