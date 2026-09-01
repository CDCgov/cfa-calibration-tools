import threading
import time
from pathlib import Path

import pytest

from calibrationtools.calibration_study import (
    CalibrationScenario,
    CalibrationStudy,
)
from calibrationtools.cloud_executor import CloudExecutor
from calibrationtools.sampler_types import ProgressEvent


class CloneTrackingExecutor(CloudExecutor):
    def __init__(self) -> None:
        self.cloned_for: list[str] = []
        self.lifecycle: list[str] = []

    async def prepare_study(self, *, progress_callback=None) -> None:
        self.lifecycle.append("prepare")

    async def cleanup_study(self) -> None:
        self.lifecycle.append("cleanup")

    def clone_for_scenario(self, scenario_name: str) -> "ScenarioExecutor":
        self.cloned_for.append(scenario_name)
        return ScenarioExecutor(scenario_name)

    async def execute_tasks(
        self, tasks, *, progress_callback=None, on_result=None
    ):
        raise AssertionError("study runner only clones the base executor")


class ScenarioExecutor(CloudExecutor):
    def __init__(self, scenario_name: str) -> None:
        self.scenario_name = scenario_name

    async def execute_tasks(
        self, tasks, *, progress_callback=None, on_result=None
    ):
        raise AssertionError("fake sampler does not submit cloud tasks")


class FakeSampler:
    def __init__(
        self,
        scenario: CalibrationScenario,
        active: list[int],
        maximum: list[int],
    ) -> None:
        self.scenario = scenario
        self.active = active
        self.maximum = maximum
        self.verbose = True

    def run(self, *, cloud_executor, progress_callback, **kwargs):
        assert cloud_executor.scenario_name == self.scenario.name
        with _active_lock:
            self.active[0] += 1
            self.maximum[0] = max(self.maximum[0], self.active[0])
        try:
            progress_callback(
                ProgressEvent(event_type="generation_started", generation=0)
            )
            time.sleep(self.scenario.parameters["delay"])
            if self.scenario.parameters.get("fail"):
                raise RuntimeError(f"{self.scenario.name} failed")
            progress_callback(
                ProgressEvent(
                    event_type="work_progressed",
                    generation=0,
                    payload={"completed": 1, "total": 1, "attempts": 1},
                )
            )
            return self.scenario.name
        finally:
            with _active_lock:
                self.active[0] -= 1


_active_lock = threading.Lock()


def test_study_limits_concurrency_clones_executors_and_preserves_order(
    tmp_path: Path,
) -> None:
    executor = CloneTrackingExecutor()
    active = [0]
    maximum = [0]
    scenarios = [
        CalibrationScenario("first", {"delay": 0.02}),
        CalibrationScenario("second", {"delay": 0.01}),
        CalibrationScenario("third", {"delay": 0.01}),
    ]
    constructed: list[str] = []

    def factory(scenario: CalibrationScenario) -> FakeSampler:
        constructed.append(scenario.name)
        return FakeSampler(scenario, active, maximum)

    study = CalibrationStudy(
        scenarios=scenarios,
        sampler_factory=factory,
        cloud_executor=executor,
        max_concurrent_scenarios=2,
        detail_log_dir=tmp_path,
        quiet=True,
    )

    results = study.run()

    assert list(results) == ["first", "second", "third"]
    assert list(results.values()) == ["first", "second", "third"]
    assert constructed == ["first", "second", "third"]
    assert executor.cloned_for == ["first", "second", "third"]
    assert executor.lifecycle == ["prepare", "cleanup"]
    assert maximum[0] == 2
    assert all(tmp_path.joinpath(f"{name}.jsonl").exists() for name in results)


def test_study_records_failure_and_cancels_remaining_scenarios(
    tmp_path: Path,
) -> None:
    executor = CloneTrackingExecutor()
    active = [0]
    maximum = [0]
    scenarios = [
        CalibrationScenario("first", {"delay": 0.001, "fail": True}),
        CalibrationScenario("second", {"delay": 0.1}),
    ]
    study = CalibrationStudy(
        scenarios=scenarios,
        sampler_factory=lambda scenario: FakeSampler(
            scenario, active, maximum
        ),
        cloud_executor=executor,
        max_concurrent_scenarios=1,
        detail_log_dir=tmp_path,
        quiet=True,
    )

    try:
        study.run()
    except RuntimeError as error:
        assert str(error) == "first failed"
    else:  # pragma: no cover - explicit failure message
        raise AssertionError("study must re-raise a scenario failure")

    assert study.reporter is not None
    snapshot = study.reporter.snapshot()
    assert snapshot.failed_count == 1
    assert snapshot.cancelled_count == 1
    assert executor.lifecycle == ["prepare", "cleanup"]
    assert "first failed" in tmp_path.joinpath("first.jsonl").read_text()
    assert tmp_path.joinpath("second.jsonl").exists()


def test_study_reports_shared_pool_preparation_failure(
    tmp_path: Path,
) -> None:
    """Surface a shared-pool failure on every scenario before scheduling work."""

    class FailingSetupExecutor(CloneTrackingExecutor):
        async def prepare_study(self, *, progress_callback=None) -> None:
            self.lifecycle.append("prepare")
            raise RuntimeError("shared pool unavailable")

    executor = FailingSetupExecutor()
    study = CalibrationStudy(
        scenarios=[
            CalibrationScenario("first"),
            CalibrationScenario("second"),
        ],
        sampler_factory=lambda scenario: (_ for _ in ()).throw(
            AssertionError(f"factory should not run for {scenario.name}")
        ),
        cloud_executor=executor,
        detail_log_dir=tmp_path,
        quiet=True,
    )

    with pytest.raises(RuntimeError, match="shared pool unavailable"):
        study.run()

    assert executor.lifecycle == ["prepare", "cleanup"]
    assert study.reporter is not None
    snapshot = study.reporter.snapshot()
    assert snapshot.failed_count == 2
    assert (
        "shared pool unavailable"
        in tmp_path.joinpath("first.jsonl").read_text()
    )
