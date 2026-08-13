"""One live progress display and durable detail logs for a calibration study."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from rich.console import Console, RenderableType
from rich.live import Live
from rich.table import Table

from .sampler_types import ProgressEvent


class ScenarioState(str, Enum):
    """Describe the lifecycle state of one study scenario."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ScenarioProgressSnapshot:
    """Expose the current observable state of one scenario."""

    name: str
    state: ScenarioState
    generation: int | None
    generation_total: int | None
    completed: int | None
    total: int | None
    attempts: int | None
    acceptance_rate: float | None
    elapsed_seconds: float | None
    eta_seconds: float | None
    status_message: str | None
    failure_summary: str | None
    detail_log_path: str | None


@dataclass(frozen=True, slots=True)
class StudyProgressSnapshot:
    """Expose the aggregate and per-scenario monitor state."""

    scenarios: tuple[ScenarioProgressSnapshot, ...]
    queued_count: int
    running_count: int
    completed_count: int
    failed_count: int
    cancelled_count: int


@dataclass(slots=True)
class _ScenarioProgress:
    name: str
    state: ScenarioState = ScenarioState.QUEUED
    parameters: Mapping[str, Any] = field(default_factory=dict)
    generation: int | None = None
    generation_total: int | None = None
    completed: int | None = None
    total: int | None = None
    attempts: int | None = None
    acceptance_rate: float | None = None
    eta_seconds: float | None = None
    status_message: str | None = None
    status_changed_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    failure_summary: str | None = None
    detail_log_path: str | None = None


class StudyProgressReporter:
    """Own the study dashboard and structured scenario detail logs."""

    def __init__(
        self,
        *,
        study_name: str,
        scenario_names: list[str] | tuple[str, ...],
        detail_log_dir: str | Path,
        quiet: bool = False,
        console: Console | None = None,
        stall_warning_seconds: float = 120.0,
    ) -> None:
        self.study_name = study_name
        self.detail_log_dir = Path(detail_log_dir)
        self.quiet = quiet
        self.stall_warning_seconds = stall_warning_seconds
        self.console = console or Console()
        self._scenarios = {
            name: _ScenarioProgress(name=name) for name in scenario_names
        }
        self._scenario_order = tuple(scenario_names)
        self._lock = threading.RLock()
        self._live: Live | None = None
        self._setup_message: str | None = None
        self._setup_started_at: float | None = None

    def start(self) -> None:
        """Start the single live display for this study."""

        with self._lock:
            if not self.quiet and self._live is None:
                self._live = Live(
                    _LiveDashboard(self),
                    console=self.console,
                    refresh_per_second=4,
                    transient=False,
                )
                self._live.start(refresh=True)

    def finish(self, *, success: bool) -> None:
        """Stop the live display after all scenario work has settled."""

        del success
        with self._lock:
            self._setup_message = None
            self._setup_started_at = None
            if self._live is not None:
                self._live.update(self._render(), refresh=True)
                self._live.stop()
                self._live = None

    def mark_started(
        self,
        scenario_name: str,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        """Record that a scenario entered the semaphore-protected run phase."""

        with self._lock:
            scenario = self._scenario(scenario_name)
            scenario.state = ScenarioState.RUNNING
            scenario.started_at = time.monotonic()
            scenario.parameters = dict(parameters or {})
            scenario.status_message = None
            self._setup_message = None
            self._setup_started_at = None
            self._write_detail(scenario, "scenario_started", {})
            self._refresh()

    def mark_shared_pool_preparing(self) -> None:
        """Show that queued scenarios are waiting for shared Azure capacity."""

        self.mark_shared_setup_status("Preparing shared Azure pool")

    def mark_shared_setup_status(self, message: str) -> None:
        """Update the study-level status shown while shared resources come up.

        Args:
            message (str): Human-readable description of the current stage.

        Returns:
            None: This method does not return a value.
        """

        with self._lock:
            if self._setup_started_at is None:
                self._setup_started_at = time.monotonic()
            self._setup_message = message
            for scenario in self._scenarios.values():
                if scenario.status_message != message:
                    scenario.status_changed_at = time.monotonic()
                scenario.status_message = message
                self._write_detail(
                    scenario, "shared_pool_preparing", {"message": message}
                )
            self._refresh()

    def handle_setup_event(self, event: ProgressEvent) -> None:
        """Route one shared-resource setup event to the study status line.

        Args:
            event (ProgressEvent): Executor event emitted during preparation.

        Returns:
            None: This method does not return a value.
        """

        message = event.payload.get("message")
        if message:
            self.mark_shared_setup_status(str(message))

    def mark_completed(self, scenario_name: str) -> None:
        """Record a successfully completed scenario."""

        with self._lock:
            scenario = self._scenario(scenario_name)
            scenario.state = ScenarioState.COMPLETED
            scenario.finished_at = time.monotonic()
            self._write_detail(scenario, "scenario_completed", {})
            self._refresh()

    def mark_failed(self, scenario_name: str, error: BaseException) -> None:
        """Record a failed scenario while retaining a concise error summary."""

        with self._lock:
            scenario = self._scenario(scenario_name)
            scenario.state = ScenarioState.FAILED
            scenario.finished_at = time.monotonic()
            scenario.failure_summary = " ".join(str(error).split())
            self._write_detail(
                scenario,
                "scenario_failed",
                {"error_type": type(error).__name__},
            )
            self._refresh()

    def mark_cancelled(self, scenario_name: str) -> None:
        """Record cancellation of a scenario after another scenario failed."""

        with self._lock:
            scenario = self._scenario(scenario_name)
            if scenario.state in (ScenarioState.QUEUED, ScenarioState.RUNNING):
                scenario.state = ScenarioState.CANCELLED
                scenario.finished_at = time.monotonic()
                self._write_detail(scenario, "scenario_cancelled", {})
                self._refresh()

    def handle_sampler_event(
        self, scenario_name: str, event: ProgressEvent
    ) -> None:
        """Route one sampler or Azure observation event to its scenario."""

        with self._lock:
            scenario = self._scenario(scenario_name)
            payload = dict(event.payload)
            if event.event_type == "generation_started":
                scenario.generation = event.generation
                scenario.generation_total = payload.get("generation_total")
            elif event.event_type == "work_progressed":
                scenario.generation = event.generation
                scenario.completed = payload.get("completed")
                scenario.total = payload.get("total")
                scenario.attempts = payload.get("attempts")
                scenario.acceptance_rate = payload.get("acceptance_rate")
                scenario.eta_seconds = payload.get("eta_seconds")
                scenario.status_message = None
                scenario.status_changed_at = time.monotonic()
            elif event.event_type == "executor_message":
                message = payload.get("message")
                if message:
                    if str(message) != scenario.status_message:
                        scenario.status_changed_at = time.monotonic()
                    scenario.status_message = str(message)
            self._write_detail(scenario, event.event_type, payload)
            self._refresh()

    def snapshot(self) -> StudyProgressSnapshot:
        """Return immutable monitor state for tests and external observers."""

        with self._lock:
            scenarios = tuple(
                self._snapshot_scenario(self._scenarios[name])
                for name in self._scenario_order
            )
        counts = {state: 0 for state in ScenarioState}
        for scenario in scenarios:
            counts[scenario.state] += 1
        return StudyProgressSnapshot(
            scenarios=scenarios,
            queued_count=counts[ScenarioState.QUEUED],
            running_count=counts[ScenarioState.RUNNING],
            completed_count=counts[ScenarioState.COMPLETED],
            failed_count=counts[ScenarioState.FAILED],
            cancelled_count=counts[ScenarioState.CANCELLED],
        )

    def _scenario(self, scenario_name: str) -> _ScenarioProgress:
        try:
            return self._scenarios[scenario_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown study scenario {scenario_name!r}"
            ) from exc

    @staticmethod
    def _elapsed(scenario: _ScenarioProgress) -> float | None:
        if scenario.started_at is None:
            return None
        return (scenario.finished_at or time.monotonic()) - scenario.started_at

    def _snapshot_scenario(
        self, scenario: _ScenarioProgress
    ) -> ScenarioProgressSnapshot:
        return ScenarioProgressSnapshot(
            name=scenario.name,
            state=scenario.state,
            generation=scenario.generation,
            generation_total=scenario.generation_total,
            completed=scenario.completed,
            total=scenario.total,
            attempts=scenario.attempts,
            acceptance_rate=scenario.acceptance_rate,
            elapsed_seconds=self._elapsed(scenario),
            eta_seconds=scenario.eta_seconds,
            status_message=scenario.status_message,
            failure_summary=scenario.failure_summary,
            detail_log_path=scenario.detail_log_path,
        )

    def _write_detail(
        self,
        scenario: _ScenarioProgress,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        self.detail_log_dir.mkdir(parents=True, exist_ok=True)
        path = self.detail_log_dir / f"{self._safe_name(scenario.name)}.jsonl"
        scenario.detail_log_path = str(path)
        record = {
            "timestamp": time.time(),
            "event_type": event_type,
            "scenario_name": scenario.name,
            "state": scenario.state.value,
            "parameters": scenario.parameters,
            "payload": payload,
            "failure_summary": scenario.failure_summary,
        }
        with path.open("a", encoding="utf-8") as detail_file:
            json.dump(record, detail_file, default=str, sort_keys=True)
            detail_file.write("\n")

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "scenario"

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def _render(self) -> RenderableType:
        snapshot = self.snapshot()
        table = Table(title=self._render_title())
        table.add_column("Scenario")
        table.add_column("State")
        table.add_column("Generation")
        table.add_column("Work")
        table.add_column("Attempts")
        table.add_column("Acceptance")
        table.add_column("Elapsed")
        table.add_column("ETA")
        table.add_column("Note", overflow="fold")
        for scenario in snapshot.scenarios:
            table.add_row(
                scenario.name,
                scenario.state.value,
                "-"
                if scenario.generation is None
                else self._format_generation(scenario),
                self._format_work(scenario),
                "-" if scenario.attempts is None else str(scenario.attempts),
                self._format_percent(scenario.acceptance_rate),
                self._format_duration(scenario.elapsed_seconds),
                self._format_duration(scenario.eta_seconds),
                scenario.failure_summary or self._format_note(scenario) or "",
            )
        return table

    def _format_note(self, scenario: ScenarioProgressSnapshot) -> str:
        """Render the status note with an age hint so stalls stay visible.

        Args:
            scenario (ScenarioProgressSnapshot): Scenario being rendered.

        Returns:
            str: Status text, annotated with its age when it is stale.
        """

        message = scenario.status_message
        if not message:
            return ""
        tracked = self._scenarios.get(scenario.name)
        changed_at = getattr(tracked, "status_changed_at", None)
        if changed_at is None:
            return message
        age = time.monotonic() - changed_at
        if age < self.stall_warning_seconds:
            return message
        return f"[yellow]{message} (unchanged {self._format_duration(age)})[/yellow]"

    def _render_title(self) -> str:
        title = f"Calibration study: {self.study_name}"
        if self._setup_message is None:
            return title
        elapsed = (
            0.0
            if self._setup_started_at is None
            else time.monotonic() - self._setup_started_at
        )
        return (
            f"{title}\n[cyan]{self._setup_message}[/cyan] "
            f"({self._format_duration(elapsed)} elapsed)"
        )

    @staticmethod
    def _format_generation(scenario: ScenarioProgressSnapshot) -> str:
        if scenario.generation is None:
            return "-"
        generation = str(scenario.generation + 1)
        if scenario.generation_total is None:
            return generation
        return f"{generation}/{scenario.generation_total}"

    @staticmethod
    def _format_work(scenario: ScenarioProgressSnapshot) -> str:
        if scenario.completed is None:
            return "-"
        return f"{scenario.completed}/{scenario.total or '?'}"

    @staticmethod
    def _format_percent(value: float | None) -> str:
        return "-" if value is None else f"{value:.1f}%"

    @staticmethod
    def _format_duration(value: float | None) -> str:
        if value is None:
            return "-"
        minutes, seconds = divmod(int(max(value, 0)), 60)
        return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


class _LiveDashboard:
    """Re-render the study dashboard on every Rich refresh tick.

    Passing a live proxy instead of a static table lets elapsed timers advance
    during long-running phases such as Azure pool creation, where no events
    arrive to trigger an explicit refresh.

    Args:
        reporter (StudyProgressReporter): Reporter owning the dashboard state.
    """

    def __init__(self, reporter: "StudyProgressReporter") -> None:
        self._reporter = reporter

    def __rich__(self) -> RenderableType:
        return self._reporter._render()
