"""Public factory-based API for coordinating Azure-backed calibrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .cloud_executor import CloudExecutor


@dataclass(frozen=True, slots=True)
class CalibrationScenario:
    """Name and immutable parameter payload for one fresh sampler run."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scenario name must not be empty")
        object.__setattr__(self, "parameters", dict(self.parameters))


class CalibrationStudy:
    """Coordinate fresh samplers whose particle work runs through Azure Batch."""

    def __init__(
        self,
        *,
        scenarios: Sequence[CalibrationScenario],
        sampler_factory: Callable[[CalibrationScenario], Any],
        cloud_executor: CloudExecutor,
        max_concurrent_scenarios: int = 1,
        detail_log_dir: str | Path = "calibration-study-logs",
        study_name: str = "calibration-study",
        quiet: bool = False,
    ) -> None:
        self.scenarios = tuple(scenarios)
        if not self.scenarios:
            raise ValueError("study requires at least one scenario")
        names = [scenario.name for scenario in self.scenarios]
        if len(names) != len(set(names)):
            raise ValueError("study scenario names must be unique")
        if max_concurrent_scenarios <= 0:
            raise ValueError("max_concurrent_scenarios must be positive")
        self.sampler_factory = sampler_factory
        self.cloud_executor = cloud_executor
        self.max_concurrent_scenarios = max_concurrent_scenarios
        self.detail_log_dir = Path(detail_log_dir)
        self.study_name = study_name
        self.quiet = quiet
        self.reporter = None

    def run(self, **sampler_kwargs: Any) -> dict[str, Any]:
        """Run all scenarios with Azure Batch and return results in input order."""

        from .study_runner import StudyRunner

        return StudyRunner(self, sampler_kwargs).run()
