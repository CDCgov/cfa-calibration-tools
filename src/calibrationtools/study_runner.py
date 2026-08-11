"""Semaphore-based coordinator for factory-created calibration scenarios."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from .async_runner import run_coroutine_from_sync
from .study_progress import ScenarioState, StudyProgressReporter

if TYPE_CHECKING:
    from .calibration_study import CalibrationScenario, CalibrationStudy


class StudyRunner:
    """Run a ``CalibrationStudy`` while preserving scenario/result ordering."""

    def __init__(
        self, study: "CalibrationStudy", sampler_kwargs: dict[str, Any]
    ) -> None:
        self.study = study
        self.sampler_kwargs = sampler_kwargs

    def run(self) -> dict[str, Any]:
        """Bridge the asynchronous study scheduler into the public sync API."""

        return run_coroutine_from_sync(self.run_async)

    async def run_async(self) -> dict[str, Any]:
        """Schedule fresh samplers under the configured scenario concurrency cap."""

        reporter = StudyProgressReporter(
            study_name=self.study.study_name,
            scenario_names=[
                scenario.name for scenario in self.study.scenarios
            ],
            detail_log_dir=self.study.detail_log_dir,
            quiet=self.study.quiet,
        )
        self.study.reporter = reporter
        reporter.start()
        semaphore = asyncio.Semaphore(self.study.max_concurrent_scenarios)

        async def run_one(
            index: int, scenario: "CalibrationScenario"
        ) -> tuple[int, Any]:
            async with semaphore:
                reporter.mark_started(
                    scenario.name, parameters=scenario.parameters
                )
                try:
                    sampler = self.study.sampler_factory(scenario)
                    executor = self.study.cloud_executor.clone_for_scenario(
                        scenario.name
                    )
                    previous_verbose = getattr(sampler, "verbose", None)
                    if previous_verbose is not None:
                        sampler.verbose = False
                    try:
                        result = await asyncio.to_thread(
                            sampler.run,
                            execution="azure_batch",
                            cloud_executor=executor,
                            progress_callback=lambda event: reporter.handle_sampler_event(
                                scenario.name, event
                            ),
                            **self.sampler_kwargs,
                        )
                    finally:
                        if previous_verbose is not None:
                            sampler.verbose = previous_verbose
                except asyncio.CancelledError:
                    reporter.mark_cancelled(scenario.name)
                    raise
                except BaseException as exc:
                    reporter.mark_failed(scenario.name, exc)
                    raise
                reporter.mark_completed(scenario.name)
                return index, result

        tasks = [
            asyncio.create_task(run_one(index, scenario))
            for index, scenario in enumerate(self.study.scenarios)
        ]
        try:
            completed = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for scenario in reporter.snapshot().scenarios:
                if scenario.state is ScenarioState.RUNNING:
                    reporter.mark_cancelled(scenario.name)
            reporter.finish(success=False)
            raise
        reporter.finish(success=True)
        return {
            self.study.scenarios[index].name: result
            for index, result in sorted(completed, key=lambda item: item[0])
        }
