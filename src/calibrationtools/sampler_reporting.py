"""Report sampler progress and summaries through Rich.

This module centralizes progress-bar construction and run-summary printing so
the execution runners can focus on sampling logic instead of Rich wiring.
"""

from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from . import formatting
from .sampler_types import GenerationStats


@dataclass(frozen=True, slots=True)
class ProgressHandle:
    """Reference an active Rich progress task.

    This carrier keeps the `Progress` instance and the active task id together
    so runners can update progress through a single object.

    Attributes:
        progress (Progress): Active Rich progress instance.
        task_id (TaskID): Task identifier within that progress instance.
    """

    progress: Progress
    task_id: TaskID


class SamplerReporter:
    """Create progress displays and print run summaries.

    This helper owns the Rich console and the formatting of generation and run
    summaries so execution engines do not need to duplicate UI setup.

    Args:
        verbose (bool): Whether progress and summary output should be visible.
        console (Console | None): Optional console override used for tests.
    """

    def __init__(
        self,
        verbose: bool,
        console: Console | None = None,
    ) -> None:
        self.console = (
            console if console is not None else formatting.get_console(verbose)
        )

    def create_collection_progress(self) -> Progress:
        """Create the progress layout used during proposal collection.

        This helper centralizes the Rich progress columns used by both the
        particlewise and batched generation paths.

        Returns:
            Progress: Configured Rich progress instance for collection work.
        """

        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("acceptance: {task.fields[acceptance]}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TextColumn("ETA: {task.fields[eta]}"),
            console=self.console,
            transient=True,
        )

    def start_collection_task(
        self,
        progress: Progress,
        description: str,
        total: int,
    ) -> ProgressHandle:
        """Start a collection-phase progress task.

        This helper creates the task used to track accepted particles during a
        generation and returns a handle for later updates.

        Args:
            progress (Progress): Active Rich progress instance.
            description (str): Description shown for the task.
            total (int): Total items required to complete the task.

        Returns:
            ProgressHandle: Handle referencing the created progress task.
        """

        task_id = progress.add_task(
            description,
            total=total,
            acceptance="N/A",
            eta="calculating...",
        )
        return ProgressHandle(progress=progress, task_id=task_id)

    def update_collection_progress(
        self,
        handle: ProgressHandle,
        completed: int,
        acceptance_rate: float,
        eta_seconds: float,
    ) -> None:
        """Update collection progress for a generation.

        This helper formats the acceptance rate and ETA consistently before
        applying the update to the active task.

        Args:
            handle (ProgressHandle): Handle referencing the active task.
            completed (int): Number of completed items for the task.
            acceptance_rate (float): Current acceptance rate as a percentage.
            eta_seconds (float): Estimated remaining time in seconds.

        Returns:
            None: This helper does not return a value.
        """

        handle.progress.update(
            handle.task_id,
            completed=completed,
            acceptance=f"{acceptance_rate:.1f}%",
            eta=formatting._format_time(eta_seconds),
        )

    def create_task_progress(self) -> Progress:
        """Create the progress layout used during weight calculation.

        This helper centralizes the simplified Rich progress display used for
        the weight-calculation phase after particle acceptance.

        Returns:
            Progress: Configured Rich progress instance for weight updates.
        """

        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )

    def start_task(
        self,
        description: str,
        progress: Progress,
        total: int,
    ) -> ProgressHandle:
        """Start a weight-calculation progress task.

        This helper creates the task used to track post-acceptance weight
        calculation and returns a handle for later updates.

        Args:
            description (str): String argument to place at the head of the prgoress bar describing the tasks being executed
            progress (Progress): Active Rich progress instance.
            total (int): Total number of accepted particles to process.

        Returns:
            ProgressHandle: Handle referencing the created progress task.
        """

        task_id = progress.add_task(description=description, total=total)
        return ProgressHandle(progress=progress, task_id=task_id)

    def advance(self, handle: ProgressHandle, steps: int = 1) -> None:
        """Advance a progress task by the requested number of steps.

        This helper keeps direct task advancement out of the execution
        runners.

        Args:
            handle (ProgressHandle): Handle referencing the active task.
            steps (int): Number of steps to advance the task.

        Returns:
            None: This helper does not return a value.
        """

        handle.progress.update(handle.task_id, advance=steps)

    def print_generation_summary(
        self,
        generation: int,
        tolerance: float,
        generation_stats: GenerationStats,
    ) -> None:
        """Print the summary for a completed generation.

        This helper prints the generation completion message with a consistent
        acceptance-rate format across execution engines.

        Args:
            generation (int): Zero-based generation index that completed.
            tolerance (float): Tolerance used by the generation.
            generation_stats (GenerationStats): Summary metrics for the
                generation.

        Returns:
            None: This helper does not return a value.
        """

        acceptance_rate = (
            100.0 * generation_stats.successes / generation_stats.attempts
            if generation_stats.attempts > 0
            else 0.0
        )
        self.console.print(
            f"[green]✓[/green] Generation {generation + 1} run complete! "
            f"Tolerance: {tolerance}, acceptance rate: {acceptance_rate:.1f}% "
            f"of {generation_stats.attempts} attempts"
        )

    def print_timing_summary(
        self,
        processing_time: float,
        total_time: float,
        weights_time: float | None = None,
    ) -> None:
        """Print the timing summary for a completed generation.

        This helper keeps the generation timing message consistent across
        particlewise and batched execution while allowing the weight phase to
        be omitted for the batched path.

        Args:
            processing_time (float): Seconds spent in the generation's main
                processing phase.
            total_time (float): Seconds elapsed since the run began.
            weights_time (float | None): Optional seconds spent in the weight
                calculation phase.

        Returns:
            None: This helper does not return a value.
        """

        if weights_time is None:
            self.console.print(
                f"(Run: {formatting._format_time(processing_time)}, "
                f"total time: {formatting._format_time(total_time)})"
            )
            return

        self.console.print(
            f"(Run: {formatting._format_time(processing_time)}, "
            f"Weights calculation: {formatting._format_time(weights_time)}, "
            f"total time: {formatting._format_time(total_time)})"
        )

    def print_run_summary(self, total_time: float) -> None:
        """Print the summary for the completed sampler run.

        This helper keeps the final run-completion message in one place so the
        public sampler facade does not need to format Rich output directly.

        Args:
            total_time (float): Seconds elapsed for the full sampler run.

        Returns:
            None: This helper does not return a value.
        """

        self.console.print(
            f"[green]✓[/green] Calibration complete! "
            f"(total time: {formatting._format_time(total_time)})"
        )
