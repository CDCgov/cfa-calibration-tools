"""Direct in-process runner for the example branching process."""

from __future__ import annotations

from typing import Any

from calibrationtools.direct_runner import CSVDirectRunner

from .example_model import Binom_BP_Model


class ExampleModelDirectRunner(CSVDirectRunner):
    """Run the example model locally while honoring staged sampler I/O."""

    def __init__(self) -> None:
        super().__init__(
            Binom_BP_Model.simulate,
            output_filename="output.csv",
            fieldnames=("generation", "population"),
            row_builder=self._build_population_row,
            input_error_message="Example model input JSON must be an object.",
        )

    @staticmethod
    def _build_population_row(
        generation: int,
        population: Any,
    ) -> dict[str, Any]:
        """Build one CSV output row for a generation population."""
        return {
            "generation": generation,
            "population": population,
        }
