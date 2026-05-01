from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


class CSVDirectRunner:
    """Run an in-process simulation while honoring staged sampler I/O.

    The runner adapts a pure simulation callable to the runner protocol used by
    ``ParticleEvaluator``. It can read staged JSON inputs, attach the sampler
    run id, call the model directly, and write a CSV output artifact.
    """

    def __init__(
        self,
        simulate_func: Callable[[dict[str, Any]], list[Any]],
        *,
        output_filename: str,
        fieldnames: tuple[str, ...],
        row_builder: Callable[[int, Any], dict[str, Any]],
        input_error_message: str = "Runner input JSON must be an object.",
    ) -> None:
        """Initialize the direct CSV runner.

        Args:
            simulate_func: In-process model simulation callable.
            output_filename: Name of the CSV file written under ``output_dir``.
            fieldnames: CSV header fields.
            row_builder: Callable that maps ``(index, value)`` to a CSV row.
            input_error_message: Error raised when staged input JSON is not an
                object.
        """
        self._simulate_func = simulate_func
        self._output_filename = output_filename
        self._fieldnames = fieldnames
        self._row_builder = row_builder
        self._input_error_message = input_error_message

    def simulate(
        self,
        params: dict[str, Any],
        *,
        input_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        run_id: str | None = None,
    ) -> list[Any]:
        """Run the simulation and optionally write staged CSV output."""
        model_inputs = self._resolve_inputs(
            params,
            input_path=input_path,
            run_id=run_id,
        )
        results = self._simulate_func(model_inputs)
        if output_dir is not None:
            self._write_output_csv(Path(output_dir), results)
        return results

    def _resolve_inputs(
        self,
        params: dict[str, Any],
        *,
        input_path: str | Path | None,
        run_id: str | None,
    ) -> dict[str, Any]:
        """Load staged JSON inputs or copy direct params."""
        if input_path is None:
            model_inputs = dict(params)
        else:
            loaded = json.loads(Path(input_path).read_text())
            if not isinstance(loaded, dict):
                raise ValueError(self._input_error_message)
            model_inputs = loaded

        if run_id is not None:
            model_inputs.setdefault("run_id", run_id)
        return model_inputs

    def _write_output_csv(
        self,
        output_dir: Path,
        results: Iterable[Any],
    ) -> None:
        """Write simulation results to a CSV file under ``output_dir``."""
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / self._output_filename).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for index, value in enumerate(results):
                writer.writerow(self._row_builder(index, value))
