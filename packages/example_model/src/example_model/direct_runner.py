"""Direct in-process runner for the example branching process."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .example_model import Binom_BP_Model


class ExampleModelDirectRunner:
    """Run the example model locally while honoring staged sampler I/O."""

    def simulate(
        self,
        params: dict[str, Any],
        *,
        input_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        run_id: str | None = None,
    ) -> list[int]:
        model_inputs = self._resolve_inputs(
            params,
            input_path=input_path,
            run_id=run_id,
        )
        results = Binom_BP_Model.simulate(model_inputs)
        if output_dir is not None:
            self._write_output_csv(Path(output_dir), results)
        return results

    @staticmethod
    def _resolve_inputs(
        params: dict[str, Any],
        *,
        input_path: str | Path | None,
        run_id: str | None,
    ) -> dict[str, Any]:
        if input_path is None:
            model_inputs = dict(params)
        else:
            loaded = json.loads(Path(input_path).read_text())
            if not isinstance(loaded, dict):
                raise ValueError("Example model input JSON must be an object.")
            model_inputs = loaded

        if run_id is not None:
            model_inputs.setdefault("run_id", run_id)
        return model_inputs

    @staticmethod
    def _write_output_csv(output_dir: Path, results: list[int]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "output.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["generation", "population"],
            )
            writer.writeheader()
            for generation, population in enumerate(results):
                writer.writerow(
                    {
                        "generation": generation,
                        "population": population,
                    }
                )
