import csv
import io
from pathlib import Path
from typing import Any

import numpy as np
from mrp import run as mrp_run


DEFAULT_MRP_CONFIG_PATH = (
    Path(__file__).resolve().parents[4] / "example_model.mrp.toml"
)
DEFAULT_DOCKER_MRP_CONFIG_PATH = (
    Path(__file__).resolve().parents[4] / "example_model.mrp.docker.toml"
)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class ExampleModelMRPRunner:
    """Run the example model through MRP and return populations as Python data."""

    def __init__(self, config_path: str | Path = DEFAULT_MRP_CONFIG_PATH):
        self.config_path = Path(config_path)

    def simulate(self, params: dict[str, Any]) -> list[int]:
        result = mrp_run(
            self.config_path,
            {
                "input": _to_jsonable(params),
                "output": {"spec": "stdout"},
            },
        )
        if not result.ok:
            raise RuntimeError(result.stderr.decode())

        output_text = result.stdout.decode()
        csv_text = self._extract_population_csv(output_text)
        rows = csv.DictReader(io.StringIO(csv_text))
        try:
            return [int(row["population"]) for row in rows]
        except KeyError as exc:
            raise ValueError(
                "MRP model output did not include a 'population' column"
            ) from exc

    @staticmethod
    def _extract_population_csv(output_text: str) -> str:
        lines = output_text.splitlines()
        for idx, line in enumerate(lines):
            if "population" in line and "generation" in line:
                return "\n".join(lines[idx:])
        return output_text
