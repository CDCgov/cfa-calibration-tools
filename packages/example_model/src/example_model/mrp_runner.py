import csv
import io
from pathlib import Path
from typing import Any

from calibrationtools.json_utils import to_jsonable
from mrp import run as mrp_run


DEFAULT_MRP_CONFIG_PATH = (
    Path(__file__).resolve().parents[4] / "example_model.mrp.toml"
)
DEFAULT_DOCKER_MRP_CONFIG_PATH = (
    Path(__file__).resolve().parents[4] / "example_model.mrp.docker.toml"
)

class ExampleModelMRPRunner:
    """Run the example model through MRP and return populations as Python data."""

    def __init__(self, config_path: str | Path = DEFAULT_MRP_CONFIG_PATH):
        self.config_path = Path(config_path)

    def simulate(
        self,
        params: dict[str, Any],
        *,
        input_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        run_id: str | None = None,
    ) -> list[int]:
        overrides: dict[str, Any]
        if input_path is not None:
            overrides = {"input": str(input_path)}
        else:
            overrides = {
                "input": to_jsonable(params),
                "output": {"spec": "stdout"},
            }

        run_kwargs: dict[str, Any] = {}
        if output_dir is not None:
            run_kwargs["output_dir"] = str(output_dir)

        result = mrp_run(self.config_path, overrides, **run_kwargs)
        if not result.ok:
            prefix = f"run {run_id}: " if run_id else ""
            raise RuntimeError(prefix + result.stderr.decode())

        if output_dir is not None:
            csv_path = Path(output_dir) / "output.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"MRP model did not write expected output file: {csv_path}"
                )
            with csv_path.open() as f:
                rows = csv.DictReader(f)
                try:
                    return [int(row["population"]) for row in rows]
                except KeyError as exc:
                    raise ValueError(
                        "MRP model output did not include a 'population' column"
                    ) from exc
        else:
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
