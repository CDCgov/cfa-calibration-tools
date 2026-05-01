from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from mrp import run as mrp_run

from .json_utils import to_jsonable

_ValueT = TypeVar("_ValueT")


def extract_csv_from_output_text(
    output_text: str,
    *,
    header_fields: tuple[str, ...] | None = None,
) -> str:
    if header_fields is None:
        return output_text

    lines = output_text.splitlines()
    for idx, line in enumerate(lines):
        if all(field in line for field in header_fields):
            return "\n".join(lines[idx:])
    return output_text


def read_csv_column_from_output_dir(
    output_dir: Path,
    *,
    output_filename: str,
    value_column: str,
    value_parser: Callable[[str], _ValueT],
) -> list[_ValueT]:
    csv_path = Path(output_dir) / output_filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MRP model did not write expected output file: {csv_path}"
        )
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        try:
            return [value_parser(row[value_column]) for row in rows]
        except KeyError as exc:
            raise ValueError(
                f"MRP model output did not include a {value_column!r} column"
            ) from exc


def make_csv_output_dir_reader(
    *,
    output_filename: str,
    value_column: str,
    value_parser: Callable[[str], _ValueT],
) -> Callable[[Path], list[_ValueT]]:
    def _reader(output_dir: Path) -> list[_ValueT]:
        return read_csv_column_from_output_dir(
            output_dir,
            output_filename=output_filename,
            value_column=value_column,
            value_parser=value_parser,
        )

    return _reader


class CSVOutputMRPRunner(Generic[_ValueT]):
    """Run an MRP config and parse a typed list from CSV output."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        output_filename: str,
        value_column: str,
        value_parser: Callable[[str], _ValueT],
        header_fields: tuple[str, ...] | None = None,
        mrp_run_func: Callable[..., Any] = mrp_run,
    ) -> None:
        self.config_path = Path(config_path)
        self.output_filename = output_filename
        self.value_column = value_column
        self._value_parser = value_parser
        self._header_fields = header_fields
        self._mrp_run = mrp_run_func
        self.read_output_dir = make_csv_output_dir_reader(
            output_filename=output_filename,
            value_column=value_column,
            value_parser=value_parser,
        )

    def simulate(
        self,
        params: dict[str, Any],
        *,
        input_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        run_id: str | None = None,
    ) -> list[_ValueT]:
        overrides: dict[str, Any]
        if input_path is not None:
            overrides = {"input": str(Path(input_path).resolve())}
            if output_dir is None:
                overrides["output"] = {"spec": "stdout"}
        else:
            overrides = {
                "input": to_jsonable(params),
                "output": {"spec": "stdout"},
            }

        run_kwargs: dict[str, Any] = {}
        if output_dir is not None:
            run_kwargs["output_dir"] = str(output_dir)

        result = self._mrp_run(self.config_path, overrides, **run_kwargs)
        if not result.ok:
            prefix = f"run {run_id}: " if run_id else ""
            raise RuntimeError(prefix + result.stderr.decode())

        if output_dir is not None:
            return self.read_output_dir(Path(output_dir))

        output_text = result.stdout.decode()
        csv_text = extract_csv_from_output_text(
            output_text,
            header_fields=self._header_fields,
        )
        rows = csv.DictReader(io.StringIO(csv_text))
        try:
            return [self._value_parser(row[self.value_column]) for row in rows]
        except KeyError as exc:
            raise ValueError(
                f"MRP model output did not include a {self.value_column!r} column"
            ) from exc
