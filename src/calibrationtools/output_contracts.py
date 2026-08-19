from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Protocol, TypeVar

OutputT = TypeVar("OutputT")
ValueT = TypeVar("ValueT")


class OutputContract(Protocol[OutputT]):
    """Parse model output from local MRP stdout or an output directory."""

    @property
    def output_filename(self) -> str: ...

    def read_output_dir(self, output_dir: Path) -> OutputT: ...

    def read_stdout(self, stdout: str | bytes) -> OutputT: ...


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


@dataclass(frozen=True)
class CSVColumnOutputContract(Generic[ValueT]):
    """Read one typed scalar column from a CSV output."""

    filename: str
    value_column: str
    value_parser: Callable[[str], ValueT]
    header_fields: tuple[str, ...] | None = None

    @property
    def output_filename(self) -> str:
        return self.filename

    def read_output_dir(self, output_dir: Path) -> list[ValueT]:
        csv_path = Path(output_dir) / self.filename
        if not csv_path.exists():
            raise FileNotFoundError(
                f"MRP model did not write expected output file: {csv_path}"
            )
        with csv_path.open(encoding="utf-8", newline="") as f:
            return self._read_csv_text(f.read())

    def read_stdout(self, stdout: str | bytes) -> list[ValueT]:
        output_text = _decode_output(stdout)
        csv_text = extract_csv_from_output_text(
            output_text,
            header_fields=self.header_fields,
        )
        return self._read_csv_text(csv_text)

    def _read_csv_text(self, csv_text: str) -> list[ValueT]:
        rows = csv.DictReader(io.StringIO(csv_text))
        if rows.fieldnames is None:
            raise ValueError("MRP model output CSV is missing a header row")
        try:
            return [self.value_parser(row[self.value_column]) for row in rows]
        except KeyError as exc:
            raise ValueError(
                f"MRP model output did not include a {self.value_column!r} column"
            ) from exc


@dataclass(frozen=True)
class CSVTableOutputContract:
    """Read a full CSV table into a JSON-serializable column mapping."""

    filename: str
    output_name: str
    orientation: Any = "columns"
    header_fields: tuple[str, ...] | None = None

    @property
    def output_filename(self) -> str:
        return self.filename

    def read_output_dir(
        self,
        output_dir: Path,
    ) -> dict[str, dict[str, list[str]]]:
        csv_path = Path(output_dir) / self.filename
        if not csv_path.exists():
            raise FileNotFoundError(
                f"MRP model did not write expected output file: {csv_path}"
            )
        return self._read_csv_text(csv_path.read_text(encoding="utf-8"))

    def read_stdout(
        self,
        stdout: str | bytes,
    ) -> dict[str, dict[str, list[str]]]:
        output_text = _decode_output(stdout)
        csv_text = extract_csv_from_output_text(
            output_text,
            header_fields=self.header_fields,
        )
        return self._read_csv_text(csv_text)

    def _read_csv_text(self, csv_text: str) -> dict[str, dict[str, list[str]]]:
        if _enum_value(self.orientation) != "columns":
            raise ValueError(
                f"unsupported CSV table orientation: {_enum_value(self.orientation)}"
            )
        rows = csv.DictReader(io.StringIO(csv_text))
        if rows.fieldnames is None:
            raise ValueError(
                "MRP model output CSV table is missing a header row"
            )
        columns = {field: [] for field in rows.fieldnames}
        for row in rows:
            for field in rows.fieldnames:
                columns[field].append(row[field])
        return {self.output_name: columns}


def make_output_contract_from_cloud_config(
    output: Any,
) -> OutputContract[Any]:
    if _enum_value(output.mode) == "csv_table":
        assert output.output_name is not None
        assert output.orientation is not None
        return CSVTableOutputContract(
            filename=output.filename,
            output_name=output.output_name,
            orientation=output.orientation,
            header_fields=getattr(output, "header_fields", None),
        )

    if output.csv_value_column is None:
        raise ValueError("cloud.output.csv_value_column is required")
    if output.csv_value_type is None:
        raise ValueError("cloud.output.csv_value_type is required")
    return CSVColumnOutputContract(
        filename=output.filename,
        value_column=output.csv_value_column,
        value_parser=output.csv_value_type.parser(),
        header_fields=getattr(output, "header_fields", None),
    )


def _decode_output(stdout: str | bytes) -> str:
    if isinstance(stdout, bytes):
        return stdout.decode()
    return stdout


def _enum_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    return value
