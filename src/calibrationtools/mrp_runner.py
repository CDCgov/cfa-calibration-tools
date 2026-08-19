from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from mrp import run as mrp_run

from .json_utils import to_jsonable
from .output_contracts import OutputContract

_OutputT = TypeVar("_OutputT")


class MRPOutputRunner(Generic[_OutputT]):
    """Run an MRP config and parse output through an output contract."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        output_contract: OutputContract[_OutputT],
        mrp_run_func: Callable[..., Any] = mrp_run,
    ) -> None:
        self.config_path = Path(config_path)
        self.output_contract = output_contract
        self.output_filename = output_contract.output_filename
        self._mrp_run = mrp_run_func
        self.read_output_dir = output_contract.read_output_dir

    def simulate(
        self,
        params: dict[str, Any],
        *,
        input_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        run_id: str | None = None,
    ) -> _OutputT:
        overrides: dict[str, Any]
        if input_path is not None:
            overrides = {"input": str(Path(input_path).resolve())}
            if output_dir is None:
                overrides["output"] = {"spec": "stdout"}
        else:
            overrides = {"input": to_jsonable(params)}
            if output_dir is None:
                overrides["output"] = {"spec": "stdout"}

        run_kwargs: dict[str, Any] = {}
        if output_dir is not None:
            run_kwargs["output_dir"] = str(output_dir)

        result = self._mrp_run(self.config_path, overrides, **run_kwargs)
        if not result.ok:
            prefix = f"run {run_id}: " if run_id else ""
            raise RuntimeError(prefix + result.stderr.decode())

        if output_dir is not None:
            return self.output_contract.read_output_dir(Path(output_dir))

        return self.output_contract.read_stdout(result.stdout)
