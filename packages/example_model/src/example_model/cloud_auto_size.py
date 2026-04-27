from __future__ import annotations

from pathlib import Path
from typing import Any

from calibrationtools.cloud.auto_size import run_memory_probe_child_main

from .example_model import Binom_BP_Model


def _run_probe_simulation(
    base_inputs: dict[str, Any],
    run_id: str,
    output_dir: Path,
) -> None:
    model_inputs = dict(base_inputs)
    model_inputs.setdefault("run_id", run_id)
    results = Binom_BP_Model.simulate(model_inputs)
    (output_dir / "output.csv").write_text(
        "generation,population\n"
        + "".join(
            f"{generation},{population}\n"
            for generation, population in enumerate(results)
        )
    )


def main() -> None:
    run_memory_probe_child_main(_run_probe_simulation)


if __name__ == "__main__":
    main()
