from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mrp import run as mrp_run

from .config import DEFAULT_POOL_MAX_NODES, load_cloud_model_config
from .sizing import (
    DEFAULT_MEMORY_RESERVE_FRACTION,
    compute_task_slots_per_node,
    resolve_vm_memory_bytes,
    resolve_vm_task_slots_per_node_limit,
)

ProbeSimulation = Callable[[dict[str, Any], str, Path], None]


@dataclass(frozen=True)
class AutoSizeSummary:
    vm_size: str
    vm_memory_bytes: int
    measured_task_peak_rss_bytes: int
    reserve: float
    memory_task_slots_per_node: int
    max_task_slots_per_node: int
    task_slots_per_node: int


@dataclass(frozen=True)
class CloudSizing:
    max_concurrent_simulations: int
    task_slots_per_node_override: int | None = None
    summary: AutoSizeSummary | None = None


def run_local_memory_probe(
    probe_module: str,
    base_inputs: dict[str, Any],
    *,
    run_id: str = "auto-size-probe",
    python_executable: str | Path = sys.executable,
) -> int:
    """Run one probe module in a subprocess and return child peak RSS."""
    with tempfile.TemporaryDirectory(prefix="cloud-auto-size-") as tmp:
        request = {
            "base_inputs": base_inputs,
            "run_id": run_id,
            "output_dir": tmp,
        }
        completed = subprocess.run(
            [str(python_executable), "-m", probe_module, "--child"],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=False,
        )

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"auto-size probe failed: {detail}")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"auto-size probe did not return valid JSON: {completed.stdout!r}"
        ) from exc

    peak_rss_bytes = payload.get("peak_rss_bytes")
    if not isinstance(peak_rss_bytes, int) or peak_rss_bytes < 1:
        raise RuntimeError(
            "auto-size probe JSON must include positive integer "
            f"peak_rss_bytes (got {peak_rss_bytes!r})"
        )
    return peak_rss_bytes


def run_local_mrp_memory_probe(
    local_mrp_config_path: str | Path,
    base_inputs: dict[str, Any],
    *,
    run_id: str = "auto-size-probe",
    python_executable: str | Path = sys.executable,
) -> int:
    """Measure peak RSS for a generic local ``mrp run`` probe."""
    return run_local_memory_probe(
        "calibrationtools.cloud.auto_size",
        {
            "mrp_config_path": str(local_mrp_config_path),
            "input": base_inputs,
        },
        run_id=run_id,
        python_executable=python_executable,
    )


def resolve_cloud_auto_size(
    *,
    auto_size: bool,
    cloud: bool,
    max_concurrent_simulations: int,
    max_concurrent_simulations_explicit: bool,
    vm_size: str | None = None,
    pool_max_nodes: int = DEFAULT_POOL_MAX_NODES,
    measure_task_peak_rss_bytes: Callable[[], int] | None = None,
    reserve: float = DEFAULT_MEMORY_RESERVE_FRACTION,
) -> CloudSizing:
    """Resolve cloud concurrency and task slots for optional auto-size."""
    if auto_size and not cloud:
        raise ValueError("--auto-size requires --cloud")
    if pool_max_nodes < 1:
        raise ValueError("pool_max_nodes must be at least 1")

    if not auto_size:
        return CloudSizing(
            max_concurrent_simulations=max_concurrent_simulations,
        )

    if vm_size is None:
        raise ValueError("vm_size is required when auto_size is enabled")
    if measure_task_peak_rss_bytes is None:
        raise ValueError(
            "measure_task_peak_rss_bytes is required when auto_size is enabled"
        )

    vm_memory_bytes = resolve_vm_memory_bytes(vm_size)
    measured_task_peak_rss_bytes = measure_task_peak_rss_bytes()
    memory_task_slots_per_node = compute_task_slots_per_node(
        measured_task_peak_rss_bytes=measured_task_peak_rss_bytes,
        vm_memory_bytes=vm_memory_bytes,
        reserve=reserve,
    )
    max_task_slots_per_node = resolve_vm_task_slots_per_node_limit(vm_size)
    # Keep the shared helper's capped path exercised here too, so future
    # callers get the same validation and clamping behavior.
    task_slots_per_node = compute_task_slots_per_node(
        measured_task_peak_rss_bytes=measured_task_peak_rss_bytes,
        vm_memory_bytes=vm_memory_bytes,
        max_task_slots_per_node=max_task_slots_per_node,
        reserve=reserve,
    )
    if not max_concurrent_simulations_explicit:
        max_concurrent_simulations = task_slots_per_node * pool_max_nodes

    return CloudSizing(
        max_concurrent_simulations=max_concurrent_simulations,
        task_slots_per_node_override=task_slots_per_node,
        summary=AutoSizeSummary(
            vm_size=vm_size,
            vm_memory_bytes=vm_memory_bytes,
            measured_task_peak_rss_bytes=measured_task_peak_rss_bytes,
            reserve=reserve,
            memory_task_slots_per_node=memory_task_slots_per_node,
            max_task_slots_per_node=max_task_slots_per_node,
            task_slots_per_node=task_slots_per_node,
        ),
    )


def resolve_cloud_sizing_from_config(
    *,
    cloud_config_path: str | Path,
    base_inputs: dict[str, Any],
    auto_size: bool,
    cloud: bool,
    max_concurrent_simulations: int,
    max_concurrent_simulations_explicit: bool,
) -> CloudSizing:
    """Resolve optional cloud auto-size settings from a cloud config file."""
    if not auto_size or not cloud:
        return resolve_cloud_auto_size(
            auto_size=auto_size,
            cloud=cloud,
            max_concurrent_simulations=max_concurrent_simulations,
            max_concurrent_simulations_explicit=(
                max_concurrent_simulations_explicit
            ),
        )

    cloud_config = load_cloud_model_config(cloud_config_path)
    settings = cloud_config.runtime_settings

    if cloud_config.auto_size.probe == "mrp":
        local_mrp_config_path = cloud_config.auto_size.local_mrp_config_path
        if local_mrp_config_path is None:
            raise ValueError(
                "cloud.auto_size.local_mrp_config_path is required"
            )

        def measure_task_peak_rss_bytes() -> int:
            """Measure RSS with the configured shared MRP probe."""
            return run_local_mrp_memory_probe(
                local_mrp_config_path,
                base_inputs,
            )

    elif cloud_config.auto_size.probe_module:
        probe_module = cloud_config.auto_size.probe_module

        def measure_task_peak_rss_bytes() -> int:
            """Measure RSS with the configured custom probe module."""
            return run_local_memory_probe(probe_module, base_inputs)

    else:
        raise ValueError(
            "cloud.auto_size requires probe='mrp' or probe_module when "
            "auto-size is enabled"
        )

    return resolve_cloud_auto_size(
        auto_size=auto_size,
        cloud=cloud,
        max_concurrent_simulations=max_concurrent_simulations,
        max_concurrent_simulations_explicit=max_concurrent_simulations_explicit,
        vm_size=settings.vm_size,
        pool_max_nodes=settings.pool_max_nodes,
        measure_task_peak_rss_bytes=measure_task_peak_rss_bytes,
    )


def format_bytes(size: int) -> str:
    """Format a byte count for concise operator output."""
    if size >= 1024**3:
        return f"{size / 1024**3:.1f} GiB"
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MiB"
    return f"{size} bytes"


def print_cloud_auto_size_summary(sizing: CloudSizing) -> None:
    """Print a human-readable auto-size summary when one was computed."""
    summary = sizing.summary
    if summary is None:
        return

    cap_note = ""
    if summary.task_slots_per_node < summary.memory_task_slots_per_node:
        cap_note = (
            f", capped_from_ram_slots={summary.memory_task_slots_per_node}"
        )

    print(
        (
            "[cloud-run] auto-size simulation RAM "
            f"measured_peak_rss="
            f"{summary.measured_task_peak_rss_bytes} bytes "
            f"({format_bytes(summary.measured_task_peak_rss_bytes)}), "
            f"vm_size={summary.vm_size}, "
            f"vm_ram={summary.vm_memory_bytes} bytes "
            f"({format_bytes(summary.vm_memory_bytes)}), "
            f"reserve={summary.reserve:.0%}, "
            f"batch_slot_limit={summary.max_task_slots_per_node}, "
            f"task_slots_per_node={summary.task_slots_per_node}"
            f"{cap_note}, "
            f"max_concurrent_simulations_per_node="
            f"{summary.task_slots_per_node}, "
            f"max_concurrent_simulations_total="
            f"{sizing.max_concurrent_simulations}"
        ),
        file=sys.stderr,
        flush=True,
    )


def run_memory_probe_child_main(run_probe_simulation: ProbeSimulation) -> None:
    """Run an auto-size memory probe child process entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if not args.child:
        raise SystemExit("cloud auto-size probe helpers are internal")
    _run_memory_probe_child(run_probe_simulation)


def _run_memory_probe_child(run_probe_simulation: ProbeSimulation) -> None:
    request = json.loads(sys.stdin.read())
    if not isinstance(request, dict):
        raise ValueError("probe request must be a JSON object")

    base_inputs = request.get("base_inputs")
    if not isinstance(base_inputs, dict):
        raise ValueError("probe request must include object base_inputs")

    run_id = request.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("probe request must include non-empty run_id")

    output_dir_value = request.get("output_dir")
    if not isinstance(output_dir_value, str) or not output_dir_value:
        raise ValueError("probe request must include non-empty output_dir")

    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_probe_simulation(base_inputs, run_id, output_dir)
    print(json.dumps({"peak_rss_bytes": _peak_rss_bytes()}), flush=True)


def _run_mrp_probe_simulation(
    base_inputs: dict[str, Any],
    run_id: str,
    output_dir: Path,
) -> None:
    """Run one local MRP simulation for the shared memory probe child."""
    config_path = base_inputs.get("mrp_config_path")
    if not isinstance(config_path, str) or not config_path:
        raise ValueError("MRP probe requires string mrp_config_path")
    input_value = base_inputs.get("input")
    if not isinstance(input_value, dict):
        raise ValueError("MRP probe requires object input")
    model_input = dict(input_value)
    model_input.setdefault("run_id", run_id)
    result = mrp_run(
        config_path,
        {"input": model_input},
        output_dir=str(output_dir),
    )
    if not result.ok:
        raise RuntimeError(result.stderr.decode())


def _peak_rss_bytes() -> int:
    """Return peak resident memory for the current process in bytes."""
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak_rss
    return peak_rss * 1024


def main() -> None:
    """Run the shared generic MRP auto-size probe child entrypoint."""
    run_memory_probe_child_main(_run_mrp_probe_simulation)


if __name__ == "__main__":
    main()
