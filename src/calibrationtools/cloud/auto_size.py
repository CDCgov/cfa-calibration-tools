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

from .config import DEFAULT_POOL_MAX_NODES
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


def run_memory_probe_child_main(run_probe_simulation: ProbeSimulation) -> None:
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


def _peak_rss_bytes() -> int:
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak_rss
    return peak_rss * 1024
