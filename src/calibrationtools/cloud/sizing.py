from __future__ import annotations

GIBIBYTE = 1024**3
DEFAULT_MEMORY_RESERVE_FRACTION = 0.15

_DOCUMENTED_VM_SPECS = {
    "xsmall": (2, 8 * GIBIBYTE),
    "small": (4, 16 * GIBIBYTE),
    "medium": (8, 32 * GIBIBYTE),
    "large": (16, 64 * GIBIBYTE),
    "xlarge": (32, 128 * GIBIBYTE),
    "standard_d2s_v3": (2, 8 * GIBIBYTE),
    "standard_d4s_v3": (4, 16 * GIBIBYTE),
    "standard_d8s_v3": (8, 32 * GIBIBYTE),
    "standard_d16s_v3": (16, 64 * GIBIBYTE),
    "standard_d32s_v3": (32, 128 * GIBIBYTE),
}


def _resolve_vm_spec(vm_size: str) -> tuple[int, int]:
    normalized = vm_size.strip().lower()
    try:
        return _DOCUMENTED_VM_SPECS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_DOCUMENTED_VM_SPECS))
        raise ValueError(
            "Cannot auto-size unknown vm_size "
            f"{vm_size!r}. Supported values are: {supported}."
        ) from exc


def resolve_vm_memory_bytes(vm_size: str) -> int:
    """Return RAM for documented cloud VM shorthands and Dsv3 SKUs."""
    return _resolve_vm_spec(vm_size)[1]


def resolve_vm_task_slots_per_node_limit(vm_size: str) -> int:
    """Return Azure Batch's task slot limit for known VM sizes."""
    vcpus = _resolve_vm_spec(vm_size)[0]
    return vcpus * 4


def compute_task_slots_per_node(
    *,
    measured_task_peak_rss_bytes: int,
    vm_memory_bytes: int,
    max_task_slots_per_node: int | None = None,
    reserve: float = DEFAULT_MEMORY_RESERVE_FRACTION,
) -> int:
    """Compute task slots from RAM headroom and an optional Batch limit."""
    if measured_task_peak_rss_bytes < 1:
        raise ValueError(
            "measured_task_peak_rss_bytes must be at least 1 "
            f"(got {measured_task_peak_rss_bytes})"
        )
    if vm_memory_bytes < 1:
        raise ValueError(
            f"vm_memory_bytes must be at least 1 (got {vm_memory_bytes})"
        )
    if reserve < 0 or reserve >= 1:
        raise ValueError(f"reserve must be in [0, 1) (got {reserve})")
    if max_task_slots_per_node is not None and max_task_slots_per_node < 1:
        raise ValueError(
            "max_task_slots_per_node must be at least 1 "
            f"(got {max_task_slots_per_node})"
        )

    usable_vm_ram = int(vm_memory_bytes * (1.0 - reserve))
    task_slots = usable_vm_ram // measured_task_peak_rss_bytes
    if task_slots < 1:
        raise ValueError(
            "A single measured task exceeds the usable VM RAM budget; "
            "choose a larger vm_size or disable --auto-size."
        )
    if max_task_slots_per_node is not None:
        task_slots = min(task_slots, max_task_slots_per_node)
    return task_slots
