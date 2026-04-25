from __future__ import annotations

GIBIBYTE = 1024**3
DEFAULT_MEMORY_RESERVE_FRACTION = 0.15

_DOCUMENTED_VM_MEMORY_BYTES = {
    "xsmall": 8 * GIBIBYTE,
    "small": 16 * GIBIBYTE,
    "medium": 32 * GIBIBYTE,
    "large": 64 * GIBIBYTE,
    "xlarge": 128 * GIBIBYTE,
    "standard_d2s_v3": 8 * GIBIBYTE,
    "standard_d4s_v3": 16 * GIBIBYTE,
    "standard_d8s_v3": 32 * GIBIBYTE,
    "standard_d16s_v3": 64 * GIBIBYTE,
    "standard_d32s_v3": 128 * GIBIBYTE,
}


def resolve_vm_memory_bytes(vm_size: str) -> int:
    """Return RAM for documented cloud VM shorthands and Dsv3 SKUs."""
    normalized = vm_size.strip().lower()
    try:
        return _DOCUMENTED_VM_MEMORY_BYTES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_DOCUMENTED_VM_MEMORY_BYTES))
        raise ValueError(
            "Cannot auto-size unknown vm_size "
            f"{vm_size!r}. Supported values are: {supported}."
        ) from exc


def compute_task_slots_per_node(
    *,
    measured_task_peak_rss_bytes: int,
    vm_memory_bytes: int,
    reserve: float = DEFAULT_MEMORY_RESERVE_FRACTION,
) -> int:
    """Compute how many measured tasks fit on a VM after RAM headroom."""
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

    usable_vm_ram = int(vm_memory_bytes * (1.0 - reserve))
    task_slots = usable_vm_ram // measured_task_peak_rss_bytes
    if task_slots < 1:
        raise ValueError(
            "A single measured task exceeds the usable VM RAM budget; "
            "choose a larger vm_size or disable --auto-size."
        )
    return task_slots
