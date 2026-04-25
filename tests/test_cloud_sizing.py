import pytest

from calibrationtools.cloud.sizing import (
    GIBIBYTE,
    compute_task_slots_per_node,
    resolve_vm_memory_bytes,
)


@pytest.mark.parametrize(
    ("vm_size", "expected_gib"),
    [
        ("xsmall", 8),
        ("small", 16),
        ("medium", 32),
        ("large", 64),
        ("xlarge", 128),
        ("Standard_D2s_v3", 8),
        ("Standard_D4s_v3", 16),
        ("Standard_D8s_v3", 32),
        ("Standard_D16s_v3", 64),
        ("Standard_D32s_v3", 128),
    ],
)
def test_resolve_vm_memory_bytes_supports_documented_sizes(
    vm_size, expected_gib
):
    assert resolve_vm_memory_bytes(vm_size) == expected_gib * GIBIBYTE


def test_resolve_vm_memory_bytes_rejects_unknown_raw_sku():
    with pytest.raises(ValueError, match="Cannot auto-size unknown vm_size"):
        resolve_vm_memory_bytes("Standard_D48s_v3")


def test_compute_task_slots_per_node_reserves_fifteen_percent():
    assert compute_task_slots_per_node(
        measured_task_peak_rss_bytes=10 * GIBIBYTE,
        vm_memory_bytes=64 * GIBIBYTE,
    ) == 5


def test_compute_task_slots_per_node_rejects_task_that_exceeds_budget():
    with pytest.raises(ValueError, match="choose a larger vm_size"):
        compute_task_slots_per_node(
            measured_task_peak_rss_bytes=7 * GIBIBYTE,
            vm_memory_bytes=8 * GIBIBYTE,
            reserve=0.15,
        )


@pytest.mark.parametrize("measured_bytes", [0, -1])
def test_compute_task_slots_per_node_rejects_invalid_measurements(
    measured_bytes,
):
    with pytest.raises(ValueError, match="measured_task_peak_rss_bytes"):
        compute_task_slots_per_node(
            measured_task_peak_rss_bytes=measured_bytes,
            vm_memory_bytes=8 * GIBIBYTE,
        )
