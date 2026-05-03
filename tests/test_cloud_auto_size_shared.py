import subprocess

import pytest

from calibrationtools.cloud.auto_size import (
    resolve_cloud_auto_size,
    run_local_memory_probe,
    run_local_mrp_memory_probe,
)


def test_run_local_memory_probe_returns_child_peak_rss(monkeypatch):
    captured = {}

    def fake_run(cmd, *, input, capture_output, text, check):
        captured["cmd"] = cmd
        captured["input"] = input
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["check"] = check
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"peak_rss_bytes": 123456}\n',
            stderr="",
        )

    monkeypatch.setattr(
        "calibrationtools.cloud.auto_size.subprocess.run",
        fake_run,
    )

    assert (
        run_local_memory_probe(
            "some_model.probe",
            {"seed": 1},
            run_id="probe-1",
        )
        == 123456
    )
    assert captured["cmd"][-3:] == [
        "-m",
        "some_model.probe",
        "--child",
    ]
    assert '"run_id": "probe-1"' in captured["input"]
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["check"] is False


def test_run_local_memory_probe_raises_with_child_stderr(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            2,
            stdout="",
            stderr="probe exploded",
        )

    monkeypatch.setattr(
        "calibrationtools.cloud.auto_size.subprocess.run",
        fake_run,
    )

    with pytest.raises(RuntimeError, match="probe exploded"):
        run_local_memory_probe("some_model.probe", {"seed": 1})


def test_run_local_memory_probe_rejects_malformed_child_output(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"not_peak_rss_bytes": 1}\n',
            stderr="",
        )

    monkeypatch.setattr(
        "calibrationtools.cloud.auto_size.subprocess.run",
        fake_run,
    )

    with pytest.raises(RuntimeError, match="peak_rss_bytes"):
        run_local_memory_probe("some_model.probe", {"seed": 1})


def test_run_local_mrp_memory_probe_uses_shared_probe_module(monkeypatch):
    captured = {}

    def fake_probe(probe_module, base_inputs, *, run_id, python_executable):
        captured["probe_module"] = probe_module
        captured["base_inputs"] = base_inputs
        captured["run_id"] = run_id
        captured["python_executable"] = python_executable
        return 123

    monkeypatch.setattr(
        "calibrationtools.cloud.auto_size.run_local_memory_probe",
        fake_probe,
    )

    assert (
        run_local_mrp_memory_probe(
            "model.mrp.toml",
            {"seed": 1},
            run_id="probe-1",
            python_executable="/usr/bin/python",
        )
        == 123
    )
    assert captured == {
        "probe_module": "calibrationtools.cloud.auto_size",
        "base_inputs": {
            "mrp_config_path": "model.mrp.toml",
            "input": {"seed": 1},
        },
        "run_id": "probe-1",
        "python_executable": "/usr/bin/python",
    }


def test_resolve_cloud_auto_size_rejects_auto_size_without_cloud():
    with pytest.raises(ValueError, match="--auto-size requires --cloud"):
        resolve_cloud_auto_size(
            auto_size=True,
            cloud=False,
            max_concurrent_simulations=50,
            max_concurrent_simulations_explicit=False,
            vm_size="large",
            measure_task_peak_rss_bytes=lambda: 1,
        )


def test_resolve_cloud_auto_size_keeps_defaults_without_auto_size():
    sizing = resolve_cloud_auto_size(
        auto_size=False,
        cloud=True,
        max_concurrent_simulations=50,
        max_concurrent_simulations_explicit=False,
    )

    assert sizing.max_concurrent_simulations == 50
    assert sizing.task_slots_per_node_override is None
    assert sizing.summary is None


def test_resolve_cloud_auto_size_sets_default_concurrency():
    sizing = resolve_cloud_auto_size(
        auto_size=True,
        cloud=True,
        max_concurrent_simulations=50,
        max_concurrent_simulations_explicit=False,
        vm_size="large",
        measure_task_peak_rss_bytes=lambda: 10 * 1024**3,
    )

    assert sizing.max_concurrent_simulations == 25
    assert sizing.task_slots_per_node_override == 5
    assert sizing.summary is not None
    assert sizing.summary.measured_task_peak_rss_bytes == 10 * 1024**3
    assert sizing.summary.memory_task_slots_per_node == 5
    assert sizing.summary.max_task_slots_per_node == 64
    assert sizing.summary.task_slots_per_node == 5


def test_resolve_cloud_auto_size_scales_default_concurrency_by_pool_max_nodes():
    sizing = resolve_cloud_auto_size(
        auto_size=True,
        cloud=True,
        max_concurrent_simulations=50,
        max_concurrent_simulations_explicit=False,
        vm_size="large",
        pool_max_nodes=3,
        measure_task_peak_rss_bytes=lambda: 10 * 1024**3,
    )

    assert sizing.max_concurrent_simulations == 15
    assert sizing.task_slots_per_node_override == 5


def test_resolve_cloud_auto_size_caps_default_concurrency_to_batch_limit():
    sizing = resolve_cloud_auto_size(
        auto_size=True,
        cloud=True,
        max_concurrent_simulations=50,
        max_concurrent_simulations_explicit=False,
        vm_size="large",
        measure_task_peak_rss_bytes=lambda: 100 * 1024**2,
    )

    assert sizing.max_concurrent_simulations == 320
    assert sizing.task_slots_per_node_override == 64
    assert sizing.summary is not None
    assert sizing.summary.memory_task_slots_per_node == 557
    assert sizing.summary.max_task_slots_per_node == 64
    assert sizing.summary.task_slots_per_node == 64


def test_resolve_cloud_auto_size_preserves_explicit_concurrency():
    sizing = resolve_cloud_auto_size(
        auto_size=True,
        cloud=True,
        max_concurrent_simulations=17,
        max_concurrent_simulations_explicit=True,
        vm_size="large",
        measure_task_peak_rss_bytes=lambda: 10 * 1024**3,
    )

    assert sizing.max_concurrent_simulations == 17
    assert sizing.task_slots_per_node_override == 5


def test_resolve_cloud_auto_size_unknown_vm_size_fails_before_probe():
    def fail_if_probe_runs():
        raise AssertionError("probe should not run")

    with pytest.raises(ValueError, match="unknown vm_size"):
        resolve_cloud_auto_size(
            auto_size=True,
            cloud=True,
            max_concurrent_simulations=50,
            max_concurrent_simulations_explicit=False,
            vm_size="unknown-sku",
            measure_task_peak_rss_bytes=fail_if_probe_runs,
        )
