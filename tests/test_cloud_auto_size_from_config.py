from __future__ import annotations

from pathlib import Path

import pytest

from calibrationtools.cloud.auto_size import resolve_cloud_sizing_from_config


def _write_cloud_config(
    tmp_path: Path,
    *,
    auto_size_body: str = (
        'probe = "mrp"\nlocal_mrp_config_path = "model.mrp.toml"\n'
    ),
) -> Path:
    """Write a minimal cloud config and return its path."""
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (tmp_path / "model.mrp.toml").write_text(
        "input = {}\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "cloud_config.toml"
    config_path.write_text(
        f"""
[cloud]
keyvault = "kv"
vm_size = "large"
pool_max_nodes = 5

[cloud.image]
local_image = "local"
repository = "repo"
build_context = "."
dockerfile = "Dockerfile"
task_mrp_config_path = "/app/task.toml"

[cloud.resources]
pool_prefix = "pool"
job_prefix = "job"
input_container_prefix = "input"
output_container_prefix = "output"
logs_container_prefix = "logs"

[cloud.output]
filename = "output.csv"
csv_value_column = "population"
csv_value_type = "int"

[cloud.auto_size]
{auto_size_body}
""",
        encoding="utf-8",
    )
    return config_path


def test_resolve_cloud_sizing_from_config_uses_mrp_probe(
    monkeypatch, tmp_path
):
    """Resolve auto-size settings from config with the shared MRP probe."""
    config_path = _write_cloud_config(tmp_path)
    captured = {}

    def fake_probe(local_mrp_config_path, base_inputs):
        captured["local_mrp_config_path"] = local_mrp_config_path
        captured["base_inputs"] = base_inputs
        return 10 * 1024**3

    monkeypatch.setattr(
        "calibrationtools.cloud.auto_size.run_local_mrp_memory_probe",
        fake_probe,
    )

    sizing = resolve_cloud_sizing_from_config(
        cloud_config_path=config_path,
        base_inputs={"seed": 1},
        auto_size=True,
        cloud=True,
        max_concurrent_simulations=50,
        max_concurrent_simulations_explicit=False,
    )

    assert sizing.max_concurrent_simulations == 25
    assert sizing.task_slots_per_node_override == 5
    assert captured == {
        "local_mrp_config_path": tmp_path / "model.mrp.toml",
        "base_inputs": {"seed": 1},
    }


def test_resolve_cloud_sizing_from_config_uses_custom_probe_module(
    monkeypatch,
    tmp_path,
):
    """Resolve auto-size settings from config with a custom probe module."""
    config_path = _write_cloud_config(
        tmp_path,
        auto_size_body='probe_module = "some_model.probe"\n',
    )
    captured = {}

    def fake_probe(probe_module, base_inputs):
        captured["probe_module"] = probe_module
        captured["base_inputs"] = base_inputs
        return 10 * 1024**3

    monkeypatch.setattr(
        "calibrationtools.cloud.auto_size.run_local_memory_probe",
        fake_probe,
    )

    sizing = resolve_cloud_sizing_from_config(
        cloud_config_path=config_path,
        base_inputs={"seed": 2},
        auto_size=True,
        cloud=True,
        max_concurrent_simulations=17,
        max_concurrent_simulations_explicit=True,
    )

    assert sizing.max_concurrent_simulations == 17
    assert sizing.task_slots_per_node_override == 5
    assert captured == {
        "probe_module": "some_model.probe",
        "base_inputs": {"seed": 2},
    }


def test_resolve_cloud_sizing_from_config_rejects_missing_probe(tmp_path):
    """Fail clearly when auto-size is enabled without a configured probe."""
    config_path = _write_cloud_config(tmp_path, auto_size_body="")

    with pytest.raises(ValueError, match="auto_size"):
        resolve_cloud_sizing_from_config(
            cloud_config_path=config_path,
            base_inputs={"seed": 1},
            auto_size=True,
            cloud=True,
            max_concurrent_simulations=50,
            max_concurrent_simulations_explicit=False,
        )
