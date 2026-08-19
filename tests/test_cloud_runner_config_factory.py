from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from calibrationtools.cloud.config import CloudRuntimeSettings
from calibrationtools.cloud.runner import create_cloud_mrp_runner_from_config


def test_create_cloud_mrp_runner_from_config_wires_resolved_settings(
    monkeypatch,
    tmp_path: Path,
):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    config_path = tmp_path / "cloud_config.toml"
    config_path.write_text(
        """
[cloud]
keyvault = "kv"
vm_size = "large"
task_slots_per_node = 8

[cloud.image]
local_image = "local-model"
repository = "remote-model"
build_context = "."
dockerfile = "Dockerfile"
task_mrp_config_path = "/app/task.toml"

[cloud.resources]
pool_prefix = "model-pool"
job_prefix = "model-job"
input_container_prefix = "model-input"
output_container_prefix = "model-output"
logs_container_prefix = "model-logs"

[cloud.output]
filename = "population.csv"
csv_value_column = "population"
csv_value_type = "int"
""",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    class FakeRunner:
        def __init__(self, config_path_arg, **kwargs):
            captured["config_path"] = config_path_arg
            captured.update(kwargs)

    monkeypatch.setattr(
        "calibrationtools.cloud.runner.CloudMRPRunner", FakeRunner
    )

    runner = create_cloud_mrp_runner_from_config(
        config_path,
        generation_count=2,
        max_concurrent_simulations=7,
        task_slots_per_node_override=3,
        auto_size_summary=SimpleNamespace(task_slots_per_node=3),
    )

    assert isinstance(runner, FakeRunner)
    assert captured["config_path"] == config_path
    assert captured["generation_count"] == 2
    assert captured["max_concurrent_simulations"] == 7
    assert captured["repo_root"] == tmp_path
    assert captured["dockerfile"] == dockerfile
    assert captured["output_filename"] == "population.csv"
    assert captured["auto_size_summary"].task_slots_per_node == 3
    settings = captured["runtime_settings"]
    assert isinstance(settings, CloudRuntimeSettings)
    assert settings.task_slots_per_node == 3
    assert settings.repository == "remote-model"


def test_cloud_mrp_runner_sync_simulate_bridges_to_native_async_path(
    tmp_path: Path,
):
    output_dir = tmp_path / "output"
    calls: dict[str, Any] = {}

    from calibrationtools.cloud.runner import CloudMRPRunner

    async def fake_simulate_async(params, **kwargs):
        calls["params"] = params
        calls["kwargs"] = kwargs
        return ["ok"]

    runner = object.__new__(CloudMRPRunner)
    runner.simulate_async = fake_simulate_async

    assert runner.simulate(
        {"seed": 1},
        output_dir=output_dir,
        run_id="g0-p0-a0",
    ) == ["ok"]
    assert calls == {
        "params": {"seed": 1},
        "kwargs": {
            "input_path": None,
            "output_dir": output_dir,
            "run_id": "g0-p0-a0",
        },
    }
