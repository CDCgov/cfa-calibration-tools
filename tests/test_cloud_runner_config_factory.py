from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from calibrationtools.cloud.config import CloudRuntimeSettings
from calibrationtools.cloud.runner import (
    create_csv_cloud_mrp_runner_from_config,
    make_cloud_executor_mrp_config,
)


def test_create_csv_cloud_mrp_runner_from_config_wires_resolved_settings(
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

    runner = create_csv_cloud_mrp_runner_from_config(
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


def test_cloud_mrp_runner_uses_synthesized_executor_config_for_mrp_run(
    tmp_path: Path,
):
    config_path = tmp_path / "cloud_config.toml"
    output_dir = tmp_path / "output"
    calls: dict[str, Any] = {}

    class FakeSession:
        print_task_durations = False

        def to_runtime_cloud(self):
            return {"session_id": "session"}

    def fake_mrp_run(config_path_arg, overrides, **kwargs):
        calls["config_path"] = config_path_arg
        calls["overrides"] = overrides
        calls["kwargs"] = kwargs
        return SimpleNamespace(ok=True, stderr=b"")

    from calibrationtools.cloud.runner import CloudMRPRunner

    runner = object.__new__(CloudMRPRunner)
    runner.config_path = config_path
    runner.executor_mrp_config = make_cloud_executor_mrp_config()
    runner.session = FakeSession()
    runner._mrp_run = fake_mrp_run
    runner._read_output_dir = lambda path: ["ok"]
    runner._select_job_name = lambda run_id: "job"

    assert runner.simulate({}, output_dir=output_dir, run_id="g0-p0-a0") == [
        "ok"
    ]
    assert calls["config_path"] == {
        "runtime": {
            "spec": "inline",
            "callable": "calibrationtools.cloud.executor:execute_cloud_run",
        },
        "output": {"spec": "filesystem", "dir": "./output"},
    }
    assert calls["overrides"]["runtime"]["cloud"]["job_name"] == "job"


def test_cloud_runner_from_config_uses_synthesized_executor_config(
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
            self.executor_mrp_config = make_cloud_executor_mrp_config()
            captured["config_path"] = config_path_arg
            captured.update(kwargs)

    monkeypatch.setattr(
        "calibrationtools.cloud.runner.CloudMRPRunner", FakeRunner
    )

    runner = create_csv_cloud_mrp_runner_from_config(
        config_path,
        generation_count=2,
        max_concurrent_simulations=7,
    )

    assert isinstance(runner, FakeRunner)
    assert runner.executor_mrp_config == make_cloud_executor_mrp_config()
    assert "simulation_mrp_config_path" not in captured
