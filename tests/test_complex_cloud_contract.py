from __future__ import annotations

from pathlib import Path
from typing import Any

from calibrationtools.cloud.config import load_cloud_model_config
from calibrationtools.cloud.runner import create_cloud_mrp_runner_from_config
from calibrationtools.cloud.task_payload import (
    CloudTaskContext,
    apply_task_payload_transforms,
    bind_shared_assets_to_session,
    resolve_shared_assets,
)
from calibrationtools.output_contracts import (
    make_output_contract_from_cloud_config,
)


def _write_complex_config(tmp_path: Path) -> Path:
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (tmp_path / "model.mrp.toml").write_text("input = {}\n", encoding="utf-8")
    (tmp_path / "reference.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "cloud_config.toml"
    config_path.write_text(
        """
[cloud]
keyvault = "kv"
max_parallel_output_downloads = 2

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
filename = "aggregated_deaths_report.csv"
mode = "csv_table"
output_name = "aggregated_deaths_report"

[cloud.shared_assets.reference]
source_json_pointer = "/epimodel.GlobalParams/reference_path"
blob_dir = "assets"
remote_path_var = "REFERENCE_PATH"

[cloud.task_payload]
task_output_dir = "{output_mount_path}/custom-output/{run_id}"

[[cloud.task_payload.transforms]]
name = "set-reference"
op = "set"
path = "/epimodel.GlobalParams/reference_remote_path"
value = "{REFERENCE_PATH}"
on_missing = "create"

[[cloud.task_payload.transforms]]
name = "set-output"
op = "set"
path = "/epimodel.GlobalParams/output_dir"
value = "{task_output_dir}"
on_missing = "create"

[cloud.auto_size]
probe = "local_task"
local_mrp_config_path = "model.mrp.toml"
memory_scope = "process_tree"
""",
        encoding="utf-8",
    )
    return config_path


def test_complex_cloud_contract_loads_transforms_and_parses_output(
    tmp_path: Path,
):
    config_path = _write_complex_config(tmp_path)
    base_inputs = {
        "epimodel.GlobalParams": {"reference_path": "reference.json"}
    }
    config = load_cloud_model_config(config_path)

    assets = resolve_shared_assets(
        config.shared_assets,
        base_payload=base_inputs,
        config_dir=tmp_path,
    )
    bound_assets = bind_shared_assets_to_session(
        assets,
        session_id="session",
        input_mount_path="/cloud-input",
    )
    context = CloudTaskContext(
        run_id="run-1",
        session_id="session",
        job_name="job",
        input_mount_path="/cloud-input",
        output_mount_path="/cloud-output",
        logs_mount_path="/cloud-logs",
        task_output_dir="/cloud-output/custom-output/run-1",
        shared_assets=bound_assets,
    )
    transformed = apply_task_payload_transforms(
        base_inputs,
        config.task_payload,
        context,
    )

    assert transformed["epimodel.GlobalParams"]["reference_remote_path"] == (
        "/cloud-input/assets/reference/session/reference.json"
    )
    assert transformed["epimodel.GlobalParams"]["output_dir"] == (
        "/cloud-output/custom-output/run-1"
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "aggregated_deaths_report.csv").write_text(
        "t_lower,t_upper,count\n0,1,10\n",
        encoding="utf-8",
    )
    contract = make_output_contract_from_cloud_config(config.output)
    assert contract.read_output_dir(output_dir) == {
        "aggregated_deaths_report": {
            "t_lower": ["0"],
            "t_upper": ["1"],
            "count": ["10"],
        }
    }


def test_complex_cloud_runner_creation_passes_resolved_contract(
    monkeypatch,
    tmp_path: Path,
):
    config_path = _write_complex_config(tmp_path)
    captured: dict[str, Any] = {}

    class FakeRunner:
        def __init__(self, config_path_arg, **kwargs):
            captured["config_path"] = config_path_arg
            captured.update(kwargs)

    monkeypatch.setattr(
        "calibrationtools.cloud.runner.CloudMRPRunner",
        FakeRunner,
    )

    runner = create_cloud_mrp_runner_from_config(
        config_path,
        generation_count=1,
        max_concurrent_simulations=2,
        base_inputs={
            "epimodel.GlobalParams": {"reference_path": "reference.json"}
        },
    )

    assert isinstance(runner, FakeRunner)
    assert captured["output_filename"] == "aggregated_deaths_report.csv"
    assert len(captured["shared_assets"]) == 1
    assert len(captured["task_payload_settings"].transforms) == 2
    assert captured["runtime_settings"].max_parallel_output_downloads == 2
