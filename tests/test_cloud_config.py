from __future__ import annotations

from pathlib import Path

import pytest

from calibrationtools.cloud.config import (
    CloudCSVValueType,
    CloudOutputMode,
    CSVTableOrientation,
    load_cloud_model_config,
)


def _write_cloud_config(tmp_path: Path, *, csv_type: str = "int") -> Path:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    config_path = tmp_path / "cloud_config.toml"
    config_path.write_text(
        f"""
[cloud]
keyvault = "kv"
vm_size = "large"
jobs_per_session = 2
task_slots_per_node = 8
pool_max_nodes = 3
dispatch_buffer = 9

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
csv_value_type = "{csv_type}"

[cloud.auto_size]
probe = "mrp"
local_mrp_config_path = "model.mrp.toml"
""",
        encoding="utf-8",
    )
    (tmp_path / "model.mrp.toml").write_text(
        "input = {}\n",
        encoding="utf-8",
    )
    return config_path


def test_load_cloud_model_config_parses_model_facing_config(tmp_path: Path):
    config = load_cloud_model_config(_write_cloud_config(tmp_path))

    assert config.config_path == tmp_path / "cloud_config.toml"
    assert config.build_context == tmp_path
    assert config.dockerfile == tmp_path / "Dockerfile"
    assert config.runtime_settings.keyvault == "kv"
    assert config.runtime_settings.local_image == "local-model"
    assert config.runtime_settings.repository == "remote-model"
    assert config.runtime_settings.task_mrp_config_path == "/app/task.toml"
    assert config.runtime_settings.jobs_per_session == 2
    assert config.runtime_settings.task_slots_per_node == 8
    assert config.runtime_settings.pool_max_nodes == 3
    assert config.runtime_settings.dispatch_buffer == 9
    assert config.output.filename == "population.csv"
    assert config.output.csv_value_column == "population"
    assert config.output.csv_value_type is CloudCSVValueType.INT
    assert config.auto_size.probe == "mrp"
    assert (
        config.auto_size.local_mrp_config_path == tmp_path / "model.mrp.toml"
    )


def test_load_cloud_model_config_rejects_config_without_top_level_cloud(
    tmp_path: Path,
):
    config_path = tmp_path / "legacy.mrp.cloud.toml"
    config_path.write_text(
        """
[runtime.cloud]
keyvault = "kv"
local_image = "local-model"
repository = "remote-model"
task_mrp_config_path = "/app/task.toml"
pool_prefix = "model-pool"
job_prefix = "model-job"
input_container_prefix = "model-input"
output_container_prefix = "model-output"
logs_container_prefix = "model-logs"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"top-level \[cloud\] table"):
        load_cloud_model_config(config_path)


def test_load_cloud_model_config_rejects_unsupported_csv_value_type(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="csv_value_type"):
        load_cloud_model_config(_write_cloud_config(tmp_path, csv_type="json"))


def test_load_cloud_model_config_parses_csv_table_output_and_runtime_limit(
    tmp_path: Path,
):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    config_path = tmp_path / "cloud_config.toml"
    config_path.write_text(
        """
[cloud]
keyvault = "kv"
max_parallel_output_downloads = 3

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
filename = "aggregated_deaths_report.csv"
mode = "csv_table"
output_name = "aggregated_deaths_report"
orientation = "columns"
header_fields = ["t_lower", "t_upper", "count"]
""",
        encoding="utf-8",
    )

    config = load_cloud_model_config(config_path)

    assert config.runtime_settings.max_parallel_output_downloads == 3
    assert config.output.mode is CloudOutputMode.CSV_TABLE
    assert config.output.output_name == "aggregated_deaths_report"
    assert config.output.csv_value_column is None
    assert config.output.orientation is CSVTableOrientation.COLUMNS
    assert config.output.header_fields == ("t_lower", "t_upper", "count")


def test_load_cloud_model_config_rejects_missing_csv_table_output_name(
    tmp_path: Path,
):
    config_path = _write_cloud_config(tmp_path)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace(
        'filename = "population.csv"\n'
        'csv_value_column = "population"\n'
        'csv_value_type = "int"\n',
        'filename = "population.csv"\nmode = "csv_table"\n',
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="output_name"):
        load_cloud_model_config(config_path)


def test_load_cloud_model_config_rejects_unsupported_output_mode(
    tmp_path: Path,
):
    config_path = _write_cloud_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "[cloud.output]\n",
        '[cloud.output]\nmode = "json"\n',
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="cloud.output.mode"):
        load_cloud_model_config(config_path)


def test_load_cloud_model_config_parses_shared_assets_and_task_payload(
    tmp_path: Path,
):
    config_path = _write_cloud_config(tmp_path)
    text = config_path.read_text(encoding="utf-8")
    text += """

[cloud.shared_assets.reference]
source_path = "reference.json"
blob_dir = "assets"
remote_path_var = "REFERENCE_PATH"

[cloud.task_payload]
task_output_dir = "{output_mount_path}/special/{run_id}"

[[cloud.task_payload.transforms]]
name = "set-reference"
op = "set"
path = "/epimodel.GlobalParams/reference_path"
value = "{REFERENCE_PATH}"
on_missing = "create"
"""
    config_path.write_text(text, encoding="utf-8")

    config = load_cloud_model_config(config_path)

    assert len(config.shared_assets) == 1
    assert config.shared_assets[0].name == "reference"
    assert config.shared_assets[0].source_path == Path("reference.json")
    assert config.shared_assets[0].remote_path_var == "REFERENCE_PATH"
    assert config.task_payload.task_output_dir == (
        "{output_mount_path}/special/{run_id}"
    )
    assert config.task_payload.transforms[0].path == (
        "/epimodel.GlobalParams/reference_path"
    )
