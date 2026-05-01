from __future__ import annotations

from pathlib import Path

import pytest

from calibrationtools.cloud.config import (
    CloudCSVValueType,
    load_cloud_model_config,
)


def _write_cloud_config(tmp_path: Path, *, csv_type: str = "int") -> Path:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    simulation_config = tmp_path / "model.mrp.cloud.toml"
    simulation_config.write_text(
        '[runtime]\nspec = "inline"\n', encoding="utf-8"
    )
    config_path = tmp_path / "cloud_config.toml"
    config_path.write_text(
        f"""
[cloud]
keyvault = "kv"
simulation_mrp_config_path = "model.mrp.cloud.toml"
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
    assert config.simulation_mrp_config_path == (
        tmp_path / "model.mrp.cloud.toml"
    )
    assert config.output.filename == "population.csv"
    assert config.output.csv_value_column == "population"
    assert config.output.csv_value_type is CloudCSVValueType.INT
    assert config.auto_size.probe == "mrp"
    assert (
        config.auto_size.local_mrp_config_path == tmp_path / "model.mrp.toml"
    )


def test_load_cloud_model_config_accepts_legacy_runtime_cloud(
    tmp_path: Path,
):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
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
task_slots_per_node = 4
""",
        encoding="utf-8",
    )

    config = load_cloud_model_config(
        config_path,
        default_build_context=tmp_path,
        default_dockerfile=dockerfile,
    )

    assert config.build_context == tmp_path
    assert config.dockerfile == dockerfile
    assert config.runtime_settings.task_slots_per_node == 4
    assert config.simulation_mrp_config_path == config_path
    assert config.output.filename == "output.csv"
    assert config.output.csv_value_column == "population"
    assert config.output.csv_value_type is CloudCSVValueType.INT


def test_load_cloud_model_config_accepts_legacy_runtime_cloud_without_build_defaults(
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

    config = load_cloud_model_config(config_path)

    assert config.build_context == tmp_path
    assert config.dockerfile == tmp_path / "Dockerfile"
    assert config.runtime_settings.repository == "remote-model"
    assert config.output.filename == "output.csv"


def test_load_cloud_model_config_does_not_require_simulation_mrp_file(
    tmp_path: Path,
):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    config_path = tmp_path / "cloud_config.toml"
    config_path.write_text(
        """
[cloud]
keyvault = "kv"
simulation_mrp_config_path = "missing.mrp.cloud.toml"

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

    config = load_cloud_model_config(config_path)

    assert config.simulation_mrp_config_path == (
        tmp_path / "missing.mrp.cloud.toml"
    )


def test_load_cloud_model_config_rejects_unsupported_csv_value_type(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="csv_value_type"):
        load_cloud_model_config(_write_cloud_config(tmp_path, csv_type="json"))
