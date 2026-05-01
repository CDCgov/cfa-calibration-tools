from __future__ import annotations

from pathlib import Path

from calibrationtools.cloud.config import (
    CloudCSVValueType,
    load_cloud_model_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MODEL_ROOT = (
    REPO_ROOT / "packages" / "example_model" / "src" / "example_model"
)


def test_example_cloud_config_matches_docker_image_contract():
    """Validate the example cloud config points at image-baked assets."""
    config = load_cloud_model_config(
        EXAMPLE_MODEL_ROOT / "example_model.cloud_config.toml"
    )

    assert config.build_context == REPO_ROOT
    assert config.dockerfile == REPO_ROOT / "packages/example_model/Dockerfile"
    assert (
        config.simulation_mrp_config_path
        == EXAMPLE_MODEL_ROOT / "example_model.mrp.cloud.toml"
    )
    assert (
        config.runtime_settings.task_mrp_config_path
        == "/app/example_model.mrp.toml"
    )
    assert config.output.filename == "output.csv"
    assert config.output.csv_value_column == "population"
    assert config.output.csv_value_type is CloudCSVValueType.INT
    assert config.auto_size.probe == "mrp"
    assert config.auto_size.local_mrp_config_path == (
        EXAMPLE_MODEL_ROOT / "example_model.mrp.toml"
    )


def test_cloud_task_config_reuses_packaged_mrp_config():
    """Validate cloud tasks can use the normal packaged MRP config."""
    task_config = EXAMPLE_MODEL_ROOT / "example_model.mrp.toml"

    assert 'input = "defaults.json"' in task_config.read_text()


def test_cloud_image_uses_cmd_instead_of_entrypoint():
    """Validate the example image remains override-friendly for Batch tasks."""
    dockerfile = REPO_ROOT / "packages" / "example_model" / "Dockerfile"

    text = dockerfile.read_text()

    assert 'CMD ["example_model"]' in text
    assert 'ENTRYPOINT ["example_model"]' not in text
