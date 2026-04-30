from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_cloud_task_config_reads_defaults_from_image_path():
    task_config = REPO_ROOT / "example_model.mrp.task.toml"

    text = task_config.read_text()

    assert 'input = "/app/defaults.json"' in text


def test_cloud_image_uses_cmd_instead_of_entrypoint():
    dockerfile = REPO_ROOT / "packages" / "example_model" / "Dockerfile"

    text = dockerfile.read_text()

    assert 'CMD ["example_model"]' in text
    assert 'ENTRYPOINT ["example_model"]' not in text
