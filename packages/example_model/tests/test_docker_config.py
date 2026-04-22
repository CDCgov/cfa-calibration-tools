from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_docker_config_uses_calling_user_identity():
    config_path = REPO_ROOT / "example_model.mrp.docker.toml"

    text = config_path.read_text()

    assert 'command = "sh"' in text
    assert "$(id -u):$(id -g)" in text
    assert "1000:1000" not in text
