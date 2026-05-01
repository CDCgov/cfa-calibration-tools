from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from calibrationtools.cloud.cleanup import parse_args


def test_parse_args_accepts_cloud_config_alias(tmp_path: Path):
    config_path = tmp_path / "cloud_config.toml"

    args = parse_args(
        ["--cloud-config", str(config_path), "--list"],
        default_config_path=Path("default.toml"),
    )

    assert args.config == config_path


def test_cleanup_main_uses_model_facing_config(monkeypatch, tmp_path: Path):
    import calibrationtools.cloud.cleanup as cleanup

    config_path = tmp_path / "cloud_config.toml"
    settings = SimpleNamespace(keyvault="kv")
    client = object()
    listing = SimpleNamespace()
    captured = {}

    monkeypatch.setattr(
        cleanup,
        "load_cloud_model_config",
        lambda path: SimpleNamespace(runtime_settings=settings),
    )

    def fake_create_cloud_client(*, keyvault):
        captured["keyvault"] = keyvault
        return client

    monkeypatch.setattr(
        cleanup, "create_cloud_client", fake_create_cloud_client
    )

    def fake_listing(client_arg, settings_arg, **kwargs):
        captured["client"] = client_arg
        captured["settings"] = settings_arg
        captured.update(kwargs)
        return listing

    monkeypatch.setattr(cleanup, "discover_cleanup_listing", fake_listing)
    monkeypatch.setattr(
        cleanup,
        "format_cleanup_listing",
        lambda listing_arg, *, include_acr: "listing",
    )

    assert cleanup.main(["--cloud-config", str(config_path), "--list"]) == 0
    assert captured["keyvault"] == "kv"
    assert captured["client"] is client
    assert captured["settings"] is settings
    assert captured["config_path"] == config_path
    assert captured["include_acr"] is True
    assert captured["allow_acr_errors"] is True
