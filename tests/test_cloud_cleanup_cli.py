from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from calibrationtools.cloud.cleanup import (
    CleanupPlan,
    CleanupResult,
    parse_args,
)


def test_parse_args_accepts_cloud_config_alias(tmp_path: Path):
    config_path = tmp_path / "cloud_config.toml"

    args = parse_args(
        ["--cloud-config", str(config_path), "--list"],
        default_config_path=Path("default.toml"),
    )

    assert args.config == config_path


def test_parse_args_accepts_session_id_for_cleanup():
    args = parse_args(
        ["--session-id", "session"],
        default_config_path=Path("default.toml"),
    )

    assert args.session_id == "session"


def test_parse_args_rejects_old_session_slug_flag():
    with pytest.raises(SystemExit):
        parse_args(
            ["--session-slug", "session"],
            default_config_path=Path("default.toml"),
        )


def test_parse_args_rejects_delete_without_selector():
    with pytest.raises(SystemExit, match="--session-id"):
        parse_args([], default_config_path=Path("default.toml"))


def test_parse_args_rejects_user_delete_without_bulk_flag():
    with pytest.raises(SystemExit, match="--all-sessions-for-user"):
        parse_args(
            ["--user", "alice"], default_config_path=Path("default.toml")
        )


def test_parse_args_requires_user_for_bulk_delete():
    with pytest.raises(SystemExit, match="requires --user"):
        parse_args(
            ["--all-sessions-for-user"],
            default_config_path=Path("default.toml"),
        )


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
    assert captured["user"] is None
    assert captured["include_acr"] is True
    assert captured["allow_acr_errors"] is True


def test_cleanup_main_deletes_by_default(monkeypatch, tmp_path: Path):
    import calibrationtools.cloud.cleanup as cleanup

    config_path = tmp_path / "cloud_config.toml"
    settings = SimpleNamespace(keyvault="kv")
    client = object()
    plan = CleanupPlan(
        config_path=config_path,
        session_id="session",
        keyvault="kv",
        registry_name=None,
        repository_name="repo",
        image_tag=None,
        job_names=("job",),
        pool_names=(),
        container_names=(),
        acr_image_exists=False,
    )
    captured = {}

    monkeypatch.setattr(
        cleanup,
        "load_cloud_model_config",
        lambda path: SimpleNamespace(runtime_settings=settings),
    )
    monkeypatch.setattr(
        cleanup,
        "create_cloud_client",
        lambda *, keyvault: client,
    )
    monkeypatch.setattr(cleanup, "discover_cleanup_plan", lambda *a, **k: plan)
    monkeypatch.setattr(
        cleanup,
        "format_cleanup_plan",
        lambda plan_arg, *, include_acr: "plan",
    )

    def fake_execute(client_arg, plan_arg, *, include_acr):
        captured["client"] = client_arg
        captured["plan"] = plan_arg
        return CleanupResult(deleted=("job:job",), failures=())

    monkeypatch.setattr(cleanup, "execute_cleanup", fake_execute)

    assert (
        cleanup.main(
            ["--cloud-config", str(config_path), "--session-id", "session"]
        )
        == 0
    )
    assert captured["client"] is client
    assert captured["plan"] is plan


def test_cleanup_main_dry_run_skips_delete(monkeypatch, tmp_path: Path):
    import calibrationtools.cloud.cleanup as cleanup

    config_path = tmp_path / "cloud_config.toml"
    settings = SimpleNamespace(keyvault="kv")
    plan = CleanupPlan(
        config_path=config_path,
        session_id="session",
        keyvault="kv",
        registry_name=None,
        repository_name="repo",
        image_tag=None,
        job_names=("job",),
        pool_names=(),
        container_names=(),
        acr_image_exists=False,
    )

    monkeypatch.setattr(
        cleanup,
        "load_cloud_model_config",
        lambda path: SimpleNamespace(runtime_settings=settings),
    )
    monkeypatch.setattr(
        cleanup, "create_cloud_client", lambda *, keyvault: object()
    )
    monkeypatch.setattr(cleanup, "discover_cleanup_plan", lambda *a, **k: plan)
    monkeypatch.setattr(
        cleanup,
        "format_cleanup_plan",
        lambda plan_arg, *, include_acr: "plan",
    )
    monkeypatch.setattr(
        cleanup,
        "execute_cleanup",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("delete should not run")
        ),
    )

    assert (
        cleanup.main(
            [
                "--cloud-config",
                str(config_path),
                "--session-id",
                "session",
                "--dry-run",
            ]
        )
        == 0
    )
