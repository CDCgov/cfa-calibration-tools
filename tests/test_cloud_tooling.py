from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from calibrationtools.cloud.tooling import (
    create_cloud_client,
    ensure_az_login_with_identity,
)


def test_ensure_az_login_uses_plain_managed_identity_login():
    sentinel = object()
    commands: list[list[str]] = []

    def run(command, *, capture_output, text):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    identity = ensure_az_login_with_identity(
        managed_identity_resource_id="/subscriptions/test/identity",
        current_identity=sentinel,
        not_logged_in_sentinel=sentinel,
        require_tool_func=lambda name: None,
        subprocess_run=run,
    )

    assert commands == [["az", "login", "--identity"]]
    assert identity == "/subscriptions/test/identity"

    ensure_az_login_with_identity(
        managed_identity_resource_id="/subscriptions/test/other-identity",
        current_identity=identity,
        not_logged_in_sentinel=sentinel,
        require_tool_func=lambda name: None,
        subprocess_run=run,
    )

    assert commands == [["az", "login", "--identity"]]


def test_create_cloud_client_logs_in_before_constructing_client(monkeypatch):
    events: list[tuple[str, str | None]] = []
    cfa = ModuleType("cfa")
    cfa.__path__ = []
    cloudops = ModuleType("cfa.cloudops")

    class CloudClient:
        def __init__(self, *, keyvault: str):
            events.append(("client", keyvault))

    setattr(cloudops, "CloudClient", CloudClient)
    monkeypatch.setitem(sys.modules, "cfa", cfa)
    monkeypatch.setitem(sys.modules, "cfa.cloudops", cloudops)

    client = create_cloud_client(
        keyvault="kv",
        ensure_az_login_with_identity_func=lambda: events.append(
            ("login", None)
        ),
    )

    assert isinstance(client, CloudClient)
    assert events == [("login", None), ("client", "kv")]
