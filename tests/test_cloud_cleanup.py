from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from calibrationtools.cloud.config import CloudRuntimeSettings
from calibrationtools.cloud.cleanup import (
    discover_cleanup_listing,
    discover_cleanup_plans_for_user,
    execute_cleanup_plans,
)


SESSION_ID = "20260101000000-alice-abc-123456abcdef"
OTHER_SESSION_ID = "20260101000000-bob-def-fedcba654321"
ALICE_SECOND_SESSION_ID = "20260102000000-alice-def-fedcba654321"


def make_settings() -> CloudRuntimeSettings:
    return CloudRuntimeSettings(
        keyvault="kv",
        local_image="local",
        repository="repo",
        task_mrp_config_path="/app/task.toml",
        pool_prefix="pool",
        job_prefix="job",
        input_container_prefix="input",
        output_container_prefix="output",
        logs_container_prefix="logs",
    )


def make_client(
    *,
    registry_name: str | None = "acr",
    pool_names: list[str] | None = None,
    job_names: list[str] | None = None,
    container_names: list[str] | None = None,
) -> SimpleNamespace:
    if pool_names is None:
        pool_names = [
            f"pool-{SESSION_ID}",
            f"pool-{OTHER_SESSION_ID}",
            f"pool-{ALICE_SECOND_SESSION_ID}",
            f"other-{SESSION_ID}",
        ]
    if job_names is None:
        job_names = [
            f"job-{SESSION_ID}-j1",
            f"job-{OTHER_SESSION_ID}-j1",
            f"job-{ALICE_SECOND_SESSION_ID}-j1",
            f"other-{SESSION_ID}-j1",
        ]
    if container_names is None:
        container_names = [
            f"input-{SESSION_ID}",
            f"output-{SESSION_ID}",
            f"logs-{SESSION_ID}",
            f"input-{OTHER_SESSION_ID}",
            f"input-{ALICE_SECOND_SESSION_ID}",
            f"other-{SESSION_ID}",
        ]

    return SimpleNamespace(
        cred=SimpleNamespace(
            azure_resource_group_name="rg",
            azure_batch_account="batch",
            azure_container_registry_account=registry_name,
            azure_user_assigned_identity="identity",
        ),
        batch_mgmt_client=SimpleNamespace(
            pool=SimpleNamespace(
                list_by_batch_account=lambda **kwargs: [
                    SimpleNamespace(name=name) for name in pool_names
                ]
            )
        ),
        batch_service_client=SimpleNamespace(
            job=SimpleNamespace(
                list=lambda: [SimpleNamespace(id=name) for name in job_names]
            )
        ),
        blob_service_client=SimpleNamespace(
            list_containers=lambda: [
                SimpleNamespace(name=name) for name in container_names
            ]
        ),
    )


def list_tags(*tags: str):
    return lambda *args, **kwargs: list(tags)


def discover(
    client: Any,
    *,
    config_path: Path = Path("cloud.toml"),
    session_id: str | None = None,
    user: str | None = None,
    image_tag: str | None = None,
    include_acr: bool = True,
    allow_acr_errors: bool = False,
    list_acr_repository_tags_fn: Callable[..., list[str]] | None = None,
):
    if list_acr_repository_tags_fn is None:
        list_acr_repository_tags_fn = list_tags("tag1", "tag2")
    return discover_cleanup_listing(
        client,
        make_settings(),
        config_path=config_path,
        session_id=session_id,
        user=user,
        image_tag=image_tag,
        include_acr=include_acr,
        allow_acr_errors=allow_acr_errors,
        list_acr_repository_tags_fn=list_acr_repository_tags_fn,
    )


def test_cleanup_listing_all_sessions_uses_project_prefixes():
    listing = discover(make_client())

    assert listing.pool_names == (
        f"pool-{SESSION_ID}",
        f"pool-{OTHER_SESSION_ID}",
        f"pool-{ALICE_SECOND_SESSION_ID}",
    )
    assert listing.job_names == (
        f"job-{SESSION_ID}-j1",
        f"job-{OTHER_SESSION_ID}-j1",
        f"job-{ALICE_SECOND_SESSION_ID}-j1",
    )
    assert listing.container_names == (
        f"input-{SESSION_ID}",
        f"input-{OTHER_SESSION_ID}",
        f"input-{ALICE_SECOND_SESSION_ID}",
        f"logs-{SESSION_ID}",
        f"output-{SESSION_ID}",
    )


def test_cleanup_listing_one_session_uses_session_id_filters():
    listing = discover(make_client(), session_id=SESSION_ID)

    assert listing.pool_names == (f"pool-{SESSION_ID}",)
    assert listing.job_names == (f"job-{SESSION_ID}-j1",)
    assert listing.container_names == (
        f"input-{SESSION_ID}",
        f"logs-{SESSION_ID}",
        f"output-{SESSION_ID}",
    )


def test_cleanup_listing_user_filter_matches_session_id_username():
    listing = discover(make_client(), user="alice")

    assert listing.pool_names == (
        f"pool-{SESSION_ID}",
        f"pool-{ALICE_SECOND_SESSION_ID}",
    )
    assert listing.job_names == (
        f"job-{SESSION_ID}-j1",
        f"job-{ALICE_SECOND_SESSION_ID}-j1",
    )
    assert listing.container_names == (
        f"input-{SESSION_ID}",
        f"input-{ALICE_SECOND_SESSION_ID}",
        f"logs-{SESSION_ID}",
        f"output-{SESSION_ID}",
    )


def test_user_cleanup_plans_delete_only_requested_users_sessions():
    calls: list[tuple[str, str]] = []
    client = make_client()
    client.delete_job = lambda name: calls.append(("job", name))
    client.delete_pool = lambda name: calls.append(("pool", name))
    client.blob_service_client.delete_container = lambda name: calls.append(
        ("container", name)
    )

    plans = discover_cleanup_plans_for_user(
        client,
        make_settings(),
        config_path=Path("cloud.toml"),
        user="alice",
        image_tag=None,
        include_acr=True,
    )

    assert [plan.session_id for plan in plans] == [
        SESSION_ID,
        ALICE_SECOND_SESSION_ID,
    ]

    result = execute_cleanup_plans(client, plans, include_acr=True)

    assert result.failures == ()
    assert ("job", f"job-{SESSION_ID}-j1") in calls
    assert ("job", f"job-{ALICE_SECOND_SESSION_ID}-j1") in calls
    assert ("job", f"job-{OTHER_SESSION_ID}-j1") not in calls
    assert ("pool", f"pool-{OTHER_SESSION_ID}") not in calls
    assert ("container", f"input-{OTHER_SESSION_ID}") not in calls


def test_cleanup_listing_include_acr_false_skips_registry_lookup():
    def fail_lookup(*args, **kwargs):
        raise AssertionError("ACR lookup should not be used")

    listing = discover(
        make_client(),
        include_acr=False,
        list_acr_repository_tags_fn=fail_lookup,
    )

    assert listing.acr_image_tags == ()
    assert listing.acr_image_tags_warning is None


def test_cleanup_listing_missing_registry_strict_raises():
    with pytest.raises(SystemExit, match="AZURE_CONTAINER_REGISTRY_ACCOUNT"):
        discover(make_client(registry_name=None))


def test_cleanup_listing_missing_registry_allowed_returns_warning():
    listing = discover(
        make_client(registry_name=None),
        allow_acr_errors=True,
    )

    assert listing.acr_image_tags == ()
    assert (
        listing.acr_image_tags_warning
        == "AZURE_CONTAINER_REGISTRY_ACCOUNT must be configured."
    )


def test_cleanup_listing_image_tag_filters_exactly():
    listing = discover(make_client(), image_tag="tag2")

    assert listing.acr_image_tags == ("tag2",)


def test_cleanup_listing_registry_lookup_failure_strict_propagates():
    def fail_lookup(*args, **kwargs):
        raise RuntimeError("registry unavailable")

    with pytest.raises(RuntimeError, match="registry unavailable"):
        discover(make_client(), list_acr_repository_tags_fn=fail_lookup)


def test_cleanup_listing_registry_lookup_failure_allowed_warns():
    def fail_lookup(*args, **kwargs):
        raise RuntimeError("registry unavailable")

    listing = discover(
        make_client(),
        allow_acr_errors=True,
        list_acr_repository_tags_fn=fail_lookup,
    )

    assert listing.acr_image_tags == ()
    assert listing.acr_image_tags_warning == "registry unavailable"
