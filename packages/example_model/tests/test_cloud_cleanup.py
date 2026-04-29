from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import example_model.cloud_cleanup as cloud_cleanup
import pytest
from example_model.cloud_cleanup import (
    discover_cleanup_listing,
    discover_cleanup_plan,
    execute_cleanup,
    parse_args,
)
from example_model.cloud_utils import (
    DEFAULT_CLOUD_RUNTIME_SETTINGS,
)

from calibrationtools.cloud.cleanup import CleanupPlan

TESTSHA_SESSION_SLUG = "20260412010101-testsha-ab12cd34ef56"
OTHERSHA_SESSION_SLUG = "20260412020202-othersha-deadbeefcafe"
TESTSHA_POOL = f"example-model-cloud-{TESTSHA_SESSION_SLUG}"
OTHERSHA_POOL = f"example-model-cloud-{OTHERSHA_SESSION_SLUG}"
TESTSHA_JOB = f"{TESTSHA_POOL}-j1"
OTHERSHA_JOB = f"{OTHERSHA_POOL}-j1"
TESTSHA_INPUT_CONTAINER = f"example-model-cloud-input-{TESTSHA_SESSION_SLUG}"
TESTSHA_OUTPUT_CONTAINER = f"example-model-cloud-output-{TESTSHA_SESSION_SLUG}"
TESTSHA_LOGS_CONTAINER = f"example-model-cloud-logs-{TESTSHA_SESSION_SLUG}"
OTHERSHA_INPUT_CONTAINER = f"example-model-cloud-input-{OTHERSHA_SESSION_SLUG}"
LOOKALIKE_POOL = f"example-model-cloudy-{TESTSHA_SESSION_SLUG}"
LOOKALIKE_JOB = f"other-project-cloud-{TESTSHA_SESSION_SLUG}-j1"
LOOKALIKE_INPUT_CONTAINER = (
    f"example-model-cloud-inputs-{TESTSHA_SESSION_SLUG}"
)


def _cloud_settings(**overrides):
    return replace(DEFAULT_CLOUD_RUNTIME_SETTINGS, **overrides)


def test_example_model_cloud_cleanup_keeps_shared_cleanup_models_private():
    assert not hasattr(cloud_cleanup, "CleanupPlan")
    assert not hasattr(cloud_cleanup, "CleanupListing")
    assert not hasattr(cloud_cleanup, "CleanupResult")


class _FakeClient:
    def __init__(self):
        self.cred = SimpleNamespace(
            azure_resource_group_name="rg-example",
            azure_batch_account="batch-example",
            azure_container_registry_account="registry-example",
            azure_user_assigned_identity=(
                "/subscriptions/test-sub/resourceGroups/test-rg/"
                "providers/Microsoft.ManagedIdentity/userAssignedIdentities/test-mi"
            ),
        )
        self.deleted_jobs: list[str] = []
        self.deleted_pools: list[str] = []
        self.deleted_containers: list[str] = []
        self.delete_job_hook: Callable[[str], None] | None = None
        self.delete_pool_hook: Callable[[str], None] | None = None
        self.delete_container_hook: Callable[[str], None] | None = None
        self.batch_mgmt_client = SimpleNamespace(
            pool=SimpleNamespace(
                list_by_batch_account=lambda **kwargs: [
                    SimpleNamespace(name=TESTSHA_POOL),
                    SimpleNamespace(name=OTHERSHA_POOL),
                    SimpleNamespace(name=LOOKALIKE_POOL),
                ]
            )
        )
        self.batch_service_client = SimpleNamespace(
            job=SimpleNamespace(
                list=lambda: [
                    SimpleNamespace(id=TESTSHA_JOB),
                    SimpleNamespace(id=OTHERSHA_JOB),
                    SimpleNamespace(id=LOOKALIKE_JOB),
                ]
            )
        )
        self.blob_service_client = SimpleNamespace(
            list_containers=lambda: [
                SimpleNamespace(name=TESTSHA_INPUT_CONTAINER),
                SimpleNamespace(name=TESTSHA_OUTPUT_CONTAINER),
                SimpleNamespace(name=TESTSHA_LOGS_CONTAINER),
                SimpleNamespace(name=OTHERSHA_INPUT_CONTAINER),
                SimpleNamespace(name=LOOKALIKE_INPUT_CONTAINER),
            ],
            delete_container=self._delete_container,
        )

    def delete_job(self, job_name: str) -> None:
        if self.delete_job_hook is not None:
            self.delete_job_hook(job_name)
            return
        self.deleted_jobs.append(job_name)

    def delete_pool(self, pool_name: str) -> None:
        if self.delete_pool_hook is not None:
            self.delete_pool_hook(pool_name)
            return
        self.deleted_pools.append(pool_name)

    def _delete_container(self, container_name: str) -> None:
        if self.delete_container_hook is not None:
            self.delete_container_hook(container_name)
            return
        self.deleted_containers.append(container_name)


def test_discover_cleanup_plan_only_matches_requested_session_resources(
    monkeypatch,
):
    fake_client = _FakeClient()
    monkeypatch.setattr(
        "example_model.cloud_cleanup.list_acr_repository_tags",
        lambda registry_name, repository_name, **kwargs: [
            "testsha",
            "othersha",
        ],
    )

    plan = discover_cleanup_plan(
        fake_client,
        _cloud_settings(),
        config_path=Path("example_model.mrp.cloud.toml"),
        session_slug=TESTSHA_SESSION_SLUG,
        image_tag="testsha",
        include_acr=True,
    )

    assert plan.session_slug == TESTSHA_SESSION_SLUG
    # Lookalike resources that share the slug suffix but come from a
    # different project prefix must be excluded from the cleanup plan.
    assert plan.pool_names == (TESTSHA_POOL,)
    assert plan.job_names == (TESTSHA_JOB,)
    assert plan.container_names == (
        TESTSHA_INPUT_CONTAINER,
        TESTSHA_LOGS_CONTAINER,
        TESTSHA_OUTPUT_CONTAINER,
    )
    assert LOOKALIKE_POOL not in plan.pool_names
    assert LOOKALIKE_JOB not in plan.job_names
    assert LOOKALIKE_INPUT_CONTAINER not in plan.container_names
    assert plan.acr_image_exists is True


def test_discover_cleanup_plan_skips_acr_lookup_without_image_tag(
    monkeypatch,
):
    fake_client = _FakeClient()
    monkeypatch.setattr(
        "example_model.cloud_cleanup.list_acr_repository_tags",
        lambda *args, **kwargs: pytest.fail(
            "cleanup plan without --image-tag must not query ACR"
        ),
    )

    plan = discover_cleanup_plan(
        fake_client,
        _cloud_settings(),
        config_path=Path("example_model.mrp.cloud.toml"),
        session_slug=TESTSHA_SESSION_SLUG,
        image_tag=None,
        include_acr=True,
    )

    assert plan.session_slug == TESTSHA_SESSION_SLUG
    assert plan.pool_names == (TESTSHA_POOL,)
    assert plan.job_names == (TESTSHA_JOB,)
    assert plan.container_names == (
        TESTSHA_INPUT_CONTAINER,
        TESTSHA_LOGS_CONTAINER,
        TESTSHA_OUTPUT_CONTAINER,
    )
    assert plan.acr_image_exists is False


def test_discover_cleanup_listing_lists_all_project_resources(monkeypatch):
    fake_client = _FakeClient()
    monkeypatch.setattr(
        "example_model.cloud_cleanup.list_acr_repository_tags",
        lambda registry_name, repository_name, **kwargs: [
            "testsha",
            "othersha",
        ],
    )

    listing = discover_cleanup_listing(
        fake_client,
        _cloud_settings(),
        config_path=Path("example_model.mrp.cloud.toml"),
        session_slug=None,
        image_tag=None,
        include_acr=True,
    )

    assert listing.session_slug is None
    assert listing.pool_names == (TESTSHA_POOL, OTHERSHA_POOL)
    assert listing.job_names == (TESTSHA_JOB, OTHERSHA_JOB)
    assert listing.container_names == (
        TESTSHA_INPUT_CONTAINER,
        OTHERSHA_INPUT_CONTAINER,
        TESTSHA_LOGS_CONTAINER,
        TESTSHA_OUTPUT_CONTAINER,
    )
    assert listing.acr_image_tags == ("othersha", "testsha")


def test_discover_cleanup_listing_filters_session_and_image(monkeypatch):
    fake_client = _FakeClient()
    monkeypatch.setattr(
        "example_model.cloud_cleanup.list_acr_repository_tags",
        lambda registry_name, repository_name, **kwargs: [
            "testsha",
            "othersha",
        ],
    )

    listing = discover_cleanup_listing(
        fake_client,
        _cloud_settings(),
        config_path=Path("example_model.mrp.cloud.toml"),
        session_slug=TESTSHA_SESSION_SLUG,
        image_tag="testsha",
        include_acr=True,
    )

    # Lookalike resources that share the slug suffix but come from a
    # different project prefix must be excluded from the session-scoped
    # listing.
    assert listing.pool_names == (TESTSHA_POOL,)
    assert listing.job_names == (TESTSHA_JOB,)
    assert listing.container_names == (
        TESTSHA_INPUT_CONTAINER,
        TESTSHA_LOGS_CONTAINER,
        TESTSHA_OUTPUT_CONTAINER,
    )
    assert LOOKALIKE_POOL not in listing.pool_names
    assert LOOKALIKE_JOB not in listing.job_names
    assert LOOKALIKE_INPUT_CONTAINER not in listing.container_names
    assert listing.acr_image_tags == ("testsha",)


def test_discover_cleanup_listing_does_not_derive_image_tag_from_session_slug(
    monkeypatch,
):
    fake_client = _FakeClient()
    monkeypatch.setattr(
        "example_model.cloud_cleanup.list_acr_repository_tags",
        lambda registry_name, repository_name, **kwargs: [
            "testsha",
            "othersha",
        ],
    )

    listing = discover_cleanup_listing(
        fake_client,
        _cloud_settings(),
        config_path=Path("example_model.mrp.cloud.toml"),
        session_slug=TESTSHA_SESSION_SLUG,
        image_tag=None,
        include_acr=True,
    )

    assert listing.image_tag is None
    assert listing.acr_image_tags == ("othersha", "testsha")


def test_parse_args_requires_session_slug_without_list():
    with pytest.raises(SystemExit, match="--session-slug is required"):
        parse_args([])

    args = parse_args(["--list"])
    assert args.list is True
    assert args.session_slug is None


def test_execute_cleanup_deletes_resources_in_safe_order(monkeypatch):
    fake_client = _FakeClient()
    operations: list[str] = []
    fake_client.delete_job_hook = lambda name: operations.append(f"job:{name}")
    fake_client.delete_pool_hook = lambda name: operations.append(
        f"pool:{name}"
    )
    fake_client.delete_container_hook = lambda name: operations.append(
        f"container:{name}"
    )
    monkeypatch.setattr(
        "example_model.cloud_cleanup.delete_acr_image_tag",
        lambda registry_name, repository_name, image_tag, **kwargs: (
            operations.append(
                f"acr:{registry_name}/{repository_name}:{image_tag}"
            )
        ),
    )

    plan = CleanupPlan(
        config_path=Path("example_model.mrp.cloud.toml"),
        session_slug=TESTSHA_SESSION_SLUG,
        keyvault="cfa-predict",
        registry_name="registry-example",
        repository_name="cfa-calibration-tools-example-model",
        image_tag="testsha",
        job_names=(TESTSHA_JOB,),
        pool_names=(TESTSHA_POOL,),
        container_names=(
            TESTSHA_INPUT_CONTAINER,
            TESTSHA_OUTPUT_CONTAINER,
            TESTSHA_LOGS_CONTAINER,
        ),
        acr_image_exists=True,
    )

    result = execute_cleanup(fake_client, plan, include_acr=True)

    assert operations == [
        f"job:{TESTSHA_JOB}",
        f"pool:{TESTSHA_POOL}",
        f"container:{TESTSHA_INPUT_CONTAINER}",
        f"container:{TESTSHA_OUTPUT_CONTAINER}",
        f"container:{TESTSHA_LOGS_CONTAINER}",
        "acr:registry-example/cfa-calibration-tools-example-model:testsha",
    ]
    assert result.ok


def test_execute_cleanup_requires_session_slug_for_batch_and_blob_resources():
    fake_client = _FakeClient()
    plan = CleanupPlan(
        config_path=Path("example_model.mrp.cloud.toml"),
        session_slug="",
        keyvault="cfa-predict",
        registry_name="registry-example",
        repository_name="cfa-calibration-tools-example-model",
        image_tag=None,
        job_names=(TESTSHA_JOB,),
        pool_names=(TESTSHA_POOL,),
        container_names=(TESTSHA_INPUT_CONTAINER,),
        acr_image_exists=False,
    )

    with pytest.raises(ValueError, match="requires a session_slug"):
        execute_cleanup(fake_client, plan, include_acr=False)


def test_delete_acr_image_tag_logs_in_with_identity_only_once(monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, capture_output, text):
        commands.append(list(command))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        cloud_cleanup,
        "_AZ_LOGGED_IN_IDENTITY",
        cloud_cleanup._AZ_NOT_LOGGED_IN,
    )
    monkeypatch.setattr(
        "example_model.cloud_cleanup.require_tool", lambda name: None
    )
    monkeypatch.setattr("example_model.cloud_cleanup.subprocess.run", fake_run)

    cloud_cleanup.delete_acr_image_tag(
        "registry-example",
        "repo-one",
        "tag-one",
    )
    cloud_cleanup.delete_acr_image_tag(
        "registry-example",
        "repo-two",
        "tag-two",
    )

    assert commands == [
        ["az", "login", "--identity"],
        [
            "az",
            "acr",
            "repository",
            "delete",
            "--name",
            "registry-example",
            "--image",
            "repo-one:tag-one",
            "--yes",
            "--output",
            "none",
        ],
        [
            "az",
            "acr",
            "repository",
            "delete",
            "--name",
            "registry-example",
            "--image",
            "repo-two:tag-two",
            "--yes",
            "--output",
            "none",
        ],
    ]


def test_list_then_delete_acr_uses_one_identity_login(monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, capture_output, text):
        commands.append(list(command))
        if command[:4] == ["az", "acr", "repository", "show-tags"]:
            return SimpleNamespace(
                returncode=0, stdout='["tag-one"]', stderr=""
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        cloud_cleanup,
        "_AZ_LOGGED_IN_IDENTITY",
        cloud_cleanup._AZ_NOT_LOGGED_IN,
    )
    monkeypatch.setattr(
        "example_model.cloud_cleanup.require_tool", lambda name: None
    )
    monkeypatch.setattr("example_model.cloud_cleanup.subprocess.run", fake_run)

    assert cloud_cleanup.list_acr_repository_tags(
        "registry-example",
        "repo-one",
    ) == ["tag-one"]
    cloud_cleanup.delete_acr_image_tag(
        "registry-example",
        "repo-one",
        "tag-one",
    )

    assert commands == [
        ["az", "login", "--identity"],
        [
            "az",
            "acr",
            "repository",
            "show-tags",
            "--name",
            "registry-example",
            "--repository",
            "repo-one",
            "--output",
            "json",
        ],
        [
            "az",
            "acr",
            "repository",
            "delete",
            "--name",
            "registry-example",
            "--image",
            "repo-one:tag-one",
            "--yes",
            "--output",
            "none",
        ],
    ]


def test_list_then_delete_acr_uses_default_identity_when_user_assigned_identity_is_configured(
    monkeypatch,
):
    commands: list[list[str]] = []

    def fake_run(command, capture_output, text):
        commands.append(list(command))
        if command[:4] == ["az", "acr", "repository", "show-tags"]:
            return SimpleNamespace(
                returncode=0, stdout='["tag-one"]', stderr=""
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    identity_resource_id = (
        "/subscriptions/test-sub/resourceGroups/test-rg/"
        "providers/Microsoft.ManagedIdentity/userAssignedIdentities/test-mi"
    )

    monkeypatch.setattr(
        cloud_cleanup,
        "_AZ_LOGGED_IN_IDENTITY",
        cloud_cleanup._AZ_NOT_LOGGED_IN,
    )
    monkeypatch.setattr(
        "example_model.cloud_cleanup.require_tool", lambda name: None
    )
    monkeypatch.setattr("example_model.cloud_cleanup.subprocess.run", fake_run)

    assert cloud_cleanup.list_acr_repository_tags(
        "registry-example",
        "repo-one",
        managed_identity_resource_id=identity_resource_id,
    ) == ["tag-one"]
    cloud_cleanup.delete_acr_image_tag(
        "registry-example",
        "repo-one",
        "tag-one",
        managed_identity_resource_id=identity_resource_id,
    )

    assert commands == [
        ["az", "login", "--identity"],
        [
            "az",
            "acr",
            "repository",
            "show-tags",
            "--name",
            "registry-example",
            "--repository",
            "repo-one",
            "--output",
            "json",
        ],
        [
            "az",
            "acr",
            "repository",
            "delete",
            "--name",
            "registry-example",
            "--image",
            "repo-one:tag-one",
            "--yes",
            "--output",
            "none",
        ],
    ]


def test_login_cache_reuses_default_identity_when_identity_parameter_changes(
    monkeypatch,
):
    commands: list[list[str]] = []

    def fake_run(command, capture_output, text):
        commands.append(list(command))
        if command[:4] == ["az", "acr", "repository", "show-tags"]:
            return SimpleNamespace(
                returncode=0, stdout='["tag-one"]', stderr=""
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    identity_resource_id = (
        "/subscriptions/test-sub/resourceGroups/test-rg/"
        "providers/Microsoft.ManagedIdentity/userAssignedIdentities/test-mi"
    )

    monkeypatch.setattr(
        cloud_cleanup,
        "_AZ_LOGGED_IN_IDENTITY",
        cloud_cleanup._AZ_NOT_LOGGED_IN,
    )
    monkeypatch.setattr(
        "example_model.cloud_cleanup.require_tool", lambda name: None
    )
    monkeypatch.setattr("example_model.cloud_cleanup.subprocess.run", fake_run)

    assert cloud_cleanup.list_acr_repository_tags(
        "registry-example",
        "repo-one",
    ) == ["tag-one"]
    assert cloud_cleanup.list_acr_repository_tags(
        "registry-example",
        "repo-one",
        managed_identity_resource_id=identity_resource_id,
    ) == ["tag-one"]

    assert commands == [
        ["az", "login", "--identity"],
        [
            "az",
            "acr",
            "repository",
            "show-tags",
            "--name",
            "registry-example",
            "--repository",
            "repo-one",
            "--output",
            "json",
        ],
        [
            "az",
            "acr",
            "repository",
            "show-tags",
            "--name",
            "registry-example",
            "--repository",
            "repo-one",
            "--output",
            "json",
        ],
    ]


def test_identity_not_found_raises_from_default_identity_login(monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, capture_output, text):
        commands.append(list(command))
        if command[:3] == ["az", "login", "--identity"]:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="ERROR: Identity not found",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    identity_resource_id = (
        "/subscriptions/test-sub/resourceGroups/test-rg/"
        "providers/Microsoft.ManagedIdentity/userAssignedIdentities/test-mi"
    )

    monkeypatch.setattr(
        cloud_cleanup,
        "_AZ_LOGGED_IN_IDENTITY",
        cloud_cleanup._AZ_NOT_LOGGED_IN,
    )
    monkeypatch.setattr(
        "example_model.cloud_cleanup.require_tool", lambda name: None
    )
    monkeypatch.setattr("example_model.cloud_cleanup.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="Identity not found"):
        cloud_cleanup.list_acr_repository_tags(
            "registry-example",
            "repo-one",
            managed_identity_resource_id=identity_resource_id,
        )

    assert commands == [["az", "login", "--identity"]]


def test_main_list_mode_is_read_only(monkeypatch, capsys):
    fake_client = _FakeClient()

    monkeypatch.setattr(
        "example_model.cloud_cleanup.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(),
    )
    monkeypatch.setattr(
        "example_model.cloud_cleanup.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_cleanup.list_acr_repository_tags",
        lambda registry_name, repository_name, **kwargs: [
            "testsha",
            "othersha",
        ],
    )
    monkeypatch.setattr(
        "example_model.cloud_cleanup.execute_cleanup",
        lambda *args, **kwargs: pytest.fail("list mode must not delete"),
    )

    exit_code = cloud_cleanup.main(["--list", "--yes"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Cloud cleanup inventory" in captured.out
    assert "batch jobs: 2" in captured.out
    assert "acr image tags: 2" in captured.out
    assert f"{TESTSHA_JOB} (session={TESTSHA_SESSION_SLUG})" in captured.out
    assert f"{OTHERSHA_POOL} (session={OTHERSHA_SESSION_SLUG})" in captured.out
    assert (
        "registry-example/cfa-calibration-tools-example-model:testsha "
        f"(session={TESTSHA_SESSION_SLUG})"
    ) in captured.out
    assert (
        "registry-example/cfa-calibration-tools-example-model:othersha "
        f"(session={OTHERSHA_SESSION_SLUG})"
    ) in captured.out


def test_main_list_mode_continues_when_acr_lookup_fails(monkeypatch, capsys):
    fake_client = _FakeClient()

    monkeypatch.setattr(
        "example_model.cloud_cleanup.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(),
    )
    monkeypatch.setattr(
        "example_model.cloud_cleanup.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_cleanup.list_acr_repository_tags",
        lambda registry_name, repository_name, **kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "Failed to authenticate Azure CLI with managed identity: "
                "ERROR: Identity not found"
            )
        ),
    )

    exit_code = cloud_cleanup.main(["--list"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Cloud cleanup inventory" in captured.out
    assert "batch jobs: 2" in captured.out
    assert "batch pools: 2" in captured.out
    assert "blob containers: 4" in captured.out
    assert "acr image tags: unavailable" in captured.out
    assert "Identity not found" in captured.out
    assert (
        "registry-example/cfa-calibration-tools-example-model:testsha"
        not in captured.out
    )
