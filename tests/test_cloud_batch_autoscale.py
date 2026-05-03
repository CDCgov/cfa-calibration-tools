from __future__ import annotations

import sys
from datetime import timedelta
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from calibrationtools.cloud.batch import (
    add_batch_task_with_short_id,
    build_pool_autoscale_formula,
    create_pool_with_blob_mounts,
)


class FakePoolClient:
    def __init__(self) -> None:
        self.created: dict[str, Any] = {}

    def create(self, **kwargs):
        self.created.update(kwargs)


class FakeScaleSettings:
    def __init__(self, *, auto_scale=None, fixed_scale=None):
        self.auto_scale = auto_scale
        self.fixed_scale = fixed_scale


class FakeAutoScaleSettings:
    def __init__(self, *, formula, evaluation_interval=None):
        self.formula = formula
        self.evaluation_interval = evaluation_interval


class FakeContainerConfiguration:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.container_image_names = kwargs.get("container_image_names", [])


class FakeContainerRegistry:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeComputeNodeIdentityReference:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeAzureBlobFileSystemConfiguration:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.blobfuse_options = kwargs.get("blobfuse_options")


class FakeMountConfiguration:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.azure_blob_file_system_configuration = kwargs[
            "azure_blob_file_system_configuration"
        ]


class FakeNodePlacementConfiguration:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.policy = kwargs["policy"]


class FakeTaskConstraints:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.max_wall_clock_time = kwargs.get("max_wall_clock_time")


class FakeTaskContainerSettings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.image_name = kwargs["image_name"]
        self.container_run_options = kwargs["container_run_options"]


class FakeTaskAddParameter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.id = kwargs["id"]
        self.command_line = kwargs["command_line"]
        self.container_settings = kwargs["container_settings"]


class FakeUserIdentity:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeAutoUserSpecification:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def install_fake_azure_batch_modules(monkeypatch) -> None:
    fake_mgmt_models = SimpleNamespace(
        AutoScaleSettings=FakeAutoScaleSettings,
        AzureBlobFileSystemConfiguration=FakeAzureBlobFileSystemConfiguration,
        ComputeNodeIdentityReference=FakeComputeNodeIdentityReference,
        ContainerConfiguration=FakeContainerConfiguration,
        ContainerRegistry=FakeContainerRegistry,
        MountConfiguration=FakeMountConfiguration,
        NodePlacementConfiguration=FakeNodePlacementConfiguration,
        NodePlacementPolicyType=SimpleNamespace(
            regional="regional",
            zonal="zonal",
        ),
        ScaleSettings=FakeScaleSettings,
    )
    fake_task_models = SimpleNamespace(
        AutoUserScope=SimpleNamespace(pool="pool"),
        AutoUserSpecification=FakeAutoUserSpecification,
        ElevationLevel=SimpleNamespace(admin="admin"),
        TaskAddParameter=FakeTaskAddParameter,
        TaskConstraints=FakeTaskConstraints,
        TaskContainerSettings=FakeTaskContainerSettings,
        UserIdentity=FakeUserIdentity,
    )
    fake_batch_module = ModuleType("azure.batch")
    setattr(fake_batch_module, "models", fake_task_models)
    fake_mgmt_batch_module = ModuleType("azure.mgmt.batch")
    setattr(fake_mgmt_batch_module, "models", fake_mgmt_models)
    fake_mgmt_module = ModuleType("azure.mgmt")
    setattr(fake_mgmt_module, "batch", fake_mgmt_batch_module)
    fake_azure_module = ModuleType("azure")
    setattr(fake_azure_module, "batch", fake_batch_module)
    setattr(fake_azure_module, "mgmt", fake_mgmt_module)

    monkeypatch.setitem(sys.modules, "azure", fake_azure_module)
    monkeypatch.setitem(sys.modules, "azure.batch", fake_batch_module)
    monkeypatch.setitem(sys.modules, "azure.mgmt", fake_mgmt_module)
    monkeypatch.setitem(
        sys.modules,
        "azure.mgmt.batch",
        fake_mgmt_batch_module,
    )


def install_fake_cloudops_modules(
    monkeypatch,
    *,
    fail_pool_lookup: bool = False,
    fail_log_mount_lookup: bool = False,
    fail_pool_mount_lookup: bool = False,
) -> list[str]:
    calls: list[str] = []
    fake_defaults = _fake_cloudops_defaults()
    fake_helpers = _fake_cloudops_helpers()
    fake_batch_helpers = _fake_cloudops_batch_helpers(
        calls,
        fail_pool_lookup=fail_pool_lookup,
        fail_log_mount_lookup=fail_log_mount_lookup,
        fail_pool_mount_lookup=fail_pool_mount_lookup,
    )
    _register_fake_cloudops_modules(
        monkeypatch,
        fake_defaults=fake_defaults,
        fake_helpers=fake_helpers,
        fake_batch_helpers=fake_batch_helpers,
    )
    return calls


def _fake_cloudops_defaults() -> ModuleType:
    def fake_default_pool_config(**kwargs):
        return SimpleNamespace(
            kwargs=kwargs,
            deployment_configuration=SimpleNamespace(
                virtual_machine_configuration=SimpleNamespace()
            ),
        )

    fake_defaults = ModuleType("cfa.cloudops.defaults")
    setattr(fake_defaults, "get_default_pool_config", fake_default_pool_config)
    setattr(
        fake_defaults,
        "assign_container_config",
        lambda pool_config, container_config: setattr(
            pool_config,
            "container_config",
            container_config,
        ),
    )
    return fake_defaults


def _fake_cloudops_helpers() -> ModuleType:
    fake_helpers = ModuleType("cfa.cloudops.helpers")
    setattr(
        fake_helpers,
        "format_rel_path",
        lambda value=None, *, rel_path=None: str(
            rel_path if rel_path is not None else value
        ).strip("/"),
    )
    return fake_helpers


def _fake_cloudops_batch_helpers(
    calls: list[str],
    *,
    fail_pool_lookup: bool,
    fail_log_mount_lookup: bool,
    fail_pool_mount_lookup: bool,
) -> ModuleType:
    fake_batch_helpers = ModuleType("cfa.cloudops.batch_helpers")

    def fake_get_pool_full_info(*args):
        calls.append("get_pool_full_info")
        if fail_pool_lookup:
            raise AssertionError("pool image lookup should not be used")
        container_configuration = SimpleNamespace(
            container_image_names=["docker://pool/image:latest"]
        )
        vm_config = SimpleNamespace(
            container_configuration=container_configuration
        )
        return SimpleNamespace(
            deployment_configuration=SimpleNamespace(
                virtual_machine_configuration=vm_config
            )
        )

    def fake_get_rel_mnt_path(**kwargs):
        calls.append("get_rel_mnt_path")
        if fail_log_mount_lookup:
            raise AssertionError("log mount lookup should not be used")
        return "pool/logs"

    def fake_get_pool_mounts(*args):
        calls.append("get_pool_mounts")
        if fail_pool_mount_lookup:
            raise AssertionError("pool mounts should not be queried")
        return [{"source": "pool-input", "target": "pool-input"}]

    def fake_generate_mount_string(mounts):
        if not mounts:
            return ""
        return " ".join(
            f"--mount {mount['source']}:{mount['target']}" for mount in mounts
        )

    def fake_generate_command_for_saving_logs(**kwargs):
        return (
            f"SAVE({kwargs['save_logs_rel_path']},"
            f"{kwargs['logs_folder']})::{kwargs['command_line']}"
        )

    setattr(fake_batch_helpers, "get_vm_size", lambda value: value)
    setattr(fake_batch_helpers, "get_pool_full_info", fake_get_pool_full_info)
    setattr(fake_batch_helpers, "get_rel_mnt_path", fake_get_rel_mnt_path)
    setattr(fake_batch_helpers, "get_pool_mounts", fake_get_pool_mounts)
    setattr(
        fake_batch_helpers,
        "_generate_mount_string",
        fake_generate_mount_string,
    )
    setattr(fake_batch_helpers, "_generate_exit_conditions", lambda value: {})
    setattr(
        fake_batch_helpers,
        "_generate_command_for_saving_logs",
        fake_generate_command_for_saving_logs,
    )
    return fake_batch_helpers


def _register_fake_cloudops_modules(
    monkeypatch,
    *,
    fake_defaults: ModuleType,
    fake_helpers: ModuleType,
    fake_batch_helpers: ModuleType,
) -> None:
    fake_cloudops_module = ModuleType("cfa.cloudops")
    setattr(fake_cloudops_module, "defaults", fake_defaults)
    setattr(fake_cloudops_module, "helpers", fake_helpers)
    setattr(fake_cloudops_module, "batch_helpers", fake_batch_helpers)
    fake_cfa_module = ModuleType("cfa")
    setattr(fake_cfa_module, "cloudops", fake_cloudops_module)

    monkeypatch.setitem(sys.modules, "cfa", fake_cfa_module)
    monkeypatch.setitem(sys.modules, "cfa.cloudops", fake_cloudops_module)
    monkeypatch.setitem(sys.modules, "cfa.cloudops.defaults", fake_defaults)
    monkeypatch.setitem(sys.modules, "cfa.cloudops.helpers", fake_helpers)
    monkeypatch.setitem(
        sys.modules,
        "cfa.cloudops.batch_helpers",
        fake_batch_helpers,
    )


def make_pool_mount_client():
    pool_client = FakePoolClient()
    added: dict[str, Any] = {}

    class FakeJobClient:
        def get(self, job_name):
            return SimpleNamespace(
                as_dict=lambda: {"execution_info": {"pool_id": "pool"}}
            )

    class FakeTaskClient:
        def add(self, *, job_id, task):
            added["job_id"] = job_id
            added["task"] = task

    client = SimpleNamespace(
        cred=SimpleNamespace(
            azure_resource_group_name="rg",
            azure_batch_account="batch",
            azure_subnet_id="subnet",
            azure_user_assigned_identity="identity",
            azure_blob_storage_account="blob",
            azure_container_registry_account="acr",
        ),
        batch_mgmt_client=SimpleNamespace(pool=pool_client),
        batch_service_client=SimpleNamespace(
            job=FakeJobClient(),
            task=FakeTaskClient(),
        ),
    )
    return client, pool_client.created, added


def create_test_pool(client: Any, **overrides: Any) -> None:
    kwargs: dict[str, Any] = {
        "client": client,
        "pool_name": "pool",
        "container_image_name": "acr.azurecr.io/model:test",
        "mounts": [{"source": "input", "target": "cloud-input"}],
        "vm_size": "large",
        "target_dedicated_nodes": 4,
    }
    kwargs.update(overrides)
    create_pool_with_blob_mounts(**kwargs)


def add_test_task(client: Any, **overrides: Any) -> str:
    kwargs: dict[str, Any] = {
        "client": client,
        "job_name": "job",
        "command_line": "do-work",
        "task_name_suffix": "run",
        "timeout": 5,
        "mount_pairs": [{"source": "output", "target": "/cloud-output"}],
        "container_image_name": "explicit/image:tag",
    }
    kwargs.update(overrides)
    return add_batch_task_with_short_id(**kwargs)


def test_build_pool_autoscale_formula_scales_dedicated_nodes_to_zero():
    formula = build_pool_autoscale_formula(
        max_dedicated_nodes=3,
        task_slots_per_node=50,
    )

    assert "maxNodes = 3;" in formula
    assert "taskSlotsPerNode = 50;" in formula
    assert "$PendingTasks" in formula
    assert "targetVMs = ceil(pendingTasks / taskSlotsPerNode);" in formula
    assert (
        "$TargetDedicatedNodes = min(maxNodes, max(0, targetVMs));" in formula
    )
    assert "$TargetLowPriorityNodes = 0;" in formula
    assert "$NodeDeallocationOption = taskcompletion;" in formula


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_dedicated_nodes": 0, "task_slots_per_node": 1},
        {"max_dedicated_nodes": 1, "task_slots_per_node": 0},
    ],
)
def test_build_pool_autoscale_formula_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError, match="must be at least 1"):
        build_pool_autoscale_formula(**kwargs)


def test_create_pool_with_blob_mounts_uses_autoscale_settings(monkeypatch):
    install_fake_azure_batch_modules(monkeypatch)
    install_fake_cloudops_modules(monkeypatch)
    client, created, _ = make_pool_mount_client()

    create_test_pool(
        client,
        task_slots_per_node=25,
        auto_scale_evaluation_interval_minutes=10,
    )

    pool_config = created["parameters"]
    scale_settings = pool_config.scale_settings
    assert scale_settings.fixed_scale is None
    assert scale_settings.auto_scale is not None
    assert scale_settings.auto_scale.evaluation_interval == timedelta(
        minutes=10
    )
    assert "maxNodes = 4;" in scale_settings.auto_scale.formula
    assert "taskSlotsPerNode = 25;" in scale_settings.auto_scale.formula


@pytest.mark.parametrize(
    ("credential_field", "message"),
    [
        ("azure_resource_group_name", "AZURE_RESOURCE_GROUP_NAME"),
        ("azure_batch_account", "AZURE_BATCH_ACCOUNT"),
        ("azure_subnet_id", "AZURE_SUBNET_ID"),
        ("azure_user_assigned_identity", "AZURE_USER_ASSIGNED_IDENTITY"),
        ("azure_blob_storage_account", "AZURE_BLOB_STORAGE_ACCOUNT"),
        (
            "azure_container_registry_account",
            "AZURE_CONTAINER_REGISTRY_ACCOUNT",
        ),
    ],
)
def test_create_pool_with_blob_mounts_requires_credentials(
    monkeypatch,
    credential_field,
    message,
):
    install_fake_azure_batch_modules(monkeypatch)
    install_fake_cloudops_modules(monkeypatch)
    client, _, _ = make_pool_mount_client()
    setattr(client.cred, credential_field, "")

    with pytest.raises(ValueError, match=message):
        create_test_pool(client)


@pytest.mark.parametrize(
    ("availability_zones", "expected_policy"),
    [
        ("regional", "regional"),
        ("zonal", "zonal"),
    ],
)
def test_create_pool_with_blob_mounts_sets_node_placement(
    monkeypatch,
    availability_zones,
    expected_policy,
):
    install_fake_azure_batch_modules(monkeypatch)
    install_fake_cloudops_modules(monkeypatch)
    client, created, _ = make_pool_mount_client()

    create_test_pool(client, availability_zones=availability_zones)

    vm_config = created[
        "parameters"
    ].deployment_configuration.virtual_machine_configuration
    assert vm_config.node_placement_configuration.policy == expected_policy


def test_create_pool_with_blob_mounts_rejects_invalid_availability_zone(
    monkeypatch,
):
    install_fake_azure_batch_modules(monkeypatch)
    install_fake_cloudops_modules(monkeypatch)
    client, _, _ = make_pool_mount_client()

    with pytest.raises(ValueError, match="zonal.*regional"):
        create_test_pool(client, availability_zones="rack-local")


def test_create_pool_with_blob_mounts_disables_blobfuse_cache(monkeypatch):
    install_fake_azure_batch_modules(monkeypatch)
    install_fake_cloudops_modules(monkeypatch)
    client, created, _ = make_pool_mount_client()

    create_test_pool(client, cache_blobfuse=False)

    mount_config = created["parameters"].kwargs["mount_configuration"][0]
    blob_config = mount_config.azure_blob_file_system_configuration
    assert blob_config.blobfuse_options == " -o direct_io"


def test_add_batch_task_explicit_container_image_avoids_pool_lookup(
    monkeypatch,
):
    install_fake_azure_batch_modules(monkeypatch)
    calls = install_fake_cloudops_modules(monkeypatch, fail_pool_lookup=True)
    client, _, added = make_pool_mount_client()

    task_id = add_test_task(client, job_name="job-explicit-image")

    assert task_id.startswith("task-run-")
    assert added["task"].container_settings.image_name == "explicit/image:tag"
    assert "get_pool_full_info" not in calls


def test_add_batch_task_explicit_log_path_takes_precedence(monkeypatch):
    install_fake_azure_batch_modules(monkeypatch)
    install_fake_cloudops_modules(monkeypatch, fail_log_mount_lookup=True)
    client, _, added = make_pool_mount_client()
    client.save_logs_to_blob = "logs"

    add_test_task(
        client,
        job_name="job-explicit-logs",
        save_logs_path="/custom/logs",
        logs_folder="folder",
    )

    assert added["task"].command_line == "SAVE(/custom/logs,folder)::do-work"


def test_add_batch_task_mount_pairs_avoid_pool_mount_lookup(monkeypatch):
    install_fake_azure_batch_modules(monkeypatch)
    calls = install_fake_cloudops_modules(
        monkeypatch,
        fail_pool_mount_lookup=True,
    )
    client, _, added = make_pool_mount_client()

    add_test_task(client, job_name="job-explicit-mounts")

    assert "--mount cloud-output:cloud-output" in (
        added["task"].container_settings.container_run_options
    )
    assert "get_pool_mounts" not in calls


def test_add_batch_task_max_override_records_and_uses_counter(monkeypatch):
    install_fake_azure_batch_modules(monkeypatch)
    install_fake_cloudops_modules(monkeypatch)
    client, _, _ = make_pool_mount_client()

    first_task_id = add_test_task(
        client,
        job_name="job-override-counter",
        task_name_suffix="abc",
        task_id_max_override=41,
    )
    next_task_id = add_test_task(
        client,
        job_name="job-override-counter",
        task_name_suffix="abc",
    )

    assert first_task_id == "task-abc-42"
    assert next_task_id == "task-abc-43"


def test_add_batch_task_logs_redirect_when_client_configured(monkeypatch):
    install_fake_azure_batch_modules(monkeypatch)
    calls = install_fake_cloudops_modules(monkeypatch)
    client, _, added = make_pool_mount_client()
    client.save_logs_to_blob = "logs"

    add_test_task(
        client,
        job_name="job-blob-logs",
        logs_folder="runlogs",
    )

    assert added["task"].command_line == "SAVE(/pool/logs,runlogs)::do-work"
    assert "get_rel_mnt_path" in calls
