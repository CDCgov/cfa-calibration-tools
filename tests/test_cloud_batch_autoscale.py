from __future__ import annotations

import sys
from datetime import timedelta
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from calibrationtools.cloud.batch import build_pool_autoscale_formula


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
    from calibrationtools.cloud.batch import create_pool_with_blob_mounts

    created: dict[str, Any] = {}

    class FakePoolClient:
        def create(self, **kwargs):
            created.update(kwargs)

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

    class FakeContainerRegistry:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeComputeNodeIdentityReference:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAzureBlobFileSystemConfiguration:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeMountConfiguration:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeNodePlacementConfiguration:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_models = SimpleNamespace(
        AutoScaleSettings=FakeAutoScaleSettings,
        AzureBlobFileSystemConfiguration=FakeAzureBlobFileSystemConfiguration,
        ComputeNodeIdentityReference=FakeComputeNodeIdentityReference,
        ContainerConfiguration=FakeContainerConfiguration,
        ContainerRegistry=FakeContainerRegistry,
        MountConfiguration=FakeMountConfiguration,
        NodePlacementConfiguration=FakeNodePlacementConfiguration,
        NodePlacementPolicyType=SimpleNamespace(
            regional="regional", zonal="zonal"
        ),
        ScaleSettings=FakeScaleSettings,
    )
    fake_batch_module = ModuleType("azure.mgmt.batch")
    setattr(fake_batch_module, "models", fake_models)
    fake_mgmt_module = ModuleType("azure.mgmt")
    setattr(fake_mgmt_module, "batch", fake_batch_module)
    fake_azure_module = ModuleType("azure")
    setattr(fake_azure_module, "mgmt", fake_mgmt_module)
    monkeypatch.setitem(sys.modules, "azure", fake_azure_module)
    monkeypatch.setitem(sys.modules, "azure.mgmt", fake_mgmt_module)
    monkeypatch.setitem(sys.modules, "azure.mgmt.batch", fake_batch_module)

    def fake_default_pool_config(**kwargs):
        return SimpleNamespace(
            kwargs=kwargs,
            deployment_configuration=SimpleNamespace(
                virtual_machine_configuration=SimpleNamespace()
            ),
        )

    fake_defaults = SimpleNamespace(
        get_default_pool_config=fake_default_pool_config,
        assign_container_config=lambda pool_config, container_config: setattr(
            pool_config, "container_config", container_config
        ),
    )
    fake_batch_helpers = SimpleNamespace(get_vm_size=lambda value: value)
    fake_cloudops_module = ModuleType("cfa.cloudops")
    setattr(fake_cloudops_module, "defaults", fake_defaults)
    setattr(fake_cloudops_module, "batch_helpers", fake_batch_helpers)
    fake_cfa_module = ModuleType("cfa")
    setattr(fake_cfa_module, "cloudops", fake_cloudops_module)
    monkeypatch.setitem(sys.modules, "cfa", fake_cfa_module)
    monkeypatch.setitem(sys.modules, "cfa.cloudops", fake_cloudops_module)
    monkeypatch.setitem(sys.modules, "cfa.cloudops.defaults", fake_defaults)
    monkeypatch.setitem(
        sys.modules, "cfa.cloudops.batch_helpers", fake_batch_helpers
    )

    client = SimpleNamespace(
        cred=SimpleNamespace(
            azure_resource_group_name="rg",
            azure_batch_account="batch",
            azure_subnet_id="subnet",
            azure_user_assigned_identity="identity",
            azure_blob_storage_account="blob",
            azure_container_registry_account="acr",
        ),
        batch_mgmt_client=SimpleNamespace(pool=FakePoolClient()),
    )

    create_pool_with_blob_mounts(
        client=client,
        pool_name="pool",
        container_image_name="acr.azurecr.io/model:test",
        mounts=[{"source": "input", "target": "cloud-input"}],
        vm_size="large",
        target_dedicated_nodes=4,
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
