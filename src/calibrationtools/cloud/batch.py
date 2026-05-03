from __future__ import annotations

import datetime
import time
from typing import Any, cast

from .config import DEFAULT_POLL_INTERVAL_SECONDS
from .naming import (
    _max_batch_task_name_suffix_length,
    _next_job_task_id_max,
    _record_job_task_id_max,
    make_batch_task_name_suffix,
)


def wait_for_task_completion(
    *,
    batch_client: Any,
    job_name: str,
    task_id: str,
    timeout_minutes: int | None,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> dict[str, Any]:
    deadline = None
    if timeout_minutes is not None:
        deadline = time.monotonic() + (timeout_minutes * 60)

    while True:
        task = batch_client.task.get(job_name, task_id)
        state = _enum_value(getattr(task, "state", None))
        if state == "completed":
            execution_info = getattr(task, "execution_info", None)
            result = _enum_value(getattr(execution_info, "result", None))
            exit_code = getattr(execution_info, "exit_code", None)
            return {
                "state": state,
                "result": result,
                "exit_code": exit_code,
                "task": task,
            }
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for Azure Batch task {task_id} in job {job_name}."
            )
        time.sleep(poll_interval_seconds)


def wait_for_pool_ready(
    *,
    batch_client: Any,
    pool_name: str,
    timeout_minutes: int | None,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> Any:
    deadline = None
    if timeout_minutes is not None:
        deadline = time.monotonic() + (timeout_minutes * 60)

    while True:
        pool = batch_client.pool.get(pool_name)
        allocation_state = _enum_value(getattr(pool, "allocation_state", None))
        current_dedicated = getattr(pool, "current_dedicated_nodes", None)
        target_dedicated = getattr(pool, "target_dedicated_nodes", None)

        if allocation_state == "steady" and (
            target_dedicated is None
            or current_dedicated is None
            or current_dedicated >= target_dedicated
        ):
            return pool

        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for Azure Batch pool {pool_name} to become ready."
            )
        time.sleep(poll_interval_seconds)


def _enum_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    return value


def build_pool_autoscale_formula(
    *,
    max_dedicated_nodes: int,
    task_slots_per_node: int,
) -> str:
    if max_dedicated_nodes < 1:
        raise ValueError("max_dedicated_nodes must be at least 1")
    if task_slots_per_node < 1:
        raise ValueError("task_slots_per_node must be at least 1")

    return "\n".join(
        [
            f"maxNodes = {max_dedicated_nodes};",
            f"taskSlotsPerNode = {task_slots_per_node};",
            "samplePercent = $PendingTasks.GetSamplePercent("
            "TimeInterval_Minute * 5);",
            "pendingTasks = samplePercent < 70 ? "
            "max(0, $PendingTasks.GetSample(1)) : "
            "max($PendingTasks.GetSample(1), "
            "avg($PendingTasks.GetSample(TimeInterval_Minute * 5)));",
            "targetVMs = ceil(pendingTasks / taskSlotsPerNode);",
            "$TargetDedicatedNodes = min(maxNodes, max(0, targetVMs));",
            "$TargetLowPriorityNodes = 0;",
            "$NodeDeallocationOption = taskcompletion;",
        ]
    )


def cancel_batch_task(
    *, batch_client: Any, job_name: str, task_id: str
) -> None:
    try:
        batch_client.task.terminate(job_name, task_id)
        return
    except Exception as exc:
        if _is_missing_or_terminal_task_error(exc):
            return

    try:
        batch_client.task.delete(job_name, task_id)
    except Exception as exc:
        if _is_missing_or_terminal_task_error(exc):
            return
        raise


def _is_missing_or_terminal_task_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if status == 404:
        return True

    message = str(exc).lower()
    return any(
        fragment in message
        for fragment in (
            "not found",
            "does not exist",
            "completed state",
            "already completed",
            "taskcompleted",
        )
    )


def add_batch_task_with_short_id(
    *,
    client: Any,
    job_name: str,
    command_line: str,
    task_name_suffix: str,
    timeout: int | None,
    mount_pairs: list[dict[str, str]] | None = None,
    container_image_name: str | None = None,
    save_logs_path: str | None = None,
    logs_folder: str | None = None,
    task_id_base: str = "task",
    task_id_max_override: int | None = None,
) -> str:
    from azure.batch import models as batch_models

    pool_name = _resolve_task_pool_name(client, job_name)
    task_number = _resolve_next_task_number(
        job_name,
        task_id_max_override=task_id_max_override,
    )
    name_suffix = make_batch_task_name_suffix(
        task_name_suffix,
        max_length=_max_batch_task_name_suffix_length(
            task_id_base=task_id_base,
            task_number=task_number,
        ),
    )
    task_id = f"{task_id_base}-{name_suffix}-{task_number}"
    task_param = _build_task_add_parameter(
        batch_models=batch_models,
        job_name=job_name,
        task_id=task_id,
        task_number=task_number,
        command_line=command_line,
        timeout=timeout,
        mounts=_resolve_task_mounts(
            client,
            pool_name=pool_name,
            mount_pairs=mount_pairs,
        ),
        full_container_name=_resolve_task_container_image_name(
            client,
            pool_name=pool_name,
            container_image_name=container_image_name,
        ),
        save_logs_rel_path=_resolve_save_logs_rel_path(
            client,
            pool_name=pool_name,
            save_logs_path=save_logs_path,
        ),
        logs_folder=logs_folder
        or getattr(client, "logs_folder", "stdout_stderr"),
    )
    client.batch_service_client.task.add(job_id=job_name, task=task_param)
    return task_id


def _resolve_task_pool_name(client: Any, job_name: str) -> str:
    job_info = client.batch_service_client.job.get(job_name)
    return job_info.as_dict()["execution_info"]["pool_id"]


def _resolve_next_task_number(
    job_name: str,
    *,
    task_id_max_override: int | None,
) -> int:
    if task_id_max_override is None:
        task_id_max = _next_job_task_id_max(job_name)
    else:
        task_id_max = task_id_max_override
        _record_job_task_id_max(job_name, task_id_max)
    return task_id_max + 1


def _resolve_task_container_image_name(
    client: Any,
    *,
    pool_name: str,
    container_image_name: str | None,
) -> str:
    if container_image_name is not None:
        return container_image_name

    full_container_name = getattr(client, "full_container_name", None)
    if full_container_name is not None:
        return full_container_name

    from cfa.cloudops import batch_helpers

    pool_info = cast(
        Any,
        batch_helpers.get_pool_full_info(
            client.cred.azure_resource_group_name,
            client.cred.azure_batch_account,
            pool_name,
            client.batch_mgmt_client,
        ),
    )
    vm_config = (
        pool_info.deployment_configuration.virtual_machine_configuration
    )
    pool_container_names = (
        vm_config.container_configuration.container_image_names
    )
    return pool_container_names[0].split("://")[-1]


def _resolve_save_logs_rel_path(
    client: Any,
    *,
    pool_name: str,
    save_logs_path: str | None,
) -> str | None:
    from cfa.cloudops import batch_helpers, helpers

    if save_logs_path is not None:
        return "/" + helpers.format_rel_path(save_logs_path)

    if not getattr(client, "save_logs_to_blob", None):
        return None

    save_logs_rel_path = batch_helpers.get_rel_mnt_path(
        blob_name=client.save_logs_to_blob,
        pool_name=pool_name,
        resource_group_name=client.cred.azure_resource_group_name,
        account_name=client.cred.azure_batch_account,
        batch_mgmt_client=client.batch_mgmt_client,
    )
    if save_logs_rel_path == "ERROR!":
        return save_logs_rel_path
    return "/" + helpers.format_rel_path(rel_path=save_logs_rel_path)


def _resolve_task_mounts(
    client: Any,
    *,
    pool_name: str,
    mount_pairs: list[dict[str, str]] | None,
) -> list[dict[str, str]] | None:
    from cfa.cloudops import batch_helpers, helpers

    if mount_pairs is None:
        return batch_helpers.get_pool_mounts(
            pool_name,
            client.cred.azure_resource_group_name,
            client.cred.azure_batch_account,
            client.batch_mgmt_client,
        )
    return [
        {
            "source": helpers.format_rel_path(mount["target"]),
            "target": helpers.format_rel_path(mount["target"]),
        }
        for mount in mount_pairs
    ]


def _build_task_add_parameter(
    *,
    batch_models: Any,
    job_name: str,
    task_id: str,
    task_number: int,
    command_line: str,
    timeout: int | None,
    mounts: list[dict[str, str]] | None,
    full_container_name: str,
    save_logs_rel_path: str | None,
    logs_folder: str,
) -> Any:
    from cfa.cloudops import batch_helpers

    full_command_line = _build_task_command_line(
        command_line=command_line,
        job_name=job_name,
        task_id=task_id,
        save_logs_rel_path=save_logs_rel_path,
        logs_folder=logs_folder,
    )
    task_constraints = batch_models.TaskConstraints(
        max_wall_clock_time=(
            datetime.timedelta(minutes=timeout)
            if timeout is not None
            else None
        )
    )
    task_param = batch_models.TaskAddParameter(
        id=task_id,
        command_line=full_command_line,
        container_settings=batch_models.TaskContainerSettings(
            image_name=full_container_name,
            container_run_options=_build_container_run_options(
                job_name=job_name,
                task_number=task_number,
                mounts=mounts,
            ),
            working_directory="containerImageDefault",
        ),
        user_identity=_build_task_user_identity(batch_models),
        constraints=task_constraints,
        depends_on=None,
        run_dependent_tasks_on_failure=False,
        exit_conditions=batch_helpers._generate_exit_conditions(False),
    )
    return task_param


def _build_task_command_line(
    *,
    command_line: str,
    job_name: str,
    task_id: str,
    save_logs_rel_path: str | None,
    logs_folder: str,
) -> str:
    if save_logs_rel_path is None:
        return command_line
    if save_logs_rel_path == "ERROR!":
        return command_line
    from cfa.cloudops import batch_helpers

    return batch_helpers._generate_command_for_saving_logs(
        command_line=command_line,
        job_name=job_name,
        task_id=task_id,
        save_logs_rel_path=save_logs_rel_path,
        logs_folder=logs_folder,
    )


def _build_container_run_options(
    *,
    job_name: str,
    task_number: int,
    mounts: list[dict[str, str]] | None,
) -> str:
    from cfa.cloudops import batch_helpers

    # Azure BlobFuse mounts on Batch nodes are not reliably writable from the
    # image's default non-root user. Keep this root override explicit so the
    # task can write output.csv and logs back to the mounted containers.
    parts = [
        f"--name={job_name}_{task_number}",
        "--rm",
        "--user 0:0",
    ]
    mount_str = batch_helpers._generate_mount_string(mounts).strip()
    if mount_str:
        parts.append(mount_str)
    return " ".join(parts)


def _build_task_user_identity(batch_models: Any) -> Any:
    return batch_models.UserIdentity(
        auto_user=batch_models.AutoUserSpecification(
            scope=batch_models.AutoUserScope.pool,
            elevation_level=batch_models.ElevationLevel.admin,
        )
    )


def create_pool_with_blob_mounts(
    *,
    client: Any,
    pool_name: str,
    container_image_name: str,
    mounts: list[dict[str, str]],
    vm_size: str,
    target_dedicated_nodes: int,
    task_slots_per_node: int = 1,
    auto_scale_evaluation_interval_minutes: int = 5,
    availability_zones: str = "regional",
    cache_blobfuse: bool = True,
) -> None:
    from azure.mgmt.batch import models
    from cfa.cloudops import defaults as d

    if auto_scale_evaluation_interval_minutes < 5:
        raise ValueError(
            "auto_scale_evaluation_interval_minutes must be at least 5"
        )

    cred = client.cred
    _validate_pool_credentials(cred)
    identity_reference, mount_configuration = _build_blob_mount_configurations(
        models,
        cred,
        mounts,
        cache_blobfuse=cache_blobfuse,
    )

    pool_config = d.get_default_pool_config(
        pool_name=pool_name,
        subnet_id=cred.azure_subnet_id,
        user_assigned_identity=cred.azure_user_assigned_identity,
        mount_configuration=mount_configuration,
        vm_size=_resolve_pool_vm_size(vm_size),
    )
    pool_config.scale_settings = _build_pool_scale_settings(
        models,
        target_dedicated_nodes=target_dedicated_nodes,
        task_slots_per_node=task_slots_per_node,
        evaluation_interval_minutes=auto_scale_evaluation_interval_minutes,
    )
    pool_config.task_slots_per_node = task_slots_per_node

    container_config = _build_container_configuration(
        models,
        cred,
        container_image_name,
        identity_reference,
    )
    d.assign_container_config(pool_config, container_config)

    vm_config = (
        pool_config.deployment_configuration.virtual_machine_configuration
    )
    _apply_node_placement(models, vm_config, availability_zones)

    client.batch_mgmt_client.pool.create(
        resource_group_name=cred.azure_resource_group_name,
        account_name=cred.azure_batch_account,
        pool_name=pool_name,
        parameters=pool_config,
    )
    client.pool_name = pool_name


def _validate_pool_credentials(cred: Any) -> None:
    if not cred.azure_resource_group_name:
        raise ValueError("AZURE_RESOURCE_GROUP_NAME must be configured.")
    if not cred.azure_batch_account:
        raise ValueError("AZURE_BATCH_ACCOUNT must be configured.")
    if not cred.azure_subnet_id:
        raise ValueError("AZURE_SUBNET_ID must be configured.")
    if not cred.azure_user_assigned_identity:
        raise ValueError("AZURE_USER_ASSIGNED_IDENTITY must be configured.")
    if not cred.azure_blob_storage_account:
        raise ValueError("AZURE_BLOB_STORAGE_ACCOUNT must be configured.")
    if not cred.azure_container_registry_account:
        raise ValueError(
            "AZURE_CONTAINER_REGISTRY_ACCOUNT must be configured."
        )


def _resolve_pool_vm_size(vm_size: str) -> str:
    from cfa.cloudops.batch_helpers import get_vm_size

    if vm_size in {"xsmall", "small", "medium", "large", "xlarge"}:
        return get_vm_size(vm_size)
    return vm_size


def _build_blob_mount_configurations(
    models: Any,
    cred: Any,
    mounts: list[dict[str, str]],
    *,
    cache_blobfuse: bool,
) -> tuple[Any, list[Any]]:
    identity_reference = models.ComputeNodeIdentityReference(
        resource_id=cred.azure_user_assigned_identity
    )
    blobfuse_options = "" if cache_blobfuse else " -o direct_io"
    mount_configuration = [
        models.MountConfiguration(
            azure_blob_file_system_configuration=(
                models.AzureBlobFileSystemConfiguration(
                    account_name=cred.azure_blob_storage_account,
                    container_name=mount["source"],
                    relative_mount_path=mount["target"].lstrip("/"),
                    blobfuse_options=blobfuse_options,
                    identity_reference=identity_reference,
                )
            )
        )
        for mount in mounts
    ]
    return identity_reference, mount_configuration


def _build_pool_scale_settings(
    models: Any,
    *,
    target_dedicated_nodes: int,
    task_slots_per_node: int,
    evaluation_interval_minutes: int,
) -> Any:
    return models.ScaleSettings(
        auto_scale=models.AutoScaleSettings(
            formula=build_pool_autoscale_formula(
                max_dedicated_nodes=target_dedicated_nodes,
                task_slots_per_node=task_slots_per_node,
            ),
            evaluation_interval=datetime.timedelta(
                minutes=evaluation_interval_minutes
            ),
        )
    )


def _build_container_configuration(
    models: Any,
    cred: Any,
    container_image_name: str,
    identity_reference: Any,
) -> Any:
    return models.ContainerConfiguration(
        type="dockerCompatible",
        container_image_names=[container_image_name],
        container_registries=[
            models.ContainerRegistry(
                user_name=cred.azure_container_registry_account,
                registry_server=(
                    f"{cred.azure_container_registry_account}.azurecr.io"
                ),
                identity_reference=identity_reference,
            )
        ],
    )


def _apply_node_placement(
    models: Any,
    vm_config: Any,
    availability_zones: str,
) -> None:
    availability_zone = availability_zones.lower()
    if availability_zone == "regional":
        policy = models.NodePlacementPolicyType.regional
    elif availability_zone == "zonal":
        policy = models.NodePlacementPolicyType.zonal
    else:
        raise ValueError(
            "Availability zone needs to be 'zonal' or 'regional'."
        )

    vm_config.node_placement_configuration = models.NodePlacementConfiguration(
        policy=policy
    )
