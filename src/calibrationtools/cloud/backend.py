from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .batch import (
    add_batch_task_with_short_id,
    cancel_batch_task,
    create_pool_with_blob_mounts,
    wait_for_pool_ready,
    wait_for_task_completion,
)
from .formatting import (
    format_task_failure_message,
    format_task_timing_summary,
)
from .naming import (
    make_resource_name,
    make_session_slug,
    parse_generation_from_run_id,
)
from .tooling import (
    build_local_image,
    create_cloud_client,
    git_short_sha,
    suppress_cloudops_info_output,
    upload_local_image,
)


@dataclass(frozen=True)
class CloudExecutorBackend:
    create_cloud_client: Callable[..., Any]
    add_batch_task_with_short_id: Callable[..., str]
    wait_for_task_completion: Callable[..., dict[str, Any]]
    cancel_batch_task: Callable[..., None]
    format_task_failure_message: Callable[..., str]
    format_task_timing_summary: Callable[..., str]
    suppress_cloudops_info_output: Callable[[], Any]


@dataclass(frozen=True)
class CloudRunnerBackend:
    create_cloud_client: Callable[..., Any]
    git_short_sha: Callable[[Path], str]
    make_session_slug: Callable[[str], str]
    build_local_image: Callable[..., str]
    upload_local_image: Callable[..., str]
    create_pool_with_blob_mounts: Callable[..., None]
    wait_for_pool_ready: Callable[..., Any]
    add_batch_task_with_short_id: Callable[..., str]
    cancel_batch_task: Callable[..., None]
    format_task_failure_message: Callable[..., str]
    format_task_timing_summary: Callable[..., str]
    make_resource_name: Callable[..., str]
    parse_generation_from_run_id: Callable[[str], int]
    suppress_cloudops_info_output: Callable[[], Any]


DEFAULT_CLOUD_EXECUTOR_BACKEND = CloudExecutorBackend(
    create_cloud_client=create_cloud_client,
    add_batch_task_with_short_id=add_batch_task_with_short_id,
    wait_for_task_completion=wait_for_task_completion,
    cancel_batch_task=cancel_batch_task,
    format_task_failure_message=format_task_failure_message,
    format_task_timing_summary=format_task_timing_summary,
    suppress_cloudops_info_output=suppress_cloudops_info_output,
)


DEFAULT_CLOUD_RUNNER_BACKEND = CloudRunnerBackend(
    create_cloud_client=create_cloud_client,
    git_short_sha=git_short_sha,
    make_session_slug=make_session_slug,
    build_local_image=build_local_image,
    upload_local_image=upload_local_image,
    create_pool_with_blob_mounts=create_pool_with_blob_mounts,
    wait_for_pool_ready=wait_for_pool_ready,
    add_batch_task_with_short_id=add_batch_task_with_short_id,
    cancel_batch_task=cancel_batch_task,
    format_task_failure_message=format_task_failure_message,
    format_task_timing_summary=format_task_timing_summary,
    make_resource_name=make_resource_name,
    parse_generation_from_run_id=parse_generation_from_run_id,
    suppress_cloudops_info_output=suppress_cloudops_info_output,
)
