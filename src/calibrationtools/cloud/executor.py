from __future__ import annotations

import json
import shlex
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from calibrationtools.json_utils import dumps_json

from .artifacts import (
    download_blob_to_path_atomic,
    read_task_log_excerpts,
    resolve_filesystem_output_dir,
)
from .backend import DEFAULT_CLOUD_EXECUTOR_BACKEND, CloudExecutorBackend
from .formatting import append_task_log_excerpts
from .naming import make_stable_batch_task_id_max
from .session import CloudSession
from .tooling import upload_files_quietly


@dataclass(frozen=True)
class _ExecutorContext:
    session: CloudSession
    run_id: str
    job_name: str
    input_payload: Any
    output_dir: Path
    client: Any


@dataclass(frozen=True)
class _UploadedExecutorInput:
    remote_input_dir: str
    elapsed_seconds: float


def execute_cloud_run(
    run_json: dict[str, Any],
    *,
    output_filename: str = "output.csv",
    backend: CloudExecutorBackend | None = None,
    create_cloud_client_func: Callable[..., Any] = (
        DEFAULT_CLOUD_EXECUTOR_BACKEND.create_cloud_client
    ),
    add_batch_task_with_short_id_func: Callable[..., str] = (
        DEFAULT_CLOUD_EXECUTOR_BACKEND.add_batch_task_with_short_id
    ),
    wait_for_task_completion_func: Callable[..., dict[str, Any]] = (
        DEFAULT_CLOUD_EXECUTOR_BACKEND.wait_for_task_completion
    ),
    cancel_batch_task_func: Callable[..., None] = (
        DEFAULT_CLOUD_EXECUTOR_BACKEND.cancel_batch_task
    ),
    format_task_failure_message_func: Callable[..., str] = (
        DEFAULT_CLOUD_EXECUTOR_BACKEND.format_task_failure_message
    ),
    format_task_timing_summary_func: Callable[..., str] = (
        DEFAULT_CLOUD_EXECUTOR_BACKEND.format_task_timing_summary
    ),
    suppress_cloudops_info_output_func: Callable[[], Any] = (
        DEFAULT_CLOUD_EXECUTOR_BACKEND.suppress_cloudops_info_output
    ),
) -> None:
    resolved_backend = _resolve_executor_backend(
        backend=backend,
        create_cloud_client_func=create_cloud_client_func,
        add_batch_task_with_short_id_func=add_batch_task_with_short_id_func,
        wait_for_task_completion_func=wait_for_task_completion_func,
        cancel_batch_task_func=cancel_batch_task_func,
        format_task_failure_message_func=format_task_failure_message_func,
        format_task_timing_summary_func=format_task_timing_summary_func,
        suppress_cloudops_info_output_func=suppress_cloudops_info_output_func,
    )
    _execute_cloud_run_with_backend(
        run_json,
        output_filename=output_filename,
        backend=resolved_backend,
    )


def _execute_cloud_run_with_backend(
    run_json: dict[str, Any],
    *,
    output_filename: str,
    backend: CloudExecutorBackend,
) -> None:
    overall_started = time.monotonic()
    context = _resolve_execution_context(
        run_json,
        output_filename=output_filename,
        backend=backend,
    )
    remote_output_dir = context.session.remote_output_dir(context.run_id)
    uploaded_input = _upload_executor_input(
        backend,
        context.client,
        context.session,
        context.run_id,
        context.input_payload,
    )
    wait_started = time.monotonic()
    task_id = _submit_executor_task(
        backend,
        context.client,
        context.session,
        context.job_name,
        context.run_id,
        uploaded_input.remote_input_dir,
        remote_output_dir,
    )
    task_status = _wait_for_executor_task(
        backend,
        context.client,
        context.job_name,
        task_id,
        context.session.task_timeout_minutes,
    )
    wait_elapsed = time.monotonic() - wait_started

    download_elapsed = None
    if task_status["result"] == "success":
        download_elapsed = _download_executor_output(
            context.client,
            context.session,
            remote_output_dir,
            context.output_dir,
            output_filename,
        )

    _emit_executor_timing_summary(
        backend,
        context.session,
        run_id=context.run_id,
        job_name=context.job_name,
        task_id=task_id,
        task_status=task_status,
        total_elapsed_seconds=time.monotonic() - overall_started,
        upload_elapsed_seconds=uploaded_input.elapsed_seconds,
        wait_elapsed_seconds=wait_elapsed,
        download_elapsed_seconds=download_elapsed,
    )

    if task_status["result"] != "success":
        _raise_executor_task_failure(
            backend,
            context.client,
            context.session,
            run_id=context.run_id,
            job_name=context.job_name,
            task_id=task_id,
            task_status=task_status,
        )


def _resolve_executor_backend(
    *,
    backend: CloudExecutorBackend | None,
    create_cloud_client_func: Callable[..., Any],
    add_batch_task_with_short_id_func: Callable[..., str],
    wait_for_task_completion_func: Callable[..., dict[str, Any]],
    cancel_batch_task_func: Callable[..., None],
    format_task_failure_message_func: Callable[..., str],
    format_task_timing_summary_func: Callable[..., str],
    suppress_cloudops_info_output_func: Callable[[], Any],
) -> CloudExecutorBackend:
    if backend is not None:
        return backend
    return CloudExecutorBackend(
        create_cloud_client=create_cloud_client_func,
        add_batch_task_with_short_id=add_batch_task_with_short_id_func,
        wait_for_task_completion=wait_for_task_completion_func,
        cancel_batch_task=cancel_batch_task_func,
        format_task_failure_message=format_task_failure_message_func,
        format_task_timing_summary=format_task_timing_summary_func,
        suppress_cloudops_info_output=suppress_cloudops_info_output_func,
    )


def _resolve_execution_context(
    run_json: dict[str, Any],
    *,
    output_filename: str,
    backend: CloudExecutorBackend,
) -> _ExecutorContext:
    runtime = run_json.get("runtime", {})
    cloud = runtime.get("cloud")
    if not isinstance(cloud, dict):
        raise ValueError(
            "Cloud runtime metadata is missing from the MRP run JSON."
        )

    session = CloudSession.from_runtime_cloud(cloud)
    input_payload = _resolve_input_payload(run_json.get("input"))
    run_id = cloud.get("run_id")
    if not run_id and isinstance(input_payload, dict):
        run_id = input_payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Cloud execution requires an input `run_id`.")

    output_dir = resolve_filesystem_output_dir(run_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = backend.create_cloud_client(keyvault=session.keyvault)
    job_name = cloud.get("job_name")
    if not isinstance(job_name, str) or not job_name:
        job_name = session.job_name_for_run(run_id)
    client.save_logs_to_blob = session.logs_container
    client.logs_folder = session.logs_folder_for_job(job_name, run_id)

    return _ExecutorContext(
        session=session,
        run_id=run_id,
        job_name=job_name,
        input_payload=input_payload,
        output_dir=output_dir,
        client=client,
    )


def _upload_executor_input(
    backend: CloudExecutorBackend,
    client: Any,
    session: CloudSession,
    run_id: str,
    input_payload: Any,
) -> _UploadedExecutorInput:
    remote_input_dir = session.remote_input_dir(run_id)
    input_filename = f"{run_id}.json"
    upload_started = time.monotonic()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        local_input_path = tmp_path / input_filename
        local_input_path.write_text(dumps_json(input_payload) + "\n")
        with backend.suppress_cloudops_info_output():
            upload_files_quietly(
                client,
                files=input_filename,
                container_name=session.input_container,
                local_root_dir=tmpdir,
                location_in_blob=remote_input_dir,
            )

    return _UploadedExecutorInput(
        remote_input_dir=remote_input_dir,
        elapsed_seconds=time.monotonic() - upload_started,
    )


def _submit_executor_task(
    backend: CloudExecutorBackend,
    client: Any,
    session: CloudSession,
    job_name: str,
    run_id: str,
    remote_input_dir: str,
    remote_output_dir: str,
) -> str:
    input_filename = f"{run_id}.json"
    remote_input_path = (
        f"{session.input_mount_path.rstrip('/')}/"
        f"{remote_input_dir}/{input_filename}"
    )
    remote_output_path = (
        f"{session.output_mount_path.rstrip('/')}/{remote_output_dir}"
    )
    task_command = _build_task_command(
        session.task_mrp_config_path,
        remote_input_path,
        remote_output_path,
    )
    return backend.add_batch_task_with_short_id(
        client=client,
        job_name=job_name,
        command_line=task_command,
        task_name_suffix=run_id,
        timeout=session.task_timeout_minutes,
        mount_pairs=session.mount_pairs(),
        container_image_name=session.remote_image_ref,
        save_logs_path=session.logs_mount_path,
        task_id_max_override=make_stable_batch_task_id_max(job_name, run_id),
    )


def _wait_for_executor_task(
    backend: CloudExecutorBackend,
    client: Any,
    job_name: str,
    task_id: str,
    timeout_minutes: int | None,
) -> dict[str, Any]:
    try:
        return backend.wait_for_task_completion(
            batch_client=client.batch_service_client,
            job_name=job_name,
            task_id=task_id,
            timeout_minutes=timeout_minutes,
        )
    except BaseException:
        try:
            backend.cancel_batch_task(
                batch_client=client.batch_service_client,
                job_name=job_name,
                task_id=task_id,
            )
        except Exception:
            pass
        raise


def _download_executor_output(
    client: Any,
    session: CloudSession,
    remote_output_dir: str,
    output_dir: Path,
    output_filename: str,
) -> float:
    download_started = time.monotonic()
    download_blob_to_path_atomic(
        client,
        src_path=f"{remote_output_dir}/{output_filename}",
        dest_path=output_dir / output_filename,
        container_name=session.output_container,
        download_file_kwargs={"do_check": False, "check_size": False},
    )
    return time.monotonic() - download_started


def _emit_executor_timing_summary(
    backend: CloudExecutorBackend,
    session: CloudSession,
    *,
    run_id: str,
    job_name: str,
    task_id: str,
    task_status: dict[str, Any],
    total_elapsed_seconds: float,
    upload_elapsed_seconds: float | None,
    wait_elapsed_seconds: float | None,
    download_elapsed_seconds: float | None,
) -> None:
    if not session.print_task_durations:
        return

    print(
        backend.format_task_timing_summary(
            run_id=run_id,
            job_name=job_name,
            task_id=task_id,
            task=task_status["task"],
            total_elapsed_seconds=total_elapsed_seconds,
            upload_elapsed_seconds=upload_elapsed_seconds,
            wait_elapsed_seconds=wait_elapsed_seconds,
            download_elapsed_seconds=download_elapsed_seconds,
        ),
        file=sys.stderr,
        flush=True,
    )


def _raise_executor_task_failure(
    backend: CloudExecutorBackend,
    client: Any,
    session: CloudSession,
    *,
    run_id: str,
    job_name: str,
    task_id: str,
    task_status: dict[str, Any],
) -> None:
    logs_folder = session.logs_folder_for_job(job_name, run_id)
    failure_message = backend.format_task_failure_message(
        run_id=run_id,
        job_name=job_name,
        task_id=task_id,
        task_status=task_status,
        logs_container=session.logs_container,
        logs_folder=logs_folder,
    )
    failure_message = append_task_log_excerpts(
        failure_message,
        task_log_excerpts=read_task_log_excerpts(
            client,
            container_name=session.logs_container,
            logs_folder=logs_folder,
        ),
    )
    raise RuntimeError(failure_message)


def read_run_json() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("Cloud MRP executor expected run JSON on stdin.")
    return json.loads(raw)


def _resolve_input_payload(input_value: Any) -> Any:
    if not isinstance(input_value, str):
        return input_value

    input_path = Path(input_value)
    if not input_path.exists():
        return input_value

    with input_path.open(encoding="utf-8") as f:
        return json.load(f)


def _build_task_command(
    task_mrp_config_path: str, remote_input_path: str, remote_output_path: str
) -> str:
    command = " ".join(
        shlex.quote(value)
        for value in [
            "mrp",
            "run",
            task_mrp_config_path,
            "--input",
            remote_input_path,
            "--output-dir",
            remote_output_path,
        ]
    )
    return f"/bin/bash -lc {shlex.quote(command)}"
