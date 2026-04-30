from __future__ import annotations

import json
import shlex
import sys
import tempfile
import time
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
    resolved_backend = backend or CloudExecutorBackend(
        create_cloud_client=create_cloud_client_func,
        add_batch_task_with_short_id=add_batch_task_with_short_id_func,
        wait_for_task_completion=wait_for_task_completion_func,
        cancel_batch_task=cancel_batch_task_func,
        format_task_failure_message=format_task_failure_message_func,
        format_task_timing_summary=format_task_timing_summary_func,
        suppress_cloudops_info_output=suppress_cloudops_info_output_func,
    )
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

    client = resolved_backend.create_cloud_client(keyvault=session.keyvault)
    job_name = cloud.get("job_name")
    if not isinstance(job_name, str) or not job_name:
        job_name = session.job_name_for_run(run_id)
    client.save_logs_to_blob = session.logs_container
    client.logs_folder = session.logs_folder_for_job(job_name, run_id)

    remote_input_dir = session.remote_input_dir(run_id)
    remote_output_dir = session.remote_output_dir(run_id)
    input_filename = f"{run_id}.json"
    overall_started = time.monotonic()
    upload_started = time.monotonic()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        local_input_path = tmp_path / input_filename
        local_input_path.write_text(dumps_json(input_payload) + "\n")
        with resolved_backend.suppress_cloudops_info_output():
            upload_files_quietly(
                client,
                files=input_filename,
                container_name=session.input_container,
                local_root_dir=tmpdir,
                location_in_blob=remote_input_dir,
            )
    upload_elapsed = time.monotonic() - upload_started

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
    wait_started = time.monotonic()
    task_id = resolved_backend.add_batch_task_with_short_id(
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

    try:
        task_status = resolved_backend.wait_for_task_completion(
            batch_client=client.batch_service_client,
            job_name=job_name,
            task_id=task_id,
            timeout_minutes=session.task_timeout_minutes,
        )
    except BaseException:
        # Best-effort cancel the submitted Batch task so a local
        # wait-path timeout or polling failure does not leave the
        # remote task running and producing late output.
        try:
            resolved_backend.cancel_batch_task(
                batch_client=client.batch_service_client,
                job_name=job_name,
                task_id=task_id,
            )
        except Exception:
            pass
        raise
    wait_elapsed = time.monotonic() - wait_started

    download_elapsed = None
    if task_status["result"] == "success":
        download_started = time.monotonic()
        download_blob_to_path_atomic(
            client,
            src_path=f"{remote_output_dir}/{output_filename}",
            dest_path=output_dir / output_filename,
            container_name=session.output_container,
            download_file_kwargs={"do_check": False, "check_size": False},
        )
        download_elapsed = time.monotonic() - download_started

    if session.print_task_durations:
        print(
            resolved_backend.format_task_timing_summary(
                run_id=run_id,
                job_name=job_name,
                task_id=task_id,
                task=task_status["task"],
                total_elapsed_seconds=time.monotonic() - overall_started,
                upload_elapsed_seconds=upload_elapsed,
                wait_elapsed_seconds=wait_elapsed,
                download_elapsed_seconds=download_elapsed,
            ),
            file=sys.stderr,
            flush=True,
        )

    if task_status["result"] != "success":
        failure_message = resolved_backend.format_task_failure_message(
            run_id=run_id,
            job_name=job_name,
            task_id=task_id,
            task_status=task_status,
            logs_container=session.logs_container,
            logs_folder=session.logs_folder_for_job(job_name, run_id),
        )
        failure_message = append_task_log_excerpts(
            failure_message,
            task_log_excerpts=read_task_log_excerpts(
                client,
                container_name=session.logs_container,
                logs_folder=session.logs_folder_for_job(job_name, run_id),
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
