"""Azure Batch task lifecycle: upload, submit, poll, harvest, cleanup."""

from __future__ import annotations

import pickle
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from ..cloud_executor import CloudAcceptanceResult, CloudAcceptanceTask
from . import _node_health
from ._console import capture_cloudops_output, suppress_upload_progress_bar


def upload_task_chunks(
    cloud_client: Any,
    tasks: list[CloudAcceptanceTask],
    *,
    base_name: str,
    blob_name: str,
    chunk_size: int,
) -> list[str]:
    """Pickle tasks into chunks and upload them to the shared Blob container."""

    task_dir = Path(tempfile.mkdtemp(prefix="calibrationtools-azure-"))
    names: list[str] = []
    try:
        width = max(1, len(str((len(tasks) - 1) // chunk_size)))
        stamp = int(time.time() * 1000)
        for index, start in enumerate(range(0, len(tasks), chunk_size)):
            name = f"tasks-{base_name}-{stamp}-{index:0{width}d}.pkl"
            with (task_dir / name).open("wb") as file:
                pickle.dump(tasks[start : start + chunk_size], file)
            names.append(name)
        with suppress_upload_progress_bar():
            cloud_client.upload_files(
                files=names,
                container_name=blob_name,
                local_root_dir=str(task_dir),
                location_in_blob=".",
            )
        return names
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


def submit_job(
    cloud_client: Any,
    *,
    job_id: str,
    pool_name: str,
    task_blobs: list[str],
    command_template: str,
    blob_name: str,
    mount_path: str,
    quiet: Callable[[], Any],
) -> None:
    """Create the job and add one task per uploaded chunk."""

    with quiet():
        cloud_client.create_job(job_id, pool_name=pool_name, exist_ok=True)
    width = max(1, len(str(len(task_blobs) - 1)))
    with quiet():
        for index, task_blob in enumerate(task_blobs):
            command = command_template.format(
                task_file=task_blob,
                blob_container=blob_name,
                mount_path=mount_path,
            )
            cloud_client.add_task(
                job_name=job_id,
                command_line=command,
                name_suffix=f"-{index:0{width}d}",
            )


def _list_batch_tasks(cloud_client: Any, job_id: str) -> list[Any]:
    """List task records across supported Azure Batch client versions."""

    batch_client = cloud_client.batch_service_client
    if hasattr(batch_client, "list_tasks"):
        return list(batch_client.list_tasks(job_id))
    return list(batch_client.task.list(job_id))


def _raise_for_failed_tasks(
    cloud_client: Any, job_id: str, task_records: list[Any]
) -> None:
    failures = [
        task
        for task in task_records
        if getattr(getattr(task, "execution_info", None), "exit_code", 0)
        not in (None, 0)
    ]
    if not failures:
        return
    failed = failures[0]
    task_id = getattr(failed, "id", "<unknown>")
    exit_code = getattr(
        getattr(failed, "execution_info", None), "exit_code", "?"
    )
    failure_info = getattr(
        getattr(failed, "execution_info", None), "failure_info", None
    )
    detail = getattr(failure_info, "message", None) or str(
        failure_info or "no details"
    )
    try:
        batch_client = cloud_client.batch_service_client
        if hasattr(batch_client, "download_task_file"):
            stream = batch_client.download_task_file(
                job_id, task_id, "stderr.txt"
            )
        else:
            stream = batch_client.file.get_from_task(
                job_id, task_id, "stderr.txt"
            )
        stderr = b"".join(stream).decode("utf-8", errors="replace")
    except Exception:
        stderr = ""
    message = (
        f"Azure Batch job {job_id} failed in task {task_id} "
        f"(exit {exit_code}): {detail}"
    )
    if stderr:
        message += f"\n\nRepresentative task stderr:\n{stderr[-4000:]}"
    raise RuntimeError(message)


def _wait_message(
    completed: int, running: int, total: int, health: dict[str, Any]
) -> str:
    """Summarize job and pool state for one progress line."""

    parts = [f"Azure task progress {completed}/{total}"]
    if running:
        parts.append(f"{running} running")
    states = health.get("node_states")
    if states:
        parts.append("nodes: " + _node_health.state_summary(states))
    elif states == {}:
        parts.append("nodes: none allocated yet")
    return " • ".join(parts)


def _try_harvest(
    cloud_client: Any,
    *,
    task_blob: str,
    blob_name: str,
    quiet_cloud_output: bool,
) -> list[CloudAcceptanceResult] | None:
    """Attempt to download and unpickle one task chunk's results.

    Workers write through a Blob mount that flushes independently of process
    exit, so a task that has already exited can briefly have no readable
    result blob. That miss is routine, not a failure, so it is swallowed
    quietly here and retried on the next poll instead of surfacing as a
    progress notice or aborting the run.

    Args:
        cloud_client (Any): Client used to download the result blob.
        task_blob (str): Task-chunk blob name whose results to fetch.
        blob_name (str): Shared Blob container holding task/result blobs.
        quiet_cloud_output (bool): Whether to swallow console output from
            the download.

    Returns:
        list[CloudAcceptanceResult] | None: The chunk's results, or ``None``
        if the result blob is not readable yet.
    """

    with tempfile.NamedTemporaryFile(
        "wb", suffix=".pkl", delete=False
    ) as file:
        destination = file.name
    try:
        with capture_cloudops_output(quiet_cloud_output):
            cloud_client.download_file(
                src_path=f"results-{task_blob}",
                dest_path=destination,
                container_name=blob_name,
            )
        with open(destination, "rb") as file:
            return pickle.load(file)
    except Exception:
        return None
    finally:
        Path(destination).unlink(missing_ok=True)


def harvest_results(
    cloud_client: Any,
    *,
    job_id: str,
    pool_name: str,
    task_blobs: list[str],
    blob_name: str,
    poll_interval: float,
    max_wait: float,
    max_wait_for_first_task_start: float,
    quiet_cloud_output: bool,
    emit: Callable[..., None],
    on_result: Callable[[CloudAcceptanceResult], None] | None,
) -> list[CloudAcceptanceResult]:
    """Poll a job to completion, downloading each chunk's results as it lands.

    Waiting for the very last task before downloading anything hides
    acceptance-rate progress for the whole generation and delays failure
    reports, so a finished chunk's results are downloaded and handed to
    ``on_result`` immediately, in blob order, while the rest of the job is
    still running.

    Args:
        cloud_client (Any): Client used to poll tasks and download results.
        job_id (str): Identifier of the submitted Azure Batch job.
        pool_name (str): Pool the job's tasks run on, for health polling.
        task_blobs (list[str]): Uploaded task-chunk blob names, in order.
        blob_name (str): Shared Blob container holding task/result blobs.
        poll_interval (float): Seconds to sleep between polls.
        max_wait (float): Maximum seconds to wait for the whole job.
        max_wait_for_first_task_start (float): Maximum seconds to wait for
            any task to start running.
        quiet_cloud_output (bool): Whether to swallow console output from
            individual result downloads.
        emit (Callable[..., None]): Progress-message sink.
        on_result (Callable[[CloudAcceptanceResult], None] | None): Called
            once per result, in download order, so callers can stream
            progress instead of waiting for every chunk to land.

    Returns:
        list[CloudAcceptanceResult]: Every task's results, in blob order.

    Raises:
        TimeoutError: If no task starts, or the job does not finish, within
            the configured limits.
    """

    started = time.monotonic()
    total = len(task_blobs)
    running_seen = False
    harvested: dict[int, list[CloudAcceptanceResult]] = {}
    while True:
        task_records = _list_batch_tasks(cloud_client, job_id)
        completed = sum(
            getattr(getattr(task, "execution_info", None), "exit_code", None)
            is not None
            for task in task_records
        )
        running = sum(
            str(getattr(task, "state", "")).lower().endswith("running")
            for task in task_records
        )
        running_seen = running_seen or running > 0 or completed > 0
        health = _node_health.pool_health(cloud_client, pool_name)
        emit(
            _wait_message(completed, running, total, health),
            job_id=job_id,
            completed=completed,
            running=running,
            total=total,
            harvested=len(harvested),
            elapsed_seconds=time.monotonic() - started,
            **health,
        )

        for index, record in enumerate(task_records):
            if index >= total or index in harvested:
                continue
            exit_code = getattr(
                getattr(record, "execution_info", None), "exit_code", None
            )
            if exit_code != 0:
                continue
            chunk_results = _try_harvest(
                cloud_client,
                task_blob=task_blobs[index],
                blob_name=blob_name,
                quiet_cloud_output=quiet_cloud_output,
            )
            if chunk_results is None:
                continue
            harvested[index] = chunk_results
            if on_result is not None:
                for result in chunk_results:
                    on_result(result)
            emit(
                f"Downloading Azure results {len(harvested)}/{total}",
                stage="download",
                job_id=job_id,
                completed=len(harvested),
                total=total,
            )

        if len(harvested) >= total:
            return [
                result for index in range(total) for result in harvested[index]
            ]
        if completed >= total:
            # Every task finished but a harvest is still missing: only a
            # failed task can explain that, since success just means the
            # result blob has not been observed yet and is retried above.
            _raise_for_failed_tasks(cloud_client, job_id, task_records)
        _node_health.raise_for_unhealthy_pool(
            pool_name, f"job {job_id}", health
        )
        elapsed = time.monotonic() - started
        if not running_seen and elapsed > max_wait_for_first_task_start:
            raise TimeoutError(
                f"No Azure Batch task in job {job_id} started within "
                f"{max_wait_for_first_task_start:.0f}s on pool {pool_name}. "
                f"Node states: {health.get('node_states') or 'unknown'}. "
                "This usually means the pool never allocated usable nodes "
                "(quota, image pull, or start-task failure)."
            )
        if elapsed > max_wait:
            raise TimeoutError(
                f"Azure Batch job {job_id} did not complete in time"
            )
        time.sleep(poll_interval)


def cleanup_job(
    cloud_client: Any,
    *,
    job_id: str,
    pool_name: str,
    delete_job_after: bool,
    delete_pool_after: bool,
    emit: Callable[..., None],
) -> None:
    """Delete the job and, if configured, the pool."""

    cleaned: list[str] = []
    if delete_job_after:
        cloud_client.delete_job(job_id)
        cleaned.append(f"job {job_id}")
    if delete_pool_after:
        cloud_client.delete_pool(pool_name)
        cleaned.append(f"pool {pool_name}")
    if cleaned:
        emit("Azure resources cleaned", resources=cleaned)
