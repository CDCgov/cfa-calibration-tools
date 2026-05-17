from __future__ import annotations

from datetime import datetime
from typing import Any


def format_task_timing_summary(
    *,
    run_id: str,
    job_name: str,
    task_id: str,
    task: Any,
    total_elapsed_seconds: float,
    upload_elapsed_seconds: float | None = None,
    wait_elapsed_seconds: float | None = None,
    download_elapsed_seconds: float | None = None,
) -> str:
    creation_time = _coerce_datetime(getattr(task, "creation_time", None))
    execution_info = getattr(task, "execution_info", None)
    start_time = _coerce_datetime(getattr(execution_info, "start_time", None))
    end_time = _coerce_datetime(getattr(execution_info, "end_time", None))

    parts = [
        f"[cloud-task] {run_id}",
        f"job={job_name}",
        f"task={task_id}",
        f"total={total_elapsed_seconds:.2f}s",
    ]
    if upload_elapsed_seconds is not None:
        parts.append(f"upload={upload_elapsed_seconds:.2f}s")
    if wait_elapsed_seconds is not None:
        parts.append(f"wait={wait_elapsed_seconds:.2f}s")
    queue_seconds = _seconds_between(creation_time, start_time)
    if queue_seconds is not None:
        parts.append(f"queue={queue_seconds:.2f}s")
    run_seconds = _seconds_between(start_time, end_time)
    if run_seconds is not None:
        parts.append(f"run={run_seconds:.2f}s")
    if download_elapsed_seconds is not None:
        parts.append(f"download={download_elapsed_seconds:.2f}s")
    return " ".join(parts)


def format_task_failure_message(
    *,
    run_id: str,
    job_name: str,
    task_id: str,
    task_status: dict[str, Any],
    logs_container: str,
    logs_folder: str,
) -> str:
    task = task_status.get("task")
    execution_info = getattr(task, "execution_info", None)
    exit_code = task_status.get("exit_code")
    failure_info = getattr(execution_info, "failure_info", None)
    container_info = getattr(execution_info, "container_info", None)
    exit_conditions = getattr(task, "exit_conditions", None)

    details: list[str] = [
        f"run_id={run_id}",
        f"result={task_status.get('result')!r}",
        f"exit_code={exit_code!r}",
    ]

    failure_summary = _format_failure_info(failure_info)
    if failure_summary is not None:
        details.append(f"failure_info={failure_summary}")

    container_error = getattr(container_info, "error", None)
    if container_error:
        details.append(f"container_error={container_error!r}")

    for attr_name in ("pre_processing_error", "file_upload_error"):
        error_value = getattr(exit_conditions, attr_name, None)
        if error_value:
            details.append(f"{attr_name}={error_value!r}")

    details.append(
        f"logs_prefix={logs_container}/{logs_folder}/stdout_stderr/"
    )

    return (
        f"Azure Batch task {task_id} in job {job_name} failed "
        f"({', '.join(details)})."
    )


def append_task_log_excerpts(
    message: str,
    *,
    task_log_excerpts: dict[str, str] | None,
) -> str:
    if not task_log_excerpts:
        return message

    details: list[str] = []
    stderr_excerpt = task_log_excerpts.get("stderr")
    if stderr_excerpt:
        details.append(f"stderr_excerpt={stderr_excerpt!r}")
    stdout_excerpt = task_log_excerpts.get("stdout")
    if stdout_excerpt:
        details.append(f"stdout_excerpt={stdout_excerpt!r}")
    if not details:
        return message
    return f"{message} Task log excerpts: {', '.join(details)}."


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None
    return None


def _seconds_between(
    start: datetime | None, end: datetime | None
) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def _format_failure_info(failure_info: Any) -> str | None:
    if failure_info is None:
        return None

    parts: list[str] = []
    category = getattr(failure_info, "category", None)
    if category is not None:
        parts.append(f"category={category!r}")
    code = getattr(failure_info, "code", None)
    if code is not None:
        parts.append(f"code={code!r}")
    message = getattr(failure_info, "message", None)
    if message:
        parts.append(f"message={message!r}")

    details = getattr(failure_info, "details", None)
    if details:
        rendered_details = []
        for detail in details:
            name = getattr(detail, "name", None)
            value = getattr(detail, "value", None)
            if name is not None or value is not None:
                rendered_details.append(f"{name}={value}")
            else:
                rendered_details.append(repr(detail))
        if rendered_details:
            parts.append(f"details=[{'; '.join(rendered_details)}]")

    if not parts:
        return repr(failure_info)
    return ", ".join(parts)
