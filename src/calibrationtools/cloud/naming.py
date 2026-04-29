from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

from calibrationtools.run_id import (
    format_generation_name as _format_generation_name,
)
from calibrationtools.run_id import (
    parse_sampler_run_id,
)

DEFAULT_BATCH_TASK_ID_MAX_LENGTH = 64

_JOB_TASK_ID_MAX_BY_JOB: dict[str, int] = {}
_JOB_TASK_ID_MAX_LOCK = Lock()


def parse_generation_from_run_id(run_id: str) -> int:
    return parse_sampler_run_id(run_id).generation_index


def parse_particle_from_run_id(run_id: str) -> int:
    return parse_sampler_run_id(run_id).proposal_index


def format_generation_name(generation_index: int) -> str:
    return _format_generation_name(generation_index)


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned.strip("-")


def make_resource_name(prefix: str, suffix: str, *, max_length: int) -> str:
    prefix_part = sanitize_name(prefix)
    suffix_part = sanitize_name(suffix)
    if not prefix_part:
        prefix_part = "cloud"
    if not suffix_part:
        suffix_part = "run"
    candidate = f"{prefix_part}-{suffix_part}"
    if len(candidate) <= max_length:
        return candidate
    max_suffix_length = max_length - len(prefix_part) - 1
    if max_suffix_length <= 0:
        return prefix_part[:max_length]
    return f"{prefix_part}-{suffix_part[-max_suffix_length:]}"


def make_session_slug(tag: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    unique_suffix = uuid4().hex[:12]
    return sanitize_name(f"{timestamp}-{tag}-{unique_suffix}")


def parse_image_tag_from_session_slug(session_slug: str) -> str | None:
    normalized = sanitize_name(session_slug)
    parts = normalized.split("-")
    if len(parts) < 2:
        return None

    timestamp = parts[0]
    if len(timestamp) != 14 or not timestamp.isdigit():
        return None

    has_unique_suffix = (
        len(parts) >= 3
        and len(parts[-1]) == 12
        and all(char in "0123456789abcdef" for char in parts[-1])
    )
    tag_parts = parts[1:-1] if has_unique_suffix else parts[1:]
    image_tag = "-".join(tag_parts)
    return image_tag or None


def make_batch_task_name_suffix(value: str, *, max_length: int = 57) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    if not cleaned:
        cleaned = "run"
    if len(cleaned) <= max_length:
        return cleaned
    digest = hashlib.sha1(cleaned.encode()).hexdigest()[:8]
    head_length = max_length - len(digest) - 1
    if head_length <= 0:
        return digest[:max_length]
    return f"{cleaned[:head_length]}-{digest}"


def make_batch_task_id(
    task_name_suffix: str,
    *,
    task_id_base: str = "task",
    task_id_max: int = 0,
) -> str:
    task_number = task_id_max + 1
    max_suffix_length = _max_batch_task_name_suffix_length(
        task_id_base=task_id_base,
        task_number=task_number,
    )
    return (
        f"{task_id_base}-"
        f"{make_batch_task_name_suffix(task_name_suffix, max_length=max_suffix_length)}-"
        f"{task_number}"
    )


def make_stable_batch_task_id_max(
    job_name: str,
    task_name_suffix: str,
) -> int:
    digest = hashlib.sha1(
        f"{job_name}:{task_name_suffix}".encode()
    ).hexdigest()
    return int(digest[:12], 16)


def _max_batch_task_name_suffix_length(
    *,
    task_id_base: str,
    task_number: int,
) -> int:
    digits = len(str(task_number))
    return max(
        1,
        DEFAULT_BATCH_TASK_ID_MAX_LENGTH - len(task_id_base) - digits - 2,
    )


def _next_job_task_id_max(job_name: str) -> int:
    with _JOB_TASK_ID_MAX_LOCK:
        current = _JOB_TASK_ID_MAX_BY_JOB.get(job_name, -1) + 1
        _JOB_TASK_ID_MAX_BY_JOB[job_name] = current
        return current


def _record_job_task_id_max(job_name: str, task_id_max: int) -> None:
    with _JOB_TASK_ID_MAX_LOCK:
        current = _JOB_TASK_ID_MAX_BY_JOB.get(job_name, -1)
        if task_id_max > current:
            _JOB_TASK_ID_MAX_BY_JOB[job_name] = task_id_max
