from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


def download_blob_to_path_atomic(
    client: Any,
    *,
    src_path: str,
    dest_path: Path,
    container_name: str,
    download_file_kwargs: dict[str, Any] | None = None,
) -> None:
    """Download a blob into ``dest_path`` using a ``.part`` side-file."""
    final_path = Path(dest_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = final_path.with_name(final_path.name + ".part")
    partial_path.unlink(missing_ok=True)
    remote_size = _get_remote_blob_size(
        client, container_name=container_name, src_path=src_path
    )
    try:
        client.download_file(
            src_path=src_path,
            dest_path=str(partial_path),
            container_name=container_name,
            **(download_file_kwargs or {}),
        )
        if not partial_path.exists():
            raise RuntimeError(
                f"Downloaded output file is missing: {partial_path}"
            )
        local_size = partial_path.stat().st_size
        if remote_size is not None and local_size < remote_size:
            raise RuntimeError(
                "Downloaded output file is shorter than the remote blob "
                f"({local_size} < {remote_size} bytes): {partial_path}"
            )
        if remote_size is None and local_size == 0:
            raise RuntimeError(
                f"Downloaded output file is empty and remote size is "
                f"unavailable: {partial_path}"
            )
        partial_path.replace(final_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise


def read_task_log_excerpts(
    client: Any,
    *,
    container_name: str,
    logs_folder: str,
    max_lines: int = 40,
    max_chars: int = 2000,
) -> dict[str, str]:
    """Best-effort read of Azure Batch stdout/stderr log blobs.

    Returns a mapping like ``{"stderr": "...", "stdout": "..."}`` for any
    blobs that could be downloaded and contained non-empty text. Failures to
    download or decode logs are intentionally ignored so the original task
    failure is not masked.
    """
    excerpts: dict[str, str] = {}
    log_prefix = logs_folder.rstrip("/") + "/stdout_stderr"
    for stream_name in ("stderr", "stdout"):
        src_path = f"{log_prefix}/{stream_name}.txt"
        excerpt = _download_blob_text_excerpt(
            client,
            container_name=container_name,
            src_path=src_path,
            max_lines=max_lines,
            max_chars=max_chars,
        )
        if excerpt is not None:
            excerpts[stream_name] = excerpt
    return excerpts


def _get_remote_blob_size(
    client: Any,
    *,
    container_name: str,
    src_path: str,
) -> int | None:
    """Return the remote blob size in bytes, or ``None`` if unavailable."""
    blob_service_client = getattr(client, "blob_service_client", None)
    if blob_service_client is None:
        return None
    try:
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=src_path
        )
        properties = blob_client.get_blob_properties()
    except Exception:
        return None
    size = getattr(properties, "size", None)
    if isinstance(size, int):
        return size
    if isinstance(properties, dict):
        candidate = properties.get("size")
        if isinstance(candidate, int):
            return candidate
    return None


def _download_blob_text_excerpt(
    client: Any,
    *,
    container_name: str,
    src_path: str,
    max_lines: int,
    max_chars: int,
) -> str | None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = Path(tmpdir) / Path(src_path).name
        try:
            client.download_file(
                src_path=src_path,
                dest_path=str(dest_path),
                container_name=container_name,
                do_check=False,
                check_size=False,
            )
        except Exception:
            return None

        try:
            text = dest_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    return _tail_text_excerpt(
        text,
        max_lines=max_lines,
        max_chars=max_chars,
    )


def _tail_text_excerpt(
    text: str,
    *,
    max_lines: int,
    max_chars: int,
) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None

    lines = stripped.splitlines()
    truncated_lines = len(lines) > max_lines
    if truncated_lines:
        lines = lines[-max_lines:]

    excerpt = "\n".join(lines).strip()
    truncated_chars = len(excerpt) > max_chars
    if truncated_chars:
        excerpt = excerpt[-max_chars:].lstrip()

    if truncated_lines or truncated_chars:
        excerpt = f"[truncated] {excerpt}"
    return excerpt or None


def resolve_filesystem_output_dir(run_json: dict[str, Any]) -> Path:
    output = run_json.get("output", {})
    if output.get("spec") == "filesystem" and output.get("dir"):
        return Path(output["dir"])
    profiles = output.get("profile", {})
    if profiles:
        if (
            "default" in profiles
            and profiles["default"].get("spec") == "filesystem"
        ):
            return Path(profiles["default"]["dir"])
        for profile in profiles.values():
            if profile.get("spec") == "filesystem" and profile.get("dir"):
                return Path(profile["dir"])
    raise ValueError("Cloud runtime requires filesystem output.")
