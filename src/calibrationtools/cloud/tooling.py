from __future__ import annotations

import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required tool not found on PATH: {name}")


def run_command(command: list[str], *, cwd: Path) -> None:
    """Run a subprocess, raising a RuntimeError on failure."""
    try:
        subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or "no output captured"
        raise RuntimeError(
            f"Command {command!r} (cwd={cwd}) failed with exit code "
            f"{exc.returncode}: {detail}"
        ) from exc


def git_short_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip() or "no stderr"
        raise RuntimeError(
            f"Failed to resolve short git SHA in {repo_root}: {stderr}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "git executable not found on PATH; required to derive the cloud "
            "image tag."
        ) from exc
    return result.stdout.strip()


def build_local_image(
    *,
    repo_root: Path,
    dockerfile: Path,
    local_image: str,
    tag: str,
) -> str:
    require_tool("docker")
    local_image_ref = f"{local_image}:{tag}"
    run_command(
        ["docker", "build", "-t", local_image_ref, "-f", str(dockerfile), "."],
        cwd=repo_root,
    )
    return local_image_ref


def upload_local_image(
    *,
    client: Any,
    local_image_ref: str,
    repository: str,
    tag: str,
) -> str:
    require_tool("az")
    registry_account = client.cred.azure_container_registry_account
    if not registry_account:
        raise SystemExit(
            "AZURE_CONTAINER_REGISTRY_ACCOUNT must be set in the environment or .env."
        )
    return client.upload_docker_image(
        image_name=local_image_ref,
        registry_name=registry_account,
        repo_name=repository,
        tag=tag,
    )


def upload_files_quietly(
    client: Any,
    *,
    files: str | list[str],
    container_name: str,
    local_root_dir: str = ".",
    location_in_blob: str = ".",
    legal_hold: bool = False,
    immutability_lock_days: int = 0,
) -> None:
    """Upload blobs without emitting cloudops progress bars or prints.

    ``cfa.cloudops.CloudClient.upload_files()`` currently writes a tqdm
    progress bar plus a final ``print(...)`` to process stdio. For our cloud
    runner/executor paths that noise is not desirable, so when a real
    ``blob_service_client`` is available we upload directly through it.

    Test doubles that only implement ``upload_files()`` still work via the
    fallback path below.
    """
    blob_service_client = getattr(client, "blob_service_client", None)
    if blob_service_client is None:
        client.upload_files(
            files=files,
            container_name=container_name,
            local_root_dir=local_root_dir,
            location_in_blob=location_in_blob,
            legal_hold=legal_hold,
            immutability_lock_days=immutability_lock_days,
        )
        return

    file_paths = [files] if isinstance(files, str) else list(files)
    if not file_paths:
        return

    immutability_policy = None
    if immutability_lock_days > 0:
        from azure.storage.blob import (
            BlobImmutabilityPolicyMode,
            ImmutabilityPolicy,
        )

        immutability_policy = ImmutabilityPolicy(
            expiry_time=datetime.now(timezone.utc)
            + timedelta(days=immutability_lock_days),
            policy_mode=BlobImmutabilityPolicyMode.UNLOCKED,
        )

    last_blob_client = None
    for file_path in file_paths:
        local_file_path = os.path.join(local_root_dir, file_path)
        remote_file_path = os.path.join(location_in_blob, file_path)
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=remote_file_path,
        )
        with open(local_file_path, "rb") as upload_data:
            blob_client.upload_blob(
                upload_data,
                overwrite=True,
                immutability_policy=immutability_policy,
            )
        if legal_hold:
            blob_client.set_legal_hold(legal_hold=legal_hold)
        last_blob_client = blob_client

    if immutability_policy is not None and last_blob_client is not None:
        last_blob_client.lock_blob_immutability_policy()


@contextmanager
def suppress_cloudops_info_output() -> Iterator[None]:
    """Silence noisy ``cfa.cloudops`` logger output while a block runs."""
    logger = logging.getLogger("cfa.cloudops")
    previous_level = logger.level
    try:
        logger.setLevel(max(previous_level, logging.WARNING))
        yield
    finally:
        logger.setLevel(previous_level)


def create_cloud_client(*, keyvault: str) -> Any:
    try:
        from cfa.cloudops import CloudClient
    except ImportError as exc:
        raise SystemExit(
            "Could not import cfa.cloudops. Install the cloud dependencies "
            "or run with the repo's `cloudops` dependency group."
        ) from exc
    return CloudClient(keyvault=keyvault)
