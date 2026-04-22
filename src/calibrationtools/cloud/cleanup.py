from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import CloudRuntimeSettings
from .naming import parse_image_tag_from_session_slug, sanitize_name
from .tooling import require_tool

# Module-level identity cache used by the default login callable.
#
# `ensure_az_login_with_identity()` is deliberately stateless so that callers
# can supply their own identity cache in tests or alternative hosts. The
# shared API, however, still needs *some* default or it would raise on every
# call. We keep a tiny cache here so the top-level helpers (and their default
# login wrapper) work out of the box without forcing every caller to thread a
# sentinel through.
_AZ_NOT_LOGGED_IN: object = object()
_AZ_LOGGED_IN_IDENTITY: object | str | None = _AZ_NOT_LOGGED_IN


def _default_ensure_az_login_with_identity(
    *, managed_identity_resource_id: str | None = None
) -> None:
    """Default login callable used by the shared cleanup helpers.

    Maintains a module-level identity cache so repeated calls during a single
    process reuse an existing `az login`.
    """
    global _AZ_LOGGED_IN_IDENTITY

    _AZ_LOGGED_IN_IDENTITY = ensure_az_login_with_identity(
        managed_identity_resource_id=managed_identity_resource_id,
        current_identity=_AZ_LOGGED_IN_IDENTITY,
        not_logged_in_sentinel=_AZ_NOT_LOGGED_IN,
    )


@dataclass(frozen=True)
class CleanupPlan:
    config_path: Path
    session_slug: str
    keyvault: str
    registry_name: str | None
    repository_name: str
    image_tag: str | None
    job_names: tuple[str, ...]
    pool_names: tuple[str, ...]
    container_names: tuple[str, ...]
    acr_image_exists: bool

    @property
    def is_empty(self) -> bool:
        return (
            not self.job_names
            and not self.pool_names
            and not self.container_names
            and not self.acr_image_exists
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "config_path": str(self.config_path),
            "session_slug": self.session_slug,
            "keyvault": self.keyvault,
            "registry_name": self.registry_name,
            "repository_name": self.repository_name,
            "image_tag": self.image_tag,
            "job_names": list(self.job_names),
            "pool_names": list(self.pool_names),
            "container_names": list(self.container_names),
            "acr_image_exists": self.acr_image_exists,
        }


@dataclass(frozen=True)
class CleanupListing:
    config_path: Path
    session_slug: str | None
    image_tag: str | None
    keyvault: str
    registry_name: str | None
    repository_name: str
    job_names: tuple[str, ...]
    pool_names: tuple[str, ...]
    container_names: tuple[str, ...]
    acr_image_tags: tuple[str, ...]
    acr_image_tags_warning: str | None = None

    @property
    def is_empty(self) -> bool:
        return (
            not self.job_names
            and not self.pool_names
            and not self.container_names
            and not self.acr_image_tags
        )


@dataclass(frozen=True)
class CleanupResult:
    deleted: tuple[str, ...]
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


def build_parser(
    *,
    default_config_path: Path,
    description: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description
        or (
            "Clean cloud resources created by one calibration session. "
            "Use --list for read-only discovery across the project naming "
            "scope, or pass --session-slug to inspect or delete one session."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path,
        help=(
            "Cloud MRP config used to derive the project-specific Batch, Blob, "
            "and ACR naming scope. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--session-slug",
        help=(
            "Exact session slug to inspect or clean up. This is printed by "
            "the cloud runner and embedded in the Batch, Blob, and log paths."
        ),
    )
    parser.add_argument(
        "--image-tag",
        help=(
            "Optional ACR image tag to delete. Omit this to leave the shared "
            "repository untouched."
        ),
    )
    parser.add_argument(
        "--skip-acr",
        action="store_true",
        help="Skip cleanup of the optional Azure Container Registry image tag.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help=(
            "list all project-scoped "
            "resources; with --session-slug and/or --image-tag, narrow the output."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Actually delete the discovered Azure resources. Without this flag, "
            "the command only prints a dry-run plan."
        ),
    )
    return parser


def parse_args(
    argv: list[str] | None = None,
    *,
    default_config_path: Path,
    description: str | None = None,
) -> argparse.Namespace:
    args = build_parser(
        default_config_path=default_config_path,
        description=description,
    ).parse_args(argv)
    if not args.list and not args.session_slug:
        raise SystemExit("--session-slug is required unless --list is used.")
    return args


def discover_cleanup_listing(
    client,
    settings: CloudRuntimeSettings,
    *,
    config_path: Path,
    session_slug: str | None,
    image_tag: str | None,
    include_acr: bool,
    allow_acr_errors: bool = False,
    list_acr_repository_tags_fn: Callable[..., list[str]] | None = None,
) -> CleanupListing:
    if list_acr_repository_tags_fn is None:
        list_acr_repository_tags_fn = list_acr_repository_tags

    cred = client.cred
    if not cred.azure_resource_group_name:
        raise SystemExit("AZURE_RESOURCE_GROUP_NAME must be configured.")
    if not cred.azure_batch_account:
        raise SystemExit("AZURE_BATCH_ACCOUNT must be configured.")

    pool_prefixes = (settings.pool_prefix,)
    job_prefixes = (settings.job_prefix,)
    container_prefixes = (
        settings.input_container_prefix,
        settings.output_container_prefix,
        settings.logs_container_prefix,
    )

    if session_slug is None:
        pool_names = _matching_project_resource_names(
            _list_pool_names(client),
            pool_prefixes,
        )
        job_names = _matching_project_resource_names(
            _list_job_names(client),
            job_prefixes,
        )
        container_names = _matching_project_resource_names(
            _list_container_names(client),
            container_prefixes,
        )
    else:
        pool_names = _filter_names_for_session_slug(
            _list_pool_names(client),
            session_slug,
            _session_slugs_for_resource_name,
            pool_prefixes,
        )
        job_names = _filter_names_for_session_slug(
            _list_job_names(client),
            session_slug,
            _session_slugs_for_job_name,
            job_prefixes,
        )
        container_names = _filter_names_for_session_slug(
            _list_container_names(client),
            session_slug,
            _session_slugs_for_resource_name,
            container_prefixes,
        )

    registry_name = getattr(cred, "azure_container_registry_account", None)
    managed_identity_resource_id = getattr(
        cred, "azure_user_assigned_identity", None
    )
    acr_image_tags: tuple[str, ...] = ()
    acr_image_tags_warning: str | None = None
    if include_acr:
        if not registry_name:
            if allow_acr_errors:
                acr_image_tags_warning = (
                    "AZURE_CONTAINER_REGISTRY_ACCOUNT must be configured."
                )
            else:
                raise SystemExit(
                    "AZURE_CONTAINER_REGISTRY_ACCOUNT must be configured."
                )
        else:
            try:
                discovered_tags = tuple(
                    sorted(
                        list_acr_repository_tags_fn(
                            registry_name,
                            settings.repository,
                            managed_identity_resource_id=(
                                managed_identity_resource_id
                            ),
                        )
                    )
                )
            except Exception as exc:
                if not allow_acr_errors:
                    raise
                acr_image_tags_warning = _summarize_error_message(exc)
            else:
                if image_tag is None:
                    acr_image_tags = discovered_tags
                else:
                    acr_image_tags = tuple(
                        tag for tag in discovered_tags if tag == image_tag
                    )

    return CleanupListing(
        config_path=config_path,
        session_slug=session_slug,
        image_tag=image_tag,
        keyvault=settings.keyvault,
        registry_name=registry_name,
        repository_name=settings.repository,
        job_names=job_names,
        pool_names=pool_names,
        container_names=container_names,
        acr_image_tags=acr_image_tags,
        acr_image_tags_warning=acr_image_tags_warning,
    )


def discover_cleanup_plan(
    client,
    settings: CloudRuntimeSettings,
    *,
    config_path: Path,
    session_slug: str,
    image_tag: str | None,
    include_acr: bool,
    list_acr_repository_tags_fn: Callable[..., list[str]] | None = None,
) -> CleanupPlan:
    listing = discover_cleanup_listing(
        client,
        settings,
        config_path=config_path,
        session_slug=session_slug,
        image_tag=image_tag,
        include_acr=include_acr and image_tag is not None,
        list_acr_repository_tags_fn=list_acr_repository_tags_fn,
    )

    return CleanupPlan(
        config_path=config_path,
        session_slug=session_slug,
        keyvault=listing.keyvault,
        registry_name=listing.registry_name,
        repository_name=listing.repository_name,
        image_tag=image_tag,
        job_names=listing.job_names,
        pool_names=listing.pool_names,
        container_names=listing.container_names,
        acr_image_exists=(
            image_tag is not None and image_tag in listing.acr_image_tags
        ),
    )


def execute_cleanup(
    client,
    plan: CleanupPlan,
    *,
    include_acr: bool,
    delete_acr_image_tag_fn: Callable[..., None] | None = None,
) -> CleanupResult:
    if delete_acr_image_tag_fn is None:
        delete_acr_image_tag_fn = delete_acr_image_tag

    if not plan.session_slug and (
        plan.job_names or plan.pool_names or plan.container_names
    ):
        raise ValueError(
            "Cleanup requires a session_slug to delete Batch or Blob resources."
        )

    deleted: list[str] = []
    failures: list[str] = []

    for job_name in plan.job_names:
        try:
            client.delete_job(job_name)
            deleted.append(f"job:{job_name}")
        except Exception as exc:
            if _is_not_found_error(exc):
                deleted.append(f"job:{job_name} (already missing)")
            else:
                failures.append(f"job:{job_name}: {exc}")

    for pool_name in plan.pool_names:
        try:
            client.delete_pool(pool_name)
            deleted.append(f"pool:{pool_name}")
        except Exception as exc:
            if _is_not_found_error(exc):
                deleted.append(f"pool:{pool_name} (already missing)")
            else:
                failures.append(f"pool:{pool_name}: {exc}")

    for container_name in plan.container_names:
        try:
            client.blob_service_client.delete_container(container_name)
            deleted.append(f"container:{container_name}")
        except Exception as exc:
            if _is_not_found_error(exc):
                deleted.append(f"container:{container_name} (already missing)")
            else:
                failures.append(f"container:{container_name}: {exc}")

    if (
        include_acr
        and plan.acr_image_exists
        and plan.registry_name is not None
        and plan.image_tag is not None
    ):
        try:
            delete_acr_image_tag_fn(
                plan.registry_name,
                plan.repository_name,
                plan.image_tag,
                managed_identity_resource_id=getattr(
                    client.cred, "azure_user_assigned_identity", None
                ),
            )
            deleted.append(
                f"acr:{plan.registry_name}/{plan.repository_name}:{plan.image_tag}"
            )
        except Exception as exc:
            if _is_not_found_error(exc):
                deleted.append(
                    "acr:"
                    f"{plan.registry_name}/{plan.repository_name}:{plan.image_tag} "
                    "(already missing)"
                )
            else:
                failures.append(
                    "acr:"
                    f"{plan.registry_name}/{plan.repository_name}:{plan.image_tag}: "
                    f"{exc}"
                )

    return CleanupResult(deleted=tuple(deleted), failures=tuple(failures))


def list_acr_repository_tags(
    registry_name: str,
    repository_name: str,
    *,
    managed_identity_resource_id: str | None = None,
    ensure_az_login_with_identity_func: Callable[..., Any] = (
        _default_ensure_az_login_with_identity
    ),
    require_tool_func: Callable[[str], None] = require_tool,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> list[str]:
    ensure_az_login_with_identity_func(
        managed_identity_resource_id=managed_identity_resource_id
    )
    require_tool_func("az")
    command = [
        "az",
        "acr",
        "repository",
        "show-tags",
        "--name",
        registry_name,
        "--repository",
        repository_name,
        "--output",
        "json",
    ]
    result = subprocess_run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "Unknown error"
        if _is_not_found_message(stderr):
            return []
        raise RuntimeError(
            f"Failed to list ACR tags for {registry_name}/{repository_name}: {stderr}"
        )
    parsed = json.loads(result.stdout or "[]")
    if not isinstance(parsed, list):
        raise RuntimeError(
            "Unexpected response while listing ACR tags for "
            f"{registry_name}/{repository_name}."
        )
    return [str(item) for item in parsed]


def delete_acr_image_tag(
    registry_name: str,
    repository_name: str,
    image_tag: str,
    *,
    managed_identity_resource_id: str | None = None,
    ensure_az_login_with_identity_func: Callable[..., Any] = (
        _default_ensure_az_login_with_identity
    ),
    require_tool_func: Callable[[str], None] = require_tool,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> None:
    ensure_az_login_with_identity_func(
        managed_identity_resource_id=managed_identity_resource_id
    )
    require_tool_func("az")
    command = [
        "az",
        "acr",
        "repository",
        "delete",
        "--name",
        registry_name,
        "--image",
        f"{repository_name}:{image_tag}",
        "--yes",
        "--output",
        "none",
    ]
    result = subprocess_run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "Unknown error"
        raise RuntimeError(
            "Failed to delete ACR image "
            f"{registry_name}/{repository_name}:{image_tag}: {stderr} "
            "The managed identity may be authenticated but still lack ACR delete permission."
        )


def ensure_az_login_with_identity(
    *,
    managed_identity_resource_id: str | None = None,
    current_identity: object | str | None,
    not_logged_in_sentinel: object,
    require_tool_func: Callable[[str], None] = require_tool,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> object | str | None:
    # Only short-circuit when the caller has already logged in (i.e. the
    # cache is not the "not yet logged in" sentinel) *and* the cached
    # identity matches the one the caller now wants. Otherwise `None ==
    # None` would silently skip `az login` on the very first call when no
    # managed identity is specified.
    if (
        current_identity is not not_logged_in_sentinel
        and current_identity == managed_identity_resource_id
    ):
        return current_identity

    require_tool_func("az")
    command = ["az", "login", "--identity"]
    if managed_identity_resource_id:
        command.extend(["--resource-id", managed_identity_resource_id])

    result = subprocess_run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return managed_identity_resource_id

    stderr = result.stderr.strip() or "Unknown error"
    raise RuntimeError(
        f"Failed to authenticate Azure CLI with managed identity: {stderr}"
    )


def format_cleanup_plan(plan: CleanupPlan, *, include_acr: bool) -> str:
    lines = [
        "Cloud cleanup plan",
        f"  config: {plan.config_path}",
        f"  session: {plan.session_slug}",
        f"  keyvault: {plan.keyvault}",
        f"  batch jobs: {len(plan.job_names)}",
    ]
    lines.extend(f"    - {name}" for name in plan.job_names)
    lines.append(f"  batch pools: {len(plan.pool_names)}")
    lines.extend(f"    - {name}" for name in plan.pool_names)
    lines.append(f"  blob containers: {len(plan.container_names)}")
    lines.extend(f"    - {name}" for name in plan.container_names)
    if include_acr:
        if plan.image_tag is None:
            lines.append(
                "  acr image tag: skipped (pass --image-tag to delete one tag)"
            )
        else:
            registry = plan.registry_name or "<unconfigured>"
            status = "present" if plan.acr_image_exists else "absent"
            lines.append(
                "  acr image tag: "
                f"{registry}/{plan.repository_name}:{plan.image_tag} ({status})"
            )
    else:
        lines.append("  acr image tag: skipped")
    return "\n".join(lines)


def format_cleanup_listing(
    listing: CleanupListing,
    *,
    include_acr: bool,
) -> str:
    session_slugs_by_tag = _session_slugs_by_image_tag(listing)
    lines = [
        "Cloud cleanup inventory",
        f"  config: {listing.config_path}",
        f"  session filter: {listing.session_slug or '<all sessions>'}",
        f"  image filter: {listing.image_tag or '<all images>'}",
        f"  keyvault: {listing.keyvault}",
        f"  batch jobs: {len(listing.job_names)}",
    ]
    lines.extend(
        _format_listing_entry(
            name,
            _session_slugs_for_job_name(name),
        )
        for name in listing.job_names
    )
    lines.append(f"  batch pools: {len(listing.pool_names)}")
    lines.extend(
        _format_listing_entry(
            name,
            _session_slugs_for_resource_name(name),
        )
        for name in listing.pool_names
    )
    lines.append(f"  blob containers: {len(listing.container_names)}")
    lines.extend(
        _format_listing_entry(
            name,
            _session_slugs_for_resource_name(name),
        )
        for name in listing.container_names
    )
    if include_acr:
        registry = listing.registry_name or "<unconfigured>"
        if listing.acr_image_tags_warning:
            lines.append(
                "  acr image tags: unavailable "
                f"from {registry}/{listing.repository_name}"
            )
            lines.append(
                "    - "
                f"{listing.acr_image_tags_warning}; rerun with --skip-acr to "
                "ignore ACR, or authenticate Azure CLI and retry"
            )
        else:
            lines.append(
                "  acr image tags: "
                f"{len(listing.acr_image_tags)} from {registry}/{listing.repository_name}"
            )
            lines.extend(
                _format_listing_entry(
                    f"{registry}/{listing.repository_name}:{tag}",
                    session_slugs_by_tag.get(tag, ()),
                )
                for tag in listing.acr_image_tags
            )
    else:
        lines.append("  acr image tags: skipped")
    return "\n".join(lines)


def _summarize_error_message(exc: Exception) -> str:
    for line in str(exc).splitlines():
        summary = line.strip()
        if summary:
            return summary
    return exc.__class__.__name__


def _list_pool_names(client) -> list[str]:
    pools = client.batch_mgmt_client.pool.list_by_batch_account(
        resource_group_name=client.cred.azure_resource_group_name,
        account_name=client.cred.azure_batch_account,
    )
    return [pool.name for pool in pools if getattr(pool, "name", None)]


def _list_job_names(client) -> list[str]:
    return [
        job.id
        for job in client.batch_service_client.job.list()
        if getattr(job, "id", None)
    ]


def _list_container_names(client) -> list[str]:
    return [
        container.name
        for container in client.blob_service_client.list_containers()
        if getattr(container, "name", None)
    ]


def _matching_project_resource_names(
    existing_names: list[str], prefixes: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                name
                for name in existing_names
                if any(
                    _matches_project_resource_name(name, prefix)
                    for prefix in prefixes
                )
            }
        )
    )


def _filter_names_for_session_slug(
    names: list[str],
    session_slug: str,
    session_slug_getter: Callable[[str], tuple[str, ...]],
    allowed_prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    """Return names whose session slug matches *and* whose prefix is one of
    the project's configured prefixes.

    Matching on session slug alone is not safe: another tool sharing the same
    Azure resource group (e.g. ``other-project-cloud-<slug>-j1``) can embed the
    same slug suffix. Cleanup must only touch resources that also satisfy the
    project prefix contract.
    """
    return tuple(
        sorted(
            {
                name
                for name in names
                if session_slug in session_slug_getter(name)
                and any(
                    _matches_project_resource_name(name, prefix)
                    for prefix in allowed_prefixes
                )
            }
        )
    )


def _session_slugs_for_job_name(job_name: str) -> tuple[str, ...]:
    match = re.search(r"(?P<session>\d{14}-[a-z0-9-]+?)-j\d+$", job_name)
    session_slug = match.group("session") if match else None
    return (session_slug,) if session_slug else ()


def _session_slugs_for_resource_name(resource_name: str) -> tuple[str, ...]:
    match = re.search(r"(?P<session>\d{14}-[a-z0-9-]+)$", resource_name)
    session_slug = match.group("session") if match else None
    return (session_slug,) if session_slug else ()


def _session_slugs_by_image_tag(
    listing: CleanupListing,
) -> dict[str, tuple[str, ...]]:
    slugs_by_tag: dict[str, set[str]] = {}

    def add_slug(session_slug: str | None) -> None:
        if not session_slug:
            return
        image_tag = parse_image_tag_from_session_slug(session_slug)
        if not image_tag:
            return
        slugs_by_tag.setdefault(image_tag, set()).add(session_slug)

    add_slug(listing.session_slug)
    for name in listing.pool_names:
        for session_slug in _session_slugs_for_resource_name(name):
            add_slug(session_slug)
    for name in listing.job_names:
        for session_slug in _session_slugs_for_job_name(name):
            add_slug(session_slug)
    for name in listing.container_names:
        for session_slug in _session_slugs_for_resource_name(name):
            add_slug(session_slug)

    return {
        image_tag: tuple(sorted(session_slugs))
        for image_tag, session_slugs in slugs_by_tag.items()
    }


def _format_listing_entry(value: str, session_slugs: tuple[str, ...]) -> str:
    if not session_slugs:
        return f"    - {value} (session=<unknown>)"
    if len(session_slugs) == 1:
        return f"    - {value} (session={session_slugs[0]})"
    return f"    - {value} (sessions={', '.join(session_slugs)})"


def _matches_project_resource_name(name: str, prefix: str) -> bool:
    normalized_prefix = sanitize_name(prefix)
    return name == normalized_prefix or name.startswith(
        f"{normalized_prefix}-"
    )


def _is_not_found_message(message: str) -> bool:
    normalized = message.lower()
    return (
        "not found" in normalized
        or "does not exist" in normalized
        or "name unknown" in normalized
    )


def _is_not_found_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 404:
        return True
    return exc.__class__.__name__ == "ResourceNotFoundError" or (
        _is_not_found_message(str(exc))
    )
