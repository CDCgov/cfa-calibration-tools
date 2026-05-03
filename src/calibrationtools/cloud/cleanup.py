from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import CloudRuntimeSettings, load_cloud_model_config
from .naming import (
    parse_image_tag_from_session_id,
    parse_username_from_session_id,
    sanitize_name,
)
from .tooling import (
    _default_ensure_az_login_with_identity,
    create_cloud_client,
    ensure_az_login_with_identity,  # noqa: F401
    require_tool,
)


@dataclass(frozen=True)
class CleanupPlan:
    config_path: Path
    session_id: str
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
            "session_id": self.session_id,
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
    session_id: str | None
    user: str | None
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
            "scope, or pass --session-id to inspect or delete one session."
        )
    )
    parser.add_argument(
        "--config",
        "--cloud-config",
        dest="config",
        type=Path,
        default=default_config_path,
        help=(
            "Cloud config used to derive the project-specific Batch, Blob, "
            "and ACR naming scope. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--session-id",
        help=(
            "Exact session ID to inspect or clean up. This is printed by "
            "the cloud runner and embedded in the Batch, Blob, and log paths."
        ),
    )
    parser.add_argument(
        "--user",
        help=(
            "Filter sessions to the sanitized username segment embedded in "
            "the session ID."
        ),
    )
    parser.add_argument(
        "--all-sessions-for-user",
        action="store_true",
        help="Delete every project-scoped session for --user.",
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
            "resources; with --session-id, --user, and/or --image-tag, "
            "narrow the output."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Print the cleanup plan without deleting any Azure resources."),
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
    if args.list:
        return args
    if args.session_id and args.all_sessions_for_user:
        raise SystemExit(
            "--session-id and --all-sessions-for-user are mutually exclusive."
        )
    if args.all_sessions_for_user:
        if not args.user:
            raise SystemExit("--all-sessions-for-user requires --user.")
        return args
    if args.user and not args.session_id:
        raise SystemExit(
            "Deleting all sessions for a user requires "
            "--all-sessions-for-user --user."
        )
    if not args.session_id:
        raise SystemExit(
            "Cleanup requires --session-id or --all-sessions-for-user --user."
        )
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the shared cloud cleanup command-line interface."""
    args = parse_args(argv, default_config_path=Path("cloud_config.toml"))
    config_path = Path(args.config)
    model_config = load_cloud_model_config(config_path)
    settings = model_config.runtime_settings
    client = create_cloud_client(keyvault=settings.keyvault)
    include_acr = not args.skip_acr

    if args.list:
        listing = discover_cleanup_listing(
            client,
            settings,
            config_path=config_path,
            session_id=args.session_id,
            user=args.user,
            image_tag=args.image_tag,
            include_acr=include_acr,
            allow_acr_errors=True,
        )
        print(format_cleanup_listing(listing, include_acr=include_acr))
        return 0

    if args.all_sessions_for_user:
        plans = discover_cleanup_plans_for_user(
            client,
            settings,
            config_path=config_path,
            user=args.user,
            image_tag=args.image_tag,
            include_acr=include_acr,
        )
        print(format_cleanup_plans(plans, include_acr=include_acr))

        if args.dry_run:
            print(
                "\nDry run only. Re-run without --dry-run to delete the "
                "resources above."
            )
            return 0

        result = execute_cleanup_plans(
            client,
            plans,
            include_acr=include_acr,
        )
        _print_cleanup_result(result)
        return 0 if result.ok else 1

    plan = discover_cleanup_plan(
        client,
        settings,
        config_path=config_path,
        session_id=args.session_id,
        image_tag=args.image_tag,
        include_acr=include_acr,
    )
    print(format_cleanup_plan(plan, include_acr=include_acr))

    if args.dry_run:
        print(
            "\nDry run only. Re-run without --dry-run to delete the "
            "resources above."
        )
        return 0

    if plan.is_empty:
        print("\nNo matching Azure resources were found.")
        return 0

    result = execute_cleanup(client, plan, include_acr=include_acr)
    _print_cleanup_result(result)
    return 0 if result.ok else 1


def _print_cleanup_result(result: CleanupResult) -> None:
    print("\nDeleted resources:")
    if result.deleted:
        for item in result.deleted:
            print(f"  - {item}")
    else:
        print("  - none")

    if result.failures:
        print("\nCleanup finished with errors:", file=sys.stderr)
        for failure in result.failures:
            print(f"  - {failure}", file=sys.stderr)
        return

    print("\nCleanup finished successfully.")


def discover_cleanup_listing(
    client,
    settings: CloudRuntimeSettings,
    *,
    config_path: Path,
    session_id: str | None,
    user: str | None,
    image_tag: str | None,
    include_acr: bool,
    allow_acr_errors: bool = False,
    list_acr_repository_tags_fn: Callable[..., list[str]] | None = None,
) -> CleanupListing:
    if list_acr_repository_tags_fn is None:
        list_acr_repository_tags_fn = list_acr_repository_tags

    cred = client.cred
    _require_cleanup_listing_credentials(cred)
    job_names, pool_names, container_names = _discover_cleanup_batch_resources(
        client,
        settings,
        session_id=session_id,
        user=user,
    )

    registry_name = getattr(cred, "azure_container_registry_account", None)
    managed_identity_resource_id = getattr(
        cred, "azure_user_assigned_identity", None
    )
    discovered_tags, acr_image_tags_warning = _discover_cleanup_acr_tags(
        registry_name=registry_name,
        repository_name=settings.repository,
        managed_identity_resource_id=managed_identity_resource_id,
        include_acr=include_acr,
        allow_acr_errors=allow_acr_errors,
        list_acr_repository_tags_fn=list_acr_repository_tags_fn,
    )
    acr_image_tags = _filter_acr_tags(discovered_tags, image_tag)

    return CleanupListing(
        config_path=config_path,
        session_id=session_id,
        user=user,
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


def _require_cleanup_listing_credentials(cred: Any) -> None:
    if not cred.azure_resource_group_name:
        raise SystemExit("AZURE_RESOURCE_GROUP_NAME must be configured.")
    if not cred.azure_batch_account:
        raise SystemExit("AZURE_BATCH_ACCOUNT must be configured.")


def _discover_cleanup_batch_resources(
    client: Any,
    settings: CloudRuntimeSettings,
    *,
    session_id: str | None,
    user: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    pool_prefixes = (settings.pool_prefix,)
    job_prefixes = (settings.job_prefix,)
    container_prefixes = (
        settings.input_container_prefix,
        settings.output_container_prefix,
        settings.logs_container_prefix,
    )

    job_names = _matching_project_resource_names(
        _list_job_names(client), job_prefixes
    )
    pool_names = _matching_project_resource_names(
        _list_pool_names(client),
        pool_prefixes,
    )
    container_names = _matching_project_resource_names(
        _list_container_names(client),
        container_prefixes,
    )

    if session_id is not None:
        job_names = _filter_names_for_session_id(
            list(job_names),
            session_id,
            _session_ids_for_job_name,
            job_prefixes,
        )
        pool_names = _filter_names_for_session_id(
            list(pool_names),
            session_id,
            _session_ids_for_resource_name,
            pool_prefixes,
        )
        container_names = _filter_names_for_session_id(
            list(container_names),
            session_id,
            _session_ids_for_resource_name,
            container_prefixes,
        )

    if user is not None:
        normalized_user = sanitize_name(user)
        job_names = _filter_names_for_session_user(
            list(job_names),
            normalized_user,
            _session_ids_for_job_name,
        )
        pool_names = _filter_names_for_session_user(
            list(pool_names),
            normalized_user,
            _session_ids_for_resource_name,
        )
        container_names = _filter_names_for_session_user(
            list(container_names),
            normalized_user,
            _session_ids_for_resource_name,
        )

    return (
        tuple(job_names),
        tuple(pool_names),
        tuple(container_names),
    )


def _discover_cleanup_acr_tags(
    *,
    registry_name: str | None,
    repository_name: str,
    managed_identity_resource_id: str | None,
    include_acr: bool,
    allow_acr_errors: bool,
    list_acr_repository_tags_fn: Callable[..., list[str]],
) -> tuple[tuple[str, ...], str | None]:
    if not include_acr:
        return (), None
    if not registry_name:
        message = "AZURE_CONTAINER_REGISTRY_ACCOUNT must be configured."
        if allow_acr_errors:
            return (), message
        raise SystemExit(message)

    try:
        return (
            tuple(
                sorted(
                    list_acr_repository_tags_fn(
                        registry_name,
                        repository_name,
                        managed_identity_resource_id=(
                            managed_identity_resource_id
                        ),
                    )
                )
            ),
            None,
        )
    except Exception as exc:
        if not allow_acr_errors:
            raise
        return (), _summarize_error_message(exc)


def _filter_acr_tags(
    discovered_tags: tuple[str, ...],
    image_tag: str | None,
) -> tuple[str, ...]:
    if image_tag is None:
        return discovered_tags
    return tuple(tag for tag in discovered_tags if tag == image_tag)


def discover_cleanup_plan(
    client,
    settings: CloudRuntimeSettings,
    *,
    config_path: Path,
    session_id: str,
    image_tag: str | None,
    include_acr: bool,
    list_acr_repository_tags_fn: Callable[..., list[str]] | None = None,
) -> CleanupPlan:
    listing = discover_cleanup_listing(
        client,
        settings,
        config_path=config_path,
        session_id=session_id,
        user=None,
        image_tag=image_tag,
        include_acr=include_acr and image_tag is not None,
        list_acr_repository_tags_fn=list_acr_repository_tags_fn,
    )

    return CleanupPlan(
        config_path=config_path,
        session_id=session_id,
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


def discover_cleanup_plans_for_user(
    client,
    settings: CloudRuntimeSettings,
    *,
    config_path: Path,
    user: str,
    image_tag: str | None,
    include_acr: bool,
    list_acr_repository_tags_fn: Callable[..., list[str]] | None = None,
) -> tuple[CleanupPlan, ...]:
    listing = discover_cleanup_listing(
        client,
        settings,
        config_path=config_path,
        session_id=None,
        user=user,
        image_tag=image_tag,
        include_acr=include_acr and image_tag is not None,
        list_acr_repository_tags_fn=list_acr_repository_tags_fn,
    )
    session_ids = _session_ids_from_listing(listing)
    acr_image_exists = (
        image_tag is not None and image_tag in listing.acr_image_tags
    )

    plans: list[CleanupPlan] = []
    for index, session_id in enumerate(session_ids):
        plans.append(
            CleanupPlan(
                config_path=config_path,
                session_id=session_id,
                keyvault=listing.keyvault,
                registry_name=listing.registry_name,
                repository_name=listing.repository_name,
                image_tag=image_tag,
                job_names=_names_for_session_id(
                    listing.job_names,
                    session_id,
                    _session_ids_for_job_name,
                ),
                pool_names=_names_for_session_id(
                    listing.pool_names,
                    session_id,
                    _session_ids_for_resource_name,
                ),
                container_names=_names_for_session_id(
                    listing.container_names,
                    session_id,
                    _session_ids_for_resource_name,
                ),
                acr_image_exists=acr_image_exists and index == 0,
            )
        )
    return tuple(plans)


def execute_cleanup(
    client,
    plan: CleanupPlan,
    *,
    include_acr: bool,
    delete_acr_image_tag_fn: Callable[..., None] | None = None,
) -> CleanupResult:
    if delete_acr_image_tag_fn is None:
        delete_acr_image_tag_fn = delete_acr_image_tag

    if not plan.session_id and (
        plan.job_names or plan.pool_names or plan.container_names
    ):
        raise ValueError(
            "Cleanup requires a session_id to delete Batch or Blob resources."
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


def execute_cleanup_plans(
    client,
    plans: tuple[CleanupPlan, ...],
    *,
    include_acr: bool,
    delete_acr_image_tag_fn: Callable[..., None] | None = None,
) -> CleanupResult:
    deleted: list[str] = []
    failures: list[str] = []
    for plan in plans:
        result = execute_cleanup(
            client,
            plan,
            include_acr=include_acr,
            delete_acr_image_tag_fn=delete_acr_image_tag_fn,
        )
        deleted.extend(result.deleted)
        failures.extend(result.failures)
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


def format_cleanup_plan(plan: CleanupPlan, *, include_acr: bool) -> str:
    lines = [
        "Cloud cleanup plan",
        f"  config: {plan.config_path}",
        f"  session: {plan.session_id}",
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


def format_cleanup_plans(
    plans: tuple[CleanupPlan, ...],
    *,
    include_acr: bool,
) -> str:
    if not plans:
        return "Cloud cleanup plan\n  sessions: 0"

    lines = [
        "Cloud cleanup plan",
        f"  sessions: {len(plans)}",
    ]
    for plan in plans:
        lines.append("")
        lines.append(format_cleanup_plan(plan, include_acr=include_acr))
    return "\n".join(lines)


def format_cleanup_listing(
    listing: CleanupListing,
    *,
    include_acr: bool,
) -> str:
    session_ids_by_tag = _session_ids_by_image_tag(listing)
    lines = [
        "Cloud cleanup inventory",
        f"  config: {listing.config_path}",
        f"  session filter: {listing.session_id or '<all sessions>'}",
        f"  user filter: {listing.user or '<all users>'}",
        f"  image filter: {listing.image_tag or '<all images>'}",
        f"  keyvault: {listing.keyvault}",
        f"  batch jobs: {len(listing.job_names)}",
    ]
    lines.extend(
        _format_listing_entry(
            name,
            _session_ids_for_job_name(name),
        )
        for name in listing.job_names
    )
    lines.append(f"  batch pools: {len(listing.pool_names)}")
    lines.extend(
        _format_listing_entry(
            name,
            _session_ids_for_resource_name(name),
        )
        for name in listing.pool_names
    )
    lines.append(f"  blob containers: {len(listing.container_names)}")
    lines.extend(
        _format_listing_entry(
            name,
            _session_ids_for_resource_name(name),
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
                    session_ids_by_tag.get(tag, ()),
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


def _filter_names_for_session_id(
    names: list[str],
    session_id: str,
    session_id_getter: Callable[[str], tuple[str, ...]],
    allowed_prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    """Return names whose session ID matches *and* whose prefix is one of
    the project's configured prefixes.

    Matching on session ID alone is not safe: another tool sharing the same
    Azure resource group (e.g. ``other-project-cloud-<id>-j1``) can embed the
    same ID suffix. Cleanup must only touch resources that also satisfy the
    project prefix contract.
    """
    return tuple(
        sorted(
            {
                name
                for name in names
                if session_id in session_id_getter(name)
                and any(
                    _matches_project_resource_name(name, prefix)
                    for prefix in allowed_prefixes
                )
            }
        )
    )


def _filter_names_for_session_user(
    names: list[str],
    user: str,
    session_id_getter: Callable[[str], tuple[str, ...]],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                name
                for name in names
                if any(
                    parse_username_from_session_id(session_id) == user
                    for session_id in session_id_getter(name)
                )
            }
        )
    )


def _session_ids_from_listing(listing: CleanupListing) -> tuple[str, ...]:
    session_ids: set[str] = set()
    for name in listing.pool_names:
        session_ids.update(_session_ids_for_resource_name(name))
    for name in listing.job_names:
        session_ids.update(_session_ids_for_job_name(name))
    for name in listing.container_names:
        session_ids.update(_session_ids_for_resource_name(name))
    return tuple(sorted(session_ids))


def _names_for_session_id(
    names: tuple[str, ...],
    session_id: str,
    session_id_getter: Callable[[str], tuple[str, ...]],
) -> tuple[str, ...]:
    return tuple(
        name for name in names if session_id in session_id_getter(name)
    )


def _session_ids_for_job_name(job_name: str) -> tuple[str, ...]:
    match = re.search(
        r"(?P<session>\d{14}-[a-z0-9-]+?-[a-z0-9]+-[0-9a-f]{12})-j\d+$",
        job_name,
    )
    session_id = match.group("session") if match else None
    return (session_id,) if session_id else ()


def _session_ids_for_resource_name(resource_name: str) -> tuple[str, ...]:
    match = re.search(
        r"(?P<session>\d{14}-[a-z0-9-]+?-[a-z0-9]+-[0-9a-f]{12})$",
        resource_name,
    )
    session_id = match.group("session") if match else None
    return (session_id,) if session_id else ()


def _session_ids_by_image_tag(
    listing: CleanupListing,
) -> dict[str, tuple[str, ...]]:
    ids_by_tag: dict[str, set[str]] = {}

    def add_slug(session_id: str | None) -> None:
        if not session_id:
            return
        image_tag = parse_image_tag_from_session_id(session_id)
        if not image_tag:
            return
        ids_by_tag.setdefault(image_tag, set()).add(session_id)

    add_slug(listing.session_id)
    for name in listing.pool_names:
        for session_id in _session_ids_for_resource_name(name):
            add_slug(session_id)
    for name in listing.job_names:
        for session_id in _session_ids_for_job_name(name):
            add_slug(session_id)
    for name in listing.container_names:
        for session_id in _session_ids_for_resource_name(name):
            add_slug(session_id)

    return {
        image_tag: tuple(sorted(session_ids))
        for image_tag, session_ids in ids_by_tag.items()
    }


def _format_listing_entry(value: str, session_ids: tuple[str, ...]) -> str:
    if not session_ids:
        return f"    - {value} (session=<unknown>)"
    if len(session_ids) == 1:
        return f"    - {value} (session={session_ids[0]})"
    return f"    - {value} (sessions={', '.join(session_ids)})"


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


if __name__ == "__main__":
    raise SystemExit(main())
