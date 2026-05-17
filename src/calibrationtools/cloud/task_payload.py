from __future__ import annotations

import copy
import shutil
import string
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from calibrationtools.json_utils import to_jsonable

from .config import (
    CloudSharedAssetSettings,
    CloudTaskPayloadSettings,
    CloudTaskPayloadTransform,
)
from .json_pointer import (
    JSONPointerError,
    JSONPointerMissingError,
    read_json_pointer,
    set_json_pointer,
)


@dataclass(frozen=True)
class ResolvedSharedAsset:
    name: str
    source_path: Path
    is_dir: bool
    remote_blob_dir: str
    remote_mount_path: str
    remote_path_var: str | None = None


@dataclass(frozen=True)
class CloudTaskContext:
    run_id: str
    session_id: str
    job_name: str
    input_mount_path: str
    output_mount_path: str
    logs_mount_path: str
    task_output_dir: str
    shared_assets: tuple[ResolvedSharedAsset, ...] = ()


def resolve_shared_assets(
    settings: tuple[CloudSharedAssetSettings, ...],
    *,
    base_payload: dict[str, Any] | None,
    config_dir: Path,
) -> tuple[ResolvedSharedAsset, ...]:
    """Resolve shared asset host paths before cloud resources are created."""
    resolved: list[ResolvedSharedAsset] = []
    for asset in settings:
        try:
            source = _resolve_asset_source(asset, base_payload, config_dir)
        except JSONPointerMissingError:
            if asset.required:
                raise
            continue

        if not source.exists():
            if asset.required:
                raise FileNotFoundError(
                    f"cloud.shared_assets.{asset.name}.source_path not found: {source}"
                )
            continue

        if not source.is_file() and not source.is_dir():
            raise FileNotFoundError(
                f"cloud.shared_assets.{asset.name}.source_path must be a file or directory: {source}"
            )

        resolved.append(
            ResolvedSharedAsset(
                name=asset.name,
                source_path=source,
                is_dir=source.is_dir(),
                remote_blob_dir=_join_blob_path(asset.blob_dir, asset.name),
                remote_mount_path="",
                remote_path_var=asset.remote_path_var,
            )
        )
    return tuple(resolved)


def bind_shared_assets_to_session(
    assets: tuple[ResolvedSharedAsset, ...],
    *,
    session_id: str,
    input_mount_path: str,
) -> tuple[ResolvedSharedAsset, ...]:
    """Attach session-scoped blob prefixes and mounted paths to assets."""
    bound: list[ResolvedSharedAsset] = []
    mount = input_mount_path.rstrip("/")
    for asset in assets:
        remote_blob_dir = _join_blob_path(
            asset.remote_blob_dir,
            session_id,
        )
        remote_mount_path = f"{mount}/{remote_blob_dir}"
        if not asset.is_dir:
            remote_mount_path = f"{remote_mount_path}/{asset.source_path.name}"
        bound.append(
            replace(
                asset,
                remote_blob_dir=remote_blob_dir,
                remote_mount_path=remote_mount_path,
            )
        )
    return tuple(bound)


def bind_shared_assets_to_local_root(
    assets: tuple[ResolvedSharedAsset, ...],
    *,
    session_id: str,
    input_root: Path,
) -> tuple[ResolvedSharedAsset, ...]:
    """Attach local probe paths using the same blob prefix shape as cloud."""
    bound: list[ResolvedSharedAsset] = []
    for asset in assets:
        remote_blob_dir = _join_blob_path(asset.remote_blob_dir, session_id)
        remote_mount_path = str(input_root / remote_blob_dir)
        if not asset.is_dir:
            remote_mount_path = str(
                Path(remote_mount_path) / asset.source_path.name
            )
        bound.append(
            replace(
                asset,
                remote_blob_dir=remote_blob_dir,
                remote_mount_path=remote_mount_path,
            )
        )
    return tuple(bound)


def copy_shared_assets_to_local_root(
    assets: tuple[ResolvedSharedAsset, ...],
    *,
    input_root: Path,
) -> None:
    """Stage resolved assets into a local probe input root."""
    for asset in assets:
        destination = input_root / asset.remote_blob_dir
        destination.mkdir(parents=True, exist_ok=True)
        if asset.is_dir:
            _copy_directory_contents(asset.source_path, destination)
        else:
            shutil.copy2(
                asset.source_path, destination / asset.source_path.name
            )


def apply_task_payload_transforms(
    payload: dict[str, Any],
    settings: CloudTaskPayloadSettings,
    context: CloudTaskContext,
) -> dict[str, Any]:
    """Return a transformed JSON-compatible task payload."""
    transformed = copy.deepcopy(to_jsonable(payload))
    for transform in settings.transforms:
        value = render_template_value(transform.value, context)
        try:
            transformed = _apply_one_transform(transformed, transform, value)
        except JSONPointerMissingError:
            if transform.on_missing == "skip":
                continue
            raise
        except JSONPointerError as exc:
            name = f" {transform.name!r}" if transform.name else ""
            raise ValueError(
                f"task payload transform{name} failed at {transform.path!r}: {exc}"
            ) from exc
    return transformed


def resolve_task_output_dir(
    settings: CloudTaskPayloadSettings,
    context: CloudTaskContext,
    *,
    default_task_output_dir: str,
) -> str:
    if settings.task_output_dir is None:
        return default_task_output_dir
    return str(render_template_value(settings.task_output_dir, context))


def validate_task_payload_templates(
    settings: CloudTaskPayloadSettings,
    *,
    shared_assets: tuple[ResolvedSharedAsset, ...] = (),
) -> None:
    """Reject unknown template variables before cloud resources are created."""
    allowed = _template_values(
        CloudTaskContext(
            run_id="run",
            session_id="session",
            job_name="job",
            input_mount_path="/input",
            output_mount_path="/output",
            logs_mount_path="/logs",
            task_output_dir="/output/run",
            shared_assets=shared_assets,
        )
    )
    for template in _iter_templates(settings):
        validate_template_string(template, allowed_variables=set(allowed))


def render_template_value(value: Any, context: CloudTaskContext) -> Any:
    if isinstance(value, str):
        return render_template_string(value, context)
    if isinstance(value, list):
        return [render_template_value(item, context) for item in value]
    if isinstance(value, tuple):
        return tuple(render_template_value(item, context) for item in value)
    if isinstance(value, dict):
        return {
            key: render_template_value(item, context)
            for key, item in value.items()
        }
    return value


def render_template_string(template: str, context: CloudTaskContext) -> str:
    values = _template_values(context)
    return _render_template_with_values(template, values)


def validate_template_string(
    template: str,
    *,
    allowed_variables: set[str],
) -> None:
    formatter = string.Formatter()
    for _, field_name, format_spec, conversion in formatter.parse(template):
        if field_name is None:
            continue
        if format_spec:
            raise ValueError(
                f"template field {field_name!r} must not use a format specifier"
            )
        if conversion:
            raise ValueError(
                f"template field {field_name!r} must not use a conversion"
            )
        if field_name not in allowed_variables:
            raise ValueError(f"unknown template variable {field_name!r}")


def _resolve_asset_source(
    asset: CloudSharedAssetSettings,
    base_payload: dict[str, Any] | None,
    config_dir: Path,
) -> Path:
    if asset.source_path is not None:
        source = asset.source_path
    else:
        if base_payload is None:
            raise ValueError(
                f"cloud.shared_assets.{asset.name}.source_json_pointer requires base inputs"
            )
        assert asset.source_json_pointer is not None
        source_value = read_json_pointer(
            base_payload, asset.source_json_pointer
        )
        if not isinstance(source_value, str) or not source_value:
            raise ValueError(
                f"cloud.shared_assets.{asset.name}.source_json_pointer must resolve to a non-empty path string"
            )
        source = Path(source_value)

    if not source.is_absolute():
        source = config_dir / source
    return source.resolve()


def _apply_one_transform(
    payload: dict[str, Any],
    transform: CloudTaskPayloadTransform,
    value: Any,
) -> dict[str, Any]:
    if transform.op != "set":
        raise ValueError(
            f"unsupported task payload transform op {transform.op!r}"
        )
    create_missing = transform.on_missing == "create"
    new_payload = set_json_pointer(
        payload,
        transform.path,
        value,
        create_missing=create_missing,
    )
    if not isinstance(new_payload, dict):
        raise ValueError("task payload transforms must produce a JSON object")
    return new_payload


def _iter_templates(settings: CloudTaskPayloadSettings) -> list[str]:
    templates: list[str] = []
    if settings.task_output_dir is not None:
        templates.append(settings.task_output_dir)
    for transform in settings.transforms:
        templates.extend(_collect_string_values(transform.value))
    return templates


def _collect_string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        found: list[str] = []
        for item in value.values():
            found.extend(_collect_string_values(item))
        return found
    if isinstance(value, (list, tuple)):
        found: list[str] = []
        for item in value:
            found.extend(_collect_string_values(item))
        return found
    return []


def _template_values(context: CloudTaskContext) -> dict[str, str]:
    values = {
        "run_id": context.run_id,
        "session_id": context.session_id,
        "job_name": context.job_name,
        "input_mount_path": context.input_mount_path,
        "output_mount_path": context.output_mount_path,
        "logs_mount_path": context.logs_mount_path,
        "task_output_dir": context.task_output_dir,
    }
    for asset in context.shared_assets:
        values[f"shared_assets.{asset.name}.remote_path"] = (
            asset.remote_mount_path
        )
        if asset.remote_path_var:
            values[asset.remote_path_var] = asset.remote_mount_path
    return values


def _render_template_with_values(
    template: str,
    values: dict[str, str],
) -> str:
    validate_template_string(template, allowed_variables=set(values))
    rendered: list[str] = []
    formatter = string.Formatter()
    for literal_text, field_name, _, _ in formatter.parse(template):
        rendered.append(literal_text)
        if field_name is not None:
            rendered.append(str(values[field_name]))
    return "".join(rendered)


def _join_blob_path(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def _copy_directory_contents(source: Path, destination: Path) -> None:
    for child in source.rglob("*"):
        if child.is_dir():
            continue
        relative = child.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child, target)
