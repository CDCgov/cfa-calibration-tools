from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Literal, cast

from .json_pointer import parse_json_pointer

DEFAULT_INPUT_MOUNT_PATH = "/cloud-input"
DEFAULT_OUTPUT_MOUNT_PATH = "/cloud-output"
DEFAULT_LOGS_MOUNT_PATH = "/cloud-logs"
DEFAULT_VM_SIZE = "large"
DEFAULT_JOBS_PER_SESSION = 1
DEFAULT_TASK_SLOTS_PER_NODE = 1
DEFAULT_POOL_MAX_NODES = 5
DEFAULT_TASK_TIMEOUT_MINUTES = 60
DEFAULT_POOL_READY_TIMEOUT_MINUTES = 20
DEFAULT_POOL_AUTO_SCALE_EVALUATION_INTERVAL_MINUTES = 5
DEFAULT_DISPATCH_BUFFER = 0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_MAX_PARALLEL_OUTPUT_DOWNLOADS = 8


@dataclass(frozen=True)
class CloudRuntimeSettings:
    keyvault: str
    local_image: str
    repository: str
    task_mrp_config_path: str
    pool_prefix: str
    job_prefix: str
    input_container_prefix: str
    output_container_prefix: str
    logs_container_prefix: str
    input_mount_path: str = DEFAULT_INPUT_MOUNT_PATH
    output_mount_path: str = DEFAULT_OUTPUT_MOUNT_PATH
    logs_mount_path: str = DEFAULT_LOGS_MOUNT_PATH
    vm_size: str = DEFAULT_VM_SIZE
    jobs_per_session: int = DEFAULT_JOBS_PER_SESSION
    task_slots_per_node: int = DEFAULT_TASK_SLOTS_PER_NODE
    pool_max_nodes: int = DEFAULT_POOL_MAX_NODES
    task_timeout_minutes: int | None = DEFAULT_TASK_TIMEOUT_MINUTES
    pool_ready_timeout_minutes: int | None = DEFAULT_POOL_READY_TIMEOUT_MINUTES
    pool_auto_scale_evaluation_interval_minutes: int = (
        DEFAULT_POOL_AUTO_SCALE_EVALUATION_INTERVAL_MINUTES
    )
    dispatch_buffer: int = DEFAULT_DISPATCH_BUFFER
    max_parallel_output_downloads: int = DEFAULT_MAX_PARALLEL_OUTPUT_DOWNLOADS
    print_task_durations: bool = False

    def __post_init__(self) -> None:
        """Validate low-level runtime settings before cloud resources exist."""
        if self.jobs_per_session < 1:
            raise ValueError("jobs_per_session must be at least 1")
        if self.task_slots_per_node < 1:
            raise ValueError("task_slots_per_node must be at least 1")
        if self.pool_max_nodes < 1:
            raise ValueError("pool_max_nodes must be at least 1")
        if self.pool_auto_scale_evaluation_interval_minutes < 5:
            raise ValueError(
                "pool_auto_scale_evaluation_interval_minutes must be at least 5"
            )
        if self.dispatch_buffer < 0:
            raise ValueError("dispatch_buffer must be at least 0")
        if self.max_parallel_output_downloads < 1:
            raise ValueError(
                "max_parallel_output_downloads must be at least 1"
            )


class CloudCSVValueType(StrEnum):
    """Supported scalar parsers for CSV-backed cloud model outputs."""

    INT = "int"
    FLOAT = "float"
    STR = "str"

    def parser(self) -> Callable[[str], Any]:
        """Return the Python callable used to parse one CSV field."""
        if self is CloudCSVValueType.INT:
            return int
        if self is CloudCSVValueType.FLOAT:
            return float
        return str


class CloudOutputMode(StrEnum):
    CSV_COLUMN = "csv_column"
    CSV_TABLE = "csv_table"


class CSVTableOrientation(StrEnum):
    COLUMNS = "columns"


@dataclass(frozen=True)
class CloudOutputSettings:
    """Describe the shared output contract for a cloud model."""

    filename: str
    mode: CloudOutputMode = CloudOutputMode.CSV_COLUMN
    csv_value_column: str | None = None
    csv_value_type: CloudCSVValueType | None = CloudCSVValueType.INT
    output_name: str | None = None
    orientation: CSVTableOrientation | None = None
    header_fields: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.mode is CloudOutputMode.CSV_COLUMN:
            if not self.csv_value_column:
                raise ValueError(
                    "cloud.output.csv_value_column is required when "
                    "cloud.output.mode = 'csv_column'"
                )
            return

        if self.mode is CloudOutputMode.CSV_TABLE:
            if not self.output_name:
                raise ValueError(
                    "cloud.output.output_name is required when "
                    "cloud.output.mode = 'csv_table'"
                )
            if self.csv_value_column is not None:
                raise ValueError(
                    "cloud.output.csv_value_column is not used when "
                    "cloud.output.mode = 'csv_table'"
                )
            if self.orientation is None:
                raise ValueError(
                    "cloud.output.orientation is required when "
                    "cloud.output.mode = 'csv_table'"
                )
            return

        raise ValueError(f"unsupported cloud.output.mode: {self.mode!r}")


@dataclass(frozen=True)
class CloudSharedAssetSettings:
    name: str
    source_path: Path | None
    source_json_pointer: str | None
    blob_dir: str
    required: bool = True
    remote_path_var: str | None = None

    def __post_init__(self) -> None:
        selector_count = int(self.source_path is not None) + int(
            self.source_json_pointer is not None
        )
        if selector_count != 1:
            raise ValueError(
                f"cloud.shared_assets.{self.name} requires exactly one of "
                "source_path or source_json_pointer"
            )
        if not self.name:
            raise ValueError("cloud.shared_assets.name is required")
        if not self.blob_dir:
            raise ValueError(
                f"cloud.shared_assets.{self.name}.blob_dir is required"
            )
        if self.source_json_pointer is not None:
            parse_json_pointer(self.source_json_pointer)


class CloudPayloadTransformOp(StrEnum):
    SET = "set"


@dataclass(frozen=True)
class CloudTaskPayloadTransform:
    name: str | None
    op: CloudPayloadTransformOp | Literal["set"]
    path: str
    value: Any
    on_missing: Literal["error", "skip", "create"] = "error"

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("cloud.task_payload.transforms.path is required")
        parse_json_pointer(self.path)
        if self.op != CloudPayloadTransformOp.SET and self.op != "set":
            raise ValueError(
                "cloud.task_payload.transforms.op must be one of: set"
            )
        if self.on_missing not in {"error", "skip", "create"}:
            raise ValueError(
                "cloud.task_payload.transforms.on_missing must be one of: "
                "error, skip, create"
            )


@dataclass(frozen=True)
class CloudTaskPayloadSettings:
    task_output_dir: str | None = None
    transforms: tuple[CloudTaskPayloadTransform, ...] = ()


class CloudAutoSizeMemoryScope(StrEnum):
    SELF = "self"
    PROCESS_TREE = "process_tree"


@dataclass(frozen=True)
class CloudAutoSizeSettings:
    """Describe optional cloud auto-size probe configuration."""

    probe: str | None = None
    local_mrp_config_path: Path | None = None
    probe_module: str | None = None
    memory_scope: CloudAutoSizeMemoryScope = CloudAutoSizeMemoryScope.SELF


@dataclass(frozen=True)
class CloudModelConfig:
    """Resolved model-facing cloud configuration.

    The config keeps build-time model integration details beside the
    low-level runtime settings used by Azure session orchestration.
    """

    config_path: Path
    build_context: Path
    dockerfile: Path
    runtime_settings: CloudRuntimeSettings
    output: CloudOutputSettings
    auto_size: CloudAutoSizeSettings
    shared_assets: tuple[CloudSharedAssetSettings, ...] = ()
    task_payload: CloudTaskPayloadSettings = CloudTaskPayloadSettings()


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    """Return a TOML table or raise a targeted validation error."""
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a TOML table")
    return value


def _required_str(table: dict[str, Any], key: str, context: str) -> str:
    """Read a required non-empty string from a parsed TOML table."""
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} is required")
    return value


def _optional_str(
    table: dict[str, Any],
    key: str,
    context: str,
) -> str | None:
    if key not in table:
        return None
    value = table[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _optional_str_tuple(
    table: dict[str, Any],
    key: str,
    context: str,
) -> tuple[str, ...] | None:
    if key not in table:
        return None
    value = table[key]
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{context}.{key} must be a non-empty string array")
    return tuple(value)


def _optional_int(
    table: dict[str, Any],
    key: str,
    default: int | None,
) -> int | None:
    """Read an optional integer from TOML using a Python default."""
    if key not in table:
        return default
    value = table[key]
    if value is None:
        return None
    return int(value)


def _resolve_dir(config_dir: Path, value: str | Path) -> Path:
    """Resolve a config-relative directory and require that it exists."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = config_dir / candidate
    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"cloud.image.build_context not found: {candidate}"
        )
    return candidate


def _resolve_existing_file(
    *,
    config_dir: Path,
    build_context: Path,
    value: str | Path,
    field_name: str,
) -> Path:
    """Resolve a file relative to build context, then config directory."""
    raw_path = Path(value)
    candidates = (
        (raw_path,)
        if raw_path.is_absolute()
        else (build_context / raw_path, config_dir / raw_path)
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    searched = ", ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(f"{field_name} not found. Looked at: {searched}")


def _load_model_cloud_runtime_settings(
    cloud: dict[str, Any],
) -> CloudRuntimeSettings:
    """Build runtime settings from the new model-facing ``[cloud]`` table."""
    image = _require_mapping(cloud.get("image"), "cloud.image")
    resources = _require_mapping(cloud.get("resources"), "cloud.resources")
    return CloudRuntimeSettings(
        keyvault=_required_str(cloud, "keyvault", "cloud"),
        local_image=_required_str(image, "local_image", "cloud.image"),
        repository=_required_str(image, "repository", "cloud.image"),
        task_mrp_config_path=_required_str(
            image,
            "task_mrp_config_path",
            "cloud.image",
        ),
        pool_prefix=_required_str(
            resources,
            "pool_prefix",
            "cloud.resources",
        ),
        job_prefix=_required_str(resources, "job_prefix", "cloud.resources"),
        input_container_prefix=_required_str(
            resources,
            "input_container_prefix",
            "cloud.resources",
        ),
        output_container_prefix=_required_str(
            resources,
            "output_container_prefix",
            "cloud.resources",
        ),
        logs_container_prefix=_required_str(
            resources,
            "logs_container_prefix",
            "cloud.resources",
        ),
        input_mount_path=str(
            resources.get("input_mount_path", DEFAULT_INPUT_MOUNT_PATH)
        ),
        output_mount_path=str(
            resources.get("output_mount_path", DEFAULT_OUTPUT_MOUNT_PATH)
        ),
        logs_mount_path=str(
            resources.get("logs_mount_path", DEFAULT_LOGS_MOUNT_PATH)
        ),
        vm_size=str(cloud.get("vm_size", DEFAULT_VM_SIZE)),
        jobs_per_session=int(
            cloud.get("jobs_per_session", DEFAULT_JOBS_PER_SESSION)
        ),
        task_slots_per_node=int(
            cloud.get("task_slots_per_node", DEFAULT_TASK_SLOTS_PER_NODE)
        ),
        pool_max_nodes=int(
            cloud.get("pool_max_nodes", DEFAULT_POOL_MAX_NODES)
        ),
        task_timeout_minutes=_optional_int(
            cloud,
            "task_timeout_minutes",
            DEFAULT_TASK_TIMEOUT_MINUTES,
        ),
        pool_ready_timeout_minutes=_optional_int(
            cloud,
            "pool_ready_timeout_minutes",
            DEFAULT_POOL_READY_TIMEOUT_MINUTES,
        ),
        pool_auto_scale_evaluation_interval_minutes=int(
            cloud.get(
                "pool_auto_scale_evaluation_interval_minutes",
                DEFAULT_POOL_AUTO_SCALE_EVALUATION_INTERVAL_MINUTES,
            )
        ),
        dispatch_buffer=int(
            cloud.get("dispatch_buffer", DEFAULT_DISPATCH_BUFFER)
        ),
        max_parallel_output_downloads=int(
            cloud.get(
                "max_parallel_output_downloads",
                DEFAULT_MAX_PARALLEL_OUTPUT_DOWNLOADS,
            )
        ),
        print_task_durations=bool(cloud.get("print_task_durations", False)),
    )


def _load_output_settings(cloud: dict[str, Any]) -> CloudOutputSettings:
    """Read the output contract from the top-level ``[cloud]`` table."""
    output = cloud.get("output", {})
    output = _require_mapping(output, "cloud.output")
    mode_value = str(output.get("mode", CloudOutputMode.CSV_COLUMN.value))
    try:
        mode = CloudOutputMode(mode_value)
    except ValueError as exc:
        raise ValueError(
            "cloud.output.mode must be one of: csv_column, csv_table"
        ) from exc

    csv_value_type = str(output.get("csv_value_type", "int"))
    try:
        value_type = CloudCSVValueType(csv_value_type)
    except ValueError as exc:
        raise ValueError(
            "cloud.output.csv_value_type must be one of: int, float, str"
        ) from exc
    orientation = None
    if mode is CloudOutputMode.CSV_TABLE:
        orientation_value = str(
            output.get("orientation", CSVTableOrientation.COLUMNS.value)
        )
        try:
            orientation = CSVTableOrientation(orientation_value)
        except ValueError as exc:
            raise ValueError(
                "cloud.output.orientation must be one of: columns"
            ) from exc

    return CloudOutputSettings(
        filename=_required_str(output, "filename", "cloud.output"),
        mode=mode,
        csv_value_column=(
            _optional_str(output, "csv_value_column", "cloud.output")
            if mode is CloudOutputMode.CSV_TABLE
            else _required_str(output, "csv_value_column", "cloud.output")
        ),
        csv_value_type=value_type
        if mode is CloudOutputMode.CSV_COLUMN
        else None,
        output_name=_optional_str(output, "output_name", "cloud.output"),
        orientation=orientation,
        header_fields=_optional_str_tuple(
            output,
            "header_fields",
            "cloud.output",
        ),
    )


def _load_shared_asset_settings(
    cloud: dict[str, Any],
) -> tuple[CloudSharedAssetSettings, ...]:
    raw_assets = cloud.get("shared_assets")
    if raw_assets is None:
        return ()

    asset_tables: list[tuple[str | None, dict[str, Any]]] = []
    if isinstance(raw_assets, list):
        for item in raw_assets:
            asset_tables.append(
                (None, _require_mapping(item, "cloud.shared_assets"))
            )
    else:
        raw_mapping = _require_mapping(raw_assets, "cloud.shared_assets")
        if "name" in raw_mapping:
            asset_tables.append((None, raw_mapping))
        else:
            for name, table in raw_mapping.items():
                asset_tables.append(
                    (
                        str(name),
                        _require_mapping(table, f"cloud.shared_assets.{name}"),
                    )
                )

    assets: list[CloudSharedAssetSettings] = []
    for name_from_table, table in asset_tables:
        name = _optional_str(table, "name", "cloud.shared_assets")
        if name is None:
            if name_from_table is None:
                raise ValueError("cloud.shared_assets.name is required")
            name = name_from_table
        source_path_value = table.get("source_path")
        source_path = (
            Path(source_path_value)
            if isinstance(source_path_value, str) and source_path_value
            else None
        )
        if source_path_value is not None and source_path is None:
            raise ValueError(
                f"cloud.shared_assets.{name}.source_path must be a non-empty string"
            )
        source_json_pointer = _optional_str(
            table,
            "source_json_pointer",
            f"cloud.shared_assets.{name}",
        )
        assets.append(
            CloudSharedAssetSettings(
                name=name,
                source_path=source_path,
                source_json_pointer=source_json_pointer,
                blob_dir=str(table.get("blob_dir", "shared-assets")),
                required=bool(table.get("required", True)),
                remote_path_var=_optional_str(
                    table,
                    "remote_path_var",
                    f"cloud.shared_assets.{name}",
                ),
            )
        )
    return tuple(assets)


def _load_task_payload_settings(
    cloud: dict[str, Any],
) -> CloudTaskPayloadSettings:
    raw_payload = cloud.get("task_payload")
    if raw_payload is None:
        return CloudTaskPayloadSettings()
    payload = _require_mapping(raw_payload, "cloud.task_payload")
    transforms = payload.get("transforms", ())
    if transforms is None:
        transform_tables: list[dict[str, Any]] = []
    elif isinstance(transforms, list):
        transform_tables = [
            _require_mapping(item, "cloud.task_payload.transforms")
            for item in transforms
        ]
    else:
        raise ValueError("cloud.task_payload.transforms must be an array")

    parsed_transforms = []
    for transform in transform_tables:
        op_value = str(transform.get("op", CloudPayloadTransformOp.SET.value))
        try:
            op = CloudPayloadTransformOp(op_value)
        except ValueError as exc:
            raise ValueError(
                "cloud.task_payload.transforms.op must be one of: set"
            ) from exc
        parsed_transforms.append(
            CloudTaskPayloadTransform(
                name=_optional_str(
                    transform,
                    "name",
                    "cloud.task_payload.transforms",
                ),
                op=op,
                path=_required_str(
                    transform,
                    "path",
                    "cloud.task_payload.transforms",
                ),
                value=transform.get("value"),
                on_missing=cast(
                    Literal["error", "skip", "create"],
                    str(transform.get("on_missing", "error")),
                ),
            )
        )

    return CloudTaskPayloadSettings(
        task_output_dir=_optional_str(
            payload,
            "task_output_dir",
            "cloud.task_payload",
        ),
        transforms=tuple(parsed_transforms),
    )


def _load_auto_size_settings(
    cloud: dict[str, Any],
    *,
    config_dir: Path,
    build_context: Path,
) -> CloudAutoSizeSettings:
    """Read optional auto-size probe settings from the cloud config."""
    auto_size = cloud.get("auto_size")
    if auto_size is None:
        return CloudAutoSizeSettings()
    auto_size = _require_mapping(auto_size, "cloud.auto_size")
    probe = auto_size.get("probe")
    probe_module = auto_size.get("probe_module")
    memory_scope_value = str(auto_size.get("memory_scope", "self"))
    if probe is not None and not isinstance(probe, str):
        raise ValueError("cloud.auto_size.probe must be a string")
    if probe_module is not None and not isinstance(probe_module, str):
        raise ValueError("cloud.auto_size.probe_module must be a string")
    try:
        memory_scope = CloudAutoSizeMemoryScope(memory_scope_value)
    except ValueError as exc:
        raise ValueError(
            "cloud.auto_size.memory_scope must be one of: self, process_tree"
        ) from exc
    local_mrp_config_path_value = auto_size.get("local_mrp_config_path")
    local_mrp_config_path = (
        None
        if local_mrp_config_path_value is None
        else _resolve_existing_file(
            config_dir=config_dir,
            build_context=build_context,
            value=local_mrp_config_path_value,
            field_name="cloud.auto_size.local_mrp_config_path",
        )
    )
    if probe in {"mrp", "local_task"} and local_mrp_config_path is None:
        raise ValueError(
            "cloud.auto_size.local_mrp_config_path is required for "
            f"probe={probe!r}"
        )
    if probe is not None and probe not in {"mrp", "local_task"}:
        raise ValueError(
            "cloud.auto_size.probe must be one of: mrp, local_task"
        )
    return CloudAutoSizeSettings(
        probe=probe,
        local_mrp_config_path=local_mrp_config_path,
        probe_module=probe_module,
        memory_scope=memory_scope,
    )


def load_cloud_model_config(config_path: str | Path) -> CloudModelConfig:
    """Load model-facing cloud settings from a TOML config with ``[cloud]``."""
    resolved_config_path = Path(config_path)
    with resolved_config_path.open("rb") as f:
        config = tomllib.load(f)

    config_dir = resolved_config_path.parent.resolve()
    if "cloud" not in config:
        raise ValueError(
            f"{resolved_config_path} must contain a top-level [cloud] table"
        )

    cloud = _require_mapping(config["cloud"], "cloud")
    image = _require_mapping(cloud.get("image"), "cloud.image")
    build_context = _resolve_dir(
        config_dir,
        image.get("build_context", "."),
    )
    dockerfile = _resolve_existing_file(
        config_dir=config_dir,
        build_context=build_context,
        value=_required_str(image, "dockerfile", "cloud.image"),
        field_name="cloud.image.dockerfile",
    )
    runtime_settings = _load_model_cloud_runtime_settings(cloud)
    output_settings = _load_output_settings(cloud)
    shared_assets = _load_shared_asset_settings(cloud)
    task_payload = _load_task_payload_settings(cloud)
    auto_size_settings = _load_auto_size_settings(
        cloud,
        config_dir=config_dir,
        build_context=build_context,
    )

    return CloudModelConfig(
        config_path=resolved_config_path,
        build_context=build_context,
        dockerfile=dockerfile,
        runtime_settings=runtime_settings,
        output=output_settings,
        auto_size=auto_size_settings,
        shared_assets=shared_assets,
        task_payload=task_payload,
    )
