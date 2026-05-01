from __future__ import annotations

import inspect
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

DEFAULT_INPUT_MOUNT_PATH = "/cloud-input"
DEFAULT_OUTPUT_MOUNT_PATH = "/cloud-output"
DEFAULT_LOGS_MOUNT_PATH = "/cloud-logs"
DEFAULT_VM_SIZE = "large"
DEFAULT_JOBS_PER_SESSION = 1
# Deprecated alias retained for callers that imported the legacy name.
# Prefer ``DEFAULT_JOBS_PER_SESSION``; this will be removed in a future
# release.
DEFAULT_JOBS_PER_GENERATION = DEFAULT_JOBS_PER_SESSION
DEFAULT_TASK_SLOTS_PER_NODE = 1
DEFAULT_POOL_MAX_NODES = 5
DEFAULT_TASK_TIMEOUT_MINUTES = 60
DEFAULT_POOL_READY_TIMEOUT_MINUTES = 20
DEFAULT_POOL_AUTO_SCALE_EVALUATION_INTERVAL_MINUTES = 5
DEFAULT_DISPATCH_BUFFER = 0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0

_CloudRuntimeSettingsT = TypeVar(
    "_CloudRuntimeSettingsT", bound="CloudRuntimeSettings"
)


def _install_jobs_per_generation_compat_init(cls: type[Any]) -> None:
    """Wrap a dataclass init without losing its public constructor signature."""
    original_init = cls.__init__
    signature = inspect.signature(original_init)

    @wraps(original_init)
    def compat_init(self: Any, *args: Any, **kwargs: Any) -> None:
        if "jobs_per_generation" in kwargs:
            import warnings

            legacy = kwargs.pop("jobs_per_generation")
            if "jobs_per_session" in kwargs:
                raise TypeError(
                    "CloudRuntimeSettings received both "
                    "`jobs_per_generation` (deprecated) and "
                    "`jobs_per_session`; pass only `jobs_per_session`."
                )
            warnings.warn(
                "CloudRuntimeSettings(jobs_per_generation=...) is "
                "deprecated; use jobs_per_session=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["jobs_per_session"] = legacy
        original_init(self, *args, **kwargs)

    compat_init.__signature__ = signature  # type: ignore[attr-defined]
    cls.__init__ = compat_init  # type: ignore[invalid-assignment]
    cls.__signature__ = signature


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

    @property
    def jobs_per_generation(self) -> int:
        """Deprecated alias for :attr:`jobs_per_session`."""
        import warnings

        warnings.warn(
            "CloudRuntimeSettings.jobs_per_generation is deprecated; "
            "use jobs_per_session instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.jobs_per_session


_install_jobs_per_generation_compat_init(CloudRuntimeSettings)


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


@dataclass(frozen=True)
class CloudOutputSettings:
    """Describe the shared CSV output contract for a cloud model."""

    filename: str
    csv_value_column: str
    csv_value_type: CloudCSVValueType


@dataclass(frozen=True)
class CloudAutoSizeSettings:
    """Describe optional cloud auto-size probe configuration."""

    probe: str | None = None
    local_mrp_config_path: Path | None = None
    probe_module: str | None = None


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
    simulation_mrp_config_path: Path | None = None


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


def _resolve_optional_existing_file(
    *,
    config_dir: Path,
    build_context: Path,
    value: str | Path | None,
    field_name: str,
) -> Path | None:
    """Resolve an optional file path when one is configured."""
    if value is None:
        return None
    return _resolve_existing_file(
        config_dir=config_dir,
        build_context=build_context,
        value=value,
        field_name=field_name,
    )


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
        jobs_per_session=_load_jobs_per_session(
            cloud,
            CloudRuntimeSettings(
                keyvault="unused",
                local_image="unused",
                repository="unused",
                task_mrp_config_path="unused",
                pool_prefix="unused",
                job_prefix="unused",
                input_container_prefix="unused",
                output_container_prefix="unused",
                logs_container_prefix="unused",
            ),
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
        print_task_durations=bool(cloud.get("print_task_durations", False)),
    )


def _load_legacy_runtime_settings(
    cloud: dict[str, Any],
) -> CloudRuntimeSettings:
    """Build runtime settings from a legacy ``[runtime.cloud]`` table."""
    required_context = "runtime.cloud"
    return CloudRuntimeSettings(
        keyvault=_required_str(cloud, "keyvault", required_context),
        local_image=_required_str(cloud, "local_image", required_context),
        repository=_required_str(cloud, "repository", required_context),
        task_mrp_config_path=_required_str(
            cloud,
            "task_mrp_config_path",
            required_context,
        ),
        pool_prefix=_required_str(cloud, "pool_prefix", required_context),
        job_prefix=_required_str(cloud, "job_prefix", required_context),
        input_container_prefix=_required_str(
            cloud,
            "input_container_prefix",
            required_context,
        ),
        output_container_prefix=_required_str(
            cloud,
            "output_container_prefix",
            required_context,
        ),
        logs_container_prefix=_required_str(
            cloud,
            "logs_container_prefix",
            required_context,
        ),
        input_mount_path=str(
            cloud.get("input_mount_path", DEFAULT_INPUT_MOUNT_PATH)
        ),
        output_mount_path=str(
            cloud.get("output_mount_path", DEFAULT_OUTPUT_MOUNT_PATH)
        ),
        logs_mount_path=str(
            cloud.get("logs_mount_path", DEFAULT_LOGS_MOUNT_PATH)
        ),
        vm_size=str(cloud.get("vm_size", DEFAULT_VM_SIZE)),
        jobs_per_session=_load_jobs_per_session(
            cloud,
            CloudRuntimeSettings(
                keyvault="unused",
                local_image="unused",
                repository="unused",
                task_mrp_config_path="unused",
                pool_prefix="unused",
                job_prefix="unused",
                input_container_prefix="unused",
                output_container_prefix="unused",
                logs_container_prefix="unused",
            ),
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
        print_task_durations=bool(cloud.get("print_task_durations", False)),
    )


def _load_output_settings(
    cloud: dict[str, Any],
    *,
    legacy: bool,
) -> CloudOutputSettings:
    """Read the CSV output contract from TOML, with legacy defaults."""
    output = cloud.get("output", {})
    if not output and legacy:
        output = {
            "filename": "output.csv",
            "csv_value_column": "population",
            "csv_value_type": "int",
        }
    output = _require_mapping(output, "cloud.output")
    csv_value_type = str(output.get("csv_value_type", "int"))
    try:
        value_type = CloudCSVValueType(csv_value_type)
    except ValueError as exc:
        raise ValueError(
            "cloud.output.csv_value_type must be one of: int, float, str"
        ) from exc
    return CloudOutputSettings(
        filename=_required_str(output, "filename", "cloud.output"),
        csv_value_column=_required_str(
            output,
            "csv_value_column",
            "cloud.output",
        ),
        csv_value_type=value_type,
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
    if probe is not None and not isinstance(probe, str):
        raise ValueError("cloud.auto_size.probe must be a string")
    if probe_module is not None and not isinstance(probe_module, str):
        raise ValueError("cloud.auto_size.probe_module must be a string")
    local_mrp_config_path = _resolve_optional_existing_file(
        config_dir=config_dir,
        build_context=build_context,
        value=auto_size.get("local_mrp_config_path"),
        field_name="cloud.auto_size.local_mrp_config_path",
    )
    if probe == "mrp" and local_mrp_config_path is None:
        raise ValueError(
            "cloud.auto_size.local_mrp_config_path is required for probe='mrp'"
        )
    return CloudAutoSizeSettings(
        probe=probe,
        local_mrp_config_path=local_mrp_config_path,
        probe_module=probe_module,
    )


def _load_simulation_mrp_config_path(
    cloud: dict[str, Any],
    *,
    config_dir: Path,
    build_context: Path,
) -> Path | None:
    """Read the optional MRP config used only for simulation execution."""
    value = cloud.get("simulation_mrp_config_path")
    if value is None:
        return None
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path
    build_context_candidate = build_context / raw_path
    if build_context_candidate.is_file():
        return build_context_candidate.resolve()
    config_dir_candidate = config_dir / raw_path
    if config_dir_candidate.is_file():
        return config_dir_candidate.resolve()
    return build_context_candidate.resolve()


def load_cloud_model_config(
    config_path: str | Path,
    *,
    default_build_context: str | Path | None = None,
    default_dockerfile: str | Path | None = None,
) -> CloudModelConfig:
    """Load model-facing cloud settings from new or legacy TOML configs.

    New configs use a top-level ``[cloud]`` table. Legacy MRP cloud configs
    with ``[runtime.cloud]`` remain supported when callers provide build
    context and Dockerfile defaults, because those fields did not exist in the
    old format.
    """
    resolved_config_path = Path(config_path)
    with resolved_config_path.open("rb") as f:
        config = tomllib.load(f)

    config_dir = resolved_config_path.parent.resolve()
    if "cloud" in config:
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
        output_settings = _load_output_settings(cloud, legacy=False)
        auto_size_settings = _load_auto_size_settings(
            cloud,
            config_dir=config_dir,
            build_context=build_context,
        )
        simulation_mrp_config_path = _load_simulation_mrp_config_path(
            cloud,
            config_dir=config_dir,
            build_context=build_context,
        )
    else:
        runtime = _require_mapping(config.get("runtime", {}), "runtime")
        cloud = _require_mapping(runtime.get("cloud", {}), "runtime.cloud")
        build_context = (
            _resolve_dir(config_dir, default_build_context)
            if default_build_context is not None
            else config_dir
        )
        if default_dockerfile is None:
            dockerfile = (build_context / "Dockerfile").resolve()
        else:
            dockerfile = _resolve_existing_file(
                config_dir=config_dir,
                build_context=build_context,
                value=default_dockerfile,
                field_name="cloud.image.dockerfile",
            )
        runtime_settings = _load_legacy_runtime_settings(cloud)
        output_settings = _load_output_settings({}, legacy=True)
        auto_size_settings = CloudAutoSizeSettings()
        simulation_mrp_config_path = resolved_config_path

    return CloudModelConfig(
        config_path=resolved_config_path,
        build_context=build_context,
        dockerfile=dockerfile,
        runtime_settings=runtime_settings,
        output=output_settings,
        auto_size=auto_size_settings,
        simulation_mrp_config_path=simulation_mrp_config_path,
    )


def _load_jobs_per_session(
    cloud: dict[str, Any], defaults: CloudRuntimeSettings
) -> int:
    """Read ``jobs_per_session`` from TOML, honoring the legacy key."""
    if "jobs_per_session" in cloud:
        if "jobs_per_generation" in cloud:
            import warnings

            warnings.warn(
                "Both `jobs_per_session` and the deprecated "
                "`jobs_per_generation` are set in the cloud config; "
                "`jobs_per_session` takes precedence.",
                DeprecationWarning,
                stacklevel=3,
            )
        return int(cloud["jobs_per_session"])
    if "jobs_per_generation" in cloud:
        import warnings

        warnings.warn(
            "The `jobs_per_generation` cloud config key is deprecated; "
            "rename it to `jobs_per_session`.",
            DeprecationWarning,
            stacklevel=3,
        )
        return int(cloud["jobs_per_generation"])
    return int(defaults.jobs_per_session)


def load_cloud_runtime_settings(
    config_path: str | Path,
    *,
    defaults: _CloudRuntimeSettingsT,
) -> _CloudRuntimeSettingsT:
    """Load low-level cloud runtime settings with caller-provided defaults."""
    with Path(config_path).open("rb") as f:
        config = tomllib.load(f)
    if "cloud" in config:
        model_config = load_cloud_model_config(
            config_path,
            default_build_context=Path(config_path).parent,
            default_dockerfile=Path(config_path).parent / "Dockerfile",
        )
        runtime_settings = model_config.runtime_settings
        settings_type = type(defaults)
        return settings_type(**runtime_settings.__dict__)

    cloud = config.get("runtime", {}).get("cloud", {})
    settings_type = type(defaults)
    return settings_type(
        keyvault=cloud.get("keyvault", defaults.keyvault),
        local_image=cloud.get("local_image", defaults.local_image),
        repository=cloud.get("repository", defaults.repository),
        task_mrp_config_path=cloud.get(
            "task_mrp_config_path",
            defaults.task_mrp_config_path,
        ),
        pool_prefix=cloud.get("pool_prefix", defaults.pool_prefix),
        job_prefix=cloud.get("job_prefix", defaults.job_prefix),
        input_container_prefix=cloud.get(
            "input_container_prefix",
            defaults.input_container_prefix,
        ),
        output_container_prefix=cloud.get(
            "output_container_prefix",
            defaults.output_container_prefix,
        ),
        logs_container_prefix=cloud.get(
            "logs_container_prefix",
            defaults.logs_container_prefix,
        ),
        input_mount_path=cloud.get(
            "input_mount_path",
            defaults.input_mount_path,
        ),
        output_mount_path=cloud.get(
            "output_mount_path",
            defaults.output_mount_path,
        ),
        logs_mount_path=cloud.get("logs_mount_path", defaults.logs_mount_path),
        vm_size=cloud.get("vm_size", defaults.vm_size),
        jobs_per_session=_load_jobs_per_session(cloud, defaults),
        task_slots_per_node=int(
            cloud.get("task_slots_per_node", defaults.task_slots_per_node)
        ),
        pool_max_nodes=int(
            cloud.get("pool_max_nodes", defaults.pool_max_nodes)
        ),
        task_timeout_minutes=cloud.get(
            "task_timeout_minutes",
            defaults.task_timeout_minutes,
        ),
        pool_ready_timeout_minutes=cloud.get(
            "pool_ready_timeout_minutes",
            defaults.pool_ready_timeout_minutes,
        ),
        pool_auto_scale_evaluation_interval_minutes=int(
            cloud.get(
                "pool_auto_scale_evaluation_interval_minutes",
                defaults.pool_auto_scale_evaluation_interval_minutes,
            )
        ),
        dispatch_buffer=int(
            cloud.get("dispatch_buffer", defaults.dispatch_buffer)
        ),
        print_task_durations=bool(
            cloud.get("print_task_durations", defaults.print_task_durations)
        ),
    )
