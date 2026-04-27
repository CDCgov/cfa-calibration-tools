from __future__ import annotations

import inspect
import tomllib
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

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
    cls.__init__ = compat_init
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
        if self.pool_max_nodes < 1:
            raise ValueError("pool_max_nodes must be at least 1")
        if self.pool_auto_scale_evaluation_interval_minutes < 5:
            raise ValueError(
                "pool_auto_scale_evaluation_interval_minutes must be at least 5"
            )

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
    with Path(config_path).open("rb") as f:
        config = tomllib.load(f)
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
