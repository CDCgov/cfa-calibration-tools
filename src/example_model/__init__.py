from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

_WORKSPACE_PACKAGE = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "example_model"
    / "src"
    / "example_model"
)
if _WORKSPACE_PACKAGE.is_dir():
    __path__.append(str(_WORKSPACE_PACKAGE))

_EXPORTS = {
    "Binom_BP_Model": (".example_model", "Binom_BP_Model"),
    "DEFAULT_CLOUD_MRP_CONFIG_PATH": (
        ".mrp_runner",
        "DEFAULT_CLOUD_MRP_CONFIG_PATH",
    ),
    "DEFAULT_DOCKER_MRP_CONFIG_PATH": (
        ".mrp_runner",
        "DEFAULT_DOCKER_MRP_CONFIG_PATH",
    ),
    "DEFAULT_MRP_CONFIG_PATH": (".mrp_runner", "DEFAULT_MRP_CONFIG_PATH"),
    "ExampleModelCloudRunner": (
        ".cloud_runner",
        "ExampleModelCloudRunner",
    ),
    "ExampleModelMRPRunner": (".mrp_runner", "ExampleModelMRPRunner"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
