from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_ROOT / "packages/example_model/src/example_model"


def _imported_modules(path: Path) -> set[str]:
    """Return modules imported by one Python source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_example_source_contains_only_model_and_thin_entrypoint_modules():
    """Assert framework runner modules have moved out of the model package."""
    assert sorted(path.name for path in PACKAGE_DIR.glob("*.py")) == [
        "__init__.py",
        "__main__.py",
        "benchmark.py",
        "calibrate.py",
        "direct_runner.py",
        "example_model.py",
    ]


def test_example_source_does_not_import_cloud_or_mrp_runner_frameworks():
    """Assert example source no longer owns cloud or MRP runner plumbing."""
    forbidden_imports = {
        "calibrationtools.cloud.auto_size",
        "calibrationtools.cloud.config",
        "calibrationtools.cloud.runner",
        "calibrationtools.mrp_csv_runner",
        "mrp.run",
    }
    offenders: dict[str, list[str]] = {}

    for path in PACKAGE_DIR.glob("*.py"):
        imported = _imported_modules(path)
        matches = sorted(
            module
            for module in imported
            if module in forbidden_imports
            or module.startswith("calibrationtools.cloud.")
        )
        if matches:
            offenders[path.name] = matches

    assert offenders == {}
