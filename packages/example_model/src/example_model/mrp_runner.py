import atexit
import shutil
import tempfile
from pathlib import Path

from mrp import run as mrp_run

from calibrationtools.mrp_csv_runner import (
    CSVOutputMRPRunner,
    make_csv_output_dir_reader,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_REPO_DEFAULTS_PATH = (
    _REPO_ROOT / "packages" / "example_model" / "defaults.json"
)
_BUNDLED_DEFAULTS_JSON = """{
  "seed": 123,
  "max_gen": 15,
  "n": 3,
  "p": 0.5,
  "max_infect": 500
}
"""
_BUNDLED_CONFIG_DIR: Path | None = None


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _repo_default_config_path(filename: str) -> Path | None:
    candidate = _REPO_ROOT / filename
    if candidate.is_file() and _REPO_DEFAULTS_PATH.is_file():
        return candidate
    return None


def _bundled_config_text(
    filename: str,
    *,
    defaults_path: Path,
) -> str:
    defaults_path_literal = _toml_string(str(defaults_path))
    if filename == "example_model.mrp.toml":
        return (
            f"input = {defaults_path_literal}\n\n"
            "[model]\n"
            'spec = "example_model"\n'
            'version = "0.0.1"\n\n'
            "[runtime]\n"
            'env = "uv"\n'
            'command = "python"\n'
            'args = ["-m", "example_model"]\n\n'
            "[output]\n"
            'spec = "filesystem"\n'
            'dir = "./output"\n'
        )
    if filename == "example_model.mrp.docker.toml":
        return (
            f"input = {defaults_path_literal}\n\n"
            "[model]\n"
            'spec = "example_model"\n'
            'version = "0.0.1"\n\n'
            "[runtime]\n"
            'command = "sh"\n'
            "args = [\n"
            '  "-c",\n'
            '  "exec docker run --rm -i --user \\"$(id -u):$(id -g)\\" -v \\"$PWD:/work\\" -w /work cfa-calibration-tools-example-model-python:latest",\n'
            "]\n\n"
            "[output]\n"
            'spec = "filesystem"\n'
            'dir = "./output"\n'
        )
    if filename == "example_model.mrp.cloud.toml":
        return (
            f"input = {defaults_path_literal}\n\n"
            "[model]\n"
            'spec = "example_model"\n'
            'version = "0.0.1"\n\n'
            "[runtime]\n"
            'spec = "process"\n'
            'command = "python"\n'
            "args = [\n"
            '  "-m",\n'
            '  "example_model.cloud_mrp_executor",\n'
            "]\n\n"
            "[runtime.cloud]\n"
            'keyvault = "cfa-predict"\n'
            'local_image = "cfa-calibration-tools-example-model-python"\n'
            'repository = "cfa-calibration-tools-example-model"\n'
            'task_mrp_config_path = "/app/example_model.mrp.task.toml"\n'
            'pool_prefix = "example-model-cloud"\n'
            'job_prefix = "example-model-cloud"\n'
            'input_container_prefix = "example-model-cloud-input"\n'
            'output_container_prefix = "example-model-cloud-output"\n'
            'logs_container_prefix = "example-model-cloud-logs"\n'
            'input_mount_path = "/cloud-input"\n'
            'output_mount_path = "/cloud-output"\n'
            'logs_mount_path = "/cloud-logs"\n'
            'vm_size = "large"\n'
            "jobs_per_session = 1\n"
            "task_slots_per_node = 50\n"
            "task_timeout_minutes = 60\n"
            "pool_ready_timeout_minutes = 20\n"
            "dispatch_buffer = 1000\n"
            "print_task_durations = false\n\n"
            "[output]\n"
            'spec = "filesystem"\n'
            'dir = "./output"\n'
        )
    raise ValueError(f"Unsupported example-model config: {filename}")


def _cleanup_bundled_config_dir() -> None:
    global _BUNDLED_CONFIG_DIR
    if _BUNDLED_CONFIG_DIR is not None:
        shutil.rmtree(_BUNDLED_CONFIG_DIR, ignore_errors=True)
        _BUNDLED_CONFIG_DIR = None


def _materialize_bundled_config_dir() -> Path:
    global _BUNDLED_CONFIG_DIR

    if _BUNDLED_CONFIG_DIR is None:
        config_dir = Path(tempfile.mkdtemp(prefix="example-model-mrp-"))
        defaults_path = config_dir / "defaults.json"
        defaults_path.write_text(_BUNDLED_DEFAULTS_JSON, encoding="utf-8")
        for filename in (
            "example_model.mrp.toml",
            "example_model.mrp.docker.toml",
            "example_model.mrp.cloud.toml",
        ):
            (config_dir / filename).write_text(
                _bundled_config_text(filename, defaults_path=defaults_path),
                encoding="utf-8",
            )
        _BUNDLED_CONFIG_DIR = config_dir
        atexit.register(_cleanup_bundled_config_dir)

    return _BUNDLED_CONFIG_DIR


def _default_config_path(filename: str) -> Path:
    repo_path = _repo_default_config_path(filename)
    if repo_path is not None:
        return repo_path
    return _materialize_bundled_config_dir() / filename


DEFAULT_MRP_CONFIG_PATH = _default_config_path("example_model.mrp.toml")
DEFAULT_DOCKER_MRP_CONFIG_PATH = _default_config_path(
    "example_model.mrp.docker.toml"
)
DEFAULT_CLOUD_MRP_CONFIG_PATH = _default_config_path(
    "example_model.mrp.cloud.toml"
)


_read_population_from_output_dir = make_csv_output_dir_reader(
    output_filename="output.csv",
    value_column="population",
    value_parser=int,
)


class ExampleModelMRPRunner(CSVOutputMRPRunner[int]):
    """Thin example-model binding over the shared CSV MRP runner."""

    _read_population_from_output_dir = staticmethod(
        _read_population_from_output_dir
    )

    def __init__(self, config_path: str | Path = DEFAULT_MRP_CONFIG_PATH):
        super().__init__(
            config_path,
            output_filename="output.csv",
            value_column="population",
            value_parser=int,
            header_fields=("generation", "population"),
            mrp_run_func=mrp_run,
        )
