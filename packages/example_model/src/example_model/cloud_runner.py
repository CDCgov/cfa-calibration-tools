from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mrp import run as mrp_run

from calibrationtools.cloud.config import DEFAULT_POLL_INTERVAL_SECONDS
from calibrationtools.cloud.runner import (
    create_cloud_mrp_runner as _create_cloud_mrp_runner,
)
from calibrationtools.cloud.runner import (
    resolve_cloud_build_context as _resolve_cloud_build_context,
)
from calibrationtools.mrp_csv_runner import make_csv_output_dir_reader

from .cloud_utils import (
    cloud_runner_backend,
    load_cloud_runtime_settings,
)
from .mrp_runner import DEFAULT_CLOUD_MRP_CONFIG_PATH

create_cloud_client = cloud_runner_backend.create_cloud_client
git_short_sha = cloud_runner_backend.git_short_sha
make_session_slug = cloud_runner_backend.make_session_slug
build_local_image = cloud_runner_backend.build_local_image
upload_local_image = cloud_runner_backend.upload_local_image
create_pool_with_blob_mounts = (
    cloud_runner_backend.create_pool_with_blob_mounts
)
wait_for_pool_ready = cloud_runner_backend.wait_for_pool_ready
add_batch_task_with_short_id = (
    cloud_runner_backend.add_batch_task_with_short_id
)
cancel_batch_task = cloud_runner_backend.cancel_batch_task
format_task_failure_message = cloud_runner_backend.format_task_failure_message
format_task_timing_summary = cloud_runner_backend.format_task_timing_summary
make_resource_name = cloud_runner_backend.make_resource_name
parse_generation_from_run_id = (
    cloud_runner_backend.parse_generation_from_run_id
)
suppress_cloudops_info_output = (
    cloud_runner_backend.suppress_cloudops_info_output
)

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_DOCKERFILE_RELATIVE_PATH = (
    Path("packages") / "example_model" / "Dockerfile"
)
_READ_POPULATION_FROM_OUTPUT_DIR = make_csv_output_dir_reader(
    output_filename="output.csv",
    value_column="population",
    value_parser=int,
)


def _current_cloud_runner_backend():
    return replace(
        cloud_runner_backend,
        create_cloud_client=create_cloud_client,
        git_short_sha=git_short_sha,
        make_session_slug=make_session_slug,
        build_local_image=build_local_image,
        upload_local_image=upload_local_image,
        create_pool_with_blob_mounts=create_pool_with_blob_mounts,
        wait_for_pool_ready=wait_for_pool_ready,
        add_batch_task_with_short_id=add_batch_task_with_short_id,
        cancel_batch_task=cancel_batch_task,
        format_task_failure_message=format_task_failure_message,
        format_task_timing_summary=format_task_timing_summary,
        make_resource_name=make_resource_name,
        parse_generation_from_run_id=parse_generation_from_run_id,
        suppress_cloudops_info_output=suppress_cloudops_info_output,
    )


def resolve_cloud_build_context(
    repo_root: str | Path | None = None,
    dockerfile: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve the Docker build context used by the example-model cloud runner.

    Source-tree callers keep the historical zero-configuration behavior, while
    wheel-installed callers can override either path explicitly.
    """
    return _resolve_cloud_build_context(
        default_repo_root=_DEFAULT_REPO_ROOT,
        default_dockerfile_relative_path=_DEFAULT_DOCKERFILE_RELATIVE_PATH,
        repo_root=repo_root,
        dockerfile=dockerfile,
        missing_dockerfile_message=(
            "Cloud mode requires the example-model Dockerfile. "
            "Looked at {dockerfile}; pass --repo-root and --dockerfile "
            "when running from an installed wheel."
        ),
    )


def ExampleModelCloudRunner(
    config_path: str | Path = DEFAULT_CLOUD_MRP_CONFIG_PATH,
    *,
    generation_count: int,
    max_concurrent_simulations: int,
    repo_root: str | Path | None = None,
    dockerfile: str | Path | None = None,
    print_task_durations: bool = False,
):
    """Run the example model through the shared cloud-backed MRP path."""

    return _create_cloud_mrp_runner(
        config_path,
        generation_count=generation_count,
        max_concurrent_simulations=max_concurrent_simulations,
        default_repo_root=_DEFAULT_REPO_ROOT,
        default_dockerfile_relative_path=_DEFAULT_DOCKERFILE_RELATIVE_PATH,
        repo_root=repo_root,
        dockerfile=dockerfile,
        missing_dockerfile_message=(
            "Cloud mode requires the example-model Dockerfile. "
            "Looked at {dockerfile}; pass --repo-root and --dockerfile "
            "when running from an installed wheel."
        ),
        settings_loader=load_cloud_runtime_settings,
        read_output_dir=_READ_POPULATION_FROM_OUTPUT_DIR,
        output_filename="output.csv",
        print_task_durations=print_task_durations,
        backend=_current_cloud_runner_backend(),
        poll_interval_seconds=DEFAULT_POLL_INTERVAL_SECONDS,
        mrp_run_func=mrp_run,
    )
