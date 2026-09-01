from pathlib import Path

from example_model_azure_smoke.run import build_smoke_sampler, parse_args

from calibrationtools.cloud_executor import (
    CloudAcceptanceTask,
    CloudExecutor,
    run_cloud_acceptance_task,
)


class InlineCloudExecutor(CloudExecutor):
    """Execute cloud tasks locally while preserving their serialization path."""

    async def execute_tasks(
        self,
        tasks: list[CloudAcceptanceTask],
        *,
        progress_callback=None,
        on_result=None,
    ):
        results = [run_cloud_acceptance_task(task) for task in tasks]
        if on_result is not None:
            for result in results:
                on_result(result)
        return results


def test_smoke_sampler_runs_through_cloud_task_serialization():
    sampler = build_smoke_sampler(particle_count=2)

    results = sampler.run(
        execution="azure_batch",
        cloud_executor=InlineCloudExecutor(),
    )

    assert results.smc_step_successes == [2]
    assert results.smc_step_attempts == [2]


def test_parse_args_exposes_small_safe_defaults():
    args = parse_args(["--build-image"])

    assert args.particle_count == 4
    assert args.chunk_size == 1
    assert args.max_autoscale_nodes == 1
    assert args.build_image is True
    assert args.delete_pool_after is True


def test_worker_dockerfile_installs_git_before_syncing_git_dependencies():
    dockerfile = Path(__file__).parents[1] / "Dockerfile"
    contents = dockerfile.read_text()

    assert "apt-get install --yes --no-install-recommends git" in contents
    assert contents.index("apt-get install") < contents.index("uv sync")
    assert "ENTRYPOINT" not in contents
