from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from calibrationtools.cloud.config import CloudRuntimeSettings
from calibrationtools.cloud.runner import CloudMRPRunner


def make_settings(**overrides: Any) -> CloudRuntimeSettings:
    kwargs: dict[str, Any] = {
        "keyvault": "kv",
        "local_image": "local-model",
        "repository": "repo",
        "task_mrp_config_path": "/app/task.toml",
        "pool_prefix": "pool",
        "job_prefix": "job",
        "input_container_prefix": "input",
        "output_container_prefix": "output",
        "logs_container_prefix": "logs",
        "jobs_per_session": 2,
        "task_slots_per_node": 3,
        "pool_max_nodes": 4,
    }
    kwargs.update(overrides)
    return CloudRuntimeSettings(**kwargs)


class FakeCloudClient:
    def __init__(
        self,
        calls: list[tuple[str, Any]],
        *,
        fail_on_job: int | None = None,
    ) -> None:
        self.calls = calls
        self.fail_on_job = fail_on_job
        self.batch_service_client = SimpleNamespace()
        self.blob_service_client = SimpleNamespace(
            delete_container=self.delete_container
        )

    def create_blob_container(self, container_name: str) -> None:
        self.calls.append(("create_container", container_name))

    def create_job(self, **kwargs: Any) -> None:
        self.calls.append(("create_job", kwargs["job_name"]))
        if self.fail_on_job is not None:
            created_jobs = [
                value for name, value in self.calls if name == "create_job"
            ]
            if len(created_jobs) == self.fail_on_job:
                raise RuntimeError("job setup failed")

    def delete_job(self, job_name: str) -> None:
        self.calls.append(("delete_job", job_name))

    def delete_pool(self, pool_name: str) -> None:
        self.calls.append(("delete_pool", pool_name))

    def delete_container(self, container_name: str) -> None:
        self.calls.append(("delete_container", container_name))


def make_runner_backend(
    calls: list[tuple[str, Any]],
    client: FakeCloudClient,
) -> dict[str, Any]:
    def create_pool_with_blob_mounts(**kwargs: Any) -> None:
        calls.append(
            (
                "create_pool",
                {
                    "pool_name": kwargs["pool_name"],
                    "mounts": kwargs["mounts"],
                    "image": kwargs["container_image_name"],
                },
            )
        )

    def make_resource_name(
        prefix: str, suffix: str, *, max_length: int
    ) -> str:
        calls.append(("make_resource_name", (prefix, suffix, max_length)))
        return f"{prefix}-{suffix}"[:max_length]

    return {
        "create_cloud_client_func": lambda *, keyvault: (
            calls.append(("create_client", keyvault)) or client
        ),
        "git_short_sha_func": lambda repo_root: (
            calls.append(("git_short_sha", repo_root)) or "abc123"
        ),
        "make_session_id_func": lambda tag: (
            calls.append(("make_session_id", tag)) or f"session-{tag}"
        ),
        "build_local_image_func": lambda **kwargs: (
            calls.append(("build_image", kwargs)) or "local:abc123"
        ),
        "upload_local_image_func": lambda **kwargs: (
            calls.append(("upload_image", kwargs)) or "remote:abc123"
        ),
        "create_pool_with_blob_mounts_func": create_pool_with_blob_mounts,
        "wait_for_pool_ready_func": lambda **kwargs: (
            calls.append(("wait_pool", kwargs["pool_name"]))
            or SimpleNamespace(pool=True)
        ),
        "make_resource_name_func": make_resource_name,
    }


def make_cloud_runner(
    tmp_path: Path,
    *,
    settings: Any | None = None,
    generation_count: int = 3,
    client: FakeCloudClient | None = None,
    calls: list[tuple[str, Any]] | None = None,
    **overrides: Any,
) -> CloudMRPRunner:
    calls = [] if calls is None else calls
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    client = client or FakeCloudClient(calls)
    kwargs: dict[str, Any] = {
        "generation_count": generation_count,
        "max_concurrent_simulations": 5,
        "repo_root": tmp_path,
        "dockerfile": dockerfile,
        "read_output_dir": lambda path: path,
        "runtime_settings": settings or make_settings(),
    }
    kwargs.update(make_runner_backend(calls, client))
    kwargs.update(overrides)
    return CloudMRPRunner(tmp_path / "cloud.toml", **kwargs)


def test_cloud_runner_missing_dockerfile_fails_before_client_creation(
    tmp_path: Path,
):
    calls: list[tuple[str, Any]] = []
    missing_dockerfile = tmp_path / "missing.Dockerfile"

    with pytest.raises(FileNotFoundError, match="Dockerfile not found"):
        CloudMRPRunner(
            tmp_path / "cloud.toml",
            generation_count=1,
            max_concurrent_simulations=1,
            repo_root=tmp_path,
            dockerfile=missing_dockerfile,
            read_output_dir=lambda path: path,
            runtime_settings=make_settings(),
            create_cloud_client_func=lambda **kwargs: calls.append(
                ("create_client", kwargs)
            ),
        )

    assert calls == []


@pytest.mark.parametrize(
    "loader,runtime_settings",
    [
        (None, None),
        (lambda path: make_settings(), make_settings()),
    ],
)
def test_cloud_runner_requires_exactly_one_settings_source(
    tmp_path: Path,
    loader,
    runtime_settings,
):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")

    with pytest.raises(TypeError, match="exactly one"):
        CloudMRPRunner(
            tmp_path / "cloud.toml",
            generation_count=1,
            max_concurrent_simulations=1,
            repo_root=tmp_path,
            dockerfile=dockerfile,
            read_output_dir=lambda path: path,
            settings_loader=loader,
            runtime_settings=runtime_settings,
        )


def test_cloud_runner_rejects_invalid_concurrency(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")

    with pytest.raises(ValueError, match="max_concurrent_simulations"):
        CloudMRPRunner(
            tmp_path / "cloud.toml",
            generation_count=1,
            max_concurrent_simulations=0,
            repo_root=tmp_path,
            dockerfile=dockerfile,
            read_output_dir=lambda path: path,
            runtime_settings=make_settings(),
        )


def test_cloud_runner_rejects_invalid_runtime_settings(tmp_path: Path):
    invalid_settings = SimpleNamespace(jobs_per_session=0)

    with pytest.raises(ValueError, match="jobs_per_session"):
        make_cloud_runner(tmp_path, settings=invalid_settings)


def test_cloud_runner_uses_injected_backend_functions(
    tmp_path: Path,
    capsys,
):
    calls: list[tuple[str, Any]] = []

    runner = make_cloud_runner(tmp_path, calls=calls)

    assert runner.session.remote_image_ref == "remote:abc123"
    assert runner.session.pool_name == "pool-session-abc123"
    assert ("create_client", "kv") in calls
    assert any(name == "build_image" for name, _ in calls)
    assert any(name == "upload_image" for name, _ in calls)
    assert any(name == "create_pool" for name, _ in calls)
    assert "created pool pool-session-abc123" in capsys.readouterr().err


def test_initialize_cloud_session_creates_resources_and_shared_jobs(
    tmp_path: Path,
    capsys,
):
    calls: list[tuple[str, Any]] = []

    runner = make_cloud_runner(tmp_path, calls=calls)

    created = [name for name, _ in calls if name.startswith("create_")]
    assert created == [
        "create_client",
        "create_container",
        "create_container",
        "create_container",
        "create_pool",
        "create_job",
        "create_job",
    ]
    assert runner.session.job_names == {
        "0": ["job-session-abc123-j1", "job-session-abc123-j2"],
        "1": ["job-session-abc123-j1", "job-session-abc123-j2"],
        "2": ["job-session-abc123-j1", "job-session-abc123-j2"],
    }
    summary = capsys.readouterr().err
    assert "created 2 reusable job(s)" in summary
    assert "max_task_capacity=12" in summary
    assert "image=remote:abc123" in summary


def test_initialize_cloud_session_rolls_back_partial_resources(
    tmp_path: Path,
):
    calls: list[tuple[str, Any]] = []
    client = FakeCloudClient(calls, fail_on_job=2)

    with pytest.raises(RuntimeError, match="Cloud session initialization"):
        make_cloud_runner(tmp_path, calls=calls, client=client)

    assert ("delete_job", "job-session-abc123-j1") in calls
    assert ("delete_pool", "pool-session-abc123") in calls
    assert ("delete_container", "logs-session-abc123") in calls
    assert ("delete_container", "output-session-abc123") in calls
    assert ("delete_container", "input-session-abc123") in calls
