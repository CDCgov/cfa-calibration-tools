from __future__ import annotations

import asyncio
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
from numpy.random import SeedSequence

from calibrationtools.azure_batch_executor import AzureBatchExecutor
from calibrationtools.cloud_executor import (
    CloudAcceptanceTask,
    run_cloud_acceptance_task,
)
from calibrationtools.particle import Particle


def make_particle(_: SeedSequence | None) -> Particle:
    return Particle({"value": 1.0})


def zero_distance(_: Particle) -> float:
    return 0.0


class FakeCloudClient:
    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        self.uploaded: dict[str, bytes] = {}
        self.results: dict[str, bytes] = {}
        self.calls: list[tuple[str, object]] = []
        self.image_tags: list[str] = []
        self.batch_service_client = SimpleNamespace(
            task=SimpleNamespace(list=self.list_tasks),
            file=SimpleNamespace(
                get_from_task=lambda *args: [b"fake worker stderr"]
            ),
        )

    def create_blob_container(self, name: str) -> None:
        self.calls.append(("container", name))

    def upload_files(
        self, *, files, container_name, local_root_dir, location_in_blob
    ) -> None:
        for name in files:
            payload = Path(local_root_dir, name).read_bytes()
            self.uploaded[name] = payload
            tasks = pickle.loads(payload)
            self.results[f"results-{name}"] = pickle.dumps(
                [run_cloud_acceptance_task(task) for task in tasks]
            )

    def create_pool(self, *args, **kwargs) -> None:
        self.calls.append(("pool", args[0]))

    def create_job(self, *args, **kwargs) -> None:
        self.calls.append(("job", args[0]))

    def add_task(self, **kwargs) -> None:
        self.calls.append(("task", kwargs["name_suffix"]))

    def list_tasks(self, job_id: str):
        return [
            SimpleNamespace(
                id="task-0",
                execution_info=SimpleNamespace(
                    exit_code=self.exit_code,
                    failure_info=SimpleNamespace(message="worker failed"),
                ),
            )
            for index in range(sum(call[0] == "task" for call in self.calls))
        ]

    def download_file(self, *, src_path, dest_path, container_name) -> None:
        Path(dest_path).write_bytes(self.results[src_path])

    def delete_job(self, job_id: str) -> None:
        self.calls.append(("delete_job", job_id))

    def delete_pool(self, pool_name: str) -> None:
        self.calls.append(("delete_pool", pool_name))

    def package_and_upload_dockerfile(self, **kwargs) -> str:
        self.calls.append(("build_image", kwargs["repo_name"]))
        return "demo.azurecr.io/demo-image:latest"

    def list_acr_tags(self, **kwargs) -> list[str]:
        return self.image_tags


def task(slot_id: int) -> CloudAcceptanceTask:
    return CloudAcceptanceTask(
        slot_id=slot_id,
        seed_sequence=SeedSequence(slot_id),
        tolerance=1.0,
        max_attempts=1,
        sample_method=make_particle,
        particle_to_distance=zero_distance,
    )


def test_azure_executor_uploads_chunks_and_returns_results_in_worker_order() -> (
    None
):
    client = FakeCloudClient()
    executor = AzureBatchExecutor(
        base_name="Demo Study",
        registry_server="demo.azurecr.io",
        chunk_size=2,
        poll_interval=0,
        cloud_client=client,
    )

    results = asyncio.run(executor.execute_tasks([task(0), task(1), task(2)]))

    assert [result.slot_id for result in results] == [0, 1, 2]
    assert len(client.uploaded) == 2
    assert any(call[0] == "delete_job" for call in client.calls)
    assert not any(call[0] == "delete_pool" for call in client.calls)


def test_azure_executor_surfaces_task_failure_before_download() -> None:
    client = FakeCloudClient(exit_code=2)
    executor = AzureBatchExecutor(
        registry_server="demo.azurecr.io",
        poll_interval=0,
        cloud_client=client,
    )

    with pytest.raises(RuntimeError) as error:
        asyncio.run(executor.execute_tasks([task(0)]))
    assert "task-0" in str(error.value)
    assert "exit 2" in str(error.value)
    assert "fake worker stderr" in str(error.value)


def test_azure_executor_supports_current_batch_client_api() -> None:
    client = FakeCloudClient(exit_code=2)
    client.batch_service_client = SimpleNamespace(
        list_tasks=client.list_tasks,
        download_task_file=lambda *args: [b"current client stderr"],
    )
    executor = AzureBatchExecutor(
        registry_server="demo.azurecr.io",
        poll_interval=0,
        cloud_client=client,
    )

    with pytest.raises(RuntimeError, match="current client stderr"):
        asyncio.run(executor.execute_tasks([task(0)]))


def test_azure_executor_requires_built_image_to_appear_in_acr() -> None:
    client = FakeCloudClient()
    executor = AzureBatchExecutor(
        registry_server="demo.azurecr.io",
        build_image=True,
        cloud_client=client,
    )

    with pytest.raises(RuntimeError, match="did not produce"):
        executor._setup_job([task(0)], callback=None)


def test_azure_executor_clone_isolates_names_and_disables_image_publish() -> (
    None
):
    executor = AzureBatchExecutor(
        base_name="Base Study",
        registry_server="demo.azurecr.io",
        build_image=True,
        client_factory=FakeCloudClient,
    )

    clone = executor.clone_for_scenario("Scenario A")

    assert clone.base_name == "base-study-scenario-a"
    assert clone.pool_name == "base-study-scenario-a-pool"
    assert clone.build_image is False
    assert clone.upload_image is False


def test_azure_executor_uses_cfa_cloudops_acr_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZURE_CONTAINER_REGISTRY_SERVER", raising=False)
    monkeypatch.setenv("AZURE_CONTAINER_REGISTRY_ACCOUNT", "exampleacr")
    monkeypatch.setenv("AZURE_CONTAINER_REGISTRY_DOMAIN", "azurecr.io")

    executor = AzureBatchExecutor(image_name="example-model")

    assert executor.image_uri == "exampleacr.azurecr.io/example-model:latest"


def test_azure_executor_uses_registry_from_cloud_client_configuration() -> (
    None
):
    cloud_client = SimpleNamespace(
        cred=SimpleNamespace(
            azure_container_registry_endpoint="configuredacr.azurecr.io"
        )
    )
    executor = AzureBatchExecutor(
        image_name="example-model", cloud_client=cloud_client
    )

    assert (
        executor.image_uri == "configuredacr.azurecr.io/example-model:latest"
    )
