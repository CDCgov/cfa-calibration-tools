from __future__ import annotations

import asyncio
import logging
import pickle
import sys
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

    events = []
    with pytest.raises(RuntimeError) as error:
        asyncio.run(
            executor.execute_tasks([task(0)], progress_callback=events.append)
        )
    assert "task-0" in str(error.value)
    assert "exit 2" in str(error.value)
    assert "fake worker stderr" in str(error.value)
    assert any(call[0] == "delete_job" for call in client.calls)
    assert [event.payload["message"] for event in events] == [
        "Authenticating with Azure",
        "Creating Blob container calibrationtools-tasks",
        "Creating pool calibrationtools-pool (STANDARD_D2S_V3)",
        "Pool calibrationtools-pool created",
        "Uploading 1 task file",
        "Azure pool ready",
        "Azure tasks submitted",
        "Azure task progress 1/1",
        "Azure resources cleaned",
    ]


class NoisyCloudClient(FakeCloudClient):
    """Emit the advisory notices that `cfa-cloudops` writes to the console."""

    def create_pool(self, *args, **kwargs) -> None:
        print(
            f"Pool {args[0]} is using a deprecated VM series. "
            "Consider updating the pool to a supported VM series."
        )
        logging.getLogger("cfa.cloudops.batch_helpers").warning(
            "The current VM is too old. Please upgrade to a newer version."
        )
        super().create_pool(*args, **kwargs)


class ProgressBarCloudClient(FakeCloudClient):
    """Write a tqdm-style upload bar to stderr, as ``cfa-cloudops`` does."""

    def upload_files(self, *args, **kwargs):
        for done in (50, 100):
            sys.stderr.write(
                f"\rUploading files: {done}%|#####| {done}/100 [00:07<00:00]"
            )
        return super().upload_files(*args, **kwargs)


def test_azure_executor_reports_cloud_notices_as_progress_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fold library notices into progress events instead of raw console noise."""

    client = NoisyCloudClient()
    executor = AzureBatchExecutor(
        registry_server="demo.azurecr.io",
        poll_interval=0,
        cloud_client=client,
    )

    events: list = []
    asyncio.run(
        executor.execute_tasks([task(0)], progress_callback=events.append)
    )

    notices = [
        event.payload["message"]
        for event in events
        if event.payload.get("stage") == "cloud_notice"
    ]
    assert any("deprecated VM series" in notice for notice in notices)
    assert any("current VM is too old" in notice for notice in notices)
    assert all(notice.startswith("cfa-cloudops: ") for notice in notices)
    assert "deprecated VM series" not in capsys.readouterr().out


def test_azure_executor_keeps_upload_progress_bars_off_the_console(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep the library's stderr progress bar out of a caller's live display."""

    executor = AzureBatchExecutor(
        registry_server="demo.azurecr.io",
        poll_interval=0,
        cloud_client=ProgressBarCloudClient(),
    )

    events: list = []
    asyncio.run(
        executor.execute_tasks([task(0)], progress_callback=events.append)
    )

    assert "Uploading files" not in capsys.readouterr().err


def test_azure_executor_reports_each_cloud_notice_once() -> None:
    """Avoid repeating one advisory on every subsequent Azure call."""

    client = NoisyCloudClient()
    executor = AzureBatchExecutor(
        registry_server="demo.azurecr.io",
        poll_interval=0,
        cloud_client=client,
    )

    events: list = []
    asyncio.run(
        executor.execute_tasks([task(0)], progress_callback=events.append)
    )
    executor._prepare_pool(events.append)

    notices = [
        event.payload["message"]
        for event in events
        if event.payload.get("stage") == "cloud_notice"
    ]
    assert len(notices) == len(set(notices))


def test_azure_executor_reports_result_download_progress() -> None:
    """Replace per-file download prints with counted progress events."""

    client = FakeCloudClient()
    executor = AzureBatchExecutor(
        registry_server="demo.azurecr.io",
        chunk_size=1,
        poll_interval=0,
        cloud_client=client,
    )

    events: list = []
    asyncio.run(
        executor.execute_tasks(
            [task(0), task(1)], progress_callback=events.append
        )
    )

    downloads = [
        event.payload["message"]
        for event in events
        if event.payload.get("stage") == "download"
    ]
    assert downloads == [
        "Downloading Azure results 1/2",
        "Downloading Azure results 2/2",
    ]


class StalledPoolClient(FakeCloudClient):
    """Report an allocated-but-unusable pool with tasks stuck in `active`."""

    def __init__(self, node_state: str = "unusable") -> None:
        super().__init__()
        self.batch_service_client = SimpleNamespace(
            task=SimpleNamespace(list=self.list_stalled_tasks),
            compute_node=SimpleNamespace(
                list=lambda pool_name: [
                    SimpleNamespace(
                        state=node_state,
                        errors=[
                            SimpleNamespace(
                                code="ContainerPullFailed",
                                message="failed to pull image",
                            )
                        ],
                        start_task_info=None,
                    )
                ]
            ),
            pool=SimpleNamespace(
                get=lambda pool_name: SimpleNamespace(resize_errors=[])
            ),
        )

    def list_stalled_tasks(self, job_id: str):
        return [
            SimpleNamespace(id="task-0", state="active", execution_info=None)
        ]


def test_azure_executor_fails_fast_when_pool_has_no_usable_nodes() -> None:
    """Detect a spoiled pool instead of polling silently until timeout."""

    client = StalledPoolClient()
    executor = AzureBatchExecutor(
        registry_server="demo.azurecr.io",
        poll_interval=0,
        cloud_client=client,
    )

    events: list = []
    with pytest.raises(RuntimeError) as error:
        asyncio.run(
            executor.execute_tasks([task(0)], progress_callback=events.append)
        )

    assert "no usable nodes" in str(error.value)
    assert "ContainerPullFailed" in str(error.value)
    assert any(
        "unusable" in str(event.payload.get("node_states")) for event in events
    )


def test_azure_executor_times_out_when_no_task_ever_starts() -> None:
    """Report a dedicated error when a healthy pool never starts any task."""

    client = StalledPoolClient(node_state="idle")
    executor = AzureBatchExecutor(
        registry_server="demo.azurecr.io",
        poll_interval=0,
        max_wait_for_first_task_start=0,
        cloud_client=client,
    )

    with pytest.raises(TimeoutError) as error:
        asyncio.run(executor.execute_tasks([task(0)]))

    assert "did not start" in str(error.value) or "started within" in str(
        error.value
    )
    assert "idle" in str(error.value)


class AllocatingPoolClient(FakeCloudClient):
    """Report a pool whose nodes reach `idle` only after several polls."""

    def __init__(self, states: list[str]) -> None:
        super().__init__()
        self.node_states = list(states)
        self.node_polls = 0
        self.batch_service_client = SimpleNamespace(
            task=SimpleNamespace(list=self.list_tasks),
            file=SimpleNamespace(
                get_from_task=lambda *args: [b"fake worker stderr"]
            ),
            compute_node=SimpleNamespace(list=self.list_compute_nodes),
            pool=SimpleNamespace(
                get=lambda pool_name: SimpleNamespace(resize_errors=[])
            ),
        )

    def list_compute_nodes(self, pool_name: str):
        index = min(self.node_polls, len(self.node_states) - 1)
        self.node_polls += 1
        state = self.node_states[index]
        if state == "none":
            return []
        return [SimpleNamespace(state=state, errors=[], start_task_info=None)]


def test_azure_executor_reports_node_allocation_until_nodes_are_ready() -> (
    None
):
    """Emit node-allocation progress between pool creation and task start."""

    client = AllocatingPoolClient(["none", "creating", "starting", "idle"])
    executor = AzureBatchExecutor(
        registry_server="demo.azurecr.io",
        poll_interval=0,
        cloud_client=client,
    )

    events: list = []
    asyncio.run(
        executor.execute_tasks([task(0)], progress_callback=events.append)
    )

    node_events = [
        event for event in events if event.payload.get("stage") == "pool_nodes"
    ]
    assert len(node_events) == 4
    assert "none allocated" not in node_events[0].payload["message"]
    assert node_events[0].payload["usable_nodes"] == 0
    assert node_events[-1].payload["usable_nodes"] == 1
    assert "creating 1" in node_events[1].payload["message"]


def test_azure_executor_times_out_when_pool_never_allocates_nodes() -> None:
    """Fail with a quota-oriented message when allocation never completes."""

    client = AllocatingPoolClient(["creating"])
    executor = AzureBatchExecutor(
        registry_server="demo.azurecr.io",
        poll_interval=0,
        max_wait_for_nodes=0,
        cloud_client=client,
    )

    with pytest.raises(TimeoutError) as error:
        asyncio.run(executor.execute_tasks([task(0)]))

    assert "allocated no usable nodes" in str(error.value)
    assert "quota-blocked resize" in str(error.value)


def test_azure_executor_skips_node_wait_when_disabled() -> None:
    """Allow opting out of the node-allocation poll."""

    client = AllocatingPoolClient(["creating"])
    executor = AzureBatchExecutor(
        registry_server="demo.azurecr.io",
        poll_interval=0,
        wait_for_nodes=False,
        max_wait_for_nodes=0,
        cloud_client=client,
    )

    events: list = []
    results = asyncio.run(
        executor.execute_tasks([task(0)], progress_callback=events.append)
    )

    assert [result.slot_id for result in results] == [0]
    assert not [
        event for event in events if event.payload.get("stage") == "pool_nodes"
    ]


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


def test_study_pool_is_shared_across_scenario_clones_and_cleaned_once() -> (
    None
):
    client = FakeCloudClient()
    executor = AzureBatchExecutor(
        base_name="Shared Study",
        registry_server="demo.azurecr.io",
        chunk_size=1,
        poll_interval=0,
        delete_pool_after=True,
        cloud_client=client,
        client_factory=lambda: client,
    )

    asyncio.run(executor.prepare_study())
    first = executor.clone_for_scenario("first")
    second = executor.clone_for_scenario("second")
    asyncio.run(first.execute_tasks([task(0)]))
    asyncio.run(second.execute_tasks([task(1)]))

    assert first.pool_name == second.pool_name == "shared-study-pool"
    assert first.blob_name == second.blob_name == "shared-study-tasks"
    assert first.base_name != second.base_name
    assert len([call for call in client.calls if call[0] == "pool"]) == 1
    assert len([call for call in client.calls if call[0] == "job"]) == 2
    assert len(client.uploaded) == 2
    assert len(set(client.uploaded)) == 2
    assert not any(call[0] == "delete_pool" for call in client.calls)

    asyncio.run(executor.cleanup_study())

    assert client.calls[-1] == ("delete_pool", "shared-study-pool")


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
