"""The public ``AzureBatchExecutor`` class.

This module owns configuration, client/registry resolution, and the overall
``execute_tasks`` flow. Pool/node health, image publication, and per-job
task lifecycle live in sibling modules; this class is the thin orchestrator
that calls them with its own configuration.
"""

from __future__ import annotations

import asyncio
import re
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from ..cloud_executor import (
    CloudAcceptanceResult,
    CloudAcceptanceTask,
    CloudExecutor,
)
from ..sampler_types import ProgressCallback, ProgressEvent
from . import _job, _node_health
from ._console import capture_cloudops_output
from ._image import (
    build_and_push_image,
    resolve_registry_server,
    upload_image,
)


class AzureBatchExecutor(CloudExecutor):
    """Run cloud acceptance tasks through Blob Storage and Azure Batch.

    ``cloud_client`` and ``client_factory`` exist for local contract tests. When neither is
    supplied, the optional ``cfa-cloudops`` dependency is imported only when the executor
    first needs to contact Azure.
    """

    def __init__(
        self,
        *,
        base_name: str = "calibrationtools",
        pool_name: str | None = None,
        blob_name: str | None = None,
        registry_server: str | None = None,
        image_name: str = "calibrationtools-model",
        image_tag: str = "latest",
        vm_size: str = "STANDARD_D2S_V3",
        max_autoscale_nodes: int = 5,
        task_slots_per_node: int = 1,
        chunk_size: int = 1,
        mount_path: str = "/mnt/batch/tasks/fsmounts",
        command_template: str = (
            "python -m calibrationtools.azure_batch_worker "
            "--tasks {task_file} --blob-container {blob_container} "
            "--mount-path {mount_path}"
        ),
        delete_job_after: bool = True,
        delete_pool_after: bool = False,
        env_path: str | None = ".env",
        use_sp: bool = False,
        use_federated: bool = False,
        build_image: bool = False,
        upload_image: bool = False,
        image_dockerfile: str = "./Dockerfile",
        base_image: str | None = None,
        poll_interval: float = 5.0,
        max_wait: float = 3600.0,
        max_wait_for_first_task_start: float = 900.0,
        wait_for_nodes: bool = True,
        max_wait_for_nodes: float = 1800.0,
        quiet_cloud_output: bool = True,
        _use_shared_study_pool: bool = False,
        _study_pool_ready: bool = False,
        cloud_client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if max_autoscale_nodes <= 0:
            raise ValueError("max_autoscale_nodes must be positive")
        if poll_interval < 0:
            raise ValueError("poll_interval must be non-negative")
        if build_image and upload_image:
            raise ValueError(
                "build_image and upload_image cannot both be true"
            )

        self.base_name = self._safe_name(base_name)
        self.pool_name = pool_name or f"{self.base_name}-pool"
        self.blob_name = blob_name or f"{self.base_name}-tasks"
        self.registry_server = registry_server
        self.image_name = image_name
        self.image_tag = image_tag
        self.vm_size = vm_size
        self.max_autoscale_nodes = max_autoscale_nodes
        self.task_slots_per_node = task_slots_per_node
        self.chunk_size = chunk_size
        self.mount_path = mount_path
        self.command_template = command_template
        self.delete_job_after = delete_job_after
        self.delete_pool_after = delete_pool_after
        self.env_path = env_path
        self.use_sp = use_sp
        self.use_federated = use_federated
        self.build_image = build_image
        self.upload_image = upload_image
        self.image_dockerfile = image_dockerfile
        self.base_image = base_image
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.max_wait_for_first_task_start = max_wait_for_first_task_start
        self.wait_for_nodes = wait_for_nodes
        self.max_wait_for_nodes = max_wait_for_nodes
        self.quiet_cloud_output = quiet_cloud_output
        self._use_shared_study_pool = _use_shared_study_pool
        self._study_pool_ready = _study_pool_ready
        self._cloud_client = cloud_client
        self._client_factory = client_factory
        self._task_blobs: list[str] = []
        self._run_index = 0
        self._reported_notices: set[str] = set()

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
        if not safe:
            raise ValueError(
                "Azure resource names need at least one alphanumeric character"
            )
        return safe

    @contextmanager
    def _quiet(
        self, callback: ProgressCallback | None = None
    ) -> Iterator[None]:
        """Fold ``cfa-cloudops`` console output into the progress stream.

        Args:
            callback (ProgressCallback | None): Observer that receives each
                distinct notice as an ``executor_message`` event.

        Yields:
            None: Control returns to the caller while capture is active.
        """

        with capture_cloudops_output(self.quiet_cloud_output) as lines:
            yield
        for line in lines:
            if line in self._reported_notices:
                continue
            self._reported_notices.add(line)
            self._emit(
                callback,
                f"cfa-cloudops: {line}",
                stage="cloud_notice",
                source_library="cfa-cloudops",
            )

    @property
    def cloud_client(self) -> Any:
        if self._cloud_client is None:
            if self._client_factory is not None:
                self._cloud_client = self._client_factory()
            else:
                try:
                    from cfa.cloudops import CloudClient
                except (
                    ImportError
                ) as exc:  # pragma: no cover - depends on optional extra
                    raise ImportError(
                        "Azure execution requires `calibrationtools[azure]`."
                    ) from exc
                kwargs: dict[str, Any] = {
                    "use_sp": self.use_sp,
                    "use_federated": self.use_federated,
                }
                if self.env_path is not None:
                    kwargs["dotenv_path"] = self.env_path
                self._cloud_client = CloudClient(**kwargs)
        return self._cloud_client

    @property
    def image_uri(self) -> str:
        return f"{self._registry_server()}/{self.image_name}:{self.image_tag}"

    def _registry_server(self) -> str:
        return resolve_registry_server(
            self.registry_server, self._cloud_client
        )

    def clone_for_scenario(self, scenario_name: str) -> "AzureBatchExecutor":
        """Create an isolated executor without repeated image publication.

        Once a study has prepared its pool, clones share only that pool and its
        mounted Blob container. Jobs, task blobs, clients, and mutable task state
        remain scenario-local.
        """

        suffix = self._safe_name(scenario_name)
        return type(self)(
            base_name=f"{self.base_name}-{suffix}",
            pool_name=self.pool_name if self._study_pool_ready else None,
            blob_name=self.blob_name if self._study_pool_ready else None,
            registry_server=self.registry_server,
            image_name=self.image_name,
            image_tag=self.image_tag,
            vm_size=self.vm_size,
            max_autoscale_nodes=self.max_autoscale_nodes,
            task_slots_per_node=self.task_slots_per_node,
            chunk_size=self.chunk_size,
            mount_path=self.mount_path,
            command_template=self.command_template,
            delete_job_after=self.delete_job_after,
            delete_pool_after=(
                False if self._study_pool_ready else self.delete_pool_after
            ),
            env_path=self.env_path,
            use_sp=self.use_sp,
            use_federated=self.use_federated,
            build_image=False,
            upload_image=False,
            image_dockerfile=self.image_dockerfile,
            base_image=self.base_image,
            poll_interval=self.poll_interval,
            max_wait=self.max_wait,
            max_wait_for_first_task_start=self.max_wait_for_first_task_start,
            wait_for_nodes=self.wait_for_nodes,
            max_wait_for_nodes=self.max_wait_for_nodes,
            quiet_cloud_output=self.quiet_cloud_output,
            _use_shared_study_pool=self._study_pool_ready,
            _study_pool_ready=self._study_pool_ready,
            client_factory=self._client_factory,
        )

    async def prepare_study(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Create the shared study pool and its mounted Blob container once.

        Args:
            progress_callback (ProgressCallback | None): Optional observer for
                image, container, and pool setup stages.
        """

        await asyncio.to_thread(
            self._prepare_shared_study_pool, progress_callback
        )

    async def cleanup_study(self) -> None:
        """Delete a completed study's shared pool when configured to do so."""

        await asyncio.to_thread(self._cleanup_shared_study_pool)

    async def execute_tasks(
        self,
        tasks: list[CloudAcceptanceTask],
        *,
        progress_callback: ProgressCallback | None = None,
        on_result: Callable[[CloudAcceptanceResult], None] | None = None,
    ) -> list[CloudAcceptanceResult]:
        if not tasks:
            return []
        job_id: str | None = None
        try:
            job_id = await asyncio.to_thread(
                self._setup_job, tasks, progress_callback
            )
            await asyncio.to_thread(
                self._submit_job, job_id, progress_callback
            )
            results = await asyncio.to_thread(
                self._harvest_results,
                job_id,
                progress_callback,
                on_result,
            )
        except BaseException:
            if job_id is not None:
                await asyncio.to_thread(
                    self._cleanup, job_id, progress_callback
                )
            raise
        await asyncio.to_thread(self._cleanup, job_id, progress_callback)
        return results

    def _emit(
        self,
        callback: ProgressCallback | None,
        message: str,
        **payload: Any,
    ) -> None:
        if callback is not None:
            callback(
                ProgressEvent(
                    event_type="executor_message",
                    payload={
                        "source": "azure_batch",
                        "message": message,
                        **payload,
                    },
                )
            )

    def _setup_job(
        self,
        tasks: list[CloudAcceptanceTask],
        callback: ProgressCallback | None,
    ) -> str:
        if self._use_shared_study_pool:
            if not self._study_pool_ready:
                raise RuntimeError(
                    "Shared Azure study pool was not prepared before scenario work"
                )
        else:
            self._prepare_pool(callback)
        self._task_blobs = self._upload_task_chunks(tasks, callback)
        self._run_index += 1
        job_id = (
            f"{self.base_name}-job-{int(time.time() * 1000)}-{self._run_index}"
        )
        self._emit(
            callback,
            (
                "Using shared Azure study pool"
                if self._use_shared_study_pool
                else "Azure pool ready"
            ),
            pool_name=self.pool_name,
            job_id=job_id,
        )
        return job_id

    def _prepare_shared_study_pool(
        self, callback: ProgressCallback | None = None
    ) -> None:
        """Provision the Blob mount and pool once for a complete study."""

        if self._study_pool_ready:
            return
        self._prepare_pool(callback)
        self._study_pool_ready = True

    def _prepare_pool(self, callback: ProgressCallback | None = None) -> None:
        """Publish the requested image and create this executor's pool."""

        self._emit(
            callback,
            "Authenticating with Azure",
            stage="authenticate",
            pool_name=self.pool_name,
        )
        client = self.cloud_client
        if self.build_image:
            self._emit(
                callback,
                f"Building and pushing image {self.image_uri}",
                stage="image",
            )
            build_and_push_image(
                client,
                registry=self._registry_server(),
                image_name=self.image_name,
                image_tag=self.image_tag,
                image_dockerfile=self.image_dockerfile,
            )
        elif self.upload_image:
            self._emit(
                callback,
                f"Uploading image {self.image_uri}",
                stage="image",
            )
            upload_image(
                client,
                registry=self._registry_server(),
                image_name=self.image_name,
                image_tag=self.image_tag,
            )
        self._emit(
            callback,
            f"Creating Blob container {self.blob_name}",
            stage="blob_container",
            blob_name=self.blob_name,
        )
        with self._quiet(callback):
            client.create_blob_container(self.blob_name)
        self._emit(
            callback,
            f"Creating pool {self.pool_name} ({self.vm_size})",
            stage="pool",
            pool_name=self.pool_name,
        )
        with self._quiet(callback):
            client.create_pool(
                self.pool_name,
                mounts=[self.blob_name],
                container_image_name=self.image_uri,
                vm_size=self.vm_size,
                max_autoscale_nodes=self.max_autoscale_nodes,
                task_slots_per_node=self.task_slots_per_node,
            )
        _node_health.wait_for_pool_nodes(
            client,
            pool_name=self.pool_name,
            wait_for_nodes=self.wait_for_nodes,
            poll_interval=self.poll_interval,
            max_wait_for_nodes=self.max_wait_for_nodes,
            emit=lambda message, **payload: self._emit(
                callback, message, **payload
            ),
        )
        self._emit(
            callback,
            f"Pool {self.pool_name} created",
            stage="pool_ready",
            pool_name=self.pool_name,
        )

    def _cleanup_shared_study_pool(self) -> None:
        """Perform deferred pool cleanup after every study scenario has settled."""

        if self._study_pool_ready and self.delete_pool_after:
            self.cloud_client.delete_pool(self.pool_name)
        self._study_pool_ready = False

    def _upload_task_chunks(
        self,
        tasks: list[CloudAcceptanceTask],
        callback: ProgressCallback | None = None,
    ) -> list[str]:
        chunk_count = -(-len(tasks) // self.chunk_size)  # ceiling division
        self._emit(
            callback,
            f"Uploading {chunk_count} task file"
            + ("" if chunk_count == 1 else "s"),
            stage="upload",
        )
        return _job.upload_task_chunks(
            self.cloud_client,
            tasks,
            base_name=self.base_name,
            blob_name=self.blob_name,
            chunk_size=self.chunk_size,
        )

    def _submit_job(
        self, job_id: str, callback: ProgressCallback | None
    ) -> None:
        _job.submit_job(
            self.cloud_client,
            job_id=job_id,
            pool_name=self.pool_name,
            task_blobs=self._task_blobs,
            command_template=self.command_template,
            blob_name=self.blob_name,
            mount_path=self.mount_path,
            quiet=lambda: self._quiet(callback),
        )
        self._emit(
            callback,
            "Azure tasks submitted",
            job_id=job_id,
            task_count=len(self._task_blobs),
        )

    def _harvest_results(
        self,
        job_id: str,
        callback: ProgressCallback | None = None,
        on_result: Callable[[CloudAcceptanceResult], None] | None = None,
    ) -> list[CloudAcceptanceResult]:
        return _job.harvest_results(
            self.cloud_client,
            job_id=job_id,
            pool_name=self.pool_name,
            task_blobs=self._task_blobs,
            blob_name=self.blob_name,
            poll_interval=self.poll_interval,
            max_wait=self.max_wait,
            max_wait_for_first_task_start=self.max_wait_for_first_task_start,
            quiet_cloud_output=self.quiet_cloud_output,
            emit=lambda message, **payload: self._emit(
                callback, message, **payload
            ),
            on_result=on_result,
        )

    def _cleanup(self, job_id: str, callback: ProgressCallback | None) -> None:
        _job.cleanup_job(
            self.cloud_client,
            job_id=job_id,
            pool_name=self.pool_name,
            delete_job_after=self.delete_job_after,
            delete_pool_after=self.delete_pool_after,
            emit=lambda message, **payload: self._emit(
                callback, message, **payload
            ),
        )
