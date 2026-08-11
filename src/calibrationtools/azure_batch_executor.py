"""Azure Batch implementation of the cloud acceptance-task contract."""

from __future__ import annotations

import asyncio
import os
import pickle
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from .cloud_executor import (
    CloudAcceptanceResult,
    CloudAcceptanceTask,
    CloudExecutor,
)
from .sampler_types import ProgressCallback, ProgressEvent


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
        self._cloud_client = cloud_client
        self._client_factory = client_factory
        self._task_blobs: list[str] = []
        self._run_index = 0

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
        if not safe:
            raise ValueError(
                "Azure resource names need at least one alphanumeric character"
            )
        return safe

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
        """Resolve an ACR server from explicit or cfa-cloudops settings."""

        registry = self.registry_server or os.getenv(
            "AZURE_CONTAINER_REGISTRY_SERVER"
        )
        if registry:
            return registry
        if self._cloud_client is not None:
            try:
                registry = (
                    self._cloud_client.cred.azure_container_registry_endpoint
                )
            except AttributeError:
                registry = None
            if registry:
                return registry
        account = os.getenv("AZURE_CONTAINER_REGISTRY_ACCOUNT")
        if account:
            domain = os.getenv("AZURE_CONTAINER_REGISTRY_DOMAIN", "azurecr.io")
            return f"{account}.{domain}"
        raise ValueError(
            "Container registry server must be configured through "
            "registry_server, AZURE_CONTAINER_REGISTRY_SERVER, or "
            "AZURE_CONTAINER_REGISTRY_ACCOUNT"
        )

    def clone_for_scenario(self, scenario_name: str) -> "AzureBatchExecutor":
        """Create an isolated executor without repeated image publication."""

        suffix = self._safe_name(scenario_name)
        return type(self)(
            base_name=f"{self.base_name}-{suffix}",
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
            delete_pool_after=self.delete_pool_after,
            env_path=self.env_path,
            use_sp=self.use_sp,
            use_federated=self.use_federated,
            build_image=False,
            upload_image=False,
            image_dockerfile=self.image_dockerfile,
            base_image=self.base_image,
            poll_interval=self.poll_interval,
            max_wait=self.max_wait,
            client_factory=self._client_factory,
        )

    async def execute_tasks(
        self,
        tasks: list[CloudAcceptanceTask],
        *,
        progress_callback: ProgressCallback | None = None,
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
            await asyncio.to_thread(
                self._wait_for_completion, job_id, progress_callback
            )
            results = await asyncio.to_thread(self._download_results, job_id)
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
        if self.build_image:
            self._build_and_push_image()
        elif self.upload_image:
            self._upload_image()
        self.cloud_client.create_blob_container(self.blob_name)
        self._task_blobs = self._upload_task_chunks(tasks)
        self.cloud_client.create_pool(
            self.pool_name,
            mounts=[self.blob_name],
            container_image_name=self.image_uri,
            vm_size=self.vm_size,
            max_autoscale_nodes=self.max_autoscale_nodes,
            task_slots_per_node=self.task_slots_per_node,
        )
        self._run_index += 1
        job_id = (
            f"{self.base_name}-job-{int(time.time() * 1000)}-{self._run_index}"
        )
        self._emit(
            callback,
            "Azure pool ready",
            pool_name=self.pool_name,
            job_id=job_id,
        )
        return job_id

    def _upload_task_chunks(
        self, tasks: list[CloudAcceptanceTask]
    ) -> list[str]:
        task_dir = Path(tempfile.mkdtemp(prefix="calibrationtools-azure-"))
        names: list[str] = []
        try:
            width = max(1, len(str((len(tasks) - 1) // self.chunk_size)))
            stamp = int(time.time() * 1000)
            for index, start in enumerate(
                range(0, len(tasks), self.chunk_size)
            ):
                name = f"tasks-{stamp}-{index:0{width}d}.pkl"
                with (task_dir / name).open("wb") as file:
                    pickle.dump(tasks[start : start + self.chunk_size], file)
                names.append(name)
            self.cloud_client.upload_files(
                files=names,
                container_name=self.blob_name,
                local_root_dir=str(task_dir),
                location_in_blob=".",
            )
            return names
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

    def _submit_job(
        self, job_id: str, callback: ProgressCallback | None
    ) -> None:
        self.cloud_client.create_job(
            job_id, pool_name=self.pool_name, exist_ok=True
        )
        width = max(1, len(str(len(self._task_blobs) - 1)))
        for index, task_blob in enumerate(self._task_blobs):
            command = self.command_template.format(
                task_file=task_blob,
                blob_container=self.blob_name,
                mount_path=self.mount_path,
            )
            self.cloud_client.add_task(
                job_name=job_id,
                command_line=command,
                name_suffix=f"-{index:0{width}d}",
            )
        self._emit(
            callback,
            "Azure tasks submitted",
            job_id=job_id,
            task_count=len(self._task_blobs),
        )

    def _wait_for_completion(
        self, job_id: str, callback: ProgressCallback | None
    ) -> None:
        started = time.monotonic()
        while True:
            task_records = list(
                self.cloud_client.batch_service_client.task.list(job_id)
            )
            completed = sum(
                getattr(
                    getattr(task, "execution_info", None), "exit_code", None
                )
                is not None
                for task in task_records
            )
            self._emit(
                callback,
                "Azure task progress",
                job_id=job_id,
                completed=completed,
                total=len(self._task_blobs),
            )
            if len(task_records) >= len(self._task_blobs) and completed >= len(
                self._task_blobs
            ):
                self._raise_for_failed_tasks(job_id, task_records)
                return
            if time.monotonic() - started > self.max_wait:
                raise TimeoutError(
                    f"Azure Batch job {job_id} did not complete in time"
                )
            time.sleep(self.poll_interval)

    def _raise_for_failed_tasks(
        self, job_id: str, task_records: list[Any]
    ) -> None:
        failures = [
            task
            for task in task_records
            if getattr(getattr(task, "execution_info", None), "exit_code", 0)
            not in (None, 0)
        ]
        if not failures:
            return
        failed = failures[0]
        task_id = getattr(failed, "id", "<unknown>")
        exit_code = getattr(
            getattr(failed, "execution_info", None), "exit_code", "?"
        )
        failure_info = getattr(
            getattr(failed, "execution_info", None), "failure_info", None
        )
        detail = getattr(failure_info, "message", None) or str(
            failure_info or "no details"
        )
        try:
            stderr = b"".join(
                self.cloud_client.batch_service_client.file.get_from_task(
                    job_id, task_id, "stderr.txt"
                )
            ).decode("utf-8", errors="replace")
        except Exception:
            stderr = ""
        message = f"Azure Batch job {job_id} failed in task {task_id} (exit {exit_code}): {detail}"
        if stderr:
            message += f"\n\nRepresentative task stderr:\n{stderr[-4000:]}"
        raise RuntimeError(message)

    def _download_results(self, job_id: str) -> list[CloudAcceptanceResult]:
        results: list[CloudAcceptanceResult] = []
        for task_blob in self._task_blobs:
            with tempfile.NamedTemporaryFile(
                "wb", suffix=".pkl", delete=False
            ) as file:
                destination = file.name
            try:
                self.cloud_client.download_file(
                    src_path=f"results-{task_blob}",
                    dest_path=destination,
                    container_name=self.blob_name,
                )
                with open(destination, "rb") as file:
                    results.extend(pickle.load(file))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download Azure result blob results-{task_blob} for job {job_id}: {exc}"
                ) from exc
            finally:
                Path(destination).unlink(missing_ok=True)
        return results

    def _cleanup(self, job_id: str, callback: ProgressCallback | None) -> None:
        cleaned: list[str] = []
        if self.delete_job_after:
            self.cloud_client.delete_job(job_id)
            cleaned.append(f"job {job_id}")
        if self.delete_pool_after:
            self.cloud_client.delete_pool(self.pool_name)
            cleaned.append(f"pool {self.pool_name}")
        if cleaned:
            self._emit(callback, "Azure resources cleaned", resources=cleaned)

    def _build_and_push_image(self) -> None:
        cloud_client = self.cloud_client
        registry = self._registry_server()
        registry_name = registry.removesuffix(".azurecr.io")
        cloud_client.package_and_upload_dockerfile(
            registry_name=registry_name,
            repo_name=self.image_name,
            tag=self.image_tag,
            path_to_dockerfile=self.image_dockerfile,
            use_device_code=False,
        )

    def _upload_image(self) -> None:
        registry = self._registry_server()
        self.cloud_client.upload_docker_image(
            image_name=f"{self.image_name}:{self.image_tag}",
            registry_name=registry.removesuffix(".azurecr.io"),
            repo_name=self.image_name,
            tag=self.image_tag,
            use_device_code=False,
        )
