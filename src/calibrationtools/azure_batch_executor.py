"""Azure Batch implementation of the cloud acceptance-task contract."""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import re
import shutil
import tempfile
import time
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from . import formatting
from .cloud_executor import (
    CloudAcceptanceResult,
    CloudAcceptanceTask,
    CloudExecutor,
)
from .sampler_types import ProgressCallback, ProgressEvent

#: Node states from which a pool can never make progress on queued tasks.
_UNUSABLE_NODE_STATES = frozenset(
    {"unusable", "starttaskfailed", "preempted", "offline", "unknown"}
)

#: Node states in which a node can accept or is already running task work.
_USABLE_NODE_STATES = frozenset({"idle", "running", "leavingpool"})

#: Terminal control sequences emitted by the progress bars in cfa-cloudops.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _clean_notice(text: str) -> str:
    """Reduce a captured ``cfa-cloudops`` line to plain single-line text.

    Captured output still carries the styling and carriage returns of the
    progress bars it came from. Re-emitting that verbatim pushes escape
    sequences into whatever renders the notice, where they show up as stray
    digits and colour codes rather than as styling.

    Args:
        text (str): Raw captured line.

    Returns:
        str: The final frame of the line, without escapes or padding.
    """

    without_ansi = _ANSI_ESCAPE.sub("", text)
    return " ".join(without_ansi.split("\r")[-1].split())


@contextmanager
def _capture_cloudops_output(enabled: bool = True) -> Iterator[list[str]]:
    """Divert ``cfa-cloudops`` console output into a list for re-reporting.

    ``cfa-cloudops`` writes advisories and per-file progress with bare
    ``print`` calls and its own log handlers. Those writes land mid-frame in
    the sampler's live display and force partial redraws, which is why a
    running calibration appears to emit duplicated, truncated progress bars.
    Capturing them keeps the display coherent while preserving the content so
    the caller can surface it as a normal progress notice.

    Args:
        enabled (bool): Whether to capture output. Disabling restores the
            default passthrough behavior for debugging.

    Yields:
        list[str]: Captured lines, appended as they are produced by logging
            and populated from stdout once the block exits.
    """

    if not enabled:
        yield []
        return

    lines: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(_clean_notice(record.getMessage()))

    buffer = StringIO()
    logger = logging.getLogger("cfa")
    handler = _Collector()
    previous_propagate = logger.propagate
    logger.addHandler(handler)
    logger.propagate = False
    try:
        with redirect_stdout(buffer):
            yield lines
    finally:
        logger.removeHandler(handler)
        logger.propagate = previous_propagate
        lines.extend(
            cleaned
            for cleaned in (
                _clean_notice(line) for line in buffer.getvalue().splitlines()
            )
            if cleaned
        )


@contextmanager
def _rich_upload_progress(
    description: str, enabled: bool = True
) -> Iterator[None]:
    """Render blob uploads with a single-line Rich progress bar.

    ``cfa-cloudops`` drives its upload loop with ``tqdm``, which emits a new
    line per refresh once its stream is not a plain terminal. This context
    manager temporarily swaps that iterator for a Rich-backed equivalent so
    upload progress matches the rest of the sampler output.

    Args:
        description (str): Label shown beside the progress bar.
        enabled (bool): Whether to install the bar. Studies own a live display
            of their own, and Rich supports only one at a time, so they pass
            ``False`` to avoid two displays fighting over the console.

    Yields:
        None: Control returns to the caller while the bar is installed.
    """

    if not enabled:
        yield
        return

    try:
        from cfa.cloudops import blob as cloudops_blob
    except Exception:
        yield
        return

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=formatting.get_console(True),
        transient=True,
    )
    original_tqdm = cloudops_blob.tqdm

    def tracked(
        iterable: Iterable[Any], *args: Any, **kwargs: Any
    ) -> Iterator[Any]:
        items = list(iterable)
        task_id = progress.add_task(description, total=len(items))
        for item in items:
            yield item
            progress.advance(task_id)

    cloudops_blob.tqdm = tracked
    try:
        with progress:
            yield
    finally:
        cloudops_blob.tqdm = original_tqdm


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

        with _capture_cloudops_output(self.quiet_cloud_output) as lines:
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
            results = await asyncio.to_thread(
                self._download_results, job_id, progress_callback
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
            self._build_and_push_image()
        elif self.upload_image:
            self._emit(
                callback,
                f"Uploading image {self.image_uri}",
                stage="image",
            )
            self._upload_image()
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
        self._wait_for_pool_nodes(callback)
        self._emit(
            callback,
            f"Pool {self.pool_name} created",
            stage="pool_ready",
            pool_name=self.pool_name,
        )

    def _wait_for_pool_nodes(
        self, callback: ProgressCallback | None = None
    ) -> None:
        """Poll until the pool allocates at least one usable compute node.

        ``create_pool`` returns as soon as Batch accepts the pool definition,
        long before any VM boots, pulls the container image, and runs its start
        task. Polling node states here turns the silent gap between pool
        creation and first task execution into visible progress, and surfaces
        allocation failures instead of letting the job wait for them.

        Args:
            callback (ProgressCallback | None): Optional observer for node
                allocation progress events.

        Returns:
            None: This method does not return a value.

        Raises:
            RuntimeError: If every allocated node reaches a terminal bad state.
            TimeoutError: If no usable node appears within
                ``max_wait_for_nodes``.
        """

        if not self.wait_for_nodes:
            return
        started = time.monotonic()
        while True:
            health = self._pool_health()
            states = health.get("node_states")
            if states is None:
                return
            usable = sum(
                count
                for name, count in states.items()
                if name in _USABLE_NODE_STATES
            )
            elapsed = time.monotonic() - started
            self._emit(
                callback,
                self._node_wait_message(states, usable, elapsed),
                stage="pool_nodes",
                pool_name=self.pool_name,
                usable_nodes=usable,
                elapsed_seconds=elapsed,
                **health,
            )
            if usable:
                return
            self._raise_for_unhealthy_pool(
                f"pool {self.pool_name} startup", health
            )
            if elapsed > self.max_wait_for_nodes:
                detail = "; ".join(
                    list(health.get("node_errors", ()))
                    + list(health.get("resize_errors", ()))
                )
                message = (
                    f"Azure Batch pool {self.pool_name} allocated no usable "
                    f"nodes within {self.max_wait_for_nodes:.0f}s "
                    f"(node states: {self._state_summary(states)}). This "
                    "usually means a quota-blocked resize, an image pull "
                    "failure, or a failing start task."
                )
                if detail:
                    message += f" Details: {detail}"
                raise TimeoutError(message)
            time.sleep(self.poll_interval)

    def _node_wait_message(
        self, states: dict[str, int], usable: int, elapsed: float
    ) -> str:
        """Summarize node allocation progress for one status line."""

        if not states:
            return (
                f"Waiting for pool {self.pool_name} to allocate nodes "
                f"({elapsed:.0f}s)"
            )
        summary = self._state_summary(states)
        if usable:
            return f"Pool {self.pool_name} nodes ready ({summary})"
        return (
            f"Waiting for pool {self.pool_name} nodes: {summary} "
            f"({elapsed:.0f}s)"
        )

    @staticmethod
    def _state_summary(states: dict[str, int]) -> str:
        """Render node state counts in a stable, compact order."""

        return ", ".join(
            f"{name} {count}" for name, count in sorted(states.items())
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
        task_dir = Path(tempfile.mkdtemp(prefix="calibrationtools-azure-"))
        names: list[str] = []
        try:
            width = max(1, len(str((len(tasks) - 1) // self.chunk_size)))
            stamp = int(time.time() * 1000)
            for index, start in enumerate(
                range(0, len(tasks), self.chunk_size)
            ):
                name = f"tasks-{self.base_name}-{stamp}-{index:0{width}d}.pkl"
                with (task_dir / name).open("wb") as file:
                    pickle.dump(tasks[start : start + self.chunk_size], file)
                names.append(name)
            noun = "file" if len(names) == 1 else "files"
            self._emit(
                callback,
                f"Uploading {len(names)} task {noun}",
                stage="upload",
                file_count=len(names),
            )
            with _rich_upload_progress(
                "Uploading task files", enabled=callback is None
            ):
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
        with self._quiet(callback):
            self.cloud_client.create_job(
                job_id, pool_name=self.pool_name, exist_ok=True
            )
        width = max(1, len(str(len(self._task_blobs) - 1)))
        with self._quiet(callback):
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
        total = len(self._task_blobs)
        running_seen = False
        while True:
            task_records = self._list_batch_tasks(job_id)
            completed = sum(
                getattr(
                    getattr(task, "execution_info", None), "exit_code", None
                )
                is not None
                for task in task_records
            )
            running = sum(
                str(getattr(task, "state", "")).lower().endswith("running")
                for task in task_records
            )
            running_seen = running_seen or running > 0 or completed > 0
            health = self._pool_health()
            self._emit(
                callback,
                self._wait_message(completed, running, total, health),
                job_id=job_id,
                completed=completed,
                running=running,
                total=total,
                elapsed_seconds=time.monotonic() - started,
                **health,
            )
            if len(task_records) >= total and completed >= total:
                self._raise_for_failed_tasks(job_id, task_records)
                return
            self._raise_for_unhealthy_pool(f"job {job_id}", health)
            elapsed = time.monotonic() - started
            if (
                not running_seen
                and elapsed > self.max_wait_for_first_task_start
            ):
                raise TimeoutError(
                    f"No Azure Batch task in job {job_id} started within "
                    f"{self.max_wait_for_first_task_start:.0f}s on pool "
                    f"{self.pool_name}. Node states: "
                    f"{health.get('node_states') or 'unknown'}. This usually "
                    "means the pool never allocated usable nodes (quota, "
                    "image pull, or start-task failure)."
                )
            if elapsed > self.max_wait:
                raise TimeoutError(
                    f"Azure Batch job {job_id} did not complete in time"
                )
            time.sleep(self.poll_interval)

    @classmethod
    def _wait_message(
        cls,
        completed: int,
        running: int,
        total: int,
        health: dict[str, Any],
    ) -> str:
        """Summarize job and pool state for one progress line."""

        parts = [f"Azure task progress {completed}/{total}"]
        if running:
            parts.append(f"{running} running")
        states = health.get("node_states")
        if states:
            parts.append("nodes: " + cls._state_summary(states))
        elif states == {}:
            parts.append("nodes: none allocated yet")
        return " • ".join(parts)

    def _pool_health(self) -> dict[str, Any]:
        """Summarize compute-node states and errors for the executor's pool.

        Batch surfaces pool problems (image pull failures, start-task errors,
        quota-blocked resizes) on the nodes and the pool rather than on the
        job, so a job that never starts otherwise looks identical to a job that
        is merely slow.

        Returns:
            dict[str, Any]: Node state counts plus any node or resize errors.
            Keys are omitted when the client does not expose that information.
        """

        batch_client = getattr(self.cloud_client, "batch_service_client", None)
        if batch_client is None:
            return {}
        try:
            nodes = self._list_compute_nodes(batch_client)
        except Exception:
            return {}
        states: dict[str, int] = {}
        errors: list[str] = []
        for node in nodes:
            state = self._state_name(getattr(node, "state", None))
            states[state] = states.get(state, 0) + 1
            for error in getattr(node, "errors", None) or []:
                errors.append(self._error_text(error))
            start_task_info = getattr(node, "start_task_info", None)
            failure_info = getattr(start_task_info, "failure_info", None)
            if failure_info is not None:
                errors.append(f"start task: {self._error_text(failure_info)}")
        health: dict[str, Any] = {"node_states": states}
        if errors:
            health["node_errors"] = sorted(set(errors))[:5]
        resize_errors = self._pool_resize_errors(batch_client)
        if resize_errors:
            health["resize_errors"] = resize_errors
        return health

    def _list_compute_nodes(self, batch_client: Any) -> list[Any]:
        """List pool nodes across supported Azure Batch client versions."""

        if hasattr(batch_client, "list_compute_nodes"):
            return list(batch_client.list_compute_nodes(self.pool_name))
        return list(batch_client.compute_node.list(self.pool_name))

    def _pool_resize_errors(self, batch_client: Any) -> list[str]:
        """Return pool resize errors, which usually indicate quota problems."""

        try:
            if hasattr(batch_client, "get_pool"):
                pool = batch_client.get_pool(self.pool_name)
            else:
                pool = batch_client.pool.get(self.pool_name)
        except Exception:
            return []
        return [
            self._error_text(error)
            for error in getattr(pool, "resize_errors", None) or []
        ]

    def _raise_for_unhealthy_pool(
        self, context: str, health: dict[str, Any]
    ) -> None:
        """Fail fast when every allocated node is in a terminal bad state.

        Args:
            context (str): Phase description included in the error message.
            health (dict[str, Any]): Result of :meth:`_pool_health`.

        Returns:
            None: This method does not return a value.

        Raises:
            RuntimeError: If no allocated node can ever run a task.
        """

        states = health.get("node_states") or {}
        if not states or not set(states) <= _UNUSABLE_NODE_STATES:
            return
        detail = "; ".join(
            list(health.get("node_errors", ()))
            + list(health.get("resize_errors", ()))
        )
        message = (
            f"Azure Batch pool {self.pool_name} has no usable nodes during "
            f"{context} (node states: {self._state_summary(states)})."
        )
        if detail:
            message += f" Details: {detail}"
        raise RuntimeError(message)

    @staticmethod
    def _state_name(state: Any) -> str:
        """Normalize an Azure Batch state enum or string to a plain name."""

        value = getattr(state, "value", state)
        return str(value).rsplit(".", 1)[-1].replace("_", "").lower()

    @staticmethod
    def _error_text(error: Any) -> str:
        """Render an Azure Batch error object as one compact line."""

        code = getattr(error, "code", None)
        message = getattr(error, "message", None)
        message = getattr(message, "value", message)
        text = " ".join(str(part) for part in (code, message) if part)
        return " ".join(text.split()) or str(error)

    def _list_batch_tasks(self, job_id: str) -> list[Any]:
        """List task records across supported Azure Batch client versions."""

        batch_client = self.cloud_client.batch_service_client
        if hasattr(batch_client, "list_tasks"):
            return list(batch_client.list_tasks(job_id))
        return list(batch_client.task.list(job_id))

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
            batch_client = self.cloud_client.batch_service_client
            if hasattr(batch_client, "download_task_file"):
                stream = batch_client.download_task_file(
                    job_id, task_id, "stderr.txt"
                )
            else:
                stream = batch_client.file.get_from_task(
                    job_id, task_id, "stderr.txt"
                )
            stderr = b"".join(stream).decode("utf-8", errors="replace")
        except Exception:
            stderr = ""
        message = f"Azure Batch job {job_id} failed in task {task_id} (exit {exit_code}): {detail}"
        if stderr:
            message += f"\n\nRepresentative task stderr:\n{stderr[-4000:]}"
        raise RuntimeError(message)

    def _download_results(
        self, job_id: str, callback: ProgressCallback | None = None
    ) -> list[CloudAcceptanceResult]:
        results: list[CloudAcceptanceResult] = []
        total = len(self._task_blobs)
        for index, task_blob in enumerate(self._task_blobs, start=1):
            with tempfile.NamedTemporaryFile(
                "wb", suffix=".pkl", delete=False
            ) as file:
                destination = file.name
            try:
                with self._quiet(callback):
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
            self._emit(
                callback,
                f"Downloading Azure results {index}/{total}",
                stage="download",
                job_id=job_id,
                completed=index,
                total=total,
            )
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
        try:
            tags = cloud_client.list_acr_tags(
                registry_name=registry_name, repo_name=self.image_name
            )
        except Exception as exc:
            raise RuntimeError(
                "Azure image publication could not be verified. Authenticate "
                "the container runtime to ACR and push the image before retrying."
            ) from exc
        if self.image_tag not in tags:
            raise RuntimeError(
                "Azure image publication did not produce "
                f"{self.image_name}:{self.image_tag} in {registry}."
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
