from __future__ import annotations

import asyncio
import json
import posixpath
import shlex
import shutil
import sys
import tempfile
import time
from concurrent.futures import Future as ThreadFuture
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Sequence

from mrp import run as mrp_run

from calibrationtools.async_runner import run_coroutine_from_sync
from calibrationtools.exceptions import (
    CloudRunnerStateError,
    SimulationCancelledError,
)
from calibrationtools.json_utils import dumps_json, to_jsonable
from calibrationtools.output_contracts import (
    OutputContract,
    make_output_contract_from_cloud_config,
)
from calibrationtools.run_id import parse_sampler_run_id

from .artifacts import download_blob_to_path_atomic, read_task_log_excerpts
from .backend import DEFAULT_CLOUD_RUNNER_BACKEND, CloudRunnerBackend
from .config import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    CloudRuntimeSettings,
    CloudTaskPayloadSettings,
    load_cloud_model_config,
)
from .formatting import append_task_log_excerpts
from .hooks import CloudRunnerHooks, CloudSessionContext
from .session import CloudSession
from .task_payload import (
    CloudTaskContext,
    ResolvedSharedAsset,
    apply_task_payload_transforms,
    bind_shared_assets_to_session,
    resolve_shared_assets,
    resolve_task_output_dir,
    validate_task_payload_templates,
)
from .tooling import upload_files_quietly

_D_CREATE_CLIENT = DEFAULT_CLOUD_RUNNER_BACKEND.create_cloud_client
_D_GIT_SHA = DEFAULT_CLOUD_RUNNER_BACKEND.git_short_sha
_D_SESSION_ID = DEFAULT_CLOUD_RUNNER_BACKEND.make_session_id
_D_BUILD_IMAGE = DEFAULT_CLOUD_RUNNER_BACKEND.build_local_image
_D_UPLOAD_IMAGE = DEFAULT_CLOUD_RUNNER_BACKEND.upload_local_image
_D_CREATE_POOL = DEFAULT_CLOUD_RUNNER_BACKEND.create_pool_with_blob_mounts
_D_WAIT_POOL = DEFAULT_CLOUD_RUNNER_BACKEND.wait_for_pool_ready
_D_ADD_TASK = DEFAULT_CLOUD_RUNNER_BACKEND.add_batch_task_with_short_id
_D_CANCEL_TASK = DEFAULT_CLOUD_RUNNER_BACKEND.cancel_batch_task
_D_FMT_FAILURE = DEFAULT_CLOUD_RUNNER_BACKEND.format_task_failure_message
_D_FMT_TIMING = DEFAULT_CLOUD_RUNNER_BACKEND.format_task_timing_summary
_D_RESOURCE_NAME = DEFAULT_CLOUD_RUNNER_BACKEND.make_resource_name
_D_SUPPRESS_INFO = DEFAULT_CLOUD_RUNNER_BACKEND.suppress_cloudops_info_output


@dataclass
class _ActiveCloudRun:
    job_name: str
    output_dir: Path
    input_payload: dict[str, Any]
    overall_started: float
    future: ThreadFuture[Any]
    task_id: str | None = None
    cancelled: bool = False
    phase: str = "queued"
    upload_elapsed_seconds: float | None = None
    submitted_at: float | None = None
    completion_seen_at: float | None = None
    download_elapsed_seconds: float | None = None
    task_status: dict[str, Any] | None = None
    controller_task: asyncio.Task[Any] | None = None
    submission_future: ThreadFuture[None] | None = None
    admission_acquired: bool = False


@dataclass(frozen=True)
class _SessionResourceNames:
    input_container: str
    output_container: str
    logs_container: str
    pool_name: str


@dataclass(frozen=True)
class _CompletedRunTask:
    output_dir: Path
    cancelled: bool


@dataclass(frozen=True)
class _TaskOutputLocation:
    command_path: str
    blob_prefix: str


def _normalize_posix_absolute_path(value: str, *, label: str) -> str:
    normalized = posixpath.normpath(value)
    if not normalized.startswith("/"):
        raise ValueError(f"{label} must be an absolute path: {value!r}")
    if normalized.startswith("//"):
        normalized = f"/{normalized.lstrip('/')}"
    return normalized


def _task_output_blob_prefix(
    task_output_dir: str,
    *,
    output_mount_path: str,
) -> str:
    if output_mount_path == "/":
        return task_output_dir.lstrip("/")
    mount_prefix = output_mount_path.rstrip("/")
    if task_output_dir != mount_prefix and not task_output_dir.startswith(
        f"{mount_prefix}/"
    ):
        raise ValueError(
            "cloud.task_payload.task_output_dir must be under "
            f"output_mount_path {output_mount_path!r}: {task_output_dir!r}"
        )
    return task_output_dir.removeprefix(mount_prefix).lstrip("/")


def create_cloud_mrp_runner_from_config(
    config_path: str | Path,
    *,
    generation_count: int,
    max_concurrent_simulations: int,
    read_output_dir: Callable[[Path], Any] | None = None,
    output_filename: str | None = None,
    output_contract: OutputContract[Any] | None = None,
    base_inputs: dict[str, Any] | None = None,
    task_slots_per_node_override: int | None = None,
    print_task_durations: bool | None = None,
    hooks: CloudRunnerHooks | None = None,
    backend: CloudRunnerBackend | None = None,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    mrp_run_func: Callable[..., Any] = mrp_run,
    auto_size_summary: Any | None = None,
) -> "CloudMRPRunner":
    """Create a cloud MRP runner from a model-facing cloud config file."""
    model_config = load_cloud_model_config(config_path)
    runtime_settings = model_config.runtime_settings
    if task_slots_per_node_override is not None:
        runtime_settings = replace(
            runtime_settings,
            task_slots_per_node=task_slots_per_node_override,
        )
    if print_task_durations is not None:
        runtime_settings = replace(
            runtime_settings,
            print_task_durations=print_task_durations,
        )
    output_contract = (
        output_contract
        or make_output_contract_from_cloud_config(model_config.output)
    )
    if read_output_dir is None:
        read_output_dir = output_contract.read_output_dir
    resolved_shared_assets = resolve_shared_assets(
        model_config.shared_assets,
        base_payload=base_inputs,
        config_dir=model_config.config_path.parent.resolve(),
    )
    validate_task_payload_templates(
        model_config.task_payload,
        shared_assets=resolved_shared_assets,
    )
    return CloudMRPRunner(
        config_path,
        generation_count=generation_count,
        max_concurrent_simulations=max_concurrent_simulations,
        repo_root=model_config.build_context,
        dockerfile=model_config.dockerfile,
        runtime_settings=runtime_settings,
        read_output_dir=read_output_dir,
        output_filename=output_filename or output_contract.output_filename,
        shared_assets=resolved_shared_assets,
        task_payload_settings=model_config.task_payload,
        hooks=hooks,
        print_task_durations=runtime_settings.print_task_durations,
        backend=backend,
        poll_interval_seconds=poll_interval_seconds,
        mrp_run_func=mrp_run_func,
        auto_size_summary=auto_size_summary,
    )


def _validate_cloud_runner_inputs(
    *,
    dockerfile: Path,
    max_concurrent_simulations: int,
    settings_loader: Callable[[str | Path], CloudRuntimeSettings] | None,
    runtime_settings: CloudRuntimeSettings | None,
) -> None:
    if not dockerfile.is_file():
        raise FileNotFoundError(
            f"Dockerfile not found at {dockerfile}. "
            "Pass an explicit `dockerfile` (and matching `repo_root`) "
            "when constructing the cloud runner."
        )
    if max_concurrent_simulations < 1:
        raise ValueError(
            "max_concurrent_simulations must be at least 1 "
            f"(got {max_concurrent_simulations})"
        )
    if (settings_loader is None) == (runtime_settings is None):
        raise TypeError(
            "Pass exactly one of settings_loader or runtime_settings."
        )


def _resolve_cloud_runner_backend(
    *,
    backend: CloudRunnerBackend | None,
    create_cloud_client_func: Callable[..., Any],
    git_short_sha_func: Callable[[Path], str],
    make_session_id_func: Callable[[str], str],
    build_local_image_func: Callable[..., str],
    upload_local_image_func: Callable[..., str],
    create_pool_with_blob_mounts_func: Callable[..., None],
    wait_for_pool_ready_func: Callable[..., Any],
    add_batch_task_with_short_id_func: Callable[..., str],
    cancel_batch_task_func: Callable[..., None],
    format_task_failure_message_func: Callable[..., str],
    format_task_timing_summary_func: Callable[..., str],
    make_resource_name_func: Callable[..., str],
    suppress_cloudops_info_output_func: Callable[[], Any],
) -> CloudRunnerBackend:
    if backend is not None:
        return backend
    return CloudRunnerBackend(
        create_cloud_client=create_cloud_client_func,
        git_short_sha=git_short_sha_func,
        make_session_id=make_session_id_func,
        build_local_image=build_local_image_func,
        upload_local_image=upload_local_image_func,
        create_pool_with_blob_mounts=create_pool_with_blob_mounts_func,
        wait_for_pool_ready=wait_for_pool_ready_func,
        add_batch_task_with_short_id=add_batch_task_with_short_id_func,
        cancel_batch_task=cancel_batch_task_func,
        format_task_failure_message=format_task_failure_message_func,
        format_task_timing_summary=format_task_timing_summary_func,
        make_resource_name=make_resource_name_func,
        suppress_cloudops_info_output=suppress_cloudops_info_output_func,
    )


def _validate_cloud_runtime_settings(settings: CloudRuntimeSettings) -> None:
    if settings.jobs_per_session < 1:
        raise ValueError("jobs_per_session must be at least 1")
    if settings.task_slots_per_node < 1:
        raise ValueError("task_slots_per_node must be at least 1")
    if settings.pool_max_nodes < 1:
        raise ValueError("pool_max_nodes must be at least 1")
    if settings.pool_auto_scale_evaluation_interval_minutes < 5:
        raise ValueError(
            "pool_auto_scale_evaluation_interval_minutes must be at least 5"
        )
    if settings.dispatch_buffer < 0:
        raise ValueError("dispatch_buffer must be at least 0")
    if settings.max_parallel_output_downloads < 1:
        raise ValueError("max_parallel_output_downloads must be at least 1")


class CloudMRPRunner:
    """Run one MRP-backed model through the shared cloud execution path."""

    prefer_simulate_async = True

    def __init__(
        self,
        config_path: str | Path,
        *,
        generation_count: int,
        max_concurrent_simulations: int,
        repo_root: Path,
        dockerfile: Path,
        read_output_dir: Callable[[Path], Any],
        settings_loader: Callable[[str | Path], CloudRuntimeSettings]
        | None = None,
        runtime_settings: CloudRuntimeSettings | None = None,
        output_filename: str = "output.csv",
        shared_assets: Sequence[ResolvedSharedAsset] = (),
        task_payload_settings: CloudTaskPayloadSettings | None = None,
        hooks: CloudRunnerHooks | None = None,
        print_task_durations: bool = False,
        backend: CloudRunnerBackend | None = None,
        create_cloud_client_func: Callable[..., Any] = _D_CREATE_CLIENT,
        git_short_sha_func: Callable[[Path], str] = _D_GIT_SHA,
        make_session_id_func: Callable[[str], str] = _D_SESSION_ID,
        build_local_image_func: Callable[..., str] = _D_BUILD_IMAGE,
        upload_local_image_func: Callable[..., str] = _D_UPLOAD_IMAGE,
        create_pool_with_blob_mounts_func: Callable[
            ..., None
        ] = _D_CREATE_POOL,
        wait_for_pool_ready_func: Callable[..., Any] = _D_WAIT_POOL,
        add_batch_task_with_short_id_func: Callable[..., str] = _D_ADD_TASK,
        cancel_batch_task_func: Callable[..., None] = _D_CANCEL_TASK,
        format_task_failure_message_func: Callable[..., str] = _D_FMT_FAILURE,
        format_task_timing_summary_func: Callable[..., str] = _D_FMT_TIMING,
        make_resource_name_func: Callable[..., str] = _D_RESOURCE_NAME,
        suppress_cloudops_info_output_func: Callable[
            [], Any
        ] = _D_SUPPRESS_INFO,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        controller_start_timeout_seconds: float | None = 10.0,
        mrp_run_func: Callable[..., Any] = mrp_run,
        auto_size_summary: Any | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = Path(repo_root)
        self.dockerfile = Path(dockerfile)
        _validate_cloud_runner_inputs(
            dockerfile=self.dockerfile,
            max_concurrent_simulations=max_concurrent_simulations,
            settings_loader=settings_loader,
            runtime_settings=runtime_settings,
        )
        self.generation_count = generation_count
        self.max_concurrent_simulations = max_concurrent_simulations
        self._load_runtime_settings = settings_loader
        self._read_output_dir_callback = read_output_dir
        self._output_filename = output_filename
        self._resolved_shared_assets = tuple(shared_assets)
        self._session_shared_assets: tuple[ResolvedSharedAsset, ...] = ()
        self._task_payload_settings = (
            task_payload_settings or CloudTaskPayloadSettings()
        )
        self._hooks = hooks or CloudRunnerHooks()
        self.print_task_durations = print_task_durations

        self._backend = _resolve_cloud_runner_backend(
            backend=backend,
            create_cloud_client_func=create_cloud_client_func,
            git_short_sha_func=git_short_sha_func,
            make_session_id_func=make_session_id_func,
            build_local_image_func=build_local_image_func,
            upload_local_image_func=upload_local_image_func,
            create_pool_with_blob_mounts_func=create_pool_with_blob_mounts_func,
            wait_for_pool_ready_func=wait_for_pool_ready_func,
            add_batch_task_with_short_id_func=add_batch_task_with_short_id_func,
            cancel_batch_task_func=cancel_batch_task_func,
            format_task_failure_message_func=format_task_failure_message_func,
            format_task_timing_summary_func=format_task_timing_summary_func,
            make_resource_name_func=make_resource_name_func,
            suppress_cloudops_info_output_func=(
                suppress_cloudops_info_output_func
            ),
        )
        self._bind_cloud_runner_backend(self._backend)
        self._initialize_controller_state(
            poll_interval_seconds=poll_interval_seconds,
            controller_start_timeout_seconds=(
                controller_start_timeout_seconds
            ),
        )
        self._mrp_run = mrp_run_func
        self.auto_size_summary = auto_size_summary
        self.settings = self._load_and_validate_runtime_settings(
            runtime_settings
        )
        self._validate_shared_assets_before_client_creation()
        validate_task_payload_templates(
            self._task_payload_settings,
            shared_assets=self._resolved_shared_assets,
        )
        self.client = self._create_cloud_client(
            keyvault=self.settings.keyvault
        )
        self.session = self._initialize_cloud_session()

    def _bind_cloud_runner_backend(
        self,
        backend: CloudRunnerBackend,
    ) -> None:
        self._create_cloud_client = backend.create_cloud_client
        self._git_short_sha = backend.git_short_sha
        self._make_session_id = backend.make_session_id
        self._build_local_image = backend.build_local_image
        self._upload_local_image = backend.upload_local_image
        self._create_pool_with_blob_mounts = (
            backend.create_pool_with_blob_mounts
        )
        self._wait_for_pool_ready = backend.wait_for_pool_ready
        self._add_batch_task_with_short_id = (
            backend.add_batch_task_with_short_id
        )
        self._cancel_batch_task = backend.cancel_batch_task
        self._format_task_failure_message = backend.format_task_failure_message
        self._format_task_timing_summary = backend.format_task_timing_summary
        self._make_resource_name = backend.make_resource_name
        self._suppress_cloudops_info_output = (
            backend.suppress_cloudops_info_output
        )

    def _load_and_validate_runtime_settings(
        self,
        runtime_settings: CloudRuntimeSettings | None,
    ) -> CloudRuntimeSettings:
        if runtime_settings is not None:
            settings = runtime_settings
        else:
            assert self._load_runtime_settings is not None
            settings = self._load_runtime_settings(self.config_path)
        _validate_cloud_runtime_settings(settings)
        return settings

    def _validate_shared_assets_before_client_creation(self) -> None:
        for asset in self._resolved_shared_assets:
            if not asset.source_path.exists():
                raise FileNotFoundError(
                    f"cloud.shared_assets.{asset.name}.source_path not found: "
                    f"{asset.source_path}"
                )
            if asset.is_dir != asset.source_path.is_dir():
                raise FileNotFoundError(
                    f"cloud.shared_assets.{asset.name}.source_path changed type: "
                    f"{asset.source_path}"
                )

    def _initialize_controller_state(
        self,
        *,
        poll_interval_seconds: float,
        controller_start_timeout_seconds: float | None,
    ) -> None:
        self._poll_interval_seconds = poll_interval_seconds
        self._controller_start_timeout_seconds = (
            controller_start_timeout_seconds
        )
        self._run_state_lock = Lock()
        self._active_runs: dict[str, _ActiveCloudRun] = {}
        self._controller_start_lock = Lock()
        self._controller_ready = Event()
        self._controller_thread: Thread | None = None
        self._controller_loop: asyncio.AbstractEventLoop | None = None
        self._controller_tasks: list[asyncio.Task[Any]] = []
        self._controller_failure: BaseException | None = None
        self._admission_semaphore: asyncio.Semaphore | None = None
        self._inflight_semaphore: asyncio.Semaphore | None = None
        self._download_semaphore: asyncio.Semaphore | None = None
        self._closed = False

    def simulate(
        self,
        params: dict[str, Any],
        *,
        input_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        run_id: str | None = None,
    ) -> Any:
        return run_coroutine_from_sync(
            lambda: self.simulate_async(
                params,
                input_path=input_path,
                output_dir=output_dir,
                run_id=run_id,
            )
        )

    async def simulate_async(
        self,
        params: dict[str, Any],
        *,
        input_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        run_id: str | None = None,
    ) -> Any:
        if output_dir is None:
            raise ValueError("Cloud runner requires an output_dir.")
        if not run_id:
            raise ValueError("Cloud runner requires a run_id.")

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        input_payload = self._load_input_payload(
            params,
            input_path=input_path,
            run_id=run_id,
        )
        future: ThreadFuture[Any] = ThreadFuture()

        # _register_active_run picks a job name and inserts the run under
        # the same lock, so two concurrent registrations cannot both
        # observe the same "least busy" job. It also checks _closed so we
        # cannot slip a run past a concurrent close().
        job_name = self._register_active_run(
            run_id,
            output_dir=output_dir_path,
            input_payload=input_payload,
            overall_started=time.monotonic(),
            future=future,
        )

        try:
            prepared_payload = self._prepare_input_payload(
                input_payload,
                run_id=run_id,
                job_name=job_name,
            )
            if not self._set_active_run_input_payload(
                run_id,
                prepared_payload,
            ):
                raise SimulationCancelledError(run_id)
            if self._is_run_cancelled(run_id):
                raise SimulationCancelledError(run_id)
            self._ensure_controller_started()
            self._raise_controller_failure()
            await self._submit_run_async(run_id)
        except asyncio.CancelledError:
            self.cancel_run(run_id)
            raise
        except BaseException as exc:
            self._resolve_run_exception(run_id, exc)
            raise

        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            self.cancel_run(run_id)
            raise

    def dispatch_buffer_size(self) -> int:
        return self.settings.dispatch_buffer

    def _read_output_dir_for_context(
        self,
        output_dir: Path,
        context: CloudTaskContext,
    ) -> Any:
        hooks = getattr(self, "_hooks", CloudRunnerHooks())
        try:
            hook_value = hooks.read_output_dir(output_dir, context)
        except Exception as exc:
            raise RuntimeError(
                f"run {context.run_id}: output hook failed"
            ) from exc
        if hook_value is not NotImplemented:
            return hook_value
        return self._read_output_dir_callback(output_dir)

    def _load_input_payload(
        self,
        params: dict[str, Any],
        *,
        input_path: str | Path | None,
        run_id: str,
    ) -> dict[str, Any]:
        if input_path is None:
            input_payload = to_jsonable(params)
        else:
            loaded = json.loads(Path(input_path).read_text())
            if not isinstance(loaded, dict):
                raise ValueError("Cloud runner input JSON must be an object.")
            input_payload = to_jsonable(loaded)
        input_payload.setdefault("run_id", run_id)
        return input_payload

    def _prepare_input_payload(
        self,
        input_payload: dict[str, Any],
        *,
        run_id: str,
        job_name: str,
    ) -> dict[str, Any]:
        context = self._build_task_context(run_id, job_name=job_name)
        transformed = apply_task_payload_transforms(
            input_payload,
            self._task_payload_settings,
            context,
        )
        try:
            hooked = self._hooks.prepare_task_payload(transformed, context)
        except Exception as exc:
            raise RuntimeError(
                f"run {run_id}: task payload hook failed"
            ) from exc
        if not isinstance(hooked, dict):
            raise TypeError(
                f"run {run_id}: task payload hook must return a dict"
            )
        return to_jsonable(hooked)

    def _build_task_context(
        self,
        run_id: str,
        *,
        job_name: str,
    ) -> CloudTaskContext:
        output_mount_path = getattr(
            self.session,
            "output_mount_path",
            "/cloud-output",
        )
        input_mount_path = getattr(
            self.session, "input_mount_path", "/cloud-input"
        )
        logs_mount_path = getattr(
            self.session, "logs_mount_path", "/cloud-logs"
        )
        remote_output_dir_func = getattr(
            self.session, "remote_output_dir", None
        )
        remote_output_dir = (
            remote_output_dir_func(run_id)
            if callable(remote_output_dir_func)
            else run_id
        )
        default_task_output_dir = (
            f"{output_mount_path.rstrip('/')}/{remote_output_dir}"
        )
        base_context = CloudTaskContext(
            run_id=run_id,
            session_id=getattr(self.session, "session_id", ""),
            job_name=job_name,
            input_mount_path=input_mount_path,
            output_mount_path=output_mount_path,
            logs_mount_path=logs_mount_path,
            task_output_dir=default_task_output_dir,
            shared_assets=getattr(self, "_session_shared_assets", ()),
        )
        task_output_dir = resolve_task_output_dir(
            getattr(
                self,
                "_task_payload_settings",
                CloudTaskPayloadSettings(),
            ),
            base_context,
            default_task_output_dir=default_task_output_dir,
        )
        return replace(base_context, task_output_dir=task_output_dir)

    def close(self) -> None:
        with self._run_state_lock:
            self._closed = True
            active_run_ids = list(self._active_runs)

        for run_id in active_run_ids:
            self.cancel_run(run_id)

        self._request_controller_shutdown()

    def cancel_run(self, run_id: str) -> None:
        with self._run_state_lock:
            state = self._active_runs.get(run_id)
            if state is None:
                return
            state.cancelled = True
            task_id = state.task_id
            job_name = state.job_name
            submission_future = state.submission_future
            controller_task = state.controller_task

        if task_id is not None:
            self._cancel_batch_task(
                batch_client=self.client.batch_service_client,
                job_name=job_name,
                task_id=task_id,
            )
            return

        if submission_future is not None:
            submission_future.cancel()

        if controller_task is not None:
            loop = self._controller_loop
            if loop is not None:
                loop.call_soon_threadsafe(controller_task.cancel)
            return

        self._resolve_run_cancelled(run_id)

    def _ensure_controller_started(self) -> None:
        if self._controller_loop is not None and self._controller_thread:
            return

        with self._controller_start_lock:
            if self._controller_loop is not None and self._controller_thread:
                return

            self._controller_ready.clear()
            self._controller_failure = None
            self._controller_thread = Thread(
                target=self._controller_main,
                name="cloud-runner-controller",
                daemon=True,
            )
            self._controller_thread.start()

        if not self._controller_ready.wait(
            timeout=self._controller_start_timeout_seconds
        ):
            raise RuntimeError(
                "Timed out starting cloud runner controller after "
                f"{self._controller_start_timeout_seconds}s. Increase "
                "`controller_start_timeout_seconds` (or set it to None to "
                "disable the bootstrap timeout) if the host is under heavy "
                "load."
            )

        self._raise_controller_failure()
        if self._controller_loop is None:
            raise RuntimeError("Cloud runner controller failed to start.")

    def _controller_main(self) -> None:
        loop = asyncio.new_event_loop()
        controller_failure: BaseException | None = None
        try:
            asyncio.set_event_loop(loop)
            self._admission_semaphore = asyncio.Semaphore(
                self.max_concurrent_simulations + self.dispatch_buffer_size()
            )
            self._inflight_semaphore = asyncio.Semaphore(
                self.max_concurrent_simulations
            )
            self._download_semaphore = asyncio.Semaphore(
                self.settings.max_parallel_output_downloads
            )
            with self._run_state_lock:
                self._controller_loop = loop
                self._controller_tasks = []
            loop.call_soon(self._controller_ready.set)
            loop.run_forever()
        except BaseException as exc:  # pragma: no cover - defensive
            controller_failure = exc
            self._controller_failure = exc
            self._controller_ready.set()
        finally:
            pending = [
                task for task in asyncio.all_tasks(loop) if not task.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()
            with self._run_state_lock:
                if self._controller_loop is loop:
                    self._controller_loop = None
                self._controller_thread = None
                self._controller_tasks = []
            self._admission_semaphore = None
            self._inflight_semaphore = None
            self._download_semaphore = None
            self._controller_ready.set()
            if controller_failure is not None:
                self._fail_controller(controller_failure)

    async def _submit_run_async(self, run_id: str) -> None:
        controller_loop = self._controller_loop
        if controller_loop is None:
            raise RuntimeError("Cloud runner controller is unavailable.")

        submission_future = asyncio.run_coroutine_threadsafe(
            self._submit_run_on_controller(run_id),
            controller_loop,
        )
        with self._run_state_lock:
            state = self._active_runs.get(run_id)
            if state is not None:
                state.submission_future = submission_future

        try:
            while True:
                with self._run_state_lock:
                    state = self._active_runs.get(run_id)
                    if state is None:
                        return
                    if state.controller_task is not None:
                        return
                if submission_future.done():
                    await asyncio.wrap_future(submission_future)
                    return
                await asyncio.sleep(min(self._poll_interval_seconds, 0.1))
        finally:
            with self._run_state_lock:
                state = self._active_runs.get(run_id)
                if (
                    state is not None
                    and state.submission_future is submission_future
                ):
                    state.submission_future = None

    async def _submit_run_on_controller(self, run_id: str) -> None:
        admission_semaphore = self._admission_semaphore
        if admission_semaphore is None:
            raise RuntimeError("Cloud runner controller is unavailable.")

        admission_acquired = False
        try:
            await admission_semaphore.acquire()
            admission_acquired = True
            with self._run_state_lock:
                state = self._active_runs.get(run_id)
                if state is None:
                    admission_semaphore.release()
                    return
                state.admission_acquired = True

            if self._closed or self._is_run_cancelled(run_id):
                self._resolve_run_cancelled(run_id)
                return

            controller_task = asyncio.create_task(
                self._execute_run(run_id),
                name=f"cloud-run-{run_id}",
            )
            with self._run_state_lock:
                state = self._active_runs.get(run_id)
                if state is None:
                    controller_task.cancel()
                    return
                state.controller_task = controller_task
            self._track_controller_task(controller_task)
        except asyncio.CancelledError:
            self._resolve_run_cancelled(run_id)
            raise
        except Exception as exc:
            if admission_acquired:
                self._resolve_run_exception(run_id, exc)
            raise

    def _request_controller_shutdown(self) -> None:
        loop = self._controller_loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._shutdown_controller_if_idle)
            except RuntimeError:
                # The loop may already be closed by the time close() runs
                # after a fast terminal task path.
                pass

    def _rollback_partial_session(
        self,
        *,
        session_id: str,
        pool_name: str | None,
        container_names: Sequence[str],
        job_names: Sequence[str],
    ) -> list[str]:
        """Best-effort teardown of resources created before a setup failure.

        Returns a list of human-readable failure descriptions for resources
        that could not be deleted. Callers should embed these in the raised
        error so operators can clean up manually when rollback itself fails.
        """
        failures: list[str] = []
        # Tear down in reverse dependency order: jobs before the pool they
        # belong to, and the pool before the storage containers it mounts.
        for job_name in job_names:
            try:
                self.client.delete_job(job_name)
            except Exception as exc:
                failures.append(f"job:{job_name}: {exc}")
        if pool_name is not None:
            try:
                self.client.delete_pool(pool_name)
            except Exception as exc:
                failures.append(f"pool:{pool_name}: {exc}")
        for container_name in container_names:
            try:
                self.client.blob_service_client.delete_container(
                    container_name
                )
            except Exception as exc:
                failures.append(f"container:{container_name}: {exc}")
        print(
            (
                f"[cloud-run] rolled back partial session {session_id}: "
                f"jobs={list(job_names)}, pool={pool_name}, "
                f"containers={list(container_names)}"
            ),
            file=sys.stderr,
            flush=True,
        )
        return failures

    def _handle_partial_session_failure(
        self,
        *,
        exc: BaseException,
        session_id: str,
        pool_name: str,
        created_pool: bool,
        created_containers: Sequence[str],
        created_jobs: Sequence[str],
        tag: str,
    ) -> RuntimeError | None:
        """Rollback partial resources and preserve control-flow exceptions."""
        rollback_failures = self._rollback_partial_session(
            session_id=session_id,
            pool_name=pool_name if created_pool else None,
            container_names=created_containers,
            job_names=created_jobs,
        )
        detail = (
            f"session_id={session_id}, "
            f"pool={pool_name if created_pool else 'not-created'}, "
            f"containers={list(created_containers)}, "
            f"jobs={list(created_jobs)}, "
            f"image_tag={tag}"
        )
        if rollback_failures:
            detail += f"; rollback_failures={rollback_failures}"

        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            if rollback_failures:
                print(
                    (
                        "[cloud-run] rollback failures after startup "
                        f"interruption ({detail})"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            return None

        return RuntimeError(
            f"Cloud session initialization failed ({detail}): {exc}"
        )

    def _initialize_cloud_session(self) -> CloudSession:
        tag = self._git_short_sha(self.repo_root)
        session_id = self._make_session_id(tag)
        remote_image_ref = self._build_cloud_image(tag)
        resource_names = self._make_session_resource_names(session_id)
        created_containers: list[str] = []
        created_pool = False
        created_jobs: list[str] = []
        self._session_shared_assets = bind_shared_assets_to_session(
            self._resolved_shared_assets,
            session_id=session_id,
            input_mount_path=self.settings.input_mount_path,
        )

        try:
            self._hooks.prepare_session(
                CloudSessionContext(
                    session_id=session_id,
                    input_mount_path=self.settings.input_mount_path,
                    output_mount_path=self.settings.output_mount_path,
                    logs_mount_path=self.settings.logs_mount_path,
                    shared_assets=self._session_shared_assets,
                )
            )
            self._create_session_containers(
                resource_names,
                created_containers=created_containers,
            )
            self._upload_session_shared_assets(resource_names.input_container)
            self._create_session_pool(
                resource_names.pool_name,
                self._build_session_mounts(resource_names),
                remote_image_ref,
            )
            created_pool = True
            self._wait_for_session_pool(resource_names.pool_name)
            shared_job_names = self._create_session_jobs(
                session_id,
                resource_names.pool_name,
                resource_names.logs_container,
                created_jobs=created_jobs,
            )
        except BaseException as exc:
            wrapped_error = self._handle_partial_session_failure(
                exc=exc,
                session_id=session_id,
                pool_name=resource_names.pool_name,
                created_pool=created_pool,
                created_containers=created_containers,
                created_jobs=created_jobs,
                tag=tag,
            )
            if wrapped_error is None:
                raise
            raise wrapped_error from exc

        job_names = self._build_generation_job_map(shared_job_names)
        self._print_session_startup_summary(
            pool_name=resource_names.pool_name,
            job_names=job_names,
            remote_image_ref=remote_image_ref,
        )
        return self._build_cloud_session(
            tag=tag,
            session_id=session_id,
            resource_names=resource_names,
            remote_image_ref=remote_image_ref,
            job_names=job_names,
        )

    def _build_cloud_image(self, tag: str) -> str:
        local_image_ref = self._build_local_image(
            repo_root=self.repo_root,
            dockerfile=self.dockerfile,
            local_image=self.settings.local_image,
            tag=tag,
        )
        return self._upload_local_image(
            client=self.client,
            local_image_ref=local_image_ref,
            repository=self.settings.repository,
            tag=tag,
        )

    def _make_session_resource_names(
        self,
        session_id: str,
    ) -> _SessionResourceNames:
        return _SessionResourceNames(
            input_container=self._make_resource_name(
                self.settings.input_container_prefix,
                session_id,
                max_length=63,
            ),
            output_container=self._make_resource_name(
                self.settings.output_container_prefix,
                session_id,
                max_length=63,
            ),
            logs_container=self._make_resource_name(
                self.settings.logs_container_prefix,
                session_id,
                max_length=63,
            ),
            pool_name=self._make_resource_name(
                self.settings.pool_prefix,
                session_id,
                max_length=64,
            ),
        )

    def _create_session_containers(
        self,
        resource_names: _SessionResourceNames,
        *,
        created_containers: list[str],
    ) -> None:
        for container_name in (
            resource_names.input_container,
            resource_names.output_container,
            resource_names.logs_container,
        ):
            self.client.create_blob_container(container_name)
            created_containers.append(container_name)

    def _upload_session_shared_assets(self, input_container: str) -> None:
        for asset in self._session_shared_assets:
            if asset.is_dir:
                files = [
                    str(path.relative_to(asset.source_path))
                    for path in asset.source_path.rglob("*")
                    if path.is_file()
                ]
                local_root_dir = str(asset.source_path)
            else:
                files = asset.source_path.name
                local_root_dir = str(asset.source_path.parent)

            if not files:
                continue
            with self._suppress_cloudops_info_output():
                upload_files_quietly(
                    self.client,
                    files=files,
                    container_name=input_container,
                    local_root_dir=local_root_dir,
                    location_in_blob=asset.remote_blob_dir,
                )

    def _build_session_mounts(
        self,
        resource_names: _SessionResourceNames,
    ) -> list[dict[str, str]]:
        return [
            {
                "source": resource_names.input_container,
                "target": self.settings.input_mount_path.lstrip("/"),
            },
            {
                "source": resource_names.output_container,
                "target": self.settings.output_mount_path.lstrip("/"),
            },
            {
                "source": resource_names.logs_container,
                "target": self.settings.logs_mount_path.lstrip("/"),
            },
        ]

    def _create_session_pool(
        self,
        pool_name: str,
        mounts: list[dict[str, str]],
        remote_image_ref: str,
    ) -> None:
        self._create_pool_with_blob_mounts(
            client=self.client,
            pool_name=pool_name,
            mounts=mounts,
            container_image_name=remote_image_ref,
            vm_size=self.settings.vm_size,
            target_dedicated_nodes=self.settings.pool_max_nodes,
            task_slots_per_node=self.settings.task_slots_per_node,
            auto_scale_evaluation_interval_minutes=(
                self.settings.pool_auto_scale_evaluation_interval_minutes
            ),
        )

    def _wait_for_session_pool(self, pool_name: str) -> None:
        self._wait_for_pool_ready(
            batch_client=self.client.batch_service_client,
            pool_name=pool_name,
            timeout_minutes=self.settings.pool_ready_timeout_minutes,
        )

    def _create_session_jobs(
        self,
        session_id: str,
        pool_name: str,
        logs_container: str,
        *,
        created_jobs: list[str],
    ) -> list[str]:
        shared_job_names: list[str] = []
        for job_index in range(1, self.settings.jobs_per_session + 1):
            job_name = self._make_resource_name(
                self.settings.job_prefix,
                f"{session_id}-j{job_index}",
                max_length=64,
            )
            self.client.create_job(
                job_name=job_name,
                pool_name=pool_name,
                save_logs_to_blob=logs_container,
                logs_folder=f"{session_id}/{job_name}",
                verify_pool=False,
            )
            created_jobs.append(job_name)
            shared_job_names.append(job_name)
        return shared_job_names

    def _build_generation_job_map(
        self,
        shared_job_names: list[str],
    ) -> dict[str, list[str]]:
        return {
            str(generation): list(shared_job_names)
            for generation in range(self.generation_count)
        }

    def _build_cloud_session(
        self,
        *,
        tag: str,
        session_id: str,
        resource_names: _SessionResourceNames,
        remote_image_ref: str,
        job_names: dict[str, list[str]],
    ) -> CloudSession:
        return CloudSession(
            keyvault=self.settings.keyvault,
            session_id=session_id,
            image_tag=tag,
            remote_image_ref=remote_image_ref,
            pool_name=resource_names.pool_name,
            job_names=job_names,
            input_container=resource_names.input_container,
            output_container=resource_names.output_container,
            logs_container=resource_names.logs_container,
            task_mrp_config_path=self.settings.task_mrp_config_path,
            input_mount_path=self.settings.input_mount_path,
            output_mount_path=self.settings.output_mount_path,
            logs_mount_path=self.settings.logs_mount_path,
            task_timeout_minutes=self.settings.task_timeout_minutes,
            print_task_durations=(
                self.print_task_durations or self.settings.print_task_durations
            ),
        )

    def _print_session_startup_summary(
        self,
        *,
        pool_name: str,
        job_names: dict[str, list[str]],
        remote_image_ref: str,
    ) -> None:
        unique_job_count = len(
            {job_name for names in job_names.values() for job_name in names}
        )
        max_task_capacity = (
            self.settings.pool_max_nodes * self.settings.task_slots_per_node
        )
        print(
            (
                f"[cloud-run] created pool {pool_name} "
                f"(vm_size={self.settings.vm_size}, "
                f"max_nodes={self.settings.pool_max_nodes}, "
                f"task_slots_per_node={self.settings.task_slots_per_node}, "
                f"max_task_capacity={max_task_capacity}, "
                f"scaling=auto(max_nodes={self.settings.pool_max_nodes}, "
                f"min_nodes=0, "
                f"interval={self.settings.pool_auto_scale_evaluation_interval_minutes}m), "
                f"image={remote_image_ref})"
            ),
            file=sys.stderr,
            flush=True,
        )
        if self.auto_size_summary is not None:
            summary = self.auto_size_summary
            print(
                (
                    "[cloud-run] auto-size "
                    f"measured_peak_rss={summary.measured_task_peak_rss_bytes} bytes, "
                    f"vm_ram={summary.vm_memory_bytes} bytes, "
                    f"reserve={summary.reserve:.0%}, "
                    f"task_slots_per_node={summary.task_slots_per_node}, "
                    f"max_concurrent_simulations_total={self.max_concurrent_simulations}"
                ),
                file=sys.stderr,
                flush=True,
            )
        print(
            (
                f"[cloud-run] created {unique_job_count} reusable job(s) for "
                f"{self.generation_count} generation(s) "
                f"({self.settings.jobs_per_session} shared job(s))"
            ),
            file=sys.stderr,
            flush=True,
        )

    def _register_active_run(
        self,
        run_id: str,
        job_name: str | None = None,
        *,
        output_dir: Path,
        input_payload: dict[str, Any],
        overall_started: float,
        future: ThreadFuture[Any],
    ) -> str:
        """Register an active run and return the assigned job name.

        When ``job_name`` is ``None`` the job is selected from the current
        active-run snapshot under the same lock that inserts the new entry
        so two concurrent registrations on different threads cannot both
        observe the same "least busy" job and pile onto it. Returning the
        chosen name lets the caller thread it through any downstream
        overrides without re-querying.
        """
        with self._run_state_lock:
            if self._closed:
                raise RuntimeError("Cloud runner is closed.")
            if run_id in self._active_runs:
                # Reject duplicates under the same lock that inserts new
                # entries so we cannot race with another caller. Allowing
                # the overwrite would orphan the first caller's Future
                # and collide on remote blob/task names derived from
                # run_id.
                raise ValueError(
                    f"run_id {run_id!r} is already active; "
                    "run_ids must be unique per CloudMRPRunner instance."
                )
            assigned = (
                job_name
                if job_name is not None
                else self._select_job_name_locked(run_id)
            )
            self._active_runs[run_id] = _ActiveCloudRun(
                job_name=assigned,
                output_dir=output_dir,
                input_payload=input_payload,
                overall_started=overall_started,
                future=future,
            )
            return assigned

    def _select_job_name_locked(self, run_id: str | None) -> str:
        """Pick a job name; caller MUST hold ``_run_state_lock``."""
        if run_id is None:
            generation = "0"
        else:
            generation = str(parse_sampler_run_id(run_id).generation_index)
        try:
            job_names = self.session.job_names[generation]
        except KeyError as exc:
            raise KeyError(
                f"No Azure Batch job configured for generation {generation}"
            ) from exc

        if len(job_names) == 1:
            return job_names[0]

        active_counts = {job_name: 0 for job_name in job_names}
        for active_run in self._active_runs.values():
            if (
                active_run.job_name in active_counts
                and not active_run.cancelled
            ):
                active_counts[active_run.job_name] += 1

        job_order = {
            job_name: index for index, job_name in enumerate(job_names)
        }
        return min(
            job_names,
            key=lambda job_name: (
                active_counts[job_name],
                job_order[job_name],
            ),
        )

    def _mark_run_submitting(self, run_id: str) -> bool:
        with self._run_state_lock:
            state = self._active_runs.get(run_id)
            if state is None:
                return False
            state.phase = "submitting"
            return True

    def _set_active_run_input_payload(
        self,
        run_id: str,
        input_payload: dict[str, Any],
    ) -> bool:
        with self._run_state_lock:
            state = self._active_runs.get(run_id)
            if state is None:
                return False
            state.input_payload = input_payload
            return True

    def _set_task_id(
        self,
        run_id: str,
        *,
        task_id: str,
        upload_elapsed_seconds: float,
        submitted_at: float,
    ) -> str:
        with self._run_state_lock:
            state = self._active_runs.get(run_id)
            if state is None:
                return "missing"
            state.task_id = task_id
            state.phase = "submitted"
            state.upload_elapsed_seconds = upload_elapsed_seconds
            state.submitted_at = submitted_at
            return "cancelled" if state.cancelled else "active"

    def _is_run_cancelled(self, run_id: str) -> bool:
        with self._run_state_lock:
            state = self._active_runs.get(run_id)
            return bool(state and state.cancelled)

    def _finish_run(self, run_id: str) -> _ActiveCloudRun | None:
        with self._run_state_lock:
            return self._active_runs.pop(run_id, None)

    def _upload_run_input(
        self,
        client: Any,
        input_filename: str,
        input_payload: dict[str, Any],
        remote_input_dir: str,
    ) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        try:
            local_input_path = tmpdir / input_filename
            local_input_path.write_text(dumps_json(input_payload) + "\n")
            with self._suppress_cloudops_info_output():
                upload_files_quietly(
                    client,
                    files=input_filename,
                    container_name=self.session.input_container,
                    local_root_dir=str(tmpdir),
                    location_in_blob=remote_input_dir,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _raise_controller_failure(self) -> None:
        if self._controller_failure is None:
            return
        if isinstance(self._controller_failure, BaseException):
            raise RuntimeError("Cloud runner controller failed.") from (
                self._controller_failure
            )
        raise RuntimeError("Cloud runner controller failed.")

    async def _execute_run(self, run_id: str) -> None:
        inflight_semaphore = self._get_inflight_semaphore_or_resolve(run_id)
        if inflight_semaphore is None:
            return

        remote_slot_acquired = False
        try:
            if self._is_run_cancelled(run_id):
                raise SimulationCancelledError(run_id)
            await inflight_semaphore.acquire()
            remote_slot_acquired = True

            submission = await self._submit_active_run(run_id)
            if submission is None:
                return

            task_status = await self._wait_for_submitted_task(
                run_id,
                submission,
            )
            completed_task = self._mark_task_completed(run_id, task_status)
            inflight_semaphore.release()
            remote_slot_acquired = False
            if completed_task is None:
                return
            download_elapsed = await self._download_successful_run_output(
                run_id,
                output_dir=completed_task.output_dir,
                task_status=task_status,
                cancelled=completed_task.cancelled,
            )
            if not self._store_run_download_elapsed(
                run_id,
                download_elapsed,
            ):
                return

            self._emit_task_timing_summary(run_id)

            if self._is_run_cancelled(run_id):
                self._resolve_run_cancelled(run_id)
                return

            if task_status.get("result") != "success":
                self._resolve_failed_task(run_id, submission, task_status)
                return

            task_context = self._build_task_context(
                run_id,
                job_name=submission["job_name"],
            )
            outputs = await self._run_in_io_executor(
                self._read_output_dir_for_context,
                completed_task.output_dir,
                task_context,
            )
            self._resolve_run_success(run_id, outputs)
        except asyncio.CancelledError:
            if self._is_run_cancelled(run_id):
                self._resolve_run_cancelled(run_id)
                return
            raise
        except SimulationCancelledError:
            self._resolve_run_cancelled(run_id)
        except Exception as exc:
            self._resolve_run_exception(run_id, exc)
        finally:
            if remote_slot_acquired:
                inflight_semaphore.release()

    def _get_inflight_semaphore_or_resolve(
        self,
        run_id: str,
    ) -> asyncio.Semaphore | None:
        inflight_semaphore = self._inflight_semaphore
        if inflight_semaphore is not None:
            return inflight_semaphore
        self._resolve_run_exception(
            run_id,
            RuntimeError("Cloud runner controller is unavailable."),
        )
        return None

    async def _submit_active_run(
        self,
        run_id: str,
    ) -> dict[str, Any] | None:
        if not self._mark_run_submitting(run_id):
            return None
        if self._is_run_cancelled(run_id):
            raise SimulationCancelledError(run_id)

        submission = await self._run_in_io_executor(
            self._submit_run_blocking,
            run_id,
        )
        submission_state = self._set_task_id(
            run_id,
            task_id=submission["task_id"],
            upload_elapsed_seconds=submission["upload_elapsed_seconds"],
            submitted_at=submission["submitted_at"],
        )
        if submission_state == "active":
            return submission

        self._cancel_submitted_task(
            submission["job_name"],
            submission["task_id"],
        )
        if submission_state == "cancelled":
            self._resolve_run_cancelled(run_id)
        return None

    def _cancel_submitted_task(self, job_name: str, task_id: str) -> None:
        self._cancel_batch_task(
            batch_client=self.client.batch_service_client,
            job_name=job_name,
            task_id=task_id,
        )

    async def _wait_for_submitted_task(
        self,
        run_id: str,
        submission: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return await self._wait_for_task_completion_async(
                client=self.client,
                job_name=submission["job_name"],
                task_id=submission["task_id"],
                run_id=run_id,
            )
        except BaseException:
            try:
                self._cancel_submitted_task(
                    submission["job_name"],
                    submission["task_id"],
                )
            except Exception:
                pass
            raise

    def _mark_task_completed(
        self,
        run_id: str,
        task_status: dict[str, Any],
    ) -> _CompletedRunTask | None:
        with self._run_state_lock:
            current = self._active_runs.get(run_id)
            if current is None:
                return None
            current.task_status = task_status
            current.completion_seen_at = time.monotonic()
            current.phase = "collecting"
            return _CompletedRunTask(
                output_dir=current.output_dir,
                cancelled=current.cancelled,
            )

    async def _download_successful_run_output(
        self,
        run_id: str,
        *,
        output_dir: Path,
        task_status: dict[str, Any],
        cancelled: bool,
    ) -> float | None:
        if cancelled or task_status.get("result") != "success":
            return None
        download_semaphore = getattr(self, "_download_semaphore", None)
        if download_semaphore is None:
            return await self._run_in_io_executor(
                self._download_output_blocking,
                run_id,
                output_dir,
            )
        await download_semaphore.acquire()
        try:
            return await self._run_in_io_executor(
                self._download_output_blocking,
                run_id,
                output_dir,
            )
        finally:
            download_semaphore.release()

    def _store_run_download_elapsed(
        self,
        run_id: str,
        download_elapsed: float | None,
    ) -> bool:
        with self._run_state_lock:
            current = self._active_runs.get(run_id)
            if current is None:
                return False
            current.download_elapsed_seconds = download_elapsed
            return True

    def _resolve_failed_task(
        self,
        run_id: str,
        submission: dict[str, Any],
        task_status: dict[str, Any],
    ) -> None:
        logs_folder = self.session.logs_folder_for_job(
            submission["job_name"],
            run_id,
        )
        failure_message = self._format_task_failure_message(
            run_id=run_id,
            job_name=submission["job_name"],
            task_id=submission["task_id"],
            task_status=task_status,
            logs_container=self.session.logs_container,
            logs_folder=logs_folder,
        )
        failure_message = append_task_log_excerpts(
            failure_message,
            task_log_excerpts=read_task_log_excerpts(
                self.client,
                container_name=self.session.logs_container,
                logs_folder=logs_folder,
            ),
        )
        self._resolve_run_exception(run_id, RuntimeError(failure_message))

    def _submit_run_blocking(self, run_id: str) -> dict[str, Any]:
        with self._run_state_lock:
            state = self._active_runs.get(run_id)
            if state is None:
                raise SimulationCancelledError(run_id)
            job_name = state.job_name
            input_payload = dict(state.input_payload)

        if self._is_run_cancelled(run_id):
            raise SimulationCancelledError(run_id)

        client = self.client
        mount_pairs = self.session.mount_pairs()
        remote_input_dir = self.session.remote_input_dir(run_id)
        task_output_location = self._resolve_task_output_location(
            run_id,
            job_name=job_name,
        )
        input_filename = f"{run_id}.json"

        upload_started = time.monotonic()
        self._upload_run_input(
            client,
            input_filename,
            input_payload,
            remote_input_dir,
        )
        upload_elapsed = time.monotonic() - upload_started

        if self._is_run_cancelled(run_id):
            raise SimulationCancelledError(run_id)

        remote_input_path = (
            f"{self.session.input_mount_path.rstrip('/')}/"
            f"{remote_input_dir}/{input_filename}"
        )
        task_command = self._build_task_command(
            self.session.task_mrp_config_path,
            remote_input_path,
            task_output_location.command_path,
        )
        task_id = self._add_batch_task_with_short_id(
            client=client,
            job_name=job_name,
            command_line=task_command,
            task_name_suffix=run_id,
            timeout=self.session.task_timeout_minutes,
            mount_pairs=mount_pairs,
            container_image_name=self.session.remote_image_ref,
            save_logs_path=self.session.logs_mount_path,
            logs_folder=self.session.logs_folder_for_job(job_name, run_id),
        )

        if self._is_run_cancelled(run_id):
            self._cancel_batch_task(
                batch_client=client.batch_service_client,
                job_name=job_name,
                task_id=task_id,
            )
            raise SimulationCancelledError(run_id)

        return {
            "job_name": job_name,
            "task_id": task_id,
            "upload_elapsed_seconds": upload_elapsed,
            "submitted_at": time.monotonic(),
        }

    def _download_output_blocking(
        self,
        run_id: str,
        output_dir: Path,
    ) -> float:
        with self._run_state_lock:
            state = self._active_runs.get(run_id)
            if state is None:
                raise SimulationCancelledError(run_id)
            job_name = state.job_name
        task_output_location = self._resolve_task_output_location(
            run_id,
            job_name=job_name,
        )
        final_path = output_dir / self._output_filename
        download_started = time.monotonic()
        download_blob_to_path_atomic(
            self.client,
            src_path=(
                f"{task_output_location.blob_prefix}/{self._output_filename}"
                if task_output_location.blob_prefix
                else self._output_filename
            ),
            dest_path=final_path,
            container_name=self.session.output_container,
            download_file_kwargs={"do_check": False, "check_size": False},
        )
        return time.monotonic() - download_started

    def _resolve_task_output_location(
        self,
        run_id: str,
        *,
        job_name: str,
    ) -> _TaskOutputLocation:
        context = self._build_task_context(run_id, job_name=job_name)
        output_mount_path = _normalize_posix_absolute_path(
            context.output_mount_path,
            label="cloud output_mount_path",
        )
        task_output_dir = _normalize_posix_absolute_path(
            context.task_output_dir,
            label="cloud.task_payload.task_output_dir",
        )
        blob_prefix = _task_output_blob_prefix(
            task_output_dir,
            output_mount_path=output_mount_path,
        )
        return _TaskOutputLocation(
            command_path=task_output_dir,
            blob_prefix=blob_prefix,
        )

    def _emit_task_timing_summary(self, run_id: str) -> None:
        if not self.session.print_task_durations:
            return

        with self._run_state_lock:
            state = self._active_runs.get(run_id)
            if (
                state is None
                or state.task_status is None
                or state.task_id is None
            ):
                return

            task_status = state.task_status
            task = task_status["task"]
            total_elapsed_seconds = time.monotonic() - state.overall_started
            wait_elapsed_seconds = None
            if (
                state.submitted_at is not None
                and state.completion_seen_at is not None
            ):
                wait_elapsed_seconds = (
                    state.completion_seen_at - state.submitted_at
                )
            summary = self._format_task_timing_summary(
                run_id=run_id,
                job_name=state.job_name,
                task_id=state.task_id,
                task=task,
                total_elapsed_seconds=total_elapsed_seconds,
                upload_elapsed_seconds=state.upload_elapsed_seconds,
                wait_elapsed_seconds=wait_elapsed_seconds,
                download_elapsed_seconds=state.download_elapsed_seconds,
            )

        print(summary, file=sys.stderr, flush=True)

    def _resolve_run_success(
        self,
        run_id: str,
        outputs: Any,
    ) -> None:
        state = self._finish_run(run_id)
        if state is None:
            return
        try:
            self._release_admission_slot(state, run_id=run_id)
        except CloudRunnerStateError as exc:
            self._shutdown_controller_if_idle(exclude_current_task=True)
            if not state.future.done():
                state.future.set_exception(exc)
            return
        if not state.future.done():
            state.future.set_result(outputs)
        if self._closed:
            self._shutdown_controller_if_idle(exclude_current_task=True)

    def _resolve_run_cancelled(self, run_id: str) -> None:
        state = self._finish_run(run_id)
        if state is None:
            return
        try:
            self._release_admission_slot(state, run_id=run_id)
        except CloudRunnerStateError as exc:
            self._shutdown_controller_if_idle(exclude_current_task=True)
            if not state.future.done():
                state.future.set_exception(exc)
            return
        self._shutdown_controller_if_idle(exclude_current_task=True)
        if not state.future.done():
            state.future.set_exception(SimulationCancelledError(run_id))

    def _resolve_run_exception(self, run_id: str, exc: BaseException) -> None:
        state = self._finish_run(run_id)
        if state is None:
            return
        try:
            self._release_admission_slot(state, run_id=run_id)
        except CloudRunnerStateError as state_exc:
            # Chain the state error as the cause of the original failure so
            # the sampler still sees the underlying problem but the
            # capacity-accounting bug is not lost.
            try:
                raise state_exc from exc
            except CloudRunnerStateError as combined:
                self._shutdown_controller_if_idle(exclude_current_task=True)
                if not state.future.done():
                    state.future.set_exception(combined)
                return
        self._shutdown_controller_if_idle(exclude_current_task=True)
        if not state.future.done():
            state.future.set_exception(exc)

    def _fail_controller(self, exc: BaseException) -> None:
        if self._controller_failure is None:
            self._controller_failure = exc
        with self._run_state_lock:
            active_run_ids = list(self._active_runs)
        for run_id in active_run_ids:
            self._resolve_run_exception(run_id, exc)

    def _shutdown_controller_if_idle(
        self,
        *,
        exclude_current_task: bool = False,
    ) -> None:
        current_task = None
        if exclude_current_task:
            try:
                current_task = asyncio.current_task()
            except RuntimeError:
                current_task = None
        loop = self._controller_loop
        with self._run_state_lock:
            if self._active_runs:
                return
            controller_tasks = [
                task
                for task in self._controller_tasks
                if task is not current_task
            ]

        for task in controller_tasks:
            if loop is not None:
                loop.call_soon_threadsafe(task.cancel)
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)

    def _release_admission_slot(
        self,
        state: _ActiveCloudRun,
        *,
        run_id: str | None = None,
    ) -> None:
        # Use the run-state lock to make release strictly idempotent: the
        # run's completion path, cancellation path, and exception path can
        # all race to call this, and we must never release the semaphore
        # twice for one acquisition.
        with self._run_state_lock:
            if not state.admission_acquired:
                return
            state.admission_acquired = False
            admission_semaphore = self._admission_semaphore
            loop = self._controller_loop
            closed = self._closed

        if admission_semaphore is None or loop is None or loop.is_closed():
            if closed:
                # True shutdown: the semaphore is being torn down with the
                # controller, so the slot has nowhere to be released to.
                return
            # The controller went away while runs are still in-flight.
            # That is a real capacity-accounting bug — surface it instead
            # of silently leaking the slot.
            raise CloudRunnerStateError(
                "cloud runner: cannot release admission slot — controller "
                "is unavailable but the runner is not closed",
                run_id=run_id,
            )

        try:
            loop.call_soon_threadsafe(admission_semaphore.release)
        except RuntimeError as exc:
            if closed:
                return
            raise CloudRunnerStateError(
                "cloud runner: admission slot release raced with controller "
                "shutdown while the runner is not closed",
                run_id=run_id,
            ) from exc

    def _track_controller_task(self, task: asyncio.Task[Any]) -> None:
        with self._run_state_lock:
            self._controller_tasks.append(task)

        def _discard(done_task: asyncio.Task[Any]) -> None:
            try:
                exc = done_task.exception()
            except asyncio.CancelledError:
                exc = None
            with self._run_state_lock:
                if done_task in self._controller_tasks:
                    self._controller_tasks.remove(done_task)
            if exc is not None:
                self._fail_controller(exc)

        task.add_done_callback(_discard)

    async def _run_in_io_executor(
        self, func: Any, *args: Any, **kwargs: Any
    ) -> Any:
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _wait_for_task_completion_async(
        self,
        *,
        client: Any,
        job_name: str,
        task_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        deadline = None
        if self.session.task_timeout_minutes is not None:
            deadline = time.monotonic() + (
                self.session.task_timeout_minutes * 60
            )

        cancel_requested = False
        while True:
            if self._is_run_cancelled(run_id) and not cancel_requested:
                await self._run_in_io_executor(
                    self._cancel_batch_task,
                    batch_client=client.batch_service_client,
                    job_name=job_name,
                    task_id=task_id,
                )
                cancel_requested = True

            try:
                task = await self._get_batch_task_with_retry(
                    client=client,
                    job_name=job_name,
                    task_id=task_id,
                    deadline=deadline,
                )
            except Exception as exc:
                if cancel_requested:
                    raise SimulationCancelledError(run_id) from exc
                raise

            state = self._enum_value(getattr(task, "state", None))
            if state == "completed":
                execution_info = getattr(task, "execution_info", None)
                result = self._enum_value(
                    getattr(execution_info, "result", None)
                )
                exit_code = getattr(execution_info, "exit_code", None)
                return {
                    "state": state,
                    "result": result,
                    "exit_code": exit_code,
                    "task": task,
                }
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for Azure Batch task {task_id} in job {job_name}."
                )
            await asyncio.sleep(self._poll_interval_seconds)

    # Azure Batch returns 4xx for well-defined terminal conditions like a
    # deleted task; everything else (5xx, 429, ECONNRESET, TLS resets) is
    # worth retrying a few times before blowing up the whole particle.
    _TASK_GET_MAX_ATTEMPTS = 5
    _TASK_GET_INITIAL_BACKOFF_SECONDS = 1.0
    _TASK_GET_MAX_BACKOFF_SECONDS = 30.0

    async def _get_batch_task_with_retry(
        self,
        *,
        client: Any,
        job_name: str,
        task_id: str,
        deadline: float | None,
    ) -> Any:
        backoff = self._TASK_GET_INITIAL_BACKOFF_SECONDS
        last_exc: Exception | None = None
        for attempt in range(1, self._TASK_GET_MAX_ATTEMPTS + 1):
            try:
                return await self._run_in_io_executor(
                    client.batch_service_client.task.get,
                    job_name,
                    task_id,
                )
            except Exception as exc:
                if not self._is_retryable_batch_error(exc):
                    raise
                last_exc = exc
                if attempt >= self._TASK_GET_MAX_ATTEMPTS:
                    break
                sleep_for = backoff
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    sleep_for = min(sleep_for, remaining)
                await asyncio.sleep(sleep_for)
                backoff = min(
                    backoff * 2.0, self._TASK_GET_MAX_BACKOFF_SECONDS
                )
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _is_retryable_batch_error(exc: BaseException) -> bool:
        """Return True for transient Azure Batch failures worth retrying.

        We treat 429 / 5xx / unknown (network) errors as retryable, and
        everything with a 4xx status other than 429 as terminal.
        """
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
        if status_code is None:
            # Likely a network / socket / TLS error before a response
            # was received. Retry.
            return True
        try:
            status_code = int(status_code)
        except (TypeError, ValueError):
            return True
        if status_code == 429:
            return True
        return 500 <= status_code < 600

    @staticmethod
    def _build_task_command(
        task_mrp_config_path: str,
        remote_input_path: str,
        remote_output_path: str,
    ) -> str:
        command = " ".join(
            shlex.quote(value)
            for value in [
                "mrp",
                "run",
                task_mrp_config_path,
                "--input",
                remote_input_path,
                "--output-dir",
                remote_output_path,
            ]
        )
        return f"/bin/bash -lc {shlex.quote(command)}"

    @staticmethod
    def _enum_value(value: Any) -> Any:
        if hasattr(value, "value"):
            return value.value
        return value
