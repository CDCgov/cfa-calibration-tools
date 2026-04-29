from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import sys
import tempfile
import time
from concurrent.futures import Future as ThreadFuture
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Sequence

from mrp import run as mrp_run

from calibrationtools.exceptions import (
    CloudRunnerStateError,
    SimulationCancelledError,
)
from calibrationtools.json_utils import dumps_json, to_jsonable

from .artifacts import download_blob_to_path_atomic, read_task_log_excerpts
from .backend import DEFAULT_CLOUD_RUNNER_BACKEND, CloudRunnerBackend
from .config import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    CloudRuntimeSettings,
)
from .formatting import append_task_log_excerpts
from .session import CloudSession
from .tooling import upload_files_quietly


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
    last_known_state: str | None = None
    task_status: dict[str, Any] | None = None
    controller_task: asyncio.Task[Any] | None = None
    submission_future: ThreadFuture[None] | None = None
    admission_acquired: bool = False


def resolve_cloud_build_context(
    *,
    default_repo_root: str | Path,
    default_dockerfile_relative_path: str | Path,
    repo_root: str | Path | None = None,
    dockerfile: str | Path | None = None,
    missing_dockerfile_message: str | None = None,
) -> tuple[Path, Path]:
    resolved_repo_root = (
        Path(repo_root) if repo_root is not None else Path(default_repo_root)
    )
    resolved_dockerfile = (
        Path(dockerfile)
        if dockerfile is not None
        else resolved_repo_root / Path(default_dockerfile_relative_path)
    )
    if not resolved_dockerfile.is_file():
        if missing_dockerfile_message is None:
            message = (
                "Cloud mode requires the model Dockerfile. "
                f"Looked at {resolved_dockerfile}."
            )
        else:
            message = missing_dockerfile_message.format(
                dockerfile=resolved_dockerfile
            )
        raise FileNotFoundError(message)
    return resolved_repo_root, resolved_dockerfile


def create_cloud_mrp_runner(
    config_path: str | Path,
    *,
    generation_count: int,
    max_concurrent_simulations: int,
    default_repo_root: str | Path,
    default_dockerfile_relative_path: str | Path,
    repo_root: str | Path | None = None,
    dockerfile: str | Path | None = None,
    missing_dockerfile_message: str | None = None,
    settings_loader: Callable[[str | Path], CloudRuntimeSettings],
    read_output_dir: Callable[[Path], Any],
    output_filename: str = "output.csv",
    print_task_durations: bool = False,
    backend: CloudRunnerBackend | None = None,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    mrp_run_func: Callable[..., Any] = mrp_run,
    auto_size_summary: Any | None = None,
) -> "CloudMRPRunner":
    resolved_repo_root, resolved_dockerfile = resolve_cloud_build_context(
        default_repo_root=default_repo_root,
        default_dockerfile_relative_path=default_dockerfile_relative_path,
        repo_root=repo_root,
        dockerfile=dockerfile,
        missing_dockerfile_message=missing_dockerfile_message,
    )
    return CloudMRPRunner(
        config_path,
        generation_count=generation_count,
        max_concurrent_simulations=max_concurrent_simulations,
        repo_root=resolved_repo_root,
        dockerfile=resolved_dockerfile,
        settings_loader=settings_loader,
        read_output_dir=read_output_dir,
        output_filename=output_filename,
        print_task_durations=print_task_durations,
        backend=backend,
        poll_interval_seconds=poll_interval_seconds,
        mrp_run_func=mrp_run_func,
        auto_size_summary=auto_size_summary,
    )


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
        settings_loader: Callable[[str | Path], CloudRuntimeSettings],
        read_output_dir: Callable[[Path], Any],
        output_filename: str = "output.csv",
        print_task_durations: bool = False,
        backend: CloudRunnerBackend | None = None,
        create_cloud_client_func: Callable[..., Any] = (
            DEFAULT_CLOUD_RUNNER_BACKEND.create_cloud_client
        ),
        git_short_sha_func: Callable[[Path], str] = (
            DEFAULT_CLOUD_RUNNER_BACKEND.git_short_sha
        ),
        make_session_slug_func: Callable[[str], str] = (
            DEFAULT_CLOUD_RUNNER_BACKEND.make_session_slug
        ),
        build_local_image_func: Callable[..., str] = (
            DEFAULT_CLOUD_RUNNER_BACKEND.build_local_image
        ),
        upload_local_image_func: Callable[..., str] = (
            DEFAULT_CLOUD_RUNNER_BACKEND.upload_local_image
        ),
        create_pool_with_blob_mounts_func: Callable[..., None] = (
            DEFAULT_CLOUD_RUNNER_BACKEND.create_pool_with_blob_mounts
        ),
        wait_for_pool_ready_func: Callable[..., Any] = (
            DEFAULT_CLOUD_RUNNER_BACKEND.wait_for_pool_ready
        ),
        add_batch_task_with_short_id_func: Callable[..., str] = (
            DEFAULT_CLOUD_RUNNER_BACKEND.add_batch_task_with_short_id
        ),
        cancel_batch_task_func: Callable[..., None] = (
            DEFAULT_CLOUD_RUNNER_BACKEND.cancel_batch_task
        ),
        format_task_failure_message_func: Callable[..., str] = (
            DEFAULT_CLOUD_RUNNER_BACKEND.format_task_failure_message
        ),
        format_task_timing_summary_func: Callable[..., str] = (
            DEFAULT_CLOUD_RUNNER_BACKEND.format_task_timing_summary
        ),
        make_resource_name_func: Callable[..., str] = (
            DEFAULT_CLOUD_RUNNER_BACKEND.make_resource_name
        ),
        parse_generation_from_run_id_func: Callable[[str], int] = (
            DEFAULT_CLOUD_RUNNER_BACKEND.parse_generation_from_run_id
        ),
        suppress_cloudops_info_output_func: Callable[[], Any] = (
            DEFAULT_CLOUD_RUNNER_BACKEND.suppress_cloudops_info_output
        ),
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        progress_refresh_interval_seconds: float | None = None,
        controller_start_timeout_seconds: float | None = 10.0,
        mrp_run_func: Callable[..., Any] = mrp_run,
        auto_size_summary: Any | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = Path(repo_root)
        self.dockerfile = Path(dockerfile)
        # Fail fast on a missing Dockerfile so that a wheel-installed caller
        # gets a clear error here instead of an obscure failure deep in
        # ``docker build`` after the cloud client has been initialized.
        if not self.dockerfile.is_file():
            raise FileNotFoundError(
                f"Dockerfile not found at {self.dockerfile}. "
                "Pass an explicit `dockerfile` (and matching `repo_root`) "
                "when constructing the cloud runner."
            )
        self.generation_count = generation_count
        # Validate concurrency before touching any cloud APIs so a
        # misconfigured CLI invocation cannot provision Azure resources
        # it will never be able to use.
        if max_concurrent_simulations < 1:
            raise ValueError(
                "max_concurrent_simulations must be at least 1 "
                f"(got {max_concurrent_simulations})"
            )
        self.max_concurrent_simulations = max_concurrent_simulations
        self._load_cloud_runtime_settings = settings_loader
        self._read_output_dir_callback = read_output_dir
        self._output_filename = output_filename
        self.print_task_durations = print_task_durations

        self._backend = backend or CloudRunnerBackend(
            create_cloud_client=create_cloud_client_func,
            git_short_sha=git_short_sha_func,
            make_session_slug=make_session_slug_func,
            build_local_image=build_local_image_func,
            upload_local_image=upload_local_image_func,
            create_pool_with_blob_mounts=create_pool_with_blob_mounts_func,
            wait_for_pool_ready=wait_for_pool_ready_func,
            add_batch_task_with_short_id=add_batch_task_with_short_id_func,
            cancel_batch_task=cancel_batch_task_func,
            format_task_failure_message=format_task_failure_message_func,
            format_task_timing_summary=format_task_timing_summary_func,
            make_resource_name=make_resource_name_func,
            parse_generation_from_run_id=parse_generation_from_run_id_func,
            suppress_cloudops_info_output=suppress_cloudops_info_output_func,
        )
        self._create_cloud_client = self._backend.create_cloud_client
        self._git_short_sha = self._backend.git_short_sha
        self._make_session_slug = self._backend.make_session_slug
        self._build_local_image = self._backend.build_local_image
        self._upload_local_image = self._backend.upload_local_image
        self._create_pool_with_blob_mounts = (
            self._backend.create_pool_with_blob_mounts
        )
        self._wait_for_pool_ready = self._backend.wait_for_pool_ready
        self._add_batch_task_with_short_id = (
            self._backend.add_batch_task_with_short_id
        )
        self._cancel_batch_task = self._backend.cancel_batch_task
        self._format_task_failure_message = (
            self._backend.format_task_failure_message
        )
        self._format_task_timing_summary = (
            self._backend.format_task_timing_summary
        )
        self._make_resource_name = self._backend.make_resource_name
        self._parse_generation_from_run_id = (
            self._backend.parse_generation_from_run_id
        )
        self._suppress_cloudops_info_output = (
            self._backend.suppress_cloudops_info_output
        )
        self._poll_interval_seconds = poll_interval_seconds
        # The progress cache is refreshed in a single bulk call per unique
        # job rather than one ``task.get`` per active run, so we can poll
        # less aggressively than the per-task wait loop without losing
        # responsiveness in ``describe_progress``. Default to 4x the per-
        # task poll interval; callers can override.
        self._progress_refresh_interval_seconds = (
            progress_refresh_interval_seconds
            if progress_refresh_interval_seconds is not None
            else max(poll_interval_seconds * 4.0, 1.0)
        )
        # ``None`` disables the bootstrap timeout and falls back to the
        # controller thread's own failure propagation; otherwise we wait
        # up to this many seconds for the controller event loop to come
        # up before assuming the thread is stuck.
        self._controller_start_timeout_seconds = (
            controller_start_timeout_seconds
        )
        self._mrp_run = mrp_run_func
        self.auto_size_summary = auto_size_summary

        self.settings = self._load_cloud_runtime_settings(self.config_path)
        if self.settings.jobs_per_session < 1:
            raise ValueError("jobs_per_session must be at least 1")
        if self.settings.task_slots_per_node < 1:
            raise ValueError("task_slots_per_node must be at least 1")
        if self.settings.pool_max_nodes < 1:
            raise ValueError("pool_max_nodes must be at least 1")
        if self.settings.pool_auto_scale_evaluation_interval_minutes < 5:
            raise ValueError(
                "pool_auto_scale_evaluation_interval_minutes must be at least 5"
            )
        if self.settings.dispatch_buffer < 0:
            raise ValueError("dispatch_buffer must be at least 0")

        self.client = self._create_cloud_client(
            keyvault=self.settings.keyvault
        )
        self._run_state_lock = Lock()
        self._active_runs: dict[str, _ActiveCloudRun] = {}
        self._last_pool_snapshot: Any | None = None
        self._controller_start_lock = Lock()
        self._controller_ready = Event()
        self._controller_thread: Thread | None = None
        self._controller_loop: asyncio.AbstractEventLoop | None = None
        self._controller_tasks: list[asyncio.Task[Any]] = []
        self._controller_failure: BaseException | None = None
        self._admission_semaphore: asyncio.Semaphore | None = None
        self._inflight_semaphore: asyncio.Semaphore | None = None
        self._closed = False
        self.session = self._initialize_cloud_session()

    def simulate(
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
        job_name = self._select_job_name(run_id)

        overrides: dict[str, Any] = {
            "runtime": {
                "command": sys.executable,
                "cloud": {
                    **self.session.to_runtime_cloud(),
                    "run_id": run_id,
                    "job_name": job_name,
                },
            }
        }

        if input_path is not None:
            overrides["input"] = str(Path(input_path).resolve())
        else:
            jsonable_params = to_jsonable(params)
            jsonable_params.setdefault("run_id", run_id)
            overrides["input"] = jsonable_params

        result = self._mrp_run(
            self.config_path,
            overrides,
            output_dir=str(output_dir),
        )
        if not result.ok:
            raise RuntimeError(f"run {run_id}: " + result.stderr.decode())
        if self.session.print_task_durations and result.stderr:
            sys.stderr.write(result.stderr.decode())
            sys.stderr.flush()

        return self._read_output_dir(Path(output_dir))

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
        input_payload = self._resolve_input_payload(
            params,
            input_path=input_path,
            run_id=run_id,
        )
        future: ThreadFuture[Any] = ThreadFuture()

        # _register_active_run picks a job name and inserts the run under
        # the same lock, so two concurrent registrations cannot both
        # observe the same "least busy" job. It also checks _closed so we
        # cannot slip a run past a concurrent close().
        self._register_active_run(
            run_id,
            output_dir=output_dir_path,
            input_payload=input_payload,
            overall_started=time.monotonic(),
            future=future,
        )

        try:
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

    def _dispatch_window_size(self) -> int:
        return self.max_concurrent_simulations + self.dispatch_buffer_size()

    def _read_output_dir(self, output_dir: Path) -> Any:
        return self._read_output_dir_callback(output_dir)

    def _resolve_input_payload(
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
                self._dispatch_window_size()
            )
            self._inflight_semaphore = asyncio.Semaphore(
                self.max_concurrent_simulations
            )
            with self._run_state_lock:
                self._controller_loop = loop
                self._controller_tasks = []
            # Schedule the cache-refresh loop here so it runs alongside
            # any in-flight runs and updates _last_pool_snapshot plus
            # per-run last_known_state at a low frequency, avoiding the
            # per-tick N-RPC fanout that describe_progress used to do.
            refresh_task = loop.create_task(
                self._progress_refresh_loop(),
                name="cloud-progress-refresh",
            )
            self._track_controller_task(refresh_task)
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
        session_slug: str,
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
                f"[cloud-run] rolled back partial session {session_slug}: "
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
        session_slug: str,
        pool_name: str,
        created_pool: bool,
        created_containers: Sequence[str],
        created_jobs: Sequence[str],
        tag: str,
    ) -> RuntimeError | None:
        """Rollback partial resources and preserve control-flow exceptions."""
        rollback_failures = self._rollback_partial_session(
            session_slug=session_slug,
            pool_name=pool_name if created_pool else None,
            container_names=created_containers,
            job_names=created_jobs,
        )
        detail = (
            f"session_slug={session_slug}, "
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
        session_slug = self._make_session_slug(tag)
        local_image_ref = self._build_local_image(
            repo_root=self.repo_root,
            dockerfile=self.dockerfile,
            local_image=self.settings.local_image,
            tag=tag,
        )
        remote_image_ref = self._upload_local_image(
            client=self.client,
            local_image_ref=local_image_ref,
            repository=self.settings.repository,
            tag=tag,
        )

        input_container = self._make_resource_name(
            self.settings.input_container_prefix,
            session_slug,
            max_length=63,
        )
        output_container = self._make_resource_name(
            self.settings.output_container_prefix,
            session_slug,
            max_length=63,
        )
        logs_container = self._make_resource_name(
            self.settings.logs_container_prefix,
            session_slug,
            max_length=63,
        )
        pool_name = self._make_resource_name(
            self.settings.pool_prefix,
            session_slug,
            max_length=64,
        )

        # Track resources that have actually been created so that a
        # failure partway through setup can roll them back and avoid
        # orphaning real Azure resources. We preserve creation order so
        # teardown can happen in reverse (jobs -> pool -> containers).
        created_containers: list[str] = []
        created_pool = False
        created_jobs: list[str] = []

        try:
            for container_name in (
                input_container,
                output_container,
                logs_container,
            ):
                self.client.create_blob_container(container_name)
                created_containers.append(container_name)

            mounts = [
                {
                    "source": input_container,
                    "target": self.settings.input_mount_path.lstrip("/"),
                },
                {
                    "source": output_container,
                    "target": self.settings.output_mount_path.lstrip("/"),
                },
                {
                    "source": logs_container,
                    "target": self.settings.logs_mount_path.lstrip("/"),
                },
            ]
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
            created_pool = True
            self._last_pool_snapshot = self._wait_for_pool_ready(
                batch_client=self.client.batch_service_client,
                pool_name=pool_name,
                timeout_minutes=self.settings.pool_ready_timeout_minutes,
            )

            shared_job_names: list[str] = []
            for job_index in range(1, self.settings.jobs_per_session + 1):
                job_name = self._make_resource_name(
                    self.settings.job_prefix,
                    f"{session_slug}-j{job_index}",
                    max_length=64,
                )
                self.client.create_job(
                    job_name=job_name,
                    pool_name=pool_name,
                    save_logs_to_blob=logs_container,
                    logs_folder=f"{session_slug}/{job_name}",
                    verify_pool=False,
                )
                created_jobs.append(job_name)
                shared_job_names.append(job_name)
        except BaseException as exc:
            wrapped_error = self._handle_partial_session_failure(
                exc=exc,
                session_slug=session_slug,
                pool_name=pool_name,
                created_pool=created_pool,
                created_containers=created_containers,
                created_jobs=created_jobs,
                tag=tag,
            )
            if wrapped_error is None:
                raise
            raise wrapped_error from exc

        job_names = {
            str(generation): list(shared_job_names)
            for generation in range(self.generation_count)
        }

        self._print_session_startup_summary(
            pool_name=pool_name,
            job_names=job_names,
            remote_image_ref=remote_image_ref,
        )

        return CloudSession(
            keyvault=self.settings.keyvault,
            session_slug=session_slug,
            image_tag=tag,
            remote_image_ref=remote_image_ref,
            pool_name=pool_name,
            job_names=job_names,
            input_container=input_container,
            output_container=output_container,
            logs_container=logs_container,
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

    def _select_job_name(self, run_id: str | None) -> str:
        with self._run_state_lock:
            return self._select_job_name_locked(run_id)

    def _select_job_name_locked(self, run_id: str | None) -> str:
        """Pick a job name; caller MUST hold ``_run_state_lock``."""
        if run_id is None:
            generation = "0"
        else:
            generation = str(self._parse_generation_from_run_id(run_id))
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

    def describe_progress(self, active_run_ids: Sequence[str]) -> str | None:
        if not active_run_ids:
            return None
        # Read from the cached snapshot to avoid an N-RPC fan-out per
        # progress tick (see _refresh_progress_cache for the lower-
        # frequency background refresh that keeps the cache fresh).
        return self._describe_progress_from_cache(active_run_ids)

    def _describe_progress_from_cache(
        self,
        active_run_ids: Sequence[str],
    ) -> str | None:
        with self._run_state_lock:
            run_states = {
                run_id: self._active_runs.get(run_id)
                for run_id in active_run_ids
            }
            pool = self._last_pool_snapshot

        pending_submit = 0
        active_count = 0
        running_count = 0
        completed_count = 0
        unknown_count = 0

        for state in run_states.values():
            if state is None:
                continue
            if state.task_id is None:
                pending_submit += 1
                continue

            task_state = state.last_known_state
            if task_state == "active":
                active_count += 1
            elif task_state == "running":
                running_count += 1
            elif task_state == "completed":
                completed_count += 1
            else:
                unknown_count += 1

        batch_status = (
            "batch("
            f"submitting={pending_submit}, "
            f"active={active_count}, "
            f"running={running_count}, "
            f"completed={completed_count}"
        )
        if unknown_count > 0:
            batch_status += f", unknown={unknown_count}"
        batch_status += ")"

        if pool is None:
            return batch_status
        return f"{batch_status}, {self._format_pool_status(pool)}"

    async def _progress_refresh_loop(self) -> None:
        """Periodically refresh the progress cache from Azure Batch.

        Runs in the controller event loop. Each iteration issues at most
        one ``task.list`` per unique active job plus one ``pool.get``,
        rather than one ``task.get`` per active run on every progress
        tick, and writes the results into ``_active_runs[*].last_known_state``
        and ``_last_pool_snapshot`` so describe_progress() can answer
        synchronously from the cache.
        """
        while True:
            try:
                await asyncio.sleep(self._progress_refresh_interval_seconds)
            except asyncio.CancelledError:
                return
            try:
                await self._run_in_io_executor(
                    self._refresh_progress_cache_blocking
                )
            except asyncio.CancelledError:
                return
            except Exception:
                # Never let a transient Batch/network failure tear down
                # the controller; the next tick will try again.
                continue

    def _refresh_progress_cache_blocking(self) -> None:
        with self._run_state_lock:
            jobs_to_task_ids: dict[str, set[str]] = {}
            for active in self._active_runs.values():
                if active.task_id is None:
                    continue
                jobs_to_task_ids.setdefault(active.job_name, set()).add(
                    active.task_id
                )

        # One task.list per unique job is O(jobs) instead of O(active_runs).
        task_state_by_job: dict[str, dict[str, str | None]] = {}
        for job_name, task_ids in jobs_to_task_ids.items():
            try:
                tasks_iter = self.client.batch_service_client.task.list(
                    job_name
                )
            except Exception:
                continue
            states: dict[str, str | None] = {}
            try:
                for task in tasks_iter:
                    task_id = getattr(task, "id", None)
                    if not isinstance(task_id, str):
                        continue
                    if task_id not in task_ids:
                        continue
                    states[task_id] = self._enum_value(
                        getattr(task, "state", None)
                    )
            except Exception:
                pass
            task_state_by_job[job_name] = states

        try:
            pool = self.client.batch_service_client.pool.get(
                self.session.pool_name
            )
        except Exception:
            pool = None

        with self._run_state_lock:
            for active in self._active_runs.values():
                if active.task_id is None:
                    continue
                states: dict[str, str | None] | None = task_state_by_job.get(
                    active.job_name
                )
                if states is None:
                    continue
                if not states:
                    continue
                new_state = states.get(active.task_id)
                if new_state is not None:
                    active.last_known_state = new_state
            if pool is not None:
                self._last_pool_snapshot = pool

    def _describe_progress_from_batch(
        self,
        active_run_ids: Sequence[str],
    ) -> str | None:
        with self._run_state_lock:
            run_states = {
                run_id: self._active_runs.get(run_id)
                for run_id in active_run_ids
            }

        pending_submit = 0
        active_count = 0
        running_count = 0
        completed_count = 0
        unknown_count = 0

        for state in run_states.values():
            if state is None or state.task_id is None:
                pending_submit += 1
                continue

            try:
                task = self.client.batch_service_client.task.get(
                    state.job_name,
                    state.task_id,
                )
            except Exception:
                unknown_count += 1
                continue

            task_state = self._enum_value(getattr(task, "state", None))
            if task_state == "active":
                active_count += 1
            elif task_state == "running":
                running_count += 1
            elif task_state == "completed":
                completed_count += 1
            else:
                unknown_count += 1

        batch_status = (
            "batch("
            f"submitting={pending_submit}, "
            f"active={active_count}, "
            f"running={running_count}, "
            f"completed={completed_count}"
        )
        if unknown_count > 0:
            batch_status += f", unknown={unknown_count}"
        batch_status += ")"

        try:
            pool = self.client.batch_service_client.pool.get(
                self.session.pool_name
            )
        except Exception:
            return f"{batch_status}, pool=unavailable"

        # Keep the cached snapshot fresh so cache-based readers and any
        # future describe_progress_from_cache() callers see current node
        # and allocation counts rather than a startup-time view.
        with self._run_state_lock:
            self._last_pool_snapshot = pool

        return f"{batch_status}, {self._format_pool_status(pool)}"

    def _format_pool_status(self, pool: Any) -> str:
        pool_state = self._enum_value(getattr(pool, "state", None))
        allocation_state = self._enum_value(
            getattr(pool, "allocation_state", None)
        )
        current_dedicated = getattr(pool, "current_dedicated_nodes", None)
        target_dedicated = getattr(pool, "target_dedicated_nodes", None)
        current_low_priority = getattr(
            pool, "current_low_priority_nodes", None
        )
        target_low_priority = getattr(pool, "target_low_priority_nodes", None)
        task_slots = getattr(pool, "task_slots_per_node", None)

        pool_status = (
            "pool("
            f"state={pool_state}, "
            f"allocation={allocation_state}, "
            f"dedicated={current_dedicated}/{target_dedicated}, "
            f"low_priority={current_low_priority}/{target_low_priority}"
        )
        if task_slots is not None:
            pool_status += f", task_slots={task_slots}"
        pool_status += ")"
        return pool_status

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
        inflight_semaphore = self._inflight_semaphore
        if inflight_semaphore is None:
            self._resolve_run_exception(
                run_id,
                RuntimeError("Cloud runner controller is unavailable."),
            )
            return

        inflight_acquired = False
        try:
            await inflight_semaphore.acquire()
            inflight_acquired = True
            if not self._mark_run_submitting(run_id):
                return
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
            if submission_state != "active":
                self._cancel_batch_task(
                    batch_client=self.client.batch_service_client,
                    job_name=submission["job_name"],
                    task_id=submission["task_id"],
                )
                if submission_state == "cancelled":
                    self._resolve_run_cancelled(run_id)
                return

            client = self.client
            try:
                task_status = await self._wait_for_task_completion_async(
                    client=client,
                    job_name=submission["job_name"],
                    task_id=submission["task_id"],
                    run_id=run_id,
                )
            except BaseException:
                # Best-effort cancel the remote Batch task so a local
                # wait-path timeout or polling failure does not leave
                # the submitted task running in Azure.
                try:
                    self._cancel_batch_task(
                        batch_client=self.client.batch_service_client,
                        job_name=submission["job_name"],
                        task_id=submission["task_id"],
                    )
                except Exception:
                    pass
                raise

            with self._run_state_lock:
                current = self._active_runs.get(run_id)
                if current is None:
                    return
                current.last_known_state = "completed"
                current.task_status = task_status
                current.completion_seen_at = time.monotonic()
                current.phase = "collecting"
                output_dir = current.output_dir
                cancelled = current.cancelled

            download_elapsed = None
            if not cancelled and task_status.get("result") == "success":
                download_elapsed = await self._run_in_io_executor(
                    self._download_output_blocking,
                    run_id,
                    output_dir,
                )

            with self._run_state_lock:
                current = self._active_runs.get(run_id)
                if current is None:
                    return
                current.download_elapsed_seconds = download_elapsed

            self._emit_task_timing_summary(run_id)

            if self._is_run_cancelled(run_id):
                self._resolve_run_cancelled(run_id)
                return

            if task_status.get("result") != "success":
                failure_message = self._format_task_failure_message(
                    run_id=run_id,
                    job_name=submission["job_name"],
                    task_id=submission["task_id"],
                    task_status=task_status,
                    logs_container=self.session.logs_container,
                    logs_folder=self.session.logs_folder_for_job(
                        submission["job_name"],
                        run_id,
                    ),
                )
                failure_message = append_task_log_excerpts(
                    failure_message,
                    task_log_excerpts=read_task_log_excerpts(
                        client,
                        container_name=self.session.logs_container,
                        logs_folder=self.session.logs_folder_for_job(
                            submission["job_name"],
                            run_id,
                        ),
                    ),
                )
                self._resolve_run_exception(
                    run_id,
                    RuntimeError(failure_message),
                )
                return

            outputs = await self._run_in_io_executor(
                self._read_output_dir,
                output_dir,
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
            if inflight_acquired:
                inflight_semaphore.release()

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
        remote_output_dir = self.session.remote_output_dir(run_id)
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
        remote_output_path = (
            f"{self.session.output_mount_path.rstrip('/')}/{remote_output_dir}"
        )
        task_command = self._build_task_command(
            self.session.task_mrp_config_path,
            remote_input_path,
            remote_output_path,
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
        remote_output_dir = self.session.remote_output_dir(run_id)
        final_path = output_dir / self._output_filename
        download_started = time.monotonic()
        download_blob_to_path_atomic(
            self.client,
            src_path=f"{remote_output_dir}/{self._output_filename}",
            dest_path=final_path,
            container_name=self.session.output_container,
            download_file_kwargs={"do_check": False, "check_size": False},
        )
        return time.monotonic() - download_started

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
        self._shutdown_controller_if_idle(exclude_current_task=True)
        if not state.future.done():
            state.future.set_result(outputs)

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
