from __future__ import annotations

import asyncio
from concurrent.futures import Future as ThreadFuture
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any

import pytest

import calibrationtools.cloud.runner as runner_module
from calibrationtools.cloud.config import (
    CloudPayloadTransformOp,
    CloudTaskPayloadSettings,
    CloudTaskPayloadTransform,
)
from calibrationtools.cloud.hooks import CloudRunnerHooks
from calibrationtools.cloud.runner import CloudMRPRunner, _ActiveCloudRun
from calibrationtools.exceptions import SimulationCancelledError

RUN_ID = "gen_0_particle_0_attempt_0"


class FakeSession:
    output_container = "output-container"
    logs_container = "logs-container"
    print_task_durations = False
    task_timeout_minutes = 9

    def logs_folder_for_job(self, job_name: str, run_id: str) -> str:
        return f"logs/{job_name}/{run_id}"


def make_execute_runner(
    tmp_path: Path,
    *,
    cancelled: bool = False,
    cancel_after_submit: bool = False,
    task_status: dict[str, Any] | None = None,
    wait_error: BaseException | None = None,
):
    calls: list[tuple[str, Any]] = []
    future: ThreadFuture[Any] = ThreadFuture()
    runner = object.__new__(CloudMRPRunner)
    runner._inflight_semaphore = asyncio.Semaphore(1)
    runner._run_state_lock = Lock()
    runner._controller_loop = None
    runner._controller_tasks = []
    runner._closed = False
    runner._admission_semaphore = None
    runner.client = SimpleNamespace(batch_service_client=SimpleNamespace())
    runner.session = FakeSession()
    runner._active_runs = {
        RUN_ID: _ActiveCloudRun(
            job_name="job-a",
            output_dir=tmp_path / "output",
            input_payload={"run_id": RUN_ID},
            overall_started=1.0,
            future=future,
            cancelled=cancelled,
        )
    }
    runner._format_task_failure_message = lambda **kwargs: "base failure"
    runner._format_task_timing_summary = lambda **kwargs: "timing"

    async def run_in_io_executor(func, *args, **kwargs):
        return func(*args, **kwargs)

    def submit_run_blocking(run_id):
        calls.append(("submit", run_id))
        if cancel_after_submit:
            runner._active_runs[run_id].cancelled = True
        return {
            "job_name": "job-a",
            "task_id": "task-1",
            "upload_elapsed_seconds": 0.1,
            "submitted_at": 2.0,
        }

    async def wait_for_task_completion_async(**kwargs):
        calls.append(("wait", kwargs))
        if wait_error is not None:
            raise wait_error
        return task_status or {"result": "success", "task": object()}

    def cancel_batch_task(**kwargs):
        calls.append(("cancel", kwargs))

    def download_output_blocking(run_id, output_dir):
        calls.append(("download", (run_id, output_dir)))
        return 0.2

    def read_output_dir(output_dir):
        calls.append(("read_output", output_dir))
        return {"ok": True}

    runner._run_in_io_executor = run_in_io_executor
    runner._submit_run_blocking = submit_run_blocking
    runner._wait_for_task_completion_async = wait_for_task_completion_async
    runner._cancel_batch_task = cancel_batch_task
    runner._download_output_blocking = download_output_blocking
    runner._read_output_dir_callback = read_output_dir
    return runner, future, calls


def assert_inflight_released(runner: CloudMRPRunner) -> None:
    inflight_semaphore = runner._inflight_semaphore
    assert inflight_semaphore is not None
    assert inflight_semaphore._value == 1


def test_execute_run_success_submits_downloads_reads_and_resolves(tmp_path):
    runner, future, calls = make_execute_runner(tmp_path)

    asyncio.run(runner._execute_run(RUN_ID))

    assert future.result() == {"ok": True}
    assert ("submit", RUN_ID) in calls
    assert any(name == "wait" for name, _ in calls)
    assert any(name == "download" for name, _ in calls)
    assert any(name == "read_output" for name, _ in calls)
    assert RUN_ID not in runner._active_runs
    assert_inflight_released(runner)


def test_resolve_run_success_keeps_controller_alive_until_close(tmp_path):
    runner, future, _ = make_execute_runner(tmp_path)
    shutdown_calls: list[dict[str, Any]] = []

    def shutdown_controller_if_idle(**kwargs):
        shutdown_calls.append(dict(kwargs))

    runner._shutdown_controller_if_idle = shutdown_controller_if_idle

    runner._resolve_run_success(RUN_ID, {"ok": True})

    assert future.result() == {"ok": True}
    assert shutdown_calls == []


def test_resolve_run_success_stops_controller_after_close(tmp_path):
    runner, future, _ = make_execute_runner(tmp_path)
    shutdown_calls: list[dict[str, Any]] = []
    runner._closed = True

    def shutdown_controller_if_idle(**kwargs):
        shutdown_calls.append(dict(kwargs))

    runner._shutdown_controller_if_idle = shutdown_controller_if_idle

    runner._resolve_run_success(RUN_ID, {"ok": True})

    assert future.result() == {"ok": True}
    assert shutdown_calls == [{"exclude_current_task": True}]


def test_execute_run_waits_for_remote_slot_before_submission(tmp_path):
    async def exercise():
        runner, future, calls = make_execute_runner(tmp_path)
        runner._inflight_semaphore = asyncio.Semaphore(0)

        task = asyncio.create_task(runner._execute_run(RUN_ID))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert not any(name == "submit" for name, _ in calls)
        assert not any(name == "wait" for name, _ in calls)
        assert not task.done()

        runner._inflight_semaphore.release()
        await task

        assert future.result() == {"ok": True}
        assert ("submit", RUN_ID) in calls
        assert any(name == "wait" for name, _ in calls)

    asyncio.run(exercise())


def test_execute_run_remote_slot_serializes_submission_and_wait(tmp_path):
    async def exercise():
        runner, future_1, calls = make_execute_runner(tmp_path)
        runner._inflight_semaphore = asyncio.Semaphore(1)
        run_id_2 = "gen_0_particle_1_attempt_0"
        future_2: ThreadFuture[Any] = ThreadFuture()
        runner._active_runs[run_id_2] = _ActiveCloudRun(
            job_name="job-a",
            output_dir=tmp_path / "output-2",
            input_payload={"run_id": run_id_2},
            overall_started=1.0,
            future=future_2,
        )
        first_wait_started = asyncio.Event()
        release_first_wait = asyncio.Event()

        async def wait_for_task_completion_async(**kwargs):
            calls.append(("wait", kwargs))
            if kwargs["run_id"] == RUN_ID:
                first_wait_started.set()
                await release_first_wait.wait()
            return {"result": "success", "task": object()}

        runner._wait_for_task_completion_async = wait_for_task_completion_async

        task_1 = asyncio.create_task(runner._execute_run(RUN_ID))
        await first_wait_started.wait()
        task_2 = asyncio.create_task(runner._execute_run(run_id_2))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert [details for name, details in calls if name == "submit"] == [
            RUN_ID
        ]

        release_first_wait.set()
        await asyncio.gather(task_1, task_2)

        assert future_1.result() == {"ok": True}
        assert future_2.result() == {"ok": True}
        assert ("submit", run_id_2) in calls
        assert_inflight_released(runner)

    asyncio.run(exercise())


def test_execute_run_cancelled_before_submission_resolves_cancelled(tmp_path):
    runner, future, calls = make_execute_runner(tmp_path, cancelled=True)

    asyncio.run(runner._execute_run(RUN_ID))

    with pytest.raises(SimulationCancelledError):
        future.result()
    assert not any(name == "submit" for name, _ in calls)
    assert_inflight_released(runner)


def test_execute_run_cancellation_after_submission_cancels_task(tmp_path):
    runner, future, calls = make_execute_runner(
        tmp_path,
        cancel_after_submit=True,
    )

    asyncio.run(runner._execute_run(RUN_ID))

    with pytest.raises(SimulationCancelledError):
        future.result()
    cancel = next(details for name, details in calls if name == "cancel")
    assert cancel["job_name"] == "job-a"
    assert cancel["task_id"] == "task-1"
    assert not any(name == "wait" for name, _ in calls)
    assert_inflight_released(runner)


def test_execute_run_wait_failure_cancels_and_resolves_exception(tmp_path):
    runner, future, calls = make_execute_runner(
        tmp_path,
        wait_error=RuntimeError("wait failed"),
    )

    asyncio.run(runner._execute_run(RUN_ID))

    with pytest.raises(RuntimeError, match="wait failed"):
        future.result()
    assert any(name == "cancel" for name, _ in calls)
    assert_inflight_released(runner)


def test_execute_run_failed_task_includes_log_excerpts(monkeypatch, tmp_path):
    runner, future, calls = make_execute_runner(
        tmp_path,
        task_status={"result": "failure", "task": object(), "exit_code": 1},
    )
    monkeypatch.setattr(
        runner_module,
        "read_task_log_excerpts",
        lambda *args, **kwargs: {"stderr": "stderr tail"},
    )

    asyncio.run(runner._execute_run(RUN_ID))

    with pytest.raises(RuntimeError) as exc_info:
        future.result()
    assert "base failure" in str(exc_info.value)
    assert "stderr_excerpt='stderr tail'" in str(exc_info.value)
    assert not any(name == "download" for name, _ in calls)
    assert_inflight_released(runner)


def test_download_successful_run_output_obeys_configured_limit(tmp_path):
    async def exercise():
        runner = object.__new__(CloudMRPRunner)
        runner._download_semaphore = asyncio.Semaphore(1)
        active = 0
        max_active = 0

        async def run_in_io_executor(func, *args, **kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return 0.1

        runner._run_in_io_executor = run_in_io_executor
        runner._download_output_blocking = lambda *args: 0.1

        await asyncio.gather(
            runner._download_successful_run_output(
                "run-1",
                output_dir=tmp_path / "one",
                task_status={"result": "success"},
                cancelled=False,
            ),
            runner._download_successful_run_output(
                "run-2",
                output_dir=tmp_path / "two",
                task_status={"result": "success"},
                cancelled=False,
            ),
        )

        assert max_active == 1

    asyncio.run(exercise())


def test_download_successful_run_output_skips_failed_without_slot(tmp_path):
    async def exercise():
        runner = object.__new__(CloudMRPRunner)
        runner._download_semaphore = asyncio.Semaphore(1)
        result = await runner._download_successful_run_output(
            "run-1",
            output_dir=tmp_path,
            task_status={"result": "failure"},
            cancelled=False,
        )
        assert result is None
        assert runner._download_semaphore._value == 1

    asyncio.run(exercise())


def test_task_payload_hook_receives_transformed_payload_and_context(tmp_path):
    events: list[tuple[str, Any]] = []

    class Hooks(CloudRunnerHooks):
        def prepare_task_payload(self, payload, context):
            events.append(("payload", dict(payload)))
            events.append(("run_id", context.run_id))
            payload["hooked"] = True
            return payload

    runner = object.__new__(CloudMRPRunner)
    runner._task_payload_settings = CloudTaskPayloadSettings(
        transforms=(
            CloudTaskPayloadTransform(
                name="output",
                op=CloudPayloadTransformOp.SET,
                path="/epimodel.GlobalParams/output_dir",
                value="{task_output_dir}",
                on_missing="create",
            ),
        )
    )
    runner._hooks = Hooks()
    runner._session_shared_assets = ()
    runner.session = SimpleNamespace(
        session_id="session",
        input_mount_path="/cloud-input",
        output_mount_path="/cloud-output",
        logs_mount_path="/cloud-logs",
        remote_output_dir=lambda run_id: f"output/session/{run_id}",
    )

    input_payload = runner._load_input_payload(
        {},
        input_path=None,
        run_id="run-1",
    )
    payload = runner._prepare_input_payload(
        input_payload,
        run_id="run-1",
        job_name="",
    )

    assert payload["epimodel.GlobalParams"]["output_dir"] == (
        "/cloud-output/output/session/run-1"
    )
    assert payload["hooked"] is True
    assert ("run_id", "run-1") in events


def test_custom_task_output_dir_drives_payload_command_and_download(
    monkeypatch,
    tmp_path,
):
    calls: dict[str, Any] = {}
    runner = object.__new__(CloudMRPRunner)
    runner._run_state_lock = Lock()
    runner._task_payload_settings = CloudTaskPayloadSettings(
        task_output_dir="{output_mount_path}/custom-output/{run_id}",
        transforms=(
            CloudTaskPayloadTransform(
                name="output",
                op=CloudPayloadTransformOp.SET,
                path="/model/output_dir",
                value="{task_output_dir}",
                on_missing="create",
            ),
        ),
    )
    runner._hooks = CloudRunnerHooks()
    runner._session_shared_assets = ()
    runner._output_filename = "result.csv"
    runner.client = SimpleNamespace(batch_service_client=SimpleNamespace())
    runner.session = SimpleNamespace(
        session_id="session",
        input_container="input-container",
        output_container="output-container",
        input_mount_path="/cloud-input",
        output_mount_path="/cloud-output",
        logs_mount_path="/cloud-logs",
        task_mrp_config_path="/app/task.toml",
        task_timeout_minutes=9,
        remote_image_ref="image:tag",
        mount_pairs=lambda: [{"source": "output", "target": "/cloud-output"}],
        remote_input_dir=lambda run_id: f"input/session/{run_id}",
        remote_output_dir=lambda run_id: f"output/session/{run_id}",
        logs_folder_for_job=lambda job_name, run_id: (
            f"logs/{job_name}/{run_id}"
        ),
    )
    input_payload = runner._load_input_payload(
        {},
        input_path=None,
        run_id=RUN_ID,
    )
    payload = runner._prepare_input_payload(
        input_payload,
        run_id=RUN_ID,
        job_name="job-a",
    )
    assert payload["model"]["output_dir"] == (
        f"/cloud-output/custom-output/{RUN_ID}"
    )
    future: ThreadFuture[Any] = ThreadFuture()
    runner._active_runs = {
        RUN_ID: _ActiveCloudRun(
            job_name="job-a",
            output_dir=tmp_path / "output",
            input_payload=payload,
            overall_started=1.0,
            future=future,
        )
    }
    runner._upload_run_input = lambda *args, **kwargs: None

    def add_batch_task_with_short_id(**kwargs):
        calls["command_line"] = kwargs["command_line"]
        return "task-1"

    def download_blob_to_path_atomic(*args, **kwargs):
        calls["download_src_path"] = kwargs["src_path"]

    runner._add_batch_task_with_short_id = add_batch_task_with_short_id
    monkeypatch.setattr(
        runner_module,
        "download_blob_to_path_atomic",
        download_blob_to_path_atomic,
    )

    runner._submit_run_blocking(RUN_ID)
    runner._download_output_blocking(RUN_ID, tmp_path)

    assert (
        f"--output-dir /cloud-output/custom-output/{RUN_ID}"
        in calls["command_line"]
    )
    assert calls["download_src_path"] == f"custom-output/{RUN_ID}/result.csv"


def test_custom_task_output_dir_must_stay_under_output_mount(tmp_path):
    runner = object.__new__(CloudMRPRunner)
    runner._run_state_lock = Lock()
    runner._task_payload_settings = CloudTaskPayloadSettings(
        task_output_dir="/other-output/{run_id}",
    )
    runner._session_shared_assets = ()
    runner._output_filename = "result.csv"
    runner.client = SimpleNamespace(batch_service_client=SimpleNamespace())
    runner.session = SimpleNamespace(
        session_id="session",
        input_container="input-container",
        output_container="output-container",
        input_mount_path="/cloud-input",
        output_mount_path="/cloud-output",
        logs_mount_path="/cloud-logs",
        task_mrp_config_path="/app/task.toml",
        task_timeout_minutes=9,
        remote_image_ref="image:tag",
        mount_pairs=lambda: [{"source": "output", "target": "/cloud-output"}],
        remote_input_dir=lambda run_id: f"input/session/{run_id}",
        remote_output_dir=lambda run_id: f"output/session/{run_id}",
        logs_folder_for_job=lambda job_name, run_id: (
            f"logs/{job_name}/{run_id}"
        ),
    )
    future: ThreadFuture[Any] = ThreadFuture()
    runner._active_runs = {
        RUN_ID: _ActiveCloudRun(
            job_name="job-a",
            output_dir=tmp_path / "output",
            input_payload={"run_id": RUN_ID},
            overall_started=1.0,
            future=future,
        )
    }
    runner._upload_run_input = lambda *args, **kwargs: pytest.fail(
        "input upload should not start for an invalid task_output_dir"
    )
    runner._add_batch_task_with_short_id = lambda **kwargs: pytest.fail(
        "Batch task should not be submitted"
    )

    with pytest.raises(ValueError, match="under output_mount_path"):
        runner._submit_run_blocking(RUN_ID)


def test_simulate_async_renders_job_name_after_registration(tmp_path):
    async def exercise():
        runner = object.__new__(CloudMRPRunner)
        runner._run_state_lock = Lock()
        runner._closed = False
        runner._active_runs = {}
        runner._controller_tasks = []
        runner._controller_loop = None
        runner._admission_semaphore = None
        runner._task_payload_settings = CloudTaskPayloadSettings(
            transforms=(
                CloudTaskPayloadTransform(
                    name="job",
                    op=CloudPayloadTransformOp.SET,
                    path="/selected_job",
                    value="{job_name}",
                    on_missing="create",
                ),
            ),
        )
        runner._hooks = CloudRunnerHooks()
        runner._session_shared_assets = ()
        runner._ensure_controller_started = lambda: None
        runner._raise_controller_failure = lambda: None
        runner.session = SimpleNamespace(
            session_id="session",
            input_mount_path="/cloud-input",
            output_mount_path="/cloud-output",
            logs_mount_path="/cloud-logs",
            job_names={"0": ["job-a", "job-b"]},
            remote_output_dir=lambda run_id: f"output/session/{run_id}",
        )
        captured: dict[str, Any] = {}

        async def submit_run_async(run_id):
            state = runner._active_runs[run_id]
            captured["payload"] = dict(state.input_payload)
            runner._resolve_run_success(run_id, {"ok": True})

        runner._submit_run_async = submit_run_async

        result = await runner.simulate_async(
            {},
            output_dir=tmp_path,
            run_id=RUN_ID,
        )

        assert result == {"ok": True}
        assert captured["payload"]["selected_job"] == "job-a"

    asyncio.run(exercise())


def test_output_hook_can_override_output_parsing(tmp_path):
    class Hooks(CloudRunnerHooks):
        def read_output_dir(self, output_dir, context):
            return {"run_id": context.run_id, "output_dir": str(output_dir)}

    runner = object.__new__(CloudMRPRunner)
    runner._hooks = Hooks()
    context = SimpleNamespace(run_id="run-1")

    assert runner._read_output_dir_for_context(tmp_path, context) == {
        "run_id": "run-1",
        "output_dir": str(tmp_path),
    }
