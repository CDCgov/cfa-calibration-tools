from __future__ import annotations

import asyncio
from concurrent.futures import Future as ThreadFuture
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any

import pytest

import calibrationtools.cloud.runner as runner_module
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
    runner._read_output_dir = read_output_dir
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


def test_execute_run_submits_before_wait_slot_is_available(tmp_path):
    async def exercise():
        runner, future, calls = make_execute_runner(tmp_path)
        runner._inflight_semaphore = asyncio.Semaphore(0)

        task = asyncio.create_task(runner._execute_run(RUN_ID))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert ("submit", RUN_ID) in calls
        assert not any(name == "wait" for name, _ in calls)
        assert not task.done()

        runner._inflight_semaphore.release()
        await task

        assert future.result() == {"ok": True}
        assert any(name == "wait" for name, _ in calls)

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
