from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from calibrationtools.cloud.backend import CloudExecutorBackend
from calibrationtools.cloud.executor import execute_cloud_run


RUN_ID = "gen_0_particle_0_attempt_0"


def make_run_json(
    tmp_path: Path,
    *,
    run_id: str = RUN_ID,
    input_value: Any | None = None,
    include_run_id: bool = True,
    include_job_name: bool = False,
    print_task_durations: bool = False,
) -> dict[str, Any]:
    cloud = {
        "keyvault": "kv",
        "session_id": "session",
        "image_tag": "tag",
        "remote_image_ref": "acr.azurecr.io/model:tag",
        "pool_name": "pool",
        "job_names": {"0": ["job-a", "job-b"]},
        "input_container": "input-container",
        "output_container": "output-container",
        "logs_container": "logs-container",
        "task_mrp_config_path": "/app/task.toml",
        "input_mount_path": "/mnt/input",
        "output_mount_path": "/mnt/output",
        "logs_mount_path": "/mnt/logs",
        "task_timeout_minutes": 7,
        "print_task_durations": print_task_durations,
    }
    if include_run_id:
        cloud["run_id"] = run_id
    if include_job_name:
        cloud["job_name"] = "explicit-job"
    if input_value is None:
        input_value = {"run_id": run_id, "value": 3}
    return {
        "runtime": {"cloud": cloud},
        "input": input_value,
        "output": {"spec": "filesystem", "dir": str(tmp_path / "output")},
    }


def make_backend(
    calls: list[tuple[str, dict[str, Any]]],
    *,
    wait_result: dict[str, Any] | None = None,
    wait_error: BaseException | None = None,
) -> CloudExecutorBackend:
    client = SimpleNamespace(batch_service_client=SimpleNamespace())

    def create_cloud_client(*, keyvault):
        calls.append(("create_client", {"keyvault": keyvault}))
        return client

    def add_batch_task_with_short_id(**kwargs):
        calls.append(("add_task", kwargs))
        return "task-1"

    def wait_for_task_completion(**kwargs):
        calls.append(("wait", kwargs))
        if wait_error is not None:
            raise wait_error
        return wait_result or {"result": "success", "task": object()}

    def cancel_batch_task(**kwargs):
        calls.append(("cancel", kwargs))

    def format_task_failure_message(**kwargs):
        calls.append(("format_failure", kwargs))
        return "formatted failure"

    def format_task_timing_summary(**kwargs):
        calls.append(("format_timing", kwargs))
        return "timing summary"

    return CloudExecutorBackend(
        create_cloud_client=create_cloud_client,
        add_batch_task_with_short_id=add_batch_task_with_short_id,
        wait_for_task_completion=wait_for_task_completion,
        cancel_batch_task=cancel_batch_task,
        format_task_failure_message=format_task_failure_message,
        format_task_timing_summary=format_task_timing_summary,
        suppress_cloudops_info_output=lambda: nullcontext(),
    )


def call_details(
    calls: list[tuple[str, dict[str, Any]]],
    name: str,
) -> dict[str, Any]:
    return next(details for call_name, details in calls if call_name == name)


def test_execute_cloud_run_success_uploads_submits_waits_and_downloads(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    import calibrationtools.cloud.executor as executor

    calls: list[tuple[str, dict[str, Any]]] = []
    backend = make_backend(calls)

    def fake_upload_files(client, **kwargs):
        input_path = Path(kwargs["local_root_dir"]) / kwargs["files"]
        calls.append(
            (
                "upload",
                {
                    **kwargs,
                    "payload": json.loads(input_path.read_text()),
                },
            )
        )

    def fake_download(client, **kwargs):
        calls.append(("download", kwargs))
        Path(kwargs["dest_path"]).write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(executor, "upload_files_quietly", fake_upload_files)
    monkeypatch.setattr(
        executor, "download_blob_to_path_atomic", fake_download
    )

    execute_cloud_run(
        make_run_json(tmp_path, print_task_durations=True),
        output_filename="result.csv",
        backend=backend,
    )

    upload = call_details(calls, "upload")
    assert upload["container_name"] == "input-container"
    assert upload["location_in_blob"].endswith(f"/{RUN_ID}")
    assert upload["payload"] == {"run_id": RUN_ID, "value": 3}

    add_task = call_details(calls, "add_task")
    assert add_task["job_name"] == "job-a"
    assert add_task["task_name_suffix"] == RUN_ID
    assert add_task["container_image_name"] == "acr.azurecr.io/model:tag"
    assert add_task["save_logs_path"] == "/mnt/logs"
    assert "/mnt/input/input/session/generation-0" in add_task["command_line"]
    assert (
        "/mnt/output/output/session/generation-0" in add_task["command_line"]
    )

    wait = call_details(calls, "wait")
    assert wait["task_id"] == "task-1"
    assert wait["timeout_minutes"] == 7

    download = call_details(calls, "download")
    assert download["container_name"] == "output-container"
    assert download["src_path"].endswith(f"/{RUN_ID}/result.csv")
    assert Path(download["dest_path"]).read_text(encoding="utf-8") == "ok\n"
    assert "timing summary" in capsys.readouterr().err


def test_execute_cloud_run_missing_cloud_metadata_raises(tmp_path: Path):
    calls: list[tuple[str, dict[str, Any]]] = []

    with pytest.raises(ValueError, match="Cloud runtime metadata is missing"):
        execute_cloud_run(
            {
                "runtime": {},
                "input": {"run_id": RUN_ID},
                "output": {
                    "spec": "filesystem",
                    "dir": str(tmp_path / "output"),
                },
            },
            backend=make_backend(calls),
        )

    assert calls == []


def test_execute_cloud_run_missing_run_id_raises(tmp_path: Path):
    calls: list[tuple[str, dict[str, Any]]] = []

    with pytest.raises(ValueError, match="input `run_id`"):
        execute_cloud_run(
            make_run_json(
                tmp_path,
                input_value={"value": 3},
                include_run_id=False,
            ),
            backend=make_backend(calls),
        )

    assert calls == []


def test_execute_cloud_run_loads_file_backed_input_payload(
    monkeypatch,
    tmp_path: Path,
):
    import calibrationtools.cloud.executor as executor

    calls: list[tuple[str, dict[str, Any]]] = []
    backend = make_backend(calls)
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"run_id": RUN_ID, "value": 9}),
        encoding="utf-8",
    )

    def fake_upload_files(client, **kwargs):
        payload_path = Path(kwargs["local_root_dir"]) / kwargs["files"]
        calls.append(
            ("upload", {"payload": json.loads(payload_path.read_text())})
        )

    monkeypatch.setattr(executor, "upload_files_quietly", fake_upload_files)
    monkeypatch.setattr(
        executor,
        "download_blob_to_path_atomic",
        lambda client, **kwargs: Path(kwargs["dest_path"]).write_text("ok\n"),
    )

    execute_cloud_run(
        make_run_json(
            tmp_path,
            input_value=str(input_path),
            include_run_id=False,
        ),
        backend=backend,
    )

    assert call_details(calls, "upload")["payload"] == {
        "run_id": RUN_ID,
        "value": 9,
    }


def test_execute_cloud_run_wait_failure_cancels_task(monkeypatch, tmp_path):
    import calibrationtools.cloud.executor as executor

    calls: list[tuple[str, dict[str, Any]]] = []
    backend = make_backend(calls, wait_error=TimeoutError("too slow"))
    monkeypatch.setattr(executor, "upload_files_quietly", lambda *a, **k: None)

    with pytest.raises(TimeoutError, match="too slow"):
        execute_cloud_run(make_run_json(tmp_path), backend=backend)

    cancel = call_details(calls, "cancel")
    assert cancel["job_name"] == "job-a"
    assert cancel["task_id"] == "task-1"
    assert not any(name == "download" for name, _ in calls)


def test_execute_cloud_run_failed_task_includes_log_excerpts(
    monkeypatch,
    tmp_path,
):
    import calibrationtools.cloud.executor as executor

    calls: list[tuple[str, dict[str, Any]]] = []
    backend = make_backend(
        calls,
        wait_result={
            "result": "failure",
            "exit_code": 1,
            "task": object(),
        },
    )
    monkeypatch.setattr(executor, "upload_files_quietly", lambda *a, **k: None)
    monkeypatch.setattr(
        executor,
        "read_task_log_excerpts",
        lambda *a, **k: {"stderr": "bad things"},
    )

    with pytest.raises(RuntimeError) as exc_info:
        execute_cloud_run(make_run_json(tmp_path), backend=backend)

    assert "formatted failure" in str(exc_info.value)
    assert "stderr_excerpt='bad things'" in str(exc_info.value)
    format_failure = call_details(calls, "format_failure")
    assert format_failure["logs_container"] == "logs-container"
    assert format_failure["logs_folder"] == f"session/job-a/{RUN_ID}"
