import asyncio
import inspect
import json
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from example_model.cloud_mrp_executor import execute_cloud_run
from example_model.cloud_runner import ExampleModelCloudRunner
from example_model.cloud_utils import (
    DEFAULT_CLOUD_RUNTIME_SETTINGS,
)

from calibrationtools.cloud import (
    CloudSession,
    add_batch_task_with_short_id,
    make_batch_task_id,
    make_batch_task_name_suffix,
    make_session_slug,
    make_stable_batch_task_id_max,
    parse_generation_from_run_id,
    parse_image_tag_from_session_slug,
    parse_particle_from_run_id,
    suppress_cloudops_info_output,
)
from calibrationtools.cloud import runner as cloud_runner_module

LONG_TASK_NAME_SUFFIX = (
    "gen-123456789_particle-123456789_" + "extra_suffix_that_needs_truncation"
)
REPO_ROOT = Path(__file__).resolve().parents[3]
SESSION_SLUG = "20260409010101-testsha-ab12cd34ef56"
POOL_NAME = f"example-model-cloud-{SESSION_SLUG}"
JOB_1 = f"{POOL_NAME}-j1"
JOB_2 = f"{POOL_NAME}-j2"
JOB_3 = f"{POOL_NAME}-j3"
JOB_4 = f"{POOL_NAME}-j4"
INPUT_CONTAINER = f"example-model-cloud-input-{SESSION_SLUG}"
OUTPUT_CONTAINER = f"example-model-cloud-output-{SESSION_SLUG}"
LOGS_CONTAINER = f"example-model-cloud-logs-{SESSION_SLUG}"


def _cloud_settings(**overrides):
    return replace(DEFAULT_CLOUD_RUNTIME_SETTINGS, **overrides)


def _require_cloudops() -> None:
    pytest.importorskip(
        "cfa.cloudops",
        reason="requires the optional cloudops dependency group",
    )


def _dummy_future():
    return SimpleNamespace(
        done=lambda: False,
        set_result=lambda value: None,
        set_exception=lambda exc: None,
    )


class _FakeClient:
    def __init__(self):
        self.cred = SimpleNamespace(
            azure_container_registry_account="fake-registry",
        )
        self.calls: list[tuple[str, tuple, dict]] = []
        self.save_logs_to_blob = None
        self.logs_folder = None
        self.upload_calls = []
        self.download_calls = []
        self.terminated_tasks = []
        self.tasks: dict[tuple[str, str], object] = {}
        self.upload_files_impl: Callable[..., None] | None = None
        self.download_file_impl: Callable[..., None] | None = None
        self.delete_job_impl: Callable[[str], None] | None = None
        self.delete_pool_impl: Callable[[str], None] | None = None
        self.blob_service_client: Any | None = None
        self.pool = SimpleNamespace(
            state="active",
            allocation_state="steady",
            current_dedicated_nodes=1,
            target_dedicated_nodes=1,
            current_low_priority_nodes=0,
            target_low_priority_nodes=0,
            task_slots_per_node=1,
        )
        self.batch_service_client = SimpleNamespace(
            task=SimpleNamespace(
                get=lambda job_id, task_id: self.tasks.get(
                    (job_id, task_id),
                    SimpleNamespace(
                        id=task_id,
                        state="completed",
                        execution_info=SimpleNamespace(
                            result="success",
                            exit_code=0,
                        ),
                    ),
                ),
                list=lambda job_id: [
                    task
                    for (stored_job_id, _), task in self.tasks.items()
                    if stored_job_id == job_id
                ],
                terminate=lambda job_id, task_id: self.terminated_tasks.append(
                    (job_id, task_id)
                ),
                delete=lambda job_id, task_id: self.tasks.pop(
                    (job_id, task_id),
                    None,
                ),
            ),
            pool=SimpleNamespace(get=lambda *_: self.pool),
        )

    def create_blob_container(self, container_name):
        self.calls.append(("create_blob_container", (container_name,), {}))

    def create_job(self, **kwargs):
        self.calls.append(("create_job", (), kwargs))

    def upload_docker_image(self, **kwargs):
        self.calls.append(("upload_docker_image", (), kwargs))
        return "fake-registry.azurecr.io/example-model:testsha"

    def upload_files(self, **kwargs):
        self.upload_calls.append(kwargs)
        if self.upload_files_impl is not None:
            self.upload_files_impl(**kwargs)

    def download_file(self, **kwargs):
        self.download_calls.append(kwargs)
        if self.download_file_impl is not None:
            self.download_file_impl(**kwargs)
            return
        Path(kwargs["dest_path"]).write_text(
            "generation,population\n0,1\n1,2\n"
        )

    def delete_job(self, job_name):
        if self.delete_job_impl is not None:
            self.delete_job_impl(job_name)

    def delete_pool(self, pool_name):
        if self.delete_pool_impl is not None:
            self.delete_pool_impl(pool_name)


def test_cloud_runner_initializes_resources_and_uses_mrp(
    monkeypatch, tmp_path, capsys
):
    fake_client = _FakeClient()
    pool_create_calls = []
    pool_wait_calls = []

    monkeypatch.setattr(
        "example_model.cloud_runner.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.git_short_sha",
        lambda repo_root: "testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.make_session_slug",
        lambda tag: SESSION_SLUG,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.build_local_image",
        lambda **kwargs: "example-local:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.upload_local_image",
        lambda **kwargs: "fake-registry.azurecr.io/example-model:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(
            vm_size="large",
            jobs_per_session=2,
            task_slots_per_node=2,
        ),
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.create_pool_with_blob_mounts",
        lambda **kwargs: pool_create_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.wait_for_pool_ready",
        lambda **kwargs: pool_wait_calls.append(kwargs) or fake_client.pool,
    )

    def fake_mrp_run(config_path, overrides, output_dir: str | None = None):
        assert config_path == Path("example_model.mrp.cloud.toml")
        assert overrides["runtime"]["command"]
        assert overrides["runtime"]["cloud"]["run_id"] == "gen-1_particle-1"
        assert overrides["runtime"]["cloud"]["job_name"].endswith("-j1")
        assert overrides["runtime"]["cloud"]["job_names"]["1"][0].endswith(
            "-j1"
        )
        assert (
            overrides["runtime"]["cloud"]["job_names"]["1"]
            == overrides["runtime"]["cloud"]["job_names"]["2"]
        )
        assert overrides["input"] == str(tmp_path / "input.json")
        assert output_dir == str(tmp_path / "output")
        assert output_dir is not None
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "output.csv").write_text(
            "generation,population\n0,1\n1,2\n"
        )
        return SimpleNamespace(ok=True, stderr=b"")

    monkeypatch.setattr("example_model.cloud_runner.mrp_run", fake_mrp_run)

    runner = ExampleModelCloudRunner(
        Path("example_model.mrp.cloud.toml"),
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        generation_count=2,
        max_concurrent_simulations=3,
    )

    captured = capsys.readouterr()
    assert f"[cloud-run] created pool {POOL_NAME}" in captured.err
    assert "vm_size=large" in captured.err
    assert "max_nodes=5" in captured.err
    assert "task_slots_per_node=2" in captured.err
    assert "max_task_capacity=10" in captured.err
    assert (
        "scaling=auto(max_nodes=5, min_nodes=0, interval=5m)" in captured.err
    )
    assert "created 2 reusable job(s) for 2 generation(s)" in captured.err

    input_path = tmp_path / "input.json"
    input_path.write_text('{"seed": 123, "run_id": "gen-1_particle-1"}')
    output_dir = tmp_path / "output"

    assert runner.simulate(
        {"seed": 123},
        input_path=input_path,
        output_dir=output_dir,
        run_id="gen-1_particle-1",
    ) == [1, 2]

    blob_calls = [
        call
        for call in fake_client.calls
        if call[0] == "create_blob_container"
    ]
    job_calls = [call for call in fake_client.calls if call[0] == "create_job"]

    assert len(blob_calls) == 3
    assert len(job_calls) == 2
    assert len(pool_create_calls) == 1
    assert len(pool_wait_calls) == 1
    assert pool_create_calls[0]["target_dedicated_nodes"] == 5
    assert pool_create_calls[0]["task_slots_per_node"] == 2
    assert pool_create_calls[0]["auto_scale_evaluation_interval_minutes"] == 5
    assert runner.session.job_name_for_run("gen-2_particle-2").endswith("-j2")


def test_cloud_runner_startup_summary_includes_auto_size(capsys):
    from calibrationtools.cloud.runner import CloudMRPRunner

    runner = object.__new__(CloudMRPRunner)
    runner.settings = _cloud_settings(vm_size="large", task_slots_per_node=5)
    runner.max_concurrent_simulations = 5
    runner.generation_count = 2
    runner.auto_size_summary = SimpleNamespace(
        measured_task_peak_rss_bytes=10 * 1024**3,
        vm_memory_bytes=64 * 1024**3,
        reserve=0.15,
        task_slots_per_node=5,
    )

    runner._print_session_startup_summary(
        pool_name=POOL_NAME,
        job_names={"1": [JOB_1], "2": [JOB_1]},
        remote_image_ref="fake-registry.azurecr.io/example-model:testsha",
    )

    captured = capsys.readouterr()
    assert "[cloud-run] auto-size" in captured.err
    assert "measured_peak_rss=10737418240 bytes" in captured.err
    assert "vm_ram=68719476736 bytes" in captured.err
    assert "reserve=15%" in captured.err
    assert "task_slots_per_node=5" in captured.err
    assert "max_concurrent_simulations_total=5" in captured.err


def test_cloud_session_accepts_attempt_suffixed_run_ids():
    session = CloudSession(
        keyvault="cfa-predict",
        session_slug=SESSION_SLUG,
        image_tag="testsha",
        remote_image_ref="fake-registry.azurecr.io/example-model:testsha",
        pool_name=POOL_NAME,
        job_names={
            "1": [JOB_1, JOB_2],
            "2": [JOB_3, JOB_4],
        },
        input_container=INPUT_CONTAINER,
        output_container=OUTPUT_CONTAINER,
        logs_container=LOGS_CONTAINER,
        task_mrp_config_path="/app/example_model.mrp.task.toml",
        input_mount_path="/cloud-input",
        output_mount_path="/cloud-output",
        logs_mount_path="/cloud-logs",
        task_timeout_minutes=60,
        print_task_durations=False,
    )
    run_id = "gen-2_particle-2-attempt-2"

    assert parse_generation_from_run_id(run_id) == 2
    assert parse_particle_from_run_id(run_id) == 2
    assert session.job_name_for_run(run_id) == JOB_4
    assert session.remote_input_dir(run_id).endswith(
        "generation-2/gen-2_particle-2-attempt-2"
    )
    assert session.remote_output_dir(run_id).endswith(
        "generation-2/gen-2_particle-2-attempt-2"
    )


def test_cloud_executor_import_does_not_eagerly_import_cloud_runner():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import example_model.cloud_mrp_executor; "
                "assert 'example_model.cloud_runner' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_cloud_executor_uploads_input_submits_task_and_downloads_output(
    monkeypatch, tmp_path
):
    class FakeExecutorClient:
        def __init__(self):
            self.save_logs_to_blob = None
            self.logs_folder = None
            self.upload_calls = []
            self.download_calls = []
            self.batch_service_client = SimpleNamespace(
                task=SimpleNamespace(get=lambda *_: None)
            )

        def upload_files(self, **kwargs):
            self.upload_calls.append(kwargs)

        def download_file(self, **kwargs):
            self.download_calls.append(kwargs)
            Path(kwargs["dest_path"]).write_text(
                "generation,population\n0,1\n1,2\n"
            )

    fake_client = FakeExecutorClient()
    task_submit_calls = []
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.add_batch_task_with_short_id",
        lambda **kwargs: task_submit_calls.append(kwargs) or "task-123",
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.wait_for_task_completion",
        lambda **kwargs: {"result": "success", "exit_code": 0},
    )

    session = CloudSession(
        keyvault="cfa-predict",
        session_slug=SESSION_SLUG,
        image_tag="testsha",
        remote_image_ref="fake-registry.azurecr.io/example-model:testsha",
        pool_name=POOL_NAME,
        job_names={"1": [JOB_1, JOB_2]},
        input_container=INPUT_CONTAINER,
        output_container=OUTPUT_CONTAINER,
        logs_container=LOGS_CONTAINER,
        task_mrp_config_path="/app/example_model.mrp.task.toml",
        input_mount_path="/cloud-input",
        output_mount_path="/cloud-output",
        logs_mount_path="/cloud-logs",
        task_timeout_minutes=60,
        print_task_durations=False,
    )
    output_dir = tmp_path / "output"
    run_json = {
        "input": {"seed": 123, "run_id": "gen-1_particle-1"},
        "output": {"spec": "filesystem", "dir": str(output_dir)},
        "runtime": {
            "cloud": {
                **session.to_runtime_cloud(),
                "run_id": "gen-1_particle-1",
                "job_name": JOB_2,
            }
        },
    }

    execute_cloud_run(run_json)

    assert fake_client.save_logs_to_blob == session.logs_container
    assert (
        fake_client.logs_folder == f"{SESSION_SLUG}/{JOB_2}/gen-1_particle-1"
    )
    assert (
        fake_client.upload_calls[0]["container_name"]
        == session.input_container
    )
    assert fake_client.upload_calls[0]["location_in_blob"].endswith(
        "generation-1/gen-1_particle-1"
    )
    assert task_submit_calls[0]["task_name_suffix"] == "gen-1_particle-1"
    assert task_submit_calls[0]["task_id_max_override"] == (
        make_stable_batch_task_id_max(
            JOB_2,
            "gen-1_particle-1",
        )
    )
    assert (
        task_submit_calls[0]["container_image_name"]
        == session.remote_image_ref
    )
    assert task_submit_calls[0]["mount_pairs"] == session.mount_pairs()
    assert task_submit_calls[0]["save_logs_path"] == session.logs_mount_path
    task_command = task_submit_calls[0]["command_line"]
    assert "mrp run /app/example_model.mrp.task.toml" in task_command
    assert (
        f"/cloud-input/input/{SESSION_SLUG}/generation-1/gen-1_particle-1"
        in task_command
    )
    assert (
        f"/cloud-output/output/{SESSION_SLUG}/generation-1/gen-1_particle-1"
        in task_command
    )
    assert fake_client.download_calls[0]["src_path"].endswith(
        "generation-1/gen-1_particle-1/output.csv"
    )
    assert (
        (output_dir / "output.csv")
        .read_text()
        .startswith("generation,population")
    )


def test_cloud_executor_reads_local_json_input_path_before_upload(
    monkeypatch, tmp_path
):
    class FakeExecutorClient:
        def __init__(self):
            self.save_logs_to_blob = None
            self.logs_folder = None
            self.upload_calls = []
            self.uploaded_payloads = []
            self.download_calls = []
            self.batch_service_client = SimpleNamespace(
                task=SimpleNamespace(get=lambda *_: None)
            )

        def upload_files(self, **kwargs):
            self.upload_calls.append(kwargs)
            uploaded_path = Path(kwargs["local_root_dir"]) / kwargs["files"]
            self.uploaded_payloads.append(
                json.loads(uploaded_path.read_text())
            )

        def download_file(self, **kwargs):
            self.download_calls.append(kwargs)
            Path(kwargs["dest_path"]).write_text(
                "generation,population\n0,1\n1,2\n"
            )

    fake_client = FakeExecutorClient()
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.add_batch_task_with_short_id",
        lambda **kwargs: "task-123",
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.wait_for_task_completion",
        lambda **kwargs: {"result": "success", "exit_code": 0},
    )

    session = CloudSession(
        keyvault="cfa-predict",
        session_slug=SESSION_SLUG,
        image_tag="testsha",
        remote_image_ref="fake-registry.azurecr.io/example-model:testsha",
        pool_name=POOL_NAME,
        job_names={"1": [JOB_1]},
        input_container=INPUT_CONTAINER,
        output_container=OUTPUT_CONTAINER,
        logs_container=LOGS_CONTAINER,
        task_mrp_config_path="/app/example_model.mrp.task.toml",
        input_mount_path="/cloud-input",
        output_mount_path="/cloud-output",
        logs_mount_path="/cloud-logs",
        task_timeout_minutes=60,
        print_task_durations=False,
    )
    staged_input_path = tmp_path / "input.json"
    staged_input_path.write_text('{"seed": 123, "run_id": "gen-1_particle-1"}')
    output_dir = tmp_path / "output"
    run_json = {
        "input": str(staged_input_path),
        "output": {"spec": "filesystem", "dir": str(output_dir)},
        "runtime": {
            "cloud": {
                **session.to_runtime_cloud(),
                "run_id": "gen-1_particle-1",
            }
        },
    }

    execute_cloud_run(run_json)

    assert fake_client.uploaded_payloads == [
        {"seed": 123, "run_id": "gen-1_particle-1"}
    ]
    assert fake_client.upload_calls[0]["location_in_blob"].endswith(
        "generation-1/gen-1_particle-1"
    )


def test_make_batch_task_name_suffix_stays_within_azure_limit():
    suffix = make_batch_task_name_suffix(LONG_TASK_NAME_SUFFIX)

    assert len(f"task-{suffix}-1") <= 64
    assert suffix


def test_make_session_slug_uses_uuid_to_avoid_same_second_collisions(
    monkeypatch,
):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 16, 10, 11, 12, tzinfo=tz)

    class FakeUuid:
        def __init__(self, hex_value: str):
            self.hex = hex_value

    uuids = iter(
        [
            FakeUuid("aaaaaaaaaaaabbbbbbbbbbbbcccccccc"),
            FakeUuid("ddddddddddddeeeeeeeeeeeeffffffff"),
        ]
    )

    monkeypatch.setattr(
        "calibrationtools.cloud.naming.datetime", FrozenDateTime
    )
    monkeypatch.setattr(
        "calibrationtools.cloud.naming.uuid4", lambda: next(uuids)
    )

    first = make_session_slug("testsha")
    second = make_session_slug("testsha")

    assert first == "20260416101112-testsha-aaaaaaaaaaaa"
    assert second == "20260416101112-testsha-dddddddddddd"
    assert first != second


def test_make_batch_task_id_uses_shortened_suffix():
    task_id = make_batch_task_id("gen-1_particle-1")

    assert task_id == "task-gen-1_particle-1-1"


def test_make_stable_batch_task_id_max_is_stable_and_distinct():
    first = make_stable_batch_task_id_max("job-1", "gen-1_particle-1")
    second = make_stable_batch_task_id_max("job-1", "gen-1_particle-1")
    third = make_stable_batch_task_id_max("job-1", "gen-1_particle-2")

    assert first == second
    assert first != third
    assert first >= 0


def test_add_batch_task_with_short_id_uses_relative_mount_paths(monkeypatch):
    _require_cloudops()
    task_add_calls = []

    class FakeBatchClient:
        def __init__(self):
            self.job = SimpleNamespace(
                get=lambda job_name: SimpleNamespace(
                    as_dict=lambda: {"execution_info": {"pool_id": "pool-1"}}
                )
            )
            self.task = SimpleNamespace(
                add=lambda **kwargs: task_add_calls.append(kwargs)
            )

    fake_client = SimpleNamespace(
        batch_service_client=FakeBatchClient(),
        batch_mgmt_client=SimpleNamespace(),
        cred=SimpleNamespace(
            azure_resource_group_name="rg",
            azure_batch_account="acct",
        ),
        logs_folder="session/job-1",
    )

    task_id = add_batch_task_with_short_id(
        client=fake_client,
        job_name="job-rel-mounts",
        command_line="echo hello",
        task_name_suffix="gen-1_particle-1",
        timeout=60,
        mount_pairs=[
            {
                "source": INPUT_CONTAINER,
                "target": "/cloud-input",
            },
            {
                "source": OUTPUT_CONTAINER,
                "target": "/cloud-output",
            },
        ],
        container_image_name="fake-registry.azurecr.io/example-model:testsha",
        save_logs_path="/cloud-logs",
    )
    second_task_id = add_batch_task_with_short_id(
        client=fake_client,
        job_name="job-rel-mounts",
        command_line="echo hello again",
        task_name_suffix=LONG_TASK_NAME_SUFFIX,
        timeout=60,
        mount_pairs=[
            {
                "source": INPUT_CONTAINER,
                "target": "/cloud-input",
            },
            {
                "source": OUTPUT_CONTAINER,
                "target": "/cloud-output",
            },
        ],
        container_image_name="fake-registry.azurecr.io/example-model:testsha",
        save_logs_path="/cloud-logs",
    )

    assert task_id == "task-gen-1_particle-1-1"
    assert second_task_id.startswith("task-")
    first_task = task_add_calls[0]["task"]
    second_task = task_add_calls[1]["task"]
    assert task_add_calls[0]["job_id"] == "job-rel-mounts"
    run_options = first_task.container_settings.container_run_options
    assert "--name=job-rel-mounts_1" in run_options
    assert "--rm" in run_options
    # Batch tasks intentionally override the image's non-root default user so
    # BlobFuse-mounted output and log directories are writable.
    assert "--user 0:0" in run_options
    assert "source=" in run_options
    assert "target=/cloud-input" in run_options
    assert "target=/cloud-output" in run_options
    assert first_task.id == "task-gen-1_particle-1-1"
    assert second_task.id.endswith("-2")
    assert len(second_task.id) <= 64


def test_add_batch_task_with_short_id_tracks_override_for_future_calls(
    monkeypatch,
):
    _require_cloudops()
    task_add_calls = []

    class FakeBatchClient:
        def __init__(self):
            self.job = SimpleNamespace(
                get=lambda job_name: SimpleNamespace(
                    as_dict=lambda: {"execution_info": {"pool_id": "pool-1"}}
                )
            )
            self.task = SimpleNamespace(
                add=lambda **kwargs: task_add_calls.append(kwargs)
            )

    fake_client = SimpleNamespace(
        batch_service_client=FakeBatchClient(),
        batch_mgmt_client=SimpleNamespace(),
        cred=SimpleNamespace(
            azure_resource_group_name="rg",
            azure_batch_account="acct",
        ),
        logs_folder="session/job-1",
    )

    add_batch_task_with_short_id(
        client=fake_client,
        job_name="job-override",
        command_line="echo hello",
        task_name_suffix="gen-1_particle-1",
        timeout=60,
        container_image_name="fake-registry.azurecr.io/example-model:testsha",
        task_id_max_override=41,
    )
    add_batch_task_with_short_id(
        client=fake_client,
        job_name="job-override",
        command_line="echo again",
        task_name_suffix="gen-1_particle-2",
        timeout=60,
        container_image_name="fake-registry.azurecr.io/example-model:testsha",
    )

    assert task_add_calls[0]["task"].id.endswith("-42")
    assert task_add_calls[1]["task"].id.endswith("-43")


def test_suppress_cloudops_info_output_hides_stdout_and_stderr(capsys):
    import logging

    logger = logging.getLogger("cfa.cloudops")
    logger.setLevel(logging.DEBUG)
    try:
        # The helper only suppresses the cfa.cloudops logger; it must not
        # touch process-global sys.stdout/sys.stderr because it runs on
        # worker threads where that would race with the main thread.
        with suppress_cloudops_info_output():
            assert logger.getEffectiveLevel() >= logging.WARNING
            print("upload stdout noise")
            print("upload stderr noise", file=sys.stderr)

        # Level restored after the block.
        assert logger.getEffectiveLevel() == logging.DEBUG
    finally:
        logger.setLevel(logging.NOTSET)

    captured = capsys.readouterr()
    # Prints go through untouched now that stdio is no longer redirected.
    assert "upload stdout noise" in captured.out
    assert "upload stderr noise" in captured.err


def test_cloud_executor_upload_input_suppresses_noisy_cloudops_progress(
    monkeypatch, tmp_path, capsys
):
    uploaded_blobs: list[tuple[str, str, bytes]] = []

    class _QuietBlobClient:
        def __init__(self, *, container: str, blob: str):
            self.container = container
            self.blob = blob

        def upload_blob(self, upload_data, **kwargs):
            uploaded_blobs.append(
                (self.container, self.blob, upload_data.read())
            )

    class _QuietBlobServiceClient:
        def get_blob_client(self, *, container, blob):
            return _QuietBlobClient(container=container, blob=blob)

    class FakeExecutorClient:
        def __init__(self):
            self.save_logs_to_blob = None
            self.logs_folder = None
            self.upload_calls = []
            self.download_calls = []
            self.blob_service_client = _QuietBlobServiceClient()
            self.batch_service_client = SimpleNamespace(
                task=SimpleNamespace(get=lambda *_: None)
            )

        def upload_files(self, **kwargs):
            self.upload_calls.append(kwargs)
            print("Uploading files: 100%")
            print("Uploaded 1 files to blob storage container")

        def download_file(self, **kwargs):
            self.download_calls.append(kwargs)
            Path(kwargs["dest_path"]).write_text(
                "generation,population\n0,1\n1,2\n"
            )

    fake_client = FakeExecutorClient()
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.add_batch_task_with_short_id",
        lambda **kwargs: "task-123",
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.wait_for_task_completion",
        lambda **kwargs: {"result": "success", "exit_code": 0, "task": None},
    )

    session = CloudSession(
        keyvault="cfa-predict",
        session_slug=SESSION_SLUG,
        image_tag="testsha",
        remote_image_ref="fake-registry.azurecr.io/example-model:testsha",
        pool_name=POOL_NAME,
        job_names={"1": [JOB_1]},
        input_container=INPUT_CONTAINER,
        output_container=OUTPUT_CONTAINER,
        logs_container=LOGS_CONTAINER,
        task_mrp_config_path="/app/example_model.mrp.task.toml",
        input_mount_path="/cloud-input",
        output_mount_path="/cloud-output",
        logs_mount_path="/cloud-logs",
        task_timeout_minutes=60,
        print_task_durations=False,
    )
    run_json = {
        "input": {"seed": 123, "run_id": "gen-1_particle-1"},
        "output": {"spec": "filesystem", "dir": str(tmp_path / "output")},
        "runtime": {
            "cloud": {
                **session.to_runtime_cloud(),
                "run_id": "gen-1_particle-1",
            }
        },
    }

    execute_cloud_run(run_json)

    captured = capsys.readouterr()
    assert "Uploading files:" not in captured.out
    assert "Uploading files:" not in captured.err
    assert "Uploaded 1 files to blob storage container" not in captured.out
    assert fake_client.upload_calls == []
    assert uploaded_blobs


def test_cloud_executor_can_print_task_durations(
    monkeypatch, tmp_path, capsys
):
    class FakeExecutorClient:
        def __init__(self):
            self.save_logs_to_blob = None
            self.logs_folder = None
            self.upload_calls = []
            self.download_calls = []
            self.batch_service_client = SimpleNamespace(
                task=SimpleNamespace(get=lambda *_: None)
            )

        def upload_files(self, **kwargs):
            self.upload_calls.append(kwargs)

        def download_file(self, **kwargs):
            self.download_calls.append(kwargs)
            Path(kwargs["dest_path"]).write_text(
                "generation,population\n0,1\n1,2\n"
            )

    fake_client = FakeExecutorClient()
    completed_task = SimpleNamespace(
        creation_time=datetime(2026, 4, 12, 17, 0, 0, tzinfo=timezone.utc),
        execution_info=SimpleNamespace(
            start_time=datetime(2026, 4, 12, 17, 0, 2, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 12, 17, 0, 7, tzinfo=timezone.utc),
        ),
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.add_batch_task_with_short_id",
        lambda **kwargs: "task-123",
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.wait_for_task_completion",
        lambda **kwargs: {
            "result": "success",
            "exit_code": 0,
            "task": completed_task,
        },
    )

    session = CloudSession(
        keyvault="cfa-predict",
        session_slug=SESSION_SLUG,
        image_tag="testsha",
        remote_image_ref="fake-registry.azurecr.io/example-model:testsha",
        pool_name=POOL_NAME,
        job_names={"1": [JOB_1]},
        input_container=INPUT_CONTAINER,
        output_container=OUTPUT_CONTAINER,
        logs_container=LOGS_CONTAINER,
        task_mrp_config_path="/app/example_model.mrp.task.toml",
        input_mount_path="/cloud-input",
        output_mount_path="/cloud-output",
        logs_mount_path="/cloud-logs",
        task_timeout_minutes=60,
        print_task_durations=True,
    )
    output_dir = tmp_path / "output"
    run_json = {
        "input": {"seed": 123, "run_id": "gen-1_particle-1"},
        "output": {"spec": "filesystem", "dir": str(output_dir)},
        "runtime": {
            "cloud": {
                **session.to_runtime_cloud(),
                "run_id": "gen-1_particle-1",
            }
        },
    }

    execute_cloud_run(run_json)

    captured = capsys.readouterr()
    assert "[cloud-task] gen-1_particle-1" in captured.err
    assert "queue=2.00s" in captured.err
    assert "run=5.00s" in captured.err


def test_cloud_executor_failure_includes_batch_details(monkeypatch, tmp_path):
    class FakeExecutorClient:
        def __init__(self):
            self.save_logs_to_blob = None
            self.logs_folder = None
            self.upload_calls = []
            self.batch_service_client = SimpleNamespace(
                task=SimpleNamespace(get=lambda *_: None)
            )

        def upload_files(self, **kwargs):
            self.upload_calls.append(kwargs)

    fake_client = FakeExecutorClient()
    failed_task = SimpleNamespace(
        id="task-gen-1_particle-1-1",
        state="completed",
        exit_conditions=SimpleNamespace(
            pre_processing_error="mount failed",
            file_upload_error=None,
        ),
        execution_info=SimpleNamespace(
            result="failure",
            exit_code=None,
            failure_info=SimpleNamespace(
                category="servererror",
                code="TaskFailedToStart",
                message="Container failed before process launch",
                details=[
                    SimpleNamespace(name="detail", value="image pull timeout")
                ],
            ),
            container_info=SimpleNamespace(error="container create error"),
        ),
    )

    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.add_batch_task_with_short_id",
        lambda **kwargs: "task-gen-1_particle-1-1",
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.wait_for_task_completion",
        lambda **kwargs: {
            "result": "failure",
            "exit_code": None,
            "task": failed_task,
        },
    )

    session = CloudSession(
        keyvault="cfa-predict",
        session_slug=SESSION_SLUG,
        image_tag="testsha",
        remote_image_ref="fake-registry.azurecr.io/example-model:testsha",
        pool_name=POOL_NAME,
        job_names={"1": [JOB_1]},
        input_container=INPUT_CONTAINER,
        output_container=OUTPUT_CONTAINER,
        logs_container=LOGS_CONTAINER,
        task_mrp_config_path="/app/example_model.mrp.task.toml",
        input_mount_path="/cloud-input",
        output_mount_path="/cloud-output",
        logs_mount_path="/cloud-logs",
        task_timeout_minutes=60,
        print_task_durations=False,
    )
    output_dir = tmp_path / "output"
    run_json = {
        "input": {"seed": 123, "run_id": "gen-1_particle-1"},
        "output": {"spec": "filesystem", "dir": str(output_dir)},
        "runtime": {
            "cloud": {
                **session.to_runtime_cloud(),
                "run_id": "gen-1_particle-1",
            }
        },
    }

    with pytest.raises(RuntimeError) as exc_info:
        execute_cloud_run(run_json)

    message = str(exc_info.value)
    assert "task-gen-1_particle-1-1" in message
    assert "result='failure'" in message
    assert "TaskFailedToStart" in message
    assert "image pull timeout" in message
    assert "container create error" in message
    assert "pre_processing_error='mount failed'" in message
    assert (
        f"logs_prefix={LOGS_CONTAINER}/{SESSION_SLUG}/{JOB_1}/gen-1_particle-1/stdout_stderr/"
    ) in message


def test_cloud_executor_failure_includes_task_log_excerpts(
    monkeypatch, tmp_path
):
    class FakeExecutorClient:
        def __init__(self):
            self.save_logs_to_blob = None
            self.logs_folder = None
            self.batch_service_client = SimpleNamespace(
                task=SimpleNamespace(get=lambda *_: None)
            )

        def upload_files(self, **kwargs):
            return None

        def download_file(self, **kwargs):
            src_path = kwargs["src_path"]
            dest_path = Path(kwargs["dest_path"])
            if src_path.endswith("/stderr.txt"):
                dest_path.write_text(
                    "Traceback (most recent call last):\nValueError: bad particle\n"
                )
                return
            if src_path.endswith("/stdout.txt"):
                dest_path.write_text("starting task\nseed=123\n")
                return
            raise FileNotFoundError(src_path)

    fake_client = FakeExecutorClient()
    failed_task = SimpleNamespace(
        id="task-gen-1_particle-1-1",
        state="completed",
        execution_info=SimpleNamespace(
            result="failure",
            exit_code=1,
            failure_info=SimpleNamespace(
                category="usererror",
                code="FailureExitCode",
                message="The task exited with an exit code representing a failure",
                details=[],
            ),
        ),
    )

    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.add_batch_task_with_short_id",
        lambda **kwargs: "task-gen-1_particle-1-1",
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.wait_for_task_completion",
        lambda **kwargs: {
            "result": "failure",
            "exit_code": 1,
            "task": failed_task,
        },
    )

    session = CloudSession(
        keyvault="cfa-predict",
        session_slug=SESSION_SLUG,
        image_tag="testsha",
        remote_image_ref="fake-registry.azurecr.io/example-model:testsha",
        pool_name=POOL_NAME,
        job_names={"1": [JOB_1]},
        input_container=INPUT_CONTAINER,
        output_container=OUTPUT_CONTAINER,
        logs_container=LOGS_CONTAINER,
        task_mrp_config_path="/app/example_model.mrp.task.toml",
        input_mount_path="/cloud-input",
        output_mount_path="/cloud-output",
        logs_mount_path="/cloud-logs",
        task_timeout_minutes=60,
        print_task_durations=False,
    )
    run_json = {
        "input": {"seed": 123, "run_id": "gen-1_particle-1"},
        "output": {"spec": "filesystem", "dir": str(tmp_path / "output")},
        "runtime": {
            "cloud": {
                **session.to_runtime_cloud(),
                "run_id": "gen-1_particle-1",
            }
        },
    }

    with pytest.raises(RuntimeError) as exc_info:
        execute_cloud_run(run_json)

    message = str(exc_info.value)
    assert (
        "stderr_excerpt='Traceback (most recent call last):\\nValueError: bad particle'"
        in message
    )
    assert "stdout_excerpt='starting task\\nseed=123'" in message


def test_cloud_runner_describe_progress_reports_batch_and_pool_state(
    monkeypatch,
):
    fake_client = _FakeClient()
    task_map = {
        (
            JOB_1,
            "task-gen-1_particle-1-1",
        ): SimpleNamespace(state="active"),
        (
            JOB_2,
            "task-gen-1_particle-2-1",
        ): SimpleNamespace(state="running"),
    }
    pool = SimpleNamespace(
        state="active",
        allocation_state="resizing",
        current_dedicated_nodes=2,
        target_dedicated_nodes=3,
        current_low_priority_nodes=0,
        target_low_priority_nodes=0,
        task_slots_per_node=1,
    )
    fake_client.batch_service_client = SimpleNamespace(
        task=SimpleNamespace(
            get=lambda job_id, task_id: task_map[(job_id, task_id)],
            list=lambda job_id: [
                SimpleNamespace(id=task_id, state=task.state)
                for (stored_job_id, task_id), task in task_map.items()
                if stored_job_id == job_id
            ],
            terminate=lambda *_: None,
            delete=lambda *_: None,
        ),
        pool=SimpleNamespace(get=lambda *_: pool),
    )

    monkeypatch.setattr(
        "example_model.cloud_runner.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.git_short_sha",
        lambda repo_root: "testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.make_session_slug",
        lambda tag: SESSION_SLUG,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.build_local_image",
        lambda **kwargs: "example-local:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.upload_local_image",
        lambda **kwargs: "fake-registry.azurecr.io/example-model:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(
            vm_size="large",
            jobs_per_session=2,
        ),
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.create_pool_with_blob_mounts",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.wait_for_pool_ready",
        lambda **kwargs: fake_client.pool,
    )

    runner = ExampleModelCloudRunner(
        Path("example_model.mrp.cloud.toml"),
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        generation_count=1,
        max_concurrent_simulations=3,
    )
    runner._register_active_run(
        "gen-1_particle-1",
        JOB_1,
        output_dir=Path("/tmp/test-output-progress-1"),
        input_payload={"run_id": "gen-1_particle-1"},
        overall_started=0.0,
        future=_dummy_future(),
    )
    runner._set_task_id(
        "gen-1_particle-1",
        task_id="task-gen-1_particle-1-1",
        upload_elapsed_seconds=0.0,
        submitted_at=0.0,
    )
    runner._register_active_run(
        "gen-1_particle-2",
        JOB_2,
        output_dir=Path("/tmp/test-output-progress-2"),
        input_payload={"run_id": "gen-1_particle-2"},
        overall_started=0.0,
        future=_dummy_future(),
    )
    runner._set_task_id(
        "gen-1_particle-2",
        task_id="task-gen-1_particle-2-1",
        upload_elapsed_seconds=0.0,
        submitted_at=0.0,
    )
    # particle-3 is admitted but has not yet been assigned a Batch task
    # id, so it should be counted as ``submitting=1`` below.
    runner._register_active_run(
        "gen-1_particle-3",
        JOB_1,
        output_dir=Path("/tmp/test-output-progress-3"),
        input_payload={"run_id": "gen-1_particle-3"},
        overall_started=0.0,
        future=_dummy_future(),
    )

    # describe_progress now reads from a cached snapshot that the
    # controller's background task refreshes; prime the cache
    # synchronously so the assertions below see the current batch state.
    runner._refresh_progress_cache_blocking()

    status = runner.describe_progress(
        (
            "gen-1_particle-1",
            "gen-1_particle-2",
            "gen-1_particle-3",
        )
    )

    assert status is not None
    assert "batch(submitting=1, active=1, running=1, completed=0)" in status
    assert (
        "pool(state=active, allocation=resizing, dedicated=2/3, "
        "low_priority=0/0, task_slots=1)"
    ) in status


def test_cloud_runner_selects_least_busy_job(monkeypatch):
    fake_client = _FakeClient()

    monkeypatch.setattr(
        "example_model.cloud_runner.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.git_short_sha",
        lambda repo_root: "testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.make_session_slug",
        lambda tag: SESSION_SLUG,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.build_local_image",
        lambda **kwargs: "example-local:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.upload_local_image",
        lambda **kwargs: "fake-registry.azurecr.io/example-model:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(
            vm_size="large",
            jobs_per_session=2,
        ),
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.create_pool_with_blob_mounts",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.wait_for_pool_ready",
        lambda **kwargs: fake_client.pool,
    )

    runner = ExampleModelCloudRunner(
        Path("example_model.mrp.cloud.toml"),
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        generation_count=1,
        max_concurrent_simulations=2,
    )
    job_1, job_2 = runner.session.job_names["1"]
    runner._register_active_run(
        "gen-1_particle-1",
        job_1,
        output_dir=Path("/tmp/test-output-select"),
        input_payload={"run_id": "gen-1_particle-1"},
        overall_started=0.0,
        future=_dummy_future(),
    )

    assert runner._select_job_name("gen-1_particle-2") == job_2


def test_cloud_runner_reports_configured_dispatch_buffer(monkeypatch):
    fake_client = _FakeClient()

    monkeypatch.setattr(
        "example_model.cloud_runner.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.git_short_sha",
        lambda repo_root: "testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.make_session_slug",
        lambda tag: SESSION_SLUG,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.build_local_image",
        lambda **kwargs: "example-local:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.upload_local_image",
        lambda **kwargs: "fake-registry.azurecr.io/example-model:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(
            vm_size="large",
            jobs_per_session=1,
            dispatch_buffer=1000,
        ),
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.create_pool_with_blob_mounts",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.wait_for_pool_ready",
        lambda **kwargs: fake_client.pool,
    )

    runner = ExampleModelCloudRunner(
        Path("example_model.mrp.cloud.toml"),
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        generation_count=1,
        max_concurrent_simulations=50,
    )

    assert runner.dispatch_buffer_size() == 1000


def test_cloud_runner_async_simulate_awaits_completion_future_without_spin(
    monkeypatch, tmp_path
):
    fake_client = _FakeClient()
    sleep_calls: list[float] = []
    original_sleep = cloud_runner_module.asyncio.sleep

    async def tracking_sleep(delay, *args, **kwargs):
        sleep_calls.append(delay)
        return await original_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(cloud_runner_module.asyncio, "sleep", tracking_sleep)
    monkeypatch.setattr(
        "example_model.cloud_runner.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.git_short_sha",
        lambda repo_root: "testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.make_session_slug",
        lambda tag: SESSION_SLUG,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.build_local_image",
        lambda **kwargs: "example-local:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.upload_local_image",
        lambda **kwargs: "fake-registry.azurecr.io/example-model:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(
            vm_size="large",
            jobs_per_session=1,
        ),
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.create_pool_with_blob_mounts",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.wait_for_pool_ready",
        lambda **kwargs: fake_client.pool,
    )

    runner = ExampleModelCloudRunner(
        Path("example_model.mrp.cloud.toml"),
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        generation_count=1,
        max_concurrent_simulations=1,
    )

    async def fake_submit_run_async(run_id: str) -> None:
        async def finish_run() -> None:
            await original_sleep(0.01)
            runner._resolve_run_success(run_id, [1, 2])

        asyncio.create_task(finish_run())

    runner._submit_run_async = fake_submit_run_async
    runner._ensure_controller_started = lambda: None
    runner._raise_controller_failure = lambda: None

    try:
        result = asyncio.run(
            runner.simulate_async(
                {"seed": 123, "run_id": "gen-1_particle-1"},
                output_dir=tmp_path / "output",
                run_id="gen-1_particle-1",
            )
        )
    finally:
        runner.close()

    assert result == [1, 2]
    assert 0 not in sleep_calls


def test_cloud_runner_async_simulate_enforces_inflight_limit(
    monkeypatch, tmp_path
):
    fake_client = _FakeClient()
    submit_lock = threading.Lock()
    release_submissions = threading.Event()
    first_wave_started = threading.Event()
    entered_run_ids: list[str] = []
    active_submissions = 0
    max_active_submissions = 0

    monkeypatch.setattr(
        "example_model.cloud_runner.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.git_short_sha",
        lambda repo_root: "testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.make_session_slug",
        lambda tag: SESSION_SLUG,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.build_local_image",
        lambda **kwargs: "example-local:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.upload_local_image",
        lambda **kwargs: "fake-registry.azurecr.io/example-model:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(
            vm_size="large",
            jobs_per_session=2,
            dispatch_buffer=1,
        ),
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.create_pool_with_blob_mounts",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.wait_for_pool_ready",
        lambda **kwargs: fake_client.pool,
    )

    runner = ExampleModelCloudRunner(
        Path("example_model.mrp.cloud.toml"),
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        generation_count=1,
        max_concurrent_simulations=2,
    )

    def fake_submit_run_blocking(run_id: str) -> dict[str, object]:
        nonlocal active_submissions, max_active_submissions
        with submit_lock:
            active_submissions += 1
            max_active_submissions = max(
                max_active_submissions,
                active_submissions,
            )
            entered_run_ids.append(run_id)
            if len(entered_run_ids) >= 2:
                first_wave_started.set()

        assert release_submissions.wait(timeout=5)

        with submit_lock:
            active_submissions -= 1

        job_name = runner.session.job_name_for_run(run_id)
        return {
            "job_name": job_name,
            "task_id": f"task-{run_id}",
            "upload_elapsed_seconds": 0.0,
            "submitted_at": 0.0,
        }

    async def fake_wait_for_task_completion_async(
        *,
        client,
        job_name: str,
        task_id: str,
        run_id: str,
    ) -> dict[str, object]:
        del client, job_name
        await asyncio.sleep(0.01)
        return {
            "result": "success",
            "exit_code": 0,
            "task": SimpleNamespace(
                id=task_id,
                state="completed",
                execution_info=SimpleNamespace(
                    result="success",
                    exit_code=0,
                ),
            ),
        }

    def fake_download_output_blocking(run_id: str, output_dir: Path) -> float:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "output.csv").write_text(
            "generation,population\n0,1\n1,2\n"
        )
        return 0.0

    runner._submit_run_blocking = fake_submit_run_blocking
    runner._wait_for_task_completion_async = (
        fake_wait_for_task_completion_async
    )
    runner._download_output_blocking = fake_download_output_blocking

    async def run_batch() -> list[list[int]]:
        run_ids = [
            "gen-1_particle-1",
            "gen-1_particle-2",
            "gen-1_particle-3",
            "gen-1_particle-4",
        ]
        tasks = [
            asyncio.create_task(
                runner.simulate_async(
                    {"seed": index, "run_id": run_id},
                    output_dir=tmp_path / run_id,
                    run_id=run_id,
                )
            )
            for index, run_id in enumerate(run_ids, start=1)
        ]
        # Generous timeout: on slow CI the first two submissions can take
        # longer than a second to be scheduled and enter the hook. The value
        # is still small enough to surface a genuine deadlock.
        assert await asyncio.to_thread(first_wave_started.wait, 30.0)
        with submit_lock:
            assert entered_run_ids == [
                "gen-1_particle-1",
                "gen-1_particle-2",
            ]
            assert max_active_submissions == 2

        release_submissions.set()
        return await asyncio.gather(*tasks)

    try:
        results = asyncio.run(run_batch())
    finally:
        runner.close()

    assert results == [[1, 2], [1, 2], [1, 2], [1, 2]]
    assert max_active_submissions == 2


def test_cloud_runner_async_simulate_uploads_submits_and_downloads(
    monkeypatch, tmp_path
):
    fake_client = _FakeClient()
    create_client_calls = []
    task_submit_calls = []

    def fake_create_cloud_client(*, keyvault):
        create_client_calls.append(keyvault)
        return fake_client

    monkeypatch.setattr(
        "example_model.cloud_runner.create_cloud_client",
        fake_create_cloud_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.git_short_sha",
        lambda repo_root: "testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.make_session_slug",
        lambda tag: SESSION_SLUG,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.build_local_image",
        lambda **kwargs: "example-local:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.upload_local_image",
        lambda **kwargs: "fake-registry.azurecr.io/example-model:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(
            vm_size="large",
            jobs_per_session=2,
        ),
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.create_pool_with_blob_mounts",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.wait_for_pool_ready",
        lambda **kwargs: fake_client.pool,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.DEFAULT_POLL_INTERVAL_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.add_batch_task_with_short_id",
        lambda **kwargs: (
            task_submit_calls.append(kwargs),
            fake_client.tasks.__setitem__(
                (kwargs["job_name"], "task-123"),
                SimpleNamespace(
                    id="task-123",
                    state="completed",
                    execution_info=SimpleNamespace(
                        result="success",
                        exit_code=0,
                    ),
                ),
            ),
            "task-123",
        )[-1],
    )

    runner = ExampleModelCloudRunner(
        Path("example_model.mrp.cloud.toml"),
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        generation_count=1,
        max_concurrent_simulations=2,
    )

    try:
        output_dir = tmp_path / "output"
        result = asyncio.run(
            runner.simulate_async(
                {"seed": 123, "run_id": "gen-1_particle-1"},
                output_dir=output_dir,
                run_id="gen-1_particle-1",
            )
        )
    finally:
        runner.close()

    assert result == [1, 2]
    assert create_client_calls == [runner.session.keyvault]
    assert (
        fake_client.upload_calls[0]["container_name"]
        == runner.session.input_container
    )
    assert fake_client.upload_calls[0]["location_in_blob"].endswith(
        "generation-1/gen-1_particle-1"
    )
    assert task_submit_calls[0]["task_name_suffix"] == "gen-1_particle-1"
    assert (
        task_submit_calls[0]["container_image_name"]
        == runner.session.remote_image_ref
    )
    assert task_submit_calls[0]["mount_pairs"] == runner.session.mount_pairs()
    assert (
        task_submit_calls[0]["save_logs_path"]
        == runner.session.logs_mount_path
    )
    assert (
        task_submit_calls[0]["logs_folder"]
        == f"{SESSION_SLUG}/{JOB_1}/gen-1_particle-1"
    )
    assert fake_client.download_calls[0]["src_path"].endswith(
        "generation-1/gen-1_particle-1/output.csv"
    )


def test_cloud_runner_upload_input_suppresses_noisy_cloudops_progress(
    monkeypatch, tmp_path, capsys
):
    uploaded_blobs: list[tuple[str, str, bytes]] = []

    class _QuietBlobClient:
        def __init__(self, *, container: str, blob: str):
            self.container = container
            self.blob = blob

        def upload_blob(self, upload_data, **kwargs):
            uploaded_blobs.append(
                (self.container, self.blob, upload_data.read())
            )

    class _QuietBlobServiceClient:
        def get_blob_client(self, *, container, blob):
            return _QuietBlobClient(container=container, blob=blob)

    fake_client = _FakeClient()
    fake_client.blob_service_client = _QuietBlobServiceClient()

    def _noisy_upload_files(**kwargs):
        print("Uploading files: 100%")
        print("Uploaded 1 files to blob storage container")

    fake_client.upload_files_impl = _noisy_upload_files

    monkeypatch.setattr(
        "example_model.cloud_runner.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.git_short_sha",
        lambda repo_root: "testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.make_session_slug",
        lambda tag: SESSION_SLUG,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.build_local_image",
        lambda **kwargs: "example-local:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.upload_local_image",
        lambda **kwargs: "fake-registry.azurecr.io/example-model:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(
            vm_size="large",
            jobs_per_session=1,
        ),
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.create_pool_with_blob_mounts",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.wait_for_pool_ready",
        lambda **kwargs: fake_client.pool,
    )

    runner = ExampleModelCloudRunner(
        Path("example_model.mrp.cloud.toml"),
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        generation_count=1,
        max_concurrent_simulations=1,
    )

    try:
        capsys.readouterr()
        runner._upload_run_input(
            fake_client,
            "gen-1_particle-1.json",
            {"seed": 123, "run_id": "gen-1_particle-1"},
            runner.session.remote_input_dir("gen-1_particle-1"),
        )
    finally:
        runner.close()

    captured = capsys.readouterr()
    assert "Uploading files:" not in captured.out
    assert "Uploading files:" not in captured.err
    assert "Uploaded 1 files to blob storage container" not in captured.out
    assert fake_client.upload_calls == []
    assert uploaded_blobs


def test_cloud_runner_async_failure_includes_batch_details(
    monkeypatch, tmp_path
):
    fake_client = _FakeClient()
    failed_task = SimpleNamespace(
        id="task-gen-1_particle-1-1",
        state="completed",
        exit_conditions=SimpleNamespace(
            pre_processing_error="mount failed",
            file_upload_error=None,
        ),
        execution_info=SimpleNamespace(
            result="failure",
            exit_code=None,
            failure_info=SimpleNamespace(
                category="servererror",
                code="TaskFailedToStart",
                message="Container failed before process launch",
                details=[
                    SimpleNamespace(name="detail", value="image pull timeout")
                ],
            ),
            container_info=SimpleNamespace(error="container create error"),
        ),
    )

    monkeypatch.setattr(
        "example_model.cloud_runner.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.git_short_sha",
        lambda repo_root: "testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.make_session_slug",
        lambda tag: SESSION_SLUG,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.build_local_image",
        lambda **kwargs: "example-local:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.upload_local_image",
        lambda **kwargs: "fake-registry.azurecr.io/example-model:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(
            vm_size="large",
            jobs_per_session=2,
        ),
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.create_pool_with_blob_mounts",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.wait_for_pool_ready",
        lambda **kwargs: fake_client.pool,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.DEFAULT_POLL_INTERVAL_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.add_batch_task_with_short_id",
        lambda **kwargs: (
            fake_client.tasks.__setitem__(
                (kwargs["job_name"], "task-gen-1_particle-1-1"),
                failed_task,
            ),
            "task-gen-1_particle-1-1",
        )[-1],
    )

    runner = ExampleModelCloudRunner(
        Path("example_model.mrp.cloud.toml"),
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        generation_count=1,
        max_concurrent_simulations=2,
    )

    try:
        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(
                runner.simulate_async(
                    {"seed": 123, "run_id": "gen-1_particle-1"},
                    output_dir=tmp_path / "output",
                    run_id="gen-1_particle-1",
                )
            )
    finally:
        runner.close()

    message = str(exc_info.value)
    assert "task-gen-1_particle-1-1" in message
    assert "result='failure'" in message
    assert "TaskFailedToStart" in message
    assert "image pull timeout" in message
    assert "container create error" in message
    assert "pre_processing_error='mount failed'" in message
    assert (
        f"logs_prefix={LOGS_CONTAINER}/{SESSION_SLUG}/{JOB_1}/gen-1_particle-1/stdout_stderr/"
    ) in message


def test_cloud_runner_failure_includes_task_log_excerpts(
    monkeypatch, tmp_path
):
    fake_client = _FakeClient()
    failed_task = SimpleNamespace(
        id="task-gen-1_particle-1-1",
        state="completed",
        execution_info=SimpleNamespace(
            result="failure",
            exit_code=1,
            failure_info=SimpleNamespace(
                category="usererror",
                code="FailureExitCode",
                message="The task exited with an exit code representing a failure",
                details=[],
            ),
        ),
    )

    def _download_file(**kwargs):
        src_path = kwargs["src_path"]
        dest_path = Path(kwargs["dest_path"])
        if src_path.endswith("/stderr.txt"):
            dest_path.write_text(
                "PermissionError: [Errno 13] Permission denied\n"
            )
            return
        if src_path.endswith("/stdout.txt"):
            dest_path.write_text("running example_model\n")
            return
        Path(kwargs["dest_path"]).write_text(
            "generation,population\n0,1\n1,2\n"
        )

    fake_client.download_file_impl = _download_file

    monkeypatch.setattr(
        "example_model.cloud_runner.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.git_short_sha",
        lambda repo_root: "testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.make_session_slug",
        lambda tag: SESSION_SLUG,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.build_local_image",
        lambda **kwargs: "example-local:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.upload_local_image",
        lambda **kwargs: "fake-registry.azurecr.io/example-model:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(
            vm_size="large",
            jobs_per_session=1,
        ),
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.create_pool_with_blob_mounts",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.wait_for_pool_ready",
        lambda **kwargs: fake_client.pool,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.DEFAULT_POLL_INTERVAL_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.add_batch_task_with_short_id",
        lambda **kwargs: (
            fake_client.tasks.__setitem__(
                (kwargs["job_name"], "task-gen-1_particle-1-1"),
                failed_task,
            ),
            "task-gen-1_particle-1-1",
        )[-1],
    )

    runner = ExampleModelCloudRunner(
        Path("example_model.mrp.cloud.toml"),
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        generation_count=1,
        max_concurrent_simulations=1,
    )

    try:
        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(
                runner.simulate_async(
                    {"seed": 123, "run_id": "gen-1_particle-1"},
                    output_dir=tmp_path / "output",
                    run_id="gen-1_particle-1",
                )
            )
    finally:
        runner.close()

    message = str(exc_info.value)
    assert (
        "stderr_excerpt='PermissionError: [Errno 13] Permission denied'"
        in message
    )
    assert "stdout_excerpt='running example_model'" in message


def test_cloud_runner_cancel_run_terminates_active_batch_task(monkeypatch):
    fake_client = _FakeClient()

    monkeypatch.setattr(
        "example_model.cloud_runner.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.git_short_sha",
        lambda repo_root: "testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.make_session_slug",
        lambda tag: SESSION_SLUG,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.build_local_image",
        lambda **kwargs: "example-local:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.upload_local_image",
        lambda **kwargs: "fake-registry.azurecr.io/example-model:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(
            vm_size="large",
            jobs_per_session=1,
        ),
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.create_pool_with_blob_mounts",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.wait_for_pool_ready",
        lambda **kwargs: fake_client.pool,
    )

    runner = ExampleModelCloudRunner(
        Path("example_model.mrp.cloud.toml"),
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        generation_count=1,
        max_concurrent_simulations=1,
    )
    runner._register_active_run(
        "gen-1_particle-1",
        JOB_1,
        output_dir=Path("/tmp/test-output-cancel"),
        input_payload={"run_id": "gen-1_particle-1"},
        overall_started=0.0,
        future=_dummy_future(),
    )
    runner._set_task_id(
        "gen-1_particle-1",
        task_id="task-gen-1_particle-1-1",
        upload_elapsed_seconds=0.0,
        submitted_at=0.0,
    )

    runner.cancel_run("gen-1_particle-1")

    assert fake_client.terminated_tasks == [
        (
            JOB_1,
            "task-gen-1_particle-1-1",
        )
    ]


# --- parse_image_tag_from_session_slug --------------------------------------


def test_parse_image_tag_from_session_slug_simple_tag():
    assert (
        parse_image_tag_from_session_slug(
            "20260412010101-testsha-ab12cd34ef56"
        )
        == "testsha"
    )


def test_parse_image_tag_from_session_slug_multi_word_tag():
    assert (
        parse_image_tag_from_session_slug(
            "20260412010101-multi-word-tag-ab12cd34ef56"
        )
        == "multi-word-tag"
    )


def test_parse_image_tag_from_session_slug_without_unique_suffix():
    # No 12-char hex suffix: every segment after the timestamp is the tag.
    assert (
        parse_image_tag_from_session_slug("20260412010101-justtag")
        == "justtag"
    )


def test_parse_image_tag_from_session_slug_rejects_invalid_inputs():
    for bad in ("", "invalid", "not-a-timestamp", "2026041201010-short-tag"):
        assert parse_image_tag_from_session_slug(bad) is None


# --- close / init-failure regression tests ----------------------------------


def _patch_runner_ok(monkeypatch, fake_client):
    """Wire up the common happy-path monkeypatches used by the tests below."""
    monkeypatch.setattr(
        "example_model.cloud_runner.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.git_short_sha",
        lambda repo_root: "testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.make_session_slug",
        lambda tag: SESSION_SLUG,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.build_local_image",
        lambda **kwargs: "example-local:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.upload_local_image",
        lambda **kwargs: "fake-registry.azurecr.io/example-model:testsha",
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(
            vm_size="large",
            jobs_per_session=1,
        ),
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.create_pool_with_blob_mounts",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "example_model.cloud_runner.wait_for_pool_ready",
        lambda **kwargs: fake_client.pool,
    )


def test_cloud_runner_close_is_idempotent_without_controller(monkeypatch):
    """Calling close() before any simulate() must not raise."""
    fake_client = _FakeClient()
    _patch_runner_ok(monkeypatch, fake_client)

    runner = ExampleModelCloudRunner(
        Path("example_model.mrp.cloud.toml"),
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        generation_count=1,
        max_concurrent_simulations=1,
    )

    # No controller has been started yet; close should be a safe no-op and
    # further close() calls must remain safe (idempotency).
    runner.close()
    runner.close()


def test_cloud_runner_close_shuts_down_controller_thread(monkeypatch):
    """close() must tear down the controller event loop cleanly."""
    import time

    fake_client = _FakeClient()
    _patch_runner_ok(monkeypatch, fake_client)

    runner = ExampleModelCloudRunner(
        Path("example_model.mrp.cloud.toml"),
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        generation_count=1,
        max_concurrent_simulations=1,
    )
    # Force controller startup without running a real simulation.
    runner._ensure_controller_started()
    assert runner._controller_thread is not None
    started_thread = runner._controller_thread

    runner.close()

    # The controller thread should wind down shortly after close().
    deadline = time.monotonic() + 5.0
    while started_thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not started_thread.is_alive()


def test_cloud_runner_init_propagates_pool_wait_failure(monkeypatch):
    """A failure while waiting for pool readiness must surface to the caller."""
    fake_client = _FakeClient()
    _patch_runner_ok(monkeypatch, fake_client)

    def _boom(**kwargs):
        raise RuntimeError("pool never became ready")

    monkeypatch.setattr(
        "example_model.cloud_runner.wait_for_pool_ready",
        _boom,
    )

    with pytest.raises(RuntimeError, match="pool never became ready"):
        ExampleModelCloudRunner(
            Path("example_model.mrp.cloud.toml"),
            repo_root=REPO_ROOT,
            dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
            generation_count=1,
            max_concurrent_simulations=1,
        )


def test_cloud_runner_init_propagates_image_upload_failure(monkeypatch):
    """Image upload failures during init must propagate cleanly."""
    fake_client = _FakeClient()
    _patch_runner_ok(monkeypatch, fake_client)

    def _boom(**kwargs):
        raise RuntimeError("ACR push rejected")

    monkeypatch.setattr(
        "example_model.cloud_runner.upload_local_image",
        _boom,
    )

    with pytest.raises(RuntimeError, match="ACR push rejected"):
        ExampleModelCloudRunner(
            Path("example_model.mrp.cloud.toml"),
            repo_root=REPO_ROOT,
            dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
            generation_count=1,
            max_concurrent_simulations=1,
        )


# --- sync executor atomic-download regression ---------------------------------


def test_execute_cloud_run_leaves_no_final_file_after_download_failure(
    monkeypatch, tmp_path
):
    """A failing blob download must not leave a truncated output.csv behind."""

    class FlakyExecutorClient:
        def __init__(self):
            self.save_logs_to_blob = None
            self.logs_folder = None
            self.batch_service_client = SimpleNamespace(
                task=SimpleNamespace(get=lambda *_: None)
            )

        def upload_files(self, **kwargs):
            pass

        def download_file(self, **kwargs):
            # Simulate a partial download: write some bytes to the
            # destination and then blow up before completion.
            Path(kwargs["dest_path"]).write_text("partial-and-corrupt")
            raise RuntimeError("network dropped mid-download")

    fake_client = FlakyExecutorClient()
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.add_batch_task_with_short_id",
        lambda **kwargs: "task-fail",
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.wait_for_task_completion",
        lambda **kwargs: {"result": "success", "exit_code": 0},
    )

    session = CloudSession(
        keyvault="cfa-predict",
        session_slug=SESSION_SLUG,
        image_tag="testsha",
        remote_image_ref="fake-registry.azurecr.io/example-model:testsha",
        pool_name=POOL_NAME,
        job_names={"1": [JOB_1, JOB_2]},
        input_container=INPUT_CONTAINER,
        output_container=OUTPUT_CONTAINER,
        logs_container=LOGS_CONTAINER,
        task_mrp_config_path="/app/example_model.mrp.task.toml",
        input_mount_path="/cloud-input",
        output_mount_path="/cloud-output",
        logs_mount_path="/cloud-logs",
        task_timeout_minutes=60,
        print_task_durations=False,
    )
    output_dir = tmp_path / "output"
    run_json = {
        "input": {"seed": 123, "run_id": "gen-1_particle-1"},
        "output": {"spec": "filesystem", "dir": str(output_dir)},
        "runtime": {
            "cloud": {
                **session.to_runtime_cloud(),
                "run_id": "gen-1_particle-1",
                "job_name": JOB_2,
            }
        },
    }

    with pytest.raises(RuntimeError, match="network dropped"):
        execute_cloud_run(run_json)

    # The partial .part file must be cleaned up and the final path must
    # never have existed with corrupted content.
    assert not (output_dir / "output.csv").exists()
    assert not (output_dir / "output.csv.part").exists()


# --- session-init rollback & sync-simulate regressions -----------------------


def _install_delete_tracking(fake_client):
    """Attach delete_job / delete_pool / delete_container tracking to fake_client."""
    deleted = {"jobs": [], "pools": [], "containers": []}

    def _delete_job(job_name):
        deleted["jobs"].append(job_name)

    def _delete_pool(pool_name):
        deleted["pools"].append(pool_name)

    def _delete_container(container_name):
        deleted["containers"].append(container_name)

    fake_client.delete_job_impl = _delete_job
    fake_client.delete_pool_impl = _delete_pool
    fake_client.blob_service_client = SimpleNamespace(
        delete_container=_delete_container,
    )
    return deleted


def test_cloud_runner_init_rolls_back_on_pool_wait_failure(monkeypatch):
    """Pool-readiness failure must best-effort delete containers and pool."""
    fake_client = _FakeClient()
    deleted = _install_delete_tracking(fake_client)
    _patch_runner_ok(monkeypatch, fake_client)

    def _boom(**kwargs):
        raise RuntimeError("pool never became ready")

    monkeypatch.setattr(
        "example_model.cloud_runner.wait_for_pool_ready",
        _boom,
    )

    with pytest.raises(RuntimeError) as excinfo:
        ExampleModelCloudRunner(
            Path("example_model.mrp.cloud.toml"),
            repo_root=REPO_ROOT,
            dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
            generation_count=1,
            max_concurrent_simulations=1,
        )

    msg = str(excinfo.value)
    assert "pool never became ready" in msg
    assert "session_slug=" in msg
    assert deleted["containers"] == [
        INPUT_CONTAINER,
        OUTPUT_CONTAINER,
        LOGS_CONTAINER,
    ]
    assert deleted["pools"] == [POOL_NAME]
    assert deleted["jobs"] == []


def test_cloud_runner_init_preserves_keyboard_interrupt_after_rollback(
    monkeypatch,
):
    """Startup rollback must not convert KeyboardInterrupt into RuntimeError."""
    fake_client = _FakeClient()
    deleted = _install_delete_tracking(fake_client)
    _patch_runner_ok(monkeypatch, fake_client)

    def _boom(**kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        "example_model.cloud_runner.wait_for_pool_ready",
        _boom,
    )

    with pytest.raises(KeyboardInterrupt):
        ExampleModelCloudRunner(
            Path("example_model.mrp.cloud.toml"),
            repo_root=REPO_ROOT,
            dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
            generation_count=1,
            max_concurrent_simulations=1,
        )

    assert deleted["containers"] == [
        INPUT_CONTAINER,
        OUTPUT_CONTAINER,
        LOGS_CONTAINER,
    ]
    assert deleted["pools"] == [POOL_NAME]
    assert deleted["jobs"] == []


def test_cloud_runner_init_rolls_back_jobs_when_job_creation_fails(
    monkeypatch,
):
    """If the second job fails to create, the first job and pool/containers roll back."""
    fake_client = _FakeClient()
    deleted = _install_delete_tracking(fake_client)
    _patch_runner_ok(monkeypatch, fake_client)

    call_count = {"n": 0}
    original_create_job = fake_client.create_job

    def _flaky_create_job(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("job quota exceeded")
        original_create_job(**kwargs)

    monkeypatch.setattr(fake_client, "create_job", _flaky_create_job)

    monkeypatch.setattr(
        "example_model.cloud_runner.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(
            vm_size="large",
            jobs_per_session=2,
        ),
    )

    with pytest.raises(RuntimeError, match="job quota exceeded"):
        ExampleModelCloudRunner(
            Path("example_model.mrp.cloud.toml"),
            repo_root=REPO_ROOT,
            dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
            generation_count=1,
            max_concurrent_simulations=1,
        )

    assert deleted["jobs"] == [JOB_1]
    assert deleted["pools"] == [POOL_NAME]
    assert deleted["containers"] == [
        INPUT_CONTAINER,
        OUTPUT_CONTAINER,
        LOGS_CONTAINER,
    ]


def test_cloud_runner_init_rollback_continues_when_individual_delete_fails(
    monkeypatch,
):
    """If a per-resource delete fails mid-rollback, the rollback continues
    with the remaining resources and the original setup error is still
    raised (with rollback failures embedded in the message so operators
    can clean up manually).
    """
    fake_client = _FakeClient()
    deleted = _install_delete_tracking(fake_client)

    # Make the first delete_job fail; remaining rollback steps must
    # still run so the pool and containers are cleaned up.
    original_delete_job = fake_client.delete_job_impl
    assert original_delete_job is not None

    def _flaky_delete_job(job_name):
        if job_name == JOB_1:
            raise RuntimeError("job delete transient failure")
        original_delete_job(job_name)

    fake_client.delete_job_impl = _flaky_delete_job

    _patch_runner_ok(monkeypatch, fake_client)

    call_count = {"n": 0}
    original_create_job = fake_client.create_job

    def _flaky_create_job(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("job quota exceeded")
        original_create_job(**kwargs)

    monkeypatch.setattr(fake_client, "create_job", _flaky_create_job)

    monkeypatch.setattr(
        "example_model.cloud_runner.load_cloud_runtime_settings",
        lambda config_path: _cloud_settings(
            vm_size="large",
            jobs_per_session=2,
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        ExampleModelCloudRunner(
            Path("example_model.mrp.cloud.toml"),
            repo_root=REPO_ROOT,
            dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
            generation_count=1,
            max_concurrent_simulations=1,
        )

    message = str(excinfo.value)
    # Original failure is preserved
    assert "job quota exceeded" in message
    # Rollback failure is surfaced in the detail block
    assert "rollback_failures" in message

    # Pool and containers must still be cleaned up even though one
    # delete_job raised mid-rollback.
    assert deleted["pools"] == [POOL_NAME]
    assert deleted["containers"] == [
        INPUT_CONTAINER,
        OUTPUT_CONTAINER,
        LOGS_CONTAINER,
    ]


def test_cloud_runner_sync_simulate_requires_run_id(monkeypatch, tmp_path):
    """Sync simulate() must reject missing run_id rather than silently pass None."""
    fake_client = _FakeClient()
    _patch_runner_ok(monkeypatch, fake_client)

    runner = ExampleModelCloudRunner(
        Path("example_model.mrp.cloud.toml"),
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        generation_count=1,
        max_concurrent_simulations=1,
    )
    try:
        with pytest.raises(ValueError, match="run_id"):
            runner.simulate(
                {"seed": 123},
                output_dir=tmp_path / "output",
            )
    finally:
        runner.close()


# --- concurrency validation & wait-path cancellation regressions ------------


def test_cloud_runner_rejects_zero_max_concurrent_simulations_before_provisioning(
    monkeypatch,
):
    """max_concurrent_simulations < 1 must fail fast without creating resources."""
    provisioned = {"client": False}

    def _should_not_create_client(*, keyvault):
        provisioned["client"] = True
        return _FakeClient()

    monkeypatch.setattr(
        "example_model.cloud_runner.create_cloud_client",
        _should_not_create_client,
    )

    with pytest.raises(ValueError, match="max_concurrent_simulations"):
        ExampleModelCloudRunner(
            Path("example_model.mrp.cloud.toml"),
            repo_root=REPO_ROOT,
            dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
            generation_count=1,
            max_concurrent_simulations=0,
        )

    assert provisioned["client"] is False


def test_execute_cloud_run_cancels_task_when_wait_times_out(
    monkeypatch, tmp_path
):
    """Wait-path failure must best-effort cancel the submitted Batch task."""

    class WaitTimeoutClient:
        def __init__(self):
            self.save_logs_to_blob = None
            self.logs_folder = None
            self.batch_service_client = SimpleNamespace(
                task=SimpleNamespace(get=lambda *_: None)
            )

        def upload_files(self, **kwargs):
            pass

    fake_client = WaitTimeoutClient()
    cancelled = {"calls": []}

    def _cancel(*, batch_client, job_name, task_id):
        cancelled["calls"].append((job_name, task_id))

    def _wait_boom(**kwargs):
        raise TimeoutError("task wait timed out")

    # The example_model wrapper injects create_cloud_client /
    # add_batch_task_with_short_id / wait_for_task_completion from
    # example_model.cloud_utils, but it does NOT override
    # cancel_batch_task, so patch that on the shared module.
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.create_cloud_client",
        lambda *, keyvault: fake_client,
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.add_batch_task_with_short_id",
        lambda **kwargs: "task-xyz",
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.wait_for_task_completion",
        _wait_boom,
    )
    monkeypatch.setattr(
        "example_model.cloud_mrp_executor.cancel_batch_task",
        _cancel,
    )

    session = CloudSession(
        keyvault="cfa-predict",
        session_slug=SESSION_SLUG,
        image_tag="testsha",
        remote_image_ref="fake-registry.azurecr.io/example-model:testsha",
        pool_name=POOL_NAME,
        job_names={"1": [JOB_1, JOB_2]},
        input_container=INPUT_CONTAINER,
        output_container=OUTPUT_CONTAINER,
        logs_container=LOGS_CONTAINER,
        task_mrp_config_path="/app/example_model.mrp.task.toml",
        input_mount_path="/cloud-input",
        output_mount_path="/cloud-output",
        logs_mount_path="/cloud-logs",
        task_timeout_minutes=60,
        print_task_durations=False,
    )
    output_dir = tmp_path / "output"
    run_json = {
        "input": {"seed": 123, "run_id": "gen-1_particle-1"},
        "output": {"spec": "filesystem", "dir": str(output_dir)},
        "runtime": {
            "cloud": {
                **session.to_runtime_cloud(),
                "run_id": "gen-1_particle-1",
                "job_name": JOB_2,
            }
        },
    }

    with pytest.raises(TimeoutError, match="task wait timed out"):
        execute_cloud_run(run_json)

    assert cancelled["calls"] == [(JOB_2, "task-xyz")]


def test_cloud_runner_rejects_zero_concurrency_before_provisioning(
    monkeypatch,
):
    """max_concurrent_simulations=0 must fail fast with no Azure provisioning."""
    fake_client = _FakeClient()
    _patch_runner_ok(monkeypatch, fake_client)

    with pytest.raises(ValueError, match="max_concurrent_simulations"):
        ExampleModelCloudRunner(
            Path("example_model.mrp.cloud.toml"),
            repo_root=REPO_ROOT,
            dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
            generation_count=1,
            max_concurrent_simulations=0,
        )

    # The constructor must have rejected the value before touching any
    # cloud APIs: no client, no blob containers, no pool, no jobs.
    assert fake_client.calls == []


def test_cloud_runner_rejects_duplicate_active_run_id(monkeypatch, tmp_path):
    """Two simulate_async calls with the same run_id must reject the second."""
    import asyncio as _asyncio

    fake_client = _FakeClient()
    _patch_runner_ok(monkeypatch, fake_client)

    runner = ExampleModelCloudRunner(
        Path("example_model.mrp.cloud.toml"),
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        generation_count=1,
        max_concurrent_simulations=2,
    )
    try:
        # Manually register the first run to simulate an in-flight state
        # without actually driving a controller loop (which would require
        # real Azure plumbing).
        runner._register_active_run(
            "gen-1_particle-1",
            JOB_1,
            output_dir=tmp_path / "out1",
            input_payload={"run_id": "gen-1_particle-1"},
            overall_started=0.0,
            future=_dummy_future(),
        )

        # Registering a second active run with the same run_id must raise
        # rather than silently overwrite and orphan the first future.
        with pytest.raises(ValueError, match="already active"):
            runner._register_active_run(
                "gen-1_particle-1",
                JOB_1,
                output_dir=tmp_path / "out2",
                input_payload={"run_id": "gen-1_particle-1"},
                overall_started=0.0,
                future=_dummy_future(),
            )

        # The end-to-end async entrypoint must surface the same rejection.
        async def _drive():
            # Invoke simulate_async directly; the duplicate must be rejected
            # at _register_active_run before any controller work starts.
            await runner.simulate_async(
                {"seed": 1},
                output_dir=tmp_path / "out3",
                run_id="gen-1_particle-1",
            )

        with pytest.raises(ValueError, match="already active"):
            _asyncio.run(_drive())
    finally:
        runner.close()


def test_calibrationtools_cloud_facade_omits_broken_helper():
    """The package facade must not advertise helpers that cannot be called."""
    import calibrationtools.cloud as facade

    # load_cloud_runtime_settings requires a model-specific `defaults=`
    # kwarg and so is not usable as a package-level entrypoint. It must
    # not be re-exported.
    assert "load_cloud_runtime_settings" not in facade.__all__
    assert not hasattr(facade, "load_cloud_runtime_settings")


def test_cloud_utils_is_compatibility_facade_for_split_modules():
    import calibrationtools.cloud.artifacts as artifacts
    import calibrationtools.cloud.batch as batch
    import calibrationtools.cloud.config as cloud_config
    import calibrationtools.cloud.formatting as formatting
    import calibrationtools.cloud.naming as naming
    import calibrationtools.cloud.session as session
    import calibrationtools.cloud.tooling as tooling
    import calibrationtools.cloud.utils as utils

    assert utils.CloudRuntimeSettings is cloud_config.CloudRuntimeSettings
    assert utils.CloudSession is session.CloudSession
    assert utils.make_session_slug is naming.make_session_slug
    assert (
        utils.format_task_failure_message
        is formatting.format_task_failure_message
    )
    assert (
        utils.download_blob_to_path_atomic
        is artifacts.download_blob_to_path_atomic
    )
    assert utils.wait_for_pool_ready is batch.wait_for_pool_ready
    assert utils.create_cloud_client is tooling.create_cloud_client


def test_cloud_backend_imports_split_modules_not_utils():
    import calibrationtools.cloud.backend as backend_module

    source = Path(backend_module.__file__).read_text()

    assert "from .utils import" not in source
    assert "from .batch import" in source
    assert "from .formatting import" in source
    assert "from .naming import" in source
    assert "from .tooling import" in source


def test_cloud_package_facade_imports_split_modules_not_utils():
    import calibrationtools.cloud as facade

    source = Path(facade.__file__).read_text()

    assert "from .utils import" not in source
    assert "from .artifacts import" in source
    assert "from .batch import" in source
    assert "from .config import" in source
    assert "from .formatting import" in source
    assert "from .naming import" in source
    assert "from .session import" in source
    assert "from .tooling import" in source


def test_example_model_cloud_utils_is_model_facing_facade():
    import example_model.cloud_utils as cloud_utils

    assert hasattr(cloud_utils, "CloudRuntimeSettings")
    assert hasattr(cloud_utils, "DEFAULT_CLOUD_RUNTIME_SETTINGS")
    assert hasattr(cloud_utils, "load_cloud_runtime_settings")
    assert hasattr(cloud_utils, "cloud_runner_backend")
    assert hasattr(cloud_utils, "cloud_executor_backend")
    assert not hasattr(cloud_utils, "CloudExecutorBackend")
    assert not hasattr(cloud_utils, "CloudRunnerBackend")
    assert not hasattr(cloud_utils, "CloudSession")
    assert not hasattr(cloud_utils, "create_cloud_client")
    assert not hasattr(cloud_utils, "make_session_slug")
    assert not hasattr(cloud_utils, "add_batch_task_with_short_id")


def test_example_model_cloud_runner_passes_backend_bundle(monkeypatch):
    import example_model.cloud_runner as cloud_runner

    captured: dict[str, Any] = {}

    def fake_init(self, config_path, **kwargs):
        captured["config_path"] = config_path
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "calibrationtools.cloud.runner.CloudMRPRunner.__init__",
        fake_init,
    )

    cloud_runner.ExampleModelCloudRunner(
        "example_model.mrp.cloud.toml",
        generation_count=2,
        max_concurrent_simulations=3,
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
    )

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    backend = kwargs["backend"]
    assert backend.create_cloud_client is cloud_runner.create_cloud_client
    assert backend.git_short_sha is cloud_runner.git_short_sha
    assert backend.make_session_slug is cloud_runner.make_session_slug
    assert backend.build_local_image is cloud_runner.build_local_image
    assert backend.upload_local_image is cloud_runner.upload_local_image
    assert (
        backend.create_pool_with_blob_mounts
        is cloud_runner.create_pool_with_blob_mounts
    )
    assert backend.wait_for_pool_ready is cloud_runner.wait_for_pool_ready
    assert (
        backend.add_batch_task_with_short_id
        is cloud_runner.add_batch_task_with_short_id
    )
    assert backend.cancel_batch_task is cloud_runner.cancel_batch_task
    assert (
        backend.format_task_failure_message
        is cloud_runner.format_task_failure_message
    )
    assert (
        backend.format_task_timing_summary
        is cloud_runner.format_task_timing_summary
    )
    assert backend.make_resource_name is cloud_runner.make_resource_name
    assert (
        backend.parse_generation_from_run_id
        is cloud_runner.parse_generation_from_run_id
    )
    assert (
        backend.suppress_cloudops_info_output
        is cloud_runner.suppress_cloudops_info_output
    )
    assert "create_cloud_client_func" not in kwargs
    assert "git_short_sha_func" not in kwargs
    assert "make_session_slug_func" not in kwargs
    assert "build_local_image_func" not in kwargs
    assert "upload_local_image_func" not in kwargs
    assert "create_pool_with_blob_mounts_func" not in kwargs
    assert "wait_for_pool_ready_func" not in kwargs
    assert "add_batch_task_with_short_id_func" not in kwargs
    assert "cancel_batch_task_func" not in kwargs
    assert "format_task_failure_message_func" not in kwargs
    assert "format_task_timing_summary_func" not in kwargs
    assert "make_resource_name_func" not in kwargs
    assert "parse_generation_from_run_id_func" not in kwargs
    assert "suppress_cloudops_info_output_func" not in kwargs


def test_example_model_cloud_runner_delegates_to_shared_factory(monkeypatch):
    import example_model.cloud_runner as cloud_runner

    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_create_cloud_mrp_runner(config_path, **kwargs):
        captured["config_path"] = config_path
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        cloud_runner,
        "_create_cloud_mrp_runner",
        fake_create_cloud_mrp_runner,
    )

    runner = cloud_runner.ExampleModelCloudRunner(
        "example_model.mrp.cloud.toml",
        generation_count=2,
        max_concurrent_simulations=3,
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        print_task_durations=True,
    )

    assert runner is sentinel
    assert captured["config_path"] == "example_model.mrp.cloud.toml"
    assert captured["kwargs"]["default_repo_root"] == REPO_ROOT
    assert captured["kwargs"]["default_dockerfile_relative_path"] == (
        Path("packages") / "example_model" / "Dockerfile"
    )
    assert captured["kwargs"]["generation_count"] == 2
    assert captured["kwargs"]["max_concurrent_simulations"] == 3
    assert captured["kwargs"]["repo_root"] == REPO_ROOT
    assert captured["kwargs"]["dockerfile"] == (
        REPO_ROOT / "packages" / "example_model" / "Dockerfile"
    )
    assert (
        captured["kwargs"]["settings_loader"]
        is cloud_runner.load_cloud_runtime_settings
    )
    assert callable(captured["kwargs"]["read_output_dir"])
    assert captured["kwargs"]["output_filename"] == "output.csv"
    assert captured["kwargs"]["print_task_durations"] is True
    assert captured["kwargs"]["auto_size_summary"] is None
    assert captured["kwargs"]["backend"].create_cloud_client is (
        cloud_runner.create_cloud_client
    )


def test_example_model_cloud_runner_wraps_settings_for_auto_size(monkeypatch):
    import example_model.cloud_runner as cloud_runner

    captured: dict[str, Any] = {}
    sentinel = object()
    summary = SimpleNamespace(task_slots_per_node=3)

    def fake_create_cloud_mrp_runner(config_path, **kwargs):
        captured["config_path"] = config_path
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        cloud_runner,
        "_create_cloud_mrp_runner",
        fake_create_cloud_mrp_runner,
    )

    runner = cloud_runner.ExampleModelCloudRunner(
        "example_model.mrp.cloud.toml",
        generation_count=2,
        max_concurrent_simulations=7,
        repo_root=REPO_ROOT,
        dockerfile=REPO_ROOT / "packages" / "example_model" / "Dockerfile",
        task_slots_per_node_override=3,
        auto_size_summary=summary,
    )

    assert runner is sentinel
    assert captured["kwargs"]["auto_size_summary"] is summary
    settings = captured["kwargs"]["settings_loader"](
        "example_model.mrp.cloud.toml"
    )
    assert settings.task_slots_per_node == 3


def test_example_model_cloud_runner_uses_default_build_context(monkeypatch):
    import example_model.cloud_runner as cloud_runner

    captured: dict[str, Any] = {}

    def fake_init(self, config_path, **kwargs):
        captured["config_path"] = config_path
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "calibrationtools.cloud.runner.CloudMRPRunner.__init__",
        fake_init,
    )

    cloud_runner.ExampleModelCloudRunner(
        "example_model.mrp.cloud.toml",
        generation_count=2,
        max_concurrent_simulations=3,
    )

    kwargs = captured["kwargs"]
    assert kwargs["repo_root"] == REPO_ROOT
    assert kwargs["dockerfile"] == (
        REPO_ROOT / "packages" / "example_model" / "Dockerfile"
    )


def test_cloud_runtime_settings_preserve_dataclass_signatures():
    import example_model.cloud_utils as cloud_utils

    import calibrationtools.cloud.config as cloud_config

    base_params = inspect.signature(
        cloud_config.CloudRuntimeSettings
    ).parameters
    example_params = inspect.signature(
        cloud_utils.CloudRuntimeSettings
    ).parameters

    assert "keyvault" in base_params
    assert "jobs_per_session" in base_params
    assert "print_task_durations" in base_params
    assert "args" not in base_params
    assert "kwargs" not in base_params

    assert "keyvault" in example_params
    assert "jobs_per_session" in example_params
    assert "print_task_durations" in example_params
    assert "args" not in example_params
    assert "kwargs" not in example_params
    assert (
        cloud_utils.CloudRuntimeSettings is cloud_config.CloudRuntimeSettings
    )


def test_example_model_cloud_defaults_object_carries_model_defaults():
    import example_model.cloud_utils as cloud_utils

    defaults = cloud_utils.DEFAULT_CLOUD_RUNTIME_SETTINGS

    assert defaults.keyvault == "cfa-predict"
    assert defaults.local_image == "cfa-calibration-tools-example-model-python"
    assert defaults.repository == "cfa-calibration-tools-example-model"
    assert defaults.task_mrp_config_path == "/app/example_model.mrp.task.toml"
    assert defaults.pool_prefix == "example-model-cloud"
    assert defaults.job_prefix == "example-model-cloud"
    assert defaults.input_container_prefix == "example-model-cloud-input"
    assert defaults.output_container_prefix == "example-model-cloud-output"
    assert defaults.logs_container_prefix == "example-model-cloud-logs"
    assert defaults.task_slots_per_node == 50
    assert defaults.pool_max_nodes == 5
    assert defaults.pool_auto_scale_evaluation_interval_minutes == 5
    assert defaults.dispatch_buffer == 1000


def test_cloud_runtime_settings_loads_autoscale_interval_override(tmp_path):
    from calibrationtools.cloud.config import (
        CloudRuntimeSettings,
        load_cloud_runtime_settings,
    )

    config_path = tmp_path / "cloud.toml"
    config_path.write_text(
        """
[runtime.cloud]
pool_auto_scale_evaluation_interval_minutes = 10
pool_max_nodes = 12
""".strip()
    )
    defaults = CloudRuntimeSettings(
        keyvault="keyvault",
        local_image="local",
        repository="repo",
        task_mrp_config_path="/app/task.toml",
        pool_prefix="pool",
        job_prefix="job",
        input_container_prefix="input",
        output_container_prefix="output",
        logs_container_prefix="logs",
    )

    settings = load_cloud_runtime_settings(config_path, defaults=defaults)

    assert settings.pool_auto_scale_evaluation_interval_minutes == 10
    assert settings.pool_max_nodes == 12


def test_cloud_runtime_settings_rejects_invalid_pool_max_nodes():
    from calibrationtools.cloud.config import CloudRuntimeSettings

    with pytest.raises(
        ValueError,
        match="pool_max_nodes must be at least 1",
    ):
        CloudRuntimeSettings(
            keyvault="keyvault",
            local_image="local",
            repository="repo",
            task_mrp_config_path="/app/task.toml",
            pool_prefix="pool",
            job_prefix="job",
            input_container_prefix="input",
            output_container_prefix="output",
            logs_container_prefix="logs",
            pool_max_nodes=0,
        )


def test_cloud_runtime_settings_rejects_too_short_autoscale_interval():
    from calibrationtools.cloud.config import CloudRuntimeSettings

    with pytest.raises(
        ValueError,
        match="pool_auto_scale_evaluation_interval_minutes must be at least 5",
    ):
        CloudRuntimeSettings(
            keyvault="keyvault",
            local_image="local",
            repository="repo",
            task_mrp_config_path="/app/task.toml",
            pool_prefix="pool",
            job_prefix="job",
            input_container_prefix="input",
            output_container_prefix="output",
            logs_container_prefix="logs",
            pool_auto_scale_evaluation_interval_minutes=4,
        )


def test_example_model_executor_passes_backend_bundle(monkeypatch):
    import example_model.cloud_mrp_executor as cloud_mrp_executor

    captured: dict[str, Any] = {}

    def fake_execute(run_json, **kwargs):
        captured["run_json"] = run_json
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        cloud_mrp_executor,
        "_execute_cloud_run",
        fake_execute,
    )

    run_json = {"runtime": {"cloud": {}}}
    cloud_mrp_executor.execute_cloud_run(run_json)

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    backend = kwargs["backend"]
    assert (
        backend.create_cloud_client is cloud_mrp_executor.create_cloud_client
    )
    assert (
        backend.add_batch_task_with_short_id
        is cloud_mrp_executor.add_batch_task_with_short_id
    )
    assert (
        backend.wait_for_task_completion
        is cloud_mrp_executor.wait_for_task_completion
    )
    assert backend.cancel_batch_task is cloud_mrp_executor.cancel_batch_task
    assert (
        backend.format_task_failure_message
        is cloud_mrp_executor.format_task_failure_message
    )
    assert (
        backend.format_task_timing_summary
        is cloud_mrp_executor.format_task_timing_summary
    )
    assert (
        backend.suppress_cloudops_info_output
        is cloud_mrp_executor.suppress_cloudops_info_output
    )
    assert kwargs["output_filename"] == "output.csv"
    assert "create_cloud_client_func" not in kwargs
    assert "add_batch_task_with_short_id_func" not in kwargs
    assert "wait_for_task_completion_func" not in kwargs
    assert "cancel_batch_task_func" not in kwargs
    assert "format_task_failure_message_func" not in kwargs
    assert "format_task_timing_summary_func" not in kwargs
    assert "suppress_cloudops_info_output_func" not in kwargs
