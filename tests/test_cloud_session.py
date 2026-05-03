from __future__ import annotations

import pytest

from calibrationtools.cloud.session import CloudSession


def make_session() -> CloudSession:
    return CloudSession(
        keyvault="kv",
        session_id="session",
        image_tag="abc123",
        remote_image_ref="acr.azurecr.io/model:abc123",
        pool_name="pool",
        job_names={"0": ["job-a", "job-b"]},
        input_container="input-container",
        output_container="output-container",
        logs_container="logs-container",
        task_mrp_config_path="/app/task.toml",
        input_mount_path="/mnt/input",
        output_mount_path="/mnt/output",
        logs_mount_path="/mnt/logs",
        task_timeout_minutes=7,
        print_task_durations=True,
    )


def test_cloud_session_runtime_metadata_uses_session_id_only():
    runtime_cloud = make_session().to_runtime_cloud()

    assert runtime_cloud["session_id"] == "session"
    assert "session_slug" not in runtime_cloud
    assert CloudSession.from_runtime_cloud(runtime_cloud) == make_session()


def test_cloud_session_requires_session_id_runtime_metadata():
    runtime_cloud = make_session().to_runtime_cloud()
    runtime_cloud["session_slug"] = runtime_cloud.pop("session_id")

    with pytest.raises(KeyError, match="session_id"):
        CloudSession.from_runtime_cloud(runtime_cloud)


def test_cloud_session_paths_and_job_selection_use_session_id():
    session = make_session()

    assert session.job_name_for_run("gen_0_particle_1_attempt_0") == "job-b"
    assert (
        session.logs_folder_for_job("job-b", "gen_0_particle_1_attempt_0")
        == "session/job-b/gen_0_particle_1_attempt_0"
    )
    assert (
        session.remote_input_dir("gen_0_particle_1_attempt_0")
        == "input/session/generation-0/gen_0_particle_1_attempt_0"
    )
    assert (
        session.remote_output_dir("gen_0_particle_1_attempt_0")
        == "output/session/generation-0/gen_0_particle_1_attempt_0"
    )
