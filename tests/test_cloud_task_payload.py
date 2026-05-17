from __future__ import annotations

from pathlib import Path

import pytest

from calibrationtools.cloud.config import (
    CloudPayloadTransformOp,
    CloudSharedAssetSettings,
    CloudTaskPayloadSettings,
    CloudTaskPayloadTransform,
)
from calibrationtools.cloud.json_pointer import (
    JSONPointerMissingError,
    JSONPointerSyntaxError,
    read_json_pointer,
)
from calibrationtools.cloud.task_payload import (
    CloudTaskContext,
    ResolvedSharedAsset,
    apply_task_payload_transforms,
    resolve_shared_assets,
    validate_task_payload_templates,
)


def _context(*, asset_path: str = "/cloud-input/assets/session/ref/data.json"):
    return CloudTaskContext(
        run_id="run-1",
        session_id="session",
        job_name="job",
        input_mount_path="/cloud-input",
        output_mount_path="/cloud-output",
        logs_mount_path="/cloud-logs",
        task_output_dir="/cloud-output/output/run-1",
        shared_assets=(
            ResolvedSharedAsset(
                name="reference",
                source_path=Path("/tmp/reference.json"),
                is_dir=False,
                remote_blob_dir="assets/session/reference",
                remote_mount_path=asset_path,
                remote_path_var="REFERENCE_PATH",
            ),
        ),
    )


def test_json_pointer_treats_literal_dots_as_key_characters():
    payload = {"epimodel.GlobalParams": {"x": 1}}

    assert read_json_pointer(payload, "/epimodel.GlobalParams/x") == 1


def test_json_pointer_supports_rfc6901_escapes():
    payload = {"a/b": {"c~d": 2}}

    assert read_json_pointer(payload, "/a~1b/c~0d") == 2


def test_json_pointer_fails_clearly_on_missing_key_and_invalid_syntax():
    with pytest.raises(JSONPointerMissingError, match="missing object key"):
        read_json_pointer({"a": {}}, "/a/b")

    with pytest.raises(JSONPointerSyntaxError, match="start with '/'"):
        read_json_pointer({"a": 1}, "a")


def test_resolve_shared_assets_supports_static_and_pointer_sources(
    tmp_path: Path,
):
    static = tmp_path / "static.json"
    pointer = tmp_path / "pointer.json"
    static.write_text("{}", encoding="utf-8")
    pointer.write_text("{}", encoding="utf-8")

    assets = resolve_shared_assets(
        (
            CloudSharedAssetSettings(
                name="static",
                source_path=Path("static.json"),
                source_json_pointer=None,
                blob_dir="assets",
            ),
            CloudSharedAssetSettings(
                name="pointer",
                source_path=None,
                source_json_pointer="/epimodel.GlobalParams/path",
                blob_dir="assets",
            ),
        ),
        base_payload={"epimodel.GlobalParams": {"path": "pointer.json"}},
        config_dir=tmp_path,
    )

    assert assets[0].source_path == static
    assert assets[1].source_path == pointer


def test_resolve_shared_assets_fails_required_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="reference"):
        resolve_shared_assets(
            (
                CloudSharedAssetSettings(
                    name="reference",
                    source_path=Path("missing.json"),
                    source_json_pointer=None,
                    blob_dir="assets",
                ),
            ),
            base_payload=None,
            config_dir=tmp_path,
        )


def test_resolve_shared_assets_requires_base_inputs_for_pointer(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="base inputs"):
        resolve_shared_assets(
            (
                CloudSharedAssetSettings(
                    name="reference",
                    source_path=None,
                    source_json_pointer="/reference_path",
                    blob_dir="assets",
                ),
            ),
            base_payload=None,
            config_dir=tmp_path,
        )


def test_apply_task_payload_transforms_sets_literal_dot_path():
    settings = CloudTaskPayloadSettings(
        transforms=(
            CloudTaskPayloadTransform(
                name="reference",
                op=CloudPayloadTransformOp.SET,
                path="/epimodel.GlobalParams/reference_path",
                value="{REFERENCE_PATH}",
            ),
        )
    )

    transformed = apply_task_payload_transforms(
        {"epimodel.GlobalParams": {"reference_path": ""}},
        settings,
        _context(),
    )

    assert transformed["epimodel.GlobalParams"]["reference_path"] == (
        "/cloud-input/assets/session/ref/data.json"
    )


def test_apply_task_payload_transforms_missing_modes():
    error_settings = CloudTaskPayloadSettings(
        transforms=(
            CloudTaskPayloadTransform(
                name=None,
                op=CloudPayloadTransformOp.SET,
                path="/missing/value",
                value=1,
            ),
        )
    )
    with pytest.raises(JSONPointerMissingError):
        apply_task_payload_transforms({}, error_settings, _context())

    skip_settings = CloudTaskPayloadSettings(
        transforms=(
            CloudTaskPayloadTransform(
                name=None,
                op=CloudPayloadTransformOp.SET,
                path="/missing/value",
                value=1,
                on_missing="skip",
            ),
        )
    )
    assert apply_task_payload_transforms({}, skip_settings, _context()) == {}

    create_settings = CloudTaskPayloadSettings(
        transforms=(
            CloudTaskPayloadTransform(
                name=None,
                op=CloudPayloadTransformOp.SET,
                path="/missing/value",
                value="{task_output_dir}",
                on_missing="create",
            ),
        )
    )
    assert apply_task_payload_transforms({}, create_settings, _context()) == {
        "missing": {"value": "/cloud-output/output/run-1"}
    }


def test_validate_task_payload_templates_rejects_unknown_variable():
    settings = CloudTaskPayloadSettings(
        transforms=(
            CloudTaskPayloadTransform(
                name="bad",
                op=CloudPayloadTransformOp.SET,
                path="/x",
                value="{does_not_exist}",
                on_missing="create",
            ),
        )
    )

    with pytest.raises(ValueError, match="unknown template variable"):
        validate_task_payload_templates(settings, shared_assets=())
