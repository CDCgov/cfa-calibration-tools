from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from calibrationtools.calibration_app import (
    CalibrationAppSpec,
    CSVOutputContract,
    parse_calibration_args,
    resolve_artifacts_dir,
    resolve_max_concurrent_simulations,
    resolve_model_runner,
    run_calibration_app,
)
from calibrationtools.cloud.auto_size import CloudSizing


def _spec(**overrides: Any) -> CalibrationAppSpec:
    """Return a minimal calibration app spec for shared CLI tests."""
    defaults: dict[str, Any] = {
        "default_inputs": {"seed": 1},
        "priors": {"priors": {}},
        "tolerance_values": [1.0, 0.5],
        "target_data": 5,
        "outputs_to_distance": lambda outputs, target: 0.0,
        "direct_runner_factory": lambda: "direct-runner",
        "output_contract": CSVOutputContract(
            filename="output.csv",
            value_column="population",
            value_parser=int,
            header_fields=("generation", "population"),
        ),
        "default_mrp_config_path": Path("model.mrp.toml"),
        "default_docker_mrp_config_path": Path("model.mrp.docker.toml"),
        "default_cloud_config_path": Path("model.cloud_config.toml"),
    }
    defaults.update(overrides)
    return CalibrationAppSpec(**defaults)


def test_parse_calibration_args_accepts_all_execution_modes():
    """Parse direct, Docker, explicit MRP, and cloud modes in shared code."""
    direct_args = parse_calibration_args([], _spec())
    docker_args = parse_calibration_args(["--docker"], _spec())
    mrp_args = parse_calibration_args(
        ["--mrp-config", "custom.toml"],
        _spec(),
    )
    cloud_args = parse_calibration_args(
        ["--cloud", "--cloud-config", "cloud.toml", "--auto-size"],
        _spec(),
    )

    assert direct_args.docker is False
    assert docker_args.docker is True
    assert mrp_args.mrp_config == Path("custom.toml")
    assert cloud_args.cloud is True
    assert cloud_args.cloud_config == Path("cloud.toml")
    assert cloud_args.auto_size is True


def test_shared_artifact_policy_rejects_cloud_without_artifacts():
    """Keep cloud artifact validation out of model packages."""
    args = parse_calibration_args(["--cloud", "--no-artifacts"], _spec())

    with pytest.raises(ValueError, match="--cloud requires artifacts"):
        resolve_artifacts_dir(args, _spec())


def test_shared_concurrency_defaults_follow_execution_mode():
    """Resolve local and cloud concurrency defaults in shared code."""
    spec = _spec(local_default_concurrency=3, cloud_default_concurrency=11)

    assert (
        resolve_max_concurrent_simulations(
            parse_calibration_args([], spec),
            spec,
        )
        == 3
    )
    assert (
        resolve_max_concurrent_simulations(
            parse_calibration_args(["--cloud"], spec),
            spec,
        )
        == 11
    )


def test_shared_runner_selection_constructs_expected_runners(monkeypatch):
    """Select direct, MRP, Docker MRP, and cloud runners from one helper."""
    captured: dict[str, Any] = {}
    spec = _spec()

    class FakeMRPRunner:
        def __init__(self, config_path, **kwargs):
            captured.setdefault("mrp", []).append((config_path, kwargs))
            self.config_path = Path(config_path)

    def fake_cloud_runner(config_path, **kwargs):
        captured["cloud"] = (config_path, kwargs)
        return SimpleNamespace(kind="cloud")

    monkeypatch.setattr(
        "calibrationtools.calibration_app.CSVOutputMRPRunner",
        FakeMRPRunner,
    )
    monkeypatch.setattr(
        "calibrationtools.calibration_app.create_csv_cloud_mrp_runner_from_config",
        fake_cloud_runner,
    )

    docker_runner = cast(
        Any,
        resolve_model_runner(
            parse_calibration_args(["--docker"], spec),
            spec,
            cloud_sizing=CloudSizing(max_concurrent_simulations=3),
        ),
    )
    mrp_runner = cast(
        Any,
        resolve_model_runner(
            parse_calibration_args(["--mrp-config", "custom.toml"], spec),
            spec,
            cloud_sizing=CloudSizing(max_concurrent_simulations=3),
        ),
    )
    cloud_runner = cast(
        Any,
        resolve_model_runner(
            parse_calibration_args(["--cloud"], spec),
            spec,
            cloud_sizing=CloudSizing(
                max_concurrent_simulations=9,
                task_slots_per_node_override=2,
            ),
        ),
    )

    assert (
        resolve_model_runner(
            parse_calibration_args([], spec),
            spec,
            cloud_sizing=CloudSizing(max_concurrent_simulations=3),
        )
        == "direct-runner"
    )
    assert docker_runner.config_path == Path("model.mrp.docker.toml")
    assert mrp_runner.config_path == Path("custom.toml")
    assert cloud_runner.kind == "cloud"

    assert captured["cloud"][0] == Path("model.cloud_config.toml")
    assert captured["cloud"][1]["generation_count"] == 2
    assert captured["cloud"][1]["max_concurrent_simulations"] == 9
    assert captured["cloud"][1]["task_slots_per_node_override"] == 2


def test_run_calibration_app_invokes_sampler_and_closes_runner(monkeypatch):
    """Run sampler lifecycle and runner cleanup through shared app code."""
    events: list[str] = []

    class FakeRunner:
        def close(self) -> None:
            events.append("closed")

    class FakeSampler:
        def __init__(self, **kwargs):
            events.append("sampler")
            self.kwargs = kwargs

        def run(self):
            events.append("run")
            return "results"

    spec = _spec(direct_runner_factory=FakeRunner)

    monkeypatch.setattr(
        "calibrationtools.calibration_app.resolve_cloud_sizing_from_config",
        lambda **kwargs: CloudSizing(max_concurrent_simulations=4),
    )
    monkeypatch.setattr(
        "calibrationtools.calibration_app.ABCSampler",
        FakeSampler,
    )

    result = run_calibration_app([], spec)

    assert result == "results"
    assert events == ["sampler", "run", "closed"]
