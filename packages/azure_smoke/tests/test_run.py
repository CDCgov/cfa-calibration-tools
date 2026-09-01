"""Tests for the externally runnable Azure smoke-study command."""

from __future__ import annotations

from typing import Any

from example_model_azure_smoke import run


def test_study_mode_builds_two_fresh_named_samplers(
    monkeypatch, tmp_path, capsys
) -> None:
    """Keep gate 2 runnable without requiring Azure in local tests."""

    captured: dict[str, Any] = {}

    class FakeStudy:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def run(self) -> dict[str, str]:
            scenarios = captured["scenarios"]
            samplers = [
                captured["sampler_factory"](scenario) for scenario in scenarios
            ]
            assert samplers[0] is not samplers[1]
            return {scenario.name: scenario.name for scenario in scenarios}

    monkeypatch.setattr(run, "CalibrationStudy", FakeStudy)

    status = run.main(["--study", "--detail-log-dir", str(tmp_path)])

    assert status == 0
    assert [scenario.name for scenario in captured["scenarios"]] == [
        "baseline",
        "higher-target",
    ]
    assert captured["max_concurrent_scenarios"] == 2
    assert captured["detail_log_dir"] == str(tmp_path)
    assert captured["cloud_executor"].max_autoscale_nodes == 2
    assert "baseline, higher-target" in capsys.readouterr().out
