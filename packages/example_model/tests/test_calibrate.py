from argparse import Namespace
from pathlib import Path

import pytest
from example_model.calibrate import (
    DEFAULT_CLOUD_MAX_CONCURRENT_SIMULATIONS,
    DEFAULT_MAX_CONCURRENT_SIMULATIONS,
    main,
    parse_args,
    print_cloud_auto_size_summary,
    resolve_cloud_sizing,
    resolve_max_concurrent_simulations,
    resolve_model_runner,
    run_calibration,
)
from example_model.mrp_runner import DEFAULT_DOCKER_MRP_CONFIG_PATH

from example_model import (
    DEFAULT_CLOUD_MRP_CONFIG_PATH,
    Binom_BP_Model,
    ExampleModelMRPRunner,
)


def test_parse_args_accepts_auto_size_for_cloud(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["calibrate", "--cloud", "--auto-size"],
    )

    args = parse_args()

    assert args.cloud is True
    assert args.auto_size is True


def test_resolve_model_runner_defaults_to_direct_simulation():
    args = Namespace(
        mrp_config=None,
        docker=False,
        cloud=False,
        max_concurrent_simulations=DEFAULT_MAX_CONCURRENT_SIMULATIONS,
        auto_size=False,
        print_task_durations=False,
    )

    assert resolve_model_runner(args) is Binom_BP_Model


def test_resolve_model_runner_uses_docker_flag():
    args = Namespace(
        mrp_config=None,
        docker=True,
        cloud=False,
        max_concurrent_simulations=DEFAULT_MAX_CONCURRENT_SIMULATIONS,
        auto_size=False,
        print_task_durations=False,
    )

    runner = resolve_model_runner(args)

    assert isinstance(runner, ExampleModelMRPRunner)
    assert runner.config_path == DEFAULT_DOCKER_MRP_CONFIG_PATH


def test_resolve_model_runner_uses_explicit_mrp_config():
    config_path = Path("/tmp/custom.mrp.toml")
    args = Namespace(
        mrp_config=config_path,
        docker=False,
        cloud=False,
        max_concurrent_simulations=DEFAULT_MAX_CONCURRENT_SIMULATIONS,
        auto_size=False,
        print_task_durations=False,
    )

    runner = resolve_model_runner(args)

    assert isinstance(runner, ExampleModelMRPRunner)
    assert runner.config_path == config_path


def test_resolve_model_runner_uses_cloud_flag(monkeypatch, tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n")

    class FakeCloudRunner:
        def __init__(
            self,
            config_path,
            *,
            generation_count,
            max_concurrent_simulations,
            repo_root,
            dockerfile,
            print_task_durations,
            task_slots_per_node_override=None,
            auto_size_summary=None,
        ):
            self.config_path = config_path
            self.generation_count = generation_count
            self.max_concurrent_simulations = max_concurrent_simulations
            self.repo_root = repo_root
            self.dockerfile = dockerfile
            self.print_task_durations = print_task_durations
            self.task_slots_per_node_override = task_slots_per_node_override
            self.auto_size_summary = auto_size_summary

    monkeypatch.setattr(
        "example_model.calibrate.ExampleModelCloudRunner", FakeCloudRunner
    )

    args = Namespace(
        mrp_config=None,
        docker=False,
        cloud=True,
        max_concurrent_simulations=7,
        auto_size=False,
        print_task_durations=True,
        repo_root=str(tmp_path),
        dockerfile=str(dockerfile),
    )

    runner = resolve_model_runner(args)

    assert isinstance(runner, FakeCloudRunner)
    assert runner.config_path == DEFAULT_CLOUD_MRP_CONFIG_PATH
    assert runner.generation_count == 2
    assert runner.max_concurrent_simulations == 7
    assert runner.task_slots_per_node_override is None
    assert runner.auto_size_summary is None
    assert Path(runner.repo_root) == tmp_path
    assert Path(runner.dockerfile) == dockerfile
    assert runner.print_task_durations is True


def test_resolve_max_concurrent_simulations_uses_cloud_default():
    args = Namespace(
        mrp_config=None,
        docker=False,
        cloud=True,
        max_concurrent_simulations=None,
        auto_size=False,
        print_task_durations=False,
    )

    assert resolve_max_concurrent_simulations(args) == (
        DEFAULT_CLOUD_MAX_CONCURRENT_SIMULATIONS
    )


def test_resolve_max_concurrent_simulations_uses_non_cloud_default():
    args = Namespace(
        mrp_config=None,
        docker=False,
        cloud=False,
        max_concurrent_simulations=None,
        auto_size=False,
        print_task_durations=False,
    )

    assert (
        resolve_max_concurrent_simulations(args)
        == DEFAULT_MAX_CONCURRENT_SIMULATIONS
    )


def test_resolve_cloud_sizing_rejects_auto_size_without_cloud():
    args = Namespace(
        cloud=False,
        auto_size=True,
        max_concurrent_simulations=None,
    )

    with pytest.raises(ValueError, match="--auto-size requires --cloud"):
        resolve_cloud_sizing(args)


def test_resolve_cloud_sizing_keeps_defaults_without_auto_size():
    args = Namespace(
        cloud=True,
        auto_size=False,
        max_concurrent_simulations=None,
    )

    sizing = resolve_cloud_sizing(args)

    assert sizing.max_concurrent_simulations == (
        DEFAULT_CLOUD_MAX_CONCURRENT_SIMULATIONS
    )
    assert sizing.task_slots_per_node_override is None
    assert sizing.summary is None


def test_resolve_cloud_sizing_auto_size_sets_default_concurrency(monkeypatch):
    args = Namespace(
        cloud=True,
        auto_size=True,
        max_concurrent_simulations=None,
    )

    monkeypatch.setattr(
        "example_model.calibrate.load_cloud_runtime_settings",
        lambda config_path: Namespace(vm_size="large", pool_max_nodes=5),
    )

    captured = {}

    def fake_probe(probe_module, base_inputs):
        captured["probe_module"] = probe_module
        captured["base_inputs"] = base_inputs
        return 10 * 1024**3

    monkeypatch.setattr(
        "example_model.calibrate.run_local_memory_probe",
        fake_probe,
    )

    sizing = resolve_cloud_sizing(args)

    assert sizing.max_concurrent_simulations == 25
    assert sizing.task_slots_per_node_override == 5
    assert sizing.summary is not None
    assert sizing.summary.measured_task_peak_rss_bytes == 10 * 1024**3
    assert sizing.summary.memory_task_slots_per_node == 5
    assert sizing.summary.max_task_slots_per_node == 64
    assert sizing.summary.task_slots_per_node == 5
    assert captured["probe_module"] == "example_model.cloud_auto_size"


def test_resolve_cloud_sizing_auto_size_caps_default_concurrency(monkeypatch):
    args = Namespace(
        cloud=True,
        auto_size=True,
        max_concurrent_simulations=None,
    )

    monkeypatch.setattr(
        "example_model.calibrate.load_cloud_runtime_settings",
        lambda config_path: Namespace(vm_size="large", pool_max_nodes=5),
    )
    monkeypatch.setattr(
        "example_model.calibrate.run_local_memory_probe",
        lambda probe_module, base_inputs: 100 * 1024**2,
    )

    sizing = resolve_cloud_sizing(args)

    assert sizing.max_concurrent_simulations == 320
    assert sizing.task_slots_per_node_override == 64
    assert sizing.summary is not None
    assert sizing.summary.memory_task_slots_per_node == 557
    assert sizing.summary.max_task_slots_per_node == 64
    assert sizing.summary.task_slots_per_node == 64


def test_print_cloud_auto_size_summary_includes_ram_sizing(capsys):
    print_cloud_auto_size_summary(
        Namespace(
            max_concurrent_simulations=320,
            summary=Namespace(
                vm_size="large",
                vm_memory_bytes=64 * 1024**3,
                measured_task_peak_rss_bytes=100 * 1024**2,
                reserve=0.15,
                memory_task_slots_per_node=557,
                max_task_slots_per_node=64,
                task_slots_per_node=64,
            ),
        )
    )

    captured = capsys.readouterr()

    assert "[cloud-run] auto-size simulation RAM" in captured.err
    assert "measured_peak_rss=104857600 bytes (100.0 MiB)" in captured.err
    assert "vm_ram=68719476736 bytes (64.0 GiB)" in captured.err
    assert "batch_slot_limit=64" in captured.err
    assert "task_slots_per_node=64" in captured.err
    assert "capped_from_ram_slots=557" in captured.err
    assert "max_concurrent_simulations_per_node=64" in captured.err
    assert "max_concurrent_simulations_total=320" in captured.err


def test_resolve_cloud_sizing_auto_size_preserves_explicit_concurrency(
    monkeypatch,
):
    args = Namespace(
        cloud=True,
        auto_size=True,
        max_concurrent_simulations=17,
    )

    monkeypatch.setattr(
        "example_model.calibrate.load_cloud_runtime_settings",
        lambda config_path: Namespace(vm_size="large", pool_max_nodes=5),
    )
    monkeypatch.setattr(
        "example_model.calibrate.run_local_memory_probe",
        lambda probe_module, base_inputs: 10 * 1024**3,
    )

    sizing = resolve_cloud_sizing(args)

    assert sizing.max_concurrent_simulations == 17
    assert sizing.task_slots_per_node_override == 5


def test_resolve_model_runner_passes_auto_size_override(monkeypatch, tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n")
    captured = {}

    class FakeCloudRunner:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "example_model.calibrate.ExampleModelCloudRunner",
        FakeCloudRunner,
    )

    args = Namespace(
        mrp_config=None,
        docker=False,
        cloud=True,
        max_concurrent_simulations=7,
        auto_size=True,
        print_task_durations=True,
        repo_root=str(tmp_path),
        dockerfile=str(dockerfile),
    )

    runner = resolve_model_runner(
        args,
        cloud_sizing=Namespace(
            max_concurrent_simulations=7,
            task_slots_per_node_override=3,
            summary=Namespace(task_slots_per_node=3),
        ),
    )

    assert isinstance(runner, FakeCloudRunner)
    assert captured["kwargs"]["max_concurrent_simulations"] == 7
    assert captured["kwargs"]["task_slots_per_node_override"] == 3
    assert captured["kwargs"]["auto_size_summary"].task_slots_per_node == 3


def test_main_auto_size_failure_happens_before_cloud_runner(monkeypatch):
    constructed = False

    def fake_cloud_runner(*args, **kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("cloud runner should not be constructed")

    monkeypatch.setattr(
        "sys.argv",
        ["calibrate", "--cloud", "--auto-size"],
    )
    monkeypatch.setattr(
        "example_model.calibrate.load_cloud_runtime_settings",
        lambda config_path: Namespace(vm_size="large", pool_max_nodes=5),
    )

    def fail_probe(probe_module, base_inputs):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(
        "example_model.calibrate.run_local_memory_probe",
        fail_probe,
    )
    monkeypatch.setattr(
        "example_model.calibrate.ExampleModelCloudRunner",
        fake_cloud_runner,
    )

    with pytest.raises(RuntimeError, match="probe failed"):
        main()

    assert constructed is False


def test_main_auto_size_passes_resolved_concurrency_to_sampler(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "sys.argv",
        ["calibrate", "--cloud", "--auto-size"],
    )
    monkeypatch.setattr(
        "example_model.calibrate.load_cloud_runtime_settings",
        lambda config_path: Namespace(vm_size="large", pool_max_nodes=5),
    )
    monkeypatch.setattr(
        "example_model.calibrate.run_local_memory_probe",
        lambda probe_module, base_inputs: 100 * 1024**2,
    )
    monkeypatch.setattr(
        "example_model.calibrate.resolve_model_runner",
        lambda args, *, cloud_sizing: "runner",
    )

    def fake_run_calibration(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "example_model.calibrate.run_calibration",
        fake_run_calibration,
    )

    main()

    assert captured["model_runner"] == "runner"
    assert captured["max_concurrent_simulations"] == 320


def test_run_calibration_passes_task_progress_to_sampler(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResults:
        def __repr__(self) -> str:
            return "FakeResults()"

        class Posterior:
            particles = [{"p": 0.4, "n": 2.0}, {"p": 0.6, "n": 3.0}]

        posterior_particles = Posterior()

    class FakeSampler:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return FakeResults()

    monkeypatch.setattr("example_model.calibrate.ABCSampler", FakeSampler)

    run_calibration(
        model_runner=Binom_BP_Model,
        max_concurrent_simulations=7,
        print_task_progress=True,
    )

    assert captured["max_concurrent_simulations"] == 7
    assert captured["print_generation_progress"] is True
