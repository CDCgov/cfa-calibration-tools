from argparse import Namespace
from pathlib import Path

from example_model.calibrate import (
    DEFAULT_CLOUD_MAX_CONCURRENT_SIMULATIONS,
    DEFAULT_MAX_CONCURRENT_SIMULATIONS,
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


def test_resolve_model_runner_defaults_to_direct_simulation():
    args = Namespace(
        mrp_config=None,
        docker=False,
        cloud=False,
        max_concurrent_simulations=DEFAULT_MAX_CONCURRENT_SIMULATIONS,
        print_task_durations=False,
    )

    assert resolve_model_runner(args) is Binom_BP_Model


def test_resolve_model_runner_uses_docker_flag():
    args = Namespace(
        mrp_config=None,
        docker=True,
        cloud=False,
        max_concurrent_simulations=DEFAULT_MAX_CONCURRENT_SIMULATIONS,
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
        ):
            self.config_path = config_path
            self.generation_count = generation_count
            self.max_concurrent_simulations = max_concurrent_simulations
            self.repo_root = repo_root
            self.dockerfile = dockerfile
            self.print_task_durations = print_task_durations

    monkeypatch.setattr(
        "example_model.calibrate.ExampleModelCloudRunner", FakeCloudRunner
    )

    args = Namespace(
        mrp_config=None,
        docker=False,
        cloud=True,
        max_concurrent_simulations=7,
        print_task_durations=True,
        repo_root=str(tmp_path),
        dockerfile=str(dockerfile),
    )

    runner = resolve_model_runner(args)

    assert isinstance(runner, FakeCloudRunner)
    assert runner.config_path == DEFAULT_CLOUD_MRP_CONFIG_PATH
    assert runner.generation_count == 2
    assert runner.max_concurrent_simulations == 7
    assert Path(runner.repo_root) == tmp_path
    assert Path(runner.dockerfile) == dockerfile
    assert runner.print_task_durations is True


def test_resolve_max_concurrent_simulations_uses_cloud_default():
    args = Namespace(
        mrp_config=None,
        docker=False,
        cloud=True,
        max_concurrent_simulations=None,
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
        print_task_durations=False,
    )

    assert (
        resolve_max_concurrent_simulations(args)
        == DEFAULT_MAX_CONCURRENT_SIMULATIONS
    )


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
