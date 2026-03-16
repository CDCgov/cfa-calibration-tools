from argparse import Namespace
from pathlib import Path

from example_model import Binom_BP_Model, ExampleModelMRPRunner
from example_model.calibrate import resolve_model_runner
from example_model.mrp_runner import DEFAULT_DOCKER_MRP_CONFIG_PATH


def test_resolve_model_runner_defaults_to_direct_simulation():
    args = Namespace(mrp_config=None, docker=False)

    assert resolve_model_runner(args) is Binom_BP_Model


def test_resolve_model_runner_uses_docker_flag():
    args = Namespace(mrp_config=None, docker=True)

    runner = resolve_model_runner(args)

    assert isinstance(runner, ExampleModelMRPRunner)
    assert runner.config_path == DEFAULT_DOCKER_MRP_CONFIG_PATH


def test_resolve_model_runner_uses_explicit_mrp_config():
    config_path = Path("/tmp/custom.mrp.toml")
    args = Namespace(mrp_config=config_path, docker=False)

    runner = resolve_model_runner(args)

    assert isinstance(runner, ExampleModelMRPRunner)
    assert runner.config_path == config_path
