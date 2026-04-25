"""Calibrate the example branching process."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
from mrp.api import apply_dict_overrides

from calibrationtools.cloud.auto_size import (
    CloudSizing,
    resolve_cloud_auto_size,
    run_local_memory_probe,
)
from calibrationtools.perturbation_kernel import (
    IndependentKernels,
    MultivariateNormalKernel,
    SeedKernel,
)
from calibrationtools.sampler import ABCSampler
from calibrationtools.variance_adapter import AdaptMultivariateNormalVariance
from example_model import (
    DEFAULT_CLOUD_MRP_CONFIG_PATH,
    DEFAULT_DOCKER_MRP_CONFIG_PATH,
    Binom_BP_Model,
    ExampleModelCloudRunner,
    ExampleModelMRPRunner,
)
from example_model.cloud_runner import resolve_cloud_build_context
from example_model.cloud_utils import load_cloud_runtime_settings

DEFAULT_INPUTS = {
    "seed": 123,
    "max_gen": 15,
    "n": 3,
    "p": 0.5,
    "max_infect": 500,
}

PRIORS = {
    "priors": {
        "p": {
            "distribution": "uniform",
            "parameters": {"min": 0.0, "max": 1.0},
        },
        "n": {
            "distribution": "uniform",
            "parameters": {"min": 0.0, "max": 5.0},
        },
    }
}
TOLERANCE_VALUES = [5.0, 1.0]
DEFAULT_MAX_CONCURRENT_SIMULATIONS = 10
DEFAULT_CLOUD_MAX_CONCURRENT_SIMULATIONS = 50


def particles_to_params(particle, **kwargs):
    base_inputs = kwargs.get("base_inputs")
    return apply_dict_overrides(base_inputs, particle)


def outputs_to_distance(model_output, target_data):
    return abs(np.sum(model_output) - target_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ABC-SMC calibration for the example model."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--docker",
        action="store_true",
        help="Run each simulation through the Docker-backed MRP config.",
    )
    mode_group.add_argument(
        "--cloud",
        action="store_true",
        help="Run each simulation through the cloud-backed MRP config.",
    )
    mode_group.add_argument(
        "--mrp-config",
        type=Path,
        help="Run simulations through the given MRP config path.",
    )
    parser.add_argument(
        "--max-concurrent-simulations",
        type=int,
        default=None,
        help=(
            "Maximum number of simulations to evaluate at once. "
            "Default: 50 for --cloud, 10 otherwise."
        ),
    )
    parser.add_argument(
        "--auto-size",
        action="store_true",
        help=(
            "Cloud mode only. Run one local probe simulation before Azure "
            "provisioning and set task slots from measured RAM usage."
        ),
    )
    parser.add_argument(
        "--print-task-durations",
        action="store_true",
        help="When running in cloud mode, print per-task timing information.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Path to the cfa-calibration-tools source tree used as the "
            "docker build context for cloud mode. Defaults to the source "
            "tree adjacent to the installed example_model package; "
            "required when running from a wheel."
        ),
    )
    parser.add_argument(
        "--dockerfile",
        type=Path,
        default=None,
        help=(
            "Path to the example-model Dockerfile used by cloud mode. "
            "Defaults to packages/example_model/Dockerfile under "
            "--repo-root."
        ),
    )
    parser.add_argument(
        "--print-task-progress",
        action="store_true",
        help="Print generation-level calibration progress.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help=(
            "Root directory where calibration writes input and output "
            "folders. Defaults to no on-disk staging for pure in-process "
            "runs; required for --docker and --cloud workflows."
        ),
    )
    return parser.parse_args()


def resolve_max_concurrent_simulations(args: argparse.Namespace) -> int:
    if args.max_concurrent_simulations is not None:
        value = args.max_concurrent_simulations
    elif args.cloud:
        value = DEFAULT_CLOUD_MAX_CONCURRENT_SIMULATIONS
    else:
        value = DEFAULT_MAX_CONCURRENT_SIMULATIONS
    # Validate here (before resolve_model_runner) so --cloud cannot
    # provision Azure resources for a concurrency value the sampler
    # would later reject.
    if value < 1:
        raise ValueError(
            f"--max-concurrent-simulations must be at least 1 (got {value})"
        )
    return value


def resolve_cloud_sizing(args: argparse.Namespace) -> CloudSizing:
    max_concurrent_simulations = resolve_max_concurrent_simulations(args)
    if not args.auto_size or not args.cloud:
        return resolve_cloud_auto_size(
            auto_size=args.auto_size,
            cloud=args.cloud,
            max_concurrent_simulations=max_concurrent_simulations,
            max_concurrent_simulations_explicit=(
                args.max_concurrent_simulations is not None
            ),
        )

    settings = load_cloud_runtime_settings(DEFAULT_CLOUD_MRP_CONFIG_PATH)
    return resolve_cloud_auto_size(
        auto_size=args.auto_size,
        cloud=args.cloud,
        max_concurrent_simulations=max_concurrent_simulations,
        max_concurrent_simulations_explicit=(
            args.max_concurrent_simulations is not None
        ),
        vm_size=settings.vm_size,
        measure_task_peak_rss_bytes=(
            lambda: run_local_memory_probe(
                "example_model.cloud_auto_size",
                DEFAULT_INPUTS,
            )
        ),
    )


def resolve_model_runner(
    args: argparse.Namespace,
    *,
    cloud_sizing: CloudSizing | None = None,
):
    if args.cloud:
        if cloud_sizing is None:
            cloud_sizing = resolve_cloud_sizing(args)
        repo_root, dockerfile = resolve_cloud_build_context(
            repo_root=args.repo_root,
            dockerfile=args.dockerfile,
        )
        return ExampleModelCloudRunner(
            DEFAULT_CLOUD_MRP_CONFIG_PATH,
            generation_count=len(TOLERANCE_VALUES),
            max_concurrent_simulations=cloud_sizing.max_concurrent_simulations,
            repo_root=repo_root,
            dockerfile=dockerfile,
            print_task_durations=args.print_task_durations,
            task_slots_per_node_override=(
                cloud_sizing.task_slots_per_node_override
            ),
            auto_size_summary=cloud_sizing.summary,
        )
    if args.mrp_config is not None:
        return ExampleModelMRPRunner(args.mrp_config)
    if args.docker:
        return ExampleModelMRPRunner(DEFAULT_DOCKER_MRP_CONFIG_PATH)
    return Binom_BP_Model


def run_calibration(
    *,
    model_runner,
    max_concurrent_simulations: int = DEFAULT_MAX_CONCURRENT_SIMULATIONS,
    print_task_progress: bool = False,
    artifacts_dir: Path | None = None,
):
    kernel = IndependentKernels(
        [
            MultivariateNormalKernel(
                [parameter for parameter in PRIORS["priors"]]
            ),
            SeedKernel("seed"),
        ]
    )
    variance_adapter = AdaptMultivariateNormalVariance()
    try:
        # Construct the sampler inside the try/finally so that any error
        # raised during ABCSampler construction (e.g. invalid
        # max_concurrent_simulations) still runs model_runner.close() and
        # releases any cloud resources the runner has already provisioned.
        sampler = ABCSampler(
            generation_particle_count=500,
            tolerance_values=TOLERANCE_VALUES,
            priors=PRIORS,
            perturbation_kernel=kernel,
            variance_adapter=variance_adapter,
            particles_to_params=particles_to_params,
            outputs_to_distance=outputs_to_distance,
            target_data=5,
            model_runner=model_runner,
            max_concurrent_simulations=max_concurrent_simulations,
            entropy=123,
            print_generation_progress=print_task_progress,
            artifacts_dir=artifacts_dir,
        )
        results = sampler.run(base_inputs=DEFAULT_INPUTS)
        print(results)

        posterior_particles = results.posterior_particles
        p_values = [
            particle["p"] for particle in posterior_particles.particles
        ]
        n_values = [
            particle["n"] for particle in posterior_particles.particles
        ]

        print(
            f"param p(25-75):{np.percentile(p_values, 25)} - {np.percentile(p_values, 75)}"
        )
        print(
            f"param n(25-75):{np.percentile(n_values, 25)} - {np.percentile(n_values, 75)}"
        )
        return results
    finally:
        close = getattr(model_runner, "close", None)
        if callable(close):
            close()


def main():
    args = parse_args()
    cloud_sizing = resolve_cloud_sizing(args)

    artifacts_dir = args.artifacts_dir
    temp_artifacts_dir: tempfile.TemporaryDirectory | None = None
    # Cloud mode requires on-disk staging because the async runner needs a
    # concrete output_dir to write downloaded blobs into. Rather than
    # failing at runtime deep inside `simulate_async()`, allocate a
    # managed temp dir here so the default `--cloud` CLI invocation just
    # works. Users can still pass `--artifacts-dir` explicitly to persist
    # artifacts.
    if args.cloud and artifacts_dir is None:
        temp_artifacts_dir = tempfile.TemporaryDirectory(
            prefix="example-model-cloud-artifacts-"
        )
        artifacts_dir = Path(temp_artifacts_dir.name)
        print(
            f"--cloud requires staged artifacts; using temporary "
            f"directory {artifacts_dir}"
        )

    try:
        run_calibration(
            model_runner=resolve_model_runner(
                args,
                cloud_sizing=cloud_sizing,
            ),
            max_concurrent_simulations=(
                cloud_sizing.max_concurrent_simulations
            ),
            print_task_progress=args.print_task_progress,
            artifacts_dir=artifacts_dir,
        )
    finally:
        if temp_artifacts_dir is not None:
            temp_artifacts_dir.cleanup()


if __name__ == "__main__":
    main()
