"""Calibrate the example branching process."""

import argparse
import sys
from pathlib import Path
from typing import Any

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
    ExampleModelCloudRunner,
    ExampleModelDirectRunner,
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
DEFAULT_ARTIFACTS_DIR = Path("artifacts")


def particles_to_params(
    particle: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    base_inputs = kwargs.get("base_inputs")
    if not isinstance(base_inputs, dict):
        raise ValueError("base_inputs must be provided as a dictionary.")
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
            f"folders. Defaults to {DEFAULT_ARTIFACTS_DIR}."
        ),
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help=(
            "Disable local input/output artifact staging. Not valid with "
            "--cloud."
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
        pool_max_nodes=settings.pool_max_nodes,
        measure_task_peak_rss_bytes=(
            lambda: run_local_memory_probe(
                "example_model.cloud_auto_size",
                DEFAULT_INPUTS,
            )
        ),
    )


def resolve_artifacts_dir(args: argparse.Namespace) -> Path | None:
    artifacts_dir = args.artifacts_dir
    if args.no_artifacts:
        if artifacts_dir is not None:
            raise ValueError(
                "Pass either --artifacts-dir or --no-artifacts, not both."
            )
        if args.cloud:
            raise ValueError(
                "--cloud requires artifacts; omit --no-artifacts or pass "
                "--artifacts-dir."
            )
        return None
    return (
        artifacts_dir if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR
    )


def _format_bytes(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.1f} GiB"
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MiB"
    return f"{size} bytes"


def print_cloud_auto_size_summary(sizing: CloudSizing) -> None:
    summary = sizing.summary
    if summary is None:
        return

    cap_note = ""
    if summary.task_slots_per_node < summary.memory_task_slots_per_node:
        cap_note = (
            f", capped_from_ram_slots={summary.memory_task_slots_per_node}"
        )

    print(
        (
            "[cloud-run] auto-size simulation RAM "
            f"measured_peak_rss="
            f"{summary.measured_task_peak_rss_bytes} bytes "
            f"({_format_bytes(summary.measured_task_peak_rss_bytes)}), "
            f"vm_size={summary.vm_size}, "
            f"vm_ram={summary.vm_memory_bytes} bytes "
            f"({_format_bytes(summary.vm_memory_bytes)}), "
            f"reserve={summary.reserve:.0%}, "
            f"batch_slot_limit={summary.max_task_slots_per_node}, "
            f"task_slots_per_node={summary.task_slots_per_node}"
            f"{cap_note}, "
            f"max_concurrent_simulations_per_node="
            f"{summary.task_slots_per_node}, "
            f"max_concurrent_simulations_total="
            f"{sizing.max_concurrent_simulations}"
        ),
        file=sys.stderr,
        flush=True,
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
    return ExampleModelDirectRunner()


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
    artifacts_dir = resolve_artifacts_dir(args)
    cloud_sizing = resolve_cloud_sizing(args)
    print_cloud_auto_size_summary(cloud_sizing)

    run_calibration(
        model_runner=resolve_model_runner(
            args,
            cloud_sizing=cloud_sizing,
        ),
        max_concurrent_simulations=cloud_sizing.max_concurrent_simulations,
        print_task_progress=args.print_task_progress,
        artifacts_dir=artifacts_dir,
    )


if __name__ == "__main__":
    main()
