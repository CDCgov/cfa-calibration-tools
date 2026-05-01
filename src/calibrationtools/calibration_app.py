from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mrp import run as mrp_run

from .calibration_results import CalibrationResults
from .cloud.auto_size import (
    CloudSizing,
    print_cloud_auto_size_summary,
    resolve_cloud_sizing_from_config,
)
from .cloud.runner import create_csv_cloud_mrp_runner_from_config
from .mrp_csv_runner import CSVOutputMRPRunner
from .sampler import ABCSampler


@dataclass(frozen=True)
class CSVOutputContract:
    """Describe how shared runners parse a model's CSV output."""

    filename: str
    value_column: str
    value_parser: Callable[[str], Any]
    header_fields: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CalibrationAppSpec:
    """Model-specific configuration consumed by the shared calibration app."""

    default_inputs: dict[str, Any]
    priors: dict[str, Any]
    tolerance_values: list[float]
    target_data: Any
    outputs_to_distance: Callable[..., float]
    direct_runner_factory: Callable[[], object]
    output_contract: CSVOutputContract
    perturbation_kernel_factory: Callable[[], Any] | None = None
    variance_adapter_factory: Callable[[], Any] | None = None
    output_reporter: Callable[[CalibrationResults], None] | None = None
    default_mrp_config_path: Path | None = None
    default_docker_mrp_config_path: Path | None = None
    default_cloud_config_path: Path | None = None
    generation_particle_count: int = 500
    cloud_default_concurrency: int = 50
    local_default_concurrency: int = 10
    default_artifacts_dir: Path = Path("artifacts")
    entropy: int | None = 123


def build_calibration_parser(
    spec: CalibrationAppSpec,
) -> argparse.ArgumentParser:
    """Build the shared calibration command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run ABC-SMC calibration for a model."
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
        "--cloud-config",
        type=Path,
        default=spec.default_cloud_config_path,
        help="Cloud config used by --cloud and --auto-size.",
    )
    parser.add_argument(
        "--max-concurrent-simulations",
        type=int,
        default=None,
        help="Maximum number of simulations to evaluate at once.",
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
        "--print-task-progress",
        action="store_true",
        help="Print generation-level calibration progress.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Root directory where calibration writes input and output folders.",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help=(
            "Disable local input/output artifact staging. Not valid with "
            "--cloud."
        ),
    )
    return parser


def parse_calibration_args(
    argv: list[str] | None,
    spec: CalibrationAppSpec,
) -> argparse.Namespace:
    """Parse shared calibration CLI arguments."""
    return build_calibration_parser(spec).parse_args(argv)


def resolve_max_concurrent_simulations(
    args: argparse.Namespace,
    spec: CalibrationAppSpec,
) -> int:
    """Resolve and validate calibration concurrency from parsed args."""
    if args.max_concurrent_simulations is not None:
        value = args.max_concurrent_simulations
    elif args.cloud:
        value = spec.cloud_default_concurrency
    else:
        value = spec.local_default_concurrency

    if value < 1:
        raise ValueError(
            f"--max-concurrent-simulations must be at least 1 (got {value})"
        )
    return value


def resolve_artifacts_dir(
    args: argparse.Namespace,
    spec: CalibrationAppSpec,
) -> Path | None:
    """Resolve the local artifact directory policy from parsed args."""
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
        artifacts_dir
        if artifacts_dir is not None
        else spec.default_artifacts_dir
    )


def resolve_cloud_sizing(
    args: argparse.Namespace,
    spec: CalibrationAppSpec,
) -> CloudSizing:
    """Resolve optional cloud sizing from shared CLI args and model spec."""
    max_concurrent_simulations = resolve_max_concurrent_simulations(args, spec)
    cloud_config_path = args.cloud_config or spec.default_cloud_config_path
    if cloud_config_path is None:
        if args.cloud:
            raise ValueError("--cloud requires --cloud-config")
        cloud_config_path = Path("cloud_config.toml")

    return resolve_cloud_sizing_from_config(
        cloud_config_path=cloud_config_path,
        base_inputs=spec.default_inputs,
        auto_size=args.auto_size,
        cloud=args.cloud,
        max_concurrent_simulations=max_concurrent_simulations,
        max_concurrent_simulations_explicit=(
            args.max_concurrent_simulations is not None
        ),
    )


def _create_csv_mrp_runner(
    config_path: Path,
    spec: CalibrationAppSpec,
) -> CSVOutputMRPRunner[Any]:
    """Create a CSV MRP runner from the model output contract."""
    return CSVOutputMRPRunner(
        config_path,
        output_filename=spec.output_contract.filename,
        value_column=spec.output_contract.value_column,
        value_parser=spec.output_contract.value_parser,
        header_fields=spec.output_contract.header_fields,
        mrp_run_func=mrp_run,
    )


def resolve_model_runner(
    args: argparse.Namespace,
    spec: CalibrationAppSpec,
    *,
    cloud_sizing: CloudSizing,
) -> object:
    """Construct the direct, MRP, Docker MRP, or cloud model runner."""
    if args.cloud:
        cloud_config_path = args.cloud_config or spec.default_cloud_config_path
        if cloud_config_path is None:
            raise ValueError("--cloud requires --cloud-config")
        return create_csv_cloud_mrp_runner_from_config(
            cloud_config_path,
            generation_count=len(spec.tolerance_values),
            max_concurrent_simulations=cloud_sizing.max_concurrent_simulations,
            task_slots_per_node_override=(
                cloud_sizing.task_slots_per_node_override
            ),
            print_task_durations=args.print_task_durations,
            auto_size_summary=cloud_sizing.summary,
        )
    if args.mrp_config is not None:
        return _create_csv_mrp_runner(args.mrp_config, spec)
    if args.docker:
        if spec.default_docker_mrp_config_path is None:
            raise ValueError("--docker requires a default Docker MRP config")
        return _create_csv_mrp_runner(
            spec.default_docker_mrp_config_path, spec
        )
    return spec.direct_runner_factory()


def run_abc_smc(
    *,
    model_runner: object,
    spec: CalibrationAppSpec,
    max_concurrent_simulations: int,
    print_task_progress: bool = False,
    artifacts_dir: Path | None = None,
) -> CalibrationResults:
    """Run ABC-SMC with the shared sampler lifecycle and cleanup policy."""
    try:
        sampler = ABCSampler(
            generation_particle_count=spec.generation_particle_count,
            tolerance_values=spec.tolerance_values,
            priors=spec.priors,
            perturbation_kernel=(
                spec.perturbation_kernel_factory()
                if spec.perturbation_kernel_factory is not None
                else None
            ),
            variance_adapter=(
                spec.variance_adapter_factory()
                if spec.variance_adapter_factory is not None
                else None
            ),
            default_parameters=spec.default_inputs,
            outputs_to_distance=spec.outputs_to_distance,
            target_data=spec.target_data,
            model_runner=model_runner,
            max_concurrent_simulations=max_concurrent_simulations,
            entropy=spec.entropy,
            print_generation_progress=print_task_progress,
            artifacts_dir=artifacts_dir,
        )
        results = sampler.run()
        if spec.output_reporter is not None:
            spec.output_reporter(results)
        return results
    finally:
        close = getattr(model_runner, "close", None)
        if callable(close):
            close()


def run_calibration_app(
    argv: list[str] | None,
    spec: CalibrationAppSpec,
) -> CalibrationResults:
    """Run the full shared calibration CLI for a model spec."""
    args = parse_calibration_args(argv, spec)
    artifacts_dir = resolve_artifacts_dir(args, spec)
    cloud_sizing = resolve_cloud_sizing(args, spec)
    print_cloud_auto_size_summary(cloud_sizing)
    return run_abc_smc(
        model_runner=resolve_model_runner(
            args,
            spec,
            cloud_sizing=cloud_sizing,
        ),
        spec=spec,
        max_concurrent_simulations=cloud_sizing.max_concurrent_simulations,
        print_task_progress=args.print_task_progress,
        artifacts_dir=artifacts_dir,
    )
