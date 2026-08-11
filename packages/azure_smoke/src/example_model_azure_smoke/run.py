"""Run one deliberately small Azure Batch calibration using ``example-model``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import numpy as np
from example_model import Binom_BP_Model
from mrp import Environment

from calibrationtools.azure_batch_executor import AzureBatchExecutor
from calibrationtools.perturbation_kernel import (
    IndependentKernels,
    NormalKernel,
    SeedKernel,
)
from calibrationtools.sampler import ABCSampler
from calibrationtools.variance_adapter import AdaptIdentityVariance

DEFAULT_PARAMETERS: dict[str, int | float] = {
    "seed": 123,
    "max_gen": 8,
    "n": 3,
    "p": 0.5,
    "max_infect": 100,
}
PRIORS: dict[str, Any] = {
    "priors": {
        "p": {
            "distribution": "uniform",
            "parameters": {"min": 0.0, "max": 1.0},
        }
    }
}
SMOKE_TOLERANCE = 10_000.0


def outputs_to_distance(model_output: list[int], target_data: float) -> float:
    """Score one branching-process output against the small smoke target."""

    return abs(float(np.sum(model_output)) - target_data)


def build_smoke_sampler(particle_count: int = 4) -> ABCSampler:
    """Build a one-generation sampler that finishes in one attempt per slot.

    The broad, finite tolerance makes this an infrastructure check rather than
    a statistically meaningful calibration. The top-level functions and model
    class are importable in the worker image, so the normal Azure task-pickle
    preflight also exercises the intended deployment boundary.
    """

    if particle_count <= 0:
        raise ValueError("particle_count must be positive")

    return ABCSampler(
        generation_particle_count=particle_count,
        tolerance_values=[SMOKE_TOLERANCE],
        priors=PRIORS,
        perturbation_kernel=IndependentKernels(
            [NormalKernel("p", 0.25), SeedKernel("seed")]
        ),
        variance_adapter=AdaptIdentityVariance(),
        default_parameters=DEFAULT_PARAMETERS,
        outputs_to_distance=outputs_to_distance,
        target_data=5.0,
        model_runner=Binom_BP_Model(
            env=Environment({"input": dict(DEFAULT_PARAMETERS)})
        ),
        max_attempts_per_proposal=1,
        entropy=123,
        verbose=False,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse Azure resource and cost-control settings for the smoke run."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the bundled branching-process model through one small Azure "
            "Batch generation."
        )
    )
    parser.add_argument("--base-name", default="example-model-smoke")
    parser.add_argument(
        "--registry-server",
        help=(
            "ACR server, e.g. myregistry.azurecr.io; defaults to "
            "AZURE_CONTAINER_REGISTRY_SERVER."
        ),
    )
    parser.add_argument(
        "--image-name", default="calibrationtools-example-smoke"
    )
    parser.add_argument("--image-tag", default="smoke")
    parser.add_argument(
        "--image-dockerfile",
        default="packages/azure_smoke/Dockerfile",
        help="Dockerfile path interpreted from the repository root.",
    )
    parser.add_argument("--particle-count", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--max-autoscale-nodes", type=int, default=1)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--max-wait", type=float, default=1800.0)
    image_actions = parser.add_mutually_exclusive_group()
    image_actions.add_argument("--build-image", action="store_true")
    image_actions.add_argument("--upload-image", action="store_true")
    parser.add_argument(
        "--keep-job",
        action="store_false",
        dest="delete_job_after",
        help="Retain the Batch job after completion for inspection.",
    )
    parser.add_argument(
        "--keep-pool",
        action="store_false",
        dest="delete_pool_after",
        help="Retain the Batch pool after completion for inspection.",
    )
    parser.set_defaults(delete_job_after=True, delete_pool_after=True)
    return parser.parse_args(argv)


def build_executor(args: argparse.Namespace) -> AzureBatchExecutor:
    """Create the Azure executor configured by the smoke-test command line."""

    return AzureBatchExecutor(
        base_name=args.base_name,
        registry_server=args.registry_server,
        image_name=args.image_name,
        image_tag=args.image_tag,
        max_autoscale_nodes=args.max_autoscale_nodes,
        chunk_size=args.chunk_size,
        delete_job_after=args.delete_job_after,
        delete_pool_after=args.delete_pool_after,
        build_image=args.build_image,
        upload_image=args.upload_image,
        image_dockerfile=args.image_dockerfile,
        poll_interval=args.poll_interval,
        max_wait=args.max_wait,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example-model Azure smoke test and print its concise outcome."""

    args = parse_args(argv)
    sampler = build_smoke_sampler(args.particle_count)
    results = sampler.run(
        execution="azure_batch",
        cloud_executor=build_executor(args),
    )
    print(
        "Azure smoke test completed: "
        f"{results.smc_step_successes[0]} accepted particles in "
        f"{results.smc_step_attempts[0]} attempts."
    )
    return 0
