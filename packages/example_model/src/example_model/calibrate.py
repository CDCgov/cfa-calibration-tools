"""Calibrate the example branching process."""

import argparse

import numpy as np
from mrp import Environment

from calibrationtools.perturbation_kernel import (
    IndependentKernels,
    MultivariateNormalKernel,
    SeedKernel,
)
from calibrationtools.sampler import ABCSampler
from calibrationtools.variance_adapter import AdaptMultivariateNormalVariance
from example_model import Binom_BP_Model


def positive_int(value: str) -> int:
    """Parse a positive integer command-line argument."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the example calibration command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run ABC-SMC calibration for the example model."
    )
    parser.add_argument(
        "--slot-lookahead",
        type=positive_int,
        default=None,
        help=(
            "Number of speculative attempts to keep submitted per generator "
            "slot during threaded parallel execution. Defaults to the "
            "sampler setting of 1."
        ),
    )
    return parser


##===================================#
## Define model
##===================================#
env = Environment(
    {
        "input": {
            "seed": 123,
            "max_gen": 15,
            "n": 3,
            "p": 0.5,
            "max_infect": 500,
        },
        "output": {"spec": "filesystem", "dir": "./output"},
    }
)
model = Binom_BP_Model(env=env)

##===================================#
## Define priors
##===================================#
P = {
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

K = IndependentKernels(
    [
        MultivariateNormalKernel(
            [p for p in P["priors"].keys()],
        ),
        SeedKernel("seed"),
    ]
)

V = AdaptMultivariateNormalVariance()


##===================================#
## Run ABC-SMC
##===================================#
def outputs_to_distance(model_output, target_data):
    return abs(np.sum(model_output) - target_data)


sampler = ABCSampler(
    generation_particle_count=500,
    tolerance_values=[5.0, 1.0],
    priors=P,
    perturbation_kernel=K,
    variance_adapter=V,
    default_parameters=model.env.input,
    outputs_to_distance=outputs_to_distance,
    target_data=5,
    model_runner=model,
    entropy=0x60636577C7AD93BBE463F30A6241FDE4,  # This value is the initial entropy for the `sampler.seed_sequence`
)


def report_results(results) -> None:
    # Default printed output is the CalibrationResults object, which includes
    # ESS, acceptance rates, and parameter details
    print(results)
    print("\nFlattened distance history (mean distance per generation):")
    print(
        [
            {
                k: np.mean(errs)
                for k, errs in results.flatten_distance_history().items()
            }
        ]
    )

    # Example user print function
    print("Posterior estimates table example:")
    for par_name in P["priors"].keys():
        print(
            f"{par_name}: {results.point_estimates[par_name]:.2f}, "
            "95% CI: "
            f"{[f'{v:.2f}' for v in results.credible_intervals[par_name]]}"
        )

    diagnostics = results.get_diagnostics()

    print("\nAvailable diagnostics metrics:")
    print(diagnostics.keys())

    print("\nQuantiles for each parameter:")
    print(diagnostics["quantiles"])

    print("\nCorrelation matrix:")
    print(diagnostics["correlation_matrix"])


def run_calibration(slot_lookahead: int | None = None):
    results = sampler.run(
        execution="parallel",
        slot_lookahead=slot_lookahead,
    )
    report_results(results)
    return results


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_calibration(slot_lookahead=args.slot_lookahead)


if __name__ == "__main__":
    main()
