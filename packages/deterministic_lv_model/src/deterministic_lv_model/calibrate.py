"""Calibrate the deterministic Lotka-Volterra model."""

import numpy as np
from mrp import Environment
from mrp.api import apply_dict_overrides

from calibrationtools.perturbation_kernel import (
    IndependentKernels,
    SeedKernel,
    UniformKernel,
)
from calibrationtools.sampler import ABCSampler
from calibrationtools.variance_adapter import AdaptIdentityVariance
from deterministic_lv_model import Deterministic_LV_Model

##===================================#
## Define model
##===================================#
env = Environment(
    {
        "input": {
            "seed": 321,
            "max_time": 15,
            "a": 1.0,
            "b": 1.0,
            "x0": 1.0,
            "y0": 0.5,
            "obs_times": [1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4],
            "obs_noise_mu": 0.0,
            "obs_noise_sigma": 0.5,
        },
        "output": {"spec": "filesystem", "dir": "./output"},
    }
)
default_inputs = {
    "seed": 321,
    "max_time": 15,
    "a": 1.0,
    "b": 1.0,
    "x0": 1.0,
    "y0": 0.5,
    "obs_times": [1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4],
    "obs_noise_mu": 0.0,
    "obs_noise_sigma": 0.5,
    "stop_x_threshold": 1e6,
}
model = Deterministic_LV_Model(env=env)

##===================================#
## Define priors
##===================================#
P = {
    "priors": {
        "a": {
            "distribution": "uniform",
            "parameters": {"min": -10.0, "max": 10.0},
        },
        "b": {
            "distribution": "uniform",
            "parameters": {"min": -10.0, "max": 10.0},
        },
    }
}

K = IndependentKernels(
    [
        UniformKernel("a", width=0.2),
        UniformKernel("b", width=0.2),
        SeedKernel("seed"),
    ]
)

V = AdaptIdentityVariance()


##===================================#
## Run ABC-SMC
##===================================#
def particles_to_params(particle, **kwargs):
    base_inputs = kwargs.get("base_inputs")
    model_params = apply_dict_overrides(base_inputs, particle)
    return model_params


def outputs_to_distance(model_output, target_data):
    model_x = np.asarray(model_output["observed_x"], dtype=float)
    model_y = np.asarray(model_output["observed_y"], dtype=float)
    target_x = np.asarray(target_data["x"], dtype=float)
    target_y = np.asarray(target_data["y"], dtype=float)

    if model_x.shape != target_x.shape or model_y.shape != target_y.shape:
        raise ValueError(
            "Model output and target_data must have matching x/y vector lengths"
        )

    x_sse = np.sum((model_x - target_x) ** 2)
    #    print(f"x_sse: {x_sse}")
    y_sse = np.sum((model_y - target_y) ** 2)
    #    print(f"y_sse: {y_sse}")
    return float(x_sse + y_sse)


sampler = ABCSampler(
    generation_particle_count=1000,
    tolerance_values=[30.0, 16.0, 6.0, 5.0, 4.3],
    priors=P,
    perturbation_kernel=K,
    variance_adapter=V,
    particles_to_params=particles_to_params,
    outputs_to_distance=outputs_to_distance,
    target_data={
        "x": [1.87, 0.65, 0.22, 0.31, 1.64, 1.15, 0.24, 2.91],
        "y": [0.49, 2.62, 1.54, 0.02, 1.14, 1.68, 1.07, 0.88],
    },
    model_runner=model,
    entropy=0x60636577C7AD93BBE463F30A6241FDE4,  # This value is the initial entropy for the `sampler.seed_sequence`
)

results = sampler.run(execution="serial", base_inputs=default_inputs)
# Default printed output is the CalibrationResults object, which includes ESS, acceptance rates, and parameter details
print(results)

# Example user print function
print("Posterior estimates table example:")
for par_name in P["priors"].keys():
    print(
        f"{par_name}: {results.point_estimates[par_name]:.2f}, 95% CI: {[f'{v:.2f}' for v in results.credible_intervals[par_name]]}"
    )

diagnostics = results.get_diagnostics()

print("\nAvailable diagnostics metrics:")
print(diagnostics.keys())

print("\nQuantiles for each parameter:")
print(diagnostics["quantiles"])

print("\nCorrelation matrix:")
print(diagnostics["correlation_matrix"])
