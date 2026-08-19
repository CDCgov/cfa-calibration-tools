from __future__ import annotations

import inspect
from collections.abc import Callable


def _assert_parameters(
    callable_obj: Callable[..., object],
    names: tuple[str, ...],
) -> None:
    signature = inspect.signature(callable_obj)
    for name in names:
        assert name in signature.parameters


def test_calibration_md_top_level_imports_are_available():
    from calibrationtools import (
        ABCSampler,
        AdaptMultivariateNormalVariance,
        CSVTableOutputContract,
        IndependentKernels,
        MRPOutputRunner,
        MultivariateNormalKernel,
        Particle,
        ParticleReader,
        SamplerReporter,
        SeedKernel,
        run_coroutine_from_sync,
    )
    from calibrationtools.calibration_results import (
        CalibrationResults,
    )
    from calibrationtools.calibration_results import (
        Particle as ResultsParticle,
    )
    from calibrationtools.particle import Particle as ModuleParticle

    assert Particle is ModuleParticle
    assert ResultsParticle is ModuleParticle
    assert callable(run_coroutine_from_sync)

    _assert_parameters(
        ABCSampler,
        (
            "generation_particle_count",
            "tolerance_values",
            "priors",
            "particles_to_params",
            "outputs_to_distance",
            "target_data",
            "model_runner",
            "perturbation_kernel",
            "variance_adapter",
            "max_concurrent_simulations",
            "entropy",
            "print_generation_progress",
            "artifacts_dir",
        ),
    )
    _assert_parameters(
        ParticleReader, ("particle_param_names", "default_params")
    )
    _assert_parameters(
        CSVTableOutputContract,
        ("filename", "output_name", "orientation"),
    )
    _assert_parameters(
        MRPOutputRunner,
        ("config_path", "output_contract", "mrp_run_func"),
    )
    _assert_parameters(SamplerReporter, ("verbose",))

    for obj in (
        AdaptMultivariateNormalVariance,
        IndependentKernels,
        MultivariateNormalKernel,
        SeedKernel,
    ):
        assert inspect.isclass(obj)

    particle = Particle({"beta": 1.2})
    assert particle["beta"] == 1.2
    assert dict(particle.items()) == {"beta": 1.2}

    assert hasattr(CalibrationResults, "get_diagnostics")
    assert hasattr(CalibrationResults, "flatten_distance_history")
    assert hasattr(CalibrationResults, "sample_posterior_particles")
    assert isinstance(CalibrationResults.fitted_params, property)
    assert isinstance(CalibrationResults.ess, property)


def test_calibration_md_cloud_imports_are_available():
    from calibrationtools.cloud import cleanup
    from calibrationtools.cloud.auto_size import (
        CloudSizing,
        print_cloud_auto_size_summary,
        resolve_cloud_sizing_from_config,
    )
    from calibrationtools.cloud.config import load_cloud_model_config
    from calibrationtools.cloud.runner import (
        CloudMRPRunner,
        create_cloud_mrp_runner_from_config,
    )
    from calibrationtools.cloud.task_payload import (
        CloudTaskContext,
        apply_task_payload_transforms,
        bind_shared_assets_to_session,
        resolve_shared_assets,
        resolve_task_output_dir,
    )

    _assert_parameters(
        CloudSizing,
        (
            "max_concurrent_simulations",
            "task_slots_per_node_override",
            "summary",
        ),
    )
    _assert_parameters(
        resolve_cloud_sizing_from_config,
        (
            "cloud_config_path",
            "base_inputs",
            "auto_size",
            "cloud",
            "max_concurrent_simulations",
            "max_concurrent_simulations_explicit",
        ),
    )
    _assert_parameters(print_cloud_auto_size_summary, ("sizing",))
    _assert_parameters(load_cloud_model_config, ("config_path",))
    _assert_parameters(
        create_cloud_mrp_runner_from_config,
        (
            "config_path",
            "generation_count",
            "max_concurrent_simulations",
            "output_contract",
            "base_inputs",
            "print_task_durations",
            "task_slots_per_node_override",
            "auto_size_summary",
        ),
    )

    assert CloudMRPRunner.prefer_simulate_async is True
    assert hasattr(CloudMRPRunner, "dispatch_buffer_size")
    assert hasattr(CloudMRPRunner, "close")

    _assert_parameters(
        CloudTaskContext,
        (
            "run_id",
            "session_id",
            "job_name",
            "input_mount_path",
            "output_mount_path",
            "logs_mount_path",
            "task_output_dir",
            "shared_assets",
        ),
    )
    _assert_parameters(
        resolve_shared_assets,
        ("settings", "base_payload", "config_dir"),
    )
    _assert_parameters(
        bind_shared_assets_to_session,
        ("assets", "session_id", "input_mount_path"),
    )
    _assert_parameters(
        resolve_task_output_dir,
        ("settings", "context", "default_task_output_dir"),
    )
    _assert_parameters(
        apply_task_payload_transforms,
        ("payload", "settings", "context"),
    )

    assert callable(cleanup.main)
