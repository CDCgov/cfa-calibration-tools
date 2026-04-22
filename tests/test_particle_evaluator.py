import inspect

import pytest

from calibrationtools.particle import Particle
from calibrationtools.particle_evaluator import (
    EVALUATION_CONTEXT_KEY,
    ParticleEvaluator,
)


class DummyModelRunner:
    def simulate(self, params):
        return 0.5 + params["p"]


def test_particle_evaluator_distance():
    evaluator = ParticleEvaluator(
        particles_to_params=lambda particle: particle,
        outputs_to_distance=lambda model_output, target_data: abs(
            model_output - target_data
        ),
        target_data=0.75,
        model_runner=DummyModelRunner(),
    )

    distance = evaluator.distance(Particle({"p": 0.1}))

    assert distance == pytest.approx(0.15)


def test_particle_evaluator_distance_passes_kwargs():
    evaluator = ParticleEvaluator(
        particles_to_params=lambda particle, scale: {
            "p": particle["p"] * scale
        },
        outputs_to_distance=lambda model_output, target_data: abs(
            model_output - target_data
        ),
        target_data=0.9,
        model_runner=DummyModelRunner(),
    )

    distance = evaluator.distance(Particle({"p": 0.2}), scale=2.0)

    assert distance == pytest.approx(0.0)


def test_particle_evaluator_prefers_simulate_when_available():
    calls: list[tuple[str, dict]] = []

    class DualModelRunner:
        def simulate(self, params, **kwargs):
            calls.append(("simulate", {"params": params, **kwargs}))
            return 1.0

        async def simulate_async(self, params, **kwargs):
            raise AssertionError("simulate_async() should not be used")

    evaluator = ParticleEvaluator(
        particles_to_params=lambda particle: particle,
        outputs_to_distance=lambda model_output, target_data: abs(
            model_output - target_data
        ),
        target_data=1.5,
        model_runner=DualModelRunner(),
    )

    distance = evaluator.distance(Particle({"p": 0.2}))

    assert distance == pytest.approx(0.5)
    assert calls == [("simulate", {"params": {"p": 0.2}})]


def test_particle_evaluator_can_prefer_simulate_async_when_runner_opts_in():
    calls: list[tuple[str, dict]] = []

    class AsyncPreferredRunner:
        prefer_simulate_async = True

        def simulate(self, params, **kwargs):
            calls.append(("simulate", {"params": params, **kwargs}))
            return 1.0

        async def simulate_async(self, params, **kwargs):
            calls.append(("simulate_async", {"params": params, **kwargs}))
            return 1.25

    evaluator = ParticleEvaluator(
        particles_to_params=lambda particle: particle,
        outputs_to_distance=lambda model_output, target_data: abs(
            model_output - target_data
        ),
        target_data=1.0,
        model_runner=AsyncPreferredRunner(),
    )

    distance = evaluator.distance(Particle({"p": 0.2}))

    assert distance == pytest.approx(0.25)
    assert calls == [("simulate_async", {"params": {"p": 0.2}})]


def test_particle_evaluator_writes_artifacts_under_correct_generation(
    tmp_path,
):
    class RecordingRunner:
        def simulate(
            self, params, *, input_path=None, output_dir=None, run_id=None
        ):
            return {"run_id": run_id, "input_path": str(input_path)}

    evaluator = ParticleEvaluator(
        particles_to_params=lambda particle: dict(particle),
        outputs_to_distance=lambda outputs, target: 0.0,
        target_data=None,
        model_runner=RecordingRunner(),
        artifacts_dir=tmp_path,
    )

    for generation_index in (0, 1, 2):
        evaluator.distance(
            Particle({"p": 0.1}),
            evaluation_context={
                "generation_index": generation_index,
                "proposal_index": 0,
                "attempt_index": 0,
            },
        )

    for human_generation in (1, 2, 3):
        gen_dir = tmp_path / "input" / f"generation-{human_generation}"
        assert gen_dir.is_dir()
        assert (gen_dir / f"gen-{human_generation}_particle-1.json").is_file()


def test_particle_evaluator_writes_direct_artifacts_without_context(
    tmp_path,
):
    class RecordingRunner:
        def simulate(
            self, params, *, input_path=None, output_dir=None, run_id=None
        ):
            return {
                "run_id": run_id,
                "input_path": str(input_path),
                "output_dir": str(output_dir),
            }

    evaluator = ParticleEvaluator(
        particles_to_params=lambda particle: dict(particle),
        outputs_to_distance=lambda outputs, target: 0.0,
        target_data=None,
        model_runner=RecordingRunner(),
        artifacts_dir=tmp_path,
    )

    assert evaluator.distance(Particle({"p": 0.1})) == 0.0

    input_files = sorted((tmp_path / "input" / "direct").glob("*.json"))
    assert len(input_files) == 1

    staged = input_files[0].read_text()
    assert '"run_id": "direct-' in staged

    output_roots = sorted((tmp_path / "output" / "direct").iterdir())
    assert len(output_roots) == 1
    assert output_roots[0].is_dir()
    assert (output_roots[0] / "result.json").is_file()


def test_particle_evaluator_signature_exposes_public_evaluation_context():
    simulate_params = inspect.signature(ParticleEvaluator.simulate).parameters
    distance_params = inspect.signature(ParticleEvaluator.distance).parameters

    assert "evaluation_context" in simulate_params
    assert "evaluation_context" in distance_params


def test_particle_evaluator_accepts_legacy_hidden_context_kwarg(tmp_path):
    class RecordingRunner:
        def simulate(
            self, params, *, input_path=None, output_dir=None, run_id=None
        ):
            return {"run_id": run_id}

    evaluator = ParticleEvaluator(
        particles_to_params=lambda particle: dict(particle),
        outputs_to_distance=lambda outputs, target: 0.0,
        target_data=None,
        model_runner=RecordingRunner(),
        artifacts_dir=tmp_path,
    )

    evaluator.distance(
        Particle({"p": 0.1}),
        **{
            EVALUATION_CONTEXT_KEY: {
                "generation_index": 1,
                "proposal_index": 2,
                "attempt_index": 0,
            }
        },
    )

    assert (
        tmp_path / "input" / "generation-2" / "gen-2_particle-3.json"
    ).is_file()


def _pickle_particles_to_params(particle):
    return dict(particle)


def _pickle_outputs_to_distance(outputs, target):
    return 0.0


class _PickleDummyModelRunner:
    def simulate(self, params, **kwargs):
        return 0.0


def test_particle_evaluator_survives_pickle():
    import pickle

    evaluator = ParticleEvaluator(
        particles_to_params=_pickle_particles_to_params,
        outputs_to_distance=_pickle_outputs_to_distance,
        target_data=None,
        model_runner=_PickleDummyModelRunner(),
    )

    restored = pickle.loads(pickle.dumps(evaluator))

    assert restored.particles_to_params is _pickle_particles_to_params
    assert restored.outputs_to_distance is _pickle_outputs_to_distance
    assert restored.target_data is None
    assert isinstance(restored.model_runner, _PickleDummyModelRunner)
    assert restored.artifacts_dir is None
