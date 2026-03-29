import pytest

from calibrationtools.particle import Particle
from calibrationtools.particle_evaluator import ParticleEvaluator


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
        particles_to_params=lambda particle, scale: {"p": particle["p"] * scale},
        outputs_to_distance=lambda model_output, target_data: abs(
            model_output - target_data
        ),
        target_data=0.9,
        model_runner=DummyModelRunner(),
    )

    distance = evaluator.distance(Particle({"p": 0.2}), scale=2.0)

    assert distance == pytest.approx(0.0)
