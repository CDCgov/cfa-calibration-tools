import pytest

from calibrationtools.particle import Particle
from calibrationtools.particle_evaluator import ParticleEvaluator
from calibrationtools.particle_reader import ParticleReader


class DummyModelRunner:
    def simulate(self, params):
        return 0.5 + params["p"]


@pytest.fixture
def basic_reader() -> ParticleReader:
    return ParticleReader(particle_param_names=["p"])


@pytest.fixture
def scale_reader() -> ParticleReader:
    return ParticleReader(
        particle_param_names=["p"],
        read_fn=lambda particle, scale: {"p": particle["p"] * scale},
    )


def test_particle_evaluator_distance(basic_reader):
    evaluator = ParticleEvaluator(
        particle_reader=basic_reader,
        outputs_to_distance=lambda model_output, target_data: abs(
            model_output - target_data
        ),
        target_data=0.75,
        model_runner=DummyModelRunner(),
    )

    distance = evaluator.distance(Particle({"p": 0.1}))

    assert distance == pytest.approx(0.15)


def test_particle_evaluator_distance_passes_kwargs(scale_reader):
    evaluator = ParticleEvaluator(
        particle_reader=scale_reader,
        outputs_to_distance=lambda model_output, target_data: abs(
            model_output - target_data
        ),
        target_data=0.9,
        model_runner=DummyModelRunner(),
    )

    distance = evaluator.distance(Particle({"p": 0.2}), scale=2.0)

    assert distance == pytest.approx(0.0)
