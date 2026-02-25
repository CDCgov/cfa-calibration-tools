from calibrationtools import Particle


def test_particle_initialization(state) -> None:
    particle = Particle(state)
    assert particle == state


def test_particle_setter(state, state2) -> None:
    particle = Particle(state)
    assert particle == state
    particle = state2
    assert particle == state2


def test_state_parameter_change(particle, state2) -> None:
    particle["x"] = 5.0
    particle["y"] = state2["y"]
    assert particle["x"] == 5.0
    assert particle["y"] == state2["y"]


def test_particle_repr(particle) -> None:
    repr_str = repr(particle)
    assert "Particle" in repr_str
    assert "x" in repr_str
    assert "state" in repr_str
