from typing import Any

import pytest

from calibrationtools.particle import Particle
from calibrationtools.particle_reader import (
    ParticleReader,
    flatten_dict,
    unflatten_parameter_name,
)


@pytest.fixture
def dense_dict() -> dict[str, Any]:
    return {
        "param1.variant.component": 3.0,
        "param2.component": 7.0,
    }


@pytest.fixture
def nested_dict() -> dict[str, Any]:
    return {
        "param1": {"variant": {"component": 3.0}},
        "param2": {"component": 7.0},
    }


@pytest.fixture
def default_param_dict() -> dict[str, Any]:
    return {
        "model_inputs": {
            "model_name.global_params": {
                "param1": {
                    "unused_variant": {"component": 2.0},
                    "variant": {"component": 2.0},
                },
                "param2": {"component": 1.0},
                "default_param": 18.0,
                "seed": 123,
            }
        }
    }


@pytest.fixture
def default_param_dict_split_params() -> dict[str, Any]:
    return {
        "model_inputs": {
            "model_name.global_params": {
                "param1": {
                    "unused_variant": {"component": 2.0},
                    "variant": {"component": 2.0},
                },
                "default_param": 18.0,
                "seed": 123,
            }
        },
        "different_model": {
            "param2": {"component": 1.0},
        },
    }


@pytest.fixture
def dense_particle(dense_dict) -> Particle:
    return Particle(dense_dict)


def test_unflatten_parameter_name():
    hierarchical = unflatten_parameter_name("param.variant.component", 1.0)
    single = unflatten_parameter_name("param", 1.0)

    assert hierarchical == {"param": {"variant": {"component": 1.0}}}
    assert single == {"param": 1.0}


def test_flatten_dict(default_param_dict):
    flat = flatten_dict(default_param_dict)
    expected_flat = {
        "model_inputs.model_name\\.global_params.param1.unused_variant.component": 2.0,
        "model_inputs.model_name\\.global_params.param1.variant.component": 2.0,
        "model_inputs.model_name\\.global_params.param2.component": 1.0,
        "model_inputs.model_name\\.global_params.default_param": 18.0,
        "model_inputs.model_name\\.global_params.seed": 123,
    }
    assert flat == expected_flat


def test_particle_init(dense_particle, default_param_dict):
    all_names = list(dense_particle.keys())

    # Exclude the extra component in the keys
    reader = ParticleReader(
        particle_param_names=all_names,
        default_params=default_param_dict,
    )

    expected_name_key = {
        "param1.variant.component": "model_inputs.model_name\\.global_params.param1.variant.component",
        "param2.component": "model_inputs.model_name\\.global_params.param2.component",
    }
    assert reader.name_key == expected_name_key


def test_particle_init_no_match(dense_particle, default_param_dict):
    all_names = list(dense_particle.keys())
    with pytest.raises(
        ValueError,
        match="No matching default parameter found for 'param1.variant.extra_component'",
    ):
        ParticleReader(
            particle_param_names=all_names
            + ["param1.variant.extra_component"],
            default_params=default_param_dict,
        )


def test_particle_init_multiple_matches(dense_particle, default_param_dict):
    all_names = list(dense_particle.keys())
    with pytest.raises(
        ValueError,
        match="Multiple matching default parameters found for 'component'",
    ):
        ParticleReader(
            particle_param_names=all_names + ["component"],
            default_params=default_param_dict,
        )


def test_particle_init_no_defaults(dense_particle, default_param_dict):
    all_names = list(dense_particle.keys())
    reader = ParticleReader(
        particle_param_names=all_names,
        default_params=None,
    )
    assert all(
        reader.name_key[param_name] == param_name for param_name in all_names
    )
    assert len(list(reader.name_key.keys())) == len(all_names)


def test_merge_particle_with_defaults(dense_particle, default_param_dict):
    all_names = list(dense_particle.keys())
    reader = ParticleReader(
        particle_param_names=all_names,
        default_params=default_param_dict,
    )
    merged = reader._merge_particle_with_defaults(dense_particle)
    print(merged)
    expected_flattened = {
        "model_inputs.model_name\\.global_params.param1.unused_variant.component": 2.0,
        "model_inputs.model_name\\.global_params.param1.variant.component": 3.0,
        "model_inputs.model_name\\.global_params.param2.component": 7.0,
        "model_inputs.model_name\\.global_params.default_param": 18.0,
        "model_inputs.model_name\\.global_params.seed": 123,
    }
    assert flatten_dict(merged) == expected_flattened
    assert "model_inputs" in merged
    assert "model_name.global_params" in merged["model_inputs"]
    assert reader.default_params != merged


def test_merge_particles_with_defaults_light(particle):
    light_defaults = {"x": 1.5, "y": 2.5, "match_x": 1.0}
    reader = ParticleReader(
        ["x", "y"],
        default_params=light_defaults,
    )
    merged = reader._merge_particle_with_defaults(particle)
    expected_merged = {"x": 1.0, "y": 2.0, "match_x": 1.0}
    assert merged == expected_merged


def test_merge_particles_with_defaults_light_structure(particle):
    light_defaults = {"x": 1.5, "y": 2.5, "match_x": 1.0}
    reader = ParticleReader(
        ["x", "y"],
        default_params={"model_header": light_defaults},
    )
    merged = reader._merge_particle_with_defaults(particle)
    print(merged)
    expected_merged = {"model_header": {"x": 1.0, "y": 2.0, "match_x": 1.0}}
    assert merged == expected_merged


def test_merge_particle_without_defaults(dense_particle, nested_dict):
    all_names = list(dense_particle.keys())
    reader = ParticleReader(
        particle_param_names=all_names,
        default_params=None,
    )
    merged = reader._merge_particle_with_defaults(dense_particle)
    assert merged == nested_dict


def test_read_particle(dense_particle, default_param_dict):
    all_names = list(dense_particle.keys())
    reader = ParticleReader(
        particle_param_names=all_names,
        default_params=default_param_dict,
    )
    read = reader.read_particle(dense_particle)

    expected_read = {
        "model_inputs": {
            "model_name.global_params": {
                "param1": {
                    "unused_variant": {"component": 2.0},
                    "variant": {"component": 3.0},
                },
                "param2": {"component": 7.0},
                "default_param": 18.0,
                "seed": 123,
            }
        }
    }
    assert read == expected_read
    assert reader.default_params != read


def test_read_particle_split_defaults(
    dense_particle, default_param_dict_split_params
):
    all_names = list(dense_particle.keys())
    reader = ParticleReader(
        particle_param_names=all_names,
        default_params=default_param_dict_split_params,
    )
    read = reader.read_particle(dense_particle)

    expected_read = {
        "model_inputs": {
            "model_name.global_params": {
                "param1": {
                    "unused_variant": {"component": 2.0},
                    "variant": {"component": 3.0},
                },
                "default_param": 18.0,
                "seed": 123,
            }
        },
        "different_model": {
            "param2": {"component": 7.0},
        },
    }
    assert read == expected_read
    assert reader.default_params != read


def test_read_particle_no_defaults(dense_particle, nested_dict):
    all_names = list(dense_particle.keys())
    reader = ParticleReader(
        particle_param_names=all_names,
        default_params=None,
    )
    read = reader.read_particle(dense_particle)
    assert read == nested_dict


def test_read_particle_with_callable(dense_particle):
    all_names = list(dense_particle.keys())

    def my_read_fn(particle: Particle) -> dict:
        return {"custom": dict(particle)}
    
    reader = ParticleReader(
        particle_param_names=all_names,
        default_params=None,
        read_fn=my_read_fn,
    )

    read = reader.read_particle(dense_particle)
    assert read == {"custom": dict(dense_particle)}


def test_read_particle_ignores_defaults_with_callable(
    dense_particle, default_param_dict
):

    def my_read_fn(particle: Particle) -> dict:
        return {"custom": dict(particle)}
    
    all_names = list(dense_particle.keys())
    reader = ParticleReader(
        particle_param_names=all_names,
        default_params=default_param_dict,
        read_fn=my_read_fn
    )

    read = reader.read_particle(dense_particle)
    assert read == {"custom": dict(dense_particle)}


def test_read_particle_callable_collects_defaults(
    dense_particle, default_param_dict
):
    def my_read_fn(particle: Particle, default_params: dict[str, Any]) -> dict:
        return {"defaults": default_params, "particle": dict(particle)}

    all_names = list(dense_particle.keys())
    reader = ParticleReader(
        particle_param_names=all_names,
        default_params=default_param_dict,
        read_fn=my_read_fn
    )

    read = reader.read_particle(dense_particle)
    assert read["defaults"] == default_param_dict
    assert read["particle"] == dict(dense_particle)
