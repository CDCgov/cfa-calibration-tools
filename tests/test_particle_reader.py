import copy
from typing import Any

import pytest

from calibrationtools.particle import Particle
from calibrationtools.particle_reader import (
    default_particle_reader,
    unflatten_parameter_name,
    unflatten_particle,
)


@pytest.fixture
def dense_dict() -> dict[str, Any]:
    return {
        "param1.variant.component": 3.0,
        "param1.variant.extra_component": 5.0,
        "param2.component": 7.0,
    }


@pytest.fixture
def default_param_dict() -> dict[str, Any]:
    return {
        "model_inputs": {
            "model_name.global_params": {
                "param1": {"unused_variant": {"component": 2.0}},
                "param2": {"component": 1.0},
                "default_param": 18.0,
                "seed": 123,
            }
        }
    }


@pytest.fixture
def dense_particle(dense_dict) -> Particle:
    return Particle(dense_dict)


def test_unflatten_parameter_name():
    hierarchical = unflatten_parameter_name("param.variant.component", 1.0)
    single = unflatten_parameter_name("param", 1.0)

    assert hierarchical == {"param": {"variant": {"component": 1.0}}}
    assert single == {"param": 1.0}


def test_unflatten_particle(dense_particle):
    param_dict = unflatten_particle(dense_particle)
    expected = {
        "param1": {"variant": {"component": 3.0, "extra_component": 5.0}},
        "param2": {"component": 7.0},
    }
    assert param_dict == expected

    param_dict_with_header = unflatten_particle(
        dense_particle, parameter_headers=["header1", "header2"]
    )
    assert param_dict_with_header == {"header1": {"header2": expected}}


def test_default_particle_reader(dense_particle, default_param_dict):
    header = ["model_inputs", "model_name.global_params"]
    model_params = default_particle_reader(
        dense_particle,
        default_params=default_param_dict,
        parameter_headers=header,
    )

    # Parameter headers should allow for '.' in their names
    expected = {
        "model_inputs": {
            "model_name.global_params": {
                "param1": {
                    "variant": {"component": 3.0, "extra_component": 5.0},
                    "unused_variant": {"component": 2.0},
                },
                "param2": {"component": 7.0},
                "default_param": 18.0,
                "seed": 123,
            }
        }
    }

    assert model_params == expected

    # Fails to yield expected results without header
    no_header_params = copy.deepcopy(default_param_dict)
    no_header_params.update(unflatten_particle(dense_particle))
    assert model_params != default_param_dict

    # Fails to yield expected results with header and using just update
    updated_params = copy.deepcopy(default_param_dict)
    updated_params.update(
        unflatten_particle(dense_particle, parameter_headers=header)
    )
    assert model_params != updated_params


def test_default_particle_reader_no_header(dense_particle, default_param_dict):
    header = []
    model_params = default_particle_reader(
        dense_particle,
        default_params=default_param_dict,
        parameter_headers=header,
    )

    # Without the matching headers, the particle params are appended instead of merged
    default_param_dict.update(unflatten_particle(dense_particle))
    assert model_params == default_param_dict
