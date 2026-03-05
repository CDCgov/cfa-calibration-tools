import json

import jsonschema
import pytest

from calibrationtools.load_priors import (
    independent_priors_from_dict,
    load_priors_from_json,
    validate_schema,
)
from calibrationtools.prior_distribution import (
    BetaPrior,
    ExponentialPrior,
    GammaPrior,
    IndependentPriors,
    LogNormalPrior,
    NormalPrior,
    SeedPrior,
    UniformPrior,
)


@pytest.fixture
def good_schema():
    return {
        "priors": {
            "param1": {
                "distribution": "uniform",
                "parameters": {"min": 0, "max": 1},
            },
            "param2": {
                "distribution": "uniform",
                "parameters": {"min": 0, "max": 1},
            },
            "param3": {
                "distribution": "normal",
                "parameters": {"mean": 0, "std_dev": 1},
            },
            "param4": {
                "distribution": "lognormal",
                "parameters": {"mean": 0, "std_dev": 1},
            },
            "param5": {
                "distribution": "exponential",
                "parameters": {"rate": 1},
            },
            "param6": {
                "distribution": "gamma",
                "parameters": {"shape": 2, "scale": 3},
            },
            "param7": {
                "distribution": "beta",
                "parameters": {"alpha": 2, "beta": 5},
            },
        }
    }


def test_independent_priors_from_dict(good_schema):
    priors = independent_priors_from_dict(
        good_schema, incl_seed_parameter=True, seed_parameter_name="seed"
    )
    assert isinstance(priors, IndependentPriors)
    assert len(priors.priors) == 8
    assert any(
        isinstance(p, UniformPrior) and p.param == "param1"
        for p in priors.priors
    )
    assert any(
        isinstance(p, UniformPrior) and p.param == "param2"
        for p in priors.priors
    )
    assert any(
        isinstance(p, NormalPrior) and p.param == "param3"
        for p in priors.priors
    )
    assert any(
        isinstance(p, LogNormalPrior) and p.param == "param4"
        for p in priors.priors
    )
    assert any(
        isinstance(p, ExponentialPrior) and p.param == "param5"
        for p in priors.priors
    )
    assert any(
        isinstance(p, SeedPrior) and p.param == "seed" for p in priors.priors
    )
    assert any(
        isinstance(p, GammaPrior) and p.param == "param6"
        for p in priors.priors
    )
    assert any(
        isinstance(p, BetaPrior) and p.param == "param7" for p in priors.priors
    )


def test_independent_priors_from_dict_no_seed(good_schema):
    priors = independent_priors_from_dict(
        good_schema, incl_seed_parameter=False
    )
    assert isinstance(priors, IndependentPriors)
    assert len(priors.priors) == 7
    assert not any(isinstance(p, SeedPrior) for p in priors.priors)


def test_independent_priors_from_dict_custom_seed_name(good_schema):
    priors = independent_priors_from_dict(
        good_schema,
        incl_seed_parameter=True,
        seed_parameter_name="custom_seed",
    )
    assert isinstance(priors, IndependentPriors)
    assert len(priors.priors) == 8
    assert any(
        isinstance(p, SeedPrior) and p.param == "custom_seed"
        for p in priors.priors
    )


def test_independent_priors_from_dict_nonexhaustive(good_schema):
    incomplete_schema = good_schema.copy()
    del incomplete_schema["priors"]["param3"]
    del incomplete_schema["priors"]["param4"]
    del incomplete_schema["priors"]["param6"]
    del incomplete_schema["priors"]["param7"]

    priors = independent_priors_from_dict(
        incomplete_schema, incl_seed_parameter=False
    )
    assert isinstance(priors, IndependentPriors)
    assert len(priors.priors) == 3
    assert any(
        isinstance(p, UniformPrior) and p.param == "param1"
        for p in priors.priors
    )
    assert any(
        isinstance(p, UniformPrior) and p.param == "param2"
        for p in priors.priors
    )
    assert any(
        isinstance(p, ExponentialPrior) and p.param == "param5"
        for p in priors.priors
    )


def test_invalid_schema_unknown_distribution():
    bad_schema = {
        "priors": {
            "param1": {"distribution": "unknown_dist", "parameters": {}}
        }
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_schema(bad_schema)


def test_valid_schema_bad_uniform_min_max():
    bad_schema = {
        "priors": {
            "param1": {
                "distribution": "uniform",
                "parameters": {"min": 1, "max": 0},
            }
        }
    }
    with pytest.raises(
        AssertionError,
        match="UniformPrior min must be less than max for parameter param1",
    ):
        independent_priors_from_dict(bad_schema)


def test_invalid_schema_normal_std_dev():
    bad_schema = {
        "priors": {
            "param3": {
                "distribution": "normal",
                "parameters": {"mean": 0, "std_dev": -1},
            }
        }
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_schema(bad_schema)


def test_invalid_schema_exponential_rate():
    bad_schema = {
        "priors": {
            "param5": {
                "distribution": "exponential",
                "parameters": {"rate": -1},
            }
        }
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_schema(bad_schema)


def test_invalid_schema_gamma_parameters():
    bad_schema = {
        "priors": {
            "param6": {
                "distribution": "gamma",
                "parameters": {"shape": -1, "scale": 3},
            }
        }
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_schema(bad_schema)


def test_invalid_schema_gamma_uses_rate_parameter():
    bad_schema = {
        "priors": {
            "param6": {
                "distribution": "gamma",
                "parameters": {"shape": 2, "rate": 3},
            }
        }
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_schema(bad_schema)


def test_invalid_schema_beta_parameters():
    bad_schema = {
        "priors": {
            "param7": {
                "distribution": "beta",
                "parameters": {"alpha": -1, "beta": 5},
            }
        }
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_schema(bad_schema)


def test_invalid_schema_missing_parameters():
    bad_schema = {
        "priors": {
            "param1": {"distribution": "uniform", "parameters": {"min": 0}}
        }
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_schema(bad_schema)


def test_invalid_schema_additional_parameters():
    bad_schema = {
        "priors": {
            "param1": {
                "distribution": "uniform",
                "parameters": {"min": 0, "max": 1, "extra": 0},
            }
        }
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_schema(bad_schema)


def test_load_priors_from_json(tmp_path, good_schema):
    json_file = tmp_path / "priors.json"
    with open(json_file, "w") as f:
        json.dump(good_schema, f)

    priors = load_priors_from_json(
        json_file, incl_seed_parameter=True, seed_parameter_name="seed"
    )
    assert isinstance(priors, IndependentPriors)
    assert len(priors.priors) == 8
    assert any(
        isinstance(p, UniformPrior) and p.param == "param1"
        for p in priors.priors
    )
    assert any(
        isinstance(p, UniformPrior) and p.param == "param2"
        for p in priors.priors
    )
    assert any(
        isinstance(p, NormalPrior) and p.param == "param3"
        for p in priors.priors
    )
    assert any(
        isinstance(p, LogNormalPrior) and p.param == "param4"
        for p in priors.priors
    )
    assert any(
        isinstance(p, ExponentialPrior) and p.param == "param5"
        for p in priors.priors
    )
    assert any(
        isinstance(p, SeedPrior) and p.param == "seed" for p in priors.priors
    )
    assert any(
        isinstance(p, GammaPrior) and p.param == "param6"
        for p in priors.priors
    )
    assert any(
        isinstance(p, BetaPrior) and p.param == "param7" for p in priors.priors
    )
