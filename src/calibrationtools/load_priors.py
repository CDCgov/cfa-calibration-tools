import importlib.resources
import json
from pathlib import Path

import jsonschema

from .prior_distribution import (
    BetaPrior,
    ExponentialPrior,
    GammaPrior,
    IndependentPriors,
    LogNormalPrior,
    NormalPrior,
    SeedPrior,
    UniformPrior,
)


def load_schema() -> dict:
    """Load the JSON schema for validating priors from the package resources."""
    with (
        importlib.resources.files("calibrationtools.assets")
        .joinpath("schema.json")
        .open("r", encoding="utf-8")
    ) as f:
        schema = json.load(f)
    return schema


def validate_schema(priors_dict: dict[str, dict]) -> None:
    """Validate the given priors dictionary against the JSON schema."""
    jsonschema.validate(instance=priors_dict, schema=load_schema())


def independent_priors_from_dict(
    priors_dict: dict[str, dict],
    incl_seed_parameter: bool = True,
    seed_parameter_name: str | None = "seed",
) -> IndependentPriors:
    """
    Convert a dictionary of priors into an IndependentPriors object.
    Args:
        priors_dict (dict[str, dict]): A dictionary stored under the key "priors" where keys are parameter names and values are dictionaries with 'distribution' and 'parameters'.
        incl_seed_parameter (bool): Whether to include a SeedPrior for random seed control.
        seed_parameter_name (str | None): The name of the seed parameter if incl_seed_parameter is True.
    Returns:
        IndependentPriors: An IndependentPriors object containing the specified priors.
    Raises:
        ValueError: If any distribution type is unsupported or if required parameters for a distribution are missing
    """
    validate_schema(priors_dict)
    priors = []

    for k, param_dict in priors_dict["priors"].items():
        distribution = param_dict["distribution"]
        parameters = param_dict["parameters"]
        match distribution:
            case "uniform":
                assert parameters["min"] < parameters["max"], (
                    f"UniformPrior min must be less than max for parameter {k}"
                )
                priors.append(
                    UniformPrior(
                        k, min=parameters["min"], max=parameters["max"]
                    )
                )
            case "normal":
                priors.append(
                    NormalPrior(
                        k,
                        mean=parameters["mean"],
                        std_dev=parameters["std_dev"],
                    )
                )
            case "lognormal":
                priors.append(
                    LogNormalPrior(
                        k,
                        mean=parameters["mean"],
                        std_dev=parameters["std_dev"],
                    )
                )
            case "exponential":
                priors.append(ExponentialPrior(k, rate=parameters["rate"]))
            case "gamma":
                priors.append(
                    GammaPrior(
                        k,
                        shape=parameters["shape"],
                        scale=parameters["scale"],
                    )
                )
            case "beta":
                priors.append(
                    BetaPrior(
                        k,
                        alpha=parameters["alpha"],
                        beta=parameters["beta"],
                    )
                )
            case _:
                raise ValueError(
                    f"Unsupported distribution type '{distribution}' for parameter '{k}'"
                )

    if incl_seed_parameter and seed_parameter_name is not None:
        priors.append(SeedPrior(seed_parameter_name))

    return IndependentPriors(priors)


def load_priors_from_json(
    json_file: str | Path,
    incl_seed_parameter: bool = True,
    seed_parameter_name: str | None = "seed",
) -> IndependentPriors:
    """
    Load priors from a JSON file and convert them into an IndependentPriors object.
    Args:
        json_file (str | Path): The path to the JSON file containing the priors in a valid schema.
        incl_seed_parameter (bool): Whether to include a SeedPrior for random seed control.
        seed_parameter_name (str | None): The name of the seed parameter if incl_seed_parameter is True.
    Returns:
        IndependentPriors: An IndependentPriors object containing the specified priors by the file, along with a SeedPrior if included.
    """
    with open(json_file, "r") as f:
        priors_dict = json.load(f)
    return independent_priors_from_dict(
        priors_dict,
        incl_seed_parameter=incl_seed_parameter,
        seed_parameter_name=seed_parameter_name,
    )
