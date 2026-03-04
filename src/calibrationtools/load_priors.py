import json
import jsonschema
import importlib.resources

from .prior_distribution import IndependentPriors, SeedPrior, UniformPrior, NormalPrior, LogNormalPrior, ExponentialPrior

def load_schema() -> dict:
    with importlib.resources.open_text("assets", "schema.json") as f:
        schema = json.load(f)
    return schema

def independent_priors_from_dict(priors_dict: dict[str, dict], incl_seed_parameter: bool = True, seed_parameter_name: str | None = 'seed') -> IndependentPriors:
    jsonschema.validate(instance=priors_dict, schema=load_schema())
    priors = []

    for k, v in priors_dict.items():
        distribution = list(v.keys())[0]
        match distribution:
            case "uniform":
                priors.append(UniformPrior(k, min=v[distribution]["min"], max=v[distribution]["max"]))
            case "normal":
                priors.append(NormalPrior(k, mean=v[distribution]["mean"], std_dev=v[distribution]["sigma"]))
            case "lognormal":
                priors.append(LogNormalPrior(k, mean=v[distribution]["mean"], std_dev=v[distribution]["sigma"]))
            case "exponential":
                priors.append(ExponentialPrior(k, rate=v[distribution]["rate"]))

    if incl_seed_parameter and seed_parameter_name is not None:
        priors.append(SeedPrior(seed_parameter_name))

    return IndependentPriors(priors)

def load_priors_from_json(
        json_file: str, 
        incl_seed_parameter: bool = True, 
        seed_parameter_name: str | None = 'seed'
    ) -> IndependentPriors:
    with open(json_file, 'r') as f:
        priors_dict = json.load(f)
    return independent_priors_from_dict(
        priors_dict["priors"], 
        incl_seed_parameter=incl_seed_parameter, 
        seed_parameter_name=seed_parameter_name
    )