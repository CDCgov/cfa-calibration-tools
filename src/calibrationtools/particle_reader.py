from typing import Any

from mrp.api import apply_dict_overrides

from .particle import Particle

def unflatten_parameter_name(flattened_name: str, value: Any) -> dict[str, Any]:
    """
    Unflattens a parameter name by splitting it on dots.

    Args:
        flattened_name (str): The flattened parameter name (e.g., "offspring_distribution.NegativeBinomial.mean").
        value (Any): The value to assign to the unflattened parameter.
    Returns:
        dict[str, Any]: A dictionary representing the unflattened parameter name (e.g., {"offspring_distribution": {"NegativeBinomial": {"mean2": value}}}).
    """
    param_vec = flattened_name.split(".")
    param_dict = {param_vec[-1]: value}
    for key in reversed(param_vec[:-1]):
        param_dict = {key: param_dict}
    return param_dict

def unflatten_particle(particle: Particle, parameter_name_header: str | None = None) -> dict[str, Any]:
    """
    Parse a particle's state into a dictionary of parameter values, unflattening parameter names as needed.

    Args:
        particle (Particle): The particle whose state is to be parsed.
        parameter_name_header (str | None): An optional header to prepend to parameter names before unflattening.
    Returns:
        dict[str, Any]: A dictionary of parameter values derived from the particle's state.
    """
    particle_params = {}
    for param_name, value in particle.items():
        if parameter_name_header:
            parameter_name_header = parameter_name_header.rstrip(".")  # Remove trailing dot if present
            par_name_to_flatten = ".".join([parameter_name_header, param_name])
        else:
            par_name_to_flatten = param_name
        unflattened = unflatten_parameter_name(par_name_to_flatten, value)
        particle_params = apply_dict_overrides(particle_params, unflattened)

    return particle_params

def default_particle_reader(
        particle: Particle,
        **kwargs
    ) -> dict[str, Any]:
    """
    Convert a particle's state into a dictionary of parameter values.

    Args:
        particle (Particle): The particle whose state is to be converted.
        **kwargs:
            - default_params (dict, optional): A dictionary of default parameter values to override with the particle's state. Defaults to an empty dictionary.
            - parameter_name_header (str, optional): An optional header to prepend to parameter names before unflattening.
    Returns:
        dict[str, Any]: A dictionary of parameter values derived from the particle's state.
    """
    default_params = kwargs.get("default_params", {})
    parameter_name_header = kwargs.get("parameter_name_header")

    # Get the unflattende particle parameter dictionary
    particle_params = unflatten_particle(particle, parameter_name_header=parameter_name_header)

    # Override default parameters with unflattened particle parameter set
    model_params = apply_dict_overrides(default_params, particle_params)

    return model_params
