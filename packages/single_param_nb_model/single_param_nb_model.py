from typing import Any

from numpy.random import default_rng
from mrp import RunnerContext


def main():
    ctx = RunnerContext.from_stdin()
    result = NB_n_Model(ctx.input)
    ctx.write_csv("incidence.csv", result)


if __name__ == "__main__":
    main()


def NB_n_Model(model_inputs: dict[str, Any]) -> list[int]:
    """
    Galton-Watson branching process model with binomial reproduction.

    Simulates population growth where each individual reproduces independently
    according to a binomial distribution. Starting with 1 individual, the model
    runs for multiple generations until reaching max_gen or max_infect threshold.

    Args:
        model_inputs: Dictionary containing model parameters:
            seed (int, optional): Random seed for reproducibility. If not provided, uses random seed.
            max_gen (int): Maximum number of generations to simulate
            n (int): Number of trials in binomial distribution (max offspring per individual)
            p (float): Success probability in binomial distribution (0 to 1)
            max_infect (int): Population threshold to stop simulation early

    Returns:
        list[int]: Population size at each generation, starting from generation 0

    Examples:
        >>> inputs = {"seed": 123, "max_gen": 15, "n": 3, "p": 0.5, "max_infect": 500}
        >>> result = NB_n_Model(inputs)
        >>> len(result)  # Number of generations simulated
    """
    seed = model_inputs.get("seed", None)
    rng = default_rng(seed)
    out: list[int] = []
    for i in range(model_inputs["max_gen"]):
        if i == 0:
            out.append(1)
        else:
            next = int(
                sum(rng.binomial(model_inputs["n"], model_inputs["p"], out[i - 1]))
            )
            out.append(next)
            if next > model_inputs["max_infect"]:
                break
    return out
