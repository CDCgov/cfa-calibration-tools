from typing import Any

from mrp import Environment, MRPModel
from numpy.random import default_rng


class Binom_BP_Model(MRPModel):
    def run(self):
        print("Running Binomial Branching Process Model...")
        results = self.simulate(self.input)
        self.write_csv(
            "output.csv",
            {
                "generation": list(range(len(results))),
                "population": results,
            },
        )

    @staticmethod
    def simulate(model_inputs: dict[str, Any]) -> list[int]:
        """
        Galton-Watson branching process model with binomial reproduction.

        Simulates population growth where each individual reproduces independently
        according to a binomial distribution. Starting with 1 individual, the model
        runs for multiple generations until reaching max_gen or max_infect threshold.

        Args:
            model_inputs (dict[str, Any]): Dictionary containing model
                parameters: seed, max_gen, n, p, and max_infect.

        Returns:
            list[int]: Population size at each generation, starting from generation 0

        Examples:
            >>> inputs = {"seed": 123, "max_gen": 15, "n": 3, "p": 0.5, "max_infect": 500}
            >>> model = Binom_BP_Model(inputs)
            >>> model.run()
            >>> len(model.simulate(inputs))  # Number of generations simulated
        """
        seed = model_inputs.get("seed", None)
        rng = default_rng(seed)
        out: list[int] = []
        for i in range(model_inputs["max_gen"]):
            if i == 0:
                out.append(1)
            else:
                next = int(
                    sum(
                        rng.binomial(
                            model_inputs["n"], model_inputs["p"], out[i - 1]
                        )
                    )
                )
                out.append(next)
                if next > model_inputs["max_infect"]:
                    break
        return out


def main():
    Binom_BP_Model().run()


def run_inline(run_json: dict[str, Any]) -> None:
    Binom_BP_Model(env=Environment(run_json)).run()
