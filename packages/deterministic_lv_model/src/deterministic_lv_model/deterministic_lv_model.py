from typing import Any

import numpy as np
from mrp import MRPModel
from numpy.random import default_rng
from scipy.integrate import solve_ivp


class Deterministic_LV_Model(MRPModel):
    def run(self):
        results = self.simulate(self.input)
        self.write_csv(
            "output.csv",
            {
                "time": results["times"],
                "x_obs": results["observed_x"],
                "y_obs": results["observed_y"],
            },
        )

    @staticmethod
    def lotka_volterra_rhs(
        _t: float, state: np.ndarray, a: float, b: float
    ) -> tuple[float, float]:
        x, y = state
        return (a * x - x * y, b * x * y - y)

    @staticmethod
    def simulate(model_inputs: dict[str, Any]) -> dict[str, list[float]]:
        """
        Deterministic Lotka-Volterra model.

        Simulates predator-prey dynamics using a system of differential equations.
        The model tracks the population of prey, x, and predators, y, over time, given initial conditions and parameters.

        The model consists of the following two differential equations:
        dx/dt = ax - xy
        dy/dt = bxy - y


        While the mechanistic model is deterministic, this simulation includes a stochastic observation process model that adds Gaussian noise to the population sizes.
        This simulation is intended to replicate the findings of Toni et al. 2009 (doi: 10.1098/rsif.2008.0172).

        Args:
            model_inputs: Dictionary containing model parameters:
                seed (int, optional): Random seed for reproducibility. If not provided, uses random seed.
                max_time (float): Maximum duration to run the simulation (in arbitrary time units)
                a (float): Maximum growth rate of prey
                b (float): Coefficient for effect of prey availability on growth rate of predators
                x0 (float): Initial prey population
                y0 (float): Initial predator population
                obs_times (list[float]): Time points at which to record population sizes
                obs_noise_mu (float): Mean of Gaussian noise added to population sizes
                obs_noise_sigma (float): Standard deviation of Gaussian noise added to population sizes
                stop_x_threshold (float, optional): Terminal event threshold for x runaway. Disabled if omitted.

        Returns:
            dict[str, list[float]]: Dictionary containing noisy prey and predator populations at observation time points.

        Examples:
            >>> inputs = {
            ...     "seed": 123,
            ...     "max_time": 15,
            ...     "a": 1.0,
            ...     "b": 1.0,
            ...     "x0": 1.0,
            ...     "y0": 0.5,
            ...     "obs_times": [1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4],
            ...     "obs_noise_mu": 0.0,
            ...     "obs_noise_sigma": 0.5,
            ... }
            >>> model = Deterministic_LV_Model(inputs)
            >>> model.run()
            >>> len(model.simulate(inputs)["times"])  # Number of observation time points
        """
        seed = model_inputs.get("seed", None)
        rng = default_rng(seed)
        obs_times = np.asarray(model_inputs["obs_times"], dtype=float)
        obs_noise_mu = float(model_inputs["obs_noise_mu"])
        obs_noise_sigma = float(model_inputs["obs_noise_sigma"])
        a = float(model_inputs["a"])
        b = float(model_inputs["b"])
        max_time = float(model_inputs["max_time"])
        stop_x_threshold = model_inputs.get("stop_x_threshold")

        #        print(f"Running simulation with parameters: a={a}, b={b}, seed={seed}")

        solver_method = str(
            model_inputs.get(
                "solver_method",
                "LSODA" if (a > 0.0 and b < 0.0) else "RK45",
            )
        )

        if obs_times.ndim != 1 or obs_times.size == 0:
            raise ValueError("obs_times must be a non-empty 1D array of times")
        if np.any(obs_times < 0) or np.any(obs_times > max_time):
            raise ValueError("obs_times values must lie within [0, max_time]")
        if np.any(np.diff(obs_times) < 0):
            raise ValueError(
                "obs_times must be sorted in non-decreasing order"
            )

        event_functions: list = []
        if stop_x_threshold is not None:
            stop_x_value = float(stop_x_threshold)

            def stop_x_event(
                _t: float, state: np.ndarray, _a: float, _b: float
            ) -> float:
                return state[0] - stop_x_value

            stop_x_event.terminal = True  # type: ignore[attr-defined]
            stop_x_event.direction = 1.0  # type: ignore[attr-defined]
            event_functions.append(stop_x_event)

        solution = solve_ivp(
            fun=Deterministic_LV_Model.lotka_volterra_rhs,
            t_span=(0.0, max_time),
            y0=np.array([model_inputs["x0"], model_inputs["y0"]], dtype=float),
            t_eval=obs_times,
            args=(a, b),
            method=solver_method,
            events=event_functions if event_functions else None,
        )

        if not solution.success:
            raise RuntimeError(f"ODE solve failed: {solution.message}")

        solved_count = solution.y.shape[1]
        requested_count = len(obs_times)
        if solved_count == 0:
            raise RuntimeError("ODE solve produced no states.")

        # The ODE solver may terminate early due to extreme values of x
        # If so, we need to fill in values for the missing time points.
        if solved_count < requested_count:
            x_values = np.empty(requested_count, dtype=float)
            y_values = np.empty(requested_count, dtype=float)
            x_values[:solved_count] = solution.y[0]
            y_values[:solved_count] = solution.y[1]
            x_values[solved_count:] = solution.y[0, solved_count - 1]
            y_values[solved_count:] = solution.y[1, solved_count - 1]
        else:
            x_values = solution.y[0]
            y_values = solution.y[1]

        observed_x = x_values + rng.normal(
            loc=obs_noise_mu,
            scale=obs_noise_sigma,
            size=len(obs_times),
        )

        observed_y = y_values + rng.normal(
            loc=obs_noise_mu,
            scale=obs_noise_sigma,
            size=len(obs_times),
        )

        return {
            "times": obs_times.tolist(),
            "observed_x": observed_x.tolist(),
            "observed_y": observed_y.tolist(),
        }


def main():
    Deterministic_LV_Model().run()
