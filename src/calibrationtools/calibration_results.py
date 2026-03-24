from typing import Any

from numpy.random import SeedSequence

from .particle import Particle
from .particle_population import ParticlePopulation
from .particle_population_metrics import ParticlePopulationMetrics
from .particle_updater import _ParticleUpdater
from .prior_distribution import nonseed_param_names


class CalibrationResults:
    """
    Stores the results of the calibration process, including the posterior particle population, population archive, success counts, and other relevant information.

     Args:
        _updater (_ParticleUpdater):The particle population updater availble from a fitted sampler object, which contains the final particle population and the perturbation kernel used for sampling particles in the final generation.
        entropy_history (dict[int, list[dict[int, int | SeedSequence]]]): A dictionary mapping generation indices to their corresponding lists of dictionaries containing particle IDs and their associated seed sequences, representing the history of particle sampling and perturbation across generations when called with the appropriate particle updater.
        population_archive (dict[int, ParticlePopulation]): A dictionary mapping generation indices to their corresponding particle populations, representing the history of particle populations across generations if saved during the sampler run.
        success_counts (dict[str, list[int]]): A dictionary containing lists of particles per generation, success counts, and attempt counts for each generation, with keys "generation_particle_count", "successes" and "attempts".
        tolerance_values (list[float]): A list of tolerance values for each generation
    Methods:
        fitted_params() -> list[str]: Returns a list of parameter names that were fitted during the calibration process, excluding any seed parameters.
        posterior_particles() -> ParticlePopulation: Returns the particle population representing the posterior distribution after the final generation of the calibration process.
        ess() -> float: Returns the effective sample size (ESS) of the posterior particle population, which is a measure of the diversity of the particles and the quality of the approximation to the posterior distribution.
        credible_intervals() -> dict[str, tuple[float, float]]: Returns a dictionary mapping parameter names to their corresponding credible intervals, which are calculated based on the quantiles of the posterior particle population for the fitted parameters.
        point_estimates() -> dict[str, float]: Returns a dictionary mapping parameter names to their corresponding point estimates, which are calculated as the weighted average of the parameter values in the posterior particle population for the fitted parameters.
        acceptance_rates() -> list[float]: Returns a list of acceptance rates for each generation, calculated as the ratio of successful particles to attempted particles for each generation.
        sample_posterior_particles(n: int = 1, perturb: bool = False) -> list[Particle]: Samples particles from the posterior distribution, with an option to apply perturbation to the sampled particles using the perturbation kernel from the particle updater.
        get_diagnostics() -> dict[str, Any]: Runs diagnostics on the calibration results, including ESS values across generations, credible intervals, point estimates, acceptance rates, quantiles, posterior weights, covariance matrix, and correlation matrix for the fitted parameters.
    """

    def __init__(
        self,
        _updater: _ParticleUpdater,
        entropy_history: dict[int, list[dict[int, int | SeedSequence]]],
        population_archive: dict[int, ParticlePopulation],
        success_counts: dict[str, list[int]],
        tolerance_values: list[float],
    ):
        self._updater = _updater
        self.posterior = ParticlePopulationMetrics(
            self._updater.particle_population
        )
        self.entropy_history = entropy_history
        self.population_archive = population_archive
        self.generation_particle_count = success_counts[
            "generation_particle_count"
        ]
        self.smc_step_successes = success_counts["successes"]
        self.smc_step_attempts = success_counts["attempts"]
        self.tolerance_values = tolerance_values
        self.priors = _updater.priors

        self.aggregate_acceptance_rate = (
            sum(self.smc_step_successes) / sum(self.smc_step_attempts)
            if sum(self.smc_step_attempts) > 0
            else 0
        )

        self._validate()

    @property
    def fitted_params(self) -> list[str]:
        return nonseed_param_names(self.priors)

    @property
    def posterior_particles(self) -> ParticlePopulation:
        return self.posterior.particle_population

    @property
    def ess(self) -> float:
        return self.posterior.particle_population.ess

    @property
    def credible_intervals(self) -> dict[str, tuple[float, float]]:
        return self.posterior.get_credible_intervals(params=self.fitted_params)

    @property
    def point_estimates(self) -> dict[str, float]:
        return self.posterior.get_point_estimates(params=self.fitted_params)

    @property
    def acceptance_rates(self) -> list[float]:
        return [
            successes / attempts if attempts > 0 else 0
            for successes, attempts in zip(
                self.smc_step_successes, self.smc_step_attempts
            )
        ]

    def __repr__(self) -> str:
        return (
            f"CalibrationResults(ESS={self.ess:.2f}, with size {self.generation_particle_count[-1]})\n"
            f"Acceptance rate [step {len(self.acceptance_rates)}={self.acceptance_rates[-1]:.2f}, overall={self.aggregate_acceptance_rate:.2f}]\n"
            f"Posterior point estimates: {[f'{k}: {v:.2f}' for k, v in self.point_estimates.items()]}\n"
        )

    def _validate(self):
        """
        Validates the calibration results to ensure consistency and correctness of the data.
        Raises:
            ValueError:
                If posterior particle population is not normalized
                If the length of success counts does not match the length of tolerance values
                If the final step successes do not match the generation particle count
                If the final step successes do not match the particle population size
        """

        if (self.posterior.particle_population.total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Particle population weights should already be normalized to 1.0, instead got {self.posterior.particle_population.total_weight}"
            )

        if len(self.smc_step_successes) != len(self.tolerance_values):
            raise ValueError(
                "Length of success counts should match length of tolerance values"
            )
        if self.smc_step_successes[-1] != self.generation_particle_count[-1]:
            raise ValueError(
                "Step successes should match generation particle count for the final generation"
            )
        if (
            self.smc_step_successes[-1]
            != self.posterior.particle_population.size
        ):
            raise ValueError(
                "Step successes should match particle population size for the final generation"
            )

    def sample_posterior_particles(
        self, n: int = 1, perturb: bool = False
    ) -> list[Particle]:
        """
        Samples particles from the posterior distribution.

        Args:
            n (int): The number of particles to sample.
            perturb (bool): Whether to apply perturbation to the sampled particles.

        Returns:
            list[Particle]: A list of sampled particles.
        """
        if perturb:
            return [
                self._updater.sample_and_perturb_particle() for _ in range(n)
            ]
        else:
            return [self._updater.sample_particle() for _ in range(n)]

    def get_diagnostics(self) -> dict[str, Any]:
        """
        Runs diagnostics on the calibration results
        """
        ess_vals = [
            p.ess for p in self.population_archive.values() if not p.is_empty()
        ]
        ess_vals.append(self.ess)

        return {
            "ess_values": ess_vals,
            "credible_intervals": self.credible_intervals,
            "point_estimates": self.point_estimates,
            "acceptance_rates": self.acceptance_rates,
            "quantiles": self.posterior.get_quantiles(
                params=self.fitted_params
            ),
            "posterior_weights": self.posterior.particle_population.weights,
            "covariance_matrix": self.posterior.get_covariance_matrix(
                self.fitted_params
            ),
            "correlation_matrix": self.posterior.get_correlation_matrix(
                self.fitted_params
            ),
        }
