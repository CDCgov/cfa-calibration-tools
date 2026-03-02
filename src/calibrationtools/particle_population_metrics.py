import numpy as np

from .particle_population import ParticlePopulation


class ParticlePopulationMetrics:
    """
    A method extension class for particle population. Generates metrics for particle populations, including
    the quantiles, point estimates, covariance and correlation.

    Args:
        particle_population (ParticlePopulation): a particle population to calculate metrics from.
    """

    def __init__(self, particle_population: ParticlePopulation):
        self.particle_population = particle_population

    def _validate_params(self, params: list[str] | None) -> list[str]:
        """
        Validate the specified parameters against the particle population.
        Args:
            params (list[str] | None): A list of parameter names to validate. If None, all parameters in the particle population are considered valid.
        Returns:
            list[str]: A list of valid parameter names.
        Raises:
            ValueError: If any specified parameter is not found in the particle population.
        """
        if params is None:
            return list(self.particle_population.particles[0].keys())
        else:
            invalid_params = [
                p
                for p in params
                if p not in self.particle_population.particles[0].keys()
            ]
            if invalid_params:
                raise ValueError(
                    f"Invalid parameters specified: {invalid_params}. Valid parameters are: {self.particle_population.particles[0].keys()}"
                )
            return params

    def get_quantiles(
        self,
        quantiles: list[float] = [0.025, 0.25, 0.5, 0.75, 0.975],
        params: list[str] | None = None,
    ) -> dict[str, dict[float, float]]:
        """
        Get the specified quantiles for the given parameters in the particle population.

        Args:
            quantiles (list[float]): A list of quantiles to compute (e.g., [0.25, 0.5, 0.75]).
                Defaults to 95% credible interval, interquartile range, and median.
            params (list[str] | None): A list of parameter names to compute quantiles for.
                If None, quantiles will be computed for all parameters in the particle population.

        Returns:
            dict[str, dict[float, float]]: A dictionary where keys are parameter names and values
                are dictionaries mapping quantiles to their corresponding values.
        Raises:
            ValueError: If any quantile is not between 0 and 1.
        """
        params = self._validate_params(params)
        if not all(0 <= q <= 1 for q in quantiles):
            raise ValueError("Quantiles must be between 0 and 1")

        quantile_results = {}
        for param in params:
            param_values = [
                particle[param]
                for particle in self.particle_population.particles
            ]
            quantile_results[param] = {
                q: np.quantile(
                    param_values,
                    q,
                    weights=self.particle_population.weights,
                    method="inverted_cdf",
                )
                for q in quantiles
            }

        return quantile_results

    def get_credible_intervals(
        self,
        lower_quantile: float = 0.025,
        upper_quantile: float = 0.975,
        params: list[str] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """
        Get the credible interval for the given parameters in the particle population.

        Args:
            lower_quantile (float): The lower quantile for the credible interval (e.g., 0.025 for a 95% credible interval).
            upper_quantile (float): The upper quantile for the credible interval (e.g., 0.975 for a 95% credible interval).
            params (list[str] | None): A list of parameter names to compute credible intervals for.
                If None, credible intervals will be computed for all parameters in the particle population.

        Returns:
            dict[str, tuple[float, float]]: A dictionary mapping parameter names to their corresponding credible intervals as tuples of (lower_bound, upper_bound).
        Raises:
            ValueError: If lower_quantile or upper_quantile is not between 0 and 1, or if lower_quantile is greater than upper_quantile.
        """
        if not (0 <= lower_quantile <= 1) or not (0 <= upper_quantile <= 1):
            raise ValueError("Quantiles must be between 0 and 1")
        if lower_quantile > upper_quantile:
            raise ValueError(
                "Lower quantile must be less than or equal to upper quantile"
            )

        params = self._validate_params(params)
        credible_intervals = {}
        for param in params:
            param_values = [
                particle[param]
                for particle in self.particle_population.particles
            ]
            lower_bound = np.quantile(
                param_values,
                lower_quantile,
                weights=self.particle_population.weights,
                method="inverted_cdf",
            )
            upper_bound = np.quantile(
                param_values,
                upper_quantile,
                weights=self.particle_population.weights,
                method="inverted_cdf",
            )
            credible_intervals[param] = (lower_bound, upper_bound)

        return credible_intervals

    def get_point_estimates(
        self, params: list[str] | None = None
    ) -> dict[str, float]:
        """
        Get point estimates (mean) for the given parameters in the particle population.

        Args:
            params (list[str] | None): A list of parameter names to compute point estimates for.
                If None, point estimates will be computed for all parameters in the particle population.

        Returns:
            dict[str, float]: A dictionary mapping parameter names to their corresponding point estimates.
        """
        params = self._validate_params(params)

        return {
            param: np.sum(
                [
                    particle[param] * particle_weight
                    for particle, particle_weight in zip(
                        self.particle_population.particles,
                        self.particle_population.weights,
                    )
                ]
            )
            for param in params
        }

    def get_variance(
        self, params: list[str] | None = None
    ) -> dict[str, float]:
        """
        Get the variance for the given parameters in the particle population.

        Args:
            params (list[str] | None): A list of parameter names to compute variances for.
                If None, variances will be computed for all parameters in the particle population.
        Returns:
            dict[str, float]: A dictionary mapping parameter names to their corresponding variances.
        """
        params = self._validate_params(params)

        return {
            param: np.sum(
                [
                    particle[param] ** 2 * particle_weight
                    for particle, particle_weight in zip(
                        self.particle_population.particles,
                        self.particle_population.weights,
                    )
                ]
            )
            - self.get_point_estimates([param])[param] ** 2
            for param in params
        }

    def get_covariance_matrix(
        self, params: list[str] | None = None
    ) -> np.ndarray:
        """
        Get the covariance matrix for the specified parameters in the particle population.

        Args:
            params (list[str] | None): A list of parameter names to include in the covariance matrix.
                If None, all parameters in the particle population will be included.

        Returns:
            np.ndarray: The covariance matrix of the specified parameters.
        Raises:
            ValueError: If no valid parameters are found in the particle population.
        Behavior:
            - If only one valid parameter is found, returns a 1x1 matrix containing the variance of that parameter, since the covariance of a single parameter is just its variance.
        """
        params = self._validate_params(params)

        if len(params) == 0:
            raise ValueError("No valid parameters found")
        if len(params) == 1:
            return np.array([[self.get_variance(params)[params[0]]]])

        data = np.array(
            [
                [particle[param] for param in params]
                for particle in self.particle_population.particles
            ]
        )
        return np.cov(
            data, rowvar=False, aweights=self.particle_population.weights
        )

    def get_correlation_matrix(
        self, params: list[str] | None = None
    ) -> np.ndarray:
        """
        Get the correlation matrix for the specified parameters in the particle population.

        Args:
            params (list[str] | None): A list of parameter names to include in the correlation matrix.
                If None, all parameters in the particle population will be included.

        Returns:
            np.ndarray: The correlation matrix of the specified parameters.
        Raises:
            ValueError: If no valid parameters are found in the particle population.
        Behavior:
            - If only one valid parameter is found, returns a 1x1 matrix with the value 1.0, since the correlation of a parameter with itself is always 1.
        """
        params = self._validate_params(params)

        if len(params) == 0:
            raise ValueError("No valid parameters found")
        if len(params) == 1:
            return np.array([[1.0]])

        cov_matrix = self.get_covariance_matrix(params)
        std_dev = np.sqrt(np.diag(cov_matrix))

        return cov_matrix / np.outer(std_dev, std_dev)
