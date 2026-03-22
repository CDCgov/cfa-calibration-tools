import asyncio
import json
import inspect
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from mrp import MRPModel
from numpy.random import SeedSequence

from .particle import Particle
from .particle_population import ParticlePopulation
from .particle_updater import _ParticleUpdater
from .perturbation_kernel import PerturbationKernel
from .prior_distribution import PriorDistribution
from .json_utils import dumps_json, to_jsonable
from .variance_adapter import VarianceAdapter


class ABCSampler:
    """
    ABCSampler is a class that implements an Approximate Bayesian Computation (ABC)
    Sequential Monte Carlo (SMC) sampler. This sampler is used to estimate posterior
    distributions of parameters for a given model by iteratively sampling and perturbing
    particles, and evaluating their distance from observed data using user-supplied functions.

    Args:
        generation_particle_count (int): Number of particles to accept per generation for a complete population.
        tolerance_values (list[float]): List of tolerance values for each generation for evaluating acceptance criterion.
        priors (PriorDistribution | dict | Path): Prior distribution of the parameters being calibrated. Can be provided as a PriorDistribution object, a dictionary, or a path to a JSON file containing a valid priors schema.
        particles_to_params (Callable[[Particle], dict]): Function to map particles to model parameters.
        outputs_to_distance (Callable[..., float]): Function to compute distance between model outputs and target data.
        target_data (Any): Observed data to compare against.
        model_runner (MRPModel): Model runner to simulate outputs given parameters.
        perturbation_kernel (PerturbationKernel): Initial kernel used to perturb particles across SMC steps.
        variance_adapter (VarianceAdapter): Adapter to adjust perturbation variance across SMC steps.
        max_attempts_per_proposal (int): Maximum number of sample and perturb attempts to propose a particle.
        max_concurrent_simulations (int): Maximum number of model evaluations to run concurrently.
        seed (int | None): Random seed for reproducibility.
        verbose (bool): Whether to print verbose output during execution.
        drop_previous_population_data (bool): Whether to drop previous population data when storing the accepted particles between SMC steps.
        seed_parameter_name (str | None): The name of the seed parameter to include in the priors if `incl_seed_parameter` is True when loading priors from a dictionary or JSON file.

    Methods:
        particle_population:
            Getter and setter for the current particle population. Automatically archives
            the previous population if `drop_previous_population_data` is False.

        get_posterior_particles() -> ParticlePopulation:
            Returns the posterior particle population after the sampler has run to completion.

        run(**kwargs: Any):
            Executes the ABC-SMC algorithm. Raises an error if any keyword argument conflicts
            with existing attributes.

        sample_priors(n: int = 1) -> Sequence[dict[str, Any]]:
            Samples `n` states from the prior distribution.

        sample_particle_from_priors() -> Particle:
            Samples a single particle from the prior distribution.

        sample_particle() -> Particle:
            Samples a particle from the current population.

        sample_and_perturb_particle() -> Particle:
            Samples and perturbs a particle from the current population.

        calculate_weight(particle) -> float:
            Calculates the weight of a given particle based on its prior and perturbed probabilities.
    """

    def __init__(
        self,
        generation_particle_count: int,
        tolerance_values: list[float],
        priors: PriorDistribution | dict | Path,
        particles_to_params: Callable[[Particle], dict],
        outputs_to_distance: Callable[..., float],
        target_data: Any,
        model_runner: MRPModel,
        perturbation_kernel: PerturbationKernel,
        variance_adapter: VarianceAdapter,
        max_attempts_per_proposal: int = np.iinfo(np.int32).max,
        max_concurrent_simulations: int = 1,
        seed: int | None = None,
        verbose: bool = True,
        drop_previous_population_data: bool = False,
        seed_parameter_name: str | None = "seed",
        artifacts_dir: Path | str | None = None,
    ):
        self.generation_particle_count = generation_particle_count
        self.max_attempts_per_proposal = max_attempts_per_proposal
        if max_concurrent_simulations < 1:
            raise ValueError(
                "max_concurrent_simulations must be at least 1"
            )
        self.max_concurrent_simulations = max_concurrent_simulations
        self.tolerance_values = tolerance_values
        self._perturbation_kernel = perturbation_kernel
        self._variance_adapter = variance_adapter
        self.particles_to_params = particles_to_params
        self.outputs_to_distance = outputs_to_distance
        self.target_data = target_data
        self.model_runner = model_runner
        self.seed = seed
        self._seed_sequence = SeedSequence(seed)
        self.drop_previous_population_data = drop_previous_population_data
        self.artifacts_dir = (
            Path(artifacts_dir) if artifacts_dir is not None else None
        )
        self.population_archive: dict[int, ParticlePopulation] = {}
        self.smc_step_successes = [0] * len(tolerance_values)
        self.verbose = verbose

        if isinstance(priors, PriorDistribution):
            self._priors = priors
        elif isinstance(priors, dict):
            from .load_priors import independent_priors_from_dict

            self._priors = independent_priors_from_dict(
                priors,
                incl_seed_parameter=seed_parameter_name is not None,
                seed_parameter_name=seed_parameter_name,
            )
        elif isinstance(priors, Path) or isinstance(priors, str):
            from .load_priors import load_priors_from_json

            self._priors = load_priors_from_json(priors)

        self._updater = _ParticleUpdater(
            self._perturbation_kernel,
            self._priors,
            self._variance_adapter,
            self._seed_sequence,
            ParticlePopulation(),
        )

    @property
    def particle_population(self) -> ParticlePopulation:
        return self._updater.particle_population

    @particle_population.setter
    def particle_population(self, population: ParticlePopulation):
        """
        Updates the particle population for the sampler.

        If `drop_previous_population_data` is set to False and there is existing
        particle population data, the current particle population is archived
        before updating to the new population.

        Args:
            population (ParticlePopulation): The new particle population to set.

        Attributes:
            drop_previous_population_data (bool): Determines whether to discard
                previous population data or archive it.
            _updater.particle_population (ParticlePopulation): The current particle
                population managed by the updater.
            population_archive (dict): A dictionary storing archived particle
                populations, indexed by step.

        Behavior:
            - If `drop_previous_population_data` is False and there is existing
              particle population data, the current population is archived with
              a step index.
            - Updates the `_updater.particle_population` with the new population.
            - Weights of the new population are normalized and the perturbation
              variance is adapted by the particle updater's setter method.
        """
        if (
            not self.drop_previous_population_data
            and self._updater.particle_population.size > 0
        ):
            step = (
                max(self.population_archive.keys()) + 1
                if self.population_archive
                else 0
            )
            self.population_archive.update({step: self.particle_population})
        self._updater.particle_population = population

    def get_posterior_particles(self) -> ParticlePopulation:
        """
        Retrieve the posterior particle population.

        This method returns the current particle population representing the posterior
        distribution after the sampling process has been completed. It ensures
        that the posterior population is fully populated before returning it.

        Returns:
            ParticlePopulation: The particle population representing the posterior
                distribution.

        Raises:
            ValueError: If the posterior population is not fully populated,
                indicating that the sampler has not been run to completion.
        """
        if self.smc_step_successes[-1] != self.generation_particle_count:
            raise ValueError(
                "Posterior population is not fully populated. Please run the sampler to completion before accessing the posterior population."
            )
        return self.particle_population

    def _simulate(
        self,
        params: dict[str, Any],
        *,
        input_path: Path | None = None,
        output_dir: Path | None = None,
        run_id: str | None = None,
    ) -> Any:
        simulate = self.model_runner.simulate

        signature = inspect.signature(simulate)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        simulate_kwargs: dict[str, Any] = {}

        if input_path is not None and (
            accepts_kwargs or "input_path" in signature.parameters
        ):
            simulate_kwargs["input_path"] = input_path
        if output_dir is not None and (
            accepts_kwargs or "output_dir" in signature.parameters
        ):
            simulate_kwargs["output_dir"] = output_dir
        if run_id is not None and (
            accepts_kwargs or "run_id" in signature.parameters
        ):
            simulate_kwargs["run_id"] = run_id

        return simulate(params, **simulate_kwargs)

    async def _simulate_async(
        self,
        params: dict[str, Any],
        *,
        input_path: Path | None = None,
        output_dir: Path | None = None,
        run_id: str | None = None,
    ) -> Any:
        simulate = self.model_runner.simulate
        if inspect.iscoroutinefunction(simulate):
            return await self._simulate(
                params,
                input_path=input_path,
                output_dir=output_dir,
                run_id=run_id,
            )

        outputs = await asyncio.to_thread(
            self._simulate,
            params,
            input_path=input_path,
            output_dir=output_dir,
            run_id=run_id,
        )
        if inspect.isawaitable(outputs):
            return await outputs
        return outputs

    def _build_run_id(
        self, generation_index: int, proposal_index: int
    ) -> str:
        return (
            f"gen-{generation_index + 1}_particle-{proposal_index + 1}"
        )

    def _stage_simulation_io(
        self,
        generation_index: int,
        proposal_index: int,
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], Path | None, Path | None, str]:
        run_id = self._build_run_id(generation_index, proposal_index)
        jsonable_params = to_jsonable(params)
        jsonable_params["run_id"] = run_id

        if self.artifacts_dir is None:
            return jsonable_params, None, None, run_id

        generation_name = f"generation-{generation_index + 1}"
        input_dir = self.artifacts_dir / "input" / generation_name
        output_dir = (
            self.artifacts_dir / "output" / generation_name / run_id
        )
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_path = input_dir / f"{run_id}.json"
        input_path.write_text(dumps_json(jsonable_params) + "\n")

        return jsonable_params, input_path, output_dir, run_id

    async def _evaluate_particle_async(
        self,
        particle: Particle,
        params: dict[str, Any],
        *,
        input_path: Path | None = None,
        output_dir: Path | None = None,
        run_id: str | None = None,
    ) -> tuple[Particle, float]:
        outputs = await self._simulate_async(
            params,
            input_path=input_path,
            output_dir=output_dir,
            run_id=run_id,
        )
        if output_dir is not None:
            output_path = output_dir / "result.json"
            output_path.write_text(
                json.dumps(to_jsonable(outputs), indent=2, sort_keys=True)
                + "\n"
            )
        err = self.outputs_to_distance(outputs, self.target_data)
        return particle, err

    async def run_async(self, **kwargs: Any):
        """
        Executes the Sequential Monte Carlo (SMC) sampling process.

        This method performs the SMC algorithm to generate a population of particles
        that approximate the posterior distribution of the model parameters. The process
        involves iteratively sampling and perturbing particles, evaluating their fitness
        using a distance metric, and accepting or rejecting them based on a tolerance value.

        Args:
            **kwargs (Any): Additional keyword arguments that can be passed to the method.
                      These arguments are supplied to the particles_to_params function.
                      Note that the keyword arguments must not conflict with existing
                      attributes of the class.

        Raises:
            ValueError: If a keyword argument conflicts with an existing attribute of the class.

        Process:
            1. For each generation, particles are sampled either from the prior distribution
               (for the first generation) or by perturbing particles from the previous generation.
            2. The sampled particles are evaluated using the model to compute a distance metric
               relative to the target data.
            3. Particles that meet the tolerance criteria are accepted and added to the population.
            4. The process continues until the desired number of particles is obtained for the generation.

        Args Updated:
            - `smc_step_successes`: A dictionary tracking the number of successful particles
              for each generation.
            - `particle_population`: The final population of particles for the current generation.
            - `perturbation_kernel`: The perturbation kernel may be updated by the variance adapter based on the successive particle population.
            - `population_archive`: If `drop_previous_population_data` is False, previous populations are archived before updating to the new population.

        Notes:
            - The method prints progress information if `verbose` is set to True.
            - The acceptance rate is displayed periodically during the sampling process.
            - When `max_concurrent_simulations > 1`, model evaluations are run concurrently
              while particle proposal sampling, weighting, and population updates remain serial.
        """
        for k in kwargs.keys():
            if k in self.__class__.__dict__:
                raise ValueError(
                    f"Keyword argument '{k}' conflicts with existing attribute. Please choose a different name for the argument. Args cannot be set from `.run()`"
                )

        proposed_population = ParticlePopulation()

        for generation in range(len(self.tolerance_values)):
            if self.verbose:
                print(
                    f"Running generation {generation + 1} with tolerance {self.tolerance_values[generation]}..."
                )

            # Rejection sampling algorithm
            attempts = 0
            while proposed_population.size < self.generation_particle_count:
                if self.verbose and attempts > 0 and attempts % 100 == 0:
                    print(
                        f"Attempt {attempts}... current population size is {proposed_population.size}. Acceptance rate is {proposed_population.size / attempts if attempts > 0 else 0:.4f}",
                        end="\r",
                    )
                batch_size = self.max_concurrent_simulations
                evaluation_tasks: list[asyncio.Task[tuple[Particle, float]]] = []

                for _ in range(batch_size):
                    attempts += 1
                    if generation == 0:
                        proposed_particle = self.sample_particle_from_priors()
                    else:
                        proposed_particle = self.sample_and_perturb_particle()
                    params = self.particles_to_params(
                        proposed_particle, **kwargs
                    )
                    (
                        params,
                        input_path,
                        output_dir,
                        run_id,
                    ) = self._stage_simulation_io(
                        generation,
                        attempts - 1,
                        params,
                    )
                    evaluation_tasks.append(
                        asyncio.create_task(
                            self._evaluate_particle_async(
                                proposed_particle,
                                params,
                                input_path=input_path,
                                output_dir=output_dir,
                                run_id=run_id,
                            )
                        )
                    )

                evaluated_particles = await asyncio.gather(
                    *evaluation_tasks
                )
                for proposed_particle, err in evaluated_particles:
                    if proposed_population.size >= self.generation_particle_count:
                        break
                    if err < self.tolerance_values[generation]:
                        if generation == 0:
                            particle_weight = 1.0
                        else:
                            particle_weight = self.calculate_weight(
                                proposed_particle
                            )
                        proposed_population.add_particle(
                            proposed_particle, particle_weight
                        )

            self.smc_step_successes[generation] = proposed_population.size
            self.particle_population = proposed_population
            proposed_population = ParticlePopulation()

    def run(self, **kwargs: Any):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(**kwargs))
        raise RuntimeError(
            "ABCSampler.run() cannot be called from an active event loop. Use `await sampler.run_async(...)` instead."
        )

    def sample_priors(self, n: int = 1) -> Sequence[dict[str, Any]]:
        """Return a sequence of states sampled from the prior distribution"""
        return self._priors.sample(n, self._seed_sequence)

    def sample_particle_from_priors(self) -> Particle:
        return Particle(self.sample_priors(1)[0])

    def sample_particle(self) -> Particle:
        return self._updater.sample_particle()

    def sample_and_perturb_particle(self) -> Particle:
        return self._updater.sample_and_perturb_particle(
            max_attempts=self.max_attempts_per_proposal
        )

    def calculate_weight(self, particle) -> float:
        return self._updater.calculate_weight(particle)
