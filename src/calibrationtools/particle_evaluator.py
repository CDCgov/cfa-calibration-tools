"""Evaluate particles by running the model and scoring its outputs.

This module isolates the particle-to-params, simulate, and distance-scoring
steps behind one small collaborator so sampler execution code stays focused on
proposal and acceptance flow.
"""

from typing import Any, Callable

from mrp import MRPModel

from .particle import Particle


class ParticleEvaluator:
    """Evaluate particles by running the model and scoring its outputs.

    This class holds the user-supplied mapping and scoring functions together
    with the model runner so particle evaluation has a single, testable
    boundary.

    Args:
        particles_to_params (Callable[[Particle], dict]): Function mapping a
            particle to model parameters.
        outputs_to_distance (Callable[..., float]): Function scoring simulated
            outputs against target data.
        target_data (Any): Observed data used for distance evaluation.
        model_runner (MRPModel): Model runner used to simulate outputs.

    Returns:
        None
    """

    def __init__(
        self,
        particles_to_params: Callable[[Particle], dict],
        outputs_to_distance: Callable[..., float],
        target_data: Any,
        model_runner: MRPModel,
    ) -> None:
        """Store the collaborators needed to score one proposed particle.

        This constructor keeps the particle-evaluation boundary explicit by
        grouping the user-supplied mapping function, scoring function, target
        data, and model runner in one object.

        Args:
            particles_to_params (Callable[[Particle], dict]): Function mapping
                a particle to model parameters.
            outputs_to_distance (Callable[..., float]): Function scoring
                simulated outputs against target data.
            target_data (Any): Observed data used for distance evaluation.
            model_runner (MRPModel): Model runner used to simulate outputs.

        Returns:
            None
        """

        self.particles_to_params = particles_to_params
        self.outputs_to_distance = outputs_to_distance
        self.target_data = target_data
        self.model_runner = model_runner

    def distance(self, particle: Particle, **kwargs: Any) -> float:
        """Return the distance between simulated outputs and target data.

        This method translates a particle into model parameters, runs the
        model, and scores the resulting outputs against the stored target data.

        Args:
            particle (Particle): Particle to evaluate.
            **kwargs (Any): Additional keyword arguments forwarded to
                `particles_to_params`.

        Returns:
            float: Distance between the simulated outputs and the target data.
        """

        params = self.particles_to_params(particle, **kwargs)
        outputs = self.model_runner.simulate(params)
        return self.outputs_to_distance(outputs, self.target_data)
