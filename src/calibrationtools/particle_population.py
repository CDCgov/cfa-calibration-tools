from typing import Sequence

from .particle import Particle


class ParticlePopulation:
    def __init__(
        self,
        states: Sequence[dict[str, any]] | None = None,
        weights: Sequence[float] | None = None,
    ):
        self._particles: list[Particle] = (
            [] if states is None else [Particle(x) for x in states]
        )

        if weights is None:
            self._weights = [1.0] * len(self._particles)
        else:
            self._weights = list(weights)

        if len(self._weights) > 0 and len(self._weights) != len(
            self._particles
        ):
            raise ValueError(
                "Length of weights must match length of particles"
            )

        self.normalize_weights()

    @property
    def particles(self) -> list[Particle]:
        return self._particles

    @property
    def weights(self) -> list[float]:
        return self._weights

    def add_particle(
        self, state: dict[str, any], weight=1.0, normalize_weights: bool = True
    ):
        self._particles.append(Particle(state))
        self._weights.append(weight)
        if normalize_weights:
            self.normalize_weights()

    @property
    def ess(self) -> float:
        if self.total_weight == 0:
            return 0.0
        else:
            return (self.total_weight**2) / sum(w**2 for w in self.weights)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    @property
    def size(self) -> int:
        return len(self.particles)

    @property
    def total_weight(self) -> float:
        if self.is_empty:
            return 0.0
        return sum(self.weights)

    def normalize_weights(self):
        if self.is_empty:
            return
        else:
            normalization_factor = 1.0 / self.total_weight
            for i in range(self.size):
                self.weights[i] *= normalization_factor

    def __repr__(self):
        return f"ParticlePopulation(size={self.size}, ESS={self.ess})"
