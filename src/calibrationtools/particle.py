from typing import Sequence


class Particle:
    def __init__(self, state: dict, weight: float = 1.0):
        self.state = state
        self.weight = weight

    def __repr__(self):
        return f"Particle(state={self.state}, weight={self.weight})"


class ParticlePopulation:
    def __init__(
        self,
        initial_states: Sequence[dict] | None = None,
        initial_weights: Sequence[float] | None = None,
    ):
        self._particles: list[Particle] = []

        initial_states = [] if initial_states is None else initial_states
        initial_weights = [] if initial_weights is None else initial_weights

        n_states = len(initial_states)

        if len(initial_weights) > 0 and len(initial_weights) != n_states:
            raise ValueError(
                "Length of initial_weights must match length of initial_states"
            )

        if n_states > 0 and not initial_weights:
            initial_weights: list[float] = [1.0] * n_states

        for state, weight in zip(initial_states, initial_weights):
            self.add(
                Particle(state=state, weight=weight), normalize_weights=False
            )

        self.normalize_weights()

    def __repr__(self):
        return f"ParticlePopulation(size={self.size}, ESS={self.ess})"

    @property
    def particles(self) -> list[Particle]:
        return self._particles

    def add(self, particle: Particle, normalize_weights: bool = True):
        self._particles.append(particle)
        if normalize_weights:
            self.normalize_weights()

    @property
    def ess(self) -> float:
        if self.total_weight == 0:
            return 0.0
        else:
            return (self.total_weight**2) / sum(
                p.weight**2.0 for p in self.particles
            )

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
        return sum(p.weight for p in self.particles)

    def normalize_weights(self):
        if self.is_empty:
            return
        else:
            normalization_factor = 1.0 / self.total_weight
            for p in self.particles:
                p.weight *= normalization_factor
