from typing import Iterator


class Particle:
    def __init__(self, state: dict, weight: float = None):
        self.state = state
        self.weight = weight

    def __repr__(self):
        return f"Particle(state={self.state}, weight={self.weight})"


class ParticlePopulation:
    def __init__(
        self,
        initial_states: Iterator[dict] | None = None,
        initial_weights: Iterator[float] | None = None,
    ):
        self.all_particles = []
        self.weights = {}
        initial_states = (
            list(initial_states) if initial_states is not None else None
        )
        if initial_states is not None:
            if initial_weights is None:
                count = len(initial_states)
                initial_weights = [1.0 / count] * count

            for state, weight in zip(initial_states, initial_weights):
                self.add(Particle(state=state, weight=weight))

    def __repr__(self):
        return f"ParticlePopulation(size={self.size}, ESS={self.ess})"

    def add(self, particle: Particle):
        self.all_particles.append(particle)
        self.weights.update({self.size: particle.weight})

    @property
    def ess(self):
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            return 0
        else:
            return (total_weight**2) / sum(w**2 for w in self.weights.values())

    @property
    def size(self):
        return len(self.all_particles)

    def normalize_weights(self):
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            uniform_weight = 1.0 / self.size
            for particle in self.all_particles:
                particle.weight = uniform_weight
        else:
            normalization_factor = 1.0 / total_weight
            for particle in self.all_particles:
                particle.weight *= normalization_factor
        self.weights = {i: p.weight for i, p in enumerate(self.all_particles)}
