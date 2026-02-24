from typing import Any


class Particle:
    def __init__(self, state: dict[str, Any], weight: float = 1.0):
        self._state = state
        self._weight = weight

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @state.setter
    def state(self, value: dict[str, Any]):
        if not isinstance(value, dict):
            raise ValueError("State must be a dictionary")
        if not all(isinstance(k, str) for k in value.keys()):
            raise ValueError("State keys must be strings")
        self._state = value

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError("Weight must be a number")
        if value < 0:
            raise ValueError("Weight must be non-negative")
        self._weight = float(value)

    def __repr__(self):
        return f"Particle(state={self.state}, weight={self.weight})"
