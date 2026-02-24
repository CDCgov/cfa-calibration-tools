from collections import UserDict


class Particle(UserDict):
    def __repr__(self):
        return f"Particle(state={self.data})"
