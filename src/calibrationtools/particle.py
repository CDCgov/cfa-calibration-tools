from collections import UserDict


class Particle(UserDict):
    """
    Particle is a subclass of `UserDict` that represents a particle with a specific state.

    Args:
        data (dict): The internal dictionary storing the state of the particle. This can
            be accessed using the standard dictionary interface provided by `UserDict`.
    """

    def __repr__(self):
        return f"Particle(state={self.data})"
