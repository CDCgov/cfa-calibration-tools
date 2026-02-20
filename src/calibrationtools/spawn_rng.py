from numpy.random import Generator, SeedSequence, default_rng


def spawn_rng(seed_sequence: SeedSequence) -> Generator:
    """Spawn a new RNG from the given seed sequence."""
    return default_rng(seed_sequence.spawn(1)[0])
