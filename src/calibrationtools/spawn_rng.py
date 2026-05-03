from numpy.random import Generator, SeedSequence, default_rng


def spawn_rng(seed_sequence: SeedSequence | None) -> Generator:
    """Spawn a new RNG from the given seed sequence."""
    if seed_sequence is None:
        return default_rng()
    return default_rng(seed_sequence.spawn(1)[0])
