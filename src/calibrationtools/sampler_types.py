"""Define typed carriers shared across sampler execution helpers.

This module replaces positional tuples and ad hoc dictionaries with small data
objects so sampler control flow reads in terms of named responsibilities.
"""

from dataclasses import dataclass

from numpy.random import SeedSequence

from .particle import Particle


@dataclass(frozen=True, slots=True)
class GeneratorSlot:
    """Identify one deterministic proposal stream for a generation.

    This carrier keeps the slot identifier and spawned seed sequence together
    so proposal ordering remains explicit across serial and parallel execution.

    Attributes:
        id (int): Stable proposal-slot identifier within the generation.
        seed_sequence (SeedSequence): Seed sequence used for the slot.
    """

    id: int
    seed_sequence: SeedSequence


@dataclass(frozen=True, slots=True)
class AcceptedProposal:
    """Store the result of proposing until one particle is accepted.

    This carrier keeps the accepted particle, or an exhausted-attempt marker,
    together with the slot identifier and attempt count used to produce it.

    Attributes:
        slot_id (int): Proposal-slot identifier within the generation.
        particle (Particle | None): Accepted particle for the slot, or `None`
            when attempts were exhausted.
        distance (float | None): Distance of the accepted particle, or `None`
            when attempts were exhausted.
        attempts (int): Proposal attempts consumed for the slot.
    """

    slot_id: int
    particle: Particle | None
    distance: float | None
    attempts: int


@dataclass(frozen=True, slots=True)
class GenerationStats:
    """Summarize attempts, successes, and timing for one generation.

    This carrier provides a named summary object for generation-level metrics
    that are shared across the particlewise and batched execution paths.

    Attributes:
        attempts (int): Total proposal attempts consumed by the generation.
        successes (int): Total accepted particles produced by the generation.
        processing_time (float): Seconds spent in the generation processing
            phase.
        total_time (float): Seconds elapsed since the full sampler run began.
    """

    attempts: int
    successes: int
    processing_time: float
    total_time: float
