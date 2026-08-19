"""Format and parse sampler-managed run identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass

_RUN_ID_PATTERN = re.compile(
    r"^gen_(?P<generation>\d+)_particle_(?P<particle>\d+)"
    r"_attempt_(?P<attempt>\d+)$"
)


@dataclass(frozen=True, slots=True)
class SamplerRunId:
    generation_index: int
    proposal_index: int
    attempt_index: int


def format_sampler_run_id(
    *,
    generation_index: int,
    proposal_index: int,
    attempt_index: int,
) -> str:
    """Return the canonical zero-based sampler run id."""

    _validate_non_negative_index("generation_index", generation_index)
    _validate_non_negative_index("proposal_index", proposal_index)
    _validate_non_negative_index("attempt_index", attempt_index)
    return (
        f"gen_{generation_index}_particle_{proposal_index}"
        f"_attempt_{attempt_index}"
    )


def parse_sampler_run_id(run_id: str) -> SamplerRunId:
    match = _RUN_ID_PATTERN.match(run_id)
    if match is None:
        raise ValueError(
            "Calibration run ids must look like "
            "`gen_0_particle_0_attempt_0`; "
            f"received {run_id!r}."
        )
    return SamplerRunId(
        generation_index=int(match.group("generation")),
        proposal_index=int(match.group("particle")),
        attempt_index=int(match.group("attempt")),
    )


def format_generation_name(generation_index: int) -> str:
    _validate_non_negative_index("generation_index", generation_index)
    return f"generation-{generation_index}"


def _validate_non_negative_index(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative; received {value}.")
