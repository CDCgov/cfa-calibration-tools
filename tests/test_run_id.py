import pytest

from calibrationtools.run_id import (
    SamplerRunId,
    format_sampler_run_id,
    parse_sampler_run_id,
)


def test_parse_sampler_run_id_accepts_canonical_zero_based_attempt_id():
    assert parse_sampler_run_id("gen_0_particle_1_attempt_2") == SamplerRunId(
        generation_index=0,
        proposal_index=1,
        attempt_index=2,
    )


@pytest.mark.parametrize(
    "run_id",
    [
        "gen-1_particle-1",
        "gen_0_particle_0",
        "gen_0_particle_0-attempt-0",
        "gen-0_particle-0_attempt_0",
    ],
)
def test_parse_sampler_run_id_rejects_legacy_or_incomplete_ids(run_id):
    with pytest.raises(ValueError, match="gen_0_particle_0_attempt_0"):
        parse_sampler_run_id(run_id)


def test_format_sampler_run_id_rejects_negative_indices():
    with pytest.raises(ValueError, match="generation_index"):
        format_sampler_run_id(
            generation_index=-1,
            proposal_index=0,
            attempt_index=0,
        )
