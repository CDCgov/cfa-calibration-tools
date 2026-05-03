from __future__ import annotations

import re

from calibrationtools.cloud.naming import (
    make_session_id,
    parse_image_tag_from_session_id,
    parse_username_from_session_id,
)


def test_make_session_id_includes_sanitized_username():
    session_id = make_session_id("ABC123", username="Alice.Smith")

    assert re.fullmatch(
        r"\d{14}-alice-smith-abc123-[0-9a-f]{12}",
        session_id,
    )


def test_session_id_parsers_extract_username_and_image_tag():
    session_id = "20260101000000-alice-smith-abc123-123456abcdef"

    assert parse_username_from_session_id(session_id) == "alice-smith"
    assert parse_image_tag_from_session_id(session_id) == "abc123"


def test_session_id_parsers_return_none_for_invalid_values():
    invalid_values = [
        "",
        "not-a-session",
        "20260101000000-abc123-123456abcdef",
        "20260101000000-alice-abc123-nothexsuffix",
    ]

    for value in invalid_values:
        assert parse_username_from_session_id(value) is None
        assert parse_image_tag_from_session_id(value) is None
