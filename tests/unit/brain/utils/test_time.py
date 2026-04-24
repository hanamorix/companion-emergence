"""Tests for brain.utils.time."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from brain.utils.time import iso_utc, parse_iso_utc


def test_iso_utc_formats_tz_aware():
    dt = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    assert iso_utc(dt) == "2026-04-24T12:00:00Z"


def test_iso_utc_raises_on_naive():
    dt = datetime(2026, 4, 24, 12, 0, 0)  # no tzinfo
    with pytest.raises(ValueError, match="tz-aware"):
        iso_utc(dt)


def test_iso_utc_non_utc_still_produces_offset():
    tz = timezone(timedelta(hours=5))
    dt = datetime(2026, 4, 24, 12, 0, 0, tzinfo=tz)
    # +05:00 is not UTC, so Z substitution doesn't apply — returns +05:00
    out = iso_utc(dt)
    assert out == "2026-04-24T12:00:00+05:00"


def test_parse_iso_utc_with_z_suffix():
    dt = parse_iso_utc("2026-04-24T12:00:00Z")
    assert dt == datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)


def test_parse_iso_utc_with_offset():
    dt = parse_iso_utc("2026-04-24T12:00:00+00:00")
    assert dt == datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)


def test_parse_iso_utc_naive_coerced_to_utc():
    dt = parse_iso_utc("2026-04-24T12:00:00")
    assert dt.tzinfo is UTC


def test_iso_utc_roundtrip():
    dt = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    assert parse_iso_utc(iso_utc(dt)) == dt
