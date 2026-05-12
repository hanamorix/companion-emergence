"""Tests for the initiate_d_calls audit table."""
from __future__ import annotations

import json

import pytest

from brain.initiate.d_call_schema import DCallRow, make_d_call_id


def test_d_call_row_roundtrip():
    row = DCallRow(
        d_call_id="dc_2026-05-12T10-00-00_ab",
        ts="2026-05-12T10:00:00+00:00",
        tick_id="tick_001",
        model_tier_used="haiku",
        candidates_in=3,
        promoted_out=1,
        filtered_out=2,
        latency_ms=420,
        tokens_input=560,
        tokens_output=180,
        failure_type=None,
        retry_count=0,
        tick_note="quiet morning weather, one worth saying",
    )
    line = row.to_jsonl()
    parsed = DCallRow.from_jsonl(line)
    assert parsed == row


def test_d_call_row_failure_type_optional():
    row = DCallRow(
        d_call_id="dc_x",
        ts="2026-05-12T10:00:00+00:00",
        tick_id="tick_002",
        model_tier_used="haiku",
        candidates_in=2,
        promoted_out=0,
        filtered_out=0,
        latency_ms=15000,
        tokens_input=0,
        tokens_output=0,
        failure_type="timeout",
        retry_count=1,
    )
    line = row.to_jsonl()
    d = json.loads(line)
    assert d["failure_type"] == "timeout"
    assert d["tick_note"] is None


def test_make_d_call_id_sortable():
    from datetime import UTC, datetime
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    ident = make_d_call_id(now)
    assert ident.startswith("dc_2026-05-12T10-00-00_")
    assert len(ident) == len("dc_2026-05-12T10-00-00_") + 4  # 2 hex bytes = 4 chars
