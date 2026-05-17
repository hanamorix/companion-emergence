"""Tests for the initiate_d_calls audit table."""

from __future__ import annotations

import json

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


def test_append_d_call_row_and_read(tmp_path):
    from datetime import UTC, datetime, timedelta

    from brain.initiate.audit import append_d_call_row, read_recent_d_calls

    persona = tmp_path / "persona"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    row = DCallRow(
        d_call_id="dc_a",
        ts=now.isoformat(),
        tick_id="t1",
        model_tier_used="haiku",
        candidates_in=2,
        promoted_out=1,
        filtered_out=1,
        latency_ms=300,
        tokens_input=400,
        tokens_output=150,
    )
    append_d_call_row(persona, row)
    out = list(read_recent_d_calls(persona, window_hours=1, now=now + timedelta(minutes=10)))
    assert len(out) == 1
    assert out[0].d_call_id == "dc_a"


def test_read_recent_d_calls_window_filter(tmp_path):
    from datetime import UTC, datetime, timedelta

    from brain.initiate.audit import append_d_call_row, read_recent_d_calls

    persona = tmp_path / "persona"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    old = DCallRow(
        d_call_id="dc_old",
        ts=(now - timedelta(hours=5)).isoformat(),
        tick_id="t1",
        model_tier_used="haiku",
        candidates_in=1,
        promoted_out=0,
        filtered_out=1,
        latency_ms=200,
        tokens_input=300,
        tokens_output=100,
    )
    recent = DCallRow(
        d_call_id="dc_new",
        ts=now.isoformat(),
        tick_id="t2",
        model_tier_used="sonnet",
        candidates_in=2,
        promoted_out=2,
        filtered_out=0,
        latency_ms=800,
        tokens_input=400,
        tokens_output=200,
    )
    append_d_call_row(persona, old)
    append_d_call_row(persona, recent)
    out = list(read_recent_d_calls(persona, window_hours=1, now=now + timedelta(minutes=10)))
    assert [r.d_call_id for r in out] == ["dc_new"]


def test_read_recent_d_calls_no_file_returns_empty(tmp_path):
    from datetime import UTC, datetime

    from brain.initiate.audit import read_recent_d_calls

    persona = tmp_path / "fresh"
    out = list(read_recent_d_calls(persona, window_hours=1, now=datetime(2026, 5, 12, tzinfo=UTC)))
    assert out == []
