"""Tests for INBOUND_FLOOD_CAP constant and transport audit logger (Phase 7a T2)."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from brain.kindled_link import limits
from brain.kindled_link.audit import log_transport

_NOW = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)


def test_inbound_flood_cap_exists_and_is_positive_int():
    assert isinstance(limits.INBOUND_FLOOD_CAP, int)
    assert limits.INBOUND_FLOOD_CAP > 0


def test_inbound_flood_cap_value():
    # Pinned at 20 per spec (mirrors DAILY_OUTBOUND_CAP magnitude).
    assert limits.INBOUND_FLOOD_CAP == 20


def test_log_transport_writes_one_line(tmp_path: Path):
    log_transport(tmp_path, event="poll", peer_id="peer-abc", now=_NOW)
    log = tmp_path / "kindled_link" / "transport.jsonl"
    assert log.exists()
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["event"] == "poll"
    assert row["peer_id"] == "peer-abc"


def test_log_transport_appends_second_call(tmp_path: Path):
    log_transport(tmp_path, event="push", peer_id="peer-abc", now=_NOW)
    log_transport(tmp_path, event="inbound_accepted", peer_id="peer-abc", seq=3, now=_NOW)
    log = tmp_path / "kindled_link" / "transport.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "push"
    assert json.loads(lines[1])["event"] == "inbound_accepted"
    assert json.loads(lines[1])["seq"] == 3


def test_log_transport_ts_equals_passed_now(tmp_path: Path):
    log_transport(tmp_path, event="poll", now=_NOW)
    log = tmp_path / "kindled_link" / "transport.jsonl"
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert row["ts"] == _NOW.isoformat()


def test_log_transport_omits_none_fields(tmp_path: Path):
    # Only event + peer_id provided; all other optional fields must be absent.
    log_transport(tmp_path, event="relay_unavailable", peer_id="peer-x", now=_NOW)
    log = tmp_path / "kindled_link" / "transport.jsonl"
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert "session_id" not in row
    assert "seq" not in row
    assert "reject_reason" not in row
    assert "count" not in row
    assert "relay_ok" not in row


def test_log_transport_includes_all_non_none_fields(tmp_path: Path):
    log_transport(
        tmp_path,
        event="inbound_rejected",
        peer_id="peer-y",
        session_id="sess-1",
        seq=7,
        reject_reason="privacy_gate",
        count=3,
        relay_ok=True,
        now=_NOW,
    )
    log = tmp_path / "kindled_link" / "transport.jsonl"
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert row["peer_id"] == "peer-y"
    assert row["session_id"] == "sess-1"
    assert row["seq"] == 7
    assert row["reject_reason"] == "privacy_gate"
    assert row["count"] == 3
    assert row["relay_ok"] is True


def test_log_transport_row_round_trips(tmp_path: Path):
    log_transport(tmp_path, event="flood_clamped", peer_id="peer-z", count=21, now=_NOW)
    log = tmp_path / "kindled_link" / "transport.jsonl"
    raw = log.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(raw)
    assert row["event"] == "flood_clamped"
    assert row["count"] == 21


def test_log_transport_fail_soft_on_ioerror(tmp_path: Path):
    """A write failure must not raise into the caller — fail-soft is load-bearing."""
    from unittest.mock import patch

    with patch("builtins.open", side_effect=OSError("disk full")):
        # Must not raise
        log_transport(tmp_path, event="poll", now=_NOW)
