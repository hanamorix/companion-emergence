"""Tests for brain.initiate.audit — audit log read/write + state transitions."""

from __future__ import annotations

import gzip
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.initiate.audit import (
    append_audit_row,
    iter_initiate_audit_full,
    read_recent_audit,
    update_audit_state,
)
from brain.initiate.schemas import AuditRow


def _row(
    audit_id: str, candidate_id: str, decision: str = "send_quiet", ts: str | None = None
) -> AuditRow:
    if ts is None:
        ts = datetime.now(UTC).isoformat()
    return AuditRow(
        audit_id=audit_id,
        candidate_id=candidate_id,
        ts=ts,
        kind="message",
        subject="the dream",
        tone_rendered="the dream from this morning landed somewhere",
        decision=decision,
        decision_reasoning="resonance is real",
        gate_check={"allowed": True, "reason": None},
    )


def test_append_audit_row_creates_file_and_writes(tmp_path: Path) -> None:
    row = _row("ia_001", "ic_001")
    append_audit_row(tmp_path, row)
    assert (tmp_path / "initiate_audit.jsonl").exists()
    rows = list(read_recent_audit(tmp_path, window_hours=24))
    assert len(rows) == 1
    assert rows[0].audit_id == "ia_001"


def test_append_audit_row_per_append_reopens(tmp_path: Path) -> None:
    """Append-write contract: each call reopens the file."""
    append_audit_row(tmp_path, _row("ia_001", "ic_001"))
    append_audit_row(tmp_path, _row("ia_002", "ic_002"))
    rows = list(read_recent_audit(tmp_path, window_hours=24))
    assert {r.audit_id for r in rows} == {"ia_001", "ia_002"}


def test_update_audit_state_mutates_row_in_place(tmp_path: Path) -> None:
    append_audit_row(tmp_path, _row("ia_001", "ic_001"))
    update_audit_state(
        tmp_path,
        audit_id="ia_001",
        new_state="delivered",
        at="2026-05-11T14:47:09.5+00:00",
    )
    update_audit_state(
        tmp_path,
        audit_id="ia_001",
        new_state="read",
        at="2026-05-11T18:34:21+00:00",
    )
    rows = list(read_recent_audit(tmp_path, window_hours=24))
    assert rows[0].delivery["current_state"] == "read"
    assert len(rows[0].delivery["state_transitions"]) == 2


def test_iter_initiate_audit_full_walks_archives(tmp_path: Path) -> None:
    """Mirrors iter_audit_full from soul.audit — chronological across archives."""
    # Active file: 2026 entry.
    append_audit_row(tmp_path, _row("ia_active", "ic_a"))
    # Archive: 2024 entry, gzipped.
    archive = tmp_path / "initiate_audit.2024.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as gz:
        gz.write(_row("ia_archive_2024", "ic_archived").to_jsonl() + "\n")
    rows = list(iter_initiate_audit_full(tmp_path))
    # Archive first, then active.
    assert rows[0].audit_id == "ia_archive_2024"
    assert rows[1].audit_id == "ia_active"


def test_read_recent_audit_filters_by_window(tmp_path: Path) -> None:
    """A 1-hour window excludes rows older than 1h ago."""
    now = datetime(2026, 5, 11, 14, 47, 9, tzinfo=UTC)
    long_ago = (now - timedelta(hours=48)).isoformat()
    recent = (now - timedelta(minutes=30)).isoformat()

    old = _row("ia_old", "ic_old")
    old.ts = long_ago
    new = _row("ia_new", "ic_new")
    new.ts = recent

    append_audit_row(tmp_path, old)
    append_audit_row(tmp_path, new)

    rows = list(read_recent_audit(tmp_path, window_hours=1, now=now))
    assert [r.audit_id for r in rows] == ["ia_new"]


# ---------------------------------------------------------------------------
# #225 — reply-lag event hook (fired from update_audit_state)
# ---------------------------------------------------------------------------


def test_update_audit_state_replied_explicit_folds_reply_lag(tmp_path: Path) -> None:
    """A send_notify/send_quiet row transitioning into replied_explicit
    folds its lag into presence_state.json's running mean."""
    from brain.initiate import presence_state

    send_ts = "2026-05-29T09:00:00+00:00"
    row = _row("ia_1", "ic_1", decision="send_notify", ts=send_ts)
    append_audit_row(tmp_path, row)

    update_audit_state(
        tmp_path, audit_id="ia_1", new_state="replied_explicit", at="2026-05-29T09:01:00+00:00"
    )

    state = presence_state.load_presence_state(tmp_path)
    assert state.reply_lag_n == 1
    assert state.reply_lag_running_mean == pytest.approx(60.0)


def test_update_audit_state_non_send_decision_does_not_fold(tmp_path: Path) -> None:
    """Only send_notify/send_quiet rows count toward reply-lag — matches
    the old _compute_response_lag_p50's own decision filter."""
    from brain.initiate import presence_state

    row = _row(
        "ia_1", "ic_1", decision="filtered_pre_compose", ts="2026-05-29T09:00:00+00:00"
    )
    append_audit_row(tmp_path, row)

    update_audit_state(
        tmp_path, audit_id="ia_1", new_state="replied_explicit", at="2026-05-29T09:01:00+00:00"
    )

    state = presence_state.load_presence_state(tmp_path)
    assert state.reply_lag_n == 0


def test_update_audit_state_non_reply_transition_does_not_fold(tmp_path: Path) -> None:
    from brain.initiate import presence_state

    row = _row("ia_1", "ic_1", decision="send_notify", ts="2026-05-29T09:00:00+00:00")
    append_audit_row(tmp_path, row)

    update_audit_state(
        tmp_path, audit_id="ia_1", new_state="delivered", at="2026-05-29T09:00:05+00:00"
    )
    update_audit_state(
        tmp_path, audit_id="ia_1", new_state="dismissed", at="2026-05-29T09:05:00+00:00"
    )

    state = presence_state.load_presence_state(tmp_path)
    assert state.reply_lag_n == 0


def test_c17_replied_explicit_posted_twice_folds_once(tmp_path: Path) -> None:
    """C17: the reply-lag fold must fire at most once per row's real ENTRY
    into replied_explicit, not once per write — a duplicate/retried post
    for a row already in that state (e.g. a client retry against the
    generic /initiate/state endpoint, which performs no legal-transition
    validation) must not double-fold.

    ST1.5f self-test embedded below: an every-write-folds design (no
    pre-transition-state guard) would fold TWICE for these same two writes
    — reply_lag_n == 2 — shown directly by calling fold_reply_lag manually
    to simulate that design, contrasted with the guarded result of 1.
    """
    from brain.initiate import presence_state

    send_ts = "2026-05-29T09:00:00+00:00"
    row = _row("ia_1", "ic_1", decision="send_notify", ts=send_ts)
    append_audit_row(tmp_path, row)

    update_audit_state(
        tmp_path, audit_id="ia_1", new_state="replied_explicit", at="2026-05-29T09:01:00+00:00"
    )
    # Retry/duplicate post of the SAME target state for the SAME row.
    update_audit_state(
        tmp_path, audit_id="ia_1", new_state="replied_explicit", at="2026-05-29T09:05:00+00:00"
    )

    state = presence_state.load_presence_state(tmp_path)
    assert state.reply_lag_n == 1  # guarded: folds once, not twice
    assert state.reply_lag_running_mean == pytest.approx(60.0)

    # Self-test: reset the sidecar and replay what an unguarded,
    # every-write-folds implementation would have done for these same two
    # transitions (60s then, from the SAME send_ts, a second "reply" 300s
    # later) — it would fold BOTH, landing at n=2.
    from brain.initiate.presence_state import PresenceState, save_presence_state

    save_presence_state(
        tmp_path, PresenceState(None, None, None, None, None, 0, False, 0)
    )
    presence_state.fold_reply_lag(tmp_path, 60.0)
    presence_state.fold_reply_lag(tmp_path, 300.0)
    unguarded = presence_state.load_presence_state(tmp_path)
    assert unguarded.reply_lag_n == 2  # the violation this guard exists to prevent


# ---------------------------------------------------------------------------
# C13 — no-lost-update on initiate_audit.jsonl's read-modify-write
# ---------------------------------------------------------------------------


class _NoOpLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_c13_concurrent_transitions_unguarded_can_lose_an_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ST1.5f self-test: with _AUDIT_LOCK bypassed, two concurrent
    transitions on DIFFERENT audit_ids can clobber each other — proving the
    interleaving technique below actually detects a lost update."""
    import brain.initiate.audit as audit_mod

    append_audit_row(tmp_path, _row("a", "ca"))
    append_audit_row(tmp_path, _row("b", "cb"))

    monkeypatch.setattr(audit_mod, "_AUDIT_LOCK", _NoOpLock())

    reached_mid = threading.Event()
    proceed = threading.Event()
    paused_once = threading.Event()
    real_from_jsonl = AuditRow.from_jsonl

    def hooked(line):
        row = real_from_jsonl(line)
        if row.audit_id == "a" and not paused_once.is_set():
            paused_once.set()
            reached_mid.set()
            proceed.wait(timeout=5)
        return row

    monkeypatch.setattr(AuditRow, "from_jsonl", staticmethod(hooked))

    def call_a():
        audit_mod.update_audit_state(
            tmp_path, audit_id="a", new_state="delivered", at="2026-05-29T09:00:00+00:00"
        )

    t = threading.Thread(target=call_a)
    t.start()
    assert reached_mid.wait(timeout=5)

    # Runs to completion (unguarded) WHILE thread A is paused mid-read.
    audit_mod.update_audit_state(
        tmp_path, audit_id="b", new_state="delivered", at="2026-05-29T09:00:05+00:00"
    )

    proceed.set()
    t.join(timeout=5)

    rows = {r.audit_id: r for r in read_recent_audit(tmp_path, window_hours=24)}
    assert rows["a"].delivery["current_state"] == "delivered"
    # Unguarded: A's later full-file rewrite (built from data read before
    # B's write existed) clobbers B's change — B's delivery is LOST.
    assert rows["b"].delivery is None


def test_c13_concurrent_transitions_guarded_no_lost_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the real _AUDIT_LOCK, both transitions survive: the second call
    blocks until the first's read-modify-write completes and releases."""
    import brain.initiate.audit as audit_mod

    append_audit_row(tmp_path, _row("a", "ca"))
    append_audit_row(tmp_path, _row("b", "cb"))

    reached_mid = threading.Event()
    proceed = threading.Event()
    paused_once = threading.Event()
    real_from_jsonl = AuditRow.from_jsonl

    def hooked(line):
        row = real_from_jsonl(line)
        if row.audit_id == "a" and not paused_once.is_set():
            paused_once.set()
            reached_mid.set()
            proceed.wait(timeout=5)
        return row

    monkeypatch.setattr(AuditRow, "from_jsonl", staticmethod(hooked))

    def call_a():
        audit_mod.update_audit_state(
            tmp_path, audit_id="a", new_state="delivered", at="2026-05-29T09:00:00+00:00"
        )

    def call_b():
        audit_mod.update_audit_state(
            tmp_path, audit_id="b", new_state="delivered", at="2026-05-29T09:00:05+00:00"
        )

    ta = threading.Thread(target=call_a)
    ta.start()
    assert reached_mid.wait(timeout=5)  # A holds _AUDIT_LOCK, paused mid-read

    tb = threading.Thread(target=call_b)
    tb.start()
    time.sleep(0.05)  # give B a chance to attempt (and block on) the real lock
    proceed.set()
    ta.join(timeout=5)
    tb.join(timeout=5)

    rows = {r.audit_id: r for r in read_recent_audit(tmp_path, window_hours=24)}
    assert rows["a"].delivery["current_state"] == "delivered"
    assert rows["b"].delivery["current_state"] == "delivered"
