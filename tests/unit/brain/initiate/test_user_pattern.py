"""Tests for brain.initiate.user_pattern."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest


def _write_turn(conv_dir: Path, *, speaker: str, ts: datetime) -> None:
    """Helper: append one turn to a session file."""
    (conv_dir / "sess_test.jsonl").open("a").write(
        json.dumps({"session_id": "sess_test", "speaker": speaker, "text": "hi", "ts": ts.isoformat()}) + "\n"
    )


def test_compute_silence_days_no_buffer_returns_zero(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_silence_days

    assert _compute_silence_days(tmp_path) == pytest.approx(0.0)


def test_compute_silence_days_recent_user_turn(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_silence_days

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    twelve_hours_ago = datetime.now(UTC) - timedelta(hours=12)
    _write_turn(conv_dir, speaker="user", ts=twelve_hours_ago)

    result = _compute_silence_days(tmp_path)
    assert 0.4 < result < 0.6  # ~0.5 days


def test_compute_silence_days_skips_companion_turns(tmp_path: Path) -> None:
    """Turns from the companion (persona_dir.name) must not reset the silence clock."""
    from brain.initiate.user_pattern import _compute_silence_days

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    # Companion spoke 5 minutes ago; user spoke 3 days ago
    _write_turn(conv_dir, speaker=tmp_path.name, ts=datetime.now(UTC) - timedelta(minutes=5))
    _write_turn(conv_dir, speaker="user", ts=datetime.now(UTC) - timedelta(days=3))

    result = _compute_silence_days(tmp_path)
    assert result > 2.9  # ~3 days — companion turn ignored


def _write_audit_rows(persona_dir: Path, rows: list[dict]) -> None:
    """Write rows to initiate_audit.jsonl."""
    path = persona_dir / "initiate_audit.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_compute_ignore_streak_no_file_returns_zero(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_ignore_streak

    assert _compute_ignore_streak(tmp_path) == 0


def test_compute_ignore_streak_consecutive_unanswered(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_ignore_streak

    _write_audit_rows(tmp_path, [
        {"audit_id": "3", "ts": "2026-05-29T08:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit"}},
        {"audit_id": "2", "ts": "2026-05-29T09:00:00+00:00", "decision": "send_quiet",
         "delivery": {"current_state": "dismissed"}},
        {"audit_id": "1", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "unanswered"}},
    ])
    # Walking newest-first: unanswered (1), dismissed (1) = streak 2, then replied_explicit -> stop
    assert _compute_ignore_streak(tmp_path) == 2


def test_compute_ignore_streak_filters_non_send_decisions(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_ignore_streak

    _write_audit_rows(tmp_path, [
        {"audit_id": "2", "ts": "2026-05-29T09:00:00+00:00", "decision": "filtered_pre_compose",
         "delivery": {"current_state": "unanswered"}},
        {"audit_id": "1", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "unanswered"}},
    ])
    # filtered_pre_compose does not count; only the send_notify row counts
    assert _compute_ignore_streak(tmp_path) == 1


def test_compute_likely_active_no_buffer_returns_true(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_likely_active

    assert _compute_likely_active(tmp_path) is True


def test_compute_likely_active_insufficient_history_returns_true(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_likely_active

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    # Write only 10 turns — below _SCHEDULE_MIN_TURNS = 50
    for i in range(10):
        _write_turn(conv_dir, speaker="user", ts=datetime.now(UTC) - timedelta(hours=i))
    assert _compute_likely_active(tmp_path) is True


def test_compute_likely_active_peak_hour_true_offpeak_false(tmp_path: Path) -> None:
    """60 turns concentrated at UTC 14:00 → peak hour True, 12 hours away False."""
    from brain.initiate.user_pattern import _compute_likely_active

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()

    # Write 60 turns all at UTC 14:00, spread over 30 days
    for i in range(60):
        ts = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC) - timedelta(days=i % 30)
        _write_turn(conv_dir, speaker="user", ts=ts)

    # Inject _now at UTC 14:00 — same local bucket as the turns
    now_peak = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)
    assert _compute_likely_active(tmp_path, _now=now_peak) is True

    # Inject _now at UTC 02:00 — 12 hours away, that bucket has 0 turns
    now_off = datetime(2026, 1, 15, 2, 0, 0, tzinfo=UTC)
    # Guard: confirm these map to different local hours (always true since 12h apart)
    if now_off.astimezone().hour != now_peak.astimezone().hour:
        assert _compute_likely_active(tmp_path, _now=now_off) is False


def test_compute_response_lag_p50_no_file_returns_none(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_response_lag_p50

    assert _compute_response_lag_p50(tmp_path) is None


def test_compute_response_lag_p50_below_cold_start_returns_none(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_response_lag_p50

    # Only 2 replied_explicit rows — below cold-start minimum of 3
    rows = [
        {"audit_id": str(i), "ts": f"2026-05-29T10:0{i}:00+00:00",
         "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit",
                      "state_transitions": [{"to": "replied_explicit",
                                             "at": f"2026-05-29T10:0{i}:30+00:00"}]}}
        for i in range(2)
    ]
    _write_audit_rows(tmp_path, rows)
    assert _compute_response_lag_p50(tmp_path) is None


def test_compute_response_lag_p50_computes_median(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_response_lag_p50

    # Three sends, lags = 60s, 120s, 300s → median = 120s
    rows = [
        {"audit_id": "1", "ts": "2026-05-29T09:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit",
                      "state_transitions": [{"to": "replied_explicit",
                                             "at": "2026-05-29T09:01:00+00:00"}]}},  # 60s
        {"audit_id": "2", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit",
                      "state_transitions": [{"to": "replied_explicit",
                                             "at": "2026-05-29T10:02:00+00:00"}]}},  # 120s
        {"audit_id": "3", "ts": "2026-05-29T11:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit",
                      "state_transitions": [{"to": "replied_explicit",
                                             "at": "2026-05-29T11:05:00+00:00"}]}},  # 300s
    ]
    _write_audit_rows(tmp_path, rows)
    assert _compute_response_lag_p50(tmp_path) == pytest.approx(120.0)


def test_compute_user_presence_cold_start_defaults(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import UserPresence, compute_user_presence

    presence = compute_user_presence(tmp_path)
    assert isinstance(presence, UserPresence)
    assert presence.silence_days == pytest.approx(0.0)
    assert presence.ignore_streak == 0
    assert presence.likely_active is True
    assert presence.response_lag_p50 is None


def test_compute_user_presence_assembles_all_signals(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import compute_user_presence

    # Write one unanswered send to create a streak of 1
    _write_audit_rows(tmp_path, [
        {"audit_id": "1", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "unanswered"}}
    ])
    # Write a recent user turn (1 hour ago) → silence_days ~ 0.04
    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    _write_turn(conv_dir, speaker="user", ts=datetime.now(UTC) - timedelta(hours=1))

    presence = compute_user_presence(tmp_path)
    assert presence.ignore_streak == 1
    assert presence.silence_days < 0.1
    assert presence.likely_active is True  # < 50 turns → permissive
    assert presence.response_lag_p50 is None  # < 3 replied rows


# ===========================================================================
# #225 — incremental/event-driven redesign
# ===========================================================================


# ---------------------------------------------------------------------------
# C1 — no full-history scan on the per-call/notes-tick path
# ---------------------------------------------------------------------------


def test_c1_no_full_scan_when_daily_cadence_not_due(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two calls to compute_user_presence in the same day: the first (cadence
    never-run -> due) may full-scan; the second (cadence not due) must not
    call the full-directory-glob-and-read reader at all."""
    import brain.initiate.user_pattern as up

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    _write_turn(conv_dir, speaker="user", ts=datetime.now(UTC) - timedelta(hours=1))
    _write_audit_rows(tmp_path, [
        {"audit_id": "1", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "unanswered"}}
    ])

    now0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    up.compute_user_presence(tmp_path, _now=now0)  # cadence due (first ever call)

    calls = {"n": 0}
    real_reader = up.read_jsonl_skipping_corrupt

    def spy(path):
        calls["n"] += 1
        return real_reader(path)

    monkeypatch.setattr(up, "read_jsonl_skipping_corrupt", spy)

    now1 = now0 + timedelta(hours=1)  # well within the 24h cadence
    up.compute_user_presence(tmp_path, _now=now1)

    assert calls["n"] == 0, "no full-directory/full-file reader call expected on the cheap path"


def test_c1_self_test_spy_would_catch_the_old_every_call_full_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ST1.5f: the same spy DOES detect a call when the reference (pre-#225)
    full-scan implementations are invoked directly — proving the spy in the
    test above is actually discriminating, not vacuously passing."""
    import brain.initiate.user_pattern as up

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    _write_turn(conv_dir, speaker="user", ts=datetime.now(UTC) - timedelta(hours=1))

    calls = {"n": 0}
    real_reader = up.read_jsonl_skipping_corrupt

    def spy(path):
        calls["n"] += 1
        return real_reader(path)

    monkeypatch.setattr(up, "read_jsonl_skipping_corrupt", spy)

    # The OLD reference implementation (kept as an oracle, no longer on the
    # hot path) calls the full reader every time it's invoked directly.
    up._compute_silence_days(tmp_path)
    up._compute_likely_active(tmp_path)

    assert calls["n"] > 0, "the spy must be capable of detecting a full-scan call"


# ---------------------------------------------------------------------------
# C2 — ignore-streak bounded-window re-scan matches the full-scan oracle
# exactly, sharing the old walk logic unchanged
# ---------------------------------------------------------------------------


def test_c2a_pure_dismiss_run_matches_oracle(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_ignore_streak, _compute_ignore_streak_bounded

    _write_audit_rows(tmp_path, [
        {"audit_id": "1", "ts": "2026-05-29T08:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
        {"audit_id": "2", "ts": "2026-05-29T09:00:00+00:00", "decision": "send_quiet",
         "delivery": {"current_state": "dismissed"}},
        {"audit_id": "3", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
    ])
    oracle = _compute_ignore_streak(tmp_path)
    bounded = _compute_ignore_streak_bounded(tmp_path)
    assert oracle == 3
    assert bounded == oracle


def test_c2b_dismiss_then_reply_reset_matches_oracle(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_ignore_streak, _compute_ignore_streak_bounded

    _write_audit_rows(tmp_path, [
        {"audit_id": "1", "ts": "2026-05-29T08:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit"}},
        {"audit_id": "2", "ts": "2026-05-29T09:00:00+00:00", "decision": "send_quiet",
         "delivery": {"current_state": "dismissed"}},
        {"audit_id": "3", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
    ])
    oracle = _compute_ignore_streak(tmp_path)
    bounded = _compute_ignore_streak_bounded(tmp_path)
    assert oracle == 2
    assert bounded == oracle


def test_c2c_mixed_skip_rows_match_oracle(tmp_path: Path) -> None:
    """pending/delivered/read/non-send-decision rows must be skipped exactly
    as the full-scan oracle skips them (no increment, no break)."""
    from brain.initiate.user_pattern import _compute_ignore_streak, _compute_ignore_streak_bounded

    _write_audit_rows(tmp_path, [
        {"audit_id": "1", "ts": "2026-05-29T07:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "delivered"}},
        {"audit_id": "2", "ts": "2026-05-29T08:00:00+00:00", "decision": "filtered_pre_compose",
         "delivery": {"current_state": "unanswered"}},
        {"audit_id": "3", "ts": "2026-05-29T09:00:00+00:00", "decision": "send_quiet",
         "delivery": {"current_state": "pending"}},
        {"audit_id": "4", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
    ])
    oracle = _compute_ignore_streak(tmp_path)
    bounded = _compute_ignore_streak_bounded(tmp_path)
    assert oracle == 1
    assert bounded == oracle


def test_c2d_out_of_send_order_resolution_matches_oracle(tmp_path: Path) -> None:
    """Multiple rows dismissed, then the OLDEST unresolved row (not the most
    recent) is replied to — must still count the newer dismissed rows."""
    from brain.initiate.user_pattern import _compute_ignore_streak, _compute_ignore_streak_bounded

    _write_audit_rows(tmp_path, [
        {"audit_id": "old", "ts": "2026-05-29T08:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit"}},
        {"audit_id": "mid", "ts": "2026-05-29T09:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
        {"audit_id": "new", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
    ])
    oracle = _compute_ignore_streak(tmp_path)
    bounded = _compute_ignore_streak_bounded(tmp_path)
    assert oracle == 2
    assert bounded == oracle


def test_c2e_reverse_order_resolution_matches_oracle(tmp_path: Path) -> None:
    """A newer row resolves first (in ts order it's just "newer, resolved"),
    then an older, previously-untouched row gets its first dismissal — the
    older row must be excluded, not counted (the walk breaks at the newer
    resolved row before ever reaching it)."""
    from brain.initiate.user_pattern import _compute_ignore_streak, _compute_ignore_streak_bounded

    _write_audit_rows(tmp_path, [
        {"audit_id": "old", "ts": "2026-05-29T08:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
        {"audit_id": "mid", "ts": "2026-05-29T09:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit"}},
        {"audit_id": "new", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
    ])
    oracle = _compute_ignore_streak(tmp_path)
    bounded = _compute_ignore_streak_bounded(tmp_path)
    assert oracle == 1  # "new" counted; "mid" breaks the walk; "old" never reached
    assert bounded == oracle


def test_c2f_single_row_cycling_plus_independent_row_matches_oracle(tmp_path: Path) -> None:
    """One row's CURRENT state reflects having cycled through a reset state
    and back into a streak state; an independent second row is concurrently
    in a streak state. Neither row's contribution may be lost."""
    from brain.initiate.user_pattern import _compute_ignore_streak, _compute_ignore_streak_bounded

    _write_audit_rows(tmp_path, [
        {"audit_id": "cycled", "ts": "2026-05-29T08:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},  # ended up back in a streak state
        {"audit_id": "other", "ts": "2026-05-29T09:00:00+00:00", "decision": "send_quiet",
         "delivery": {"current_state": "unanswered"}},
    ])
    oracle = _compute_ignore_streak(tmp_path)
    bounded = _compute_ignore_streak_bounded(tmp_path)
    assert oracle == 2
    assert bounded == oracle


def test_c2g_tied_send_timestamps_inside_window_both_orderings(tmp_path: Path) -> None:
    """Two rows with byte-identical ts, one later dismissed, one later
    replied — must match the old scan's stable-sort/file-order tiebreak
    exactly, in BOTH possible append orderings of the tied pair."""
    from brain.initiate.user_pattern import _compute_ignore_streak, _compute_ignore_streak_bounded

    tied_ts = "2026-05-29T10:00:00+00:00"

    # Ordering 1: dismissed row appended first, replied row second.
    _write_audit_rows(tmp_path, [
        {"audit_id": "dismissed_row", "ts": tied_ts, "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
        {"audit_id": "replied_row", "ts": tied_ts, "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit"}},
    ])
    oracle_1 = _compute_ignore_streak(tmp_path)
    bounded_1 = _compute_ignore_streak_bounded(tmp_path)
    assert bounded_1 == oracle_1

    # Ordering 2: replied row appended first, dismissed row second.
    _write_audit_rows(tmp_path, [
        {"audit_id": "replied_row", "ts": tied_ts, "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit"}},
        {"audit_id": "dismissed_row", "ts": tied_ts, "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
    ])
    oracle_2 = _compute_ignore_streak(tmp_path)
    bounded_2 = _compute_ignore_streak_bounded(tmp_path)
    assert bounded_2 == oracle_2

    # The two orderings must actually produce DIFFERENT counts — proof the
    # file-order tiebreak genuinely matters here, not a test that happens to
    # be insensitive to ordering.
    assert oracle_1 != oracle_2


def test_c2h_tied_timestamps_straddling_window_boundary(tmp_path: Path) -> None:
    """A tied-ts cluster sits exactly at the naive window_n-row cutoff: some
    members would fall inside a naive window, some just outside. The
    tie-safe extension must capture the whole cluster; a tie-unaware plain
    cutoff must NOT (ST1.5f self-test, proving this test is discriminating).
    """
    from brain.health.jsonl_reader import read_last_n_jsonl_lines
    from brain.initiate.user_pattern import (
        _compute_ignore_streak,
        _compute_ignore_streak_bounded,
        _ignore_streak_from_rows,
    )

    tied_ts = "2026-05-29T10:00:00+00:00"
    rows = [
        # Padding — distinct, earlier timestamps, inert state.
        {"audit_id": "pad0", "ts": "2026-05-29T06:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "delivered"}},
        {"audit_id": "pad1", "ts": "2026-05-29T07:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "delivered"}},
        # Tied cluster at `tied_ts`: reset row FIRST in file order, then two
        # dismissed rows also at `tied_ts`.
        {"audit_id": "reset_at_tie", "ts": tied_ts, "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit"}},
        {"audit_id": "dismissed_at_tie_1", "ts": tied_ts, "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
        {"audit_id": "dismissed_at_tie_2", "ts": tied_ts, "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
        # Newest row, distinct ts, dismissed.
        {"audit_id": "newest", "ts": "2026-05-29T11:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
    ]
    _write_audit_rows(tmp_path, rows)

    window_n = 3  # naive cutoff would keep only the last 3 rows: the two
    # tied dismissed rows + newest, SPLITTING OFF the tied reset row.

    oracle = _compute_ignore_streak(tmp_path)
    bounded = _compute_ignore_streak_bounded(tmp_path, window_n=window_n, window_n_max=50)
    assert bounded == oracle

    # Self-test: a tie-unaware plain window_n-row cutoff must diverge.
    path = tmp_path / "initiate_audit.jsonl"
    naive_lines = read_last_n_jsonl_lines(path, window_n)
    naive_rows = [json.loads(line) for line in naive_lines]
    naive_value = _ignore_streak_from_rows(naive_rows)
    assert naive_value != oracle, (
        "the naive tie-unaware cutoff must diverge from the oracle here — "
        "otherwise this test cannot detect the round-6 straddling defect"
    )


def test_read_ignore_streak_window_row_missing_ts_at_boundary_skips_gracefully(
    tmp_path: Path,
) -> None:
    """Round-1 red-team MINOR finding: `_read_ignore_streak_window` used
    direct `["ts"]` indexing at the tie-boundary check, while the oracle
    walk (`_ignore_streak_from_rows`) uses tolerant `.get("ts")` access and
    simply filters out any row with a missing/falsy ts. A row missing "ts"
    landing exactly at the naive window_n cutoff must be skipped gracefully
    here too (matching the oracle's tolerant filtering), not raise
    KeyError."""
    from brain.initiate.user_pattern import (
        _compute_ignore_streak,
        _compute_ignore_streak_bounded,
        _read_ignore_streak_window,
    )

    rows = [
        {"audit_id": "r0", "ts": "2026-05-29T05:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "delivered"}},
        {"audit_id": "r1", "ts": "2026-05-29T06:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "delivered"}},
        {"audit_id": "r2", "decision": "send_notify",  # NO "ts" at all
         "delivery": {"current_state": "dismissed"}},
        {"audit_id": "r3", "ts": "2026-05-29T08:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
    ]
    _write_audit_rows(tmp_path, rows)

    # window_n=2 -> naive cutoff is rows[-2:], so rows[-n] == rows[2] == the
    # ts-less row is exactly the boundary reference this must not crash on.
    result = _read_ignore_streak_window(tmp_path, 2)
    assert result == rows[2:]  # no extension attempted for a ts-less boundary

    # End-to-end: the bounded production path must not raise either, and
    # must still match the (tolerant) oracle's count.
    oracle = _compute_ignore_streak(tmp_path)
    bounded = _compute_ignore_streak_bounded(tmp_path, window_n=2, window_n_max=4)
    assert bounded == oracle


def test_c19_ignore_streak_bounded_read_does_not_full_scan_large_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bounded production path must never call the full-file reader —
    its cost must scale with window_n, not with total file size."""
    import brain.initiate.user_pattern as up

    rows = []
    for i in range(10_000):
        rows.append({
            "audit_id": f"old_{i}", "ts": f"2026-01-{(i % 27) + 1:02d}T00:00:{i % 60:02d}+00:00",
            "decision": "send_notify", "delivery": {"current_state": "dismissed"},
        })
    # A resolution well within the primary window_n=300 default (near the tail).
    rows.append({
        "audit_id": "resolved", "ts": "2026-06-01T00:00:00+00:00",
        "decision": "send_notify", "delivery": {"current_state": "replied_explicit"},
    })
    for i in range(10):
        rows.append({
            "audit_id": f"recent_{i}", "ts": f"2026-06-01T01:{i:02d}:00+00:00",
            "decision": "send_notify", "delivery": {"current_state": "dismissed"},
        })
    _write_audit_rows(tmp_path, rows)

    calls = {"n": 0}
    real_reader = up.read_jsonl_skipping_corrupt

    def spy(path):
        calls["n"] += 1
        return real_reader(path)

    monkeypatch.setattr(up, "read_jsonl_skipping_corrupt", spy)

    result = up._compute_ignore_streak_bounded(tmp_path)
    assert result == 10  # the 10 trailing dismissed rows, stopping at "resolved"
    assert calls["n"] == 0, "the bounded path must never call the full-file reader"

    # Self-test: the oracle DOES call the full reader for the same file.
    oracle = up._compute_ignore_streak(tmp_path)
    assert calls["n"] > 0
    assert oracle == result


def test_c20a_widen_on_no_resolution_finds_it_and_matches_oracle(tmp_path: Path) -> None:
    """No resolution in the primary window_n rows, but one exists further
    back within window_n_max: the widened read must find it and match the
    oracle's exact count."""
    from brain.initiate.user_pattern import _compute_ignore_streak, _compute_ignore_streak_bounded

    rows = [
        {"audit_id": "resolved", "ts": "2026-05-01T00:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit"}},
    ]
    for i in range(29):
        rows.append({
            "audit_id": f"d_{i}", "ts": f"2026-05-{2 + i:02d}T00:00:00+00:00",
            "decision": "send_notify", "delivery": {"current_state": "dismissed"},
        })
    _write_audit_rows(tmp_path, rows)  # 30 rows total; last 29 all dismissed

    oracle = _compute_ignore_streak(tmp_path)
    assert oracle == 29
    bounded = _compute_ignore_streak_bounded(tmp_path, window_n=5, window_n_max=30)
    assert bounded == oracle


def test_c20a_widen_path_is_actually_exercised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirms the widen step (a second, larger read) actually fires when
    the primary window has no resolution — not merely that the final count
    happens to be right by some other means."""
    import brain.initiate.user_pattern as up

    rows = [
        {"audit_id": "resolved", "ts": "2026-05-01T00:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit"}},
    ]
    for i in range(29):
        rows.append({
            "audit_id": f"d_{i}", "ts": f"2026-05-{2 + i:02d}T00:00:00+00:00",
            "decision": "send_notify", "delivery": {"current_state": "dismissed"},
        })
    _write_audit_rows(tmp_path, rows)

    seen_ns: list[int] = []
    real_window = up._read_ignore_streak_window

    def spy(persona_dir, n):
        seen_ns.append(n)
        return real_window(persona_dir, n)

    monkeypatch.setattr(up, "_read_ignore_streak_window", spy)
    up._compute_ignore_streak_bounded(tmp_path, window_n=5, window_n_max=30)
    assert seen_ns == [5, 30]  # primary window, then the widened read


def test_c20b_saturation_within_window_n_max_logs_and_returns_sane_value(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Even window_n_max contains no resolution: return the count found
    there (a documented, disclosed saturation), not an exception or a
    silently wrong number, and log the condition."""
    from brain.initiate.user_pattern import _compute_ignore_streak_bounded

    rows = []
    for i in range(10):
        rows.append({
            "audit_id": f"d_{i}", "ts": f"2026-05-{1 + i:02d}T00:00:00+00:00",
            "decision": "send_notify", "delivery": {"current_state": "dismissed"},
        })
    _write_audit_rows(tmp_path, rows)  # no resolution anywhere in the file

    with caplog.at_level("DEBUG", logger="brain.initiate.user_pattern"):
        result = _compute_ignore_streak_bounded(tmp_path, window_n=3, window_n_max=10)

    assert result == 10
    assert any(
        "no resolution found within window_n_max" in r.message for r in caplog.records
    )


# ---------------------------------------------------------------------------
# C3 — reply-lag updates incrementally and matches the old median within
# tolerance; cold-start guard still holds
# ---------------------------------------------------------------------------


def test_c3_incremental_reply_lag_matches_old_median_within_tolerance(tmp_path: Path) -> None:
    from brain.initiate import presence_state
    from brain.initiate.audit import append_audit_row, update_audit_state
    from brain.initiate.schemas import AuditRow
    from brain.initiate.user_pattern import _compute_response_lag_p50

    # Lags chosen close together (no heavy-tailed outlier) so the running
    # MEAN and the exact MEDIAN naturally land within the stated tolerance
    # of each other -- the two statistics are only equal by construction
    # for symmetric data, so a skewed sample (e.g. one huge outlier) would
    # legitimately fail this even for a correct implementation.
    lags_and_sends = [
        ("2026-05-29T09:00:00+00:00", 100.0),
        ("2026-05-29T10:00:00+00:00", 110.0),
        ("2026-05-29T11:00:00+00:00", 120.0),
        ("2026-05-29T12:00:00+00:00", 130.0),
        ("2026-05-29T13:00:00+00:00", 140.0),
    ]
    for i, (send_ts, _lag) in enumerate(lags_and_sends):
        row = AuditRow(
            audit_id=f"ia_{i}", candidate_id=f"ic_{i}", ts=send_ts, kind="message",
            subject="x", tone_rendered="x", decision="send_notify",
            decision_reasoning="x", gate_check={"allowed": True, "reason": None},
        )
        append_audit_row(tmp_path, row)

    for i, (send_ts, lag) in enumerate(lags_and_sends):
        send_dt = datetime.fromisoformat(send_ts)
        reply_at = (send_dt + timedelta(seconds=lag)).isoformat()
        update_audit_state(tmp_path, audit_id=f"ia_{i}", new_state="replied_explicit", at=reply_at)

    old_median = _compute_response_lag_p50(tmp_path)
    new_state = presence_state.load_presence_state(tmp_path)
    new_value = new_state.reply_lag_running_mean

    assert old_median is not None
    assert new_value is not None
    tolerance = max(0.05 * old_median, 1.0)
    assert abs(new_value - old_median) <= tolerance
    assert new_state.reply_lag_n == len(lags_and_sends)


def test_c3_cold_start_guard_still_holds(tmp_path: Path) -> None:
    """Below _COLD_START_LAG_MIN folded lags -> response_lag_p50 stays None."""
    from brain.initiate.audit import append_audit_row, update_audit_state
    from brain.initiate.schemas import AuditRow
    from brain.initiate.user_pattern import compute_user_presence

    for i in range(2):  # below the cold-start minimum of 3
        row = AuditRow(
            audit_id=f"ia_{i}", candidate_id=f"ic_{i}", ts=f"2026-05-29T0{i}:00:00+00:00",
            kind="message", subject="x", tone_rendered="x", decision="send_notify",
            decision_reasoning="x", gate_check={"allowed": True, "reason": None},
        )
        append_audit_row(tmp_path, row)
        update_audit_state(
            tmp_path, audit_id=f"ia_{i}", new_state="replied_explicit",
            at=f"2026-05-29T0{i}:01:00+00:00",
        )

    presence = compute_user_presence(tmp_path)
    assert presence.response_lag_p50 is None


# ---------------------------------------------------------------------------
# C4 — daily recompute at most once/24h, cached value matches a fresh
# from-scratch compute (also covers C14's non-UTC fixture requirement)
# ---------------------------------------------------------------------------


def test_c4_daily_recompute_not_reentered_within_same_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import brain.initiate.user_pattern as up

    calls = {"n": 0}
    real_recompute = up._run_daily_presence_recompute

    def spy(persona_dir, *, _now=None):
        calls["n"] += 1
        return real_recompute(persona_dir, _now=_now)

    monkeypatch.setattr(up, "_run_daily_presence_recompute", spy)

    t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    up.compute_user_presence(tmp_path, _now=t0)
    assert calls["n"] == 1

    up.compute_user_presence(tmp_path, _now=t0 + timedelta(hours=23))
    assert calls["n"] == 1  # still not due

    up.compute_user_presence(tmp_path, _now=t0 + timedelta(hours=25))
    assert calls["n"] == 2  # due again


@pytest.mark.parametrize(
    "tz",
    [UTC, timezone(timedelta(hours=9)), timezone(timedelta(hours=-8))],
    ids=["utc", "utc_plus_9", "utc_minus_8"],
)
def test_c4_c14_cached_histogram_matches_fresh_compute_for_current_hour(
    tmp_path: Path, tz
) -> None:
    """C4's correctness-of-cached-value check, parametrized (C14) across a
    UTC and two non-UTC fixed-offset zones so a formula regression (raw-hour
    vs. local-hour) can't hide behind a UTC-only test run."""
    from brain.initiate.user_pattern import _compute_likely_active, compute_user_presence

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    for i in range(60):
        ts = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC) - timedelta(days=i % 30)
        _write_turn(conv_dir, speaker="user", ts=ts)

    now = datetime(2026, 1, 15, 14, 0, 0, tzinfo=tz)

    compute_user_presence(tmp_path, _now=now)  # triggers the due daily recompute

    presence_after = compute_user_presence(tmp_path, _now=now + timedelta(minutes=1))
    fresh_oracle = _compute_likely_active(tmp_path, _now=now + timedelta(minutes=1))
    assert presence_after.likely_active == fresh_oracle


# ---------------------------------------------------------------------------
# C5 — hour-boundary correctness: classification is evaluated against the
# CURRENT hour every call, even though the histogram is cached
# ---------------------------------------------------------------------------


def test_c5_likely_active_flips_across_hour_boundary_without_a_new_recompute(
    tmp_path: Path,
) -> None:
    from brain.initiate.user_pattern import compute_user_presence

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    # 60 turns concentrated ONLY at UTC 14:00 -> that bucket is "active",
    # every other hour (including 02:00) has zero turns -> "inactive".
    for i in range(60):
        ts = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC) - timedelta(days=i % 30)
        _write_turn(conv_dir, speaker="user", ts=ts)

    now_peak = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)
    presence_peak = compute_user_presence(tmp_path, _now=now_peak)  # due -> recompute
    assert presence_peak.likely_active is True

    now_off = now_peak + timedelta(hours=12)  # well within the 24h cadence -> NOT due
    if now_off.astimezone().hour == now_peak.astimezone().hour:
        pytest.skip("12h-apart hours coincide on this host's local zone")
    presence_off = compute_user_presence(tmp_path, _now=now_off)
    assert presence_off.likely_active is False, (
        "classification must reflect the CURRENT hour, not a boolean frozen "
        "at the time of the last recompute"
    )


# ---------------------------------------------------------------------------
# C6 — returning-user staleness window: silence_days reflects a fresh event
# promptly, without waiting for the next daily recompute
# ---------------------------------------------------------------------------


def test_c6_new_inbound_turn_reflected_without_forcing_a_recompute(tmp_path: Path) -> None:
    from brain.initiate import presence_state
    from brain.initiate.user_pattern import compute_user_presence

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    _write_turn(conv_dir, speaker="user", ts=t0 - timedelta(days=5))

    presence_stale = compute_user_presence(tmp_path, _now=t0)  # due -> caches 5-day-old last_seen
    assert presence_stale.silence_days > 4.5

    # A REAL inbound turn arrives (the actual event hook engine.py fires).
    fresh_ts = t0.isoformat()
    presence_state.record_inbound_turn(tmp_path, fresh_ts)

    # Cadence must NOT be due yet (only a moment has passed) -- this call
    # must NOT trigger a new daily recompute to reflect the fresh event.
    presence_fresh = compute_user_presence(tmp_path, _now=t0 + timedelta(seconds=1))
    assert presence_fresh.silence_days < 0.01, (
        "silence_days must reflect the fresh event immediately, not the "
        "stale cached day-count from the last daily recompute"
    )


# ---------------------------------------------------------------------------
# C7 — fail-open under missing/corrupt state (5 sub-cases, each a separate
# assertion) + check_send_allowed equivalence
# ---------------------------------------------------------------------------


def test_c7a_missing_sidecar_yields_permissive_defaults(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import compute_user_presence

    presence = compute_user_presence(tmp_path)
    assert presence.silence_days == pytest.approx(0.0)
    assert presence.ignore_streak == 0
    assert presence.likely_active is True
    assert presence.response_lag_p50 is None


def test_c7b_corrupt_sidecar_yields_permissive_defaults(tmp_path: Path) -> None:
    from brain.bridge.persisted_cadence import CadenceState, save_cadence
    from brain.initiate.user_pattern import compute_user_presence

    (tmp_path / "presence_state.json").write_text("{not valid json", encoding="utf-8")
    # Pre-seed the daily cadence as NOT due, so this call derives straight
    # from the corrupt sidecar rather than triggering a recompute that would
    # overwrite it first.
    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    save_cadence(tmp_path, "presence_daily_cadence.json", CadenceState(next_at=now + timedelta(days=1)))

    presence = compute_user_presence(tmp_path, _now=now)
    assert presence.silence_days == pytest.approx(0.0)
    assert presence.likely_active is True
    assert presence.response_lag_p50 is None


def test_c7c_missing_conversations_dir_defaults_silence_and_active_only(
    tmp_path: Path,
) -> None:
    """Missing active_conversations/ must not wrongly zero out ignore_streak,
    which is independently derived from the (present) audit log."""
    from brain.initiate.user_pattern import compute_user_presence

    _write_audit_rows(tmp_path, [
        {"audit_id": "1", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
    ])
    presence = compute_user_presence(tmp_path)
    assert presence.silence_days == pytest.approx(0.0)
    assert presence.likely_active is True
    assert presence.ignore_streak == 1  # unaffected by the missing conv dir


def test_c7d_missing_audit_file_defaults_streak_and_lag_only(tmp_path: Path) -> None:
    """Missing initiate_audit.jsonl must not corrupt silence_days, which is
    independently derived from the (present) conversation buffer."""
    from brain.initiate.user_pattern import compute_user_presence

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    _write_turn(conv_dir, speaker="user", ts=datetime.now(UTC) - timedelta(hours=1))

    presence = compute_user_presence(tmp_path)
    assert presence.ignore_streak == 0
    assert presence.response_lag_p50 is None
    assert presence.silence_days < 0.1  # unaffected by the missing audit file


def test_c7e_unexpected_exception_in_a_computation_path_yields_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import brain.initiate.user_pattern as up
    from brain.initiate import presence_state

    def boom(_persona_dir):
        raise RuntimeError("boom")

    monkeypatch.setattr(presence_state, "load_presence_state", boom)

    presence = up.compute_user_presence(tmp_path)
    assert presence.silence_days == pytest.approx(0.0)
    assert presence.ignore_streak == 0
    assert presence.likely_active is True
    assert presence.response_lag_p50 is None


def test_c7_fail_open_presence_matches_none_presence_in_check_send_allowed(
    tmp_path: Path,
) -> None:
    """Feeding the fail-open UserPresence through check_send_allowed for a
    representative input produces the same allow/deny outcome as
    user_presence=None."""
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo

    from brain.initiate.gates import check_send_allowed
    from brain.initiate.user_pattern import compute_user_presence

    now = _dt(2026, 5, 29, 12, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    fail_open_presence = compute_user_presence(tmp_path, _now=now)

    allowed_a, reason_a = check_send_allowed(
        tmp_path, urgency="notify", now=now, user_presence=None
    )
    allowed_b, reason_b = check_send_allowed(
        tmp_path, urgency="notify", now=now, user_presence=fail_open_presence
    )
    assert (allowed_a, reason_a) == (allowed_b, reason_b)


# ---------------------------------------------------------------------------
# C8 — file-removal / underlying-data-disappearance edge case
# ---------------------------------------------------------------------------


def test_c8_source_removal_after_populated_fails_open_at_next_recompute(
    tmp_path: Path,
) -> None:
    from brain.initiate.user_pattern import compute_user_presence

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    # 60 turns concentrated at hour 14 so likely_active's histogram is
    # populated (non-None) before the removal.
    for i in range(60):
        ts = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC) - timedelta(days=i % 30)
        _write_turn(conv_dir, speaker="user", ts=ts)
    _write_audit_rows(tmp_path, [
        {"audit_id": "1", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "dismissed"}},
    ])

    now_peak = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)
    presence_before = compute_user_presence(tmp_path, _now=now_peak)
    assert presence_before.likely_active is True
    assert presence_before.ignore_streak == 1

    # The underlying sources disappear (persona reset / tmp dir cleared).
    for f in conv_dir.glob("*.jsonl"):
        f.unlink()
    (tmp_path / "initiate_audit.jsonl").unlink()

    # ignore_streak is never cached -- it must reflect the removal on the
    # very next call, cadence or no cadence.
    presence_immediately_after = compute_user_presence(tmp_path, _now=now_peak + timedelta(seconds=1))
    assert presence_immediately_after.ignore_streak == 0

    # silence_days/likely_active are cached at daily granularity; force the
    # cadence due again so the removal is actually noticed, and assert the
    # result is fail-open (not an exception, not still trusting the vanished
    # histogram as if it were still valid).
    presence_after_next_recompute = compute_user_presence(
        tmp_path, _now=now_peak + timedelta(hours=25)
    )
    assert presence_after_next_recompute.likely_active is True  # hour_counts reset to None
    # silence_days can only ever grow (never shrink) once its source
    # vanishes -- per gates.py, a HIGHER silence_days only loosens gates
    # further, so "no less permissive than the fail-open default" holds
    # even though the exact cached last_seen_ts isn't reset to None.
    assert presence_after_next_recompute.silence_days >= presence_before.silence_days


def test_c8_reply_lag_fails_open_after_source_removed_post_bootstrap(tmp_path: Path) -> None:
    """#225 stage-6 round-1 red-team MAJOR finding: `response_lag_p50` never
    failed open once initiate_audit.jsonl disappeared AFTER a real
    reply-lag bootstrap had already completed with data — `do_bootstrap`
    (`not pre_scan_state.bootstrapped`) is permanently False once
    bootstrapped, so nothing re-checked whether the source file the
    bootstrap seeded from still existed, and the daily recompute's merge
    just re-persisted the stale reply_lag_running_mean/reply_lag_n forever.

    Reproduces the reviewer's exact repro: seed 3 replied_explicit rows
    with a 1000s lag (well above the >=600s threshold that actively
    tightens check_send_allowed's gate gaps), let the bootstrap seed
    reply_lag_running_mean/reply_lag_n from them, delete
    initiate_audit.jsonl, force a daily recompute 25h later, and assert
    response_lag_p50 comes back None (the cold-start default) instead of
    the stale 1000.0 -- this is the test gap C8's own existing test
    (test_c8_source_removal_after_populated_fails_open_at_next_recompute)
    left open: it never seeded reply-lag data or asserted on
    response_lag_p50 at all.
    """
    from brain.initiate import presence_state
    from brain.initiate.audit import append_audit_row, update_audit_state
    from brain.initiate.schemas import AuditRow
    from brain.initiate.user_pattern import compute_user_presence

    t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)

    for i in range(3):
        send_ts = (t0 - timedelta(hours=5 - i)).isoformat()
        row = AuditRow(
            audit_id=f"ia_{i}", candidate_id=f"ic_{i}", ts=send_ts,
            kind="message", subject="x", tone_rendered="x", decision="send_notify",
            decision_reasoning="x", gate_check={"allowed": True, "reason": None},
        )
        append_audit_row(tmp_path, row)
        reply_at = (datetime.fromisoformat(send_ts) + timedelta(seconds=1000)).isoformat()
        update_audit_state(tmp_path, audit_id=f"ia_{i}", new_state="replied_explicit", at=reply_at)

    # Force the daily recompute -> the one-time bootstrap runs and seeds
    # reply_lag_running_mean/reply_lag_n from the 3 seeded rows.
    presence_before = compute_user_presence(tmp_path, _now=t0)
    assert presence_before.response_lag_p50 == pytest.approx(1000.0)
    state_before = presence_state.load_presence_state(tmp_path)
    assert state_before.bootstrapped is True
    assert state_before.reply_lag_n == 3

    # The underlying source disappears (persona reset / tmp dir cleared).
    (tmp_path / "initiate_audit.jsonl").unlink()

    # Force the daily cadence due again so the removal is actually noticed
    # by a recompute (ignore_streak would notice immediately regardless of
    # cadence, but reply-lag's fields are cached at daily granularity).
    presence_after = compute_user_presence(tmp_path, _now=t0 + timedelta(hours=25))
    assert presence_after.response_lag_p50 is None, (
        "response_lag_p50 must fail open once the source file it was "
        "computed from has disappeared, not keep serving the stale value "
        "computed while it still existed"
    )
    state_after = presence_state.load_presence_state(tmp_path)
    assert state_after.reply_lag_n == 0
    assert state_after.reply_lag_running_mean is None
    # bootstrapped resets to False too, so a fresh persona later reusing
    # the same directory (the file reappearing) can re-bootstrap correctly
    # instead of being permanently skipped.
    assert state_after.bootstrapped is False


# ---------------------------------------------------------------------------
# C10 — behavior parity: same underlying data drives the same gate decision
# through the new mechanism and the old from-scratch computation
# ---------------------------------------------------------------------------


def test_c10_same_input_same_gate_decision_old_vs_new_mechanism(tmp_path: Path) -> None:
    from brain.initiate.audit import append_audit_row, update_audit_state
    from brain.initiate.gates import check_send_allowed
    from brain.initiate.schemas import AuditRow
    from brain.initiate.user_pattern import (
        UserPresence,
        _compute_ignore_streak,
        _compute_likely_active,
        _compute_response_lag_p50,
        _compute_silence_days,
        compute_user_presence,
    )

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    _write_turn(conv_dir, speaker="user", ts=now - timedelta(hours=2))

    for i in range(3):
        row = AuditRow(
            audit_id=f"ia_{i}", candidate_id=f"ic_{i}", ts=(now - timedelta(hours=10 - i)).isoformat(),
            kind="message", subject="x", tone_rendered="x", decision="send_notify",
            decision_reasoning="x", gate_check={"allowed": True, "reason": None},
        )
        append_audit_row(tmp_path, row)
        update_audit_state(
            tmp_path, audit_id=f"ia_{i}", new_state="replied_explicit",
            at=(now - timedelta(hours=10 - i) + timedelta(seconds=120)).isoformat(),
        )
    # One more, unresolved (streak of 1) -- needs an explicit transition to
    # "unanswered", since a freshly-appended row's delivery block is None
    # (skipped by the walk, not counted as a streak) until transitioned.
    row = AuditRow(
        audit_id="ia_last", candidate_id="ic_last", ts=(now - timedelta(hours=1)).isoformat(),
        kind="message", subject="x", tone_rendered="x", decision="send_notify",
        decision_reasoning="x", gate_check={"allowed": True, "reason": None},
    )
    append_audit_row(tmp_path, row)
    update_audit_state(
        tmp_path, audit_id="ia_last", new_state="unanswered",
        at=(now - timedelta(minutes=30)).isoformat(),
    )

    # New mechanism, settled.
    new_presence = compute_user_presence(tmp_path, _now=now)

    # Old from-scratch mechanism, over the IDENTICAL final on-disk data.
    old_presence = UserPresence(
        silence_days=_compute_silence_days(tmp_path, _now=now),
        ignore_streak=_compute_ignore_streak(tmp_path),
        likely_active=_compute_likely_active(tmp_path, _now=now),
        response_lag_p50=_compute_response_lag_p50(tmp_path),
    )

    allowed_new, _ = check_send_allowed(tmp_path, urgency="notify", now=now, user_presence=new_presence)
    allowed_old, _ = check_send_allowed(tmp_path, urgency="notify", now=now, user_presence=old_presence)
    assert allowed_new == allowed_old
    assert new_presence.ignore_streak == old_presence.ignore_streak == 1


# ---------------------------------------------------------------------------
# C15 — bootstrap runs at most once, even with zero historical data
# ---------------------------------------------------------------------------


def test_c15_bootstrap_runs_at_most_once_even_with_zero_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import brain.initiate.user_pattern as up

    calls = {"n": 0}
    real_read_lags = up._read_valid_reply_lags

    def spy(persona_dir):
        calls["n"] += 1
        return real_read_lags(persona_dir)

    monkeypatch.setattr(up, "_read_valid_reply_lags", spy)

    t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    up.compute_user_presence(tmp_path, _now=t0)  # day 1 -> due, bootstrap fires
    up.compute_user_presence(tmp_path, _now=t0 + timedelta(days=1, hours=1))  # day 2 -> due again

    assert calls["n"] == 1, "the bootstrap scan must fire at most once, ever, per persona"

    from brain.initiate import presence_state
    assert presence_state.load_presence_state(tmp_path).bootstrapped is True
