"""user_pattern.py — infer user availability and responsiveness from audit + buffer.

Produces a UserPresence snapshot each initiate-review tick, consumed by
check_send_allowed to adjust gate thresholds in real time.

#225 redesign: the four signals used to each re-read active_conversations/*.
jsonl and/or initiate_audit.jsonl from scratch on EVERY call. As of this
redesign, only ignore_streak still reads initiate_audit.jsonl on every call
— and only a small bounded tail window of it, never the whole file (see
``_compute_ignore_streak_bounded``). silence_days and likely_active are
derived from a small persisted sidecar (``brain.initiate.presence_state``)
that's rebuilt from a full scan at most once/24h; response_lag_p50 is
folded incrementally at the point a reply is recorded
(``brain.initiate.audit.update_audit_state``'s reply-lag hook). The
original from-scratch ``_compute_*`` helpers below are KEPT as reference/
bootstrap/oracle implementations — some are refactored minimally (their
scan bodies extracted into shared helpers) but their own external behavior
is unchanged, and they remain the independent comparison oracles the tests
check the new incremental paths against.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.bridge.persisted_cadence import advance, is_due, load_cadence, save_cadence
from brain.health.jsonl_reader import read_jsonl_skipping_corrupt, read_last_n_jsonl_lines
from brain.initiate import presence_state

log = logging.getLogger(__name__)

_ACTIVE_CONVERSATIONS_DIR = "active_conversations"
_SCHEDULE_LOOKBACK_DAYS = 30
_SCHEDULE_MIN_TURNS = 50
_SCHEDULE_ACTIVE_PERCENTILE = 0.20
_COLD_START_LAG_MIN = 3
_COLD_START_STREAK_AUDIT_FILENAME = "initiate_audit.jsonl"
_SEND_DECISIONS = frozenset({"send_notify", "send_quiet"})
_STREAK_STATES = frozenset({"unanswered", "dismissed"})
_RESET_STATES = frozenset({"replied_explicit", "acknowledged_unclear"})

# Daily recompute cadence (silence-days last-seen bootstrap, likely-active
# histogram rebuild, and the one-time reply-lag bootstrap) — reuses the
# generic persisted_cadence.py mechanism (#225).
_PRESENCE_DAILY_CADENCE_FILENAME = "presence_daily_cadence.json"
_PRESENCE_DAILY_INTERVAL_S = 86400.0

# Ignore-streak's bounded-window re-scan (#225 round 6/7). Fallback after
# four consecutive rounds of red-team found distinct genuine defects in
# every attempt to translate the old backward scan into persisted running
# state (see changes/user-presence-incremental-225/decisions.md) — this
# reverts to a direct, stateless, bounded re-scan every call instead.
_TIE_MARGIN = 50  # generous relative to cap_per_tick's default of 3


def cap_per_tick_default() -> int:
    """The hardcoded cap_per_tick default (brain.bridge.supervisor.py's
    ``_run_initiate_review_tick`` fallback and
    brain.initiate.review.run_initiate_review_tick's own default parameter).

    Referenced by ``_read_ignore_streak_window``'s safety-factor assert so a
    future increase to cap_per_tick past ``_TIE_MARGIN``'s headroom fails
    loudly instead of silently reopening the round-6/7 tie-straddling bug.
    """
    return 3


@dataclass(frozen=True)
class UserPresence:
    silence_days: float           # days since last inbound chat turn; 0.0 when uncertain
    ignore_streak: int            # consecutive unanswered/dismissed proactive sends
    likely_active: bool           # within inferred active window; True when unknown
    response_lag_p50: float | None  # median response lag in seconds; None = cold start


# ---------------------------------------------------------------------------
# silence-days — reference implementation (kept, unchanged; not used by
# compute_user_presence anymore, which derives silence_days from
# presence_state.last_seen_ts instead — see _daily_scan_conversations /
# _run_daily_presence_recompute below for the incremental replacement).
# ---------------------------------------------------------------------------


def _compute_silence_days(persona_dir: Path, *, _now: datetime | None = None) -> float:
    """Return days since the most recent inbound (non-companion) chat turn.

    Returns 0.0 when no buffer files exist or no inbound turns are found —
    uncertainty stays permissive.
    """
    conversations_dir = persona_dir / _ACTIVE_CONVERSATIONS_DIR
    if not conversations_dir.exists():
        return 0.0

    now = _now or datetime.now(UTC)
    companion_name = persona_dir.name.lower()
    latest_ts: datetime | None = None

    for jsonl_file in conversations_dir.glob("*.jsonl"):
        for row in read_jsonl_skipping_corrupt(jsonl_file):
            speaker = str(row.get("speaker", "")).lower()
            if speaker == companion_name:
                continue
            # A compaction `summary` row is not a real user turn — its
            # compaction-time ts must not corrupt the "last user message" signal.
            if speaker == "summary":
                continue
            ts_str = row.get("ts")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if latest_ts is None or ts > latest_ts:
                    latest_ts = ts
            except (ValueError, TypeError):
                continue

    if latest_ts is None:
        return 0.0
    return max(0.0, (now - latest_ts).total_seconds() / 86400.0)


# ---------------------------------------------------------------------------
# ignore-streak — walk logic shared, unchanged, between the full-scan oracle
# and the new bounded-window production path (#225 round 6/7).
# ---------------------------------------------------------------------------


def _ignore_streak_from_rows(rows: list[dict]) -> int:
    """Count consecutive unanswered/dismissed proactive sends, most recent
    first, stopping at the first replied_explicit/acknowledged_unclear row.

    Extracted verbatim from what `_compute_ignore_streak` has always done —
    byte-identical walk, operating on plain dicts (matching what
    read_jsonl_skipping_corrupt/read_last_n_jsonl_lines produce). Shared,
    unchanged, between `_compute_ignore_streak`'s full-file oracle and
    `_compute_ignore_streak_bounded`'s bounded production path so exact
    parity holds by construction, not by re-derivation.
    """
    rows_desc = sorted((r for r in rows if r.get("ts")), key=lambda r: r["ts"], reverse=True)

    streak = 0
    for row in rows_desc:
        if row.get("decision") not in _SEND_DECISIONS:
            continue
        state = (row.get("delivery") or {}).get("current_state", "")
        if state in _STREAK_STATES:
            streak += 1
        elif state in _RESET_STATES:
            break
    return streak


def _compute_ignore_streak(persona_dir: Path) -> int:
    """Full-file entrypoint — UNCHANGED behavior. Retained as-is: it has no
    callers outside this module today, and it is the independent
    comparison oracle `_compute_ignore_streak_bounded`'s tests check parity
    against. Never modified again, so the oracle and the production path
    can't share a bug.
    """
    audit_path = persona_dir / _COLD_START_STREAK_AUDIT_FILENAME
    if not audit_path.exists():
        return 0

    rows = read_jsonl_skipping_corrupt(audit_path)
    return _ignore_streak_from_rows(rows)


def _parse_lines_skipping_corrupt(lines: list[str]) -> list[dict]:
    """Parse raw JSONL text lines with the same skip/log/continue discipline
    `read_jsonl_skipping_corrupt` uses, for lines already obtained via a
    bounded tail read (so this never re-reads the file itself)."""
    rows: list[dict] = []
    for raw in lines:
        stripped = raw.rstrip("\r\n")
        if not stripped.strip():
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            log.warning(
                "skipping malformed jsonl line in bounded ignore-streak read: %s | content: %r",
                exc,
                stripped[:201],
            )
            continue
        if isinstance(data, dict):
            rows.append(data)
        else:
            log.warning(
                "skipping non-dict jsonl line in bounded ignore-streak read (value type=%s)",
                type(data).__name__,
            )
    return rows


def _read_ignore_streak_window(persona_dir: Path, n: int) -> list[dict]:
    """Read the most recent n rows, extended backward as needed so a
    send-ts-tied cluster straddling the n-row cutoff is never split.

    `run_initiate_review_tick` (cap_per_tick=3 default) routinely stamps
    several same-tick candidates with an identical `ts`, so a naive
    line-count cutoff can fall inside such a tied cluster — some members
    inside the window, some just outside — breaking exact parity with the
    full-scan oracle's stable-sort/file-order tiebreak. Reading a small,
    generous safety margin beyond the nominal window and extending
    backward, in-memory, until the full tied group at the cutoff is
    included closes that gap by construction.
    """
    path = persona_dir / _COLD_START_STREAK_AUDIT_FILENAME
    raw_lines = read_last_n_jsonl_lines(path, n + _TIE_MARGIN)
    rows = _parse_lines_skipping_corrupt(raw_lines)
    if len(rows) <= n:
        return rows  # whole read fits inside the nominal window; no tie risk
    # rows are in original file order (oldest first); the naive cutoff would
    # keep only rows[-n:]. Extend the start index backward while the
    # boundary row's ts matches the row just before it (a tied cluster).
    #
    # Tolerant `.get("ts")` access (round-1 red-team MINOR finding): a row
    # missing "ts" at the boundary must be skipped gracefully here, exactly
    # like `_ignore_streak_from_rows`'s oracle walk does (it filters out any
    # row with a falsy/missing "ts" before ever comparing timestamps) —
    # direct `["ts"]` indexing would instead raise KeyError. A missing
    # boundary_ts can't be a genuine tie (the oracle never tie-groups a
    # ts-less row with anything), so the extension is skipped in that case
    # rather than treating two ts-less rows as "tied".
    boundary_ts = rows[-n].get("ts")
    start = len(rows) - n
    while start > 0 and boundary_ts is not None and rows[start - 1].get("ts") == boundary_ts:
        start -= 1
    if start == 0 and boundary_ts is not None and rows[0].get("ts") == boundary_ts:
        # The tied cluster extends to (or past) the edge of what we read.
        # This is only unreachable today because cap_per_tick is hardcoded
        # to 3 — _TIE_MARGIN=50 is a 16x safety factor against that specific
        # constant, not a structural guarantee. If cap_per_tick ever becomes
        # configurable past this factor, this assertion turns a silent
        # reopening of the tie-straddling bug into a loud, immediate failure.
        assert cap_per_tick_default() * 10 <= _TIE_MARGIN, (
            "cap_per_tick has grown past _TIE_MARGIN's safety factor; "
            "the tie-safe window can no longer guarantee correctness"
        )
        log.debug(
            "ignore-streak: tie-margin (%d) exhausted at boundary ts=%s",
            _TIE_MARGIN,
            boundary_ts,
        )
    return rows[start:]


def _compute_ignore_streak_bounded(
    persona_dir: Path, *, window_n: int = 300, window_n_max: int = 3000
) -> int:
    """Bounded-window production path for ignore_streak (#225 round 6/7).

    Reads only a tie-safe tail window of initiate_audit.jsonl (never the
    whole file) and feeds it into `_ignore_streak_from_rows` — the SAME walk
    `_compute_ignore_streak`'s full scan uses — so exact parity holds by
    construction, not re-derivation. Widens once to window_n_max if the
    primary window contains no resolution (round 6's "don't silently
    undercount" rule); if even that finds none, returns the (disclosed,
    logged) saturation count rather than falling back to an unbounded read.

    No persisted state, no event hook, no lock, no bootstrap, no cadence
    gating — this signal has zero concurrency surface, immune by
    construction to the class of bug found translating this scan into
    running state across rounds 2-5.
    """
    audit_path = persona_dir / _COLD_START_STREAK_AUDIT_FILENAME
    if not audit_path.exists():
        return 0

    rows = _read_ignore_streak_window(persona_dir, window_n)
    if not _has_resolution(rows):
        rows = _read_ignore_streak_window(persona_dir, window_n_max)
        if not _has_resolution(rows):
            log.debug(
                "ignore-streak: no resolution found within window_n_max=%d rows; "
                "returning saturated count",
                window_n_max,
            )
    return _ignore_streak_from_rows(rows)


def _has_resolution(rows: list[dict]) -> bool:
    return any((r.get("delivery") or {}).get("current_state") in _RESET_STATES for r in rows)


# ---------------------------------------------------------------------------
# likely-active-at-hour — histogram-building scan extracted so the daily
# recompute can reuse it (returns hour_counts/threshold); the bool-returning
# `_compute_likely_active` is kept as a thin wrapper — its own external
# behavior (and the direct unit tests exercising it) is unchanged.
# ---------------------------------------------------------------------------


def _build_hour_histogram(
    persona_dir: Path, *, _now: datetime | None = None
) -> tuple[list[int] | None, int | None]:
    """Scan active_conversations/*.jsonl and build the 24-bucket local-hour
    histogram + active-percentile threshold used by likely-active-at-hour.

    Returns (None, None) when there's no conversations dir or fewer than
    _SCHEDULE_MIN_TURNS turns in the _SCHEDULE_LOOKBACK_DAYS window
    (insufficient data -> permissive default upstream).
    """
    conversations_dir = persona_dir / _ACTIVE_CONVERSATIONS_DIR
    if not conversations_dir.exists():
        return None, None
    now = _now or datetime.now(UTC)
    cutoff = now - timedelta(days=_SCHEDULE_LOOKBACK_DAYS)
    hour_counts: list[int] = [0] * 24
    total = 0
    for jsonl_file in conversations_dir.glob("*.jsonl"):
        for row in read_jsonl_skipping_corrupt(jsonl_file):
            speaker = str(row.get("speaker", "")).lower()
            if speaker == persona_dir.name.lower():
                continue
            # A compaction `summary` row is not a real user turn — exclude from
            # hour-distribution so the active-window inference isn't skewed by
            # compaction-time timestamps.
            if speaker == "summary":
                continue
            ts_str = row.get("ts")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if ts < cutoff:
                    continue
                hour_counts[ts.astimezone().hour] += 1
                total += 1
            except (ValueError, TypeError):
                continue
    if total < _SCHEDULE_MIN_TURNS:
        return None, None
    sorted_counts = sorted(hour_counts)
    threshold = max(sorted_counts[int(len(sorted_counts) * _SCHEDULE_ACTIVE_PERCENTILE)], 1)
    return hour_counts, threshold


def _compute_likely_active(persona_dir: Path, *, _now: datetime | None = None) -> bool:
    """Return True if current hour is within the user's inferred active window.

    Reference implementation — a thin wrapper over `_build_hour_histogram`.
    Kept as-is (unchanged external behavior) as the from-scratch comparison
    oracle for C4/C5's tests.
    """
    hour_counts, threshold = _build_hour_histogram(persona_dir, _now=_now)
    if hour_counts is None or threshold is None:
        return True
    now = _now or datetime.now(UTC)
    return hour_counts[now.astimezone().hour] >= threshold


# ---------------------------------------------------------------------------
# median reply-lag — valid-lag extraction shared between the full-scan
# oracle and the daily recompute's one-time bootstrap.
# ---------------------------------------------------------------------------


def _read_valid_reply_lags(persona_dir: Path) -> list[float]:
    """Return every valid (non-negative) reply lag in seconds from
    replied_explicit audit rows. Shared by `_compute_response_lag_p50`
    (the full-scan oracle) and the daily recompute's one-time historical
    bootstrap of `reply_lag_running_mean`/`reply_lag_n`."""
    audit_path = persona_dir / _COLD_START_STREAK_AUDIT_FILENAME
    if not audit_path.exists():
        return []

    lags: list[float] = []
    for row in read_jsonl_skipping_corrupt(audit_path):
        if row.get("decision") not in _SEND_DECISIONS:
            continue
        delivery = row.get("delivery") or {}
        if delivery.get("current_state") != "replied_explicit":
            continue
        send_ts_str = row.get("ts", "")
        transitions = delivery.get("state_transitions", [])
        reply_ts_str = next(
            (t["at"] for t in reversed(transitions) if t.get("to") == "replied_explicit"),
            None,
        )
        if not reply_ts_str or not send_ts_str:
            continue
        try:
            send_ts = datetime.fromisoformat(send_ts_str)
            reply_ts = datetime.fromisoformat(reply_ts_str)
            if send_ts.tzinfo is None:
                send_ts = send_ts.replace(tzinfo=UTC)
            if reply_ts.tzinfo is None:
                reply_ts = reply_ts.replace(tzinfo=UTC)
            lag = (reply_ts - send_ts).total_seconds()
            if lag >= 0:
                lags.append(lag)
        except (ValueError, TypeError):
            continue
    return lags


def _compute_response_lag_p50(persona_dir: Path) -> float | None:
    """Return median response lag in seconds from replied_explicit audit rows.

    Returns None when fewer than _COLD_START_LAG_MIN rows with confirmed
    replies exist (cold-start guard). Reference implementation — a thin
    wrapper over `_read_valid_reply_lags`, kept as the full-scan comparison
    oracle for C3's tolerance test.
    """
    lags = _read_valid_reply_lags(persona_dir)

    if len(lags) < _COLD_START_LAG_MIN:
        return None

    lags = sorted(lags)
    mid = len(lags) // 2
    return lags[mid] if len(lags) % 2 == 1 else (lags[mid - 1] + lags[mid]) / 2.0


# ---------------------------------------------------------------------------
# Daily recompute — the one full scan that survives, gated to at most
# once/24h via persisted_cadence. Scan phase runs UNLOCKED; only the final
# merge into PresenceState is lock-guarded (#225 round 2 fix 1).
# ---------------------------------------------------------------------------


def _daily_scan_conversations(
    persona_dir: Path, *, _now: datetime | None = None
) -> tuple[list[int] | None, int | None, str | None]:
    """One pass over active_conversations/*.jsonl for the daily recompute:
    builds the likely-active hour histogram (30-day lookback) AND finds the
    max inbound-turn ts across ALL history (no lookback cutoff) in the same
    loop, so silence-days' bootstrap doesn't need a second full scan.

    Returns (hour_counts, active_threshold, latest_ts_iso) — any of which
    may be None (no conversations dir, or no qualifying rows).
    """
    conversations_dir = persona_dir / _ACTIVE_CONVERSATIONS_DIR
    if not conversations_dir.exists():
        return None, None, None
    now = _now or datetime.now(UTC)
    cutoff = now - timedelta(days=_SCHEDULE_LOOKBACK_DAYS)
    companion_name = persona_dir.name.lower()
    hour_counts: list[int] = [0] * 24
    total = 0
    latest_ts: datetime | None = None

    for jsonl_file in conversations_dir.glob("*.jsonl"):
        for row in read_jsonl_skipping_corrupt(jsonl_file):
            speaker = str(row.get("speaker", "")).lower()
            if speaker == companion_name or speaker == "summary":
                continue
            ts_str = row.get("ts")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
            except (ValueError, TypeError):
                continue
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
            if ts < cutoff:
                continue
            hour_counts[ts.astimezone().hour] += 1
            total += 1

    threshold: int | None = None
    if total >= _SCHEDULE_MIN_TURNS:
        sorted_counts = sorted(hour_counts)
        threshold = max(sorted_counts[int(len(sorted_counts) * _SCHEDULE_ACTIVE_PERCENTILE)], 1)
    else:
        hour_counts = None  # type: ignore[assignment]

    latest_ts_iso = latest_ts.isoformat() if latest_ts is not None else None
    return hour_counts, threshold, latest_ts_iso


def _run_daily_presence_recompute(persona_dir: Path, *, _now: datetime | None = None) -> None:
    """Run the once-daily bounded recompute: silence-days last-seen
    discovery, likely-active histogram rebuild, and (once ever) the
    reply-lag bootstrap.

    Scan phase runs UNLOCKED (matches today's un-locked full-scan reads —
    read_jsonl_skipping_corrupt already tolerates a concurrent writer via
    its corrupt-line-skip discipline). Merge phase is a small,
    presence-lock-guarded, version-reconciled read-merge-write so a live
    event (record_inbound_turn/fold_reply_lag) that fires during the scan
    window is never silently clobbered (#225 round 3 fix).
    """
    now = _now or datetime.now(UTC)

    # Snapshot version + bootstrap-need BEFORE the unlocked scan starts, so
    # the merge phase can detect whether anything wrote since.
    pre_scan_state = presence_state.load_presence_state(persona_dir)
    v0 = pre_scan_state.version
    do_bootstrap = not pre_scan_state.bootstrapped

    # --- unlocked scan phase (the one full scan/day) ---
    hour_counts, active_threshold, scanned_latest_ts = _daily_scan_conversations(
        persona_dir, _now=now
    )

    bootstrap_mean: float | None = None
    bootstrap_n = 0
    if do_bootstrap:
        lags = _read_valid_reply_lags(persona_dir)
        bootstrap_n = len(lags)
        bootstrap_mean = (sum(lags) / bootstrap_n) if bootstrap_n else None

    # --- locked merge phase: small read-merge-write, version-reconciled ---
    with presence_state._presence_lock(persona_dir):  # noqa: SLF001
        fresh = presence_state.load_presence_state(persona_dir)

        # last_seen_ts: always a safe max-merge — no version check needed,
        # taking the max of two timestamps never loses information.
        candidates = [t for t in (fresh.last_seen_ts, scanned_latest_ts) if t]
        merged_last_seen = max(candidates) if candidates else None

        # hour_counts/active_threshold: no other accessor ever writes these
        # fields, so no race is possible here regardless of version.
        merged_hour_counts = tuple(hour_counts) if hour_counts is not None else None
        merged_active_threshold = active_threshold

        # #225 stage-6 round-1 red-team MAJOR finding (C8): `do_bootstrap`
        # (`not pre_scan_state.bootstrapped`) is permanently False once a
        # real bootstrap has run, so nothing ever re-checked whether the
        # source file that seeded reply_lag_running_mean/reply_lag_n still
        # exists — a persona reset that removes initiate_audit.jsonl left
        # those fields serving a stale value forever. Fixed by checking
        # existence on EVERY recompute, mirroring
        # _daily_scan_conversations/_read_valid_reply_lags's own "check the
        # source exists, every call" discipline, instead of trusting a
        # one-time bootstrap flag to stay valid forever.
        #
        # Checked fresh here, inside the lock, immediately before the
        # decision (not earlier, in the unlocked scan phase) — this is what
        # lets the reset below apply with NO version-check needed, unlike
        # the bootstrap-snapshot branch just below it: the only path that
        # can fold a LIVE reply-lag update, update_audit_state
        # (brain/initiate/audit.py), itself requires the audit file to
        # exist (its own `if not path.exists(): return`) and folds via
        # `presence_state.fold_reply_lag`, which takes this SAME presence
        # lock. So a concurrent fold can never interleave with this
        # check-then-write: it either already completed before we took the
        # lock (and `fresh`, loaded above, already reflects it — so
        # reply_lag_n > 0 below is real, current data, not stale) or it is
        # blocked behind this lock right now and will correctly fold onto
        # whatever we write here, reset or not.
        #
        # Gated on `fresh.reply_lag_n > 0` (evidence real historical data
        # was actually captured) so this can't fight C15's "bootstrap runs
        # at most once ever, even with zero historical data" contract: a
        # persona that has NEVER had an audit file (do_bootstrap True,
        # reply_lag_n already 0) must keep `bootstrapped=True` after its
        # one bootstrap attempt, not be treated as a same-day "removal" —
        # that path is left to the do_bootstrap branch below, untouched.
        audit_path = persona_dir / _COLD_START_STREAK_AUDIT_FILENAME
        audit_missing = not audit_path.exists()

        if not do_bootstrap and audit_missing and fresh.reply_lag_n > 0:
            merged_reply_lag_mean = None
            merged_reply_lag_n = 0
            merged_bootstrapped = False
        elif do_bootstrap:
            # Bootstrap-only fields: applied only the first time, and only
            # if no live write raced the scan (fresh.version == v0). If
            # something did race, the bootstrap snapshot for these fields
            # is discarded — losing a live real-time update to satisfy a
            # one-time historical seed would be the wrong trade;
            # undercounting the retrospective seed is strictly on the
            # permissive side of the fail-open contract. bootstrapped is
            # set True regardless (C15: at most once ever, even zero data).
            if fresh.version == v0:
                merged_reply_lag_mean = bootstrap_mean
                merged_reply_lag_n = bootstrap_n
            else:
                merged_reply_lag_mean = fresh.reply_lag_running_mean
                merged_reply_lag_n = fresh.reply_lag_n
            merged_bootstrapped = True
        else:
            merged_reply_lag_mean = fresh.reply_lag_running_mean
            merged_reply_lag_n = fresh.reply_lag_n
            merged_bootstrapped = fresh.bootstrapped

        merged = presence_state.PresenceState(
            last_seen_ts=merged_last_seen,
            daily_computed_at=now.isoformat(),
            hour_counts=merged_hour_counts,
            active_threshold=merged_active_threshold,
            reply_lag_running_mean=merged_reply_lag_mean,
            reply_lag_n=merged_reply_lag_n,
            bootstrapped=merged_bootstrapped,
            version=fresh.version + 1,
        )
        presence_state.save_presence_state(persona_dir, merged)


def compute_user_presence(persona_dir: Path, *, _now: datetime | None = None) -> UserPresence:
    """Compute current UserPresence from persisted presence state + audit log.

    All four signals default to permissive values on failure — uncertainty
    never tightens gates. The heavy full-history scans (silence-days
    last-seen discovery, the likely-active histogram, the one-time
    reply-lag bootstrap) run at most once/24h via the daily recompute
    below; the normal per-call path is a cheap small-file read plus
    arithmetic. ignore_streak is the one exception: it always uses its own
    bounded (never full-file) tail read (`_compute_ignore_streak_bounded`),
    independent of the daily cadence and of PresenceState entirely.
    """
    now = _now or datetime.now(UTC)

    try:
        cadence = load_cadence(persona_dir, _PRESENCE_DAILY_CADENCE_FILENAME)
        if is_due(cadence, now=now):
            _run_daily_presence_recompute(persona_dir, _now=now)
            save_cadence(
                persona_dir,
                _PRESENCE_DAILY_CADENCE_FILENAME,
                advance(now=now, interval_s=_PRESENCE_DAILY_INTERVAL_S),
            )
    except Exception:
        log.debug("user_pattern: daily presence recompute failed", exc_info=True)

    try:
        state = presence_state.load_presence_state(persona_dir)
    except Exception:
        log.debug("user_pattern: load_presence_state failed", exc_info=True)
        state = None

    try:
        if state is None or state.last_seen_ts is None:
            silence_days = 0.0
        else:
            last_seen = datetime.fromisoformat(state.last_seen_ts)
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=UTC)
            silence_days = max(0.0, (now - last_seen).total_seconds() / 86400.0)
    except Exception:
        log.debug("user_pattern: silence_days derivation failed", exc_info=True)
        silence_days = 0.0

    try:
        ignore_streak = _compute_ignore_streak_bounded(persona_dir)
    except Exception:
        log.debug("user_pattern: _compute_ignore_streak_bounded failed", exc_info=True)
        ignore_streak = 0

    try:
        if state is None or state.hour_counts is None or state.active_threshold is None:
            likely_active = True
        else:
            likely_active = state.hour_counts[now.astimezone().hour] >= state.active_threshold
    except Exception:
        log.debug("user_pattern: likely_active derivation failed", exc_info=True)
        likely_active = True

    try:
        if state is None or state.reply_lag_n < _COLD_START_LAG_MIN:
            response_lag_p50 = None
        else:
            response_lag_p50 = state.reply_lag_running_mean
    except Exception:
        log.debug("user_pattern: response_lag_p50 derivation failed", exc_info=True)
        response_lag_p50 = None

    return UserPresence(
        silence_days=silence_days,
        ignore_streak=ignore_streak,
        likely_active=likely_active,
        response_lag_p50=response_lag_p50,
    )
