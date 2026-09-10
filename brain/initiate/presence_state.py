"""presence_state.py — persisted sidecar backing the compute_user_presence
incremental/event-driven redesign (#225).

Holds a small ``<persona_dir>/presence_state.json`` used to avoid a full
history re-scan on every call: the silence-days last-seen timestamp, the
likely-active-at-hour histogram + threshold, and the reply-lag running
mean/count. ignore_streak has NO state here at all — round 6 of #225's
red-team abandoned every attempt to translate its full backward scan into
persisted running state (four distinct defects across rounds 2-5) and
reverted it to a direct, bounded, stateless re-scan every call (see
``user_pattern.py``'s ``_compute_ignore_streak_bounded``).

Every write goes through a per-persona-keyed lock (``_presence_lock``) held
ONLY around the small final read-merge-write step, never across a source
file scan — the scans that feed this state (active_conversations/*.jsonl,
initiate_audit.jsonl) are read unlocked, exactly like today's un-cached
reads. See ``changes/user-presence-incremental-225/2-plan.md``'s "Design
overview" for the full concurrency rationale.

Persistence follows the house convention (persisted_cadence.py,
notes/state.py): atomic temp-file + ``Path.replace``, fail-open to a cold
default state on missing or corrupt JSON. Fail-open is the #1 invariant
this module exists to preserve — a missing/corrupt sidecar must never
tighten a downstream gate.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from brain.paths import presence_state_path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PresenceState:
    """Persisted presence signal state for one persona.

    Cold/missing/corrupt file loads as the all-defaults sentinel
    (``_SENTINEL`` below) — every derived UserPresence signal's permissive
    default is computed FROM this sentinel, not hard-coded a second time.
    """

    last_seen_ts: str | None  # ISO ts of the most recent inbound (user) turn
    daily_computed_at: str | None  # when hour_counts/active_threshold were last rebuilt
    hour_counts: tuple[int, ...] | None  # 24 buckets, or None = never computed
    active_threshold: int | None  # None = insufficient data seen (permissive)
    reply_lag_running_mean: float | None
    reply_lag_n: int  # count of lags folded so far
    bootstrapped: bool  # True once the one-time historical-audit reply-lag
    # bootstrap scan has run, EVEN IF it found zero rows — distinguishes
    # "never bootstrapped" from "bootstrapped, genuinely empty," so a
    # cold/low-traffic persona's daily recompute doesn't re-scan forever.
    version: int  # incremented on every write, by every mutator in this
    # module. An optimistic-concurrency check so the daily-recompute's
    # (deliberately unlocked) scan can detect whether a live event wrote to
    # this persona's state while the scan was in flight, without ever
    # holding the lock across it.


_SENTINEL = PresenceState(
    last_seen_ts=None,
    daily_computed_at=None,
    hour_counts=None,
    active_threshold=None,
    reply_lag_running_mean=None,
    reply_lag_n=0,
    bootstrapped=False,
    version=0,
)


def _state_path(persona_dir: Path) -> Path:
    return presence_state_path(persona_dir)


def load_presence_state(persona_dir: Path) -> PresenceState:
    """Load the persisted presence state; fail-open to the cold sentinel on
    any missing/corrupt/malformed file."""
    path = _state_path(persona_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _SENTINEL
    if not isinstance(raw, dict):
        return _SENTINEL
    try:
        hour_counts_raw = raw.get("hour_counts")
        return PresenceState(
            last_seen_ts=raw.get("last_seen_ts"),
            daily_computed_at=raw.get("daily_computed_at"),
            hour_counts=tuple(hour_counts_raw) if hour_counts_raw is not None else None,
            active_threshold=raw.get("active_threshold"),
            reply_lag_running_mean=raw.get("reply_lag_running_mean"),
            reply_lag_n=int(raw.get("reply_lag_n", 0)),
            bootstrapped=bool(raw.get("bootstrapped", False)),
            version=int(raw.get("version", 0)),
        )
    except (TypeError, ValueError):
        return _SENTINEL


def save_presence_state(persona_dir: Path, state: PresenceState) -> None:
    """Atomically persist presence state (temp file + rename), best-effort.

    Mirrors persisted_cadence.save_cadence's fail-soft posture: a raised
    OSError here must never propagate and break a caller (an event hook, a
    supervisor tick) — swallowing it just means the state re-derives sooner
    from source data on the next call.
    """
    path = _state_path(persona_dir)
    payload = {
        "last_seen_ts": state.last_seen_ts,
        "daily_computed_at": state.daily_computed_at,
        "hour_counts": list(state.hour_counts) if state.hour_counts is not None else None,
        "active_threshold": state.active_threshold,
        "reply_lag_running_mean": state.reply_lag_running_mean,
        "reply_lag_n": state.reply_lag_n,
        "bootstrapped": state.bootstrapped,
        "version": state.version,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        log.warning("save_presence_state: could not persist %s (best-effort)", path, exc_info=True)
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink()


# ---------------------------------------------------------------------------
# Per-persona-keyed lock
# ---------------------------------------------------------------------------
#
# Keyed by str(persona_dir) even though one bridge process serves exactly one
# persona today (BridgeAppState holds a single persona_dir) — cheap hardening
# against any future cross-persona coupling (#225 stage-3 finding 8), same
# idiom as server.py's per-session in_flight_locks.setdefault(...).

_locks: dict[str, threading.Lock] = {}
_locks_meta_lock = threading.Lock()


def _presence_lock(persona_dir: Path) -> threading.Lock:
    """Return the (get-or-create) lock guarding `persona_dir`'s presence state.

    Held ONLY around each accessor's own small final read-merge-write step —
    never across a source-file scan. See module docstring / 2-plan.md.
    """
    key = str(persona_dir)
    with _locks_meta_lock:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


def record_inbound_turn(persona_dir: Path, ts: str) -> None:
    """Event hook (silence-days): update last_seen_ts to `ts` if newer.

    ISO-8601 strings sort correctly lexicographically, so a cheap string
    compare suffices — no parse needed. O(1): one small JSON read+write
    under the per-persona presence lock. A no-op (no write at all) when
    `ts` is not newer than what's already cached.
    """
    with _presence_lock(persona_dir):
        state = load_presence_state(persona_dir)
        if state.last_seen_ts is not None and ts <= state.last_seen_ts:
            return
        new_state = dataclasses.replace(
            state, last_seen_ts=ts, version=state.version + 1
        )
        save_presence_state(persona_dir, new_state)


def fold_reply_lag(persona_dir: Path, lag_seconds: float) -> None:
    """Event hook (median reply-lag): fold one lag into the running mean.

    Standard incremental/streaming mean update: n = reply_lag_n + 1;
    running_mean = (running_mean or 0.0) + (lag - (running_mean or 0.0)) / n.
    O(1): one small JSON read+write under the per-persona presence lock.
    """
    with _presence_lock(persona_dir):
        state = load_presence_state(persona_dir)
        n = state.reply_lag_n + 1
        prev_mean = state.reply_lag_running_mean or 0.0
        new_mean = prev_mean + (lag_seconds - prev_mean) / n
        new_state = dataclasses.replace(
            state,
            reply_lag_running_mean=new_mean,
            reply_lag_n=n,
            version=state.version + 1,
        )
        save_presence_state(persona_dir, new_state)
