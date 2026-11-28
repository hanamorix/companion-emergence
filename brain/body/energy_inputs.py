"""Gathers the inputs of one body energy step (#650, spec 2.4): the feelings, the interval's messages, the kindled's start.

Reads only; nothing is written and nothing is saved between heartbeats. The arithmetic is `brain.body.energy`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from brain import dev_constants
from brain.body.energy import EnergyInputs, EnergyMessage, length_factor
from brain.emotion import engine as emotion_engine
from brain.emotion import engine_files
from brain.health.json_io import iter_jsonl
from brain.utils.time import lenient_utc

logger = logging.getLogger(__name__)

# The speakers whose rows are messages: hers and the user's (image and summary rows are not).
_SPEAKERS = ("assistant", "user")


@dataclass(frozen=True)
class _Row:
    """A user or assistant conversation row reduced to what the step needs."""

    ts: datetime
    speaker: str
    words: int
    identity: tuple


def _row_of(raw: dict) -> _Row | None:
    """The row as a message, or None for an image or summary row, a row with no usable text or a time that does not parse."""
    from brain.ingest.buffer import turn_identity

    speaker = raw.get("speaker")
    text = raw.get("text")
    ts = lenient_utc(raw.get("ts"))
    if speaker not in _SPEAKERS or not isinstance(text, str) or ts is None:
        return None
    return _Row(ts=ts, speaker=speaker, words=len(text.split()), identity=turn_identity(raw))


def _speaker_weights() -> dict[str, float]:
    return {
        "assistant": dev_constants.BODY_ENERGY_HER_MESSAGE_WEIGHT,
        "user": dev_constants.BODY_ENERGY_USER_MESSAGE_WEIGHT,
    }


def _active_session_ids(persona_dir: Path) -> list[str]:
    from brain.ingest.buffer import list_active_sessions

    try:
        return sorted(list_active_sessions(persona_dir))
    except OSError:
        logger.warning("body energy: the active conversations folder cannot be listed; no messages read", exc_info=True)
        return []


def _interval_rows(persona_dir: Path, interval_start: datetime, now: datetime) -> list[_Row]:
    """The user and assistant rows after `interval_start` up to `now`, read as `count_chat_turns_since` reads them.

    A buffer that cannot be read costs only its own rows for this heartbeat (logged); felt time's tick still runs.

    Accepted edge (spec builder note 8, P4): this reads the active buffers only. A message that compaction archived before
    the heartbeat that would have read it is missed. That needs the row to be past the protected 40-row tail and 24 hours
    old (`brain/chat/compaction.py`), so it is rare; the 30-day length read in `_window_rows` does see archived rows.
    """
    from brain.ingest.buffer import read_session_after

    out: list[_Row] = []
    for session_id in _active_session_ids(persona_dir):
        try:
            raws = read_session_after(persona_dir, session_id, interval_start.isoformat())
        except Exception:
            logger.warning("body energy: session %s cannot be read; its messages are skipped", session_id, exc_info=True)
            continue
        for raw in raws:
            row = _row_of(raw)
            if row is not None and interval_start < row.ts <= now:
                out.append(row)
    return out


def _window_rows(persona_dir: Path, cutoff: datetime) -> dict[tuple, _Row]:
    """Every user and assistant row at or after `cutoff` in the active buffers and the archive, each turn once.

    Active rows come first, then archive files in name order; `turn_identity` drops a turn that is in more than one place.
    A file that cannot be read costs only its own rows (logged).
    """
    from brain.ingest.buffer import archived_files_written_since, read_session

    pool: dict[tuple, _Row] = {}

    def add(raw: dict) -> None:
        row = _row_of(raw)
        if row is not None and row.ts >= cutoff:
            pool.setdefault(row.identity, row)

    for session_id in _active_session_ids(persona_dir):
        try:
            raws = read_session(persona_dir, session_id)
        except Exception:
            logger.warning("body energy: session %s cannot be read; its messages are skipped", session_id, exc_info=True)
            continue
        for raw in raws:
            add(raw)
    for path in archived_files_written_since(persona_dir, cutoff):
        try:
            for raw in iter_jsonl(path):
                add(raw)
        except Exception:
            logger.warning("body energy: archive file %s cannot be read; its messages are skipped", path.name, exc_info=True)
    return pool


def _interval_messages(persona_dir: Path, interval_start: datetime, now: datetime) -> tuple[EnergyMessage, ...]:
    """The interval's messages with their weights and length factors (spec 2.3).

    The 30-day read for each speaker's average length runs only when the interval has a message. A message's average is
    over that speaker's earlier messages in the last 30 days (this heartbeat's earlier ones included): a first message, or
    an average of 0, counts 1. Words are `len(text.split())`.
    """
    interval = _interval_rows(persona_dir, interval_start, now)
    if not interval:
        return ()
    cutoff = now - timedelta(days=dev_constants.BODY_ENERGY_LENGTH_WINDOW_DAYS)
    pool = _window_rows(persona_dir, cutoff)
    for row in interval:
        pool.setdefault(row.identity, row)  # a message archived since the interval read is still counted
    interval_ids = {row.identity for row in interval}
    weights = _speaker_weights()
    totals: dict[str, int] = {}
    counts: dict[str, int] = {}
    messages: list[EnergyMessage] = []
    for row in sorted(pool.values(), key=lambda r: r.ts):
        if row.identity in interval_ids:
            count = counts.get(row.speaker, 0)
            average = totals.get(row.speaker, 0) / count if count else 0.0
            messages.append(EnergyMessage(weight=weights[row.speaker], length_factor=length_factor(row.words, average)))
        totals[row.speaker] = totals.get(row.speaker, 0) + row.words
        counts[row.speaker] = counts.get(row.speaker, 0) + 1
    return tuple(messages)


def gather_energy_inputs(
    persona_dir: Path, *, interval_start: datetime, now: datetime, heartbeat_s: float
) -> EnergyInputs:
    """Everything the energy step of the tick at `now` reads, for the interval that began at `interval_start`.

    `heartbeat_s` is the supervisor's own heartbeat interval, handed through unchanged for the step's catch-up count.

    One `read()` of the emotion engine for the short and long averages, `feels_high` per lift and drain name (it reads the
    engine's cached state itself), the kindled's start from the emotion state file (lock-free; None when there is no state,
    which closes the 30-day gate) and the interval's messages.
    """
    live = emotion_engine.engine_for(persona_dir)
    readings = live.read()
    names = dev_constants.BODY_ENERGY_LIFT_FEELINGS + dev_constants.BODY_ENERGY_DRAIN_FEELINGS
    high = frozenset(name for name in names if readings and live.feels_high(name))
    state = engine_files.read_state(persona_dir).state
    return EnergyInputs(
        interval_start=interval_start,
        heartbeat_s=heartbeat_s,
        short=emotion_engine.short_values(readings),
        long={name: reading.long for name, reading in readings.items()},
        high=high,
        messages=_interval_messages(persona_dir, interval_start, now),
        persona_start_at=state.persona_start_at if state is not None else None,
    )
