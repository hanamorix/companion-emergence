"""The emotion engine's replay and catch-up (Engine 4, #579). The rest of this file (the rebuild, the recovery job) is Engine 6's.

`replay` folds what happened in a window of time into a state: the rows of `memories.db` created or opened in it, the day each
row was written on, and the change-log lines. `catch_up` is the recovery-first step (design 2.8, "Recovery first", of `docs/source-spec/2026-10-28-emotion-rework-ThinkerOfThoughts-half-design.md`): a state that had to be loaded
from a backup copy is replayed forward from that copy's `saved_at` to now.

Both are called with the engine's lock already held by the caller (the drain, a direct apply or a batch). Neither takes it,
and the lock is not re-entrant. `catch_up` holds the caller's lock for as long as the replay takes (accepted edge, design section 6
of `docs/source-spec/2026-10-28-emotion-rework-ThinkerOfThoughts-half-design.md`: the post-reply drain, or a writer that waits at the lock, may wait for that time).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

from brain import dev_constants
from brain.emotion import engine_files
from brain.emotion.engine_utils import (
    EngineState,
    Event,
    RepeatEntry,
    close_days,
    day_of,
    fold_entry,
    fold_event,
    fold_guarded,
)
from brain.memory.store import Memory, MemoryStore
from brain.utils import file_ops
from brain.utils.time import lenient_utc

logger = logging.getLogger(__name__)

_MEMORIES_DB = "memories.db"

# The order of points that fall on the same instant: a row's creation stamp first, then the row's own entry, then a change.
_STAMP, _ROW, _CHANGE = 0, 1, 2


class CatchUpError(Exception):
    """The catch-up cannot be done: the state it would start from is too old to replay, or the memories are not there."""


class _Point(NamedTuple):
    at: datetime
    kind: int
    order: int
    payload: object  # an `Event` for a stamp or a change; a `_RowPoint` for a row


class _RowPoint(NamedTuple):
    memory: Memory
    created: datetime
    emotions: Mapping[str, float]
    weight: float


def _usable_time(value: object) -> datetime | None:
    """`value` as an aware UTC time that `day_of` can turn into a local day, else None (nothing is folded at such a time)."""
    when = lenient_utc(value)
    if when is None:
        return None
    try:
        day_of(when)
    except (OverflowError, OSError, ValueError):
        return None
    return when


def replay(
    state: EngineState,
    *,
    rows: Sequence[Memory],
    stamps: Sequence[datetime],
    changes: Sequence[Event],
    now: datetime,
    after: datetime,
    stop: datetime | None,
    emotions_of: Callable[[Memory], Mapping[str, float]],
    modifier_of: Callable[[Memory], float],
) -> frozenset[str]:
    """Fold a window of history into `state` and return the ids of the rows it folded.

    Points: a creation stamp for each of `stamps` (it closes the day it falls on and keeps the persona's start date through a
    start-only event); one entry for each counted row of `rows` that has emotions (below); each of `changes`. Every point is
    folded in time order, `close_days` first, at `min(its time, now)`: a row dated ahead of `now` is folded at `now`. Ties
    fold in the order the points are read, stamps before rows before changes: a catch-up and the live fold can differ for two
    things at one instant (accepted edge, design section 6, same-time order).

    A row folds ONCE, with the emotions it has now (`emotions_of`) at the weight `modifier_of` gives: at the later of its
    creation and its last access when that access is in `[after, stop)`; else at its creation when it was created at or after
    `after` (its access, at or after `stop` or none, is left to the showing that is still queued); else not at all (created
    before `after` and opened at or after `stop`: its queued showing folds it). The lower bound `after` is INCLUSIVE for every
    point (rows, stamps, change lines), matching `MemoryStore.list_engine_window` ("on or after"); the upper bound `stop` is
    exclusive. `stop=None` means no bound: accesses and changes replay up to now. A row created in the last
    `EMOTION_REPEAT_WINDOW_H` hours gets a repeat entry (its first showing, count 0, dated like the fold, never after now), so the
    next showing is halved. A climax
    reset is not replayed. A change dated exactly at `after` can fold twice (no skip id exists for a change): the S42-class
    overlap, accepted.

    Each point folds through `fold_guarded`: one that raises (a time `day_of` cannot convert, a bad value) is put back as it was,
    logged and skipped, and the others still fold. A skipped row is not in the returned ids.
    """
    points: list[_Point] = []
    order = 0
    for stamp in stamps:
        at = _usable_time(stamp)
        if at is None or at < after:
            continue
        points.append(_Point(at, _STAMP, order, Event(kind="start", at=at)))
        order += 1
    for row in rows:
        created = _usable_time(row.created_at)
        accessed = _usable_time(row.last_accessed_at) if row.last_accessed_at is not None else None
        if created is None:
            logger.warning("emotion catch-up: memory %s skipped, its creation time cannot be used", row.id)
            continue
        if accessed is not None and accessed >= after and (stop is None or accessed < stop):
            at = max(created, accessed)
        elif created >= after:
            at = created
        else:
            continue
        try:
            emotions = dict(emotions_of(row))
            weight = modifier_of(row) if emotions else 1.0
        except Exception as exc:  # noqa: BLE001 - one memory's bad emotions or trust must not stop the replay
            logger.warning("emotion catch-up: memory %s skipped (%s: %s)", row.id, type(exc).__name__, exc)
            continue
        if not emotions:
            continue
        points.append(_Point(at, _ROW, order, _RowPoint(row, created, emotions, weight)))
        order += 1
    for change in changes:
        if change.at < after or (stop is not None and change.at >= stop):
            continue
        points.append(_Point(change.at, _CHANGE, order, change))
        order += 1

    repeat_window = timedelta(hours=dev_constants.EMOTION_REPEAT_WINDOW_H)
    folded: set[str] = set()
    for point in sorted(points, key=lambda p: (p.at, p.kind, p.order)):
        held = min(point.at, now)

        def step(point: _Point = point, held: datetime = held) -> None:
            close_days(state, held)
            payload = point.payload
            if isinstance(payload, _RowPoint):
                for name, value in payload.emotions.items():
                    fold_entry(state, name, value, payload.weight, held)
                if payload.created >= after and now - payload.created < repeat_window:
                    state.repeats[payload.memory.id] = RepeatEntry(first_at=min(payload.created, now), count=0)
            else:
                fold_event(state, payload, now)  # type: ignore[arg-type]

        if isinstance(point.payload, _RowPoint):
            label = f"memory {point.payload.memory.id}"
        else:
            label = f"{point.payload.kind} event at {point.at}"  # type: ignore[union-attr]
        if fold_guarded(state, step, f"catch-up {label}", key="engine_rebuild:replay") and isinstance(point.payload, _RowPoint):
            folded.add(point.payload.memory.id)
    return frozenset(folded)


def catch_up(
    persona_dir: Path,
    state: EngineState,
    now: datetime,
    *,
    queued: Sequence[Event],
    emotions_of: Callable[[Memory], Mapping[str, float]],
    modifier_of: Callable[[Memory], float],
) -> frozenset[str]:
    """Bring a state loaded from a backup copy forward from its `saved_at` to now; return the ids of the rows it folded.

    Reads the rows created or opened since `saved_at` (`MemoryStore.list_engine_window`, opened and closed here), their creation
    stamps, and the change-log lines (`engine_files.read_changes`, which skips a line it cannot load), and replays them
    (`replay`). Directly applied kinds (a memory's own commit) go up to now; queued kinds (showings, chat changes) go up to the
    OLDEST event among `queued`, all the lines of the queue file, which the drain folds after this once each. The queue
    position is set to 0, so that drain folds every line in the file (a backup's own position can name a line of an older
    queue). Writes nothing itself: the caller stamps `saved_at` and `reset_at` and saves. The returned ids are what the caller
    tests its commit and assigned events against: those memories are already counted.

    Raises `CatchUpError` when `saved_at` is missing, more than `EMOTION_LONG_HALF_LIFE_D` days before now (a replay cannot
    stand in for a backup that old), or `memories.db` is missing (opening a store on a missing path would create an empty
    one). Any other failure (a database that cannot be opened or read) propagates; the caller treats every exception as a
    failed catch-up.

    Accepted edges (design section 6): a queued event dated before its time (a late-dated one) narrows the window, and an append
    that straddles the backup's save is missed. A change dated exactly at `saved_at` can fold twice (no skip id exists for
    it). After a clock regression together with a corrupt main state, `saved_at` can be ahead of now and is held to now, so the
    rows between the backup's true time and now are not replayed (a clock set back does not drag `saved_at` backwards, by design).
    """
    saved = lenient_utc(state.saved_at)
    if saved is None:
        raise CatchUpError("the backup state has no saved_at")
    if now - saved > timedelta(days=dev_constants.EMOTION_LONG_HALF_LIFE_D):
        raise CatchUpError(f"the backup state was saved {now - saved} ago, longer than the replay can stand in for")
    after = min(saved, now)
    database = persona_dir / _MEMORIES_DB
    if file_ops.stat_path(database) is None:
        raise CatchUpError(f"{database} does not exist")
    store = MemoryStore(database, integrity_check=False)
    try:
        rows = store.list_engine_window(after)
        stamps = store.creation_stamps_since(after)
    finally:
        store.close()
    changes = engine_files.read_changes(persona_dir, after)
    stop = min((event.at for event in queued), default=None)
    folded = replay(
        state,
        rows=rows,
        stamps=stamps,
        changes=changes,
        now=now,
        after=after,
        stop=stop,
        emotions_of=emotions_of,
        modifier_of=modifier_of,
    )
    state.queue_line = 0
    return folded
