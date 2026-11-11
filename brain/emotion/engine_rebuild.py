"""The emotion engine's replay and catch-up (Engine 4, #579) and its rebuild (Engine 6, #584).

`replay` folds what happened in a window of time into a state: the rows of `memories.db` created or opened in it, the day each
row was written on, and the change-log lines. `catch_up` is the recovery-first step (design 2.8, "Recovery first", of `docs/source-spec/2026-10-28-emotion-rework-ThinkerOfThoughts-half-design.md`): a state that had to be loaded
from a backup copy is replayed forward from that copy's `saved_at` to now. `rebuild` builds a whole state from the last 30 days
of rows and the change log: at a persona's first start (the switch-over, which records `switch_over_at` in its one state write) and as the recovery
of a live persona whose state cannot be used. Both read the window through `_read_window`.

All three are called with the engine's lock already held by the caller (the drain, a direct apply or a batch, the first start, the
recovery job). None takes it, and the lock is not re-entrant. `catch_up` holds the caller's lock for as long as the replay takes
(accepted edge, design section 6 of `docs/source-spec/2026-10-28-emotion-rework-ThinkerOfThoughts-half-design.md`: the post-reply drain, or a writer that waits at the lock, may wait for that time); a rebuild holds it for its own length.
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
    counted_kind,
    day_of,
    fold_entry,
    fold_event,
    fold_guarded,
    resolve_emotions,
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


class _Window(NamedTuple):
    rows: list[Memory]
    stamps: list[datetime]
    oldest: datetime | None  # the creation time of the oldest row of any type and state, in the whole database


def _read_window(persona_dir: Path, since: datetime) -> _Window:
    """The rows, creation stamps and oldest creation time `catch_up` and `rebuild` start from, read in ONE open of `memories.db`.

    Opens `MemoryStore(db, integrity_check=False)` (the rebuild's reads skip the check, E7.14; the one-time x10 fix before it
    ran the check on a first start), reads `list_engine_window(since)`, `creation_stamps_since(since)` and `oldest_created_at()`,
    and closes. It does not stat the file: opening a store on a missing path would create an empty database, so the missing-file
    rule belongs to each caller.
    """
    store = MemoryStore(persona_dir / _MEMORIES_DB, integrity_check=False)
    try:
        return _Window(store.list_engine_window(since), store.creation_stamps_since(since), store.oldest_created_at())
    finally:
        store.close()


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
    window = _read_window(persona_dir, after)
    changes = engine_files.read_changes(persona_dir, after)
    stop = min((event.at for event in queued), default=None)
    folded = replay(
        state,
        rows=window.rows,
        stamps=window.stamps,
        changes=changes,
        now=now,
        after=after,
        stop=stop,
        emotions_of=emotions_of,
        modifier_of=modifier_of,
    )
    state.queue_line = 0
    return folded


class RebuildCheckError(Exception):
    """The rebuilt state failed its check or its verified write: nothing the engine reads was changed by it (a queue that was
    already emptied stays emptied, see `rebuild`)."""


class RebuildStateWriteError(Exception):
    """`rebuild`'s own `save_state` call raised an `OSError` (a directory or an unreadable file at the state path, a directory at
    a backup name the rotation unlinks, a full disk): the one named exception that tells that I/O fault apart from every other
    rebuild failure, so `EmotionEngine._locked_rebuild` can skip a first start for it. `path` is the state file, `cause` the
    `OSError`."""

    def __init__(self, path: Path, cause: OSError) -> None:
        super().__init__(f"{path}: {type(cause).__name__}: {cause}")
        self.path = path
        self.cause = cause


def _check_rebuilt(state: EngineState) -> None:
    """Raise `RebuildCheckError` unless the rebuilt state is one the reader would accept and every average is in 0-10.

    The state is turned into its on-disk form (`state_to_json`) and that form goes through the reader's own validator
    (`validate_state`: a non-finite number, a wrong shape or a bad date is refused there), so a state the reader would call
    unreadable is never written. Then every average and every snapshot average must be a number from 0 to the value cap.
    """
    try:
        engine_files.validate_state(engine_files.state_to_json(state))
    except ValueError as exc:
        raise RebuildCheckError(f"the rebuilt state is not one the reader accepts ({exc})") from exc
    cap = dev_constants.EMOTION_VALUE_CAP
    for name, emotion in state.emotions.items():
        averages = [emotion.short.avg, emotion.long.avg]
        for day in emotion.snapshot:
            averages += [day.short, day.long]
        for value in averages:
            if not 0.0 <= value <= cap:
                raise RebuildCheckError(f"an average of {name!r} is {value!r}, outside 0 to {cap}")


def rebuild(
    persona_dir: Path,
    mode: str,
    now: datetime,
    *,
    faded_emotions_of: Callable[[Memory], Mapping[str, float]],
    modifier_of: Callable[[Memory], float],
    switch_over_at: datetime | None = None,
) -> None:
    """Build the engine's state from the last 30 days of `memories.db` and the change log, and write it (Engine 6, #584).

    Called with the engine lock held by the caller; it never takes the lock (it is not re-entrant). `mode` is "first" (a
    persona's switch-over: the stored values of its counted rows are folded as they are; a missing `memories.db` is an empty
    window and creates nothing) or "recovery" (a live persona whose state cannot be used: the faded values `faded_emotions_of`
    gives, faded to this rebuild's clock as `catch_up` does; a missing `memories.db` raises `CatchUpError`, because an empty
    rebuild would rotate a readable backup out). `modifier_of` is the trust-aware weight, as for `catch_up`.

    The switch-over time. First mode records it as the rebuild's own `now`, in the one verified state write (there is no
    switch file). A recovery rebuild copies it from the old state it loads (main, or a backup copy: the walk accepts either),
    so a recovered persona stays switched over; when no readable copy holds one it uses `switch_over_at`, the value the
    caller's engine already saw, and when it has neither it refuses (`RebuildCheckError`, nothing written): a recovered
    state without the field would read as not switched. Accepted edge (Planning B25 SW2): a backup copy older than the first
    rebuild holds no `switch_over_at` (see `engine_files.switch_over_state`, edge (b)); a recovery that loaded only such a copy
    takes `switch_over_at` from the caller.

    The window is `now - EMOTION_LONG_HALF_LIFE_D days` up to now; a row dated after `now` is folded at `now` (the window has
    no creation upper bound). The state is built from empty by `replay` (no stop) and the days close at `now`. Past days are
    the creation stamps of every row of any type, one extra stamp at the access time of each counted row that is opened in the
    window and has no feelings to fold (`replay` folds a row with feelings at its access time; an emotion-less counted row would
    otherwise leave its access day unmarked), and the change-log lines. `persona_start_at` is the earliest of the oldest row of
    any type, the replay's own and (recovery) the old state's.

    The order is fixed. (1) The check (`_check_rebuilt`). (2) The queue step: `empty_queue`, then the new state's `queue_line` is 0;
    if the emptying fails, `queue_line` is the queue's end line, so no line is folded on top of the rebuild. (3) ONE
    `save_state(..., verify=True)` with `saved_at` and `reset_at` both `now` and `switch_over_at` set as above; a False result
    is a failed check. That write is the last step of the switch-over, and it replaces a state that cannot be read like any
    other (its normal backup rotation; nothing is moved aside or deleted). A failed check raises `RebuildCheckError`; an
    `OSError` of the write itself raises `RebuildStateWriteError`; either way the old state stays. Nothing is written before
    step 2, so a failed check leaves the queue intact.
    """
    if mode not in ("first", "recovery"):
        raise ValueError(f"mode must be 'first' or 'recovery', found {mode!r}")
    when = lenient_utc(now)
    if when is None:
        raise ValueError(f"now is not a usable time: {now!r}")
    now = when
    after = now - timedelta(days=dev_constants.EMOTION_LONG_HALF_LIFE_D)
    database = persona_dir / _MEMORIES_DB
    if file_ops.stat_path(database) is not None:
        window = _read_window(persona_dir, after)
    elif mode == "first":
        window = _Window([], [], None)  # a fresh persona: an empty state, and nothing is created
    else:
        raise CatchUpError(f"{database} does not exist")

    # E7.10 (accepted): a missing or corrupt part of the change log loses those changes at a rebuild; the log is a
    # crash-recovery record. E6.19 and E1.15 (accepted): the first rebuild has no chat-change history at all, because the
    # change log is only written once the persona is live.
    changes = engine_files.read_changes(persona_dir, after)

    previous_start: datetime | None = None
    carried_switch: datetime | None = None
    if mode == "recovery":
        # E1.4a (accepted): with no readable old state (the state and all three backups unreadable), `persona_start_at` has only
        # the oldest row to go on, and the write-back moves a row's creation time forward, so the gate can only stay shut
        # longer, by at most 30 days. No source exists that the write-back cannot move; there is no later fix.
        previous = engine_files.read_state(persona_dir).state
        previous_start = None if previous is None else previous.persona_start_at
        carried_switch = previous.switch_over_at if previous is not None and previous.switch_over_at is not None else switch_over_at
        if carried_switch is None:
            raise RebuildCheckError(
                "no readable copy of the state holds the switch-over time and the caller has none to carry: "
                "a recovered state without it would read as not switched"
            )

    if mode == "first":
        emotions_of = _stored_emotions
    else:
        emotions_of = faded_emotions_of
    # E3.5 (accepted): a day whose only rows were refreshed by the temp-gate write-back (which moves `created_at` forward)
    # counts as an off day in a later rebuild; the scan below sees the new date, not the old one.
    stamps = list(window.stamps)
    for memory in window.rows:
        accessed = _usable_time(memory.last_accessed_at) if memory.last_accessed_at is not None else None
        if accessed is None or accessed < after or counted_kind(memory) is None:
            continue
        try:
            has_feelings = bool(emotions_of(memory))
        except Exception:  # noqa: BLE001 - `replay` logs and skips this same row; here it only decides whether to add a stamp
            has_feelings = False
        if not has_feelings:
            stamps.append(accessed)

    state = EngineState()
    # E5.27 (accepted): a rebuilt persona's snapshot is a fresh 30-day history, so the flags after a rebuild or a catch-up can
    # differ from those before it. E6.16 (accepted): a showing beyond the first within 24 hours is not restored by a rebuild; the
    # row's access time folds once. S42 class (accepted), as for E6.14 case 1 in `EmotionEngine._apply_direct`: a writer that
    # committed its memory before the window read above and applies its event after this lock hold counts that memory twice
    # (one extra entry in the short and long averages); this is true of the first rebuild as well as of a recovery.
    #
    # Accepted edge (owner, 2026-10-27): a row's last_accessed_at can have been stamped by a read that did not
    # show her the memory. Before the access fixes the recall-resonance check did this for up to 50 memories
    # every 15 minutes, and some background reads may still do it. An install whose first rebuild reads such
    # rows counts them as accessed in this window, once: a small nudge to the starting mood, mostly in the
    # long average, which works it out over about a month. Nothing waits on a fix, because installs
    # update at different times, so this is not a deferred fix.
    replay(
        state,
        rows=window.rows,
        stamps=stamps,
        changes=changes,
        now=now,
        after=after,
        stop=None,
        emotions_of=emotions_of,
        modifier_of=modifier_of,
    )
    close_days(state, now)
    starts = [t for t in (window.oldest, state.persona_start_at, previous_start) if t is not None]
    state.persona_start_at = min(starts, default=None)
    state.saved_at = state.reset_at = now
    state.switch_over_at = now if mode == "first" else carried_switch
    _check_rebuilt(state)

    # E7.18 (accepted): a change still queued at a rebuild with no change-log line (a failed append, or phase B, which writes no
    # log) is lost, and so is a queued line when the state write below fails after the emptying. A queued showing is already in
    # the rebuild through its row's `last_accessed_at` (one fold), and a queued change through its change-log line.
    if engine_files.empty_queue(persona_dir):
        state.queue_line = 0
    else:
        state.queue_line = engine_files.read_queue(persona_dir, from_line=0).end_line
    # A state that exists but cannot be read (owner S88, Planning B36 RD2: "in that case the state file should just get replaced
    # with a new one with anything needed reconstructed") is replaced by this write like any other state. `save_state`'s normal
    # rotation moves the unreadable main file into `.bak1`, and the old copies age out over the next saves; nothing here moves,
    # keeps or deletes them. Accepted edges (P4), both for the I/O fault skip in `EmotionEngine._locked_rebuild`: the queue was
    # emptied just above, so when this write fails and that start is skipped the queued changes of that start are lost (the E7.18
    # class), each skipped start losing its own; and a recovery rebuild is not skipped, it raises and the recovery job retries.
    # Accepted edge (F3-rebuild, owner's review list): a fresh process rebuilding a state that was lost folds each memory's STORED
    # values unfaded and sets `switch_over_at` to `now`, which also restarts the fade clock of every older memory, so the
    # rebuilt mood reads stronger than the persona's real state until the averages work it off. No data is lost or corrupted.
    try:
        verified = engine_files.save_state(persona_dir, state, verify=True)
    except OSError as exc:
        raise RebuildStateWriteError(engine_files.state_path(persona_dir), exc) from exc
    if not verified:
        raise RebuildCheckError("the rebuilt state did not verify after it was written (the previous copy is put back when there is one)")


def _stored_emotions(memory: Memory) -> Mapping[str, float]:
    """What a FIRST rebuild folds for a row: its stored feelings (names canonical, each held to 0-10) when it counts, else nothing."""
    return resolve_emotions(memory._stored_emotions) if counted_kind(memory) is not None else {}
