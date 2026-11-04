"""The emotion engine's interface and its drain (Engine 0 #575, hooks Engine 3 #578, drain and shadow engine Engine 4 #579).

Every public call exists with its final signature, so the code that calls the engine can be written and tested before the
engine behind every call exists.

Real now:
- `half_life`, the lookup of an emotion's half-life in days (the persona's vocabulary file, then
  `dev_constants.EMOTION_HALF_LIFE_DAYS`, then `EMOTION_DEFAULT_HALF_LIFE_DAYS`), and `is_live()`, which is True once
  `state/emotion_switch.json` exists, or cannot be inspected (nothing writes that file until the switch-over, so it answers
  False for every persona today unless the stat itself fails; only a switch file that was seen is remembered).
- The four hooks. A memory's own emotions (`on_memory_committed`, `on_emotions_assigned`, and the start-only date event
  that the commit hook makes for a row that counts for nothing) are APPLIED to the state under the engine's lock when the
  memory is written (`_apply_direct`, or once per `batch()` block). A showing (`on_memory_shown`) and a chat change
  (`fold_change`) are APPENDED to the queue (`state/emotion_pending.jsonl`), and `note_active` folds them later. A hook never
  raises into the write that called it. Everything an event needs is worked out before the lock is taken, and nothing inside
  the lock hold touches a model.
- `note_active(now)`, the drain and the day close, and the one apply step both it and the direct apply run (`_run_apply`):
  load the state, recovery first when it came from a backup copy (`engine_rebuild.catch_up`), fold the events in time order
  with the days closing, prune, save once. The drain consumes the queue by position (`EngineState.queue_line`), then empties
  the queue file; it never deletes it.
- `read()` and `status()`: the readings and whether the state is current. Reads never write and never take the lock.
- `batch()`: a bulk writer's block; the events of its hooks apply once, when the outermost block ends.

Shadow mode (the phase before the switch-over): the engine writes a shadow `state/`, nothing reads it, `faded_value` returns the
stored value, no change-log line is written and nothing rebuilds.

Stubbed now:
- `oldest_entry_at()` returns None.
- `faded_value(memory, emotion)` returns the stored value and raises `KeyError`, naming the
  memory id and the emotion, when the memory has no such emotion. It never returns 0.0:
  "absent" is for the caller to decide.
- `needs_recovery`, `recovery_rebuild` and `ensure_engine_ready` do nothing; `EngineStartupError` is defined and
  nothing raises it yet.

Callers use the receiver form `engine.engine_for(persona_dir).<call>` and import the module
(`from brain.emotion import engine`), not its names, so one patch reaches every call site.
"""

from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from brain import dev_constants
from brain.emotion import engine_files, engine_rebuild, persona_loader, vocabulary
from brain.emotion.engine_files import VocabularyHalfLives
from brain.emotion.engine_utils import (
    EngineState,
    Event,
    close_days,
    counted_kind,
    day_of,
    fold_event,
    fold_guarded,
    resolve_emotions,
    shift_for,
    source_modifier,
)
from brain.utils import file_ops
from brain.utils.file_lock import LockTimeoutError
from brain.utils.log_throttle import count_throttled, warn_throttled
from brain.utils.time import lenient_utc

if TYPE_CHECKING:
    from brain.memory.store import Memory

logger = logging.getLogger(__name__)

_UNSET = object()  # no clean drain yet: the early-out may not fire


class Reading(NamedTuple):
    """One feeling as `read()` reports it: the short and long averages, their difference, and the shift."""

    short: float
    long: float
    difference: float
    shift: int


class EngineStatus(NamedTuple):
    caught_up: bool
    reset_at: datetime | None


class EngineStartupError(Exception):
    """The engine could not be made ready. `what` names the step that failed, `cause` why."""

    def __init__(self, what: str, cause: object) -> None:
        super().__init__(what, cause)
        self.what = what
        self.cause = cause

    def __str__(self) -> str:
        return f"{self.what}: {self.cause}"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _never_raises(hook: Callable[..., None]) -> Callable[..., None]:
    """Make a hook unable to raise into the write that called it (E6.9): the one guard of all four hooks.

    Three outcomes. A `LockTimeoutError` is the engine lock not coming within its wait (a drain, a rebuild or a wedged
    holder has it): the event is not recorded or applied, one throttled line says so. Any other `OSError` is the engine's
    folder or the lock's file refusing us (a read-only or full disk, a share that went away). It stays true for every write
    until the cause is fixed, so it is logged once per interval, without a traceback (`count_throttled`). Anything else is a
    bug in the hook: one error line with the traceback, every time.
    """

    @functools.wraps(hook)
    def guarded(self: EmotionEngine, *args: object, **kwargs: object) -> None:
        try:
            hook(self, *args, **kwargs)
        except LockTimeoutError as exc:
            # Before `except OSError`: `LockTimeoutError` IS an `OSError` (a `TimeoutError`), and a lock wait that ran out is
            # not a disk fault. Its own throttle key, so neither line hides the other.
            count_throttled(
                logger,
                f"engine:lock_wait:{self.persona_dir}",
                logging.ERROR,
                "emotion engine: %s gave up on the engine lock (%s); its event was not recorded or applied",
                hook.__name__,
                exc,
            )
        except OSError as exc:
            count_throttled(
                logger,
                f"engine:hook:{self.persona_dir}",
                logging.ERROR,
                "emotion engine: %s could not record its event (%s: %s)",
                hook.__name__,
                type(exc).__name__,
                exc,
            )
        except Exception:
            logger.error("emotion engine: %s failed; its event is lost", hook.__name__, exc_info=True)

    return guarded


class EmotionEngine:
    """The engine for one persona directory. Get it through `engine_for(persona_dir)`."""

    def __init__(self, persona_dir: Path, *, now: Callable[[], datetime] = _utcnow) -> None:
        self.persona_dir = Path(persona_dir)
        self._now = now
        self._vocabulary = VocabularyHalfLives(self.persona_dir)
        # The parsed state, kept by the state file's stat key (RD6): the hooks read the stored `persona_start_at` through it
        # without a lock. Owned here; `read()` and `status()` and the drain's early-out read the state through this same cache.
        self._state_cache = engine_files.StateCache(self.persona_dir)
        self._live = False  # a True answer for a switch file that was SEEN is kept for the life of the instance
        # Per-instance state of the apply step, all of it in memory (a second process has its own):
        self._catch_up_failed = False  # set by a failed recovery or an unreadable live state; cleared by a state load or write
        self._failure_logged_day: str | None = None  # the local day the failure was last logged for (once per active day)
        self._last_queue_key: object = _UNSET  # the queue file's stat key as this instance's last clean drain left it
        self._warned_names: set[str] = set()  # unregistered emotion names `read()` already named, once per name
        self._refresh_logged: set[str] = set()  # distinct refresh failures `read()` already logged
        self._batch = threading.local()  # `events` is a list while this thread is inside `batch()`, else absent or None

    # --- hooks the memory store and the extractor call ------------------------------------------------
    #
    # A hook never raises (`_never_raises`). Its steps: the cheap `counted_kind` test first; everything the event needs is
    # worked out BEFORE the lock is taken (the row, the trust lookup, the half-lives, the faded values). A memory's own
    # emotions (commit, assigned, the start-only date event) are then applied directly (`_apply`); a showing and a change are
    # appended to the queue (`_queue`) under the lock. Nothing inside the lock hold except the catch-up of a recovery touches
    # `memories.db`, and nothing touches a model.

    @_never_raises
    def on_memory_committed(self, memory: Memory) -> None:
        kind = counted_kind(memory)
        if kind is None:
            self._apply_start_if_needed(memory.created_at)
            return
        emotions = self._resolved_faded(memory)
        if not emotions:  # nothing to fold, but the row's date still reaches the persona's start (B02.6)
            self._apply_start_if_needed(memory.created_at)
            return
        self._apply(
            Event(
                kind="commit",
                at=memory.created_at,
                memory_id=memory.id,
                emotions=emotions,
                modifier=self._modifier(memory, kind),
                climax=self._stored_climax(memory),
            )
        )

    @_never_raises
    def on_emotions_assigned(self, memory: Memory) -> None:
        kind = counted_kind(memory)
        if kind is None:
            return
        emotions = self._resolved_faded(memory)
        if not emotions:
            return
        self._apply(
            Event(
                kind="assigned",
                at=memory.created_at,
                memory_id=memory.id,
                emotions=emotions,
                modifier=self._modifier(memory, kind),
            )
        )

    @_never_raises
    def fold_change(self, emotions: Mapping[str, float], at: datetime | None, source: str) -> None:
        amounts = resolve_emotions(emotions, signed=True)
        if not amounts:  # an empty change folds nothing
            return
        self._queue(Event(kind="change", at=at if at is not None else self._now(), emotions=amounts, source=source))

    @_never_raises
    def on_memory_shown(self, memory: Memory, at: datetime | None) -> None:
        kind = counted_kind(memory)
        if kind is None:
            return
        emotions = self._resolved_faded(memory)
        if not emotions:
            return
        self._queue(
            Event(
                kind="showing",
                at=at if at is not None else self._now(),
                memory_id=memory.id,
                emotions=emotions,
                modifier=self._modifier(memory, kind),
            )
        )

    def _resolved_faded(self, memory: Memory) -> dict[str, float]:
        """The memory's emotions as the engine folds them: faded, names canonical, each held to 0-10, bad values dropped.

        A name whose faded value cannot be worked out is left out with a log line, so one bad name does not stop the others.
        The memory is read, never changed.
        """
        faded: dict = {}
        for name in memory.emotions:
            try:
                faded[name] = self.faded_value(memory, name)
            except Exception:
                logger.warning("emotion engine: no faded value for %r of memory %s; left out", name, memory.id, exc_info=True)
        return resolve_emotions(faded)

    @staticmethod
    def _stored_climax(memory: Memory) -> float:
        """The memory's STORED climax value (the climax test is on the stored value, the fold uses the faded one), 0.0 if none."""
        stored = {name: value for name, value in memory.emotions.items() if isinstance(name, str) and vocabulary.canonical_name(name) == "climax"}
        return resolve_emotions(stored).get("climax", 0.0)

    def _modifier(self, memory: Memory, kind: str) -> float:
        """The source modifier: trust is asked for a peer memory that names its peer, and is 0.0 otherwise."""
        trust = 0.0
        peer_id = memory.metadata.get("peer_id")
        if kind == "peer" and isinstance(peer_id, str) and peer_id:
            trust = engine_files.peer_trust(self.persona_dir, peer_id)
        return source_modifier(memory, trust)

    def _start_needed(self, created_at: datetime) -> bool:
        """True when the persona's stored start date is missing, unreadable (a state that cannot be read reads as none) or later than this row's date.

        Read through the cache with no lock. The check only says "not needed" from a stored start that is already at or
        before the row's date, and the fold keeps the earliest date it has seen (a min-fold, `fold_event`), so a state
        replaced meanwhile cannot make that verdict wrong.
        A backup copy of the state is older, so its start is never earlier than the main file's: it can only add a duplicate
        start event, which the drain's min-fold makes harmless.
        """
        created = lenient_utc(created_at)
        if created is None:
            return False
        state = self._state_cache.get().state
        return state is None or state.persona_start_at is None or created < state.persona_start_at

    def _apply_start_if_needed(self, created_at: datetime) -> None:
        """Apply the start-only event when the stored start date is missing or later than the row's.

        The start-only event is a memory's own date event: a direct apply, like a commit (spec 3.0). It carries no memory id,
        so recovery never skips it; its min-fold is idempotent.
        """
        if self._start_needed(created_at):
            self._apply(Event(kind="start", at=created_at))

    def _queue(self, event: Event) -> None:
        """Append `event` (a showing or a chat change) to the queue, holding the engine's lock for the append only.

        Accepted edge (E2, R12, owner 2026-10-21: "treat it as a fairly minor edge case, noted in code"): an entry can be lost
        (a failed append, a hook that raised, a lock wait that timed out, a crash between the commit and this append, a line
        damaged by an unclean stop). `state_lock` is the bounded acquisition: a hook's caller does not wait for ever, and the
        guard (`_never_raises`) handles its `LockTimeoutError`. It is not recovered. A lost entry is a minor edge while
        appends succeed; a fault that persists loses every entry for as long as it lasts, and the log line is then the only
        signal. `append_event` logs a failed append and returns False; nothing more is done with that here.
        """
        with engine_files.state_lock(self.persona_dir):
            engine_files.append_event(self.persona_dir, event)

    # --- the direct apply and the batch scope --------------------------------------------------------

    def _apply(self, event: Event) -> None:
        """Apply a memory's own event now, or, inside this thread's `batch()` block, collect it for the block's end."""
        pending = getattr(self._batch, "events", None)
        if pending is not None:
            pending.append(event)
            return
        self._apply_direct([event])

    def _apply_direct(self, events: list[Event]) -> None:
        """The direct apply of 3.0: under the engine lock, the drain's own apply step on `events` (no queue write).

        Never raises: the memory is already saved (R8) and a failure here only costs its emotions. The clock is read once,
        inside the lock.

        Accepted edges (spec section 6), the code comments the spec asks for here:
        - S42 (E6.14 case 1): a writer that committed its memory and then waited at the lock while a recovery ran, and that
          recovery counted that memory, applies it again: in that overlap one memory's emotions count twice (one extra entry in
          the short and long averages). No stored catch-up range is added.
        - F-DA: a failed direct apply while the state is still readable (a failed save, a crash between the memory's commit and
          this apply, a lock wait that timed out, which includes a wait behind a recovery, a batch lost in a crash) leaves the
          memory saved and nothing applies its event again: one memory's emotions are missing from the short and long averages
          until the next rebuild folds its row. Not recovered, not retried, no issue.
        - A commit event that the recovery's catch-up already counts gets no climax reset: a replay gives none (E4.2, E5.19),
          so the first write to meet a corrupted state, if its memory has stored climax 7 or more, is folded without one and
          its own apply is skipped (`_run_apply`, the skip of `skip_ids`).
        - A day can close while a queued event dated in it still waits (E5.13): the apply closes days up to now.
        """
        try:
            with engine_files.state_lock(self.persona_dir):
                self._run_apply(events, drain=False, now=self._now())
        except LockTimeoutError as exc:
            count_throttled(
                logger,
                f"engine:apply_lock:{self.persona_dir}",
                logging.ERROR,
                "emotion engine: a memory's emotions were not applied, the engine lock did not come (%s); the memory is saved",
                exc,
            )
        except OSError as exc:
            count_throttled(
                logger,
                f"engine:apply_os:{self.persona_dir}",
                logging.ERROR,
                "emotion engine: a memory's emotions were not applied (%s: %s); the memory is saved",
                type(exc).__name__,
                exc,
            )
        except Exception as exc:
            warn_throttled(
                logger,
                f"engine:apply:{self.persona_dir}:{type(exc).__name__}",
                "emotion engine: a memory's emotions were not applied, an unexpected %s; the memory is saved",
                type(exc).__name__,
            )

    @contextmanager
    def batch(self) -> Iterator[None]:
        """A bulk writer wraps its loop in this block: its hooks' events apply ONCE, when the outermost block ends.

        Inside the block, on this thread, the commit, assigned and start-only hooks add their event to a list and apply
        nothing; the outermost block's end (normal or by an exception, which goes on out unchanged) runs one direct apply on
        the whole list: one lock hold, the events in time order with the days closing, one save. Nested blocks join the
        outermost. A hook on another thread, or outside a block, applies directly. A crash inside a block loses the block's
        events from the engine until a rebuild (accepted edge F-DA).
        """
        local = self._batch
        depth = getattr(local, "depth", 0)
        if depth == 0:
            local.events = []
        local.depth = depth + 1
        try:
            yield
        finally:
            local.depth -= 1
            if local.depth == 0:
                collected, local.events = local.events, None
                if collected:
                    self._apply_direct(collected)

    # --- the one apply step (the drain and the direct apply) -----------------------------------------

    def _mark_failed(self, why: str, now: datetime, *, exc: BaseException | None = None) -> None:
        """A live persona's state cannot be used: set the catch-up-failed flag (it empties `read()`), log once per active day.

        Nothing is written, the queue and its position are kept, and a memory stays saved. The retry is the next call: a failed
        live catch-up reruns on every tick with the lock held (the spec's cost, 3.0 "Costs recorded").
        """
        self._catch_up_failed = True
        try:
            day = day_of(now)
        except (OverflowError, OSError, ValueError):
            day = None
        if day is None or self._failure_logged_day != day:
            self._failure_logged_day = day
            logger.error(
                "emotion engine: %s; nothing is applied until a state write succeeds, the queue and its position are kept",
                why,
                exc_info=exc,
            )

    def _replay_emotions(self, memory: Memory) -> Mapping[str, float]:
        """What the catch-up folds for a stored row: its faded emotions when it counts, nothing when it does not."""
        return self._resolved_faded(memory) if counted_kind(memory) is not None else {}

    def _replay_modifier(self, memory: Memory) -> float:
        kind = counted_kind(memory)
        return self._modifier(memory, kind) if kind is not None else 1.0

    def _fold_events(self, state: EngineState, events: list[Event], now: datetime, skip_ids: frozenset[str]) -> bool:
        """Fold `events` in time order, each with the days closing first; True when at least one event folded.

        A commit or assigned event of a memory the recovery already counted (`skip_ids`), or already folded in this same
        apply (E6.7: two events for one id in a batch), folds once at most; a showing is never skipped, and a start-only
        event has no memory id, so it is never skipped (its min-fold is idempotent). An event that raises is put back and
        skipped (`fold_guarded`), and the others still fold.
        """
        applied = False
        folded_ids: set[str] = set()
        for event in sorted(events, key=lambda e: e.at):
            memory_id = event.memory_id if event.kind in ("commit", "assigned") else None
            if memory_id is not None and (memory_id in skip_ids or memory_id in folded_ids):
                continue

            def step(event: Event = event) -> None:
                close_days(state, min(event.at, now))
                fold_event(state, event, now)

            if fold_guarded(state, step, f"{event.kind} event at {event.at}", key=f"engine:fold:{self.persona_dir}"):
                applied = True
                if memory_id is not None:
                    folded_ids.add(memory_id)
        return applied

    def _run_apply(self, events: list[Event], *, drain: bool, now: datetime) -> None:
        """The one apply step of the drain and the direct apply. The caller holds the engine lock (it is not re-entrant).

        Steps (spec 3.1 S4): (2) read the state, recovery first when it came from a backup copy, then (for the drain) read the
        queue by position; (3-5) fold the events in time order with `close_days` before each and once at `now`, prune the
        change-log files and the repeat list; (6) one `save_state`, with the queue position, when anything was read, folded or
        closed, when a catch-up ran, or when an empty state replaced one that could not be used; (7) the drain then empties
        the queue file and saves position 0. A direct apply passes `events` (its own, with
        `drain=False`) and writes no queue and reads none except the catch-up's one read. `now` is held to the loaded
        `saved_at` for the whole step, so a clock read before a wait for the lock cannot rewind the state, and `saved_at`
        never moves backwards (a clock set back does not drag it back, by design).
        """
        persona = self.persona_dir
        when = lenient_utc(now)
        if when is None:
            raise ValueError(f"now is not a usable time: {now!r}")
        now = when
        if drain:
            self._last_queue_key = _UNSET  # set again only by a drain that ends clean
        loaded = engine_files.read_state(persona)
        state = loaded.state
        skip_ids: frozenset[str] = frozenset()
        queue_read: engine_files.QueueRead | None = None
        caught_up = False
        fallback_empty = False
        pre_day = state.active_day if state is not None else None  # the active day as it stood BEFORE this step's own closes
        if state is None:
            if self.is_live():
                self._mark_failed(f"the emotion state is {loaded.problem}", now)
                return
            # A persona that is not live starts from an empty state (E6.17). A state that does not exist yet is nothing to
            # save by itself; one that exists but cannot be read is replaced, so the failing load is not repeated every call.
            state, fallback_empty = EngineState(), loaded.problem != "missing"
        else:
            if state.saved_at is not None:
                now = max(now, state.saved_at)
            if loaded.source == "main":
                self._catch_up_failed = False  # a readable main state ends a failure (a state repaired outside the engine)
            else:
                # Recovery first (sfa S40). Known risk (RD7): the walk skips a main file with an I/O fault that outlasts the
                # shared read's own retries, and loads `.bak1`. That runs this recovery, which gives a close approximation of the
                # state, not the state: a longer hold or a real disk error can replace a good main with a replayed one.
                try:
                    queue_read = engine_files.read_queue(persona, from_line=0)
                    skip_ids = engine_rebuild.catch_up(
                        persona,
                        state,
                        now,
                        queued=queue_read.events,
                        emotions_of=self._replay_emotions,
                        modifier_of=self._replay_modifier,
                    )
                    caught_up = True
                except Exception as exc:  # noqa: BLE001 - every failure of the catch-up is the failed-catch-up path
                    if self.is_live():
                        self._mark_failed(f"the catch-up of the backup state failed ({type(exc).__name__}: {exc})", now, exc=exc)
                        return
                    logger.warning(
                        "emotion engine: the catch-up failed on a persona that is not live; starting from an empty state (%s: %s)",
                        type(exc).__name__,
                        exc,
                    )
                    state, queue_read, fallback_empty, pre_day = EngineState(), None, True, None
        if drain:
            if queue_read is None:
                queue_read = engine_files.read_queue(persona, from_line=state.queue_line)
            effective_from = 0 if (caught_up or state.queue_line > queue_read.end_line) else state.queue_line
            lines_read = queue_read.end_line > effective_from
            to_fold = list(queue_read.events)
        else:
            lines_read = False
            to_fold = []
            for event in events:
                try:  # the check every queued line already passed: UTC time, a day `day_of` can convert, a held modifier
                    to_fold.append(engine_files.event_from_json(engine_files.event_to_json(event)))
                except (ValueError, OverflowError, OSError, TypeError) as exc:
                    logger.warning("emotion engine: a %s event was dropped (%s: %s)", event.kind, type(exc).__name__, exc)
        applied = self._fold_events(state, to_fold, now, skip_ids)
        fold_guarded(state, lambda: close_days(state, now), "the day close", key=f"engine:close:{persona}")
        window = timedelta(hours=dev_constants.EMOTION_REPEAT_WINDOW_H)
        for memory_id in [m for m, entry in state.repeats.items() if entry.first_at + window <= now]:
            del state.repeats[memory_id]
        day_closed = pre_day is not None and state.active_day != pre_day
        if day_closed:
            engine_files.prune_changes(persona, pre_day)
        if not (lines_read or day_closed or caught_up or fallback_empty or (not drain and applied)):
            if drain and queue_read is not None and queue_read.end_line > 0:
                self._empty_and_remember(state)  # a retry of an emptying that failed before
            elif drain:
                self._remember_queue_key()
            return
        state.saved_at = now
        if caught_up:
            state.reset_at = now
        if drain:
            state.queue_line = queue_read.end_line  # type: ignore[union-attr]
        engine_files.save_state(persona, state)
        self._catch_up_failed = False
        if drain:
            if queue_read.end_line > 0:  # type: ignore[union-attr]
                self._empty_and_remember(state)
            else:
                self._remember_queue_key()

    def _empty_and_remember(self, state: EngineState) -> None:
        """Drain step 7: empty the queue file, and on success save position 0; remember the queue's key when all went clean.

        If the emptying fails the position stays at the end of the queue, so the next drain reads only lines after it
        (nothing is applied twice) and retries the emptying without a new save; the key is then left unset so the early-out
        does not hide the retry.

        Accepted edge S22 (spec section 6; owner 2026-11-03 "That's a very narrow edge, note it in a comment but otherwise
        accepted."): a crash between the emptying and the save of position 0 leaves the stored position past the end of the
        emptied queue. If the next reply's queued lines outgrow it before the next drain, those lines are skipped. A failed
        second save leaves the same state on disk and is read as the same edge. Also knowingly accepted: a crash can leave a
        torn last line, which the emptying discards (an R12 loss; the reader does not log a torn tail).
        """
        persona = self.persona_dir
        if not engine_files.empty_queue(persona):
            return
        state.queue_line = 0
        engine_files.save_state(persona, state)
        self._remember_queue_key()

    def _remember_queue_key(self) -> None:
        """Keep the queue file's stat key, taken NOW (after any emptying: the emptied file is a new file with a new key)."""
        try:
            stat = file_ops.stat_path(engine_files.queue_path(self.persona_dir))
        except OSError:
            self._last_queue_key = _UNSET
            return
        self._last_queue_key = None if stat is None else stat.key

    def _can_skip(self, now: datetime) -> bool:
        """The drain's early-out (1): no lock and no write when nothing can have changed since this instance's last clean drain.

        Needs a clean drain behind it (`_last_queue_key` set), the queue file's stat key unchanged since, the failed flag
        clear, a state loaded from the main file (a backup is never skipped past), an open day, and `now` still on that day.
        A stat error means cannot skip.
        """
        if self._last_queue_key is _UNSET or self._catch_up_failed:
            return False
        try:
            stat = file_ops.stat_path(engine_files.queue_path(self.persona_dir))
        except OSError:
            return False
        if (None if stat is None else stat.key) != self._last_queue_key:
            return False
        loaded = self._state_cache.get()
        state = loaded.state
        if state is None or loaded.source != "main" or state.active_day is None:
            return False
        return day_of(now) <= state.active_day

    # --- reads: change nothing, take no lock ---------------------------------------------------------

    def read(self) -> dict[str, Reading]:
        """Each present emotion's `Reading(short, long, difference, shift)`; `{}` when nothing can be read.

        Present: its long average has weight. Only names in the vocabulary registry appear (an unregistered name is logged
        once per name per engine instance; the stored state keeps it, and registering it later makes it appear). `difference`
        is `short - long`. The shift is worked out from the snapshot and the start gate once per state load and kept with the
        cached state. Empty while the state is missing or unreadable, while a catch-up failed (the flag), and when the read
        itself fails. A state loaded from a backup copy is read as it is (its values are older). Never writes, never recovers,
        never takes the lock.
        """
        try:
            self._refresh_vocabulary()
            if self._catch_up_failed:
                return {}
            loaded = self._state_cache.get()
            state = loaded.state
            if state is None:
                return {}
            shifts = loaded.shifts
            if shifts is None:
                shifts = shift_for({name: e.snapshot for name, e in state.emotions.items()}, state.persona_start_at)
                self._state_cache.store_shifts(loaded, shifts)
            readings: dict[str, Reading] = {}
            for name, emotion in state.emotions.items():
                if emotion.long.weight <= 0.0:
                    continue
                if vocabulary.get(name) is None:
                    if name not in self._warned_names:
                        self._warned_names.add(name)
                        logger.warning("emotion engine: read() left out %r, which is not in the vocabulary registry", name)
                    continue
                short, long = emotion.short.avg, emotion.long.avg
                readings[name] = Reading(short, long, short - long, shifts.get(name, 0))
            return readings
        except Exception:
            warn_throttled(logger, f"engine:read:{self.persona_dir}", "emotion engine: read() failed; nothing is returned")
            return {}

    def _refresh_vocabulary(self) -> None:
        """Ask the persona loader to re-read the vocabulary file if another process changed it, so a word registered there is seen.

        Its own `try`: a refresh that raises, takes another argument list, or does not exist yet is logged (once per distinct
        failure) and `read()` goes on with the registry as it is, never with `{}`.
        """
        refresh = getattr(persona_loader, "refresh_if_changed", None)
        if refresh is None:
            return
        try:
            refresh(engine_files.vocabulary_path(self.persona_dir))
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if failure not in self._refresh_logged:
                self._refresh_logged.add(failure)
                logger.warning("emotion engine: the vocabulary refresh failed (%s); read() goes on with the registry as it is", failure, exc_info=True)

    def status(self) -> EngineStatus:
        """`EngineStatus(caught_up, reset_at)`: False while the state is a backup copy or a catch-up failed, `reset_at` from the state.

        A missing state is `(True, None)` (there is nothing to catch up); an unreadable one, or any failure of this call, is
        `(False, None)`. Changes nothing.
        """
        try:
            loaded = self._state_cache.get()
            state = loaded.state
            reset_at = state.reset_at if state is not None else None
            if self._catch_up_failed:
                return EngineStatus(False, reset_at)
            if state is None:
                return EngineStatus(loaded.problem == "missing", None)
            return EngineStatus(loaded.caught_up, reset_at)
        except Exception:
            return EngineStatus(False, None)

    def oldest_entry_at(self) -> datetime | None:
        return None

    def is_live(self) -> bool:
        """True when the switch file exists, and also when that cannot be told (a stat error other than "no such file").

        Only a switch file that was actually SEEN is remembered for the life of the instance; an unknown answer is not
        cached, so one stat error cannot make a phase-B persona live for good. A False answer is asked again (one stat) each
        time, so the switch-over is seen without a restart.
        """
        if self._live:
            return True
        seen = engine_files.switch_state(self.persona_dir)
        if seen is True:
            self._live = True
        return seen is not False

    def faded_value(self, memory: Memory, emotion: str) -> float:
        try:
            return memory.emotions[emotion]
        except KeyError:
            raise KeyError(
                f"memory {memory.id} has no stored value for emotion {emotion!r}"
            ) from None

    # --- the real lookup ---------------------------------------------------------------------------

    def half_life(self, emotion: str) -> float:
        """The emotion's half-life in days.

        The name is made canonical first (`vocabulary.canonical_name`), because the persona file's
        names are made canonical the same way. Order: the persona's vocabulary file (an entry with a
        numeric half-life; a null entry is "never fades" and the lookup goes on), then
        `dev_constants.EMOTION_HALF_LIFE_DAYS`, then `dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS`.
        """
        name = vocabulary.canonical_name(emotion)
        persona = self._vocabulary.get(name)
        if persona is not None:
            return persona
        table = dev_constants.EMOTION_HALF_LIFE_DAYS.get(name)
        if table is not None:
            return table
        return dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS

    # --- the daily pass and recovery ---------------------------------------------------------------

    def note_active(self, now: datetime) -> None:
        """The drain and the day close (spec 3.1 S4): fold the queued events, close the days, save the position. Never raises.

        Early-out with no lock and no write when nothing can have changed (`_can_skip`). Otherwise, under the engine lock,
        the one apply step (`_run_apply`). A failure is logged and the queue and its position are kept, so the next drain
        retries: a lock wait that ran out, an I/O fault, a bug.
        """
        try:
            when = lenient_utc(now)
            if when is None:
                logger.error("emotion engine: note_active was given %r, which is not a time; nothing was done", now)
                return
            day_of(when)  # a time `day_of` cannot convert would fail every drain
            if self._can_skip(when):
                return
            with engine_files.state_lock(self.persona_dir):
                self._run_apply([], drain=True, now=when)
        except LockTimeoutError as exc:
            count_throttled(
                logger,
                f"engine:drain_lock:{self.persona_dir}",
                logging.ERROR,
                "emotion engine: the drain gave up on the engine lock (%s); the queue is kept and the next drain retries",
                exc,
            )
        except OSError as exc:
            count_throttled(
                logger,
                f"engine:drain_os:{self.persona_dir}",
                logging.ERROR,
                "emotion engine: the drain failed (%s: %s); the queue and its position are kept and the next drain retries",
                type(exc).__name__,
                exc,
            )
        except Exception as exc:
            warn_throttled(
                logger,
                f"engine:drain:{self.persona_dir}:{type(exc).__name__}",
                "emotion engine: the drain failed, an unexpected %s; the queue and its position are kept",
                type(exc).__name__,
            )

    def needs_recovery(self) -> bool:
        return False

    def recovery_rebuild(self) -> None:
        return None


# One engine per resolved persona directory. Any other code that touches `_engines` (a test fixture that
# resets the registry) must hold `_engines_lock`. The lock is not reentrant, so no constructor may call
# `engine_for`, and it is never taken while an engine's own lock is held.
_engines: dict[Path, EmotionEngine] = {}
_engines_lock = threading.Lock()


def engine_for(persona_dir: Path) -> EmotionEngine:
    """The one engine for this persona directory; two threads' first calls cannot build two."""
    key = Path(persona_dir).resolve()
    with _engines_lock:
        engine = _engines.get(key)
        if engine is None:
            engine = _engines[key] = EmotionEngine(key)
        return engine


def ensure_engine_ready(persona_dir: Path) -> None:
    """Make the engine ready to run (inert until Engine 6: writes nothing, raises nothing)."""
    return None
