"""The emotion engine's interface (Engine 0, #575; hooks, Engine 3, #578): the one module the rest of the emotion rework builds against.

Every public call exists with its final signature, so the code that calls the engine can be written and tested before the
engine behind every call exists.

Real now:
- `half_life`, the lookup of an emotion's half-life in days (the persona's vocabulary file, then
  `dev_constants.EMOTION_HALF_LIFE_DAYS`, then `EMOTION_DEFAULT_HALF_LIFE_DAYS`), and `is_live()`, which is True once
  `state/emotion_switch.json` exists (nothing writes that file until the switch-over, so it answers False for every persona
  today).
- The four hooks `on_memory_committed`, `on_emotions_assigned`, `fold_change` and `on_memory_shown`. A hook only APPENDS one
  event to the queue (`state/emotion_pending.jsonl`) under the engine's lock; it never folds anything and never raises into
  the write that called it. Everything the event needs is worked out before the lock is taken, and nothing inside the lock
  hold touches `memories.db` or a model. A memory of a type that does not count queues nothing, except a start-only event
  when the stored start date of the persona is missing or later than the row's date.

Stubbed now:
- `read()` returns `{}`; `status()` returns `EngineStatus(True, None)`; `oldest_entry_at()` returns None.
- `faded_value(memory, emotion)` returns the stored value and raises `KeyError`, naming the
  memory id and the emotion, when the memory has no such emotion. It never returns 0.0:
  "absent" is for the caller to decide.
- `note_active`, `needs_recovery`, `recovery_rebuild` and `ensure_engine_ready` do nothing; `EngineStartupError` is defined and
  nothing raises it yet.

Callers use the receiver form `engine.engine_for(persona_dir).<call>` and import the module
(`from brain.emotion import engine`), not its names, so one patch reaches every call site.
"""

from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from brain import dev_constants
from brain.emotion import engine_files, vocabulary
from brain.emotion.engine_files import VocabularyHalfLives
from brain.emotion.engine_utils import Event, counted_kind, resolve_emotions, source_modifier
from brain.utils.log_throttle import count_throttled
from brain.utils.time import lenient_utc

if TYPE_CHECKING:
    from brain.memory.store import Memory

logger = logging.getLogger(__name__)


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

    An `OSError` is the queue's folder or the lock's file refusing us (a read-only or full disk, a share that went away). It
    stays true for every write until the cause is fixed, so it is logged once per interval, without a traceback
    (`count_throttled`). Anything else is a bug in the hook: one error line with the traceback, every time.
    """

    @functools.wraps(hook)
    def guarded(self: EmotionEngine, *args: object, **kwargs: object) -> None:
        try:
            hook(self, *args, **kwargs)
        except OSError as exc:
            count_throttled(
                logger,
                f"engine:hook:{self.persona_dir}",
                logging.ERROR,
                "emotion engine: %s could not queue its event (%s: %s)",
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
        # without a lock. Owned here; the drain (Engine 4) reads the state through this same cache.
        self._state_cache = engine_files.StateCache(self.persona_dir)
        self._live = False  # a True answer is kept for the life of the instance

    # --- hooks the memory store and the extractor call: each appends one event to the queue ----------
    #
    # A hook never folds and never raises (`_never_raises`). Its steps: the cheap `counted_kind` test first; everything the
    # event needs is worked out BEFORE the lock is taken (the row, the trust lookup, the half-lives, the faded values); then
    # the lock, one append, the release. Nothing inside the lock hold touches `memories.db`, the kindled-link store or a model.

    @_never_raises
    def on_memory_committed(self, memory: Memory) -> None:
        kind = counted_kind(memory)
        if kind is None:
            self._queue_start_if_needed(memory.created_at)
            return
        emotions = self._resolved_faded(memory)
        if not emotions:  # nothing to fold, but the row's date still reaches the persona's start (B02.6)
            self._queue_start_if_needed(memory.created_at)
            return
        self._queue(
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
        self._queue(
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

        Read through the cache with no lock. The check only ever says "not needed" from a stored start that is already
        at or before the row, and the stored start only moves earlier, so a state replaced meanwhile cannot make that wrong.
        A backup copy of the state is older, so its start is never earlier than the main file's: it can only add a duplicate
        start event, which the drain's min-fold makes harmless.
        """
        created = lenient_utc(created_at)
        if created is None:
            return False
        state = self._state_cache.get().state
        return state is None or state.persona_start_at is None or created < state.persona_start_at

    def _queue_start_if_needed(self, created_at: datetime) -> None:
        if self._start_needed(created_at):
            self._queue(Event(kind="start", at=created_at))

    def _queue(self, event: Event) -> None:
        """Append `event` to the queue, holding the engine's lock for the append only.

        Accepted edge (E2, R12, owner 2026-10-21: "treat it as a fairly minor edge case, noted in code"): an entry can be lost
        (a failed append, a hook that raised, a crash between the commit and this append, a line damaged by an unclean stop).
        It is not recovered. A lost entry is a minor edge while appends succeed; a fault that persists loses every entry for
        as long as it lasts, and the log line is then the only signal. `append_event` logs a failed append and returns False;
        nothing more is done with that here.
        """
        with engine_files.state_lock(self.persona_dir):
            engine_files.append_event(self.persona_dir, event)

    # --- reads (stubbed) ---------------------------------------------------------------------------

    def read(self) -> dict[str, Reading]:
        return {}

    def status(self) -> EngineStatus:
        return EngineStatus(True, None)

    def oldest_entry_at(self) -> datetime | None:
        return None

    def is_live(self) -> bool:
        """True when the switch file exists. A True answer is cached for the life of the instance; a False one is asked
        again (one stat) each time, so the switch-over is seen without a restart."""
        if not self._live:
            self._live = engine_files.switch_exists(self.persona_dir)
        return self._live

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

    # --- recovery and the daily pass (inert until the slices that build them) ----------------------

    def note_active(self, now: datetime) -> None:
        return None

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
    """Make the engine ready to run (inert in this slice: writes nothing, raises nothing)."""
    return None
