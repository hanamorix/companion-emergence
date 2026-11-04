"""The emotion engine's interface (Engine 0, #575): the one module the rest of the emotion rework builds against.

This is the stub. Every public call exists with its final signature, so the code that will call
the engine can be written and tested before the engine behind the calls exists.

Real now: `half_life`, the lookup of an emotion's half-life in days (the persona's vocabulary
file, then `dev_constants.EMOTION_HALF_LIFE_DAYS`, then `EMOTION_DEFAULT_HALF_LIFE_DAYS`), and
`is_live()`, which is True once `state/emotion_switch.json` exists (nothing writes that file until
the switch-over, so it answers False for every persona today).

Stubbed now:
- `on_memory_committed`, `on_emotions_assigned`, `fold_change` and `on_memory_shown` append their
  arguments to `recorded_calls` and do nothing else.
- `batch()` is a context manager that does nothing: it takes no argument, yields None and records nothing.
- `read()` returns `{}`; `status()` returns `EngineStatus(True, None)`; `oldest_entry_at()` returns None.
- `faded_value(memory, emotion)` returns the stored value and raises `KeyError`, naming the
  memory id and the emotion, when the memory has no such emotion. It never returns 0.0:
  "absent" is for the caller to decide.
- `note_active`, `needs_recovery`, `recovery_rebuild` and `ensure_engine_ready` do nothing and
  are not recorded; `EngineStartupError` is defined and nothing raises it yet.

Callers use the receiver form `engine.engine_for(persona_dir).<call>` and import the module
(`from brain.emotion import engine`), not its names, so one patch reaches every call site.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from brain import dev_constants
from brain.emotion import engine_files, vocabulary
from brain.emotion.engine_files import VocabularyHalfLives

if TYPE_CHECKING:
    from brain.memory.store import Memory


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


class EmotionEngine:
    """The engine for one persona directory. Get it through `engine_for(persona_dir)`."""

    def __init__(self, persona_dir: Path, *, now: Callable[[], datetime] = _utcnow) -> None:
        self.persona_dir = Path(persona_dir)
        self._now = now
        # One entry per recorded call: `(call_name, *arguments in the call's own order)`, for example
        # `("on_memory_shown", memory, at)` or `("fold_change", emotions, at, source)`. The entries hold
        # the caller's own objects, not copies. Read by tests; the stub writes it.
        self.recorded_calls: list[tuple[Any, ...]] = []
        self._vocabulary = VocabularyHalfLives(self.persona_dir)
        self._live = False  # a True answer is kept for the life of the instance

    # --- hooks the memory store and the extractor call (stubbed: record and return) ---------------

    def on_memory_committed(self, memory: Memory) -> None:
        self.recorded_calls.append(("on_memory_committed", memory))

    def on_emotions_assigned(self, memory: Memory) -> None:
        self.recorded_calls.append(("on_emotions_assigned", memory))

    def fold_change(self, emotions: Mapping[str, float], at: datetime | None, source: str) -> None:
        self.recorded_calls.append(("fold_change", emotions, at, source))

    def on_memory_shown(self, memory: Memory, at: datetime | None) -> None:
        self.recorded_calls.append(("on_memory_shown", memory, at))

    # --- batch scope (stubbed: does nothing) -------------------------------------------------------

    @contextmanager
    def batch(self) -> Iterator[None]:
        """A bulk writer wraps its loop in this block. The stub does nothing.

        The engine built later collects the block's events and applies them once when the outermost
        block ends; here nothing is collected, applied or recorded, and an exception from the block
        goes out unchanged.
        """
        yield

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
