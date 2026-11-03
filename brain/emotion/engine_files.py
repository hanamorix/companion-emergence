"""The files the emotion engine owns, and the one place that knows their paths and their lock (Engine 0, #575; Engine 2, #577).

Everything lives under ``<persona_dir>/state/`` (``paths.state_dir``); no code of ours creates that folder, because
``write_json``, ``append_jsonl`` and ``file_lock`` each create a missing parent.

* ``emotion_state.json`` (+ ``.bak1``-``.bak3``): the averages, the weights and a 30-day snapshot per emotion. Written with
  ``write_json(backups=3, indent=None)``. Read through ``json_io.read_json_walk``, the read-only walk with the validator
  below: a read never heals, quarantines, rotates or rewrites anything, and says which copy it loaded. ``read_state`` reads
  fresh objects (what a drain mutates); ``StateCache`` keeps the parsed state for readers by a stat key.
* ``emotion_pending.json``: the queue, a JSON list of events waiting to be folded. No backups (a drained queue must not come
  back from one); removed when empty; a missing file is an empty queue.
* ``emotion_changes/YYYY-MM-DD.jsonl``: the change log, one file a day, a crash-recovery record of chat-derived changes.
  Whole files are deleted once they are older than the retention boundary.
* ``emotion_switch.json``: ``{"switch_over_at": ...}`` and nothing else, written once; its existence means the engine is live.
* ``emotion_vocabulary.json`` (the persona's own file, owned by the persona loader and the growth code): only READ here, for
  the half-life cache (``VocabularyHalfLives``, behind ``EmotionEngine.half_life``).

The lock. ONE lock guards the four files: ``state_lock(persona_dir)``, a blocking, non-re-entrant ``file_lock`` of the state
file. No function in this module takes it; every caller holds it (a hook around its appends, a drain around its whole run, the
rebuild around everything) and no code path may take it twice. Lock-free by design: ``read_state``, ``StateCache``,
``switch_exists``, ``read_switch``, ``read_queue``, ``read_changes`` and ``list_change_files`` (a reader never waits for a
writer; the writers replace files atomically). Expecting the caller's lock: ``save_state``, ``append_event``, ``write_queue``,
``clear_queue``, ``append_change``, ``prune_changes`` and ``write_switch``.

Times are UTC ISO strings written with ``iso_utc`` (``Z``) and read with ``lenient_utc``; day labels are local ``YYYY-MM-DD``
from ``engine_utils.day_of``. All constants (file and folder names, the retention length, the value cap) are read as
``dev_constants.NAME`` at call time.

The half-life cache (Engine 0). The persona's own ``emotion_vocabulary.json`` is read, never written, never repaired and
never quarantined (``Recovery.DEFAULT``: bad content takes the default). It is re-read when its stat key
``(st_mtime_ns, st_size, st_ino)`` changes. The key is taken BEFORE the read, so a file replaced between the stat and the read
is read again on the next call. A failure that says nothing about the content (a filesystem error on the read, an error from
``stat`` other than "no such file") is not cached: the previously loaded half-lives stay in force and the next call tries
again (the same bound as ``brain/tunables.py``, #530). A fault in the content (not JSON, wrong shape, a bad entry) is cached,
because the same bytes give the same fault. A bad entry is reported once per change of the file; a file that is not JSON or
not the right shape is reported by ``read_json_result``, which follows its own throttle (one warning per path and kind per
interval).
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

from brain import dev_constants, paths
from brain.emotion.engine_utils import (
    Average,
    EmotionState,
    EngineState,
    Event,
    RepeatEntry,
    RunRecord,
    SnapshotDay,
    day_of,
)
from brain.emotion.vocabulary import canonical_name
from brain.health.json_io import (
    JsonFileError,
    Recovery,
    append_jsonl,
    iter_jsonl,
    read_json_result,
    read_json_walk,
    write_json,
)
from brain.utils.file_lock import file_lock
from brain.utils.log_throttle import count_throttled, warn_throttled
from brain.utils.time import iso_utc, lenient_utc

logger = logging.getLogger(__name__)

_VOCABULARY_FILENAME = "emotion_vocabulary.json"

_StatKey = tuple[int, int, int]


class _Entry(NamedTuple):
    name: str
    days: float | None  # None: stored as "never fades"; the lookup goes on to the next step


class _Fault(NamedTuple):
    name: str | None  # None: the entry has no name to report
    field: str
    why: str


def _stat_key(path: Path) -> _StatKey | None:
    """The file's stat key, or None when there is no such file. Any other error from `stat` is raised."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _emotions_list(data: dict) -> None:
    """The same rule as the persona loader's schema check: a top-level `emotions` list."""
    emotions = data.get("emotions")
    if not isinstance(emotions, list):
        found = "missing" if "emotions" not in data else type(emotions).__name__
        raise ValueError(f"'emotions' must be a list, found {found}")


def _classify(entry: Any) -> _Entry | _Fault:
    """One vocabulary entry: its name and half-life, or the specific fault.

    Only `name` and `decay_half_life_days` matter to the engine. A present `name` that is not
    text is turned into text with `str()`, as the persona loader does; only a missing one is
    a fault.
    """
    if not isinstance(entry, dict):
        return _Fault(None, "entry", "not an object")
    if "name" not in entry:
        return _Fault(None, "name", "missing")
    name = str(entry["name"])
    if "decay_half_life_days" not in entry:
        return _Fault(name, "decay_half_life_days", "missing")
    value = entry["decay_half_life_days"]
    if value is None:
        return _Entry(name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _Fault(
            name, "decay_half_life_days", f"must be a number or null, found {type(value).__name__}"
        )
    try:
        number = float(value)
    except OverflowError:
        return _Fault(name, "decay_half_life_days", "too large to be a number of days")
    if not math.isfinite(number) or number <= 0:
        return _Fault(
            name, "decay_half_life_days", f"must be a finite number above 0, found {number!r}"
        )
    return _Entry(name, number)


class VocabularyHalfLives:
    """The persona vocabulary file's numeric half-lives, by canonical name, kept current by stat key."""

    def __init__(self, persona_dir: Path) -> None:
        self._path = Path(persona_dir) / _VOCABULARY_FILENAME
        # One lock covers the stat, the reload and the lookup, so a second thread's first call waits
        # for the load instead of reading the file again or seeing a key without its table.
        self._lock = threading.Lock()
        self._loaded = False
        self._key: _StatKey | None = None
        self._table: dict[str, float] = {}

    def get(self, name: str) -> float | None:
        """The persona file's numeric half-life for the canonical `name`, or None.

        None means the persona file has no usable number for it (no file, not listed, a null
        entry, a faulted entry, or a failure that was not cached): the caller goes on to the
        built-in table.
        """
        with self._lock:
            self._refresh()
            return self._table.get(name)

    def _refresh(self) -> None:
        try:
            key = _stat_key(self._path)
        except OSError as exc:
            # Says nothing about the content: keep what is loaded, store nothing, try again next call.
            warn_throttled(
                logger,
                f"engine_files:stat:{self._path}:{exc}",
                "emotion vocabulary %s cannot be inspected (%s); the half-lives already loaded stay in force",
                self._path,
                exc,
            )
            return
        if self._loaded and key == self._key:
            return
        if key is None:
            self._table, self._key, self._loaded = {}, None, True
            return
        result = read_json_result(
            self._path, default=dict, recover=Recovery.DEFAULT, expect=dict, validate=_emotions_list
        )
        if result.outcome == "defaulted" and result.kind == "io":
            # `read_json_result` has already logged it (throttled). Not cached: same key, read again.
            return
        self._table = self._parse(result.data)
        self._key = key
        self._loaded = True

    def _parse(self, data: dict) -> dict[str, float]:
        decided: set[str] = set()
        table: dict[str, float] = {}
        for index, entry in enumerate(data.get("emotions", ())):
            parsed = _classify(entry)
            if isinstance(parsed, _Fault):
                label = repr(parsed.name) if parsed.name is not None else f"emotions[{index}]"
                logger.warning(
                    "emotion vocabulary %s: entry %s ignored (field %s: %s); the built-in half-life applies to it",
                    self._path,
                    label,
                    parsed.field,
                    parsed.why,
                )
                continue
            key = canonical_name(parsed.name)
            if key in decided:
                continue  # the first valid entry for a name decides, as the persona loader registers it
            decided.add(key)
            if parsed.days is not None:
                table[key] = parsed.days
        return table


# ================================================================================================
# Paths and the lock
# ================================================================================================


def state_path(persona_dir: Path) -> Path:
    return paths.state_dir(persona_dir) / dev_constants.EMOTION_STATE_FILENAME


def queue_path(persona_dir: Path) -> Path:
    return paths.state_dir(persona_dir) / dev_constants.EMOTION_PENDING_FILENAME


def switch_path(persona_dir: Path) -> Path:
    return paths.state_dir(persona_dir) / dev_constants.EMOTION_SWITCH_FILENAME


def changes_dir(persona_dir: Path) -> Path:
    return paths.state_dir(persona_dir) / dev_constants.EMOTION_CHANGES_DIRNAME


def state_lock(persona_dir: Path) -> AbstractContextManager[bool]:
    """The one lock of the engine's four files: ``file_lock`` of the state file (sidecar ``emotion_state.json.lock``).

    Blocking, no timeout, NOT re-entrant. The hooks, the drain and the rebuild take it through this helper, once each;
    nothing in this module takes it (see the module docstring).
    """
    return file_lock(state_path(persona_dir))


# ================================================================================================
# The state file
# ================================================================================================


def _finite(value: object, what: str) -> float:
    """A finite number (not a bool, not too large for a float), else ValueError naming `what`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, found {type(value).__name__}")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{what} is too large to be a number") from None
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite, found {number!r}")
    return number


def _time(value: object, what: str) -> datetime:
    """A time the engine can use: text `lenient_utc` accepts and `day_of` can turn into a local day, else ValueError.

    The second test is not decoration: a time that parses but cannot be converted (year 1, or before 1970 on Windows) would
    make every close of that state fail for ever.
    """
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a time as text, found {type(value).__name__}")
    parsed = lenient_utc(value)
    if parsed is None:
        raise ValueError(f"{what} is not a time: {value!r}")
    try:
        day_of(parsed)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f"{what} cannot be turned into a local day: {value!r}") from exc
    return parsed


def _day(value: object, what: str) -> str:
    """A real calendar day written exactly ``YYYY-MM-DD``, else ValueError ("20260101" parses but is not the form)."""
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a day as text, found {type(value).__name__}")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{what} is not a day: {value!r}") from None
    if parsed.isoformat() != value:
        raise ValueError(f"{what} is not written YYYY-MM-DD: {value!r}")
    return value


def _hold(value: float) -> float:
    return min(max(value, 0.0), dev_constants.EMOTION_VALUE_CAP)


def _mapping(parent: Mapping, key: str, what: str) -> Mapping:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{what}.{key} must be an object, found {type(value).__name__ if key in parent else 'nothing'}")
    return value


def validate_state(data: object) -> None:
    """Raise ValueError unless `data` is a state the engine can use (the validator `read_json_walk` applies to every copy).

    Rejects: not an object; a missing or unparseable `saved_at`; an unparseable `reset_at`, `last_update`, `oldest_entry_at`
    or `persona_start_at` (a missing or null `persona_start_at` is accepted: a closed gate until the first memory); an
    `active_day` that is not a real day, or unset while `last_update` is set; `emotions` or `repeats` of the wrong type; an
    emotion that is not an object or lacks `short`, `long` or `last_entry_day`; a non-finite number, a bool, or a negative
    weight in an average; a `last_entry_day` that is not a real day or is later than `active_day`. An average outside 0-10 is
    NOT a rejection (`state_from_json` holds it), and neither is anything wrong with a snapshot or a repeat record (those are
    dropped one at a time, never the whole state). Every refusal is a ValueError.
    """
    if not isinstance(data, dict):
        raise ValueError(f"the state must be an object, found {type(data).__name__}")
    if "saved_at" not in data:
        raise ValueError("saved_at is missing")
    _time(data["saved_at"], "saved_at")
    for key in ("reset_at", "last_update", "oldest_entry_at", "persona_start_at"):
        if data.get(key) is not None:
            _time(data[key], key)
    active_day = data.get("active_day")
    if active_day is not None:
        _day(active_day, "active_day")
    elif data.get("last_update") is not None:
        raise ValueError("active_day is unset while last_update is set")
    if "repeats" in data and not isinstance(data["repeats"], dict):
        raise ValueError(f"repeats must be an object, found {type(data['repeats']).__name__}")
    emotions = data.get("emotions")
    if not isinstance(emotions, dict):
        raise ValueError(f"emotions must be an object, found {type(emotions).__name__ if 'emotions' in data else 'nothing'}")
    for name, emotion in emotions.items():
        where = f"emotion {name!r}"
        if not isinstance(emotion, dict):
            raise ValueError(f"{where} must be an object, found {type(emotion).__name__}")
        for key in ("short", "long"):
            average = _mapping(emotion, key, where)
            for field in ("avg", "weight"):
                if field not in average:
                    raise ValueError(f"{where}.{key}.{field} is missing")
                number = _finite(average[field], f"{where}.{key}.{field}")
                if field == "weight" and number < 0:
                    raise ValueError(f"{where}.{key}.weight must not be negative, found {number!r}")
        if "last_entry_day" not in emotion:
            raise ValueError(f"{where}.last_entry_day is missing")
        last_entry_day = emotion["last_entry_day"]
        if last_entry_day is not None:
            _day(last_entry_day, f"{where}.last_entry_day")
            if active_day is not None and last_entry_day > active_day:
                raise ValueError(f"{where}.last_entry_day {last_entry_day} is later than active_day {active_day}")


def _snapshot_day(item: object, previous_day: str | None) -> SnapshotDay:
    """One snapshot day, or ValueError saying what is wrong with it. Averages outside 0-10 are held, not refused."""
    if not isinstance(item, list) or len(item) != 5:
        raise ValueError("a snapshot day is an array of five: day, short, long, felt, run")
    day = _day(item[0], "day")
    if previous_day is not None and day <= previous_day:
        raise ValueError(f"day {day} is not later than the day before it, {previous_day}")
    short = _hold(_finite(item[1], "short average"))
    long = _hold(_finite(item[2], "long average"))
    felt = item[3]
    if type(felt) is not int or felt not in (0, 1):
        raise ValueError(f"felt must be 0 or 1, found {felt!r}")
    run: RunRecord | None = None
    if item[4] is not None:
        record = item[4]
        if not isinstance(record, list) or len(record) != 3:
            raise ValueError("a run record is null or an array of three: start day, direction, last day with the direction")
        start = _day(record[0], "run start day")
        direction = record[1]
        if type(direction) is not int or direction not in (1, -1):
            raise ValueError(f"run direction must be 1 or -1, found {direction!r}")
        last_with = _day(record[2], "run last with-direction day")
        if start > day:
            raise ValueError(f"run start day {start} is later than the day {day}")
        if last_with > day:
            raise ValueError(f"run last with-direction day {last_with} is later than the day {day}")
        if last_with < start:
            raise ValueError(f"run last with-direction day {last_with} is before its start day {start}")
        if (date.fromisoformat(day) - date.fromisoformat(last_with)).days > 1:
            raise ValueError(f"run last with-direction day {last_with} is more than one day before the day {day}")
        run = RunRecord(start, direction, last_with)
    return SnapshotDay(day, short, long, bool(felt), run)


def _snapshot_from_json(name: str, raw: object, label: str) -> list[SnapshotDay]:
    if not isinstance(raw, list):
        count_throttled(
            logger,
            f"engine_files:snapshot:{label}:{name}",
            logging.ERROR,
            "emotion state %s: emotion %r has no usable snapshot list; treated as an empty snapshot",
            label,
            name,
        )
        return []
    days: list[SnapshotDay] = []
    previous: str | None = None
    for index, item in enumerate(raw):
        try:
            day = _snapshot_day(item, previous)
        except ValueError as exc:
            count_throttled(
                logger,
                f"engine_files:snapshot:{label}:{name}:{index}",
                logging.ERROR,
                "emotion state %s: snapshot day %d of emotion %r dropped as a missing day (%s)",
                label,
                index,
                name,
                exc,
            )
            continue
        days.append(day)
        previous = day.day
    return days


def state_from_json(data: dict, *, label: str = "state") -> EngineState:
    """The `EngineState` of a state that passed `validate_state`.

    Averages (the state's and the snapshot's) are held to 0-10. A bad snapshot day is dropped as a missing day and a bad
    repeat record is dropped, each with an error log line naming the emotion or the memory; the state stays valid. `label`
    names the file in those lines.
    """
    repeats: dict[str, RepeatEntry] = {}
    for memory_id, record in data.get("repeats", {}).items():
        try:
            if not isinstance(record, dict):
                raise ValueError("a repeat record is an object with first_at and count")
            count = record.get("count")
            if type(count) is not int or count < 0:
                raise ValueError(f"count must be a whole number, 0 or more, found {count!r}")
            repeats[memory_id] = RepeatEntry(_time(record.get("first_at"), "first_at"), count)
        except ValueError as exc:
            logger.error("emotion state %s: repeat record for memory %r dropped (%s)", label, memory_id, exc)
    emotions: dict[str, EmotionState] = {}
    for name, emotion in data["emotions"].items():
        emotions[name] = EmotionState(
            short=Average(_hold(float(emotion["short"]["avg"])), float(emotion["short"]["weight"])),
            long=Average(_hold(float(emotion["long"]["avg"])), float(emotion["long"]["weight"])),
            last_entry_day=emotion["last_entry_day"],
            snapshot=_snapshot_from_json(name, emotion.get("snapshot"), label),
        )

    def optional(key: str) -> datetime | None:
        value = data.get(key)
        return None if value is None else lenient_utc(value)

    return EngineState(
        saved_at=lenient_utc(data["saved_at"]),
        reset_at=optional("reset_at"),
        last_update=optional("last_update"),
        active_day=data.get("active_day"),
        oldest_entry_at=optional("oldest_entry_at"),
        persona_start_at=optional("persona_start_at"),
        repeats=repeats,
        emotions=emotions,
    )


def _iso(value: datetime, what: str) -> str:
    """A time as a UTC ISO string with the Z suffix; a naive time means UTC, as everywhere in the repo."""
    utc = lenient_utc(value)
    if utc is None:
        raise ValueError(f"{what} is not a usable time: {value!r}")
    return iso_utc(utc)


def _optional_iso(value: datetime | None, what: str) -> str | None:
    return None if value is None else _iso(value, what)


def state_to_json(state: EngineState) -> dict:
    """The on-disk form of a state (the shape of ``emotion_state.json``); `state.saved_at` must be set."""
    if state.saved_at is None:
        raise ValueError("cannot write a state with no saved_at")
    emotions: dict[str, dict] = {}
    for name, emotion in state.emotions.items():
        emotions[name] = {
            "short": {"avg": emotion.short.avg, "weight": emotion.short.weight},
            "long": {"avg": emotion.long.avg, "weight": emotion.long.weight},
            "last_entry_day": emotion.last_entry_day,
            "snapshot": [
                [
                    day.day,
                    day.short,
                    day.long,
                    1 if day.felt else 0,
                    None if day.run is None else [day.run.start_day, day.run.direction, day.run.last_with_day],
                ]
                for day in emotion.snapshot
            ],
        }
    return {
        "saved_at": _iso(state.saved_at, "saved_at"),
        "reset_at": _optional_iso(state.reset_at, "reset_at"),
        "last_update": _optional_iso(state.last_update, "last_update"),
        "active_day": state.active_day,
        "oldest_entry_at": _optional_iso(state.oldest_entry_at, "oldest_entry_at"),
        "persona_start_at": _optional_iso(state.persona_start_at, "persona_start_at"),
        "repeats": {
            memory_id: {"first_at": _iso(entry.first_at, "first_at"), "count": entry.count}
            for memory_id, entry in state.repeats.items()
        },
        "emotions": emotions,
    }


def save_state(persona_dir: Path, state: EngineState) -> None:
    """Write the state file atomically with three backups, compact. The caller holds the lock and has set `state.saved_at`.

    Does not change `state`. Serialization is strict: a non-finite number fails before any file is touched.
    """
    write_json(state_path(persona_dir), state_to_json(state), backups=3, indent=None)


class LoadedState(NamedTuple):
    """One load of the state file, immutable as a whole (the cache replaces it with one assignment).

    `state` is None when no copy was usable (`problem` says why). `source` is the copy it came from. `key` is the stat key
    ``(st_mtime_ns, st_size, st_ino)`` of the bytes parsed, from the walk's ``fstat`` signature. `shifts` is what Engine 4
    works out and stores with `StateCache.store_shifts`. The `state` inside is a mutable object shared by everyone holding
    this tuple: treat it as read-only (a drain that folds reads its own copy with `read_state`).
    """

    key: tuple[int, int, int] | None
    state: EngineState | None
    source: str | None
    problem: str | None
    shifts: Mapping[str, int] | None = None

    @property
    def caught_up(self) -> bool:
        """False while the state came from a backup copy, which is as old as that copy's `saved_at` (E5.19)."""
        return self.source == "main"


def read_state(persona_dir: Path) -> LoadedState:
    """Read the state through the no-heal walk, every time, into fresh objects. Takes no lock and writes nothing."""
    path = state_path(persona_dir)
    walk = read_json_walk(path, expect=dict, validate=validate_state)
    if walk.problem == "unreadable":
        count_throttled(
            logger,
            f"engine_files:unreadable:{path}",
            logging.ERROR,
            "emotion state %s and its backups exist but none is usable; the state is unreadable",
            path,
        )
    if walk.problem is not None:
        return LoadedState(None, None, None, walk.problem)
    if walk.source != "main":
        count_throttled(
            logger,
            f"engine_files:backup:{path}:{walk.source}",
            logging.WARNING,
            "emotion state %s is not usable; loaded its older copy %s",
            path,
            walk.source,
        )
    state = state_from_json(walk.data, label=f"{path} ({walk.source})")
    ino, mtime_ns, size = walk.signature
    return LoadedState((mtime_ns, size, ino), state, walk.source, None)


class StateCache:
    """The parsed state of one persona, kept in process and re-read when the main file's stat key changes (RD6).

    Only a state loaded from the main file is kept. A state from a backup, or none at all, is read again on the next call (a
    backup is a transient condition that the next drain's catch-up ends by writing a new main file). A hit needs the cached
    key to equal the main file's current stat key; the key is the signature of the bytes that were parsed, so a file replaced
    after the read does not match and is read again. The cache is replaced by one assignment of an immutable `LoadedState`,
    so a reader on another thread sees the old one or the new one, never half of one. The state inside is shared: read-only.
    """

    def __init__(self, persona_dir: Path) -> None:
        self._persona_dir = Path(persona_dir)
        self._loaded: LoadedState | None = None

    def get(self) -> LoadedState:
        loaded = self._loaded
        if loaded is not None:
            try:
                current = _stat_key(state_path(self._persona_dir))
            except OSError:
                current = None  # cannot tell: read again
            if current is not None and current == loaded.key:
                return loaded
        fresh = read_state(self._persona_dir)
        self._loaded = fresh if fresh.source == "main" else None
        return fresh

    def store_shifts(self, loaded: LoadedState, shifts: Mapping[str, int]) -> bool:
        """Keep `shifts`, worked out from `loaded`, with the cached state. False when the cache has moved on to another load.

        Two threads racing here can leave an older tuple in the cache; its key no longer matches the file, so the next
        `get` reads again.
        """
        if self._loaded is not loaded:
            return False
        self._loaded = loaded._replace(shifts=shifts)
        return True


# ================================================================================================
# Events: the queue's and the change log's one shape and one check
# ================================================================================================

# The kinds the queue knows. "start" is the start-only event Engine 1 added: it carries only `at`, the creation time of a
# committed row of a type that does not count, and keeps the persona's start date.
_MEMORY_KINDS = frozenset({"commit", "assigned", "showing"})
_EVENT_KINDS = _MEMORY_KINDS | {"change", "start"}


class InvalidEventError(ValueError):
    """An event the engine will not fold; the message says why (it is written to the log as it is)."""


def _event_number(raw: Mapping, key: str, default: float) -> float:
    if key not in raw:
        return default
    try:
        return _finite(raw[key], key)
    except ValueError as exc:
        raise InvalidEventError(str(exc)) from None


def event_from_json(raw: object) -> Event:
    """The `Event` of one queue or change-log record, or `InvalidEventError` saying why not.

    Invalid (A3-L3 3.2): not an object; an unknown `kind`; a missing `at`, or an `at` that `lenient_utc` rejects or that
    `day_of` cannot turn into a local day (such an event would fail the drain on every call, and a failure that comes from an
    event's own content must not repeat for ever, so the drop is the remedy); a commit, assigned or showing with no
    `memory_id` (a non-empty text); a commit or assigned with no emotion values; an emotion name that is not text or an amount
    that is not a finite number (a bool is not a number); a `modifier` or `climax`, when present, that is not a finite
    number; a `source`, when present, that is not text. A missing `modifier` is 1.0 and a missing `climax` is 0.0. The
    `modifier` of a change is not read (a change has weight 1, no repeat factor).
    """
    if not isinstance(raw, dict):
        raise InvalidEventError(f"not an object ({type(raw).__name__})")
    kind = raw.get("kind")
    if not isinstance(kind, str) or kind not in _EVENT_KINDS:
        raise InvalidEventError(f"unknown kind {kind!r}")
    if "at" not in raw:
        raise InvalidEventError("missing at")
    at_text = raw["at"]
    at = lenient_utc(at_text) if isinstance(at_text, str) else None
    if at is None:
        raise InvalidEventError(f"at is not a time: {at_text!r}")
    try:
        day_of(at)
    except (OverflowError, OSError, ValueError) as exc:
        raise InvalidEventError(f"at cannot be turned into a local day: {at_text!r}") from exc
    if kind == "start":
        return Event(kind="start", at=at)
    memory_id: str | None = None
    if kind in _MEMORY_KINDS:
        memory_id = raw.get("memory_id")
        if not isinstance(memory_id, str) or not memory_id:
            raise InvalidEventError(f"{kind} with no usable memory_id: {memory_id!r}")
    emotions_raw = raw.get("emotions", {})
    if not isinstance(emotions_raw, dict):
        raise InvalidEventError(f"emotions must be an object, found {type(emotions_raw).__name__}")
    emotions: dict[str, float] = {}
    for name, amount in emotions_raw.items():
        try:
            emotions[name] = _finite(amount, f"amount of {name}")
        except ValueError as exc:
            raise InvalidEventError(str(exc)) from None
    if kind in ("commit", "assigned") and not emotions:
        raise InvalidEventError(f"{kind} with no emotion values")
    if kind == "change":
        source = raw.get("source")
        if source is not None and not isinstance(source, str):
            raise InvalidEventError(f"source must be text, found {type(source).__name__}")
        return Event(kind="change", at=at, emotions=emotions, source=source)
    modifier = _event_number(raw, "modifier", 1.0)
    climax = _event_number(raw, "climax", 0.0) if kind == "commit" else 0.0
    return Event(kind=kind, at=at, memory_id=memory_id, emotions=emotions, modifier=modifier, climax=climax)


def event_to_json(event: Event) -> dict:
    """The record of an event (A3-L3 3.2): commit, assigned and showing carry a memory id, emotions and a modifier (a commit
    also its climax); a change carries its source and signed amounts and no modifier; a start-only event carries only `at`."""
    if event.kind not in _EVENT_KINDS:
        raise ValueError(f"unknown event kind {event.kind!r}")
    out: dict[str, Any] = {"kind": event.kind, "at": _iso(event.at, "at")}
    if event.kind == "start":
        return out
    if event.kind in _MEMORY_KINDS:
        out["memory_id"] = event.memory_id
        out["emotions"] = dict(event.emotions)
        out["modifier"] = event.modifier
        if event.kind == "commit":
            out["climax"] = event.climax
        return out
    if event.source is not None:
        out["source"] = event.source
    out["emotions"] = dict(event.emotions)
    return out


# ================================================================================================
# The queue
# ================================================================================================


def _read_raw_queue(persona_dir: Path) -> list:
    """The queue file's list as it is, unchecked. A missing file is an empty queue.

    Accepted edge E3 (A3 E6.18; owner 2026-10-21 15:02, ledger:337 "Agreed with your extension of the logic to the full
    rebuild"): a queue file that does not parse, is empty, or is not a list is DISCARDED with an ordinary error log line and
    NO rebuild follows. What a discard loses is at most one turn's worth of events (the accepted loss of R12); a rebuild would
    trade the long average's history older than 30 days for them. The read itself writes nothing: the next append replaces
    the file and the drain removes it.

    A fault in the READING (permissions, a sharing violation) is not a discard: it raises, the queue file is left alone, and
    the caller's failure handling keeps the queue for the next drain (E6.10).
    """
    path = queue_path(persona_dir)
    try:
        return read_json_result(path, default=list, recover=Recovery.RAISE, expect=list).data
    except JsonFileError as exc:
        if exc.kind == "io":
            raise
        logger.error("emotion queue %s discarded (%s: %s); no rebuild follows", path, exc.kind, exc.detail)
        return []


def read_queue(persona_dir: Path) -> list[Event]:
    """The valid events of the queue, in file order. An invalid event is dropped with an error log line saying why."""
    path = queue_path(persona_dir)
    events: list[Event] = []
    for index, raw in enumerate(_read_raw_queue(persona_dir)):
        try:
            events.append(event_from_json(raw))
        except InvalidEventError as exc:
            logger.error("emotion queue %s: event %d dropped (%s)", path, index, exc)
    return events


def _write_raw_queue(persona_dir: Path, raw: list) -> None:
    path = queue_path(persona_dir)
    if not raw:
        path.unlink(missing_ok=True)
        return
    write_json(path, raw, indent=None, fsync=False)  # no backups: a drained queue must not come back from one


def append_event(persona_dir: Path, event: Event) -> None:
    """Add an event to the end of the queue (the caller holds the lock). The events already there are kept as they are."""
    raw = _read_raw_queue(persona_dir)
    raw.append(event_to_json(event))
    _write_raw_queue(persona_dir, raw)


def write_queue(persona_dir: Path, events: Sequence[Event]) -> None:
    """Replace the queue with `events`; no events removes the file."""
    _write_raw_queue(persona_dir, [event_to_json(event) for event in events])


def clear_queue(persona_dir: Path) -> None:
    """Remove the queue file (a missing one is fine)."""
    queue_path(persona_dir).unlink(missing_ok=True)


# ================================================================================================
# The change log
# ================================================================================================


def _name_to_day(name: str) -> str | None:
    """The day of a change-log file name ``YYYY-MM-DD.jsonl``, or None for any other name."""
    stem, dot, suffix = name.rpartition(".")
    if not dot or suffix != "jsonl":
        return None
    try:
        return _day(stem, "file name")
    except ValueError:
        return None


def append_change(persona_dir: Path, event: Event) -> bool:
    """Append a change event as one line to the file of its local day (the caller holds the lock). True if it was written.

    Best effort, by design: a failure (a full disk, a value that cannot be encoded, a time that cannot be dated) gets an
    ordinary error log line and False, and never stops the caller's queue append; the change still folds through the queue.
    Only a `change` event belongs here (anything else is a ValueError).
    """
    if event.kind != "change":
        raise ValueError(f"the change log holds change events, not {event.kind!r}")
    try:
        path = changes_dir(persona_dir) / f"{day_of(event.at)}.jsonl"
        written = append_jsonl(path, event_to_json(event), lock="held", on_error="log")
    except (ValueError, TypeError, OverflowError, OSError) as exc:
        logger.error("emotion change log: a change at %s was not recorded (%s: %s)", event.at, type(exc).__name__, exc)
        return False
    return written is not None


def list_change_files(persona_dir: Path) -> list[str]:
    """The names (only) of the change-log files, oldest first: regular files named ``YYYY-MM-DD.jsonl`` for a real day.

    A missing folder is empty. A folder that cannot be listed is logged and treated as missing.
    """
    folder = changes_dir(persona_dir)
    try:
        entries = list(folder.iterdir())
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.error("emotion change log folder %s cannot be listed (%s); treated as missing", folder, exc)
        return []
    names: list[str] = []
    for entry in entries:
        if _name_to_day(entry.name) is None:
            continue
        try:
            if entry.is_file():
                names.append(entry.name)
        except OSError:
            continue
    return sorted(names)


def prune_changes(persona_dir: Path, anchor_day: str | None) -> list[str]:
    """Delete whole change-log files whose day is before the label 30 calendar days before `anchor_day`; return their names.

    `anchor_day` is the active day as it stood BEFORE the drain's own day closes (the caller reads it at the start of the
    drain), so a system clock set far forward cannot delete the folder in the drain that jumps the active day. Nothing is
    deleted when it is None (unset). The length is ``dev_constants.EMOTION_LONG_HALF_LIFE_D``. A file on or after the
    boundary stays. Only regular files named ``YYYY-MM-DD.jsonl`` are ever touched. A delete that fails is logged and the
    rest go on; the file is tried again at the next day close. The caller holds the lock.
    """
    if anchor_day is None:
        return []
    try:
        boundary = (date.fromisoformat(_day(anchor_day, "anchor day")) - timedelta(days=dev_constants.EMOTION_LONG_HALF_LIFE_D)).isoformat()
    except (ValueError, OverflowError) as exc:
        logger.error("emotion change log: no files pruned, the active day %r is unusable (%s)", anchor_day, exc)
        return []
    deleted: list[str] = []
    for name in list_change_files(persona_dir):
        day = _name_to_day(name)
        if day is None or day >= boundary:
            continue
        try:
            (changes_dir(persona_dir) / name).unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.error("emotion change log: %s could not be deleted (%s); it is tried again at the next day close", name, exc)
            continue
        deleted.append(name)
    return deleted


def read_changes(persona_dir: Path, since: datetime) -> list[Event]:
    """The change events dated at or after `since`, for the replay of a rebuild or a catch-up.

    Reads the files in date order starting at the day BEFORE ``day_of(since)`` (files are named by the local day at write
    time, so a time-zone change could put a first in-window line in an earlier-named file) and keeps the lines dated at or
    after `since`; the caller applies the end of its window and folds in time order. The log is a crash-recovery record and
    its absence or damage is normal: a missing file anywhere in the series, a missing folder, a file not named
    ``YYYY-MM-DD.jsonl``, an unlistable folder (logged), a torn or unparseable line and a file that cannot be read in full
    (what was read is used) add nothing and are not errors. A line that fails the event check, or is not a change, is dropped
    with an error log line. The reader repairs nothing.
    """
    start = (date.fromisoformat(day_of(since)) - timedelta(days=1)).isoformat()
    folder = changes_dir(persona_dir)
    events: list[Event] = []
    for name in list_change_files(persona_dir):
        day = _name_to_day(name)
        if day is None or day < start:
            continue
        path = folder / name
        try:
            for record in iter_jsonl(path):
                try:
                    event = event_from_json(record)
                    if event.kind != "change":
                        raise InvalidEventError(f"a {event.kind} event is not a change")
                except InvalidEventError as exc:
                    logger.error("emotion change log %s: a line was dropped (%s)", path, exc)
                    continue
                if event.at >= since:
                    events.append(event)
        except OSError as exc:
            logger.error("emotion change log %s could not be read (%s); what was read is used", path, exc)
    return events


# ================================================================================================
# The switch
# ================================================================================================


def write_switch(persona_dir: Path, switch_over_at: datetime) -> bool:
    """Write ``{"switch_over_at": ...}`` once (the caller holds the lock). True if written, False if the switch already exists.

    ``write_json(..., backups=3, if_missing=True)``: an existing switch is never replaced, so the persona's switch-over time
    never moves. Its existence is what makes a persona live.
    """
    return write_json(
        switch_path(persona_dir),
        {"switch_over_at": _iso(switch_over_at, "switch_over_at")},
        backups=3,
        if_missing=True,
    )


def switch_exists(persona_dir: Path) -> bool:
    """True when the switch file exists. A stat error other than "no such file" is logged and answers False."""
    path = switch_path(persona_dir)
    try:
        path.stat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        warn_throttled(
            logger,
            f"engine_files:switch:{path}:{exc}",
            "emotion switch %s cannot be inspected (%s); the engine is treated as not live for now",
            path,
            exc,
        )
        return False
    return True


def _switch_time(data: object) -> None:
    if not isinstance(data, dict):
        raise ValueError("the switch must be an object")
    _time(data.get("switch_over_at"), "switch_over_at")


def read_switch(persona_dir: Path) -> datetime | None:
    """The switch-over time, or None when there is no switch or no copy of it holds a usable time. Read-only walk."""
    walk = read_json_walk(switch_path(persona_dir), expect=dict, validate=_switch_time)
    return None if walk.problem is not None else lenient_utc(walk.data["switch_over_at"])
