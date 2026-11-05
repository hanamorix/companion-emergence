"""The files the emotion engine owns, and the one place that knows their paths and their lock (Engine 0, #575; Engine 2, #577).

Everything lives under ``<persona_dir>/state/`` (``paths.state_dir``); no code of ours creates that folder, because
``write_json``, ``write_jsonl``, ``append_jsonl`` and ``file_lock`` each create a missing parent.

* ``emotion_state.json`` (+ ``.bak1``-``.bak3``): the averages, the weights and a 30-day snapshot per emotion. Written with
  ``write_json(backups=3, indent=None)``. Read through ``json_io.read_json_walk``, the read-only walk with the validator
  below: a read never heals, quarantines, rotates or rewrites anything, and says which copy it loaded. ``read_state`` reads
  fresh objects (what a drain mutates); ``StateCache`` keeps the parsed state for readers by a stat key.
* ``emotion_pending.jsonl``: the queue of events waiting to be folded, one JSON event per line. Append-only: an append writes
  only its own line, so it costs the same at any depth (a whole-file rewrite per append was quadratic). No backups (a drained
  queue must not come back from one). It is read by position: ``read_queue(persona_dir, from_line=N)`` returns the lines after
  line N and the line count, and the position is kept in the state (``EngineState.queue_line``). The drain empties the file
  after it has folded the events (``empty_queue``); the engine never deletes it. A missing file is an empty queue; a bad line
  is dropped on its own with an error log line (``read_queue``); a last line with no newline yet is not a line and is read
  whole once it is complete.
* ``emotion_changes/YYYY-MM-DD.jsonl``: the change log, one file a day, a crash-recovery record of chat-derived changes.
  Whole files are deleted once they are older than the retention boundary.
* ``emotion_switch.json``: ``{"switch_over_at": ...}`` and nothing else, written once; its existence means the engine is live.
* ``emotion_vocabulary.json`` (the persona's own file, owned by the persona loader and the growth code): only READ here, for
  the half-life cache (``VocabularyHalfLives``, behind ``EmotionEngine.half_life``).

The lock. ONE lock guards the four files: ``state_lock(persona_dir)``, a non-re-entrant ``file_lock`` of the state file that
waits at most ``dev_constants.EMOTION_LOCK_WAIT_S`` (``file_lock.locked_within``) and raises ``LockTimeoutError`` when the wait
runs out. Every acquisition of it passes the wait; what a caller does on the timeout is that caller's. No other function in
this module takes it; every caller holds it (a hook around its appends, a drain around its whole run, the rebuild around
everything) and no code path may take it twice. Lock-free by design: ``read_state``, ``StateCache``, ``switch_state``, ``switch_exists``,
``read_switch``, ``read_queue``, ``read_changes`` and ``list_change_files`` (a reader never waits for a writer; the state and
switch files are replaced atomically, the queue is appended to and emptied atomically, the change log is appended to and
deleted whole; a queue reader that is not the drain, which holds the lock, can meet a line being appended and does not count
it until it is complete). Expecting the caller's lock: ``save_state``, ``append_event``, ``empty_queue``, ``append_change``,
``prune_changes`` and ``write_switch``.

Times are UTC ISO strings written with ``iso_utc`` (``Z``) and read with ``lenient_utc``; day labels are local ``YYYY-MM-DD``
from ``engine_utils.day_of``. All constants (file and folder names, the retention length, the value cap) are read as
``dev_constants.NAME`` at call time.

The half-life cache (Engine 0). The persona's own ``emotion_vocabulary.json`` is read, never written, never repaired and
never quarantined (``Recovery.DEFAULT``: bad content takes the default). It is re-read when its stat key,
``file_ops.stat_path(path).key`` (``(inode, mtime_ns, size)``), changes. The key is taken BEFORE the read, so a file replaced between the stat and the read
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
import reprlib
import threading
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

from brain import dev_constants, paths
from brain.emotion.engine_utils import (
    _MODIFIER_FLOOR,  # with _MODIFIER_SPAN: the range `source_modifier` makes, which a stored modifier is held to
    _MODIFIER_SPAN,
    Average,
    EmotionState,
    EngineState,
    Event,
    RepeatEntry,
    RunRecord,
    SnapshotDay,
    _hold,  # the one hold-to-0-10 rule, shared with the fold (same package, no second copy)
    day_of,
)
from brain.emotion.vocabulary import canonical_name
from brain.health.json_io import (
    Recovery,
    append_jsonl,
    iter_jsonl_raw,
    read_json_result,
    read_json_walk,
    read_jsonl_from,
    write_json,
    write_jsonl,
)
from brain.utils import file_lock, file_ops
from brain.utils.log_throttle import count_throttled, warn_throttled
from brain.utils.time import iso_utc, lenient_utc

logger = logging.getLogger(__name__)

# The state file and the switch keep three backups; `json_io.read_json_walk` reads exactly `.bak1` to `.bak3`.
_BACKUPS = 3

_VOCABULARY_FILENAME = "emotion_vocabulary.json"


def vocabulary_path(persona_dir: Path) -> Path:
    """The persona's vocabulary file (``emotion_vocabulary.json``): the one place its name is joined to a persona folder.

    The half-life cache reads it through this, and the engine's `read()` passes it to the persona loader's refresh.
    """
    return Path(persona_dir) / _VOCABULARY_FILENAME

_StatKey = tuple[int, int, int]


class _Entry(NamedTuple):
    name: str
    days: float | None  # None: stored as "never fades"; the lookup goes on to the next step


class _Fault(NamedTuple):
    name: str | None  # None: the entry has no name to report
    field: str
    why: str


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
        self._path = vocabulary_path(persona_dir)
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
            stat = file_ops.stat_path(self._path)
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
        key = None if stat is None else stat.key
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


def state_lock(persona_dir: Path) -> AbstractContextManager[None]:
    """The one lock of the engine's four files: the lock of the state file (sidecar ``emotion_state.json.lock``), bounded.

    ``file_lock.locked_within`` of the state file path (the lock adds the sidecar name itself), waiting at most
    ``dev_constants.EMOTION_LOCK_WAIT_S``, read here at call time. When the wait runs out it raises ``LockTimeoutError`` and
    the ``with`` body never runs. The lock is NOT re-entrant (taking it again inside the body times out). The hooks, the drain
    and the rebuild take it through this helper, once each; nothing else in this module takes it (see the module docstring).

    ``LockTimeoutError`` is an ``OSError`` (a ``TimeoutError``): a caller that wraps the ``with`` in ``except OSError`` would
    swallow it, so it catches ``LockTimeoutError`` first. What a caller does on the timeout is that caller's.
    """
    return file_lock.locked_within(state_path(persona_dir), dev_constants.EMOTION_LOCK_WAIT_S)


def peer_trust(persona_dir: Path, peer_id: str) -> float:
    """How far this persona trusts one peer (0 to 1): the engine's one door to the peer-trust score.

    Engine-internal stand-in that answers 0.0 for every peer and does no I/O, so until Engine 6 (#584) every peer memory
    weighs the modifier floor, 0.1. Engine 6 replaces this body with a call to the kindled-link accessor
    ``peer_trust(persona_dir, peer_id)`` (not built here, not ours). Its expected shape: open
    ``KindledLinkStore(kindled_db_path(persona_dir), integrity_check=False)``, read
    ``get_relationship_state(...).trust_score``, and give 0.0 for no database or no relationship row.

    Heads-up for that wiring: the engine's `modifier_of` (which calls this) runs INSIDE the catch-up, which holds the engine
    lock (``engine_rebuild.catch_up``), once per peer row of the replay. A kindled-link read there is a database read under
    the engine lock.
    """
    return 0.0


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
        # A start later than the day is refused by the next two checks: its last with-direction day would be after the day
        # or before the start.
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


_VALUE_REPR = reprlib.Repr()
_VALUE_REPR.maxstring = _VALUE_REPR.maxother = 40
_VALUE_REPR.maxlevel = 1
_VALUE_REPR.maxlist = _VALUE_REPR.maxdict = _VALUE_REPR.maxtuple = _VALUE_REPR.maxset = 3


def _queue_line_from(data: Mapping, label: str) -> int:
    """The stored queue position: a whole number of 0 or more, else 0 with a log line (P12b: a value outside its range is held to it).

    A missing key reads as 0 too (a state written before the position existed). `validate_state` never refuses a state for this
    key: the worst a damaged position does is make the next drain fold the queue from its start. The log line is throttled per
    label and kind of fault; its key never holds the value, and its text carries only a short, capped form of it.
    """
    if "queue_line" not in data:
        found, shown = "missing", "missing"
    else:
        value = data["queue_line"]
        if type(value) is int and value >= 0:
            return value
        found = type(value).__name__
        shown = _VALUE_REPR.repr(value)
    count_throttled(
        logger,
        f"engine_files:queue_line:{label}:{found}",
        logging.ERROR,
        "emotion state %s: queue_line is %s, read as 0 (it must be a whole number, 0 or more); the next drain folds the queue from its start",
        label,
        shown,
    )
    return 0


def state_from_json(data: dict, *, label: str = "state") -> EngineState:
    """The `EngineState` of a state that passed `validate_state`.

    Averages (the state's and the snapshot's) are held to 0-10. A bad snapshot day is dropped as a missing day and a bad
    repeat record is dropped, each with an error log line naming the emotion or the memory; the state stays valid. A missing
    or unusable `queue_line` reads as 0, with a log line. `label` names the file in those lines.
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
        queue_line=_queue_line_from(data, label),
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
        "queue_line": state.queue_line,
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
    write_json(state_path(persona_dir), state_to_json(state), backups=_BACKUPS, indent=None)


class LoadedState(NamedTuple):
    """One load of the state file, immutable as a whole (the cache replaces it with one assignment).

    `state` is None when no copy was usable (`problem` says why). `source` is the copy it came from. `key` is the stat key
    ``(inode, mtime_ns, size)`` of the bytes parsed, the walk's ``fstat`` signature as returned (the triple that
    ``file_ops.stat_path(path).key`` gives for the same file), None when there is no state. `shifts` is what `EmotionEngine.read`
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
    return LoadedState(walk.signature, state, walk.source, None)


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
                stat = file_ops.stat_path(state_path(self._persona_dir))
            except OSError:
                stat = None  # cannot tell: read again
            if stat is not None and stat.key == loaded.key:
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
# committed row of a type that does not count, and keeps the persona's start date. It is a memory's own date event: like a
# commit and an assigned event it is APPLIED directly when the memory is written (`engine.EmotionEngine._apply_direct`), and
# only a showing and a change are appended to the queue. The check and the JSON form are shared by both routes.
_MEMORY_KINDS = frozenset({"commit", "assigned", "showing"})
_EVENT_KINDS = _MEMORY_KINDS | {"change", "start"}
# The range of a modifier: what `engine_utils.source_modifier` can make (1.0 for the user and her own, a peer or research memory
# from the floor up to floor + span). A stored modifier outside it is held to it (P12b); one far outside would otherwise fold
# to an infinite average and make every save of the state fail.
_MODIFIER_LOW = _MODIFIER_FLOOR
_MODIFIER_HIGH = _MODIFIER_FLOOR + _MODIFIER_SPAN


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
    number; a `source`, when present, that is not text. A missing `climax` is 0.0. A missing `modifier` is 1.0; a finite one
    outside the range `source_modifier` can make (0.1 to 1.0) is not invalid: it is held to the range, with a warning line
    (P12b). The `modifier` of a change is not read (a change has weight 1, no repeat factor).
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
    held = min(max(modifier, _MODIFIER_LOW), _MODIFIER_HIGH)
    if held != modifier:
        logger.warning(
            "emotion event %s %s: modifier %r is outside %s to %s, held to %s",
            kind,
            memory_id,
            modifier,
            _MODIFIER_LOW,
            _MODIFIER_HIGH,
            held,
        )
        modifier = held
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


class QueueRead(NamedTuple):
    """What `read_queue` found: the valid events after the position, and the position to keep for next time.

    `end_line` is the number of complete lines in the queue file (bad lines and blank lines count; a last line with no
    newline yet does not), which is the `from_line` of the next read once these events are folded.
    """

    events: list[Event]
    end_line: int


def read_queue(persona_dir: Path, *, from_line: int) -> QueueRead:
    """The valid events on the queue lines after line `from_line`, in file order, and the line count (`QueueRead`).

    `from_line` is the position the engine saved with its state (`EngineState.queue_line`): the lines up to it are already
    folded. A missing file is `([], 0)`. If `from_line` is larger than the file's line count, the queue was emptied since the
    position was saved, so it is read again from line 0 (`end_line` is the file's line count either way). A negative
    `from_line` is not a position: it is read as 0, with a log line (P12b).

    Accepted edge E3 (A3 E6.18, applied per line; owner 2026-10-21 15:02, ledger:337 "Agreed with your extension of the logic
    to the full rebuild"): a line that is not valid JSON (a line of zero bytes left by an unclean machine stop, anything else)
    or that fails the event check is DROPPED with its own ordinary error log line naming the file, the line number and the
    reason, and the other lines are read. NO rebuild follows. What a drop loses is at most one turn's worth of events (the
    accepted loss of R12); a rebuild would trade the long average's history older than 30 days for them. A line damaged in the
    middle takes anything glued to it along: nothing is salvaged. A dropped line still counts as a line, and so does a blank
    one. A last line with no newline yet is a torn write, or an append in flight: it is not returned, not counted and not
    logged, and is read whole once it is complete (if a crash left it torn, the next append starts a new line, and it is then
    a complete, bad line). The read itself writes nothing: the bad lines stay in the file (and are logged again by a read from
    before them) until the drain empties it.

    A fault in the READING (permissions, a sharing violation, a directory in the file's place) is not a drop: it raises, the
    queue file is left alone, and the caller's failure handling keeps the queue for the next drain (E6.10).

    The file is read by ``json_io.read_jsonl_from`` (one bytes read, the one line classifier); this function adds the event
    check and the log lines, which name the physical line number and the reason and echo no content.
    """
    path = queue_path(persona_dir)
    if from_line < 0:
        count_throttled(
            logger,
            f"engine_files:queue_from_line:{path}",
            logging.ERROR,
            "emotion queue %s: the position %d is below 0, read as 0 (the queue is read from its start)",
            path,
            from_line,
        )
        from_line = 0
    result = read_jsonl_from(path, from_line)
    if from_line > result.end_line:
        result = read_jsonl_from(path, 0)
    events: list[Event] = []
    for record in result.records:
        if record.reason is not None or record.data is None:
            logger.error("emotion queue %s: line %d dropped (%s)", path, record.line, record.reason)
            continue
        try:
            events.append(event_from_json(record.data))
        except InvalidEventError as exc:
            logger.error("emotion queue %s: line %d dropped (%s)", path, record.line, exc)
    return QueueRead(events, result.end_line)


def append_event(persona_dir: Path, event: Event) -> bool:
    """Append an event as one line at the end of the queue (the caller holds the lock). True if it was written.

    Earlier lines are never read or rewritten, so an append costs the same at any depth of the queue. Best effort, by design
    (E6.11): a failure (a full disk, a folder that is a file, a value that cannot be encoded) is logged and gives False, and
    never raises into the caller. An OSError is logged by ``append_jsonl`` (a throttled warning); a value that cannot be
    encoded gets an error line here. No fsync, no backups.

    R12 (B04.2): an entry lost to a failed append, or damaged by an unclean machine stop, is an accepted minor edge. It is not
    recovered; a damaged line is dropped on read (once it is a complete line), and takes any record glued to it along.
    """
    return _append_line(lambda: queue_path(persona_dir), event, "queue")


def empty_queue(persona_dir: Path) -> bool:
    """Empty the queue file, atomically: True once it is empty, False (with a throttled log line) when it could not be. Never raises.

    ``write_jsonl(path, [], fsync=False)`` replaces the file with an empty one; the engine never deletes it. A failure leaves
    the queue exactly as it was. The caller holds the lock and has folded every line it read; the position it holds
    (``EngineState.queue_line``) is the caller's to reset.
    """
    path = queue_path(persona_dir)
    try:
        write_jsonl(path, [], fsync=False)
    except OSError as exc:
        warn_throttled(
            logger,
            f"engine_files:empty_queue:{path}:{exc}",
            "emotion queue %s could not be emptied (%s); the queue is kept as it is",
            path,
            exc,
        )
        return False
    return True


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


def _append_line(path_of: Callable[[], Path], event: Event, what: str) -> bool:
    """Append `event` as one line to the file `path_of()` names, never raising: the shared failure rule of the queue and the log.

    The caller holds the lock (``lock="held"``). A failure of the append itself is logged by ``append_jsonl``; a value that
    cannot be encoded, or a time that cannot be dated (the path is built inside the guard), gets an ordinary error line here.
    """
    try:
        written = append_jsonl(path_of(), event_to_json(event), lock="held", on_error="log")
    except (ValueError, TypeError, OverflowError, OSError) as exc:
        logger.error("emotion %s: an event at %s was not recorded (%s: %s)", what, event.at, type(exc).__name__, exc)
        return False
    return written is not None


def append_change(persona_dir: Path, event: Event) -> bool:
    """Append a change event as one line to the file of its local day (the caller holds the lock). True if it was written.

    Best effort, by design: a failure (a full disk, a value that cannot be encoded, a time that cannot be dated) gets an
    ordinary error log line and False, and never stops the caller's queue append; the change still folds through the queue.
    Only a `change` event belongs here (anything else is a ValueError).
    """
    if event.kind != "change":
        raise ValueError(f"the change log holds change events, not {event.kind!r}")
    return _append_line(lambda: changes_dir(persona_dir) / f"{day_of(event.at)}.jsonl", event, "change log")


def list_change_files(persona_dir: Path) -> list[str]:
    """The names (only) of the change-log files, oldest first: regular files named ``YYYY-MM-DD.jsonl`` for a real day.

    The folder is listed with ``file_ops.list_dir``; a name that matches the pattern is kept only when
    ``file_ops.stat_path`` says it is a regular file, so a folder named like a day is never listed (and never deleted). An
    entry that vanished (no stat) or whose stat fails is skipped. A missing folder is empty. A folder that cannot be listed is
    logged and treated as missing.
    """
    folder = changes_dir(persona_dir)
    try:
        entries = file_ops.list_dir(folder)
    except OSError as exc:
        logger.error("emotion change log folder %s cannot be listed (%s); treated as missing", folder, exc)
        return []
    names: list[str] = []
    for name in entries:
        if _name_to_day(name) is None:
            continue
        try:
            stat = file_ops.stat_path(folder / name)
        except OSError:
            continue
        if stat is not None and stat.is_file:
            names.append(name)
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
            file_ops.delete_file(changes_dir(persona_dir) / name)
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
    ``YYYY-MM-DD.jsonl``, an unlistable folder (logged), a torn or unparseable line (including one the JSON parser cannot
    load, such as an integer of more than 4300 digits, skipped with a warning naming the file and line) and a file that
    cannot be read in full (what was read is used) add nothing and are not errors. A line that fails the event check, or is not a change, is dropped
    with an error log line. The reader repairs nothing.
    """
    when = lenient_utc(since)  # a naive time means UTC, as everywhere in the repo
    if when is None:
        raise ValueError(f"since is not a usable time: {since!r}")
    since = when
    start = (date.fromisoformat(day_of(since)) - timedelta(days=1)).isoformat()
    folder = changes_dir(persona_dir)
    events: list[Event] = []
    for name in list_change_files(persona_dir):
        day = _name_to_day(name)
        if day is None or day < start:
            continue
        path = folder / name
        for _text, record in iter_jsonl_raw(path):  # a line the parser cannot load is skipped; an I/O failure ends the file there
            try:
                kind = record.get("kind") if isinstance(record, dict) else None
                if kind in _MEMORY_KINDS or kind == "start":  # before the check, so a held-modifier line is not logged for it
                    raise InvalidEventError(f"a {kind} event is not a change")
                event = event_from_json(record)
                if event.kind != "change":
                    raise InvalidEventError(f"a {event.kind} event is not a change")
            except InvalidEventError as exc:
                logger.error("emotion change log %s: a line was dropped (%s)", path, exc)
                continue
            if event.at >= since:
                events.append(event)
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
        backups=_BACKUPS,
        if_missing=True,
    )


def switch_state(persona_dir: Path) -> bool | None:
    """Whether the switch file exists: True when it was seen, False when it is missing, None when the stat failed.

    A stat error other than "no such file" (a permission fault, a share that went away) cannot say either way. It is logged
    (throttled) and answers None, so a caller can tell a file that was SEEN from one that is only unknown. The engine treats an
    unknown as not switched (owner 2026-11-05: "If the system can't tell if its switched, assume its not.") and never remembers it.
    """
    path = switch_path(persona_dir)
    try:
        return file_ops.stat_path(path) is not None
    except OSError as exc:
        warn_throttled(
            logger,
            f"engine_files:switch:{path}:{exc}",
            "emotion switch %s cannot be inspected (%s); the persona is treated as not switched for now, and this is not remembered",
            path,
            exc,
        )
        return None


def switch_exists(persona_dir: Path) -> bool:
    """True when the switch file was seen; False when it is missing, and False when that cannot be told (a stat error).

    An unknown is not switched: the same rule `EmotionEngine.is_live` follows. `switch_state` is the three-way form for a
    caller that must tell a missing file from an unknown one.
    """
    return switch_state(persona_dir) is True


def _switch_time(data: object) -> None:
    if not isinstance(data, dict):
        raise ValueError("the switch must be an object")
    _time(data.get("switch_over_at"), "switch_over_at")


def read_switch(persona_dir: Path) -> datetime | None:
    """The switch-over time, or None when there is no switch or no copy of it holds a usable time. Read-only walk."""
    walk = read_json_walk(switch_path(persona_dir), expect=dict, validate=_switch_time)
    return None if walk.problem is not None else lenient_utc(walk.data["switch_over_at"])
