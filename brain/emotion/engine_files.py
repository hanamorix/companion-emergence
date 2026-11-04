"""The emotion engine's reader of the persona's vocabulary file (Engine 0, #575).

In this slice the module holds one thing: the half-life cache behind
`brain.emotion.engine.EmotionEngine.half_life`. The persona's own
`emotion_vocabulary.json` is owned by the persona loader and the growth code; the
engine only reads it, never writes it, never repairs it and never quarantines it
(`Recovery.DEFAULT`: bad content takes the default).

The file is re-read when its stat key, `file_ops.stat_path(path).key`
(`(inode, mtime_ns, size)`), changes. The key is taken BEFORE the read, so a file replaced
between the stat and the read is read again on the next call. A failure that says nothing about the content (a filesystem error on
the read, an error from `stat` other than "no such file") is not cached: the previously
loaded half-lives stay in force and the next call tries again (the same bound as
`brain/tunables.py`, #530). A fault in the content (not JSON, wrong shape, a bad entry)
is cached, because the same bytes give the same fault. A bad entry is reported once per change
of the file; a file that is not JSON or not the right shape is reported by `read_json_result`,
which follows its own throttle (one warning per path and kind per interval).
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any, NamedTuple

from brain.emotion.vocabulary import canonical_name
from brain.health.json_io import Recovery, read_json_result
from brain.utils import file_ops
from brain.utils.log_throttle import warn_throttled

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
