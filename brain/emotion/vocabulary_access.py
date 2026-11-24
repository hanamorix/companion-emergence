"""How a process asks about, and adds to, her vocabulary (#801 section 5).

The bridge is the one owner: only there do the shared writer's changes (`brain.emotion.vocabulary_file`) run and only there is the
registry the record. Any other process (a `nell` command, the memory server) reaches the owner over the bridge's local web API,
`POST /vocabulary/lookup` and `POST /vocabulary/add`, through `brain.bridge.client.post_to_bridge`: one call per batch of names,
never one per name, and no copy of her vocabulary kept in the process. A name added through the bridge is therefore seen by the
next call of any process without a restart.

`lookup` is the read and `check` is the lookup outcome of section 3 for a raw name: its tidied form matches an entry (that
entry's spelling is used), else the gate accepts it and it is added (the bridge adds it), else it is discarded.

When the bridge cannot be reached, a process reads `emotion_vocabulary.json` read-only for that run
(`persona_loader.load_persona_vocabulary(read_only=True)`): a name whose tidied form is in the file is used under the file's
spelling; any other name is discarded (`check` reports it `missing`, and nothing is written); the process adds no name and runs no
memory heal.

In the owner (`vocabulary.is_bridge_owner()`) both calls run in process, with no HTTP.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from brain import dev_constants
from brain.bridge.client import post_to_bridge
from brain.emotion import vocabulary, vocabulary_file
from brain.emotion.vocabulary import Emotion
from brain.emotion.vocabulary_file import NameResult
from brain.health.json_io import JsonFileError
from brain.utils import file_ops
from brain.utils.log_throttle import warn_throttled

logger = logging.getLogger(__name__)

LOOKUP_ROUTE = "/vocabulary/lookup"
ADD_ROUTE = "/vocabulary/add"

# The entry fields a caller may give a new name; the name itself is the writer's (the tidied form).
ENTRY_FIELDS = ("description", "category", "decay_half_life_days", "intensity_clamp")

ALL: Literal["all"] = "all"


@dataclass(frozen=True)
class Found:
    """What the lookup knows about one entry of her vocabulary."""

    spelling: str
    category: str
    description: str
    half_life_days: float | None


# ------------------------------------------------------------------------------------------------
# In process: the registry
# ------------------------------------------------------------------------------------------------


def _found(emotion: Emotion) -> Found:
    return Found(emotion.name, emotion.category, emotion.description, emotion.decay_half_life_days)


def lookup_local(names: Sequence[str] | Literal["all"]) -> dict[str, Found | None]:
    """The registry's answer: one entry per raw name (None when unknown); every registered entry, keyed by its tidied name, for "all"."""
    if names == ALL:
        return {vocabulary.canonical_name(e.name): _found(e) for e in vocabulary.list_all()}
    out: dict[str, Found | None] = {}
    for raw in names:
        emotion = vocabulary.get(raw)
        out[raw] = None if emotion is None else _found(emotion)
    return out


# The stat key of her file at this process's last read-only load of it, by path: the same bytes are read once, not once per call
# (a run with no bridge asks once per name). A changed file is read again.
_READ_ONLY_LOADED: dict[Path, tuple[int, int, int]] = {}


def load_read_only(persona_dir: Path) -> None:
    """Register what her file holds into this process's registry for this run; a file that cannot be read adds nothing.

    The load every command process makes (spec section 5): no memory heal, no rebuild, no quarantine. Never writes: a damaged file raises (json_io `Recovery.RAISE`) instead of being quarantined and reset, and is reported once.
    """
    from brain.emotion.persona_loader import load_persona_vocabulary

    path = vocabulary_file.vocabulary_path(persona_dir)
    try:
        stat = file_ops.stat_path(path)
    except OSError:
        stat = None  # cannot say whether the file changed: read it (below) and remember nothing
        key = None
    else:
        if stat is None:
            return  # no file, nothing to load
        key = stat.key
    if key is not None and _READ_ONLY_LOADED.get(path) == key:
        return
    try:
        load_persona_vocabulary(path, read_only=True)
    except (JsonFileError, OSError, ValueError) as exc:
        warn_throttled(
            logger,
            f"vocabulary_access:read_only:{path}",
            "vocabulary: no bridge, and her vocabulary file cannot be read (%s: %s); every name outside the framework's own is unknown this run",
            type(exc).__name__,
            exc,
        )
    if key is not None:
        _READ_ONLY_LOADED[path] = key  # the same bytes give the same fault: a bad file is not read again until it changes


# ------------------------------------------------------------------------------------------------
# The wire
# ------------------------------------------------------------------------------------------------


def found_to_wire(raw: str, found: Found | None) -> dict[str, Any]:
    """One lookup answer as the bridge sends it."""
    if found is None:
        return {"name": raw, "unknown": True}
    return {
        "name": raw,
        "spelling": found.spelling,
        "category": found.category,
        "description": found.description,
        "half_life_days": found.half_life_days,
    }


def _found_from_wire(item: Any) -> tuple[str, Found | None]:
    if item.get("unknown") is True:
        return item["name"], None
    half_life = item["half_life_days"]
    return item["name"], Found(
        spelling=item["spelling"],
        category=item["category"],
        description=item["description"],
        half_life_days=None if half_life is None else float(half_life),
    )


def result_to_wire(result: NameResult) -> dict[str, Any]:
    """One add result as the bridge sends it."""
    return {"name": result.raw, "spelling": result.spelling, "outcome": result.outcome, "changed": result.changed}


def _result_from_wire(item: Any) -> NameResult:
    return NameResult(item["name"], item["spelling"], item["outcome"], bool(item["changed"]))


# ------------------------------------------------------------------------------------------------
# The two calls
# ------------------------------------------------------------------------------------------------


def _ask(persona_dir: Path, route: str, body: dict[str, Any]) -> Any:
    """One call to the bridge. A body that is not JSON is reported as an error result like the other failures."""
    try:
        return post_to_bridge(persona_dir, route, body, timeout=dev_constants.VOCABULARY_BRIDGE_TIMEOUT_S)
    except ValueError:
        return {"error": "bridge answer is not JSON"}


def _note_failure(route: str, reply: Any) -> None:
    """Say why the read-only file is used for a call. A bridge that is simply not running is the quiet case; every other failure
    (a refused token, a server error, a timeout, an answer of the wrong shape) is reported once per kind."""
    error = reply.get("error") if isinstance(reply, dict) else None
    if error == "bridge unreachable":
        return
    detail = error if isinstance(error, str) else "an answer that is not a vocabulary reply"
    if route == ADD_ROUTE and error == "bridge timeout":
        detail += "; the bridge may still have added the names"
    warn_throttled(
        logger,
        f"vocabulary_access:bridge:{route}:{detail}",
        "vocabulary: the bridge did not answer %s (%s); her file is read read-only for this run",
        route,
        detail,
    )


def lookup(persona_dir: Path, names: Sequence[str] | Literal["all"]) -> dict[str, Found | None]:
    """The entry of each raw name (None = unknown), or every entry keyed by tidied name for "all". One call for the whole batch.

    The owner answers from its registry; any other process asks the bridge and, when the bridge cannot be reached or answers
    anything but a lookup, reads her file read-only for this run.
    """
    names = ALL if names == ALL else list(names)
    if vocabulary.is_bridge_owner():
        return lookup_local(names)
    if names != ALL and not names:
        return {}
    reply = _ask(persona_dir, LOOKUP_ROUTE, {"names": names})
    try:
        return dict(_found_from_wire(item) for item in reply["entries"])
    except (KeyError, TypeError, ValueError, AttributeError):
        _note_failure(LOOKUP_ROUTE, reply)
    load_read_only(persona_dir)
    return lookup_local(names)


def _fields_valid(fields: Mapping[str, Any]) -> bool:
    """True when every given entry field is what her file's schema takes: text for description and category, a finite number above 0
    (null allowed for the half-life: "never fades") for the half-life and the clamp."""

    def number(value: Any, *, null_ok: bool) -> bool:
        if value is None:
            return null_ok
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(value) and value > 0
        except OverflowError:
            return False

    return (
        all(isinstance(fields[k], str) for k in ("description", "category") if k in fields)
        and ("decay_half_life_days" not in fields or number(fields["decay_half_life_days"], null_ok=True))
        and ("intensity_clamp" not in fields or number(fields["intensity_clamp"], null_ok=False))
    )


def add_local(persona_dir: Path, items: Sequence[tuple[str, Mapping[str, Any] | None]]) -> list[NameResult]:
    """The owner's add: only entry fields are taken from a caller, and an item whose fields are not valid is `refused` (nothing is
    written for it); the rest go through the shared writer in one locked write. One result per item, in order."""
    kept = [(raw, None if fields is None else {k: v for k, v in fields.items() if k in ENTRY_FIELDS}) for raw, fields in items]
    valid_at = [i for i, (_raw, fields) in enumerate(kept) if fields is None or _fields_valid(fields)]
    if len(valid_at) < len(kept):
        warn_throttled(logger, "vocabulary_access:invalid_fields", "vocabulary: an add carried an entry field that is not valid; that name is refused", exc_info=False)
    results = [NameResult(raw, None, "refused") for raw, _fields in kept]
    if valid_at:
        written = vocabulary_file.add(vocabulary_file.vocabulary_path(persona_dir), [kept[i] for i in valid_at])
        for i, result in zip(valid_at, written, strict=True):
            results[i] = result
    return results


def check(
    persona_dir: Path, items: Sequence[tuple[str, Mapping[str, Any] | None]]
) -> list[NameResult]:
    """The section 3 outcome for each `(raw_name, fields_or_None)` item, in order; a name that passes the gate is added when new.

    `used` carries the spelling in her vocabulary; `refused` is a name the gate refused (or an item with an invalid field);
    `cannot_write` is a name the owner could not add; `missing` is a name that is not in her vocabulary and cannot be added from
    here (no bridge), which the caller discards. One call for the whole batch.
    """
    items = [(raw, None if fields is None else dict(fields)) for raw, fields in items]
    if not items:
        return []
    if vocabulary.is_bridge_owner():
        return add_local(persona_dir, items)
    body = {"items": [{"name": raw, "fields": fields} for raw, fields in items]}
    reply = _ask(persona_dir, ADD_ROUTE, body)
    try:
        results = [_result_from_wire(item) for item in reply["results"]]
        if len(results) == len(items):
            return results
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    _note_failure(ADD_ROUTE, reply)
    load_read_only(persona_dir)
    out: list[NameResult] = []
    for raw, _fields in items:
        if vocabulary.gate_name(raw) is None:
            out.append(NameResult(raw, None, "refused"))
            continue
        emotion = vocabulary.get(raw)
        out.append(NameResult(raw, None, "missing") if emotion is None else NameResult(raw, emotion.name, "used"))
    return out


# ------------------------------------------------------------------------------------------------
# The check on a memory's feeling names (spec section 6)
# ------------------------------------------------------------------------------------------------

# (persona dir, tidied name) -> time.monotonic() before which the name is not asked about again, after an add that was `cannot_write`.
_ADD_RETRY_AFTER: dict[tuple[Path, str], float] = {}


def names_used(persona_dir: Path | None, raw_names: Iterable[str]) -> dict[str, str | None]:
    """The spelling to use for each raw feeling name, or None when the name is discarded (spec section 3 outcome, one batch).

    A framework name is answered from the registry with nothing asked and nothing written. With no `persona_dir` (an in-memory
    store, a reader with no engine) only this process's registry answers. Otherwise the owner answers a name its registry holds
    without any file work and adds the rest through the shared writer; any other process makes ONE `check` for all of the
    non-framework names (the bridge, else her file read-only, else the name is `missing` and discarded). A name that could not be
    written (`cannot_write`) is kept tidied, and is not asked about again for `dev_constants.EMOTION_VOCABULARY_ADD_RETRY_S`.
    """
    unique = list(dict.fromkeys(n for n in raw_names if isinstance(n, str)))
    out: dict[str, str | None] = {}
    ask: list[str] = []
    for raw in unique:
        if vocabulary.is_baseline(raw):
            emotion = vocabulary.get(raw)
            out[raw] = None if emotion is None else emotion.name
            continue
        if persona_dir is None:
            emotion = vocabulary.get(raw)
            out[raw] = None if emotion is None else emotion.name
            continue
        if vocabulary.is_bridge_owner() and (emotion := vocabulary.get(raw)) is not None:
            out[raw] = emotion.name
            continue
        tidied = vocabulary.gate_name(raw)
        if tidied is None:
            out[raw] = None
            continue
        if _ADD_RETRY_AFTER.get((persona_dir, tidied), 0.0) > time.monotonic():
            out[raw] = tidied
            continue
        ask.append(raw)
    if ask and persona_dir is not None:
        results = check(persona_dir, [(raw, None) for raw in ask])
        for raw, result in zip(ask, results, strict=True):
            tidied = vocabulary.canonical_name(raw)
            if result.outcome == "used":
                out[raw] = result.spelling
                _ADD_RETRY_AFTER.pop((persona_dir, tidied), None)
            elif result.outcome == "cannot_write":
                out[raw] = tidied
                _ADD_RETRY_AFTER[(persona_dir, tidied)] = time.monotonic() + dev_constants.EMOTION_VOCABULARY_ADD_RETRY_S
            else:
                out[raw] = None
                if result.outcome == "missing":
                    warn_throttled(
                        logger,
                        "vocabulary_access:discarded_missing",
                        "vocabulary: a feeling name is not in her vocabulary and no bridge can add it; it is discarded here (first seen: %r)",
                        raw,
                        exc_info=False,
                    )
    return out


def names_known(persona_dir: Path | None, raw_names: Iterable[str]) -> dict[str, str | None]:
    """The spelling her vocabulary has for each raw name, or None when it holds no such entry: a lookup that never adds a name.

    For a reader of another file's feeling names (a reflex arc's trigger, a growth-log event) whose names say what the file
    refers to, not what is felt now: such a read must not mint a stub entry in her vocabulary for a name she does not have. One
    `lookup` for the batch (the owner answers from its registry, another process asks the bridge, else the read-only file); with
    no `persona_dir` only this process's registry answers. A framework name is answered from the registry.
    """
    unique = list(dict.fromkeys(n for n in raw_names if isinstance(n, str)))
    out: dict[str, str | None] = {}
    ask: list[str] = []
    for raw in unique:
        if persona_dir is None or vocabulary.is_baseline(raw):
            emotion = vocabulary.get(raw)
            out[raw] = None if emotion is None else emotion.name
        else:
            ask.append(raw)
    if ask and persona_dir is not None:
        found = lookup(persona_dir, ask)
        for raw in ask:
            entry = found.get(raw)
            out[raw] = None if entry is None else entry.spelling
    return out


def finite_number(value: Any) -> bool:
    """True for an int or float that is finite (a bool, a string, NaN, inf and an integer too large for a float are not)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _mean_of_numbers(values: list[Any]) -> Any:
    numbers = [v for v in values if finite_number(v)]
    if len(values) == 1 or not numbers:
        return values[0]
    return sum(numbers) / len(numbers)


def check_feelings(persona_dir: Path | None, raw: Mapping[str, Any]) -> dict[str, Any]:
    """The check of spec section 6 on a map of feelings that is about to be WRITTEN: each kept name under the spelling her vocabulary uses.

    A name the check discards is left out; two raw names that land on one spelling carry the mean of their numeric values (a value
    that is not a number is kept as given for a lone name). A key that is not text is dropped with a log line. Returns a new dict.
    """
    texts = {name: value for name, value in raw.items() if isinstance(name, str)}
    if len(texts) < len(raw):
        logger.warning("dropping an emotion whose name is not text")
    spellings = names_used(persona_dir, texts)
    groups: dict[str, list[Any]] = {}
    for name, value in texts.items():
        spelling = spellings.get(name)
        if spelling is not None:
            groups.setdefault(spelling, []).append(value)
    return {spelling: _mean_of_numbers(values) for spelling, values in groups.items()}
