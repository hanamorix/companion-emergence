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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from brain import dev_constants
from brain.bridge.client import post_to_bridge
from brain.emotion import vocabulary, vocabulary_file
from brain.emotion.vocabulary import Emotion
from brain.emotion.vocabulary_file import NameResult
from brain.health.json_io import JsonFileError
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


def _load_read_only(persona_dir: Path) -> None:
    """Register what her file holds into this process's registry for this run; a file that cannot be read adds nothing."""
    from brain.emotion.persona_loader import load_persona_vocabulary

    try:
        load_persona_vocabulary(vocabulary_file.vocabulary_path(persona_dir), read_only=True)
    except (JsonFileError, OSError, ValueError) as exc:
        warn_throttled(
            logger,
            f"vocabulary_access:read_only:{persona_dir}",
            "vocabulary: no bridge, and her vocabulary file cannot be read (%s: %s); every name outside the framework's own is unknown this run",
            type(exc).__name__,
            exc,
        )


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
    try:
        reply = post_to_bridge(persona_dir, LOOKUP_ROUTE, {"names": names}, timeout=dev_constants.VOCABULARY_BRIDGE_TIMEOUT_S)
        return dict(_found_from_wire(item) for item in reply["entries"])
    except (KeyError, TypeError, ValueError, AttributeError):
        pass  # no bridge, or an answer that is not a lookup (a body that is not JSON included): the read-only file
    _load_read_only(persona_dir)
    return lookup_local(names)


def check(
    persona_dir: Path, items: Sequence[tuple[str, Mapping[str, Any] | None]]
) -> list[NameResult]:
    """The section 3 outcome for each `(raw_name, fields_or_None)` item, in order; a name that passes the gate is added when new.

    `used` carries the spelling in her vocabulary; `refused` is a name the gate refused; `cannot_write` is a name the owner could
    not add; `missing` is a name that is not in her vocabulary and cannot be added from here (no bridge), which the caller
    discards. One call for the whole batch.
    """
    items = [(raw, None if fields is None else dict(fields)) for raw, fields in items]
    if not items:
        return []
    if vocabulary.is_bridge_owner():
        return vocabulary_file.add(vocabulary_file.vocabulary_path(persona_dir), items)
    body = {"items": [{"name": raw, "fields": fields} for raw, fields in items]}
    try:
        reply = post_to_bridge(persona_dir, ADD_ROUTE, body, timeout=dev_constants.VOCABULARY_BRIDGE_TIMEOUT_S)
        results = [_result_from_wire(item) for item in reply["results"]]
        if len(results) == len(items):
            return results
    except (KeyError, TypeError, ValueError, AttributeError):
        pass  # no bridge, or an answer that is not an add result: the read-only file
    _load_read_only(persona_dir)
    out: list[NameResult] = []
    for raw, _fields in items:
        if vocabulary.gate_name(raw) is None:
            out.append(NameResult(raw, None, "refused"))
            continue
        emotion = vocabulary.get(raw)
        out.append(NameResult(raw, None, "missing") if emotion is None else NameResult(raw, emotion.name, "used"))
    return out
