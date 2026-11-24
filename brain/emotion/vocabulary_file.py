"""The one writer of her vocabulary file (#741, #801 section 4).

`emotion_vocabulary.json` is changed by this module and by nothing else: adding a name, changing an entry's fields, merging two
entries that tidy to the same name, and removing an entry are `add`, `replace_fields`, `merge` and `remove`. (json_io's own
recovery of a corrupt file, which adds no name, is the only other code that writes it. The rebuild step and the OG migrator, the
two noted exceptions to "only the bridge calls this" (section 9), call this module too.) `load` registers what the file holds
into the registry and writes nothing.

A core feeling (one of the framework's 30, `vocabulary.is_baseline`) is never written to her file (#801 section 8): `add` and
`replace_fields` refuse it, `merge` leaves a core entry where it is (never merged into her file), and `remove_core_entries` takes
out any entry the file holds for one, at every bridge start-up and in the rebuild step.

Each change is ONE read-check-write, built once in `_transact`: the cross-process sidecar lock of the file taken with a bounded
wait (`dev_constants.EMOTION_VOCABULARY_LOCK_WAIT_S`), the file re-read, the change applied to what was just read, a verified
`json_io.write_json` with the shared backup count, then the registry changed under one `threading.Lock` (still inside the file
lock, so registry changes follow file changes in the same order). The file comes first and the registry second, so a write that
fails leaves the registry as it was. An error in the registry step itself (a bug, not a file problem) is raised to the caller
after the lock is released; the file is already written and the next `load` registers what it holds. The lock is not re-entrant, so the public operations
never call each other and `load` takes no lock. Lock order for any later caller: `engine_files.state_lock` is the OUTER lock and
this file's lock the INNER one.

A name is matched by its tidied form (`vocabulary.canonical_name`) and the spelling used is the one already in the file. A new
name passes `vocabulary.gate_name` first and is added under its tidied form.

Each name gets a `NameResult`: `used` (with the spelling used), `refused` (the gate refused the name, or it is a core feeling,
which `add` and `replace_fields` never write), `missing` (no entry carries
the name: `replace_fields`, `remove`) or `cannot_write` (the file could not be read or written, its schema is wrong, or the lock
wait ran out; the cause is logged). `changed` says this call wrote that entry.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from brain import dev_constants
from brain.emotion import vocabulary
from brain.emotion.vocabulary import Emotion
from brain.health.adaptive import DEFAULT_BACKUP_COUNT, compute_treatment
from brain.health.json_io import JsonRead, Recovery, read_json_result, write_json
from brain.utils import file_lock

if TYPE_CHECKING:
    from brain.health.anomaly import BrainAnomaly

logger = logging.getLogger(__name__)

Outcome = Literal["used", "refused", "missing", "cannot_write"]

# One lock for every change to the registry that a file operation makes.
_REGISTRY_LOCK = threading.Lock()


@dataclass(frozen=True)
class NameResult:
    raw: str  # the name as the caller gave it
    spelling: str | None  # the spelling used in the file (None for refused, missing and cannot_write)
    outcome: Outcome
    changed: bool = False  # this call wrote the entry


@dataclass(frozen=True)
class MergeResult:
    pairs: list[tuple[str, str]]  # (dropped entry's name, kept spelling), one per dropped entry
    cannot_write: bool = False


# ------------------------------------------------------------------------------------------------
# The schema, the read and the backup count (each exists once)
# ------------------------------------------------------------------------------------------------


def vocabulary_path(persona_dir: Path) -> Path:
    """Her vocabulary file (``emotion_vocabulary.json``): the one place its name is joined to a persona folder."""
    return Path(persona_dir) / "emotion_vocabulary.json"


def default_vocabulary() -> dict:
    """The content of a file with no entry."""
    return {"version": 1, "emotions": []}


def validate_schema(data: object) -> None:
    """The one check that the file holds an `emotions` list."""
    if not isinstance(data, dict) or not isinstance(data.get("emotions"), list):
        raise ValueError("emotion_vocabulary schema invalid: missing 'emotions' list")


def read(path: Path, *, recover: Recovery = Recovery.HEAL) -> JsonRead:
    """Read her vocabulary file under a recovery policy (see `json_io.read_json_result`); a missing file reads as no entries."""
    return read_json_result(path, default=default_vocabulary, recover=recover, validate=validate_schema)


def backup_count(path: Path) -> int:
    """How many backups a write of her vocabulary file keeps: the adaptive count, else the default when the audit log cannot be read."""
    try:
        return compute_treatment(path.parent, path.name).backup_count
    except (OSError, ValueError):
        return DEFAULT_BACKUP_COUNT


# ------------------------------------------------------------------------------------------------
# The registry side
# ------------------------------------------------------------------------------------------------


def _entry_to_emotion(entry: dict) -> Emotion:
    """Build an Emotion from a file entry. Raises on a missing or invalid field."""
    if not isinstance(entry, dict):
        raise TypeError("entry must be a dict")
    for key in ("name", "description", "category", "decay_half_life_days"):
        if key not in entry:
            raise KeyError(f"missing required field {key!r}")
    return Emotion(
        name=str(entry["name"]),
        description=str(entry["description"]),
        category=str(entry["category"]),  # type: ignore[arg-type]
        decay_half_life_days=(
            None if entry["decay_half_life_days"] is None else float(entry["decay_half_life_days"])
        ),
        intensity_clamp=float(entry.get("intensity_clamp", 10.0)),
    )


def register_entries(data: dict) -> int:
    """Register the entries of a parsed vocabulary dict; the count newly registered.

    An entry already registered under any spelling is skipped (idempotent); an entry the gate refuses (empty once tidied, or too
    long) is skipped with the gate's warning and stays in the file; an entry that fails validation is skipped with a warning.
    """
    registered = 0
    for entry in data.get("emotions", []):
        try:
            emotion = _entry_to_emotion(entry)
        except (KeyError, ValueError, TypeError) as exc:
            entry_name = entry.get("name", "<unnamed>") if isinstance(entry, dict) else "<not a dict>"
            logger.warning("skipping emotion entry %r: %s", entry_name, exc)
            continue
        if vocabulary.gate_name(emotion.name) is None:
            continue  # the gate has logged why
        with _REGISTRY_LOCK:  # the same lock the registry steps of the file operations take: no check-then-register race
            if vocabulary.get(emotion.name) is not None:
                continue
            vocabulary.register(emotion)
        registered += 1
    return registered


def _register_new(entry: dict) -> None:
    """Registry step of an add: register the entry unless its name is registered already."""
    try:
        emotion = _entry_to_emotion(entry)
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("vocabulary: entry %r written but not registered: %s", entry.get("name"), exc)
        return
    if vocabulary.get(emotion.name) is None:
        vocabulary.register(emotion)


def _replace_registered(entry: dict) -> None:
    """Registry step of a change: the registered entry becomes the file's entry; a framework entry is left alone."""
    try:
        emotion = _entry_to_emotion(entry)
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("vocabulary: entry %r written but not re-registered: %s", entry.get("name"), exc)
        return
    if not vocabulary.is_baseline(emotion.name):
        vocabulary.replace_entry(emotion)


# ------------------------------------------------------------------------------------------------
# The one transaction
# ------------------------------------------------------------------------------------------------

# A mutation edits the data it is given and returns (payload, registry_step); a registry_step of None means nothing changed and
# nothing is written.
_Mutation = Callable[[dict], tuple[Any, "Callable[[], None] | None"]]


def _transact(path: Path, mutate: _Mutation, action: str) -> tuple[Any, bool]:
    """Lock, re-read, mutate, write verified, then run the registry step. Returns (payload, ok); ok False = `cannot_write`.

    The backup count is computed before the lock is taken (it scans the audit log). Any OSError or ValueError from the lock, the
    read or the write, a write that did not verify, and a lock wait that ran out are all `cannot_write`; the cause is logged.
    """
    count = backup_count(path)
    registry_error: Exception | None = None
    try:
        with file_lock.locked_within(path, dev_constants.EMOTION_VOCABULARY_LOCK_WAIT_S):
            data = read_json_result(
                path, default=default_vocabulary, recover=Recovery.RAISE, validate=validate_schema
            ).data
            payload, registry_step = mutate(data)
            if registry_step is not None:
                if not write_json(path, data, backups=count, verify=True):
                    raise OSError("the write did not verify (the previous file was restored)")
                # Still inside the file lock, so two writers' registry changes follow their file changes in the same order.
                # A registry error is kept apart from the file errors handled below: the file is written, so it is not a
                # `cannot_write`, and it is raised once the lock is released.
                try:
                    with _REGISTRY_LOCK:
                        registry_step()
                except Exception as exc:  # noqa: BLE001  re-raised below, after the file lock is released
                    registry_error = exc
    except (OSError, ValueError) as exc:
        logger.warning("vocabulary: %s of %s failed (%s: %s); the file and the registry are as they were", action, path.name, type(exc).__name__, exc)
        return None, False
    if registry_error is not None:
        raise registry_error
    return payload, True


def _entries_by_key(emotions: Iterable[Any]) -> dict[str, dict]:
    """The first dict entry of each tidied name (entries that are not dicts, have no string name or tidy to nothing are not indexed)."""
    index: dict[str, dict] = {}
    for entry in emotions:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            key = vocabulary.canonical_name(entry["name"])
            if key and key not in index:
                index[key] = entry
    return index


# ------------------------------------------------------------------------------------------------
# The operations
# ------------------------------------------------------------------------------------------------


def _writable_key(raw: str) -> str | None:
    """The tidied name an `add` or `replace_fields` may write, or None (`refused`): the gate refused it, or it is a core feeling."""
    key = vocabulary.gate_name(raw)
    if key is None or vocabulary.is_baseline(key):
        return None  # a core feeling is never written to her file (#801 section 8): the framework holds its definition
    return key


def add(path: Path, items: Sequence[tuple[str, Mapping[str, Any] | None]]) -> list[NameResult]:
    """Add names to her vocabulary file: one result per `(raw_name, fields_or_None)` item, in order.

    A name the gate refuses, and a core feeling, is `refused`. A name whose tidied form is already in the FILE is `used` under the spelling there and
    nothing is written for it (the registry is not consulted: the file is the record). Any other name is appended under its
    tidied form with the given fields over the defaults (placeholder description, `persona_extension`, the placeholder half-life,
    clamp 10.0) and registered. One locked write for the whole list.
    """
    from brain.health.reconstruct import PLACEHOLDER_DECAY_DAYS, PLACEHOLDER_DESCRIPTION

    gated = [(raw, _writable_key(raw), dict(fields or {})) for raw, fields in items]
    accepted = [(i, raw, key, fields) for i, (raw, key, fields) in enumerate(gated) if key is not None]
    if not accepted:
        return [NameResult(raw, None, "refused") for raw, _key, _fields in gated]

    def mutate(data: dict) -> tuple[dict[int, NameResult], Callable[[], None] | None]:
        emotions = data["emotions"]
        index = _entries_by_key(emotions)
        results: dict[int, NameResult] = {}
        new_entries: list[dict] = []
        for i, raw, key, fields in accepted:
            if key in index:
                results[i] = NameResult(raw, index[key]["name"], "used")
                continue
            entry: dict[str, Any] = {
                "description": PLACEHOLDER_DESCRIPTION,
                "category": "persona_extension",
                "decay_half_life_days": PLACEHOLDER_DECAY_DAYS,
                "intensity_clamp": 10.0,
            }
            entry.update({k: v for k, v in fields.items() if k != "name"})
            entry = {"name": key, **entry}
            emotions.append(entry)
            index[key] = entry
            new_entries.append(entry)
            results[i] = NameResult(raw, key, "used", True)
        if not new_entries:
            return results, None
        return results, lambda: [_register_new(e) for e in new_entries]

    by_index, ok = _transact(path, mutate, "add")
    out: list[NameResult] = []
    for i, (raw, key, _fields) in enumerate(gated):
        if key is None:
            out.append(NameResult(raw, None, "refused"))
        elif not ok:
            out.append(NameResult(raw, None, "cannot_write"))
        else:
            out.append(by_index[i])
    return out


def replace_fields(
    path: Path,
    changes: Mapping[str, Mapping[str, Any]],
    *,
    only_if: Callable[[dict], bool] | None = None,
) -> list[NameResult]:
    """Change fields of existing entries: `changes` maps a raw name to the fields to set; one result per name, one locked write.

    The entry found (by tidied name) is updated in place, so any other key it carries is kept; its name is never changed.
    `only_if(entry)`, when given, is asked about the entry as it is in the file now: False leaves that entry alone (`used`, not
    `changed`), so a caller never overwrites a value another writer set since it looked. A name no entry carries is `missing`; a
    core feeling is `refused` (its definition is the framework's).
    """
    gated = [(raw, _writable_key(raw), dict(fields)) for raw, fields in changes.items()]
    wanted = [(raw, key, fields) for raw, key, fields in gated if key is not None]
    if not wanted:
        return [NameResult(raw, None, "refused") for raw, _key, _fields in gated]

    def mutate(data: dict) -> tuple[dict[str, NameResult], Callable[[], None] | None]:
        index = _entries_by_key(data["emotions"])
        results: dict[str, NameResult] = {}
        touched: list[dict] = []
        for raw, key, fields in wanted:
            entry = index.get(key)
            if entry is None:
                results[raw] = NameResult(raw, None, "missing")
                continue
            if only_if is not None and not only_if(entry):
                results[raw] = NameResult(raw, entry["name"], "used")
                continue
            update = {k: v for k, v in fields.items() if k != "name"}
            if all(k in entry and entry[k] == v for k, v in update.items()):
                results[raw] = NameResult(raw, entry["name"], "used")
                continue
            entry.update(update)
            touched.append(entry)
            results[raw] = NameResult(raw, entry["name"], "used", True)
        if not touched:
            return results, None
        return results, lambda: [_replace_registered(e) for e in touched]

    by_raw, ok = _transact(path, mutate, "replace_fields")
    out: list[NameResult] = []
    for raw, key, _fields in gated:
        if key is None:
            out.append(NameResult(raw, None, "refused"))
        elif not ok:
            out.append(NameResult(raw, None, "cannot_write"))
        else:
            out.append(by_raw[raw])
    return out


def merge_twin_entries(data: dict, only: set[str] | None = None) -> list[tuple[str, str]]:
    """Merge entries whose names tidy to the same key (`canonical_name`) into one (#174, #801 section 3).

    Mutates ``data["emotions"]`` in place. Returns ``(dropped_name, kept_spelling)`` pairs, one per entry dropped; empty when
    nothing was merged, and then ``data`` is untouched (a lone entry keeps its spelling: nothing is renamed). The merged entry
    takes all its fields (every key of that entry, not a mix of twins) from the first twin with a real description, else from the
    first; its name is the spelling of the twin that comes first in the file. It sits where that first twin sat. Entries that are
    not dicts, have no string name, or have a name that tidies to nothing pass through unchanged and in place. So does an entry
    for a core feeling (`vocabulary.is_baseline` only, the check `add` and `replace_fields` use, not the gate: a name over the gate's
    length is merged as before): a core twin is never merged into her file, it stays for the start-up removal (#801 section 8).
    `only`, when given, is a set of tidied names: twins of any other name are left as they are.
    """
    from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION

    emotions = data.get("emotions", [])

    def key_of(entry: Any) -> str:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            return ""
        key = vocabulary.canonical_name(entry["name"])
        if vocabulary.is_baseline(key):
            return ""  # a core feeling is never merged into her file (the same check as `add`'s); the start-up removal takes it out
        return key if only is None or key in only else ""

    groups: dict[str, list[dict]] = {}
    for entry in emotions:
        key = key_of(entry)
        if key:
            groups.setdefault(key, []).append(entry)
    if all(len(twins) == 1 for twins in groups.values()):
        return []

    dropped: list[tuple[str, str]] = []
    merged: list = []
    done: set[str] = set()
    for entry in emotions:
        key = key_of(entry)
        if not key:
            merged.append(entry)
            continue
        if key in done:
            continue
        done.add(key)
        twins = groups[key]
        first_spelling = twins[0]["name"]
        donor = next((t for t in twins if t.get("description") != PLACEHOLDER_DESCRIPTION), twins[0])
        dropped.extend((t["name"], first_spelling) for t in twins if t is not donor)
        donor["name"] = first_spelling
        merged.append(donor)
    data["emotions"] = merged
    return dropped


def merge(path: Path, names: Iterable[str] | None = None) -> MergeResult:
    """Merge twin entries in her vocabulary file (the rule of `merge_twin_entries`); `names` limits it to those names.

    Nothing is written when there is no twin. The registered entry of each merged name becomes the merged entry.
    """
    only = None if names is None else {vocabulary.canonical_name(n) for n in names}

    def mutate(data: dict) -> tuple[list[tuple[str, str]], Callable[[], None] | None]:
        pairs = merge_twin_entries(data, only)
        if not pairs:
            return pairs, None
        index = _entries_by_key(data["emotions"])
        kept = [index[k] for k in {vocabulary.canonical_name(spelling) for _dropped, spelling in pairs} if k in index]
        return pairs, lambda: [_replace_registered(e) for e in kept]

    pairs, ok = _transact(path, mutate, "merge")
    return MergeResult(pairs or [], cannot_write=not ok)


def remove(path: Path, name: str) -> NameResult:
    """Remove every entry whose name tidies to `name` from her vocabulary file, with a verified write.

    The registry entry goes too, except a framework (baseline) entry, which stays registered. A name that tidies to nothing is
    `refused`; a name no entry carries is `missing` and nothing is written.
    """
    key = vocabulary.canonical_name(name)
    if not key:
        return NameResult(name, None, "refused")
    removed, ok = _remove_where(path, lambda tidied: tidied == key, "remove")
    if not ok:
        return NameResult(name, None, "cannot_write")
    return NameResult(name, key, "used", True) if removed else NameResult(name, None, "missing")


def _remove_where(path: Path, wanted: Callable[[str], bool], action: str) -> tuple[list[str], bool]:
    """The one removal: drop every dict entry whose tidied name satisfies `wanted`, in one verified locked write.

    Returns (the spellings removed, ok); ok False = `cannot_write`. Entries that are not dicts, have no string name or tidy to
    nothing stay, in their order. Nothing is written when none matches. A framework (baseline) entry stays registered.
    """
    def mutate(data: dict) -> tuple[list[str], Callable[[], None] | None]:
        kept: list = []
        gone: list[str] = []
        for e in data["emotions"]:
            if isinstance(e, dict) and isinstance(e.get("name"), str):
                tidied = vocabulary.canonical_name(e["name"])
                if tidied and wanted(tidied):
                    gone.append(e["name"])
                    continue
            kept.append(e)
        if not gone:
            return gone, None
        data["emotions"] = kept
        keys = {vocabulary.canonical_name(s) for s in gone}

        def registry_step() -> None:
            for k in keys:
                if not vocabulary.is_baseline(k):  # a core entry stays registered: the framework holds it
                    vocabulary._unregister(k)

        return gone, registry_step

    gone, ok = _transact(path, mutate, action)
    return (gone if ok else []), ok


@dataclass(frozen=True)
class CoreRemoval:
    removed: list[str]  # the spellings removed from her file
    cannot_write: bool = False


def remove_core_entries(path: Path) -> CoreRemoval:
    """Remove every entry her file holds for a core feeling (#801 section 8), with a verified write; run at each bridge start-up and
    in the rebuild step. After the first run it finds nothing and writes nothing.

    A missing file, or a file with no core entry, returns at once: no lock, no write, no backup-count scan, no file created. The
    unlocked read that decides this is safe because a core feeling can no longer be added (`add` and `replace_fields` refuse it) and
    the file is replaced atomically, so a "none found" cannot hide a core entry that is still there; the removal itself re-reads
    under the lock. A file that cannot be read or written is logged and left as it is (`cannot_write`); nothing here raises for a file
    problem. The framework's own entries stay in the registry.
    """
    if not path.exists():
        return CoreRemoval([])
    try:
        emotions = read(path, recover=Recovery.RAISE).data["emotions"]
    except (OSError, ValueError) as exc:
        logger.warning("vocabulary: the check for core feelings in %s could not read it (%s: %s); it is left as it is", path.name, type(exc).__name__, exc)
        return CoreRemoval([], cannot_write=True)
    if not any(
        isinstance(e, dict) and isinstance(e.get("name"), str) and vocabulary.is_baseline(e["name"]) for e in emotions
    ):
        return CoreRemoval([])
    removed, ok = _remove_where(path, vocabulary.is_baseline, "remove core entries")
    if removed:
        logger.warning("vocabulary: removed core feeling entries %s from %s (the framework holds their definitions)", sorted(removed), path.name)
    return CoreRemoval(removed, cannot_write=not ok)


def load(path: Path, *, recover: Recovery = Recovery.HEAL) -> tuple[int, BrainAnomaly | None]:
    """Register what her vocabulary file holds into the registry; no write of the vocabulary. Returns (count newly registered, anomaly).

    `recover` is the read policy (`read`): HEAL quarantines and restores a damaged file (json_io's own recovery), RAISE raises
    `JsonFileError` for it.
    """
    res = read(path, recover=recover)
    return register_entries(res.data), res.anomaly
