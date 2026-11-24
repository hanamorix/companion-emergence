"""The one writer of her vocabulary file (#741, #801 section 4).

`emotion_vocabulary.json` is changed by this module and by nothing else: adding a name, changing an entry's fields, merging two
entries that tidy to the same name, and removing an entry are `add`, `replace_fields`, `merge` and `remove`. (json_io's own
recovery of a corrupt file, which adds no name, is the only other code that writes it. The rebuild step and the OG migrator, the
two noted exceptions to "only the bridge calls this" (section 9), call this module too.) `reload_entries` and `reload` make the
registry match what the file holds, exactly (names the file no longer holds are removed), and write nothing: every load of the file
uses them, and `reload` is what a process's copy runs when the file changed under it (`vocabulary.follow_file`, #803 step 15).

A core feeling (one of the framework's 30, `vocabulary.is_baseline`) is never written to her file (#801 section 8): `add` and
`replace_fields` refuse it, `merge` leaves a core entry where it is (never merged into her file), and `remove_core_entries` takes
out any entry the file holds for one, at every bridge start-up and in the rebuild step.

Each change is ONE read-check-write, built once in `_transact`: the cross-process sidecar lock of the file taken with a bounded
wait (`dev_constants.EMOTION_VOCABULARY_LOCK_WAIT_S`), the file re-read, the change applied to what was just read, a verified
`json_io.write_json` with the shared backup count, then the registry changed under one `threading.Lock` (still inside the file
lock, so registry changes follow file changes in the same order). The file comes first and the registry second, so a write that
fails leaves the registry as it was. An error in the registry step itself (a bug, not a file problem) is raised to the caller
after the lock is released; the file is already written and the next read reloads what it holds. The lock is not re-entrant, so the
public operations never call each other. Lock order for any later caller: `engine_files.state_lock` is the OUTER lock, this file's
lock the next and `_REGISTRY_LOCK` the INNER one; a reload takes the registry lock only.

A name is matched by its tidied form (`vocabulary.canonical_name`) and the spelling used is the one already in the file. A new
name passes `vocabulary.gate_name` first and is added under its tidied form.

Each name gets a `NameResult`: `used` (with the spelling used), `refused` (the gate refused the name, or it is a core feeling,
which `add` and `replace_fields` never write), `missing` (no entry carries
the name: `replace_fields`, `remove`) or `cannot_write` (the file could not be read or written, its schema is wrong, or the lock
wait ran out; the cause is logged). `changed` says this call wrote that entry.
"""

from __future__ import annotations

import copy
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


def _match_registry(data: dict) -> int:
    """Make the registry hold exactly what a parsed vocabulary dict holds, plus the framework's own entries; the count of changes.

    The caller holds `_REGISTRY_LOCK` and has the check switched off in this thread (`vocabulary.suspended`): this reads the registry
    with `vocabulary.get` while the lock, which is not re-entrant, is held.

    An entry of the file is added, or has its fields replaced when its Emotion differs from the registered one; a registered name
    the file no longer holds is removed. An entry the gate refuses (empty once tidied, or too long) is skipped with the gate's warning
    and stays in the file; an entry that fails validation is skipped with a warning (so a registered name whose file entry has turned
    invalid is removed, which is what a fresh load gives); a core name stays the framework's entry; of two entries that tidy the same
    the first in the file wins. The framework's entries are never removed. An unchanged file changes nothing (0).
    """
    wanted: dict[str, Emotion] = {}
    for entry in data.get("emotions", []):
        try:
            emotion = _entry_to_emotion(entry)
        except (KeyError, ValueError, TypeError) as exc:
            entry_name = entry.get("name", "<unnamed>") if isinstance(entry, dict) else "<not a dict>"
            logger.warning("skipping emotion entry %r: %s", entry_name, exc)
            continue
        key = vocabulary.gate_name(emotion.name)
        if key is None:
            continue  # the gate has logged why
        if vocabulary.is_baseline(key) or key in wanted:
            continue
        wanted[key] = emotion
    changes = 0
    for key, emotion in wanted.items():
        held = vocabulary.get(key)
        if held is None:
            vocabulary.register(emotion)
            changes += 1
        elif held != emotion:
            vocabulary.replace_entry(emotion)
            changes += 1
    for held in vocabulary.list_all():
        key = vocabulary.canonical_name(held.name)
        if not vocabulary.is_baseline(key) and key not in wanted:
            vocabulary._unregister(key)
            changes += 1
    return changes


def reload_entries(data: dict) -> int:
    """Make the registry match a parsed vocabulary dict exactly (see `_match_registry`); the count of registry changes (added,
    replaced, removed). An unchanged file, or an absent one with nothing to drop, returns 0. The one exact sync for an explicit load.
    """
    with _REGISTRY_LOCK, vocabulary.suspended():
        return _match_registry(data)


def reload(path: Path) -> tuple[int, JsonRead]:
    """Read her file (never repairing it: `Recovery.RAISE`) and make the registry match it exactly, in ONE critical section under
    `_REGISTRY_LOCK`; then note the stat key of the bytes read (from the same open handle) as the key the copy matches.

    Returns (the count of registry changes, the read). A file that cannot be read raises (`JsonFileError`, which is both an OSError
    and a ValueError) and the registry is untouched. An absent file reads as no entries: every non-framework entry is dropped and the
    noted key is None. Lock order: the registry lock only (json_io's own read takes no file lock under RAISE); the writers take the file
    lock and then the registry lock, so there is no cycle. The registry lock is held across the file read, so on Windows a read that
    has to retry a sharing violation can make a writer's registry step wait for a moment.
    """
    with _REGISTRY_LOCK, vocabulary.suspended():
        res = read(path, recover=Recovery.RAISE)
        key = _key_of(res)  # before the sync: a read with no key raises with the registry untouched
        count = _match_registry(res.data)
        vocabulary.note_copy_key(path, key)
    return count, res


def _key_of(res: JsonRead) -> tuple[int, int, int] | None:
    """The stat key a read's data matches: the key of the bytes read, None for an absent file."""
    if res.outcome == "missing":
        return None
    if res.sig is None:
        raise ValueError("a vocabulary read with no stat key")  # RAISE reads give one; the caller treats this as an unreadable file
    return res.sig


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

    The copy follows the file (#803 step 15): before anything else the owner's copy is checked against the file
    (`vocabulary.follow_file`), so a hand edit is in the registry before this change is applied to it. The file-lock body runs with
    the check switched off (the file lock is not re-entrant, and the registry step reads the registry under the registry lock).

    The backup count is computed before the lock is taken (it scans the audit log). Any OSError or ValueError from the lock, the
    read or the write, a write that did not verify, and a lock wait that ran out are all `cannot_write`; the cause is logged.
    """
    vocabulary.follow_file()
    count = backup_count(path)
    registry_error: Exception | None = None
    try:
        with file_lock.locked_within(path, dev_constants.EMOTION_VOCABULARY_LOCK_WAIT_S), vocabulary.suspended():
            res = read_json_result(path, default=default_vocabulary, recover=Recovery.RAISE, validate=validate_schema)
            data = res.data
            payload, registry_step = mutate(data)
            if registry_step is not None:
                if not write_json(path, data, backups=count, verify=True):
                    raise OSError("the write did not verify (the previous file was restored)")
                # Still inside the file lock, so two writers' registry changes follow their file changes in the same order.
                # A registry error is kept apart from the file errors handled below: the file is written, so it is not a
                # `cannot_write`, and it is raised once the lock is released.
                with _REGISTRY_LOCK:
                    # Decided HERE, under the registry lock (D6 M1): the copy still matches the file this write started from
                    # (`res.sig`, the key of the bytes read under the file lock; None for an absent file), or no copy of it is
                    # followed. A reader thread's reload that ran after the read above noted its own key inside this lock, so the
                    # check sees the mismatch. Then the step is SKIPPED, not only the key: the copy was rebuilt from the file
                    # after this write's read (or was already behind it), so it already holds, or the next read reloads, what the
                    # file now holds, this write included; applying the step would put an entry in the copy the file may lack.
                    if vocabulary.copy_matches(path, None if res.outcome == "missing" else res.sig):
                        try:
                            registry_step()
                        except Exception as exc:  # noqa: BLE001  re-raised below, after the file lock is released
                            registry_error = exc
                        else:
                            _note_written_key(path, data)
    except (OSError, ValueError) as exc:
        logger.warning("vocabulary: %s of %s failed (%s: %s); the file and the registry are as they were", action, path.name, type(exc).__name__, exc)
        return None, False
    if registry_error is not None:
        raise registry_error
    return payload, True


def _note_written_key(path: Path, written: dict) -> None:
    """After a verified write and its registry step, in the same registry-lock block: the copy matches the file again.

    Called only when the copy matched the file this write started from (the caller decided that under the registry lock, see
    `_transact`), and only when a copy of this file is followed.

    The file is read back (its own `try`: a failure here never reaches the caller of a write that succeeded) and the key noted is the
    one of that read's handle, and only when the bytes parse to what was written: a hand edit between the write and the read makes
    the data differ, the key is not advanced, and the next read reloads. One extra read per write: a write is rare, and the read-back
    is what lets the saved key come from a handle that read exactly what was written.
    """
    if vocabulary.followed_path() != Path(path):
        return
    try:
        back = read(path, recover=Recovery.RAISE)
    except (OSError, ValueError):
        return
    if back.outcome == "ok" and back.sig is not None and back.data == written:
        vocabulary.note_copy_key(path, back.sig)


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

    Nothing is written when there is no twin. The registered entry of each merged name becomes the merged entry. A core feeling is
    never merged (its entries stay where they are for the start-up removal), whether or not `names` lists it.
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


@dataclass(frozen=True)
class OwnerCleanup:
    core: CoreRemoval | None  # None: the removal stopped on an error (logged)
    merged: MergeResult | None  # None: the merge stopped on an error (logged)
    ran: bool = True  # False: the file held no core entry and no twin, so nothing was run


def owner_file_needs_cleanup(data: dict) -> bool:
    """True when the parsed file holds a core name or two entries that tidy the same (the one merge rule decides the twins)."""
    emotions = data.get("emotions", [])
    if any(isinstance(e, dict) and isinstance(e.get("name"), str) and vocabulary.is_baseline(e["name"]) for e in emotions):
        return True
    return bool(merge_twin_entries(copy.deepcopy(data)))


def clean_owner_file(path: Path, data: dict) -> OwnerCleanup:
    """The owner's start-up steps on her file (#801 sections 3 and 8), run at the bridge's start-up and by every owner load or reload
    that finds a core name or twins in `data` (what was just read): remove any entry for a core feeling, then merge twins under the
    first spelling, both through the shared writer. Nothing is run, and no lock taken, when `data` holds neither.

    Each step has its own try: a bug in the removal does not skip the merge. A step that returns `cannot_write` is logged by the
    writer and left; the owner's next explicit load (the heartbeat and soul-review loads, each tick) finds the same core name or twin
    in the file and runs it again. Accepted edge: a reload by `vocabulary.get` that noted the file's key does not retry a failed clean-up
    until the file changes or the next explicit load (the old start-up retried only at the next start).
    """
    if not owner_file_needs_cleanup(data):
        return OwnerCleanup(None, None, ran=False)
    core: CoreRemoval | None = None
    merged: MergeResult | None = None
    try:
        core = remove_core_entries(path)  # through the module name: callers and tests may replace it
    except Exception:
        logger.exception("vocabulary: the removal of core feelings from %s stopped; the twin merge goes on", path.name)
    try:
        merged = merge(path)
        if merged.cannot_write:
            logger.warning("vocabulary: the twin merge of %s could not be written (the writer logged why); the twins stay until the next load", path.name)
        if merged.pairs:
            logger.warning("vocabulary: merged %d twin entr(ies) in %s under their first spelling", len(merged.pairs), path.name)
    except Exception:
        logger.exception("vocabulary: the twin merge of %s stopped", path.name)
    return OwnerCleanup(core, merged)


def load(path: Path, *, recover: Recovery = Recovery.HEAL) -> tuple[int, BrainAnomaly | None]:
    """Make the registry match what her vocabulary file holds, exactly (see `reload_entries`); no write of the vocabulary. Returns
    (count of registry changes, anomaly). Test-only API: no production code calls it (the loader reads and syncs itself).

    `recover` is the read policy (`read`): HEAL quarantines and restores a damaged file (json_io's own recovery), RAISE raises
    `JsonFileError` for it.
    """
    res = read(path, recover=recover)
    return reload_entries(res.data), res.anomaly
