"""Per-persona emotion-vocabulary loader.

Loads a persona's `emotion_vocabulary.json` at engine startup and
registers each entry with the vocabulary registry. Idempotent on
re-register so multiple loaders in the same process don't fail.

Spec: docs/superpowers/specs/2026-04-25-vocabulary-split-design.md
"""

from __future__ import annotations

import copy
import logging
import math
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from brain import dev_constants
from brain.emotion import vocabulary
from brain.emotion.vocabulary import Emotion
from brain.health.adaptive import compute_treatment
from brain.health.json_io import JsonFileError, Recovery, read_json, read_json_result, write_json
from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION
from brain.memory.store import MemoryStore
from brain.paths import cadence_state_path
from brain.utils import file_ops
from brain.utils.file_lock import file_lock

if TYPE_CHECKING:
    from brain.health.anomaly import BrainAnomaly

logger = logging.getLogger(__name__)

_DEFAULT_VOCAB: dict = {"version": 1, "emotions": []}


def _default_vocab_factory() -> dict:
    return {"version": 1, "emotions": []}


def _vocab_schema_validator(data: object) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("emotions"), list):
        raise ValueError("emotion_vocabulary schema invalid: missing 'emotions' list")


def load_persona_vocabulary_with_anomaly(
    path: Path,
    *,
    store: MemoryStore | None = None,
    read_only: bool = False,
) -> tuple[int, BrainAnomaly | None]:
    """Load persona vocabulary with self-healing from .bak rotation if corrupt.

    Returns (registered_count, anomaly_or_None).
      - Missing file → 0, no anomaly.
      - Corrupt file → quarantine, restore from .bak1/.bak2/.bak3 or reset to
        empty default. Anomaly set.
      - Healthy file → counts of newly registered emotions, no anomaly.
    """
    if read_only and store is not None:
        raise ValueError("read_only takes no store: the store's rebuild and heal both write")
    if read_only or (store is None and not path.exists()):
        return _load_vocabulary(path, store=store, read_only=read_only)
    if not _load_may_write(path, store=store):
        # Healthy, nothing broken, already healed: a read-only load does the same work without waiting on a writer that
        # holds the lock. A file damaged since the check raises there, and the locked load below repairs it.
        try:
            return _load_vocabulary(path, store=None, read_only=True)
        except JsonFileError:
            pass
    # Every write below (the json_io heal, a rebuild from memories, the heal of referenced words) happens under the lock
    # register_new_word and vocab repair take, on data read inside it, so a word they add meanwhile is never dropped.
    with file_lock(path):
        return _load_vocabulary(path, store=store, read_only=False)


def _load_may_write(path: Path, *, store: MemoryStore | None) -> bool:
    """True when a writing load of `path` could write: a missing file to rebuild, a damaged one, a broken entry, or the heal due."""
    if not path.exists():
        return store is not None
    if store is not None and not cadence_state_path(path.parent, "vocab_heal_final.json").exists():
        return True
    try:
        data = read_json(path, default=_default_vocab_factory, recover=Recovery.RAISE, validate=_vocab_schema_validator)
    except JsonFileError:
        return True
    return bool(_repair_broken_entries(copy.deepcopy(data)))  # the repair's own rule, run on a copy


def _vocab_backups(path: Path) -> int:
    """The backup count for a write of the vocabulary file: the file's own treatment (more after repeated anomalies).

    The treatment reads the heartbeat audit log; when that can't be read, the write still happens, with the usual three.
    """
    try:
        return compute_treatment(path.parent, path.name).backup_count
    except Exception:  # noqa: BLE001 - a bad audit log must not stop a vocabulary write
        logger.warning("could not read the backup treatment for %s; writing with 3 backups", path.name, exc_info=True)
        return 3


def _load_vocabulary(
    path: Path,
    *,
    store: MemoryStore | None,
    read_only: bool,
) -> tuple[int, BrainAnomaly | None]:
    """The body of `load_persona_vocabulary_with_anomaly`; a writing load runs it under `file_lock(path)`."""
    if not path.exists():
        if store is not None:
            # Recover-dropped-vocab self-heal: `nell recover` backs up only
            # memories.db + hebbian.db, so a grown emotion_vocabulary.json is
            # lost. If memories reference non-baseline emotions, reconstruct the
            # vocabulary from them rather than orphaning the emotions (which
            # broke soul crystallization) and misdirecting the user to the OG
            # migrate path. A missing file is not an integrity anomaly → no
            # BrainAnomaly. A persona referencing only baseline emotions writes
            # nothing (fresh-persona behavior preserved).
            from brain.health.reconstruct import reconstruct_vocabulary_from_memories

            recon_data = reconstruct_vocabulary_from_memories(store)
            extensions = [
                e for e in recon_data["emotions"] if e["category"] == "persona_extension"
            ]
            if extensions:
                write_json(path, recon_data, backups=_vocab_backups(path))
                registered = _register_from_data(recon_data)
                logger.warning(
                    "emotion_vocabulary.json was missing but memories reference "
                    "persona emotions %s — reconstructed %d entries from memories "
                    "and wrote %s. (If this followed `nell recover`, the recover "
                    "backup omitted the vocabulary file.)",
                    sorted(e["name"] for e in extensions),
                    len(recon_data["emotions"]),
                    path.name,
                )
                return registered, None
        return 0, None

    recover = Recovery.RAISE if read_only else Recovery.HEAL
    _res = read_json_result(path, default=_default_vocab_factory, recover=recover, validate=_vocab_schema_validator)
    data, anomaly = _res.data, _res.anomaly

    # Reconstruct from memories when reset_to_default fires on vocabulary.
    # The default factory writes empty `{"version":1,"emotions":[]}` — that's
    # a truthful empty default but it loses the persona-extension entries
    # the brain has been operating with. If we have memory access, the brain
    # can re-learn its own vocabulary from how it has been using emotions.
    if anomaly is not None and anomaly.action == "reset_to_default" and store is not None:
        from brain.health.reconstruct import reconstruct_vocabulary_from_memories

        recon_data = reconstruct_vocabulary_from_memories(store)
        write_json(path, recon_data, backups=_vocab_backups(path))
        data = recon_data
        # Replace the anomaly with one whose action reflects the reconstruction.
        # Same kind (json_parse_error / schema_mismatch — that's why we needed
        # to reconstruct) and same forensic quarantine path; the heal path
        # advanced beyond reset.
        from brain.health.anomaly import BrainAnomaly

        anomaly = BrainAnomaly(
            timestamp=anomaly.timestamp,
            file=anomaly.file,
            kind=anomaly.kind,
            action="reconstructed_from_memories",
            quarantine_path=anomaly.quarantine_path,
            likely_cause=anomaly.likely_cause,
            detail=(
                f"{anomaly.detail}; reconstructed "
                f"{len(recon_data['emotions'])} entries from memories"
            ),
        )

    if anomaly is not None:
        logger.warning(
            "emotion_vocabulary anomaly detected: %s action=%s file=%s",
            anomaly.kind,
            anomaly.action,
            anomaly.file,
        )

    if not read_only:
        repaired = _repair_broken_entries(data)
        if repaired:
            write_json(path, data, backups=_vocab_backups(path))
            logger.warning("emotion_vocabulary.json had entries the loader couldn't register %s; repaired them in place", sorted(repaired))

    registered = _register_from_data(data)

    if store is not None:
        # One-time backfill: new words now register when written, so the scan of every active
        # memory runs only until its marker exists. Reconstruction above is not gated by it, and
        # a file this load repaired (an older .bak restored, or a reset) may lack words only the
        # memories still carry, so an anomaly runs the scan again. The health walker, which
        # repairs with no store, deletes the marker for the same reason.
        marker = cadence_state_path(path.parent, "vocab_heal_final.json")
        if anomaly is not None or not marker.exists():
            written, healed = _heal_referenced_but_unregistered(path, data, store)
            registered += healed
            write_json(
                marker,
                {"healed_at": datetime.now(UTC).isoformat(), "count": written},
            )

    return registered, anomaly


def load_persona_vocabulary(
    path: Path,
    *,
    store: MemoryStore | None = None,
    read_only: bool = False,
) -> int:
    """Load persona vocabulary from JSON file and register each entry.

    Returns the count of emotions newly registered. Re-registering an
    already-registered name is a silent no-op (idempotent), so calling
    this twice for the same persona returns 0 the second time.

    Missing `path` → returns 0 silently. Fresh personas don't need a file.
    Corrupt JSON → quarantine + heal from .bak, log a warning.
    Per-entry validation failure → that entry skipped + warning,
    other entries proceed.

    If `store` is provided, after registration the loader heals any emotion
    names referenced in memories but absent from the vocabulary — registering
    each as a persona_extension and persisting — rather than warning.

    `read_only=True` writes nothing and takes no `store` (ValueError): a corrupt file raises `JsonFileError` instead of being
    quarantined and reset, so a process that cannot rebuild from memories leaves the repair to one that can.
    """
    count, _anomaly = load_persona_vocabulary_with_anomaly(path, store=store, read_only=read_only)
    return count


def register_new_word(
    vocab_path: Path,
    raw_name: str,
    *,
    half_life_days: float | None = None,
    description: str = PLACEHOLDER_DESCRIPTION,
) -> str | None:
    """Register a new emotion word in this process and persist it to the vocabulary file.

    Returns the canonical name, or None when the gate refuses the name or the file cannot be
    read or written safely (a corrupt file is left untouched). The file is re-read under the
    cross-process lock so an entry another process wrote is kept. A word this process already
    holds is still checked against the file: a repair can drop it there, and it is then written
    back with its registered fields, so the scheduler's `emotion_added` is true of the file.
    """
    name = vocabulary.validate_new_name(raw_name)
    if name is None:
        return None
    held = vocabulary.get(name)
    if held is not None:
        held_half_life = held.decay_half_life_days
        if held_half_life is not None and not math.isfinite(held_half_life):
            held_half_life = dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS
        new_entry = {
            "name": name,
            "description": held.description,
            "category": held.category,
            "decay_half_life_days": held_half_life,
            "intensity_clamp": held.intensity_clamp if math.isfinite(held.intensity_clamp) else 10.0,
        }
    else:
        new_entry = {
            "name": name,
            "description": description,
            "category": "persona_extension",
            "decay_half_life_days": half_life_days or dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS,
            "intensity_clamp": 10.0,
        }
    with file_lock(vocab_path):
        try:
            data = read_json(
                vocab_path,
                default=_default_vocab_factory,
                recover=Recovery.RAISE,
                expect=dict,
                validate=_vocab_schema_validator,
            )
        except JsonFileError:
            logger.warning(
                "emotion vocabulary %s is unreadable; not registering %r", vocab_path, name, exc_info=True
            )
            return None
        entries = data["emotions"]
        # A broken entry (NaN, a missing field) is fixed in place first: left as it is, it would make the write raise,
        # or count as present while the loader skips it.
        repaired = _repair_broken_entries(data)
        known = {_entry_canonical_name(e) for e in entries}
        if name not in known:
            entries.append(new_entry)
        if name not in known or repaired:
            try:
                write_json(vocab_path, data, backups=_vocab_backups(vocab_path))
            except (OSError, ValueError):
                # ValueError: a value the repair doesn't cover (an extra field holding NaN) that JSON can't write back.
                logger.warning("could not write %s; not registering %r", vocab_path, name, exc_info=True)
                return None
        # A word another thread registered meanwhile is skipped there, not raised.
        _register_from_data({"emotions": [e for e in entries if _entry_canonical_name(e) == name]})
    return name


_REFRESH_KEYS: dict[str, object] = {}
_REFRESH_LOCK = threading.Lock()


def refresh_if_changed(vocab_path: Path) -> int:
    """Register words another process added to the vocabulary file; read-only.

    Costs one ``stat`` when the file is unchanged since the last call for this path. A changed
    file is loaded with ``read_only=True`` (no heal, quarantine or write) and its count of
    newly registered words returned. A failed read re-raises, and the file's key is recorded
    all the same so a corrupt file is not re-parsed on every call; a failed ``stat`` records
    nothing and propagates.
    """
    path_key = Path(vocab_path).as_posix()
    with _REFRESH_LOCK:
        stat = file_ops.stat_path(vocab_path)
        key = None if stat is None else stat.key
        if path_key in _REFRESH_KEYS and _REFRESH_KEYS[path_key] == key:
            return 0
        if key is None:
            _REFRESH_KEYS[path_key] = None
            return 0
        try:
            count = load_persona_vocabulary(vocab_path, read_only=True)
        except JsonFileError as exc:
            if exc.kind != "io":
                _REFRESH_KEYS[path_key] = key
            raise
        _REFRESH_KEYS[path_key] = key
        return count


def _register_from_data(data: dict) -> int:
    """Register emotions from a parsed vocab dict; return count of newly registered."""
    registered = 0
    for entry in data.get("emotions", []):
        try:
            emotion = _entry_to_emotion(entry)
        except (KeyError, ValueError, TypeError) as exc:
            entry_name = (
                entry.get("name", "<unnamed>") if isinstance(entry, dict) else "<not a dict>"
            )
            logger.warning("skipping emotion entry %r: %s", entry_name, exc)
            continue

        if vocabulary.get(emotion.name) is not None:
            # Already registered — idempotent skip.
            continue

        try:
            vocabulary.register(emotion)
        except ValueError:
            # Another thread registered it between the check above and here: it is registered.
            continue
        registered += 1
    return registered


def _entry_to_emotion(entry: dict) -> Emotion:
    """Build an Emotion from a JSON entry. Raises on missing/invalid fields."""
    if not isinstance(entry, dict):
        raise TypeError("entry must be a dict")
    required = ("name", "description", "category", "decay_half_life_days")
    for key in required:
        if key not in entry:
            raise KeyError(f"missing required field {key!r}")

    half_life = None if entry["decay_half_life_days"] is None else float(entry["decay_half_life_days"])
    clamp = float(entry.get("intensity_clamp", 10.0))
    # Python's JSON reads NaN and Infinity; registered, they would poison decay and make every write of the file raise.
    if (half_life is not None and not math.isfinite(half_life)) or not math.isfinite(clamp):
        raise ValueError("decay_half_life_days and intensity_clamp must be finite")
    return Emotion(
        name=str(entry["name"]),
        description=str(entry["description"]),
        category=str(entry["category"]),  # type: ignore[arg-type]
        decay_half_life_days=half_life,
        intensity_clamp=clamp,
    )


def _is_registrable(entry: object) -> bool:
    """True when `_entry_to_emotion` accepts the file entry, so a load would register it."""
    try:
        _entry_to_emotion(entry)  # type: ignore[arg-type]
    except (KeyError, ValueError, TypeError):
        return False
    return True


def _entry_canonical_name(entry: object) -> str | None:
    """The entry's name, canonical; None when it has no string name."""
    if isinstance(entry, dict) and isinstance(entry.get("name"), str):
        return vocabulary.canonical_name(entry["name"])
    return None


def _finite_or(value: object, fallback: float) -> float:
    """`value` as a finite float, else `fallback` (a missing, non-numeric, NaN or infinite value)."""
    if isinstance(value, bool):
        return fallback
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _first_usable_number(values: list[object], fallback: float) -> float:
    """The first of `values` that is a finite number (not a bool), as a float; `fallback` when none is."""
    for value in values:
        number = _finite_or(value, math.nan)
        if not math.isnan(number):
            return number
    return fallback


def _repair_broken_entries(data: dict) -> list[str]:
    """Fix in place every word whose entries the loader can't all register; return the names of the words changed.

    Entries are grouped by canonical name. A group with no broken entry is left as it is (plain spelling twins are vocab
    repair's to merge). A group with one becomes one entry, so a repaired copy can never come first and win the registry:
      - the base is the first whole entry with a real description, else the first whole one, else the first broken one
        with a real description, else the first broken one;
      - a real description from any entry of the word fills a placeholder base, so dropping a twin never deletes it;
      - a broken base takes each unusable field (missing, the wrong type, or a NaN or infinite number, which JSON can't
        write back) from the first other entry where it's usable, else the placeholder's value. A half-life of None (no
        decay) is usable;
      - the word's other broken entries are dropped; its other whole entries are kept.
    Then any other field of any named entry holding NaN or infinity, at any depth, becomes null, so the file can be
    written again. An entry with no string name is left alone: it names no word, so it hides none.
    """
    from brain.health.reconstruct import PLACEHOLDER_DECAY_DAYS

    entries = data.get("emotions", [])
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        name = _entry_canonical_name(entry)
        if name is not None:
            groups.setdefault(name, []).append(entry)  # type: ignore[arg-type]

    def _real(entry: dict) -> bool:
        description = entry.get("description")
        return isinstance(description, str) and description != PLACEHOLDER_DESCRIPTION

    repaired: list[str] = []
    dropped: set[int] = set()
    for group in groups.values():
        broken = [e for e in group if not _is_registrable(e)]
        if not broken:
            continue
        whole = [e for e in group if _is_registrable(e)]
        pool = whole or broken
        base = next((e for e in pool if _real(e)), pool[0])
        others = [e for e in group if e is not base]
        if not _real(base):
            donor = next((e for e in others if _real(e)), None)
            if donor is not None:
                base["description"] = donor["description"]
        if base in broken:
            if not isinstance(base.get("description"), str):
                base["description"] = PLACEHOLDER_DESCRIPTION
            if not isinstance(base.get("category"), str):
                base["category"] = next(
                    (e["category"] for e in others if isinstance(e.get("category"), str)), "persona_extension"
                )
            if not ("decay_half_life_days" in base and base["decay_half_life_days"] is None):  # None: no decay, usable
                base["decay_half_life_days"] = _first_usable_number(
                    [base.get("decay_half_life_days")] + [e.get("decay_half_life_days") for e in others],
                    PLACEHOLDER_DECAY_DAYS,
                )
            base["intensity_clamp"] = _first_usable_number(
                [base.get("intensity_clamp")] + [e.get("intensity_clamp") for e in others], 10.0
            )
        dropped.update(id(e) for e in broken if e is not base)
        repaired.append(base["name"])

    def _null_non_finite(value: object) -> tuple[object, bool]:
        if isinstance(value, float) and not math.isfinite(value):
            return None, True
        if isinstance(value, dict):
            changed = False
            for key, inner in value.items():
                value[key], inner_changed = _null_non_finite(inner)
                changed = changed or inner_changed
            return value, changed
        if isinstance(value, list):
            changed = False
            for index, inner in enumerate(value):
                value[index], inner_changed = _null_non_finite(inner)
                changed = changed or inner_changed
            return value, changed
        return value, False

    # The core numbers were given usable values above, so a NaN half-life never turns into "no decay" here.
    for group in groups.values():
        for entry in group:
            if id(entry) in dropped:
                continue
            changed = False
            for key, value in entry.items():
                if key in ("decay_half_life_days", "intensity_clamp"):
                    continue
                entry[key], key_changed = _null_non_finite(value)
                changed = changed or key_changed
            if changed and entry["name"] not in repaired:
                repaired.append(entry["name"])
    if dropped:
        entries[:] = [e for e in entries if id(e) not in dropped]
    return repaired


def _heal_referenced_but_unregistered(path: Path, data: dict, store: MemoryStore) -> tuple[int, int]:
    """Persist (and register) emotions referenced in active memories but absent from the vocabulary FILE.

    E.g. extractor-minted channel names the growth crystallizer never added, or words a repair of the file dropped. The
    comparison is against the file's entries, not this process's registry: a long-running process (the bridge) still
    holds a word in its registry after the file lost it, and only writing it back keeps it across a restart. Baseline
    emotions are never written. A word this process already has is written with its registered fields; any other gets
    a placeholder entry and is registered. Returns ``(entries written to the file, words newly registered)``.

    Replaces the old misdirecting "run nell migrate" warning. Those orphaned
    emotions broke the emotion readers and soul crystallization (the names
    were silently skipped). Healing them as persona_extension entries works for
    native and OG personas alike, and persists so the next load is clean.
    """
    from brain.health.reconstruct import PLACEHOLDER_DECAY_DAYS, PLACEHOLDER_DESCRIPTION

    entries = data.setdefault("emotions", [])
    # Only an entry the loader can register counts as present: a malformed one (`{"name": x}`) would hide its word for good.
    in_file = {_entry_canonical_name(e) for e in entries if _is_registrable(e)}
    skip = in_file | {e.name for e in vocabulary._BASELINE}
    new_entries: list[dict] = []
    for mem in store.list_active(limit=None):
        # Names only: a stored emotion whose value is not finite still has a name the vocabulary needs.
        for raw_name in MemoryStore.stored_emotion_names(mem):
            # #174: heal under the canonical spelling so a variant memory key
            # (present-ness, Love_Grief_blend) never mints a twin entry.
            name = vocabulary.canonical_name(raw_name)
            if name in skip:
                continue
            skip.add(name)
            known = vocabulary.get(name)
            if known is not None:
                # JSON has no NaN or infinity: a non-finite registered number is written as the placeholder's.
                half_life = known.decay_half_life_days
                if half_life is not None and not math.isfinite(half_life):
                    half_life = PLACEHOLDER_DECAY_DAYS
                clamp = known.intensity_clamp if math.isfinite(known.intensity_clamp) else 10.0
                new_entries.append(
                    {
                        "name": name,
                        "description": known.description,
                        "category": known.category,
                        "decay_half_life_days": half_life,
                        "intensity_clamp": clamp,
                    }
                )
            else:
                new_entries.append(
                    {
                        "name": name,
                        "description": PLACEHOLDER_DESCRIPTION,
                        "category": "persona_extension",
                        "decay_half_life_days": PLACEHOLDER_DECAY_DAYS,
                        "intensity_clamp": 10.0,
                    }
                )

    if not new_entries:
        return 0, 0

    # Registers only the words missing from the registry; one another thread registered meanwhile is skipped there.
    healed = _register_from_data({"emotions": new_entries})
    written = {e["name"] for e in new_entries}
    # A malformed entry for a word written here is replaced, not left beside the whole one.
    data["emotions"] = [e for e in entries if _is_registrable(e) or _entry_canonical_name(e) not in written]
    data["emotions"].extend(new_entries)
    write_json(path, data, backups=_vocab_backups(path))
    logger.warning(
        "emotion_vocabulary.json was missing %d referenced emotion(s) %s — "
        "appended them to %s (extractor-minted words the vocabulary crystallizer hadn't added, "
        "or words a repair of the file dropped).",
        len(new_entries),
        sorted(e["name"] for e in new_entries),
        path.name,
    )
    return len(new_entries), healed
