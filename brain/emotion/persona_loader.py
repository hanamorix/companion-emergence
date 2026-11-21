"""Per-persona emotion-vocabulary loader.

Loads a persona's `emotion_vocabulary.json` at engine startup and
registers each entry with the vocabulary registry. Idempotent on
re-register so multiple loaders in the same process don't fail.

Spec: docs/superpowers/specs/2026-04-25-vocabulary-split-design.md
"""

from __future__ import annotations

import logging
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
                write_json(path, recon_data, backups=3)
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
        write_json(path, recon_data, backups=3)
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
    read safely (a corrupt file is left untouched). A word already registered returns its
    name and touches nothing. The file is re-read under the cross-process lock so an entry
    another process wrote is kept.
    """
    name = vocabulary.validate_new_name(raw_name)
    if name is None:
        return None
    if vocabulary.get(name) is not None:
        return name
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
        known = {e.get("name") for e in entries if isinstance(e, dict)}
        if name not in known:
            entries.append(
                {
                    "name": name,
                    "description": description,
                    "category": "persona_extension",
                    "decay_half_life_days": half_life_days or dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS,
                    "intensity_clamp": 10.0,
                }
            )
            try:
                backups = compute_treatment(vocab_path.parent, vocab_path.name).backup_count
                write_json(vocab_path, data, backups=backups)
            except OSError:
                logger.warning("could not write %s; not registering %r", vocab_path, name, exc_info=True)
                return None
        # A word another thread registered meanwhile is skipped there, not raised.
        _register_from_data({"emotions": [e for e in entries if isinstance(e, dict) and e.get("name") == name]})
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

    return Emotion(
        name=str(entry["name"]),
        description=str(entry["description"]),
        category=str(entry["category"]),  # type: ignore[arg-type]
        decay_half_life_days=(
            None if entry["decay_half_life_days"] is None else float(entry["decay_half_life_days"])
        ),
        intensity_clamp=float(entry.get("intensity_clamp", 10.0)),
    )


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
    in_file = {
        vocabulary.canonical_name(e["name"]) for e in entries if isinstance(e, dict) and isinstance(e.get("name"), str)
    }
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
                new_entries.append(
                    {
                        "name": name,
                        "description": known.description,
                        "category": known.category,
                        "decay_half_life_days": known.decay_half_life_days,
                        "intensity_clamp": known.intensity_clamp,
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
    entries.extend(new_entries)
    write_json(path, data, backups=3)
    logger.warning(
        "emotion_vocabulary.json was missing %d referenced emotion(s) %s — "
        "appended them to %s (extractor-minted words the vocabulary crystallizer hadn't added, "
        "or words a repair of the file dropped).",
        len(new_entries),
        sorted(e["name"] for e in new_entries),
        path.name,
    )
    return len(new_entries), healed
