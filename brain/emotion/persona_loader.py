"""Per-persona emotion-vocabulary loader.

Loads a persona's `emotion_vocabulary.json` at engine startup and
registers each entry with the vocabulary registry. Idempotent on
re-register so multiple loaders in the same process don't fail.

Every write this loader makes to the file (the rebuild of a missing or corrupt file, the memory heal) goes through
`brain.emotion.vocabulary_file`, the one writer of that file.

Spec: the 2026-04-25 vocabulary-split design, never committed and lost (#797).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from brain.emotion import vocabulary, vocabulary_file
from brain.health.json_io import Recovery
from brain.memory.store import MemoryStore

if TYPE_CHECKING:
    from brain.health.anomaly import BrainAnomaly

logger = logging.getLogger(__name__)


def _rebuild_from_memories(path: Path, recon_data: dict) -> tuple[int, bool]:
    """Write the reconstructed entries through the shared writer.

    The reconstructed entries are persona extensions only (a core feeling is never written to her file). Returns (count newly
    registered, whether the file was written).
    """
    before = len(vocabulary.list_all())
    results = vocabulary_file.add(
        path,
        [(entry["name"], {k: v for k, v in entry.items() if k != "name"}) for entry in recon_data["emotions"]],
    )
    written = not any(r.outcome == "cannot_write" for r in results)
    return len(vocabulary.list_all()) - before, written


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
                registered, written = _rebuild_from_memories(path, recon_data)
                if written:
                    logger.warning(
                        "emotion_vocabulary.json was missing but memories reference "
                        "persona emotions %s, reconstructed %d entries from memories "
                        "and wrote %s. (If this followed `nell recover`, the recover "
                        "backup omitted the vocabulary file.)",
                        sorted(e["name"] for e in extensions),
                        len(recon_data["emotions"]),
                        path.name,
                    )
                return registered, None
        return 0, None

    recover = Recovery.RAISE if read_only else Recovery.HEAL
    _res = vocabulary_file.read(path, recover=recover)
    data, anomaly = _res.data, _res.anomaly
    registered = 0

    # Reconstruct from memories when reset_to_default fires on vocabulary.
    # The default factory writes empty `{"version":1,"emotions":[]}` — that's
    # a truthful empty default but it loses the persona-extension entries
    # the brain has been operating with. If we have memory access, the brain
    # can re-learn its own vocabulary from how it has been using emotions.
    if anomaly is not None and anomaly.action == "reset_to_default" and store is not None:
        from brain.health.reconstruct import reconstruct_vocabulary_from_memories

        recon_data = reconstruct_vocabulary_from_memories(store)
        rebuilt, written = _rebuild_from_memories(path, recon_data)
        if written:
            registered += rebuilt
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
        # A rebuild the file could not take (the writer logged why) registers nothing and leaves the anomaly as the heal
        # reported it: the registry never holds what the file does not.

    if anomaly is not None:
        logger.warning(
            "emotion_vocabulary anomaly detected: %s action=%s file=%s",
            anomaly.kind,
            anomaly.action,
            anomaly.file,
        )

    registered += vocabulary_file.register_entries(data)

    if store is not None:
        registered += _heal_referenced_but_unregistered(path, store)

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


def _heal_referenced_but_unregistered(path: Path, store: MemoryStore) -> int:
    """Register + persist emotions referenced in active memories but absent from
    the vocabulary — e.g. extractor-minted channel names the growth crystallizer
    never added. Returns the count healed.

    Replaces the old misdirecting "run nell migrate" warning. Those orphaned
    emotions broke the emotion readers and soul crystallization (the names
    were silently skipped). Healing them as persona_extension entries works for
    native and OG personas alike, and persists so the next load is clean.

    The file is written through `vocabulary_file.add` (file first, registry second): a write that cannot be made leaves the
    names unregistered, and the next load heals them again.
    """
    seen: set[str] = set()
    names: list[str] = []
    for mem in store.list_active(limit=None):
        # Names only: a stored emotion whose value is not finite still has a name the vocabulary needs.
        for raw_name in MemoryStore.stored_emotion_names(mem):
            # #174: heal under the canonical spelling so a variant memory key
            # (present-ness, Love_Grief_blend) never mints a twin entry; #801: a name
            # the gate refuses (empty once tidied, too long) is not healed.
            name = vocabulary.gate_name(raw_name)
            if name is None or name in seen or vocabulary.get(name) is not None:
                continue
            seen.add(name)
            names.append(name)

    if not names:
        return 0

    results = vocabulary_file.add(path, [(name, None) for name in names])
    healed = sum(1 for r in results if r.changed)
    if healed:
        logger.warning(
            "emotion_vocabulary.json was missing %d referenced emotion(s) %s: "
            "registered them as persona_extension and appended to %s (extractor-"
            "minted emotions the vocabulary crystallizer hadn't added yet).",
            healed,
            sorted(r.spelling for r in results if r.changed and r.spelling),
            path.name,
        )
    return healed
