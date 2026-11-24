"""Per-persona emotion-vocabulary loader.

Loads a persona's `emotion_vocabulary.json` and makes the vocabulary registry match it exactly (entries are added or have their
fields replaced, names the file no longer holds are removed, the framework's own stay). An unchanged file changes nothing, so
loading twice is harmless. The loader also hands the registry the file's path and a reload function (`vocabulary.set_file_copy`):
from then on this process's copy follows the file, and a read after the file changed (a hand edit, another process's write)
reloads it first (#803 step 15).

Every write this loader makes to the file (the rebuild of a missing or corrupt file, the memory heal) goes through
`brain.emotion.vocabulary_file`, the one writer of that file.

Spec: the 2026-04-25 vocabulary-split design, never committed and lost (#797).
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from brain.emotion import vocabulary, vocabulary_file
from brain.health.json_io import JsonFileError, Recovery
from brain.memory.store import MemoryStore
from brain.utils import file_ops

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

    Returns (registry_changes, anomaly_or_None): the count of registry changes (added, replaced, removed) that made the registry match
    the file; an unchanged file, or an absent one with nothing to drop, gives 0.
      - Missing file → the copy matches that (non-framework entries dropped); with a store, the rebuild from memories. No anomaly.
      - Corrupt file → quarantine, restore from .bak1/.bak2/.bak3 or reset to
        empty default. Anomaly set.
      - Healthy file → the registry matches it, no anomaly.
    In the owner (`vocabulary.is_bridge_owner()`) a file holding a core name or twins is then cleaned (`vocabulary_file.clean_owner_file`).
    """
    if read_only and store is not None:
        raise ValueError("read_only takes no store: the store's rebuild and heal both write")
    # This process's copy follows the file from here on (#803 step 15). It is set stale BEFORE the file is read, so a first read that
    # fails (a damaged file at a read-only first load) still leaves a followed copy: the next read's reload logs it once and notes
    # the key. One persona's registry per process: an exact load of persona B removes persona A's names (production has one persona
    # per process).
    adopt_copy(path, stale=True)
    # The whole load runs with the check switched off in this thread: its own reads of the registry must not reload it again.
    with vocabulary.suspended():
        return _load_followed(path, store=store, read_only=read_only)


def adopt_copy(path: Path, *, stale: bool = False) -> None:
    """Make this process's copy follow `path` (a no-op when it already does, unless `stale` forces a reload at the next check)."""
    if stale or vocabulary.followed_path() != Path(path):
        vocabulary.set_file_copy(path, partial(reload_copy, path))


def _rebuild_missing_file(path: Path, store: MemoryStore) -> int:
    """The missing-file branch: rebuild the file from the memories that name feelings outside the framework. Returns the count registered.

    Recover-dropped-vocab self-heal: `nell recover` backs up only memories.db + hebbian.db, so a grown emotion_vocabulary.json is
    lost. If memories reference non-baseline emotions, reconstruct the vocabulary from them rather than orphaning the emotions (which
    broke soul crystallization) and misdirecting the user to the OG migrate path. A missing file is not an integrity anomaly, so no
    BrainAnomaly. A persona referencing only baseline emotions writes nothing (fresh-persona behavior preserved). Shared by the
    start-up and explicit load and by the owner's reload of a file that vanished (`reload_copy`).
    """
    from brain.health.reconstruct import reconstruct_vocabulary_from_memories

    recon_data = reconstruct_vocabulary_from_memories(store)
    extensions = [e for e in recon_data["emotions"] if e["category"] == "persona_extension"]
    if not extensions:
        return 0
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
    return registered


def _log_anomaly(anomaly: BrainAnomaly | None) -> None:
    """The one log line for a repair of her vocabulary file (the start-up load and the owner's reload both write it)."""
    if anomaly is not None:
        logger.warning(
            "emotion_vocabulary anomaly detected: %s action=%s file=%s",
            anomaly.kind,
            anomaly.action,
            anomaly.file,
        )


def reload_copy(path: Path) -> None:
    """What `vocabulary.follow_file` runs when the file changed under this process's copy: make the copy match the file again.

    Everyone: read the file without repairing it (`Recovery.RAISE`) and match the registry to it exactly (`vocabulary_file.reload`).
    A file that cannot be read leaves the copy as it was, is logged once, and the saved key becomes the key of the file as found
    (stat taken BEFORE the read, the safe direction), so it is not read and logged again until it changes. A stat that fails cannot
    say which file was found: no key is saved. The owner repairs a damaged file (bad content, not an io error) through
    `vocabulary_file.repair` (VC1: quarantine, restore a backup or reset to the default, under the file's lock, ending with the copy
    matching the file), logs the anomaly as the start-up load does, and goes on with the steps below on what the repair left; a repair
    that fails is logged and the file and the copy stay as they were. After a RESET the owner's reload does not rebuild names from
    memories (the start-up load does); the heartbeat's explicit load heals names that memories reference.

    Owner only, after a successful read: an absent file with a `memories.db` beside it takes the start-up's missing-file path with a
    short-lived store (opened here, closed when done, none kept); a file holding a core name or twins gets the start-up clean-up
    (`vocabulary_file.clean_owner_file`). A process that is not the owner only reads: no write, no clean-up, no repair; a core entry in
    the file stays the framework's, the first twin wins.

    The owner's reload may wait on the file lock (up to `EMOTION_VOCABULARY_LOCK_WAIT_S`) on the thread that did the read. The store's
    open runs `executescript(_SCHEMA)` with IF NOT EXISTS, which takes no write lock on existing tables, so a wait on a write transaction
    this same thread holds is unlikely; a store that raises is caught by `follow_file`, which logs it once and goes on.
    """
    owner = vocabulary.is_bridge_owner()
    try:
        before = file_ops.stat_path(path)
        found_key: tuple[int, int, int] | None = None if before is None else before.key
        found_known = True
    except OSError:
        found_key, found_known = None, False
    try:
        try:
            _count, res = vocabulary_file.reload(path)
        except JsonFileError as damaged:
            if not owner or damaged.kind == "io":
                raise
            res, _changes = vocabulary_file.repair(path)  # the owner repairs; it logged and raised if it could not
            _log_anomaly(res.anomaly)
    except (OSError, ValueError) as exc:  # JsonFileError is both
        logger.warning(
            "emotion_vocabulary.json cannot be read or repaired (%s: %s); the copy stays as it was until the file changes",
            type(exc).__name__,
            exc,
        )
        if found_known:
            vocabulary.note_copy_key(path, found_key)
        return
    if not owner:
        return
    if res.outcome == "missing":
        db = path.parent / "memories.db"
        if file_ops.stat_path(db) is not None:
            store = MemoryStore(db, integrity_check=False)
            try:
                _rebuild_missing_file(path, store)
            finally:
                store.close()
    else:
        # Accepted edge (#803 step 15, D6 m3): a clean-up that could not be written (the lock wait ran out) is not retried by a later read,
        # because the reload has noted the file's key; it is retried when the file changes or by the owner's next explicit load.
        vocabulary_file.clean_owner_file(path, res.data)


def _load_followed(path: Path, *, store: MemoryStore | None, read_only: bool) -> tuple[int, BrainAnomaly | None]:
    """The body of `load_persona_vocabulary_with_anomaly`, run with the check off and the copy already followed (stale)."""
    try:
        present = file_ops.stat_path(path) is not None
    except OSError:
        return 0, None  # cannot say whether the file is there (what `path.exists()` answered before): nothing loaded, the copy stays stale
    if not present:
        # An absent file: the copy matches that too (every non-framework entry goes), then the rebuild from memories when a store is given.
        registered = vocabulary_file.reload_entries(vocabulary_file.default_vocabulary())
        vocabulary.note_copy_key(path, None)
        if store is not None:
            registered += _rebuild_missing_file(path, store)
        return registered, None

    if read_only:
        _res, registered = vocabulary_file.read(path, recover=Recovery.RAISE), 0
    else:
        # The owner's load repairs a damaged file through the accessor, which ends with the registry matching the repaired file;
        # its count of registry changes is part of this load's count.
        _res, registered = vocabulary_file.repair(path)
    data, anomaly = _res.data, _res.anomaly

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

    _log_anomaly(anomaly)

    registered += vocabulary_file.reload_entries(data)
    if _res.outcome == "ok" and _res.sig is not None:
        vocabulary.note_copy_key(path, _res.sig)  # the copy matches the bytes this read parsed; a heal's own reload noted its key already, a rebuild leaves it stale

    # The owner's clean-up on what was just read (core entries out, twins merged), so a load that sees a hand edit first does not
    # record its key with a core name or twin left in the file. After the key above, so the clean-up's own write advances it.
    if vocabulary.is_bridge_owner() and not read_only:
        vocabulary_file.clean_owner_file(path, data)

    if store is not None:
        registered += _heal_referenced_but_unregistered(path, store)

    return registered, anomaly


def load_persona_vocabulary(
    path: Path,
    *,
    store: MemoryStore | None = None,
    read_only: bool = False,
) -> int:
    """Load persona vocabulary from JSON file and make the registry match it.

    Returns the count of registry changes (added, replaced, removed). An unchanged file changes nothing, so calling
    this twice for the same persona returns 0 the second time. The copy then follows the file (`vocabulary.follow_file`).

    Missing `path` → the copy matches that, returns 0 when it has nothing to drop. Fresh personas don't need a file.
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
