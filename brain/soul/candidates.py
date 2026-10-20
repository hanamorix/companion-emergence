"""brain.soul.candidates — soul candidate utility functions.

Provides helpers consumed by brain.forgetting to determine which memory IDs
are currently protected from the forgetting pass by virtue of being linked
to an active (under-review) soul candidate.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from brain.health.json_io import Recovery, iter_jsonl

if TYPE_CHECKING:
    from brain.soul.store import SoulStore

# Statuses that mean a candidate is actively pending a review decision.
# ONLY these statuses protect the linked memory from the forgetting engine.
# 'expired', 'accepted', and 'rejected' are all terminal — their placeholder
# memories are no longer under active review and should become forgettable
# (or, in the accepted case, protected via the crystallised-ids path instead).
_UNDER_REVIEW_STATUSES = {"pending", "auto_pending"}


def list_under_review_memory_ids(persona_dir: Path) -> list[str]:
    """Return the memory_id of every candidate that is actively under review.

    Only candidates with status in {'pending', 'auto_pending'} are considered
    under review.  'expired', 'accepted', and 'rejected' candidates are
    terminal — their placeholder memories become forgettable again.

    A missing file → empty list; corrupt lines are skipped by the reader. A real
    read failure (permissions, I/O) is NOT swallowed here: the caller
    (``brain.forgetting._load_soul_linked_ids``) logs it and skips its pass (#370, #392).
    """
    candidates_path = persona_dir / "soul_candidates.jsonl"

    ids: list[str] = []
    # RAISE: a failed read must not read as "no candidates under review" (the forgetting pass fails closed on it).
    for r in iter_jsonl(candidates_path, recover=Recovery.RAISE):
        status = r.get("status", "auto_pending")
        if status in _UNDER_REVIEW_STATUSES:
            mid = str(r.get("memory_id") or "").strip()
            if mid:
                ids.append(mid)
    return ids


def _accepted_candidate_ids(persona_dir: Path, store: SoulStore | None) -> list[str]:
    """Memory ids of accepted candidates whose crystallization has not been revoked."""
    candidates_path = persona_dir / "soul_candidates.jsonl"
    accepted = [
        r for r in iter_jsonl(candidates_path, recover=Recovery.RAISE) if r.get("status") == "accepted"
    ]
    revoked = {c.id for c in store.list_revoked()} if accepted and store else set()

    ids: list[str] = []
    for r in accepted:
        if str(r.get("crystallization_id") or "") in revoked:
            continue
        mid = str(r.get("memory_id") or "").strip()
        if mid:
            ids.append(mid)
    return ids


def merge_target_map(persona_dir: Path) -> dict[str, set[str]]:
    """Map each merged-away candidate id to the row(s) the consolidation gate merged it into (#430).

    When the gate merges a gated memory into an existing row, the candidate id never becomes a row,
    but a soul candidate queued for it keeps that id. The moment now lives in the merge target, so
    the forgetting/narrative exemption follows this map. Read from ``consolidation_archive.jsonl``
    (``reason == "consolidation_merge"`` rows). Set-valued: a candidate re-enqueued after its target
    was deleted can merge again into another row; a dead target id simply matches no row.

    The forgetting exemption (and the narrative pass and dedup sweep, which share its loader) reads
    this archive. Rotating, archiving away or truncating it silently drops protection from every
    merged soul moment, with no error. ``test_consolidation_archive_is_never_rotated`` keeps it out of
    the supervisor's rotation tables.

    A missing archive → ``{}`` (checked before the lock, so no ``.lock`` sidecar is created). A torn or
    corrupt line is skipped with a warning. A real read failure raises: the caller fails closed (#392).
    """
    from brain.engines.consolidation import _ARCHIVE_FILENAME  # lazy: one source of the name
    from brain.utils.file_lock import file_lock

    path = persona_dir / _ARCHIVE_FILENAME
    if not path.exists():
        return {}
    targets: dict[str, set[str]] = {}
    with file_lock(path):  # the writers append under this lock: no half-written tail
        for r in iter_jsonl(path, recover=Recovery.RAISE):
            if r.get("reason") != "consolidation_merge":
                continue
            source = r.get("merged_from_candidate")
            target = r.get("target")
            target_id = target.get("id") if isinstance(target, dict) else None
            if isinstance(source, str) and source and isinstance(target_id, str) and target_id:
                targets.setdefault(source, set()).add(target_id)
    return targets


def list_crystallised_memory_ids(persona_dir: Path) -> list[str]:
    """Return the memory ids protected by a live crystallization, candidate ids first.

    Two sources, one store open:

    1. Accepted candidates. Accepting a candidate crystallises it (``review._apply_accept``)
       and keeps its ``memory_id`` on the candidate record. ``revoke_crystallization`` only
       touches crystallizations.db, so the candidate stays ``accepted`` and the store is
       consulted for revoked ids. Only an affirmative revoke drops protection: a legacy
       record with no ``crystallization_id`` or a dangling one stays protected, because
       losing a memory is the worse error (#362 ruling).
    2. Direct links (#371). A crystallization made by the ``crystallize_soul`` tool has no
       candidate record; when the caller named a memory, ``SoulStore.link_source_memory``
       recorded it. Revoked crystallizations drop out of that read by themselves.

    Candidate ids come first, then direct links (sorted by memory id), de-duplicated.
    Reading never creates ``crystallizations.db``.

    A source that cannot be read raises. Returning what the other source found would look like a
    complete answer, and the caller would run its pass with the missing memories un-exempt, so the
    caller (``brain.forgetting._load_soul_linked_ids``) logs it and skips the pass instead (#392).
    """
    db_path = persona_dir / "crystallizations.db"
    with ExitStack() as stack:
        store: SoulStore | None = None
        if db_path.exists():  # opening a SoulStore would create the file
            from brain.soul.store import SoulStore

            store = stack.enter_context(SoulStore(db_path))
        candidate_ids = _accepted_candidate_ids(persona_dir, store)
        linked = store.list_active_source_memory_ids() if store else []
    return candidate_ids + [mid for mid in linked if mid not in candidate_ids]
