"""brain.soul.candidates — soul candidate utility functions.

Provides helpers consumed by brain.forgetting to determine which memory IDs
are currently protected from the forgetting pass by virtue of being linked
to an active (under-review) soul candidate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.soul.store import SoulStore

logger = logging.getLogger(__name__)

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

    Fail-soft: corrupt or missing soul_candidates.jsonl → empty list.
    """
    from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

    candidates_path = persona_dir / "soul_candidates.jsonl"
    if not candidates_path.exists():
        return []

    try:
        records = read_jsonl_skipping_corrupt(candidates_path)
    except Exception:
        return []

    ids: list[str] = []
    for r in records:
        status = r.get("status", "auto_pending")
        if status in _UNDER_REVIEW_STATUSES:
            mid = str(r.get("memory_id") or "").strip()
            if mid:
                ids.append(mid)
    return ids


def _one_source(what: str, read: Callable[[], list[str]]) -> list[str]:
    """Run one source of crystallised ids. A failure is logged and costs only that source."""
    try:
        return read()
    except Exception:
        logger.warning(
            "soul-crystallised memory ids: could not read %s; the memories it protects "
            "are NOT exempt this pass",
            what,
            exc_info=True,
        )
        return []


def _accepted_candidate_ids(persona_dir: Path, store: SoulStore | None) -> list[str]:
    """Memory ids of accepted candidates whose crystallization has not been revoked."""
    from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

    candidates_path = persona_dir / "soul_candidates.jsonl"
    if not candidates_path.exists():
        return []
    accepted = [
        r for r in read_jsonl_skipping_corrupt(candidates_path) if r.get("status") == "accepted"
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

    Each source fails on its own: one that cannot be read is logged and contributes nothing,
    so a bad link table cannot take the accepted-candidate protection down, nor an unreadable
    candidates file the direct links. If ``crystallizations.db`` itself cannot be opened the
    error propagates, and the caller (``brain.forgetting._load_soul_linked_ids``) owns the
    fail-soft fallback and logs it.
    """
    db_path = persona_dir / "crystallizations.db"
    with ExitStack() as stack:
        store: SoulStore | None = None
        if db_path.exists():  # opening a SoulStore would create the file
            from brain.soul.store import SoulStore

            store = stack.enter_context(SoulStore(db_path))
        candidate_ids = _one_source(
            "accepted candidates", lambda: _accepted_candidate_ids(persona_dir, store)
        )
        linked = _one_source(
            "direct crystallisation links",
            lambda: store.list_active_source_memory_ids() if store else [],
        )
    return candidate_ids + [mid for mid in linked if mid not in candidate_ids]
