"""brain.soul.candidates — soul candidate utility functions.

Provides helpers consumed by brain.forgetting to determine which memory IDs
are currently protected from the forgetting pass by virtue of being linked
to an active (under-review) soul candidate.
"""

from __future__ import annotations

from pathlib import Path

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


def _revoked_crystallization_ids(persona_dir: Path) -> set[str]:
    """Ids of revoked crystallizations. Opens crystallizations.db only if it exists
    (SoulStore would create it) — an absent store simply has nothing revoked."""
    db_path = persona_dir / "crystallizations.db"
    if not db_path.exists():
        return set()

    from brain.soul.store import SoulStore

    with SoulStore(db_path) as store:
        return {c.id for c in store.list_revoked()}


def list_crystallised_memory_ids(persona_dir: Path) -> list[str]:
    """Return the memory_id of every accepted candidate whose crystallization is live.

    Accepting a candidate crystallises it (``review._apply_accept``) and keeps its
    ``memory_id`` on the candidate record; that record is the only link from a
    crystallisation back to a memory, so this is where "soul-crystallised
    memories are exempt from forgetting" gets its ids. (Crystallizations made by
    ``crystallize_soul`` or the migrator have no candidate and no memory link.)

    A revoked crystallization stops protecting its memory (#362 ruling) —
    ``revoke_crystallization`` only touches crystallizations.db, the candidate
    stays ``accepted``, so the store is consulted. Only an affirmative revoke
    drops protection: a legacy record with no ``crystallization_id`` or a
    dangling one stays protected, because losing a memory is the worse error.

    Unlike ``list_under_review_memory_ids`` this does not swallow read errors:
    the caller (``brain.forgetting._load_soul_linked_ids``) owns the fail-soft
    fallback and logs it.
    """
    from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

    candidates_path = persona_dir / "soul_candidates.jsonl"
    if not candidates_path.exists():
        return []

    accepted = [
        r for r in read_jsonl_skipping_corrupt(candidates_path) if r.get("status") == "accepted"
    ]
    if not accepted:
        return []

    revoked = _revoked_crystallization_ids(persona_dir)
    ids: list[str] = []
    for r in accepted:
        if str(r.get("crystallization_id") or "") in revoked:
            continue
        mid = str(r.get("memory_id") or "").strip()
        if mid:
            ids.append(mid)
    return ids
