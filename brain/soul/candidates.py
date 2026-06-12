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
