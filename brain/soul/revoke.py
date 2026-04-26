"""revoke_crystallization — move a crystallization to revoked state.

Revocation is permanent in the sense that it cannot be undone programmatically.
The record is soft-deleted (moved to revoked state) rather than dropped —
per OG design, revoked crystallizations stay in the database for audit.

This is Hana's override — the only human-controlled mutation path for the soul.
Brain decisions are autonomous (review.py), human revocations are intentional.
"""

from __future__ import annotations

from brain.soul.crystallization import Crystallization
from brain.soul.store import SoulStore


def revoke_crystallization(
    soul_store: SoulStore,
    id: str,
    reason: str,
) -> Crystallization | None:
    """Move a crystallization from active to revoked.

    Permanent in the audit sense — kept in revoked state (soft delete),
    never physically deleted. Returns the revoked Crystallization, or None
    if not found.

    Parameters
    ----------
    soul_store:
        Open SoulStore instance. Caller is responsible for lifecycle.
    id:
        UUID of the crystallization to revoke.
    reason:
        Human-readable explanation for the revocation.
    """
    return soul_store.mark_revoked(id, reason)
