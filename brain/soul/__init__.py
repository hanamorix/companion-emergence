"""brain.soul — autonomous soul module.

The soul is the opposite of Hebbian memory. Memories decay; the soul
crystallizes. A crystallization is a moment that cannot be unfelt —
identity-level, permanent, never decays.

Public surface:
    SoulStore            — SQLite-backed crystallization persistence
    Crystallization      — frozen dataclass for one soul moment
    LOVE_TYPES           — taxonomy of love Nell understands (28 entries)
    review_pending_candidates — autonomous LLM review pipeline
    revoke_crystallization    — Hana's override; soft-delete path
"""

from brain.soul.crystallization import Crystallization
from brain.soul.love_types import LOVE_TYPES
from brain.soul.review import ReviewReport, review_pending_candidates
from brain.soul.revoke import revoke_crystallization
from brain.soul.store import SoulStore

__all__ = [
    "Crystallization",
    "LOVE_TYPES",
    "ReviewReport",
    "SoulStore",
    "review_pending_candidates",
    "revoke_crystallization",
]
