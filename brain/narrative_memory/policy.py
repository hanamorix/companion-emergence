"""Arc open/close gates + module-top tunable constants.

Spec §2 §4 §5. Constants here are intentionally duplicated from
membership.py for grep-ability — the membership thresholds are
clustered with the per-call function, the lifecycle thresholds with
the per-pass decisions.
"""

from __future__ import annotations

from brain.narrative_memory.arc import Arc

# Spec §2 module-top constants
ARC_STALE_LIVED_HOURS: float = 72.0
MAX_ARC_MEMBERS: int = 50


def should_close(
    arc: Arc,
    *,
    lived_age_now: float | None,
    last_extended_lived_age: float,
) -> bool:
    """True iff arc's last extension was >=ARC_STALE_LIVED_HOURS lived-hours ago.

    `lived_age_now` is `FeltTimeState.lived_age_hours` at pass time.
    `last_extended_lived_age` is the lived-age snapshot recorded on the most
    recent member_added event for this arc.

    If `lived_age_now` is None (FeltTimeState absent during cold start), the
    pass logs a warning and skips closure for this tick — wall-clock-only
    closure is intentionally NOT implemented to keep all arc lifecycle math
    bound to experiential time (spec §1 cross-cluster invariant).
    """
    if lived_age_now is None:
        return False
    return (lived_age_now - last_extended_lived_age) >= ARC_STALE_LIVED_HOURS


def should_open(
    seed_memory_ids: tuple[str, ...],
    *,
    open_arcs: dict[str, Arc],
) -> bool:
    """True iff none of the seed memory ids are already members of an open arc.

    The anchor-sweep step uses this to decide: extend an existing arc vs.
    open a fresh one. When False, the caller appends the seed memories as
    members of the matching open arc instead.
    """
    existing_members: set[str] = set()
    for arc in open_arcs.values():
        existing_members.update(m.memory_id for m in arc.members)
    return not any(sid in existing_members for sid in seed_memory_ids)
