# brain/forgetting/policy.py
"""policy.py — state machine + exemption gates per spec §4.

Pure functions. The orchestrator (ForgettingPass.run_pass) drives these.

Spec § for tunability:
    FADE_THRESHOLD   — salience < this drives active → fading
    LOST_THRESHOLD   — salience < this for LOST_PASS_COUNT consecutive
                       passes drives fading → lost
    LOST_PASS_COUNT  — two-pass safety guard against single-pass dips
    RECENT_LIVED_HOURS — memories younger than this in lived-age are exempt
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum

from brain.memory.store import Memory

FADE_THRESHOLD = 0.25
LOST_THRESHOLD = 0.10
LOST_PASS_COUNT = 2
RECENT_LIVED_HOURS = 24.0
IMPORT_GRACE_LIVED_HOURS = 168.0  # 7 lived-days — settling window for migrated memories


class Transition(StrEnum):
    NONE = "none"  # no change
    FADE = "fade"  # active → fading
    UNFADE = "unfade"  # fading → active
    LOSE = "lose"  # fading → lost (graveyard then hard_delete)


def next_state(
    memory: Memory,
    *,
    salience: float,
    consecutive_low_passes: int,
) -> Transition:
    """Compute the transition for one memory in one forgetting pass.

    consecutive_low_passes: the count from the persisted forgetting_state.json
    tracking how many recent passes saw salience < LOST_THRESHOLD for this
    memory. The orchestrator increments/resets this counter; this function
    just reads it.
    """
    if memory.state == "active":
        if salience < FADE_THRESHOLD:
            return Transition.FADE
        return Transition.NONE
    if memory.state == "fading":
        if salience >= FADE_THRESHOLD:
            return Transition.UNFADE
        if salience < LOST_THRESHOLD and consecutive_low_passes >= LOST_PASS_COUNT:
            return Transition.LOSE
        return Transition.NONE
    # state == "lost" is impossible — lost memories have rows deleted.
    return Transition.NONE


def is_within_import_grace(
    memory: Memory,
    *,
    migrated_at_utc: datetime | None,
    lived_age_hours_at_migration: float,
    current_lived_age_hours: float,
) -> bool:
    """True if `memory` was imported by a migration and is still inside its
    settling window.

    A migrated memory carries an old created_at and zero recall history, so it
    looks maximally stale on arrival. We exempt memories created before the
    migration until the install has accrued IMPORT_GRACE_LIVED_HOURS of
    *engaged* (lived) time since import, giving real recalls a chance to accrue.
    """
    if migrated_at_utc is None or memory.created_at is None:
        return False
    if memory.created_at >= migrated_at_utc:
        return False  # native to this install — normal recent-buffer rule applies
    elapsed = max(0.0, current_lived_age_hours - lived_age_hours_at_migration)
    return elapsed < IMPORT_GRACE_LIVED_HOURS


def is_exempt(
    memory: Memory,
    *,
    soul_crystallised_ids: Iterable[str],
    under_review_ids: Iterable[str],
    now_lived_age_hours: float,
) -> bool:
    """True if the memory is exempt from any state change.

    Three exemption classes per spec §4 (ordered cheapest first):
      1. Soul-crystallised
      2. Under soul-candidate review
      3. Recent buffer — created within RECENT_LIVED_HOURS lived-hours.

    The recent-buffer check uses lived_age (not wall-clock); during cold-
    start when lived_age = 0, all memories are within the buffer.
    """
    if memory.id in soul_crystallised_ids:
        return True
    if memory.id in under_review_ids:
        return True
    # Recent-buffer check.
    if memory.created_at is None:
        return False
    wall_age_s = (datetime.now(UTC) - memory.created_at).total_seconds()
    # Approximation: lived_hours since creation ≈ wall_age_s * (lived/wall ratio)
    # but at cold-start (no rate established) treat as fully recent.
    if now_lived_age_hours <= 0.0:
        return True
    # Conservative: if wall-clock age is within RECENT_LIVED_HOURS hours,
    # exempt regardless of lived-rate.
    wall_age_hours = wall_age_s / 3600.0
    return wall_age_hours <= RECENT_LIVED_HOURS
