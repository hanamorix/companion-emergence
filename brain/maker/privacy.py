"""brain.maker.privacy — the disposition gate (LOAD-BEARING).

A private or discard making must NEVER reach the user through any AUTOMATIC
surface (feed, outbound reach, volunteered ambient). The ONLY path a private
making's content reaches the user is her DELIBERATE choice in the disclosure
turn (Task 16). Every automatic read path consults is_auto_surfaceable().
"""
from __future__ import annotations

from brain.works import Work

_AUTO_SURFACEABLE = {"eventual_share"}


def is_auto_surfaceable(work: Work) -> bool:
    """True only if this making may surface automatically (shared + ready)."""
    if work.disposition not in _AUTO_SURFACEABLE:
        return False
    return work.shared_at is not None


def is_disclosable_on_request(work: Work) -> bool:
    """All non-discard makings are available to HER for deliberate disclosure;
    she chooses per-item to share or decline-with-reason. (discard has no content.)"""
    return work.disposition != "discard"
