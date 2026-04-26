"""Tests for brain.soul.revoke."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from brain.soul.crystallization import Crystallization
from brain.soul.revoke import revoke_crystallization
from brain.soul.store import SoulStore


def _make_store() -> SoulStore:
    return SoulStore(":memory:")


def _make_crystal(moment: str = "a permanent moment") -> Crystallization:
    return Crystallization(
        id=str(uuid.uuid4()),
        moment=moment,
        love_type="romantic",
        why_it_matters="because it matters",
        crystallized_at=datetime.now(UTC),
    )


def test_revoke_crystallization_moves_entry() -> None:
    """revoke_crystallization moves the entry to revoked state."""
    store = _make_store()
    c = _make_crystal()
    store.create(c)
    assert store.count() == 1

    revoked = revoke_crystallization(store, c.id, "it no longer resonates")
    assert revoked is not None
    assert revoked.revoked_at is not None
    assert revoked.revoked_reason == "it no longer resonates"
    assert store.count() == 0

    revoked_list = store.list_revoked()
    assert len(revoked_list) == 1
    assert revoked_list[0].id == c.id

    store.close()


def test_revoke_crystallization_returns_none_for_unknown_id() -> None:
    """revoke_crystallization returns None if the id is not found."""
    store = _make_store()

    result = revoke_crystallization(store, "nonexistent-uuid", "whatever")
    assert result is None

    store.close()
