"""Tests for brain.soul.store.SoulStore."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.soul.crystallization import Crystallization
from brain.soul.store import SoulStore


def _make_crystal(
    love_type: str = "romantic",
    resonance: int = 9,
    moment: str = "a test moment",
) -> Crystallization:
    return Crystallization(
        id=str(uuid.uuid4()),
        moment=moment,
        love_type=love_type,
        why_it_matters="because it matters",
        crystallized_at=datetime.now(UTC),
        who_or_what="hana",
        resonance=resonance,
    )


def _make_store() -> SoulStore:
    return SoulStore(":memory:")


def test_create_and_get_roundtrip() -> None:
    """create + get returns the same crystallization."""
    store = _make_store()
    c = _make_crystal()
    store.create(c)

    fetched = store.get(c.id)
    assert fetched is not None
    assert fetched.id == c.id
    assert fetched.moment == c.moment
    assert fetched.love_type == c.love_type
    assert fetched.resonance == c.resonance
    assert fetched.revoked_at is None
    store.close()


def test_list_active_excludes_revoked() -> None:
    """list_active does not include revoked crystallizations."""
    store = _make_store()
    active = _make_crystal(moment="active moment")
    revoked = _make_crystal(moment="revoked moment")
    store.create(active)
    store.create(revoked)
    store.mark_revoked(revoked.id, "test revocation")

    active_list = store.list_active()
    ids = [c.id for c in active_list]
    assert active.id in ids
    assert revoked.id not in ids
    store.close()


def test_list_revoked_includes_only_revoked() -> None:
    """list_revoked returns only the revoked crystallization."""
    store = _make_store()
    active = _make_crystal(moment="still active")
    revoked = _make_crystal(moment="was revoked")
    store.create(active)
    store.create(revoked)
    store.mark_revoked(revoked.id, "reason")

    revoked_list = store.list_revoked()
    ids = [c.id for c in revoked_list]
    assert revoked.id in ids
    assert active.id not in ids
    store.close()


def test_mark_revoked_moves_entry() -> None:
    """mark_revoked sets revoked_at and moves entry out of active list."""
    store = _make_store()
    c = _make_crystal()
    store.create(c)

    result = store.mark_revoked(c.id, "no longer relevant")
    assert result is not None
    assert result.revoked_at is not None
    assert result.revoked_reason == "no longer relevant"

    # Subsequent list_active should not return it
    active = store.list_active()
    assert all(x.id != c.id for x in active)
    store.close()


def test_count_returns_active_only() -> None:
    """count() reflects only active (non-revoked) crystallizations."""
    store = _make_store()
    c1 = _make_crystal(moment="one")
    c2 = _make_crystal(moment="two")
    c3 = _make_crystal(moment="three")
    for c in (c1, c2, c3):
        store.create(c)
    store.mark_revoked(c3.id, "test")

    assert store.count() == 2
    store.close()


def test_integrity_check_raises_on_corrupt_db(tmp_path: Path) -> None:
    """SoulStore raises BrainIntegrityError on corrupt SQLite file."""
    from brain.health.anomaly import BrainIntegrityError

    corrupt_path = tmp_path / "corrupt.db"
    corrupt_path.write_bytes(b"this is not a sqlite file at all")

    with pytest.raises(BrainIntegrityError):
        SoulStore(str(corrupt_path))


def test_save_and_list_voice_evolution(tmp_path: Path) -> None:
    """save_voice_evolution persists a record; list_voice_evolution returns it."""
    from brain.soul.store import VoiceEvolution

    store = SoulStore(str(tmp_path / "crystallizations.db"))
    try:
        evolution = VoiceEvolution(
            id="ve_001",
            accepted_at="2026-05-11T14:32:04+00:00",
            diff="- old\n+ new",
            old_text="old",
            new_text="new",
            rationale="feels truer",
            evidence=["dream_a", "cryst_b"],
            audit_id="ia_001",
            user_modified=False,
        )
        store.save_voice_evolution(evolution)
        retrieved = store.list_voice_evolution()
        assert len(retrieved) == 1
        assert retrieved[0].id == "ve_001"
        assert retrieved[0].evidence == ["dream_a", "cryst_b"]
    finally:
        store.close()


def test_list_voice_evolution_chronological_order(tmp_path: Path) -> None:
    """list_voice_evolution returns records ordered by accepted_at ascending."""
    from brain.soul.store import VoiceEvolution

    store = SoulStore(str(tmp_path / "crystallizations.db"))
    try:
        for i, ts in enumerate(
            [
                "2026-01-01T00:00:00+00:00",
                "2026-03-15T00:00:00+00:00",
                "2026-05-11T00:00:00+00:00",
            ]
        ):
            store.save_voice_evolution(
                VoiceEvolution(
                    id=f"ve_{i}",
                    accepted_at=ts,
                    diff="",
                    old_text="",
                    new_text="",
                    rationale="",
                    evidence=[],
                    audit_id=f"ia_{i}",
                    user_modified=False,
                )
            )
        retrieved = store.list_voice_evolution()
        assert [v.id for v in retrieved] == ["ve_0", "ve_1", "ve_2"]
    finally:
        store.close()
