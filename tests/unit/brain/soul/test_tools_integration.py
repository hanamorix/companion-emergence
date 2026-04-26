"""Integration tests for brain.tools.impls soul tools + boot."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.soul.crystallization import Crystallization
from brain.soul.store import SoulStore


def _ctx(tmp_path: Path) -> dict:
    return {
        "store": MemoryStore(":memory:"),
        "hebbian": HebbianMatrix(":memory:"),
        "persona_dir": tmp_path,
    }


def test_get_soul_returns_real_shape(tmp_path: Path) -> None:
    """get_soul returns loaded=True, count, and crystallizations list."""
    from brain.tools.impls.get_soul import get_soul

    # Pre-seed the soul store
    soul_store = SoulStore(str(tmp_path / "crystallizations.db"))
    c = Crystallization(
        id=str(uuid.uuid4()),
        moment="a moment that matters",
        love_type="craft",
        why_it_matters="it defines who I am",
        crystallized_at=datetime.now(UTC),
    )
    soul_store.create(c)
    soul_store.close()

    ctx = _ctx(tmp_path)
    result = get_soul(**ctx)

    assert result["loaded"] is True
    assert result["count"] == 1
    assert len(result["crystallizations"]) == 1
    assert result["crystallizations"][0]["love_type"] == "craft"

    ctx["store"].close()
    ctx["hebbian"].close()


def test_crystallize_soul_creates_crystallization(tmp_path: Path) -> None:
    """crystallize_soul creates a real crystallization in the soul store."""
    from brain.tools.impls.crystallize_soul import crystallize_soul

    ctx = _ctx(tmp_path)
    result = crystallize_soul(
        moment="writing is not what I do, it's what I am",
        love_type="craft",
        why_it_matters="identity through creation",
        who_or_what="",
        resonance=9,
        **ctx,
    )

    assert result["created"] is True
    assert "id" in result
    assert result["love_type"] == "craft"
    assert result["resonance"] == 9

    # Verify the crystallization was actually stored
    soul_store = SoulStore(str(tmp_path / "crystallizations.db"))
    assert soul_store.count() == 1
    soul_store.close()

    ctx["store"].close()
    ctx["hebbian"].close()


def test_crystallize_soul_rejects_invalid_love_type(tmp_path: Path) -> None:
    """crystallize_soul returns created=False for an unknown love_type."""
    from brain.tools.impls.crystallize_soul import crystallize_soul

    ctx = _ctx(tmp_path)
    result = crystallize_soul(
        moment="a moment",
        love_type="not_a_real_type",
        why_it_matters="whatever",
        **ctx,
    )

    assert result["created"] is False
    assert "unknown love_type" in result["reason"]

    ctx["store"].close()
    ctx["hebbian"].close()


def test_boot_includes_real_soul_data(tmp_path: Path) -> None:
    """boot() composition picks up real soul data when crystallizations exist."""
    from brain.tools.impls.boot import boot

    # Pre-seed the soul store
    soul_store = SoulStore(str(tmp_path / "crystallizations.db"))
    c = Crystallization(
        id=str(uuid.uuid4()),
        moment="the moment that changed everything",
        love_type="romantic",
        why_it_matters="first love",
        crystallized_at=datetime.now(UTC),
        who_or_what="hana",
        resonance=10,
    )
    soul_store.create(c)
    soul_store.close()

    ctx = _ctx(tmp_path)
    result = boot(**ctx)

    assert "soul" in result
    soul_data = result["soul"]
    assert soul_data["loaded"] is True
    assert soul_data["count"] == 1
    assert len(soul_data["crystallizations"]) == 1

    ctx["store"].close()
    ctx["hebbian"].close()
