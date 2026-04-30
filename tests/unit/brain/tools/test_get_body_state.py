"""Unit tests for the real get_body_state tool impl.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md §3.3.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools.impls.get_body_state import get_body_state


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


@pytest.fixture
def hebbian(tmp_path: Path) -> HebbianMatrix:
    return HebbianMatrix(tmp_path / "hebbian.db")


def test_returns_loaded_true(store, hebbian, tmp_path):
    out = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
    assert out["loaded"] is True


def test_returns_real_schema(store, hebbian, tmp_path):
    out = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
    assert set(out.keys()) == {
        "loaded", "energy", "temperature", "exhaustion",
        "session_hours", "days_since_contact", "body_emotions", "computed_at",
    }
    assert isinstance(out["energy"], int)
    assert isinstance(out["temperature"], int)
    assert isinstance(out["exhaustion"], int)
    assert set(out["body_emotions"].keys()) == {
        "arousal", "desire", "climax",
        "touch_hunger", "comfort_seeking", "rest_need",
    }


def test_baseline_energy_when_empty(store, hebbian, tmp_path):
    out = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
    assert out["energy"] == 8


def test_session_hours_default_zero(store, hebbian, tmp_path):
    """No session_hours kwarg → 0.0 (CLI mode)."""
    out = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
    assert out["session_hours"] == 0.0


def test_session_hours_passed_through(store, hebbian, tmp_path):
    out = get_body_state(
        store=store, hebbian=hebbian, persona_dir=tmp_path, session_hours=2.5,
    )
    assert out["session_hours"] == 2.5


def test_recomputes_each_call(store, hebbian, tmp_path):
    """Inviolate property #8 from spec §7.1 — no cache."""
    out1 = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
    out2 = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
    assert out1["computed_at"] != out2["computed_at"]
