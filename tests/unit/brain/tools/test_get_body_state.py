"""Unit tests for the real get_body_state tool impl.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md §3.3.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.tools.impls import get_body_state as body_tool
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
        "loaded",
        "energy",
        "temperature",
        "exhaustion",
        "session_hours",
        "days_since_contact",
        "body_emotions",
        "computed_at",
    }
    assert isinstance(out["energy"], int)
    assert isinstance(out["temperature"], int)
    assert isinstance(out["exhaustion"], int)
    assert set(out["body_emotions"].keys()) == {
        "arousal",
        "desire",
        "climax",
        "touch_hunger",
        "comfort_seeking",
        "rest_need",
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
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
        session_hours=2.5,
    )
    assert out["session_hours"] == 2.5


def test_body_state_reads_a_variant_spelled_stored_emotion(store, hebbian, tmp_path):
    """A feeling stored as `comfort-seeking` shows up as comfort_seeking in body state (#557)."""
    memory = Memory.create_new(
        content="held",
        memory_type="conversation",
        domain="us",
        emotions={"comfort-seeking": 7.0},
    )
    store.create(memory)

    out = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)

    assert out["body_emotions"]["comfort_seeking"] == 7.0
    # Reading must not rewrite the stored spelling.
    assert store.get(memory.id).emotions == {"comfort-seeking": 7.0}


def test_body_state_keeps_the_in_range_spelling_when_another_is_out_of_range(
    store, hebbian, tmp_path
):
    """A dream-sized value in one spelling must not hide the in-range spellings of the emotion (#557)."""
    stored = {
        "comfort-seeking": 12.0,
        "comfort_seeking": 5.0,
        "Comfort_Seeking": 6.0,
    }
    ids = {}
    for spelling, value in stored.items():
        memory = Memory.create_new(
            content=f"held {spelling}",
            memory_type="conversation",
            domain="us",
            emotions={spelling: value},
        )
        store.create(memory)
        ids[spelling] = memory.id

    out = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)

    assert out["body_emotions"]["comfort_seeking"] == 6.0
    # Reading must not rewrite any stored spelling or value.
    for spelling, value in stored.items():
        assert store.get(ids[spelling]).emotions == {spelling: value}


def test_recomputes_each_call(store, hebbian, tmp_path, monkeypatch):
    """Inviolate property #8 from spec §7.1 — no cache."""
    calls = 0
    original = body_tool.compute_body_state

    def counting_compute_body_state(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(body_tool, "compute_body_state", counting_compute_body_state)

    out1 = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
    out2 = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)

    assert calls == 2
    assert "computed_at" in out1
    assert "computed_at" in out2


def test_live_body_state_has_no_key_for_an_absent_body_emotion(store, hebbian, tmp_path, emotion_engine_stub):
    """B2 phase B: when the engine is live the body emotions are the engine's short averages, and only those."""
    from brain.emotion.engine import Reading

    store.create(
        Memory.create_new(
            content="stored", memory_type="fact", domain="us", emotions={"rest_need": 9.0, "arousal": 9.0}
        )
    )
    emotion_engine_stub.live = True
    emotion_engine_stub.readings = {"arousal": Reading(short=4.0, long=0.0, difference=0.0, shift=0)}
    out = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
    assert out["body_emotions"] == {"arousal": 4.0}


def test_live_energy_and_temperature_follow_the_engines_predicate_not_a_level(
    store, hebbian, tmp_path, emotion_engine_stub
):
    from brain.emotion.engine import Reading

    def numbers(readings: dict[str, float], high: set[str]) -> tuple[int, int]:
        emotion_engine_stub.live = True
        emotion_engine_stub.high = set(high)
        emotion_engine_stub.readings = {
            n: Reading(short=v, long=0.0, difference=0.0, shift=0) for n, v in readings.items()
        }
        out = get_body_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
        return out["energy"], out["temperature"]

    energy, temperature = numbers({}, set())
    top = {"rest_need": 10.0, "desire": 10.0, "love": 10.0, "touch_hunger": 10.0}
    assert numbers(top, set()) == (energy, temperature)
    low = {"rest_need": 0.5, "desire": 0.5, "love": 0.5, "touch_hunger": 0.5}
    assert numbers(low, set(low)) == (energy - 1, temperature + 1)
