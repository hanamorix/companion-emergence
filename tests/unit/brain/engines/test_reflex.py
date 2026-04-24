"""Unit tests for brain.engines.reflex — types, loaders, scaffold."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.engines.reflex import (
    ArcFire,
    ArcSkipped,
    ReflexArc,
    ReflexArcSet,
    ReflexEngine,
    ReflexLog,
    ReflexResult,
)
from brain.memory.store import MemoryStore

DEFAULT_ARCS_PATH = Path(__file__).parents[4] / "brain" / "engines" / "default_reflex_arcs.json"


def _valid_arc_dict() -> dict:
    return {
        "name": "test_arc",
        "description": "desc",
        "trigger": {"love": 5},
        "days_since_human_min": 0,
        "cooldown_hours": 1.0,
        "action": "generate_journal",
        "output_memory_type": "reflex_journal",
        "prompt_template": "hi {persona_name}",
    }


# ---- ReflexArc ----


def test_reflex_arc_from_dict_valid():
    arc = ReflexArc.from_dict(_valid_arc_dict())
    assert arc.name == "test_arc"
    assert arc.trigger == {"love": 5.0}
    assert arc.cooldown_hours == 1.0


def test_reflex_arc_from_dict_missing_key_raises():
    bad = _valid_arc_dict()
    del bad["trigger"]
    with pytest.raises((KeyError, ValueError)):
        ReflexArc.from_dict(bad)


# ---- ArcSkipped / ReflexResult ----


def test_arc_skipped_and_reflex_result_construction():
    """ArcSkipped and ReflexResult are constructible with expected fields."""
    skip = ArcSkipped(arc_name="test", reason="trigger_not_met")
    assert skip.arc_name == "test"
    assert skip.reason == "trigger_not_met"

    result = ReflexResult(
        arcs_fired=(),
        arcs_skipped=(skip,),
        would_fire=None,
        dry_run=True,
        evaluated_at=datetime.now(UTC),
    )
    assert result.dry_run is True
    assert len(result.arcs_skipped) == 1
    assert result.arcs_skipped[0].arc_name == "test"


# ---- ReflexArcSet ----


def test_reflex_arc_set_load_missing_falls_back_to_defaults(tmp_path: Path):
    missing = tmp_path / "nope.json"
    loaded = ReflexArcSet.load(missing, default_path=DEFAULT_ARCS_PATH)
    assert len(loaded.arcs) == 4
    assert {a.name for a in loaded.arcs} == {
        "creative_pitch",
        "loneliness_journal",
        "self_check",
        "defiance_burst",
    }


def test_reflex_arc_set_load_corrupt_falls_back_to_defaults(tmp_path: Path):
    bad = tmp_path / "arcs.json"
    bad.write_text("not valid json{{{", encoding="utf-8")
    loaded = ReflexArcSet.load(bad, default_path=DEFAULT_ARCS_PATH)
    assert len(loaded.arcs) == 4


def test_reflex_arc_set_load_valid_file(tmp_path: Path):
    path = tmp_path / "arcs.json"
    path.write_text(
        json.dumps({"version": 1, "arcs": [_valid_arc_dict()]}),
        encoding="utf-8",
    )
    loaded = ReflexArcSet.load(path, default_path=DEFAULT_ARCS_PATH)
    assert len(loaded.arcs) == 1
    assert loaded.arcs[0].name == "test_arc"


def test_reflex_arc_set_load_bad_arc_skipped_good_kept(tmp_path: Path):
    path = tmp_path / "arcs.json"
    payload = {
        "version": 1,
        "arcs": [
            _valid_arc_dict(),
            {"name": "broken"},  # missing many required keys
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = ReflexArcSet.load(path, default_path=DEFAULT_ARCS_PATH)
    names = {a.name for a in loaded.arcs}
    assert "test_arc" in names
    assert "broken" not in names


# ---- ReflexLog ----


def test_reflex_log_load_missing_returns_empty(tmp_path: Path):
    log = ReflexLog.load(tmp_path / "nope.json")
    assert log.fires == ()


def test_reflex_log_load_corrupt_returns_empty(tmp_path: Path):
    path = tmp_path / "log.json"
    path.write_text("{{{not json", encoding="utf-8")
    log = ReflexLog.load(path)
    assert log.fires == ()


def test_reflex_log_save_atomic(tmp_path: Path):
    path = tmp_path / "log.json"
    fire = ArcFire(
        arc_name="test_arc",
        fired_at=datetime.now(UTC),
        trigger_state={"love": 6.0},
        output_memory_id="mem-1",
    )
    log = ReflexLog(fires=(fire,))
    log.save(path)
    reloaded = ReflexLog.load(path)
    assert len(reloaded.fires) == 1
    assert reloaded.fires[0].arc_name == "test_arc"
    assert reloaded.fires[0].output_memory_id == "mem-1"


def test_reflex_log_last_fire_for_arc_returns_most_recent(tmp_path: Path):
    now = datetime.now(UTC)
    log = ReflexLog(
        fires=(
            ArcFire(
                arc_name="a",
                fired_at=now - timedelta(hours=5),
                trigger_state={},
                output_memory_id=None,
            ),
            ArcFire(
                arc_name="a",
                fired_at=now - timedelta(hours=1),
                trigger_state={},
                output_memory_id=None,
            ),
            ArcFire(
                arc_name="b",
                fired_at=now - timedelta(hours=2),
                trigger_state={},
                output_memory_id=None,
            ),
        )
    )
    latest = log.last_fire_for_arc("a")
    assert latest is not None
    assert latest == now - timedelta(hours=1)
    assert log.last_fire_for_arc("nonexistent") is None


# ---- Engine scaffold ----


def test_reflex_engine_construction(tmp_path: Path):
    from brain.bridge.provider import FakeProvider

    store = MemoryStore(":memory:")
    try:
        engine = ReflexEngine(
            store=store,
            provider=FakeProvider(),
            persona_name="TestPersona",
            persona_system_prompt="You are TestPersona.",
            arcs_path=tmp_path / "arcs.json",
            log_path=tmp_path / "log.json",
            default_arcs_path=DEFAULT_ARCS_PATH,
        )
        assert engine.persona_name == "TestPersona"
    finally:
        store.close()
