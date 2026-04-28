"""Unit tests for brain.engines.reflex — types, loaders, scaffold."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.provider import FakeProvider
from brain.engines.reflex import (
    ArcFire,
    ArcSkipped,
    ReflexArc,
    ReflexArcSet,
    ReflexEngine,
    ReflexLog,
    ReflexResult,
)
from brain.memory.store import Memory, MemoryStore


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not find pyproject.toml above {here}")


DEFAULT_ARCS_PATH = _find_repo_root() / "brain" / "engines" / "default_reflex_arcs.json"


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


# ---- Health T10: attempt_heal wiring ----


def test_reflex_arc_set_load_corrupt_quarantines_restores_from_bak(tmp_path: Path):
    """Corrupt live arcs file + valid .bak1 → restore .bak1, return its arcs, anomaly set."""
    path = tmp_path / "reflex_arcs.json"
    bak1 = tmp_path / "reflex_arcs.json.bak1"

    bak1.write_text(json.dumps({"version": 1, "arcs": [_valid_arc_dict()]}), encoding="utf-8")
    path.write_text("{corrupt{{", encoding="utf-8")

    result, anomaly = ReflexArcSet.load_with_anomaly(path, default_path=DEFAULT_ARCS_PATH)

    assert anomaly is not None
    assert "bak1" in anomaly.action
    assert len(result.arcs) == 1
    assert result.arcs[0].name == "test_arc"
    corrupt_files = list(tmp_path.glob("reflex_arcs.json.corrupt-*"))
    assert len(corrupt_files) == 1


def test_reflex_arc_set_load_corrupt_no_bak_uses_defaults(tmp_path: Path):
    """Corrupt arcs file + no .bak → falls back to default_path arcs, anomaly with reset_to_default."""
    path = tmp_path / "reflex_arcs.json"
    path.write_text("{corrupt{{", encoding="utf-8")

    result, anomaly = ReflexArcSet.load_with_anomaly(path, default_path=DEFAULT_ARCS_PATH)

    assert anomaly is not None
    assert anomaly.action == "reset_to_default"
    # default_path arcs loaded (4 OG arcs)
    assert len(result.arcs) == 4


def test_reflex_log_save_with_backup_creates_bak(tmp_path: Path):
    """save() via save_with_backup creates .bak1 on second write."""
    path = tmp_path / "reflex_log.json"
    fire = ArcFire(
        arc_name="test_arc",
        fired_at=datetime.now(UTC),
        trigger_state={"love": 6.0},
        output_memory_id="mem-1",
    )
    log = ReflexLog(fires=(fire,))
    log.save(path)
    log.save(path)  # second write should rotate the first into .bak1
    bak1 = tmp_path / "reflex_log.json.bak1"
    assert bak1.exists()


# ---- ReflexLog ----


def test_reflex_log_load_missing_returns_empty(tmp_path: Path):
    log = ReflexLog.load(tmp_path / "nope.json")
    assert log.fires == ()


def test_reflex_log_load_corrupt_returns_empty(tmp_path: Path):
    path = tmp_path / "log.json"
    path.write_text("{{{not json", encoding="utf-8")
    log = ReflexLog.load(path)
    assert log.fires == ()


def test_reflex_log_load_corrupt_quarantines_and_warns(tmp_path: Path, caplog) -> None:
    """Corrupt file is quarantined and WARNING is emitted (attempt_heal path)."""
    import logging

    caplog.set_level(logging.WARNING)
    path = tmp_path / "log.json"
    path.write_text("{{{not json", encoding="utf-8")
    log = ReflexLog.load(path)
    assert log.fires == ()
    # Quarantine file should have been created
    corrupt_files = list(tmp_path.glob("log.json.corrupt-*"))
    assert len(corrupt_files) == 1
    # Warning logged
    warn_msgs = [r.getMessage() for r in caplog.records if "ReflexLog anomaly" in r.getMessage()]
    assert len(warn_msgs) == 1


def test_reflex_log_load_heals_from_bak(tmp_path: Path) -> None:
    """When primary file is corrupt, load restores from .bak1."""
    fire = ArcFire(
        arc_name="restored_arc",
        fired_at=datetime.now(UTC),
        trigger_state={},
        output_memory_id=None,
    )
    good_payload = {"version": 1, "fires": [fire.to_dict()]}
    path = tmp_path / "log.json"
    bak1 = tmp_path / "log.json.bak1"
    import json

    bak1.write_text(json.dumps(good_payload), encoding="utf-8")
    path.write_text("{{{not json", encoding="utf-8")  # corrupt primary
    log = ReflexLog.load(path)
    assert len(log.fires) == 1
    assert log.fires[0].arc_name == "restored_arc"


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


# ---- run_tick ----


def _build_engine(tmp_path: Path, store: MemoryStore) -> ReflexEngine:
    return ReflexEngine(
        store=store,
        provider=FakeProvider(),
        persona_name="Nell",
        persona_system_prompt="You are Nell.",
        arcs_path=tmp_path / "arcs.json",
        log_path=tmp_path / "log.json",
        default_arcs_path=DEFAULT_ARCS_PATH,
    )


def _write_single_arc(
    path: Path,
    *,
    trigger: dict,
    cooldown_hours: float = 1.0,
    days_since_human_min: float = 0.0,
    name: str = "test_arc",
) -> None:
    arc = {
        "name": name,
        "description": "test",
        "trigger": trigger,
        "days_since_human_min": days_since_human_min,
        "cooldown_hours": cooldown_hours,
        "action": "generate_journal",
        "output_memory_type": "reflex_journal",
        "prompt_template": "You are {persona_name}. Write something.",
    }
    path.write_text(json.dumps({"version": 1, "arcs": [arc]}, indent=2), encoding="utf-8")


def _seed_emotion_memory(store: MemoryStore, emotions: dict[str, float]) -> str:
    mem = Memory.create_new(
        content="seed",
        memory_type="conversation",
        domain="us",
        emotions=emotions,
    )
    store.create(mem)
    return mem.id


def test_run_tick_returns_no_arcs_defined_when_empty(tmp_path: Path):
    path = tmp_path / "arcs.json"
    path.write_text(json.dumps({"version": 1, "arcs": []}), encoding="utf-8")

    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        engine.arcs_path = path
        result = engine.run_tick(trigger="manual", dry_run=False)
        assert result.arcs_fired == ()
        assert len(result.arcs_skipped) == 1
        assert result.arcs_skipped[0].reason == "no_arcs_defined"
    finally:
        store.close()


def test_run_tick_skips_when_trigger_not_met(tmp_path: Path):
    arcs_path = tmp_path / "arcs.json"
    _write_single_arc(arcs_path, trigger={"love": 8})

    store = MemoryStore(":memory:")
    try:
        _seed_emotion_memory(store, {"love": 2.0})
        engine = _build_engine(tmp_path, store)
        engine.arcs_path = arcs_path
        result = engine.run_tick(dry_run=False)
        assert result.arcs_fired == ()
        assert any(s.reason == "trigger_not_met" for s in result.arcs_skipped)
    finally:
        store.close()


def test_run_tick_fires_arc_when_trigger_met(tmp_path: Path):
    arcs_path = tmp_path / "arcs.json"
    _write_single_arc(arcs_path, trigger={"love": 5})

    store = MemoryStore(":memory:")
    try:
        _seed_emotion_memory(store, {"love": 8.0})
        engine = _build_engine(tmp_path, store)
        engine.arcs_path = arcs_path
        result = engine.run_tick(dry_run=False)
        assert len(result.arcs_fired) == 1
        assert result.arcs_fired[0].arc_name == "test_arc"
        # Memory was written
        mem = store.get(result.arcs_fired[0].output_memory_id)
        assert mem is not None
        assert mem.memory_type == "reflex_journal"
    finally:
        store.close()


def test_run_tick_dry_run_reports_would_fire(tmp_path: Path):
    arcs_path = tmp_path / "arcs.json"
    _write_single_arc(arcs_path, trigger={"love": 5})

    store = MemoryStore(":memory:")
    try:
        _seed_emotion_memory(store, {"love": 8.0})
        engine = _build_engine(tmp_path, store)
        engine.arcs_path = arcs_path
        result = engine.run_tick(dry_run=True)
        assert result.dry_run is True
        assert result.would_fire == "test_arc"
        assert result.arcs_fired == ()
        # No memory written beyond the seed
        assert store.count() == 1
        # No log file written
        assert not (tmp_path / "log.json").exists()
    finally:
        store.close()


def test_run_tick_respects_cooldown(tmp_path: Path):
    arcs_path = tmp_path / "arcs.json"
    _write_single_arc(arcs_path, trigger={"love": 5}, cooldown_hours=24.0)

    # Pre-populate log with a recent fire
    log_path = tmp_path / "log.json"
    recent = datetime.now(UTC) - timedelta(hours=1)
    log = ReflexLog(
        fires=(
            ArcFire(
                arc_name="test_arc",
                fired_at=recent,
                trigger_state={"love": 8.0},
                output_memory_id="prev",
            ),
        )
    )
    log.save(log_path)

    store = MemoryStore(":memory:")
    try:
        _seed_emotion_memory(store, {"love": 8.0})
        engine = _build_engine(tmp_path, store)
        engine.arcs_path = arcs_path
        result = engine.run_tick(dry_run=False)
        assert result.arcs_fired == ()
        assert any(s.reason == "cooldown_active" for s in result.arcs_skipped)
    finally:
        store.close()


def test_run_tick_ranks_highest_threshold_excess(tmp_path: Path):
    # Two eligible arcs; the one whose trigger is most exceeded wins.
    arcs_path = tmp_path / "arcs.json"
    payload = {
        "version": 1,
        "arcs": [
            {
                "name": "low_excess",
                "description": "d",
                "trigger": {"love": 5},
                "days_since_human_min": 0,
                "cooldown_hours": 1.0,
                "action": "a",
                "output_memory_type": "reflex_journal",
                "prompt_template": "t",
            },
            {
                "name": "high_excess",
                "description": "d",
                "trigger": {"defiance": 3},
                "days_since_human_min": 0,
                "cooldown_hours": 1.0,
                "action": "a",
                "output_memory_type": "reflex_journal",
                "prompt_template": "t",
            },
        ],
    }
    arcs_path.write_text(json.dumps(payload), encoding="utf-8")

    store = MemoryStore(":memory:")
    try:
        _seed_emotion_memory(store, {"love": 6.0, "defiance": 9.0})
        engine = _build_engine(tmp_path, store)
        engine.arcs_path = arcs_path
        result = engine.run_tick(dry_run=False)
        # love excess = 6-5 = 1; defiance excess = 9-3 = 6 — defiance wins
        assert len(result.arcs_fired) == 1
        assert result.arcs_fired[0].arc_name == "high_excess"
        assert any(
            s.arc_name == "low_excess" and s.reason == "single_fire_cap"
            for s in result.arcs_skipped
        )
    finally:
        store.close()


def test_run_tick_llm_failure_does_not_poison_cooldown(tmp_path: Path):
    arcs_path = tmp_path / "arcs.json"
    _write_single_arc(arcs_path, trigger={"love": 5})

    class FailingProvider(FakeProvider):
        def generate(self, prompt, *, system=None):
            raise RuntimeError("simulated LLM failure")

    store = MemoryStore(":memory:")
    try:
        _seed_emotion_memory(store, {"love": 8.0})
        engine = ReflexEngine(
            store=store,
            provider=FailingProvider(),
            persona_name="Nell",
            persona_system_prompt="",
            arcs_path=arcs_path,
            log_path=tmp_path / "log.json",
            default_arcs_path=DEFAULT_ARCS_PATH,
        )
        with pytest.raises(RuntimeError):
            engine.run_tick(dry_run=False)
        # Log file NOT written — next tick can retry
        assert not (tmp_path / "log.json").exists()
    finally:
        store.close()


def test_run_tick_template_missing_key_substitutes_zero(tmp_path: Path):
    arcs_path = tmp_path / "arcs.json"
    arc = {
        "name": "test_arc",
        "description": "d",
        "trigger": {"love": 5},
        "days_since_human_min": 0,
        "cooldown_hours": 1.0,
        "action": "a",
        "output_memory_type": "reflex_journal",
        "prompt_template": "Unknown: {undefined_var} Love: {love}",
    }
    arcs_path.write_text(json.dumps({"version": 1, "arcs": [arc]}), encoding="utf-8")

    captured: list[str] = []

    class CapturingProvider(FakeProvider):
        def generate(self, prompt, *, system=None):
            captured.append(prompt)
            return "ok"

    store = MemoryStore(":memory:")
    try:
        _seed_emotion_memory(store, {"love": 7.0})
        engine = ReflexEngine(
            store=store,
            provider=CapturingProvider(),
            persona_name="Nell",
            persona_system_prompt="",
            arcs_path=arcs_path,
            log_path=tmp_path / "log.json",
            default_arcs_path=DEFAULT_ARCS_PATH,
        )
        result = engine.run_tick(dry_run=False)
        assert len(result.arcs_fired) == 1
        assert captured[0] == "Unknown: 0 Love: 7.0"
    finally:
        store.close()


def test_run_tick_days_since_human_gate(tmp_path: Path):
    arcs_path = tmp_path / "arcs.json"
    _write_single_arc(arcs_path, trigger={"love": 5}, days_since_human_min=5.0)

    store = MemoryStore(":memory:")
    try:
        # Recent conversation memory — days_since_human ~0
        _seed_emotion_memory(store, {"love": 8.0})
        engine = _build_engine(tmp_path, store)
        engine.arcs_path = arcs_path
        result = engine.run_tick(dry_run=False)
        assert result.arcs_fired == ()
        assert any(s.reason == "days_since_human_too_low" for s in result.arcs_skipped)
    finally:
        store.close()


# ---- ReflexArc provenance (Phase 2) ----


def test_reflex_arc_has_created_by_field():
    arc = ReflexArc(
        name="creative_pitch",
        description="creative hunger overwhelmed",
        trigger={"creative_hunger": 8.0},
        days_since_human_min=0.0,
        cooldown_hours=48.0,
        action="generate_pitch",
        output_memory_type="reflex_pitch",
        prompt_template="...",
        created_by="brain_emergence",
        created_at=datetime(2026, 4, 28, tzinfo=UTC),
    )
    assert arc.created_by == "brain_emergence"
    assert arc.created_at == datetime(2026, 4, 28, tzinfo=UTC)


def test_reflex_arc_from_dict_backward_compat_no_created_by():
    """Loading an arc from old persona file (pre-Phase-2): missing created_by
    defaults to 'og_migration', missing created_at defaults to epoch sentinel."""
    arc = ReflexArc.from_dict({
        "name": "x",
        "description": "y",
        "trigger": {"e": 5.0},
        "days_since_human_min": 0.0,
        "cooldown_hours": 12.0,
        "action": "z",
        "output_memory_type": "reflex_x",
        "prompt_template": "t",
    })
    assert arc.created_by == "og_migration"
    assert arc.created_at == datetime(1970, 1, 1, tzinfo=UTC)


def test_reflex_arc_from_dict_with_created_by():
    arc = ReflexArc.from_dict({
        "name": "x",
        "description": "y",
        "trigger": {"e": 5.0},
        "days_since_human_min": 0.0,
        "cooldown_hours": 12.0,
        "action": "z",
        "output_memory_type": "reflex_x",
        "prompt_template": "t",
        "created_by": "brain_emergence",
        "created_at": "2026-04-28T10:00:00+00:00",
    })
    assert arc.created_by == "brain_emergence"
    assert arc.created_at == datetime(2026, 4, 28, 10, 0, 0, tzinfo=UTC)


def test_reflex_arc_from_dict_rejects_invalid_created_by():
    with pytest.raises(ValueError, match="created_by"):
        ReflexArc.from_dict({
            "name": "x",
            "description": "y",
            "trigger": {"e": 5.0},
            "days_since_human_min": 0.0,
            "cooldown_hours": 12.0,
            "action": "z",
            "output_memory_type": "reflex_x",
            "prompt_template": "t",
            "created_by": "alien_source",  # not in allowed enum
            "created_at": "2026-04-28T10:00:00+00:00",
        })
