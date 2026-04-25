"""Unit tests for brain.engines.research — scaffold + types."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.provider import FakeProvider
from brain.engines._interests import InterestSet
from brain.engines.research import ResearchEngine
from brain.memory.store import Memory, MemoryStore
from brain.search.base import NoopWebSearcher

DEFAULT_INTERESTS_PATH = Path(__file__).parents[4] / "brain" / "engines" / "default_interests.json"


# ---- run_tick ----


def _build_engine(
    tmp_path: Path, store: MemoryStore, provider=None, searcher=None
) -> ResearchEngine:
    return ResearchEngine(
        store=store,
        provider=provider or FakeProvider(),
        searcher=searcher or NoopWebSearcher(),
        persona_name="Nell",
        persona_system_prompt="You are Nell.",
        interests_path=tmp_path / "interests.json",
        research_log_path=tmp_path / "research_log.json",
        default_interests_path=DEFAULT_INTERESTS_PATH,
    )


def _write_interests(path: Path, interests: list[dict]) -> None:
    path.write_text(json.dumps({"version": 1, "interests": interests}, indent=2), encoding="utf-8")


def _seed_conversation_memory(
    store: MemoryStore, content: str, emotions: dict[str, float] | None = None
) -> str:
    mem = Memory.create_new(
        content=content,
        memory_type="conversation",
        domain="us",
        emotions=emotions or {},
    )
    store.create(mem)
    return mem.id


def _interest_dict(**overrides) -> dict:
    base = {
        "id": "i1",
        "topic": "marine bioluminescence",
        "pull_score": 7.0,
        "scope": "either",
        "related_keywords": ["marine", "bioluminescence", "ocean"],
        "notes": "",
        "first_seen": "2026-04-01T10:00:00Z",
        "last_fed": "2026-04-15T10:00:00Z",
        "last_researched_at": None,
        "feed_count": 3,
        "source_types": ["manual"],
    }
    base.update(overrides)
    return base


def test_run_tick_no_interests_defined(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(trigger="manual", dry_run=False, days_since_human_override=5.0)
        assert result.fired is None
        assert result.reason == "no_interests_defined"
    finally:
        store.close()


def test_run_tick_not_due_returns_reason(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        # Low days + no emotion signal provided → not due
        result = engine.run_tick(
            trigger="manual",
            dry_run=False,
            days_since_human_override=0.0,
            emotion_state_override=None,
        )
        assert result.fired is None
        assert result.reason == "not_due"
    finally:
        store.close()


def test_run_tick_days_since_human_triggers(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(
            trigger="days_since_human", dry_run=False, days_since_human_override=5.0
        )
        # Now eligible + selected + fired
        assert result.fired is not None
        assert result.fired.topic == "marine bioluminescence"
    finally:
        store.close()


def test_run_tick_emotion_high_triggers(tmp_path: Path):
    from brain.emotion.state import EmotionalState

    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        es = EmotionalState()
        es.set("curiosity", 8.0)
        result = engine.run_tick(
            trigger="emotion_high",
            dry_run=False,
            days_since_human_override=0.0,
            emotion_state_override=es,
        )
        assert result.fired is not None
    finally:
        store.close()


def test_run_tick_pull_threshold_filters(tmp_path: Path):
    _write_interests(
        tmp_path / "interests.json", [_interest_dict(pull_score=5.0)]
    )  # below threshold 6.0
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(days_since_human_override=5.0)
        assert result.fired is None
        assert result.reason == "no_eligible_interest"
    finally:
        store.close()


def test_run_tick_cooldown_filters(tmp_path: Path):
    now = datetime.now(UTC)
    recent = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    _write_interests(tmp_path / "interests.json", [_interest_dict(last_researched_at=recent)])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(days_since_human_override=5.0)
        assert result.fired is None
        assert result.reason == "no_eligible_interest"
    finally:
        store.close()


def test_run_tick_ranks_highest_pull(tmp_path: Path):
    _write_interests(
        tmp_path / "interests.json",
        [
            _interest_dict(id="low", topic="Topic A", pull_score=6.5, related_keywords=["a"]),
            _interest_dict(id="high", topic="Topic B", pull_score=8.0, related_keywords=["b"]),
        ],
    )
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(days_since_human_override=5.0)
        assert result.fired is not None
        assert result.fired.topic == "Topic B"
    finally:
        store.close()


def test_run_tick_dry_run_reports_would_fire(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(dry_run=True, days_since_human_override=5.0)
        assert result.dry_run is True
        assert result.would_fire == "marine bioluminescence"
        assert result.fired is None
        # No memory written
        assert store.count() == 0
        # No log file
        assert not (tmp_path / "research_log.json").exists()
    finally:
        store.close()


def test_run_tick_fire_writes_research_memory(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        result = engine.run_tick(days_since_human_override=5.0)
        assert result.fired is not None
        mem = store.get(result.fired.output_memory_id)
        assert mem is not None
        assert mem.memory_type == "research"
        assert mem.metadata["interest_topic"] == "marine bioluminescence"
        assert mem.metadata["web_used"] is False  # Noop searcher returns []
    finally:
        store.close()


def test_run_tick_fire_updates_interest_last_researched_at(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        engine.run_tick(days_since_human_override=5.0)
        # Reload interests and verify last_researched_at was updated
        reloaded = InterestSet.load(
            tmp_path / "interests.json", default_path=DEFAULT_INTERESTS_PATH
        )
        assert reloaded.interests[0].last_researched_at is not None
    finally:
        store.close()


def test_run_tick_fire_appends_to_log(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    try:
        engine = _build_engine(tmp_path, store)
        engine.run_tick(days_since_human_override=5.0)
        log_data = json.loads((tmp_path / "research_log.json").read_text(encoding="utf-8"))
        assert len(log_data["fires"]) == 1
        assert log_data["fires"][0]["topic"] == "marine bioluminescence"
    finally:
        store.close()


def test_run_tick_internal_scope_skips_searcher(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict(scope="internal")])
    store = MemoryStore(":memory:")

    class TrackingSearcher(NoopWebSearcher):
        calls = 0

        def search(self, query: str, *, limit: int = 5):
            TrackingSearcher.calls += 1
            return []

    ts = TrackingSearcher()
    try:
        engine = _build_engine(tmp_path, store, searcher=ts)
        result = engine.run_tick(days_since_human_override=5.0)
        assert result.fired is not None
        assert TrackingSearcher.calls == 0
        assert result.fired.web_used is False
    finally:
        store.close()


def test_run_tick_llm_failure_does_not_touch_files(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")

    class FailingProvider(FakeProvider):
        def generate(self, prompt, *, system=None):
            raise RuntimeError("simulated LLM failure")

    try:
        engine = _build_engine(tmp_path, store, provider=FailingProvider())
        with pytest.raises(RuntimeError):
            engine.run_tick(days_since_human_override=5.0)
        # No memory, no log
        assert store.count() == 0
        assert not (tmp_path / "research_log.json").exists()
        # last_researched_at still None
        reloaded = InterestSet.load(
            tmp_path / "interests.json", default_path=DEFAULT_INTERESTS_PATH
        )
        assert reloaded.interests[0].last_researched_at is None
    finally:
        store.close()


def test_run_tick_renders_prompt_with_context(tmp_path: Path):
    _write_interests(tmp_path / "interests.json", [_interest_dict()])
    store = MemoryStore(":memory:")
    _seed_conversation_memory(store, "I love how deep-sea creatures make their own light.")

    captured = {}

    class CapturingProvider(FakeProvider):
        def generate(self, prompt, *, system=None):
            captured["prompt"] = prompt
            captured["system"] = system
            return "I spent some time today exploring marine bioluminescence..."

    try:
        engine = _build_engine(tmp_path, store, provider=CapturingProvider())
        engine.run_tick(days_since_human_override=5.0)
    finally:
        store.close()

    assert "marine bioluminescence" in captured["prompt"]
    assert "Nell" in (captured["system"] or "")


# ---- ResearchLog unit tests ----


def test_research_log_load_missing_returns_empty(tmp_path: Path) -> None:
    from brain.engines.research import ResearchLog

    log = ResearchLog.load(tmp_path / "nope.json")
    assert log.fires == ()


def test_research_log_load_corrupt_returns_empty(tmp_path: Path) -> None:
    """Corrupt file is healed to empty via attempt_heal; fires == ()."""
    from brain.engines.research import ResearchLog

    path = tmp_path / "research_log.json"
    path.write_text("{{{not json", encoding="utf-8")
    log = ResearchLog.load(path)
    assert log.fires == ()


def test_research_log_load_corrupt_quarantines_and_warns(tmp_path: Path, caplog) -> None:
    """Corrupt primary is quarantined and a WARNING is emitted."""
    import logging

    from brain.engines.research import ResearchLog

    caplog.set_level(logging.WARNING)
    path = tmp_path / "research_log.json"
    path.write_text("{{{not json", encoding="utf-8")
    ResearchLog.load(path)
    corrupt_files = list(tmp_path.glob("research_log.json.corrupt-*"))
    assert len(corrupt_files) == 1
    warn_msgs = [r.getMessage() for r in caplog.records if "ResearchLog anomaly" in r.getMessage()]
    assert len(warn_msgs) == 1


def test_research_log_load_heals_from_bak(tmp_path: Path) -> None:
    """When primary is corrupt, load restores the most recent valid .bak."""
    from brain.engines.research import ResearchFire, ResearchLog

    fire = ResearchFire(
        interest_id="i1",
        topic="marine bioluminescence",
        fired_at=datetime.now(UTC),
        trigger="days_since_human",
        web_used=False,
        web_result_count=0,
        output_memory_id="mem-1",
    )
    good_payload = {"version": 1, "fires": [fire.to_dict()]}
    path = tmp_path / "research_log.json"
    bak1 = tmp_path / "research_log.json.bak1"
    bak1.write_text(json.dumps(good_payload), encoding="utf-8")
    path.write_text("{{{not json", encoding="utf-8")
    log = ResearchLog.load(path)
    assert len(log.fires) == 1
    assert log.fires[0].topic == "marine bioluminescence"
