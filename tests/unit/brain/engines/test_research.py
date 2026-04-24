"""Unit tests for brain.engines.research — scaffold + types."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.bridge.provider import FakeProvider
from brain.engines.research import (
    ResearchEngine,
    ResearchFire,
    ResearchResult,
)
from brain.memory.store import MemoryStore
from brain.search.base import NoopWebSearcher

DEFAULT_INTERESTS_PATH = Path(__file__).parents[4] / "brain" / "engines" / "default_interests.json"


def test_research_fire_construction():
    fire = ResearchFire(
        interest_id="abc",
        topic="Test",
        fired_at=datetime.now(UTC),
        trigger="manual",
        web_used=False,
        web_result_count=0,
        output_memory_id="mem_xyz",
    )
    assert fire.topic == "Test"
    assert fire.web_used is False


def test_research_result_construction():
    r = ResearchResult(
        fired=None,
        would_fire=None,
        reason="not_due",
        dry_run=False,
        evaluated_at=datetime.now(UTC),
    )
    assert r.fired is None
    assert r.reason == "not_due"


def test_research_engine_construction(tmp_path: Path):
    store = MemoryStore(":memory:")
    try:
        engine = ResearchEngine(
            store=store,
            provider=FakeProvider(),
            searcher=NoopWebSearcher(),
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
            interests_path=tmp_path / "interests.json",
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
        )
        assert engine.persona_name == "Nell"
    finally:
        store.close()


def test_run_tick_raises_not_implemented_yet(tmp_path: Path):
    store = MemoryStore(":memory:")
    try:
        engine = ResearchEngine(
            store=store,
            provider=FakeProvider(),
            searcher=NoopWebSearcher(),
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
            interests_path=tmp_path / "interests.json",
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
        )
        with pytest.raises(NotImplementedError):
            engine.run_tick(trigger="manual", dry_run=False)
    finally:
        store.close()
