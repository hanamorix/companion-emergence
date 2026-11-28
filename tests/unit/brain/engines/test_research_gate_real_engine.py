"""The research gate's feeling half on the REAL emotion engine, not the stub (#651, spec A6).

A synthetic persona directory holds a saved engine state; the engine is installed as the persona's engine and `run_tick` reads it.
Nothing here touches a real persona.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.provider import FakeProvider
from brain.emotion.engine_utils import Average, EmotionState, EngineState, SnapshotDay
from brain.engines.research import ResearchEngine
from brain.memory.store import MemoryStore
from brain.search.base import NoopWebSearcher
from tests.unit.brain.emotion._engine_support import install, make_persona, save

SAVED_AT = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
DEFAULT_INTERESTS = Path(__file__).parents[4] / "brain" / "engines" / "default_interests.json"
INTEREST = {
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


def _feeling(short: float) -> EmotionState:
    """A feeling whose short average sits at the top of its own 30-day range, so the engine calls it high (when short >= 1)."""
    days = [SnapshotDay(f"2026-10-{i + 1:02d}", s, 0.0, True, None) for i, s in enumerate((0.5, 1.0, 0.8))]
    return EmotionState(short=Average(short, 1.0), long=Average(1.0, 1.0), last_entry_day=None, snapshot=days)


def _reason(tmp_path: Path, shorts: dict[str, float]) -> str | None:
    persona = make_persona(tmp_path / "persona")
    state = EngineState(
        saved_at=SAVED_AT,
        last_update=SAVED_AT,
        active_day=SAVED_AT.date().isoformat(),
        persona_start_at=SAVED_AT - timedelta(days=60),
        emotions={name: _feeling(v) for name, v in shorts.items()},
    )
    save(persona, state)
    install(persona, now=SAVED_AT)
    (persona / "interests.json").write_text(json.dumps({"version": 1, "interests": [INTEREST]}), encoding="utf-8")
    store = MemoryStore(persona / "memories.db")
    try:
        runner = ResearchEngine(
            store=store,
            provider=FakeProvider(),
            searcher=NoopWebSearcher(),
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
            interests_path=persona / "interests.json",
            research_log_path=persona / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS,
        )
        return runner.run_tick(
            trigger="manual", dry_run=True, days_since_human_override=0.0, emotion_state_override=None
        ).reason
    finally:
        store.close()


@pytest.mark.no_base_failure(reason="spec A6: with no opposing feeling present the new gate must give today's answer, so the old gate passes it by design; it guards against a wrong default for a missing name")
def test_real_engine_an_absent_opposing_feeling_counts_as_zero(tmp_path: Path):
    # awe is high at 5.0 and the persona has no grief, shame, comfort_seeking or rest_need: opp 0, T 0, so it opens
    # (a missing name counted as 10 would make T 8 and keep it closed)
    assert _reason(tmp_path, {"awe": 5.0}) != "not_due"


@pytest.mark.no_base_failure(reason="tests the engine's feels_high on a name the persona lacks, which this change does not touch; spec A6 asks for it as a guard")
def test_real_engine_an_opening_feeling_she_lacks_is_never_high(tmp_path: Path):
    persona = make_persona(tmp_path / "persona")
    state = EngineState(
        saved_at=SAVED_AT,
        last_update=SAVED_AT,
        active_day=SAVED_AT.date().isoformat(),
        emotions={"joy": _feeling(5.0)},
    )
    save(persona, state)
    eng = install(persona, now=SAVED_AT)
    assert eng.feels_high("joy") is True
    assert eng.feels_high("awe") is False


def test_real_engine_only_an_opposing_feeling_high_stays_closed(tmp_path: Path):
    assert _reason(tmp_path, {"grief": 9.0}) == "not_due"


def test_real_engine_a_moderate_opposing_feeling_raises_the_bar(tmp_path: Path):
    # grief 3.0 -> T 5.3: awe high at 5.0 stays closed; awe at 5.4 opens
    assert _reason(tmp_path, {"awe": 5.0, "grief": 3.0}) == "not_due"
    assert _reason(tmp_path / "again", {"awe": 5.4, "grief": 3.0}) != "not_due"
