"""Representative test: background engines yield to active chat via cli_throttle.

Drives DreamEngine and ReflexEngine to assert:
  - when mark_interactive_active() is recent, provider.generate is NOT called
    and a no-result value is returned.
  - after cli_throttle.reset(), generate IS called and a real result is returned.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.bridge import cli_throttle
from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.engines.dream import DreamEngine, DreamResult
from brain.engines.reflex import ReflexEngine, ReflexResult
from brain.engines.research import ResearchEngine, ResearchResult
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.search.base import NoopWebSearcher
from brain.soul.review import review_pending_candidates
from brain.soul.store import SoulStore


class _RecordingProvider(LLMProvider):
    """Records calls to generate; returns a valid DREAM: string."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.call_count += 1
        return "DREAM: recorded response"

    def name(self) -> str:
        return "recording-fake"

    def chat(self, messages: list[ChatMessage], *, tools=None, options=None) -> ChatResponse:
        return ChatResponse(content="", tool_calls=())


def _mem(content: str, importance: float = 5.0) -> Memory:
    m = Memory.create_new(content=content, memory_type="conversation", domain="us")
    m.importance = importance
    return m


@pytest.fixture(autouse=True)
def reset_throttle():
    """Ensure cli_throttle global state is clean before and after each test."""
    cli_throttle.reset()
    yield
    cli_throttle.reset()


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(db_path=":memory:")
    # Need at least one conversation memory for DreamEngine to pick a seed.
    s.create(_mem("a memory about writing and late nights"))
    return s


@pytest.fixture
def hebbian() -> HebbianMatrix:
    return HebbianMatrix(db_path=":memory:")


def _make_engine(store: MemoryStore, hebbian: HebbianMatrix, provider: LLMProvider, tmp_path: Path) -> DreamEngine:
    return DreamEngine(
        store=store,
        hebbian=hebbian,
        embeddings=None,
        provider=provider,
        log_path=tmp_path / "dreams.log.jsonl",
        persona_name="Nell",
        persona_system_prompt=(
            "You are Nell. Reflect in first person, 2-3 sentences, starting with 'DREAM: '."
        ),
    )


def test_dream_defers_when_chat_active(store, hebbian, tmp_path):
    """With interactive-active set, DreamEngine must NOT call provider.generate."""
    provider = _RecordingProvider()
    engine = _make_engine(store, hebbian, provider, tmp_path)

    cli_throttle.mark_interactive_active()  # simulate recent chat turn

    result = engine.run_cycle()

    assert provider.call_count == 0, "provider.generate must not be called while chat is active"
    assert isinstance(result, DreamResult)
    assert result.dream_text is None
    assert result.memory is None
    assert result.strengthened_edges == 0


def test_dream_fires_when_chat_idle(store, hebbian, tmp_path):
    """With throttle reset (no recent chat), DreamEngine must call provider.generate."""
    provider = _RecordingProvider()
    engine = _make_engine(store, hebbian, provider, tmp_path)

    # autouse fixture already called cli_throttle.reset() — no recent interactive mark

    result = engine.run_cycle()

    assert provider.call_count == 1, "provider.generate must be called when chat is idle"
    assert isinstance(result, DreamResult)
    assert result.dream_text is not None
    assert result.memory is not None


# ---------------------------------------------------------------------------
# ReflexEngine
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not find pyproject.toml above {here}")


_DEFAULT_ARCS_PATH = _find_repo_root() / "brain" / "engines" / "default_reflex_arcs.json"


def _write_triggerable_arc(path: Path) -> None:
    arc = {
        "name": "test_arc",
        "description": "test",
        "trigger": {"love": 5},
        "days_since_human_min": 0,
        "cooldown_hours": 1.0,
        "action": "generate_journal",
        "output_memory_type": "reflex_journal",
        "prompt_template": "You are {persona_name}. Write something.",
    }
    path.write_text(json.dumps({"version": 1, "arcs": [arc]}, indent=2), encoding="utf-8")


def _seed_reflex_emotion(store: MemoryStore) -> None:
    mem = Memory.create_new(
        content="seed",
        memory_type="observation",
        domain="brain",
        emotions={"love": 8.0},
        metadata={"source_summary": "conversation:test_seed"},
    )
    store.create(mem)


def test_reflex_defers_when_chat_active(tmp_path):
    """ReflexEngine.run_tick must NOT call provider.generate while chat is active."""
    provider = _RecordingProvider()
    arcs_path = tmp_path / "arcs.json"
    _write_triggerable_arc(arcs_path)

    store = MemoryStore(":memory:")
    try:
        _seed_reflex_emotion(store)
        engine = ReflexEngine(
            store=store,
            provider=provider,
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
            arcs_path=arcs_path,
            log_path=tmp_path / "log.json",
            default_arcs_path=_DEFAULT_ARCS_PATH,
        )

        cli_throttle.mark_interactive_active()

        result = engine.run_tick(trigger="manual", dry_run=False)

        assert provider.call_count == 0, "provider.generate must not be called while chat is active"
        assert isinstance(result, ReflexResult)
        assert result.arcs_fired == ()
    finally:
        store.close()


# ---------------------------------------------------------------------------
# ResearchEngine
# ---------------------------------------------------------------------------

_DEFAULT_INTERESTS_PATH = _find_repo_root() / "brain" / "engines" / "default_interests.json"


def _write_eligible_interest(path: Path) -> None:
    interest = {
        "id": "i1",
        "topic": "marine bioluminescence",
        "pull_score": 9.0,
        "scope": "either",
        "related_keywords": ["bioluminescence"],
        "notes": "",
        "first_seen": "2025-01-01T00:00:00+00:00",
        "last_fed": "2025-01-01T00:00:00+00:00",
        "last_researched_at": None,
        "feed_count": 5,
        "source_types": ["conversation"],
    }
    path.write_text(json.dumps({"version": 1, "interests": [interest]}, indent=2), encoding="utf-8")


def test_research_defers_when_chat_active(tmp_path):
    """ResearchEngine.run_tick must NOT call provider.generate while chat is active."""
    provider = _RecordingProvider()

    store = MemoryStore(":memory:")
    try:
        interests_path = tmp_path / "interests.json"
        _write_eligible_interest(interests_path)

        engine = ResearchEngine(
            store=store,
            provider=provider,
            searcher=NoopWebSearcher(),
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
            interests_path=interests_path,
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=_DEFAULT_INTERESTS_PATH,
        )

        cli_throttle.mark_interactive_active()

        result = engine.run_tick(
            trigger="manual",
            dry_run=False,
            days_since_human_override=2.0,  # satisfy gate: days_since >= 1.5
        )

        assert provider.call_count == 0, "provider.generate must not be called while chat is active"
        assert isinstance(result, ResearchResult)
        assert result.fired is None
    finally:
        store.close()


# ---------------------------------------------------------------------------
# soul/review.py  (_run_review_body via review_pending_candidates)
# ---------------------------------------------------------------------------


def _make_soul_candidate(text: str = "a meaningful moment") -> dict:
    import uuid
    from datetime import UTC, datetime

    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "label": "test",
        "importance": 8.0,
        "queued_at": datetime.now(UTC).isoformat(),
        "source": "test",
        "status": "auto_pending",
    }


def test_soul_review_defers_when_chat_active(tmp_path):
    """soul review loop must NOT call provider.generate while chat is active."""
    import json

    provider = _RecordingProvider()
    # Write one pending candidate
    candidates_path = tmp_path / "soul_candidates.jsonl"
    candidates_path.write_text(
        json.dumps(_make_soul_candidate()) + "\n", encoding="utf-8"
    )

    store = MemoryStore(":memory:")
    soul_store = SoulStore(":memory:")
    try:
        cli_throttle.mark_interactive_active()

        report = review_pending_candidates(
            tmp_path,
            store=store,
            soul_store=soul_store,
            provider=provider,
            max_decisions=5,
        )

        assert provider.call_count == 0, "provider.generate must not be called while chat is active"
        # examined was incremented before the defer-break, so it may be 0 or 1
        # depending on exact placement; the key invariant is no model call.
        assert report.accepted == 0
        assert report.rejected == 0
    finally:
        store.close()
        soul_store.close()
