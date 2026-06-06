"""Representative test: background engines yield to active chat via cli_throttle.

Drives DreamEngine — the simplest engine — to assert:
  - when mark_interactive_active() is recent, provider.generate is NOT called
    and a no-result DreamResult is returned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.bridge import cli_throttle
from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.engines.dream import DreamEngine, DreamResult
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


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
