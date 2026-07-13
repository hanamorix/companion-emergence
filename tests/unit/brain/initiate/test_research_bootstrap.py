"""Tests for research engine interest bootstrap from voice.md.

When interests.json is empty and voice.md exists, the engine should:
- Make a single LLM call to extract starter interests
- Write them to interests.json
- Continue the normal research flow (not return no_interests_defined)
"""

from __future__ import annotations

import json
from pathlib import Path

from brain.bridge.provider import FakeProvider
from brain.engines._interests import InterestSet
from brain.engines.research import ResearchEngine
from brain.memory.store import MemoryStore
from brain.search.base import NoopWebSearcher

DEFAULT_INTERESTS_PATH = (
    Path(__file__).parents[4] / "brain" / "engines" / "default_interests.json"
)

_BOOTSTRAP_TOPICS = [
    "gothic architecture history",
    "deep sea bioluminescence",
    "Victorian-era letter writing",
    "mycorrhizal fungal networks",
    "liminal spaces in fiction",
]

_BOOTSTRAP_JSON = json.dumps(_BOOTSTRAP_TOPICS)


class _BootstrapProvider(FakeProvider):
    """Provider that returns a JSON array on the first generate() call (bootstrap),
    then falls back to FakeProvider for subsequent calls."""

    def __init__(self):
        self._calls: list[str] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self._calls.append(prompt)
        # First call is the bootstrap prompt — return a JSON array of topics
        if len(self._calls) == 1:
            return _BOOTSTRAP_JSON
        return super().generate(prompt, system=system)

    def name(self) -> str:
        return "bootstrap_test"


def _build_engine(tmp_path: Path, provider: FakeProvider) -> ResearchEngine:
    return ResearchEngine(
        store=MemoryStore(":memory:"),
        provider=provider,
        searcher=NoopWebSearcher(),
        persona_name="Nell",
        persona_system_prompt="You are Nell.",
        interests_path=tmp_path / "interests.json",
        research_log_path=tmp_path / "research_log.json",
        default_interests_path=DEFAULT_INTERESTS_PATH,
    )


def test_bootstrap_writes_interests_from_voice(tmp_path: Path):
    """When interests.json is empty and voice.md exists, bootstrap writes 5 interests."""
    # Arrange
    (tmp_path / "interests.json").write_text(
        json.dumps({"version": 1, "interests": []}), encoding="utf-8"
    )
    (tmp_path / "voice.md").write_text(
        "Nell is a sweater-wearing novelist with a love of the strange and the "
        "liminal. She's fascinated by the hidden architectures of the world — "
        "mycelial networks, Gothic cathedrals, the way light behaves at dusk. "
        "She reads Victorian letters for fun and is moved by deep-sea footage.",
        encoding="utf-8",
    )
    provider = _BootstrapProvider()
    engine = _build_engine(tmp_path, provider)

    # Act — bootstrap only; no gate pass needed to verify the write
    engine._seed_interests_from_voice()

    # Assert: interests.json now has 5 non-empty interests
    loaded = InterestSet.load(tmp_path / "interests.json", default_path=DEFAULT_INTERESTS_PATH)
    assert len(loaded.interests) == 5
    topics = [i.topic for i in loaded.interests]
    for topic in topics:
        assert topic.strip(), "interest topic must not be blank"


def test_bootstrap_result_not_no_interests_defined(tmp_path: Path):
    """run_tick should not return no_interests_defined after a successful bootstrap."""
    # Arrange
    (tmp_path / "interests.json").write_text(
        json.dumps({"version": 1, "interests": []}), encoding="utf-8"
    )
    (tmp_path / "voice.md").write_text(
        "Nell is a sweater-wearing novelist fascinated by liminal spaces.",
        encoding="utf-8",
    )
    provider = _BootstrapProvider()
    engine = _build_engine(tmp_path, provider)

    # Act — trigger gate override so we get past the gate check
    result = engine.run_tick(
        trigger="manual",
        dry_run=True,  # dry_run avoids the full LLM + memory write path
        days_since_human_override=5.0,  # forces gate open
    )

    assert result.reason != "no_interests_defined", (
        f"Expected bootstrap to populate interests; got reason={result.reason!r}"
    )


def test_bootstrap_no_voice_file_falls_back(tmp_path: Path):
    """When voice.md is absent, bootstrap returns False and tick returns no_interests_defined."""
    (tmp_path / "interests.json").write_text(
        json.dumps({"version": 1, "interests": []}), encoding="utf-8"
    )
    # No voice.md written
    provider = _BootstrapProvider()
    engine = _build_engine(tmp_path, provider)

    result = engine._seed_interests_from_voice()
    assert result is False

    # run_tick should still fall back gracefully
    tick_result = engine.run_tick(
        trigger="manual",
        dry_run=False,
        days_since_human_override=5.0,
    )
    assert tick_result.reason == "no_interests_defined"


def test_bootstrap_bad_llm_response_falls_back(tmp_path: Path):
    """If the LLM returns garbage (not a JSON array), bootstrap returns False."""

    class _GarbageProvider(FakeProvider):
        def generate(self, prompt: str, *, system: str | None = None) -> str:
            return "Sure! Here are some topics: history, science, art."

        def name(self) -> str:
            return "garbage_test"

    (tmp_path / "interests.json").write_text(
        json.dumps({"version": 1, "interests": []}), encoding="utf-8"
    )
    (tmp_path / "voice.md").write_text("Nell loves books.", encoding="utf-8")
    engine = _build_engine(tmp_path, _GarbageProvider())

    result = engine._seed_interests_from_voice()
    assert result is False


def test_bootstrap_prompt_includes_conversation_memories(tmp_path: Path):
    """Prompt should include up to 10 recent conversation memories."""
    # Arrange: add conversation memories to the store
    store = MemoryStore(":memory:")
    from brain.memory.store import Memory
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    # Add a memory with distinctive content that we'll search for in the prompt
    mem1 = Memory.create_new(
        content="the user talked about wulfsyarn at length yesterday",
        memory_type="conversation",
        domain="us",
        emotions={},
    )
    store.create(mem1)

    # Build engine with this store and capturing provider
    provider = _BootstrapProvider()
    engine = ResearchEngine(
        store=store,
        provider=provider,
        searcher=NoopWebSearcher(),
        persona_name="Nell",
        persona_system_prompt="You are Nell.",
        interests_path=tmp_path / "interests.json",
        research_log_path=tmp_path / "research_log.json",
        default_interests_path=DEFAULT_INTERESTS_PATH,
    )
    (tmp_path / "interests.json").write_text(
        json.dumps({"version": 1, "interests": []}), encoding="utf-8"
    )
    (tmp_path / "voice.md").write_text("Nell is a novelist.", encoding="utf-8")

    # Act
    engine._seed_interests_from_voice()

    # Assert: captured prompt should include the conversation memory content
    assert provider._calls, "Provider should have been called"
    bootstrap_prompt = provider._calls[0]
    assert "wulfsyarn" in bootstrap_prompt.lower(), (
        f"Prompt should include conversation memory content; got: {bootstrap_prompt}"
    )


def test_bootstrap_parses_json_with_preamble(tmp_path: Path):
    """Parser should handle preamble before JSON array (e.g., 'Sure! Here you go:\\n[...]')."""

    class _PreambleProvider(FakeProvider):
        def generate(self, prompt: str, *, system: str | None = None) -> str:
            # First call is bootstrap — return JSON with preamble
            if "suggest 5 specific" in prompt:
                return f'Sure! Here you go:\n{_BOOTSTRAP_JSON}'
            return super().generate(prompt, system=system)

        def name(self) -> str:
            return "preamble_test"

    (tmp_path / "interests.json").write_text(
        json.dumps({"version": 1, "interests": []}), encoding="utf-8"
    )
    (tmp_path / "voice.md").write_text("Nell loves books.", encoding="utf-8")
    engine = _build_engine(tmp_path, _PreambleProvider())

    # Act
    result = engine._seed_interests_from_voice()

    # Assert: should successfully parse despite the preamble
    assert result is True, "Should successfully parse JSON with preamble"
    loaded = InterestSet.load(tmp_path / "interests.json", default_path=DEFAULT_INTERESTS_PATH)
    assert len(loaded.interests) == 5
