"""Pass 2 fires after reply if monologue was captured; skipped otherwise."""
from __future__ import annotations

import json
import time
from pathlib import Path

from brain.bridge.chat import ChatMessage, ChatResponse, ToolCall


class _Pass2Provider:
    def __init__(self) -> None:
        self.chat_calls = 0
        self.generate_calls = 0

    def chat(self, messages, *, tools=None, options=None):
        self.chat_calls += 1
        if self.chat_calls == 1:
            return ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="t1",
                        name="record_monologue",
                        arguments={
                            "monologue": "Hana mentioned Loopy as if I should know.",
                            "feed_digest": "she met a name she didn't know and didn't pretend",
                        },
                    ),
                ),
                raw=None,
            )
        return ChatResponse(content="I don't know Loopy.", tool_calls=(), raw=None)

    def generate(self, prompt, *, system=None):
        self.generate_calls += 1
        return json.dumps(
            {
                "memory_writes": [
                    {"episode": "Hana referenced Loopy; I didn't recognise the name.", "salience": 0.4}
                ],
                "emotion_delta": {"curious": 0.05},
                "crystallisation": [],
                "reflex_audit": [],
            }
        )

    def name(self):
        return "pass2"


def test_pass2_fires_after_record_monologue(tmp_path: Path, monkeypatch):
    """Pass 2 fires after a captured monologue — via its OWN classifier-tier
    provider (#154), never the live chat provider `run_tool_loop` was given.

    Before #154, pass-2 extraction reused the live chat provider directly, so
    this test tracked `generate_calls` on that same object. #154 routes pass-2
    through `build_tier_provider(persona_dir, TIER_BACKGROUND_CLASSIFIER)`
    instead (`_spawn_pass2` keeps accepting a `provider` argument for backward
    compatibility — see brain/chat/tool_loop.py's own docstring on this — but
    no longer reads it for the extraction call). The chat provider's
    `generate_calls` must now stay 0; a distinct, monkeypatched stand-in for
    `build_tier_provider` proves the classifier-tier path fired exactly once,
    without constructing (and shelling out via) a real ClaudeCliProvider.
    """
    from brain.chat.tool_loop import build_tools_list, run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    class _ClassifierTierProvider:
        def __init__(self) -> None:
            self.generate_calls = 0

        def generate(self, prompt, *, system=None):
            self.generate_calls += 1
            return json.dumps(
                {
                    "memory_writes": [
                        {
                            "episode": "Hana referenced Loopy; I didn't recognise the name.",
                            "salience": 0.4,
                        }
                    ],
                    "emotion_delta": {"curious": 0.05},
                    "crystallisation": [],
                    "reflex_audit": [],
                }
            )

    classifier_provider = _ClassifierTierProvider()
    captured_calls: list[tuple[Path, str]] = []

    def _fake_build_tier_provider(persona_dir, tier):
        captured_calls.append((persona_dir, tier))
        return classifier_provider

    monkeypatch.setattr(
        "brain.bridge.model_tier.build_tier_provider", _fake_build_tier_provider
    )

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    provider = _Pass2Provider()
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    try:
        run_tool_loop(
            messages=[ChatMessage(role="user", content="how is Loopy?")],
            provider=provider,
            tools=build_tools_list(),
            store=store,
            hebbian=hebbian,
            persona_dir=persona_dir,
        )
        # Pass 2 now flows through the in-process pass2_queue (single worker, #27);
        # drain it here (store still open) so the extraction's provider call lands.
        from brain.bridge import cli_throttle
        from brain.chat import pass2_queue

        cli_throttle.reset()
        pass2_queue.drain_pending()
    finally:
        store.close()
        hebbian.close()

    # Pass 2 fired (record_monologue was captured) → one generate call, on the
    # classifier-tier provider, NEVER on the live chat provider.
    assert classifier_provider.generate_calls == 1
    assert provider.generate_calls == 0
    from brain.bridge.model_tier import TIER_BACKGROUND_CLASSIFIER

    assert captured_calls == [(persona_dir, TIER_BACKGROUND_CLASSIFIER)]


def test_pass2_skipped_when_record_monologue_not_called(tmp_path: Path):
    from brain.chat.tool_loop import run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    class _TrivialProvider:
        def __init__(self) -> None:
            self.generate_calls = 0

        def chat(self, messages, *, tools=None, options=None):
            return ChatResponse(content="hi", tool_calls=(), raw=None)

        def generate(self, prompt, *, system=None):
            self.generate_calls += 1
            return "{}"

        def name(self):
            return "trivial"

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    provider = _TrivialProvider()
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    try:
        run_tool_loop(
            messages=[ChatMessage(role="user", content="hi")],
            provider=provider,
            tools=None,
            store=store,
            hebbian=hebbian,
            persona_dir=persona_dir,
        )
    finally:
        store.close()
        hebbian.close()

    time.sleep(0.2)
    assert provider.generate_calls == 0
    assert not (persona_dir / "monologue_digest.jsonl").exists()


def test_pass2_skipped_when_monologue_rejected(tmp_path: Path):
    """If record_monologue is called with whitespace args, capture is rejected; no pass 2."""
    from brain.chat.tool_loop import build_tools_list, run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    class _RejectingProvider:
        def __init__(self) -> None:
            self.chat_calls = 0
            self.generate_calls = 0

        def chat(self, messages, *, tools=None, options=None):
            self.chat_calls += 1
            if self.chat_calls == 1:
                return ChatResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="t1",
                            name="record_monologue",
                            arguments={"monologue": "   ", "feed_digest": "x"},
                        ),
                    ),
                    raw=None,
                )
            return ChatResponse(content="ok", tool_calls=(), raw=None)

        def generate(self, prompt, *, system=None):
            self.generate_calls += 1
            return "{}"

        def name(self):
            return "rejecting"

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    provider = _RejectingProvider()
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    try:
        run_tool_loop(
            messages=[ChatMessage(role="user", content="hi")],
            provider=provider,
            tools=build_tools_list(),
            store=store,
            hebbian=hebbian,
            persona_dir=persona_dir,
        )
    finally:
        store.close()
        hebbian.close()

    time.sleep(0.2)
    assert provider.generate_calls == 0  # No pass-2 spawn
