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


def test_pass2_fires_after_record_monologue(tmp_path: Path):
    from brain.chat.tool_loop import build_tools_list, run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

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
    finally:
        store.close()
        hebbian.close()

    # Pass 2 runs in a daemon thread -- give it 5s.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if provider.generate_calls >= 1:
            break
        time.sleep(0.05)

    assert provider.generate_calls == 1


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
