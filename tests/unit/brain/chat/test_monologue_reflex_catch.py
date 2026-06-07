"""Reflex catch: search_memories interleaves with record_monologue + visible reply."""
from __future__ import annotations

import json
from pathlib import Path

from brain.bridge.chat import ChatMessage, ChatResponse, ToolCall


class _ReflexProvider:
    """Three-call sequence: search → record_monologue → reply.

    Mirrors the design property — the model can call any tool before
    composing the visible reply.
    """

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
                        name="search_memories",
                        arguments={"query": "Loopy"},
                    ),
                ),
                raw=None,
            )
        if self.chat_calls == 2:
            return ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="t2",
                        name="record_monologue",
                        arguments={
                            "monologue": "Searched and got nothing. I genuinely don't know Loopy.",
                            "feed_digest": "she searched and acknowledged the gap",
                        },
                    ),
                ),
                raw=None,
            )
        return ChatResponse(
            content="I checked — no record of Loopy.",
            tool_calls=(),
            raw=None,
        )

    def generate(self, prompt, *, system=None):
        self.generate_calls += 1
        return json.dumps(
            {
                "memory_writes": [],
                "emotion_delta": {},
                "crystallisation": [],
                "reflex_audit": [],
            }
        )

    def name(self):
        return "reflex"


def test_search_interleaves_with_record_monologue(tmp_path: Path):
    from brain.chat.tool_loop import build_tools_list, run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    provider = _ReflexProvider()
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    try:
        resp, invocations = run_tool_loop(
            messages=[ChatMessage(role="user", content="how is Loopy?")],
            provider=provider,
            tools=build_tools_list(),
            store=store,
            hebbian=hebbian,
            persona_dir=persona_dir,
        )
        # All three model calls happened.
        assert provider.chat_calls == 3
        # The visible reply made it.
        assert "no record" in resp.content
        # Both tool invocations were recorded.
        names = [inv["name"] for inv in invocations]
        assert "search_memories" in names
        assert "record_monologue" in names
        # The record_monologue invocation captured monologue_text.
        monologue_rec = next(inv for inv in invocations if inv["name"] == "record_monologue")
        assert monologue_rec.get("monologue_text", "").startswith("Searched")
        # Digest written synchronously.
        assert (persona_dir / "monologue_digest.jsonl").exists()
        # Pass 2 now flows through the in-process pass2_queue (single worker, #27);
        # drain it here (store still open) so the extraction's provider call lands.
        from brain.bridge import cli_throttle
        from brain.chat import pass2_queue

        cli_throttle.reset()
        pass2_queue.drain_pending()
    finally:
        store.close()
        hebbian.close()

    # Pass 2 spawned (because record_monologue was captured) → one generate call.
    assert provider.generate_calls == 1
