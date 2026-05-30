"""End-to-end: a substantive chat turn fires record_monologue → digest + pass 2 → memory."""
from __future__ import annotations

import json
import time
from pathlib import Path

from brain.bridge.chat import ChatMessage, ChatResponse, ToolCall


class _SubstantiveProvider:
    def __init__(self) -> None:
        self.chat_calls = 0

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
                            "monologue": (
                                "Hana mentioned Loopy as if I should know. "
                                "I searched my memory and nothing surfaced. "
                                "There's a small warmth in not faking it."
                            ),
                            "feed_digest": "she met a name she didn't know and didn't pretend",
                        },
                    ),
                ),
                raw=None,
            )
        return ChatResponse(
            content="I don't recognise Loopy — what is that?",
            tool_calls=(),
            raw=None,
        )

    def generate(self, prompt, *, system=None):
        return json.dumps(
            {
                "memory_writes": [
                    {
                        "episode": "Hana referenced Loopy as known; I'd never met the name. Acknowledged honestly.",
                        "salience": 0.45,
                    }
                ],
                "emotion_delta": {"curious": 0.05},
                "crystallisation": [],
                "reflex_audit": [],
            }
        )

    def name(self):
        return "substantive"


def test_full_turn_updates_all_surfaces(tmp_path: Path):
    from brain.chat.tool_loop import build_tools_list, run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    provider = _SubstantiveProvider()
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
        assert "Loopy" in resp.content

        # Digest written synchronously — already present when run_tool_loop returns.
        digest_log = persona_dir / "monologue_digest.jsonl"
        assert digest_log.exists()
        assert "didn't pretend" in digest_log.read_text()

        # Pass 2 is async — wait for memory write to land.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            recent = list(store.list_by_type("monologue", active_only=True, limit=10))
            if recent:
                break
            time.sleep(0.05)

        recent = list(store.list_by_type("monologue", active_only=True, limit=10))
        assert any("Loopy" in m.content for m in recent), (
            f"no Loopy memory found in {[m.content for m in recent]}"
        )
    finally:
        store.close()
        hebbian.close()
