"""End-to-end: whitespace-only monologue → rejection logged → no digest, reply still ships."""
from __future__ import annotations

import time
from pathlib import Path

from brain.bridge.chat import ChatMessage, ChatResponse, ToolCall


class _MalformedProvider:
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
        return ChatResponse(content="reply continues", tool_calls=(), raw=None)

    def generate(self, prompt, *, system=None):
        self.generate_calls += 1
        return "{}"

    def name(self):
        return "malformed"


def test_malformed_args_rejected_no_digest_no_pass2(tmp_path: Path):
    from brain.chat.tool_loop import build_tools_list, run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    provider = _MalformedProvider()
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    try:
        resp, invocations = run_tool_loop(
            messages=[ChatMessage(role="user", content="hi")],
            provider=provider,
            tools=build_tools_list(),
            store=store,
            hebbian=hebbian,
            persona_dir=persona_dir,
        )
        assert resp.content == "reply continues"

        time.sleep(0.2)
        assert not (persona_dir / "monologue_digest.jsonl").exists()
        assert provider.generate_calls == 0
        rec = next(inv for inv in invocations if inv["name"] == "record_monologue")
        assert "error" in rec.get("result_summary", rec)
    finally:
        store.close()
        hebbian.close()
