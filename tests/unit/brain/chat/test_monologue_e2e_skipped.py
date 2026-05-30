"""End-to-end: trivial turn doesn't call record_monologue → no digest, no pass 2."""
from __future__ import annotations

import time
from pathlib import Path

from brain.bridge.chat import ChatMessage, ChatResponse


class _TrivialProvider:
    def __init__(self) -> None:
        self.generate_calls = 0

    def chat(self, messages, *, tools=None, options=None):
        return ChatResponse(content="hi back", tool_calls=(), raw=None)

    def generate(self, prompt, *, system=None):
        self.generate_calls += 1
        return "{}"

    def name(self):
        return "trivial"


def test_trivial_turn_no_monologue(tmp_path: Path):
    from brain.chat.tool_loop import run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    provider = _TrivialProvider()
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    try:
        resp, _ = run_tool_loop(
            messages=[ChatMessage(role="user", content="hi")],
            provider=provider,
            tools=None,
            store=store,
            hebbian=hebbian,
            persona_dir=persona_dir,
        )
        assert resp.content == "hi back"

        time.sleep(0.2)
        assert not (persona_dir / "monologue_digest.jsonl").exists()
        assert provider.generate_calls == 0
    finally:
        store.close()
        hebbian.close()
