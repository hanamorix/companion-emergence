"""tool_loop intercepts record_monologue, captures the args, writes digest synchronously."""
from __future__ import annotations

import json
from pathlib import Path

from brain.bridge.chat import ChatMessage, ChatResponse, ToolCall


class _RecordingProvider:
    """Provider that emits record_monologue tool call on call 1, plain reply on call 2."""

    def __init__(self) -> None:
        self.call_count = 0

    def chat(self, messages, *, tools=None, options=None):
        self.call_count += 1
        if self.call_count == 1:
            return ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="t1",
                        name="record_monologue",
                        arguments={
                            "monologue": "I was thinking about Loopy.",
                            "feed_digest": "she searched for Loopy and felt fond",
                        },
                    ),
                ),
                raw=None,
            )
        return ChatResponse(content="hello back", tool_calls=(), raw=None)

    def name(self):
        return "recording"


def test_tool_loop_captures_record_monologue_args(tmp_path: Path):
    from brain.chat.tool_loop import build_tools_list, run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    provider = _RecordingProvider()
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
        assert resp.content == "hello back"

        # Digest written synchronously, before reply shipped.
        digest_log = persona_dir / "monologue_digest.jsonl"
        assert digest_log.exists()
        entry = json.loads(digest_log.read_text().splitlines()[0])
        assert entry["digest"] == "she searched for Loopy and felt fond"

        # Invocation record carries the captured monologue text.
        rec = next(inv for inv in invocations if inv["name"] == "record_monologue")
        assert rec.get("monologue_text") == "I was thinking about Loopy."
    finally:
        store.close()
        hebbian.close()


def test_tool_loop_rejects_malformed_record_monologue(tmp_path: Path):
    """Whitespace monologue → CaptureRejected → tool result is an error, no digest written."""
    from brain.chat.tool_loop import build_tools_list, run_tool_loop
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    class _BadProvider(_RecordingProvider):
        def chat(self, messages, *, tools=None, options=None):
            self.call_count += 1
            if self.call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="t1",
                            name="record_monologue",
                            arguments={"monologue": "   ", "feed_digest": "digest"},
                        ),
                    ),
                    raw=None,
                )
            return ChatResponse(content="ok", tool_calls=(), raw=None)

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    provider = _BadProvider()
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
        # Reply ships normally.
        assert resp.content == "ok"
        # No digest written.
        assert not (persona_dir / "monologue_digest.jsonl").exists()
        # Invocation records the rejection.
        rec = next(inv for inv in invocations if inv["name"] == "record_monologue")
        assert "error" in rec.get("result_summary", rec)
    finally:
        store.close()
        hebbian.close()
