"""#227 — replies are stored and returned exactly as the model produced them.

An earlier attempt stripped a leaked ``</s>`` or ``", "ts": "<iso>"}`` tail before the reply was
persisted. That hid the symptom and removed the evidence of how often it happens, so the filter is
gone and these tests pin the opposite: the raw tail survives into the returned content and the buffer.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.chat.engine import respond
from brain.chat.session import reset_registry
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

_RAW_REPLIES = [
    "hi there\n</s>",
    'ok", "ts": "2026-09-07T21:21:30-04:00"}',
    'fine.", "ts": "2026-09-08T00:10:55-04:00"}\n</s>',
]


class _RawProvider(LLMProvider):
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def name(self) -> str:
        return "raw"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "ok"

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        return ChatResponse(content=self._reply, tool_calls=())


@pytest.mark.parametrize("raw", _RAW_REPLIES)
def test_respond_returns_and_persists_the_raw_reply(tmp_path: Path, raw: str) -> None:
    reset_registry()
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "noop"}), encoding="utf-8"
    )
    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")
    try:
        result = respond(
            persona_dir,
            "how did it go?",
            store=store,
            hebbian=hebbian,
            provider=_RawProvider(raw),
            voice_md_override="# Nell",
        )
    finally:
        store.close()
        hebbian.close()
        reset_registry()

    assert result.content == raw
    rows = [
        json.loads(line)
        for p in (persona_dir / "active_conversations").glob("*.jsonl")
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r["text"] for r in rows if r.get("speaker") == "assistant"] == [raw]


def test_the_scaffold_filter_module_is_gone() -> None:
    assert importlib.util.find_spec("brain.chat.reply_scaffold") is None
    engine_source = (Path(__file__).parents[4] / "brain" / "chat" / "engine.py").read_text(encoding="utf-8")
    assert "reply_scaffold" not in engine_source
    assert "strip_scaffold_tail" not in engine_source
