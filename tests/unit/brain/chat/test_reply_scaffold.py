"""#227 — scaffolding tokens the model leaks onto the tail of its own reply.

Two variants, both logged in production:
  1. the literal ``</s>`` end-of-sequence token;
  2. the JSON wrapper history is replayed in, completed by the model:
     ``", "ts": "2026-09-08T00:10:55-04:00"`` (sometimes with a closing ``"}``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.chat.engine import respond
from brain.chat.reply_scaffold import strip_scaffold_tail
from brain.chat.session import reset_registry
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        # variant 1: </s>
        ("unsettling-strange or both at once?\n</s>", "unsettling-strange or both at once?"),
        ("good outcome for the photographer at least.</s>", "good outcome for the photographer at least."),
        ("tail twice.\n</s>\n</s>", "tail twice."),
        # variant 2: JSON ts wrapper, with and without the closing brace
        ('must be solved.", "ts": "2026-08-21T00:12:46+00:00"', "must be solved."),
        ('from here.", "ts": "2026-09-07T21:21:30-04:00"}', "from here."),
        ('not small.", "ts": "2026-09-08T00:10:55-04:00', "not small."),
        # both stacked
        ('fine.", "ts": "2026-09-08T00:10:55-04:00"}\n</s>', "fine."),
    ],
)
def test_strips_leaked_scaffold_tail(raw: str, clean: str) -> None:
    assert strip_scaffold_tail(raw) == clean


@pytest.mark.parametrize(
    "text",
    [
        "plain reply, nothing to strip.",
        "",
        # mid-reply mentions are the model's own words, not a leak
        "the stop token </s> is how a model ends a turn, I was saying.",
        'a "ts": "field" in prose, no timestamp.',
        # a JSON example whose ts is not an ISO date is left alone
        'config looks like {"a": "x", "ts": "soon"}',
    ],
)
def test_leaves_clean_text_untouched(text: str) -> None:
    assert strip_scaffold_tail(text) == text


# ── through the live path: respond() ─────────────────────────────────────────


class _LeakyProvider(LLMProvider):
    def name(self) -> str:
        return "leaky"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "ok"

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        leaked = 'good outcome at least.", "ts": "2026-09-08T00:10:55-04:00"}\n</s>'
        return ChatResponse(content=leaked, tool_calls=())


def test_respond_returns_and_persists_clean_reply(tmp_path: Path) -> None:
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
            provider=_LeakyProvider(),
            voice_md_override="# Nell",
        )
    finally:
        store.close()
        hebbian.close()
        reset_registry()

    assert result.content == "good outcome at least."
    rows = [
        json.loads(line)
        for p in (persona_dir / "active_conversations").glob("*.jsonl")
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    replies = [r["text"] for r in rows if r.get("speaker") == "assistant"]
    assert replies == ["good outcome at least."]  # replay never sees the leaked tail
