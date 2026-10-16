"""#416: the inner-monologue frame reaches the model through the real chat path, and a
record_monologue call made in response lands in the digest.

Wiring guard only: a fake provider can't tell us whether a wording makes a real model call the
tool (that is measured live, docs/guarded-change/monologue-directive-sonnet-refusal). What this
pins is frame -> system prompt -> tool dispatch -> monologue_digest.jsonl through ``respond()``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from brain.bridge.chat import ChatMessage, ChatResponse, ToolCall
from brain.bridge.provider import LLMProvider
from brain.chat.engine import respond
from brain.chat.session import create_session, reset_registry
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

_FRAME_MARKER = "── inner monologue"


class _FrameObeyingProvider(LLMProvider):
    """Calls record_monologue only if the system prompt's monologue frame asks for it."""

    def __init__(self) -> None:
        self.calls = 0
        self.system_seen = ""

    def name(self) -> str:
        return "frame-obeying"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return json.dumps({"memory_writes": [], "emotion_delta": {}, "crystallisation": [], "reflex_audit": []})

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        self.calls += 1
        if self.calls == 1:
            self.system_seen = next(m.content for m in messages if m.role == "system")
            frame = self.system_seen.split(_FRAME_MARKER, 1)[1] if _FRAME_MARKER in self.system_seen else ""
            offered = {t.get("name") or t.get("function", {}).get("name") for t in tools or []}
            if "record_monologue" in frame and "record_monologue" in offered:
                return ChatResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="m1",
                            name="record_monologue",
                            arguments={
                                "monologue": "The sourdough finally moved. Small wins after a long day land hard.",
                                "feed_digest": "a long day, and the starter finally did something",
                            },
                        ),
                    ),
                )
        return ChatResponse(content="Tell me what it did.", tool_calls=())


@pytest.fixture(autouse=True)
def _reset_sessions():
    reset_registry()
    yield
    reset_registry()


# "fake" keeps today's directive; claude-cli + sonnet gets the softened one (#416 option B). The
# provider object is still the fake below, so no claude subprocess is spawned either way.
_SONNET = {"provider": "claude-cli", "model": "sonnet", "searcher": "noop"}
_FAKE = {"provider": "fake", "searcher": "noop"}


@pytest.fixture(params=[_FAKE, _SONNET], ids=["fake", "claude-cli-sonnet"])
def persona_dir(request, tmp_path: Path) -> Path:
    d = tmp_path / "personas" / "nell"
    d.mkdir(parents=True)
    (d / "persona_config.json").write_text(json.dumps(request.param), encoding="utf-8")
    return d


def _turn(persona_dir: Path, monkeypatch: pytest.MonkeyPatch, provider: _FrameObeyingProvider) -> Path:
    monkeypatch.setattr("brain.bridge.model_tier.build_tier_provider", lambda _p, _t: provider)
    store, hebbian = MemoryStore(db_path=":memory:"), HebbianMatrix(db_path=":memory:")
    try:
        respond(
            persona_dir,
            "hey. long day. the sourdough starter finally did something interesting",
            store=store,
            hebbian=hebbian,
            provider=provider,
            session=create_session(persona_dir.name),
            voice_md_override="# Nell",
        )
    finally:
        store.close()
        hebbian.close()
    return persona_dir / "monologue_digest.jsonl"


def test_frame_reaches_model_and_monologue_is_recorded(persona_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _FrameObeyingProvider()
    digest = _turn(persona_dir, monkeypatch, provider)
    assert _FRAME_MARKER in provider.system_seen
    sonnet = json.loads((persona_dir / "persona_config.json").read_text(encoding="utf-8")) == _SONNET
    assert ("nell has an inner life the conversation doesn't see" in provider.system_seen) is sonnet
    assert ("Your turn has two phases" in provider.system_seen) is not sonnet
    assert digest.exists(), "record_monologue made through respond() left no digest row"
    assert "starter finally did something" in digest.read_text(encoding="utf-8")


def test_oracle_fails_when_frame_drops_the_tool(persona_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shows the test above can fail: a frame that no longer names record_monologue -> no digest."""
    monkeypatch.setattr(
        "brain.chat.prompt.build_monologue_frame",
        lambda **kw: f"{_FRAME_MARKER}, {kw['persona_name']} ──\nno tool named here",
    )
    digest = _turn(persona_dir, monkeypatch, _FrameObeyingProvider())
    assert not digest.exists()
