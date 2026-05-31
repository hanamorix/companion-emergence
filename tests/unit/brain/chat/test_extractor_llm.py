"""Tests for the pass-2 LLM extraction call wrapper."""
from __future__ import annotations

import json

from brain.chat.extractor import ExtractorOutput, extract_from_thinking


class _FakeProvider:
    def __init__(self, output_text: str) -> None:
        self._out = output_text
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        return self._out

    def name(self) -> str:
        return "fake"


def test_returns_extractor_output_on_valid_json():
    provider = _FakeProvider(
        json.dumps(
            {
                "memory_writes": [{"episode": "noticed Loopy gap", "salience": 0.3}],
                "emotion_delta": {"curious": 0.1},
                "crystallisation": [],
                "reflex_audit": [],
            }
        )
    )
    out = extract_from_thinking(
        provider=provider,
        monologue_blocks=("thinking about Loopy",),
        visible_reply="I don't think I know Loopy.",
        recent_turn_context=("who's Loopy again?",),
    )
    assert isinstance(out, ExtractorOutput)
    assert out.memory_writes[0].episode == "noticed Loopy gap"
    assert len(provider.calls) == 1


def test_returns_empty_output_on_empty_thinking_blocks():
    """No LLM call when thinking is empty — pass 2 is a no-op."""
    provider = _FakeProvider("should not be called")
    out = extract_from_thinking(
        provider=provider,
        monologue_blocks=(),
        visible_reply="hi",
        recent_turn_context=(),
    )
    assert out == ExtractorOutput()
    assert provider.calls == []


def test_returns_empty_output_when_provider_raises():
    """Provider failure → empty output, caller responsible for logging."""

    class _BrokenProvider:
        def generate(self, prompt: str, *, system: str | None = None) -> str:
            raise RuntimeError("haiku borked")

        def name(self) -> str:
            return "broken"

    out = extract_from_thinking(
        provider=_BrokenProvider(),
        monologue_blocks=("anything",),
        visible_reply="hi",
        recent_turn_context=(),
    )
    assert out == ExtractorOutput()


def test_returns_empty_output_on_malformed_json():
    provider = _FakeProvider("not json at all")
    out = extract_from_thinking(
        provider=provider,
        monologue_blocks=("something",),
        visible_reply="hi",
        recent_turn_context=(),
    )
    assert out == ExtractorOutput()


def test_returns_empty_output_on_schema_violation():
    provider = _FakeProvider(
        json.dumps({"memory_writes": [{"episode": "x", "salience": 99.0}]})
    )
    out = extract_from_thinking(
        provider=provider,
        monologue_blocks=("something",),
        visible_reply="hi",
        recent_turn_context=(),
    )
    assert out == ExtractorOutput()


def test_system_prompt_names_the_schema():
    """The system prompt instructs the LLM what shape to return."""
    provider = _FakeProvider(json.dumps({}))
    extract_from_thinking(
        provider=provider,
        monologue_blocks=("x",),
        visible_reply="y",
        recent_turn_context=("z",),
    )
    _, system = provider.calls[0]
    assert system is not None
    assert "memory_writes" in system
    assert "emotion_delta" in system


def test_strips_json_code_fence():
    """Fence-stripping enables Ollama compatibility."""
    provider = _FakeProvider(
        "```json\n"
        '{"memory_writes": [{"episode": "fenced response", "salience": 0.4}]}\n'
        "```"
    )
    out = extract_from_thinking(
        provider=provider,
        monologue_blocks=("x",),
        visible_reply="y",
        recent_turn_context=(),
    )
    assert len(out.memory_writes) == 1
    assert out.memory_writes[0].episode == "fenced response"


def test_strips_bare_code_fence():
    """Code fence without language tag also stripped."""
    provider = _FakeProvider("```\n{}\n```")
    out = extract_from_thinking(
        provider=provider,
        monologue_blocks=("x",),
        visible_reply="y",
        recent_turn_context=(),
    )
    assert out == ExtractorOutput()


def test_user_prompt_uses_xml_delimiters():
    """Thinking blocks are wrapped so blank lines inside don't confuse the LLM."""
    provider = _FakeProvider("{}")
    extract_from_thinking(
        provider=provider,
        monologue_blocks=("first block\n\nwith blank line", "second block"),
        visible_reply="visible",
        recent_turn_context=("user said hi",),
    )
    prompt, _ = provider.calls[0]
    assert "<inner_monologue>" in prompt
    assert "</inner_monologue>" in prompt
    assert '<block n="1">' in prompt
    assert '<block n="2">' in prompt
    assert "<visible_reply>" in prompt
    assert "</visible_reply>" in prompt
    assert "<recent_user_messages>" in prompt
