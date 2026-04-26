"""Tests for brain.ingest.extract — EXTRACT + SCORE stages."""

from __future__ import annotations

from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.ingest.extract import extract_items, format_transcript, parse_extraction

# ---------------------------------------------------------------------------
# Fake providers for extraction tests
# ---------------------------------------------------------------------------


class _ValidJsonProvider(LLMProvider):
    """Returns a canned valid JSON array from generate()."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self._payload

    def name(self) -> str:
        return "fake-valid-json"

    def chat(self, messages: list[ChatMessage], *, tools=None, options=None) -> ChatResponse:
        return ChatResponse(content=self._payload, tool_calls=())


class _GarbageProvider(LLMProvider):
    """Always returns garbage — forces parse failure."""

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "this is definitely not json {]"

    def name(self) -> str:
        return "fake-garbage"

    def chat(self, messages: list[ChatMessage], *, tools=None, options=None) -> ChatResponse:
        return ChatResponse(content="garbage", tool_calls=())


class _FailingProvider(LLMProvider):
    """Always raises on generate() — simulates network failure."""

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise RuntimeError("provider exploded")

    def name(self) -> str:
        return "fake-failing"

    def chat(self, messages: list[ChatMessage], *, tools=None, options=None) -> ChatResponse:
        raise RuntimeError("provider exploded")


# ---------------------------------------------------------------------------
# format_transcript
# ---------------------------------------------------------------------------


def test_format_transcript_joins_turns_as_speaker_colon_text() -> None:
    """format_transcript produces 'speaker: text' lines joined by newline."""
    turns = [
        {"speaker": "Hana", "text": "hello there"},
        {"speaker": "Nell", "text": "hey love"},
    ]
    result = format_transcript(turns)
    assert result == "Hana: hello there\nNell: hey love"


def test_format_transcript_caps_long_transcript_keeps_tail() -> None:
    """format_transcript caps at max_tokens*4 chars and keeps the tail (recent context)."""
    # Create a transcript that exceeds the cap.
    long_text = "x" * 100
    turns = [{"speaker": "A", "text": long_text} for _ in range(10)]
    result = format_transcript(turns, max_tokens=50)  # cap = 200 chars
    assert len(result) <= 200
    # The tail should end with the last turn.
    assert result.endswith(f"A: {long_text}"[:200]) or result.endswith(long_text[-10:])


# ---------------------------------------------------------------------------
# parse_extraction
# ---------------------------------------------------------------------------


def test_parse_extraction_handles_bare_json_array() -> None:
    """parse_extraction parses a clean JSON array."""
    raw = '[{"text": "Nell loves Hana", "label": "feeling", "importance": 9}]'
    result = parse_extraction(raw)
    assert result is not None
    assert len(result) == 1
    assert result[0].text == "Nell loves Hana"
    assert result[0].label == "feeling"
    assert result[0].importance == 9


def test_parse_extraction_strips_code_fences() -> None:
    """parse_extraction handles ```json ... ``` wrapping from the LLM."""
    raw = '```json\n[{"text": "important fact", "label": "fact", "importance": 7}]\n```'
    result = parse_extraction(raw)
    assert result is not None
    assert len(result) == 1
    assert result[0].text == "important fact"


def test_parse_extraction_strips_plain_code_fences() -> None:
    """parse_extraction handles ``` (no language tag) wrapping."""
    raw = '```\n[{"text": "a decision", "label": "decision", "importance": 6}]\n```'
    result = parse_extraction(raw)
    assert result is not None
    assert result[0].label == "decision"


def test_parse_extraction_returns_none_on_malformed() -> None:
    """parse_extraction returns None when JSON cannot be parsed."""
    result = parse_extraction("this is not json at all {]")
    assert result is None


def test_parse_extraction_returns_none_on_empty_input() -> None:
    """parse_extraction returns None on empty/None input."""
    assert parse_extraction(None) is None
    assert parse_extraction("") is None


def test_parse_extraction_returns_empty_list_for_empty_array() -> None:
    """parse_extraction returns [] when the LLM says no memories extracted."""
    result = parse_extraction("[]")
    assert result == []


def test_parse_extraction_handles_prose_before_array() -> None:
    """parse_extraction finds the array even when prose precedes it."""
    raw = 'Here are the extracted memories:\n[{"text": "deep thought", "label": "observation", "importance": 5}]'
    result = parse_extraction(raw)
    assert result is not None
    assert result[0].text == "deep thought"


# ---------------------------------------------------------------------------
# extract_items
# ---------------------------------------------------------------------------


def test_extract_items_succeeds_on_first_try() -> None:
    """extract_items calls generate() once and returns parsed items."""
    payload = '[{"text": "Nell is a novelist", "label": "fact", "importance": 6}]'
    provider = _ValidJsonProvider(payload)
    items = extract_items("Hana: tell me about yourself\nNell: I write", provider=provider)
    assert len(items) == 1
    assert items[0].text == "Nell is a novelist"


def test_extract_items_returns_empty_on_empty_transcript() -> None:
    """extract_items returns [] without calling the provider for empty transcripts."""
    provider = _GarbageProvider()
    items = extract_items("   ", provider=provider)
    assert items == []


def test_extract_items_retries_once_on_bad_output_then_returns_empty() -> None:
    """extract_items retries max_retries times then gives up with []."""
    provider = _GarbageProvider()
    items = extract_items("Hana: something happened", provider=provider, max_retries=1)
    assert items == []


def test_extract_items_returns_empty_when_provider_raises() -> None:
    """extract_items handles provider exceptions gracefully and returns []."""
    provider = _FailingProvider()
    items = extract_items("Hana: test", provider=provider, max_retries=0)
    assert items == []
