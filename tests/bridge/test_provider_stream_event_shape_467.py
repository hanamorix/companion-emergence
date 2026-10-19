"""#467 (+ the same-class frames found with it): `_run_chat_stream` must survive a valid frame whose nested value is
not the expected shape, and `null` must never become the literal text "None" in a reply.

`{"type":"stream_event","event":5}` parses fine, but the reader then called `.get` on the non-object `event`,
raising `AttributeError` out of the generator and ending the streaming reply. Same class one level down: a `delta`
and an `assistant` frame's `message`. A malformed frame is skipped and changes NOTHING: in particular a malformed
`assistant` frame must not erase the text snapshot an earlier valid frame stored. And `str(None)` put "None" into the
reply for `"text": null` / `"result": null`, on the streaming path and on the non-streaming ones (`generate()`, both
blocking `chat()` paths, the over-budget `partial`).

Cases that already pass on the old code are marked PIN.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from brain.bridge.chat import StreamDone, TextDelta
from brain.bridge.provider import ClaudeCliProvider, ProviderError, _claude_failure_detail_text
from tests.bridge.test_provider_builtin_tool_audit import _blocking, _persona, _stream, _text_branch

_INIT = {"type": "system", "subtype": "init"}
_RESULT = {"type": "result", "subtype": "success", "is_error": False, "result": "the reply"}
_BAD = pytest.mark.parametrize("bad", [5, None, [1], "x"], ids=["int", "null", "array", "string"])


def _delta(text) -> dict:
    return {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}}


def _assistant(text: str) -> dict:
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


def _events(tmp_path, frames: list[dict]) -> list:
    lines = [json.dumps(f) + "\n" for f in frames]
    return _stream(_persona(tmp_path), iter(lines))


def _texts(events) -> list[str]:
    return [e.text for e in events if isinstance(e, TextDelta)]


# ---------------------------------------------------------------------------
# stream_event / delta / assistant message shapes
# ---------------------------------------------------------------------------


@_BAD
def test_a_stream_event_whose_event_is_not_an_object_is_skipped(tmp_path, bad):
    # C467-a: `event.get("type")` raised AttributeError.
    events = _events(tmp_path, [_INIT, {"type": "stream_event", "event": bad}, _RESULT])

    assert isinstance(events[-1], StreamDone) and events[-1].content == "the reply"


def test_a_malformed_stream_event_costs_only_itself(tmp_path):
    # C467-b: the deltas around it are delivered, in order.
    frames = [_INIT, _delta("A "), {"type": "stream_event", "event": 5}, _delta("B"), _RESULT]

    events = _events(tmp_path, frames)

    assert _texts(events) == ["A ", "B"]
    assert isinstance(events[-1], StreamDone)


@_BAD
def test_a_content_block_delta_whose_delta_is_not_an_object_is_skipped(tmp_path, bad):
    # C467-d (FB-1): `delta.get("type")` raised AttributeError.
    bad_event = {"type": "stream_event", "event": {"type": "content_block_delta", "delta": bad}}

    events = _events(tmp_path, [_INIT, _delta("A "), bad_event, _delta("B"), _RESULT])

    assert _texts(events) == ["A ", "B"]
    assert isinstance(events[-1], StreamDone)


@_BAD
def test_an_assistant_frame_whose_message_is_not_an_object_is_skipped(tmp_path, bad):
    # C467-d (FB-1): `obj.get("message", {}).get(...)` raised AttributeError.
    events = _events(tmp_path, [_INIT, {"type": "assistant", "message": bad}, _RESULT])

    assert isinstance(events[-1], StreamDone) and events[-1].content == "the reply"


def test_a_malformed_assistant_frame_does_not_erase_the_stored_snapshot(tmp_path):
    # C467-e: valid text, then `message: 5`, then EOF with no result frame. The snapshot is the fallback reply; a skipped
    # frame must leave it alone (a `[]` fallback would run `"".join([])` and blank it).
    events = _events(tmp_path, [_INIT, _assistant("kept"), {"type": "assistant", "message": 5}])

    assert isinstance(events[-1], StreamDone) and events[-1].content == "kept"


def test_a_valid_assistant_message_without_content_still_resets_the_snapshot_as_before(tmp_path):
    # C467-f PIN: a VALID message with no text blocks has always replaced the snapshot with "".
    events = _events(tmp_path, [_INIT, _assistant("first"), {"type": "assistant", "message": {}}])

    assert isinstance(events[-1], StreamDone) and events[-1].content == ""


# ---------------------------------------------------------------------------
# an assistant text block whose text is not a string (the snapshot join)
# ---------------------------------------------------------------------------

_BAD_TEXT = pytest.mark.parametrize("bad", [None, 5], ids=["null", "int"])


def _assistant_blocks(*texts) -> dict:
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": t} for t in texts]}}


@_BAD_TEXT
def test_an_assistant_text_block_whose_text_is_not_a_string_contributes_nothing(tmp_path, bad):
    # C467-j: `"".join([... None ...])` raised TypeError out of the generator. Only string text is kept, as
    # `_tool_result_text` does; the frame's other blocks still count. EOF with no result frame, so the snapshot is the reply.
    events = _events(tmp_path, [_INIT, _assistant_blocks("kept ", bad, "on")])

    assert isinstance(events[-1], StreamDone) and events[-1].content == "kept on"


@_BAD_TEXT
def test_a_text_block_with_no_usable_text_after_an_earlier_snapshot_ends_the_stream_normally(tmp_path, bad):
    # C467-j: no crash after an earlier valid snapshot. The later frame is a VALID message whose only text block has no
    # usable text, so it replaces the snapshot with "" exactly as a valid message without text does (C467-f).
    events = _events(tmp_path, [_INIT, _assistant("earlier"), _assistant_blocks(bad)])

    assert isinstance(events[-1], StreamDone) and events[-1].content == ""


# ---------------------------------------------------------------------------
# null text must not become "None" (streaming)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [None, 5], ids=["null", "int"])
def test_a_text_delta_whose_text_is_not_a_string_contributes_nothing(tmp_path, text):
    # C467-h: `str(delta.get("text", ""))` streamed the literal "None".
    events = _events(tmp_path, [_INIT, _delta("A "), _delta(text), _delta("B"), _RESULT])

    assert _texts(events) == ["A ", "B"]


@pytest.mark.parametrize("result", [None, 5], ids=["null", "int"])
def test_a_result_that_is_not_a_string_falls_back_to_the_streamed_text(tmp_path, result):
    # C467-h: `str(obj.get("result", ""))` turned null into "None" and, being non-empty, beat the streamed text.
    frames = [_INIT, _delta("hello "), _delta("there"), {**_RESULT, "result": result}]

    events = _events(tmp_path, frames)

    assert isinstance(events[-1], StreamDone) and events[-1].content == "hello there"


def test_a_null_result_with_only_an_assistant_snapshot_uses_the_snapshot(tmp_path):
    # C467-h.
    events = _events(tmp_path, [_INIT, _assistant("snap"), {**_RESULT, "result": None}])

    assert events[-1].content == "snap"


def test_a_null_result_with_nothing_else_is_empty_not_none(tmp_path):
    # C467-h.
    events = _events(tmp_path, [_INIT, {**_RESULT, "result": None}])

    assert events[-1].content == ""


def test_a_string_result_is_used_as_before(tmp_path):
    # C467-h PIN.
    events = _events(tmp_path, [_INIT, _delta("streamed"), _RESULT])

    assert events[-1].content == "the reply"


# ---------------------------------------------------------------------------
# null result must not become "None" (non-streaming paths)
# ---------------------------------------------------------------------------

_NULL_RESULT = json.dumps({**_RESULT, "result": None})
_OVER_BUDGET = json.dumps(
    {"type": "result", "subtype": "error_max_budget_usd", "is_error": True, "result": None, "errors": ["Reached maximum budget"]}
)


def _generate(stdout: str) -> str:
    done = subprocess.CompletedProcess(["claude"], 0, stdout, "")
    with patch("brain.bridge.provider.subprocess.run", return_value=done):
        return ClaudeCliProvider(model="haiku").generate("hi")


def test_generate_with_a_null_result_returns_empty_not_none():
    # C467-i(1): `str(payload["result"])` returned "None".
    assert _generate(_NULL_RESULT) == ""


def test_generate_with_a_string_result_is_unchanged():
    # C467-i(1) PIN.
    assert _generate(json.dumps(_RESULT)) == "the reply"


def test_generate_without_a_result_key_is_still_the_parse_error():
    # C467-i(1) PIN: a MISSING key keeps raising (KeyError -> the existing "unexpected output format").
    with pytest.raises(RuntimeError, match="unexpected output format"):
        _generate(json.dumps({"type": "result", "subtype": "success", "is_error": False}))


def test_blocking_tools_chat_with_a_null_result_is_empty_not_none(tmp_path):
    # C467-i(2).
    resp, _ = _blocking(_persona(tmp_path), _NULL_RESULT)

    assert resp.content == ""


def test_blocking_text_chat_with_a_null_result_is_empty_not_none(tmp_path):
    # C467-i(2).
    resp, _, _ = _text_branch(_persona(tmp_path), _NULL_RESULT)

    assert resp.content == ""


def test_blocking_chats_without_a_result_key_are_still_the_parse_error(tmp_path):
    # C467-i(2) PIN.
    no_result = json.dumps({"type": "result", "subtype": "success", "is_error": False})
    with pytest.raises(ProviderError) as tools:
        _blocking(_persona(tmp_path / "tools"), no_result)
    with pytest.raises(ProviderError) as text:
        _text_branch(_persona(tmp_path / "text"), no_result)

    assert tools.value.stage == "claude_cli_parse" and text.value.stage == "claude_cli_parse"


def test_over_budget_tools_chat_with_a_null_result_has_no_leading_none(tmp_path):
    # C467-i(3): the `partial` text in front of the budget message was "None".
    resp, _ = _blocking(_persona(tmp_path), _OVER_BUDGET)

    assert not resp.content.startswith("None") and "None" not in resp.content


def test_over_budget_text_chat_with_a_null_result_has_no_leading_none(tmp_path):
    # C467-i(3).
    resp, _, _ = _text_branch(_persona(tmp_path), _OVER_BUDGET)

    assert not resp.content.startswith("None") and "None" not in resp.content


def test_failure_detail_for_an_error_frame_with_a_null_result_has_no_none():
    # C467-i(4) PIN: site `_claude_failure_detail_text` is guarded by `if payload.get("result")`.
    failed = subprocess.CompletedProcess(
        ["claude"], 1, json.dumps({"type": "result", "is_error": True, "api_error_status": 500, "result": None}), ""
    )

    assert "None" not in _claude_failure_detail_text(failed)
