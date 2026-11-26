"""#468 (+ the same-class paths found with it): the Ollama readers must survive a hostile or oddly shaped line.

`OllamaProvider.chat_stream` caught only `json.JSONDecodeError`, so a line nested past the recursion limit
(`RecursionError`) or a 5000-digit run (a plain `ValueError`) escaped it, and a line that parses to something other
than an object (`7`, `[1]`) raised `AttributeError` on `.get`. Ruled behaviour: an undecodable line keeps raising the
existing handled `ProviderError("ollama_parse")` (now for the shared `(ValueError, RecursionError)` tuple); a line that
parses to a non-object is skipped. The same class one level down: a frame whose `message` is not an object, a
non-string `content`, and in `chat()` a body, `message`, `tool_calls` or tool-call `arguments` of the wrong shape.

Cases that already pass on the old code are marked PIN.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from brain.bridge.chat import ChatMessage, ToolCall
from brain.bridge.provider import OllamaProvider, ProviderError

_DEEP = "[" * 200_000
_DIGITS = "1" * 5000
_HOSTILE = pytest.mark.parametrize("hostile", [_DEEP, _DIGITS], ids=["deep-nesting", "digit-run"])
_USER = [ChatMessage(role="user", content="hi")]


def _frame(text: str, done: bool = False) -> str:
    return json.dumps({"message": {"role": "assistant", "content": text}, "done": done})


def _done() -> str:
    return json.dumps({"message": {}, "done": True})


def _streaming_response(lines):
    response = MagicMock()
    response.status_code = 200
    response.iter_lines = MagicMock(return_value=iter(lines))
    response.read = MagicMock(return_value=b"")
    response.raise_for_status = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=response)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _chunks(lines) -> list:
    with patch("httpx.stream", return_value=_streaming_response(lines)):
        return list(OllamaProvider().chat_stream(_USER))


def _chat(body: str) -> object:
    """Drive the real `chat()` with a real `httpx.Response`, so `.json()` really parses `body` (a mock whose
    `.json()` returns a prebuilt object could not reach a RecursionError)."""
    resp = httpx.Response(200, text=body, request=httpx.Request("POST", "http://localhost:11434/api/chat"))
    with patch("httpx.post", return_value=resp):
        return OllamaProvider().chat(_USER)


# ---------------------------------------------------------------------------
# chat_stream: lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("line", ["7", "[1]", "null", '"x"'], ids=["number", "array", "null", "string"])
def test_stream_skips_a_json_line_that_is_not_an_object(line):
    # C468-a. `frame.get("message")` raised AttributeError for each of these.
    assert _chunks([_frame("A "), line, _frame("B"), _done()]) == ["A ", "B"]


@_HOSTILE
def test_stream_raises_the_handled_parse_error_for_a_hostile_line(hostile):
    # C468-b. RecursionError (deep) / bare ValueError (digit run) escaped the JSONDecodeError-only except.
    with pytest.raises(ProviderError) as excinfo:
        _chunks([_frame("A "), hostile, _done()])

    assert excinfo.value.stage == "ollama_parse"


def test_stream_still_raises_the_handled_parse_error_for_a_plain_non_json_line():
    # C468-c PIN: the documented contract ("A streamed line could not be JSON-decoded").
    with pytest.raises(ProviderError) as excinfo:
        _chunks([_frame("A "), "not json at all", _done()])

    assert excinfo.value.stage == "ollama_parse"


# ---------------------------------------------------------------------------
# chat_stream: frame shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"message": 5},
        {"message": [1]},
        {"message": "x"},
        {"message": {"content": 5}},
        {"message": {"content": ["x"]}},
    ],
    ids=["message-int", "message-list", "message-string", "content-int", "content-list"],
)
def test_stream_frame_with_a_malformed_message_yields_no_chunk(bad):
    # C468-d. `msg.get` raised AttributeError, or a non-string chunk was yielded to a caller that joins strings.
    assert _chunks([_frame("A "), json.dumps(bad), _frame("B"), _done()]) == ["A ", "B"]


def test_stream_done_is_honoured_on_a_frame_with_a_malformed_message():
    # C468-d: the bad frame yields nothing but its `done` still stops the stream.
    lines = [_frame("A "), json.dumps({"message": 5, "done": True}), _frame("ghost")]

    assert _chunks(lines) == ["A "]


@pytest.mark.parametrize("frame", [{"message": None, "done": False}, {"done": False}], ids=["message-null", "no-message"])
def test_stream_null_or_absent_message_is_an_empty_frame(frame):
    # C468-d PIN: `frame.get("message") or {}` already handled these.
    assert _chunks([_frame("A "), json.dumps(frame), _frame("B"), _done()]) == ["A ", "B"]


# ---------------------------------------------------------------------------
# chat(): body shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "[1]",
        "7",
        json.dumps({"message": 5}),
        json.dumps({"message": {"content": 5}}),
        json.dumps({"message": {"content": "x", "tool_calls": 5}}),
        _DEEP,
    ],
    ids=["list-body", "number-body", "message-int", "content-int", "tool-calls-int", "deep-body"],
)
def test_chat_raises_the_handled_parse_error_for_a_wrongly_shaped_body(body):
    # C468-e. AttributeError / TypeError / RecursionError escaped `chat()` before.
    with pytest.raises(ProviderError) as excinfo:
        _chat(body)

    assert excinfo.value.stage == "ollama_parse"


@pytest.mark.parametrize("falsy", ["", 0, [], False, None], ids=["empty-string", "zero", "empty-list", "false", "null"])
def test_chat_falsy_message_is_still_an_empty_response(falsy):
    # C468-e PIN: `data.get("message") or {}` accepts every falsy value as an empty message, as it always did.
    assert _chat(json.dumps({"message": falsy})).content == ""


def test_chat_valid_body_and_tool_call_still_parse():
    # C468-e PIN.
    body = {
        "message": {
            "content": "hello",
            "tool_calls": [{"id": "c1", "function": {"name": "search_memories", "arguments": {"q": "x"}}}],
        }
    }

    resp = _chat(json.dumps(body))

    assert resp.content == "hello"
    assert resp.tool_calls[0].name == "search_memories"


def test_chat_tool_call_arguments_nested_past_the_recursion_limit_is_a_parse_error_with_a_short_detail():
    # C468-e: a model-controlled `arguments` string. `from_provider_dict` only caught JSONDecodeError.
    body = {"message": {"content": "x", "tool_calls": [{"id": "c1", "function": {"name": "n", "arguments": _DEEP}}]}}

    with pytest.raises(ProviderError) as excinfo:
        _chat(json.dumps(body))

    assert excinfo.value.stage == "ollama_parse"
    assert len(str(excinfo.value)) < 1000  # a 200,000-character argument is not copied into the error


def test_chat_per_tool_call_handler_catches_a_recursion_error_from_the_parser():
    # C468-e: the handler itself catches the shared tuple, whatever the parser raises.
    body = {"message": {"content": "x", "tool_calls": [{"id": "c1", "function": {"name": "n", "arguments": {}}}]}}

    with patch.object(ToolCall, "from_provider_dict", side_effect=RecursionError("deep")):
        with pytest.raises(ProviderError) as excinfo:
            _chat(json.dumps(body))

    assert excinfo.value.stage == "ollama_parse"


@pytest.mark.parametrize("call", [
    {"id": 5, "function": {"name": "n", "arguments": {}}},
    {"id": "c1", "function": {"name": None, "arguments": {}}},
], ids=["id-int", "name-null"])
def test_chat_tool_call_with_a_non_string_id_or_name_is_a_parse_error(call):
    # #474 through chat(): rejected where it is read, as the handled ollama_parse.
    with pytest.raises(ProviderError) as excinfo:
        _chat(json.dumps({"message": {"content": "x", "tool_calls": [call]}}))

    assert excinfo.value.stage == "ollama_parse"


# ---------------------------------------------------------------------------
# chat.py: ToolCall.from_provider_dict
# ---------------------------------------------------------------------------


def test_from_provider_dict_deep_arguments_is_a_value_error_with_a_short_message():
    # C468-f. `except json.JSONDecodeError` let RecursionError out; the message embedded the whole string.
    with pytest.raises(ValueError) as excinfo:
        ToolCall.from_provider_dict({"id": "c1", "function": {"name": "n", "arguments": _DEEP}})

    assert len(str(excinfo.value)) < 600


def test_from_provider_dict_digit_run_arguments_is_a_value_error():
    # C468-f: a 5000-digit run is a ValueError that is not a JSONDecodeError; same message path.
    with pytest.raises(ValueError):
        ToolCall.from_provider_dict({"id": "c1", "function": {"name": "n", "arguments": _DIGITS}})


def test_from_provider_dict_message_builder_survives_a_dict_nested_past_the_recursion_limit():
    # C468-f: `{d!r}` in the message raised RecursionError inside the except clause that builds it.
    nested: list = []
    for _ in range(200_000):
        nested = [nested]

    with pytest.raises(ValueError) as excinfo:
        ToolCall.from_provider_dict({"id": "c1", "function": nested})

    assert len(str(excinfo.value)) < 600


def test_from_provider_dict_valid_calls_are_unchanged():
    # C468-f PIN.
    call = ToolCall.from_provider_dict({"id": "c1", "function": {"name": "n", "arguments": '{"a": 1}'}})

    assert call.arguments == {"a": 1}
