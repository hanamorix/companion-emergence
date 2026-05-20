"""ChatStreamEvent discriminated union — shape + kind tags."""

from brain.bridge.chat import StreamDone, StreamError, TextDelta, ToolCallEvent


def test_text_delta_kind():
    assert TextDelta(text="hi").kind == "text_delta"


def test_text_delta_text_default_empty():
    assert TextDelta().text == ""


def test_tool_call_kind():
    assert ToolCallEvent(name="WebSearch").kind == "tool_call"


def test_tool_call_arguments_default_empty_dict():
    ev = ToolCallEvent(name="WebSearch")
    assert ev.arguments == {}


def test_done_kind():
    assert StreamDone(content="x").kind == "done"


def test_done_metadata_default_empty_dict():
    assert StreamDone(content="x").metadata == {}


def test_error_kind():
    assert StreamError(stage="claude_cli_timeout").kind == "error"


def test_error_detail_default_empty():
    assert StreamError(stage="x").detail == ""
