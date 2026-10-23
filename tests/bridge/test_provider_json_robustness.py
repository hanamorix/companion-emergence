"""#400: the provider's JSON line readers must survive a hostile line, not just a malformed one.

`json.loads` raises `RecursionError` on a line nested beyond the recursion limit (about 5000 levels) and a
plain `ValueError` (not `JSONDecodeError`) on a 5000-digit run. Every reader caught only `JSONDecodeError`, so
one such line escaped `chat()`, `generate()` or the stream generator. Position matters per site (the scanners
read in a particular order and stop at the result frame), so each test puts the hostile line where the
OLD code parses it.
"""

from __future__ import annotations

import json

import pytest

from brain.bridge.provider import (
    ClaudeCliProvider,
    ProviderError,
    _claude_failure_detail_text,
    _last_result_line,
    _scan_blocking_stream_stdout,
)

_DEEP = "[" * 200_000
_DIGITS = "1" * 5000
_HOSTILE = pytest.mark.parametrize("hostile", [_DEEP, _DIGITS], ids=["deep-nesting", "digit-run"])
_RESULT = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "the reply"})
_INIT = json.dumps({"type": "system", "subtype": "init"})


@_HOSTILE
def test_scanner_per_line_site_skips_a_hostile_line_before_the_result(hostile):
    # Reads every line in order, so the hostile line must come first.
    payload = _scan_blocking_stream_stdout(f"{_INIT}\n{hostile}\n{_RESULT}\n", {})

    assert payload is not None and payload["result"] == "the reply"


@_HOSTILE
def test_scanner_whole_stdout_fallback_survives_a_hostile_first_line(hostile):
    # The fallback `json.loads(stdout)` only runs when no result frame was found.
    assert _scan_blocking_stream_stdout(f"{hostile}\n{_INIT}\n", {}) is None


@_HOSTILE
def test_last_result_line_skips_a_hostile_line_after_the_result(hostile):
    # Scans in reverse and returns at the first result line, so the hostile line must come after it.
    frame = _last_result_line(f"{_INIT}\n{_RESULT}\n{hostile}\n")

    assert frame is not None and frame["result"] == "the reply"


@_HOSTILE
@pytest.mark.parametrize("position", ["first", "after-the-result"])
def test_failure_detail_survives_a_hostile_stdout(hostile, position):
    # Two readers, two placements: the whole-stdout `json.loads` only parses a hostile value that comes FIRST
    # (after a complete first value it stops at "Extra data"); `_last_result_line` scans in reverse, so it only
    # reaches a hostile line that comes AFTER the result.
    import subprocess

    stdout = f"{hostile}\n{_RESULT}\n" if position == "first" else f"{_RESULT}\n{hostile}\n"
    failed = subprocess.CompletedProcess(["claude"], 1, stdout, "")

    assert isinstance(_claude_failure_detail_text(failed), str)


@_HOSTILE
def test_generate_with_a_hostile_stdout_raises_its_normal_error(hostile):
    import subprocess
    from unittest.mock import patch

    done = subprocess.CompletedProcess(["claude"], 0, f"{hostile}\n{_RESULT}\n", "")
    with patch("brain.bridge.provider.subprocess.run", return_value=done):
        with pytest.raises(RuntimeError, match="unexpected output format"):
            ClaudeCliProvider(probe_model="haiku").generate("hi")


@_HOSTILE
@pytest.mark.parametrize("path", ["tools", "text"])
def test_blocking_chat_survives_a_hostile_line_before_the_result(tmp_path, path, hostile):
    from tests.bridge.test_provider_builtin_tool_audit import _blocking, _persona, _text_branch

    stdout = f"{_INIT}\n{hostile}\n{_RESULT}\n"
    if path == "tools":
        resp, _ = _blocking(_persona(tmp_path), stdout)
    else:
        resp, _, _ = _text_branch(_persona(tmp_path), stdout)

    assert resp.content == "the reply"


@_HOSTILE
def test_blocking_chat_without_a_result_is_a_parse_error_not_an_escape(tmp_path, hostile):
    from tests.bridge.test_provider_builtin_tool_audit import _blocking, _persona

    with pytest.raises(ProviderError) as excinfo:
        _blocking(_persona(tmp_path), f"{hostile}\n{_INIT}\n")

    assert excinfo.value.stage == "claude_cli_parse"


@_HOSTILE
def test_stream_skips_a_hostile_line_before_the_result(tmp_path, hostile):
    # `_run_chat_stream` breaks at the result frame, so the hostile line must come before it.
    from brain.bridge.chat import StreamDone
    from tests.bridge.test_provider_builtin_tool_audit import _persona, _stream

    events = _stream(_persona(tmp_path), iter([f"{_INIT}\n", f"{hostile}\n", f"{_RESULT}\n"]))

    assert isinstance(events[-1], StreamDone) and events[-1].content == "the reply"


@pytest.mark.parametrize("line", ["null", "[1]", '"text"', "7"], ids=["null", "array", "string", "number"])
def test_stream_skips_a_json_line_that_is_not_an_object(tmp_path, line):
    # `obj.get("type")` raised AttributeError out of the generator for any of these.
    from brain.bridge.chat import StreamDone
    from tests.bridge.test_provider_builtin_tool_audit import _persona, _stream

    events = _stream(_persona(tmp_path), iter([f"{_INIT}\n", f"{line}\n", f"{_RESULT}\n"]))

    assert isinstance(events[-1], StreamDone) and events[-1].content == "the reply"


@pytest.mark.parametrize("stdout", ["[1]", "null", '"text"', "7"], ids=["array", "null", "string", "number"])
def test_generate_with_a_json_stdout_that_is_not_an_object_raises_its_normal_error(stdout):
    import subprocess
    from unittest.mock import patch

    done = subprocess.CompletedProcess(["claude"], 0, stdout, "")
    with patch("brain.bridge.provider.subprocess.run", return_value=done):
        with pytest.raises(RuntimeError, match="unexpected output format"):
            ClaudeCliProvider(probe_model="haiku").generate("hi")


@_HOSTILE
def test_audit_log_reader_skips_a_hostile_line(tmp_path, hostile):
    from brain.bridge.provider import _read_audit_lines_since

    log = tmp_path / "tool_invocations.log.jsonl"
    good = json.dumps({"name": "search_memories", "arguments": {}, "result_summary": "{}"})
    log.write_text(f"{hostile}\n{good}\n", encoding="utf-8")

    assert [r["name"] for r in _read_audit_lines_since(log, 0)] == ["search_memories"]


@pytest.mark.parametrize("line", ["null", "[1]", '"text"', "7"], ids=["null", "array", "string", "number"])
def test_audit_log_reader_skips_a_json_line_that_is_not_an_object(tmp_path, line):
    # `entry.get("origin")` raised AttributeError out of the reader, and from there out of chat() on the tools
    # branch; the function's own contract is that telemetry never breaks a chat response.
    from brain.bridge.provider import _read_audit_lines_since

    log = tmp_path / "tool_invocations.log.jsonl"
    good = json.dumps({"name": "search_memories", "arguments": {}, "result_summary": "{}"})
    log.write_text(f"{line}\n{good}\n", encoding="utf-8")

    assert [r["name"] for r in _read_audit_lines_since(log, 0)] == ["search_memories"]
