"""read_file results must reach the model whole (#595).

The Claude CLI passes an MCP tool result to the model only up to 50,000 characters of its
text; anything longer is saved to a file and the model gets a 2 KB preview it cannot open
in chat (no Read tool). Measured on claude 2.1.321, docs/cli-provider-capabilities.md.
These tests go through the brain-tools MCP handler, so they measure the text the CLI gets.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools.impls import _read_cache

CLI_INLINE_CAP_CHARS = 50_000


@pytest.fixture
def call(tmp_path):
    from mcp.server import Server
    from mcp.types import CallToolRequest, CallToolRequestParams

    from brain.mcp_server.tools import register_tools

    _read_cache.reset()
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    server = Server("brain-tools")
    register_tools(
        server,
        persona_dir=persona_dir,
        store=MemoryStore(":memory:"),
        hebbian=HebbianMatrix(":memory:"),
        tool_prefix="mcp__brain-tools__",
    )
    handler = server.request_handlers[CallToolRequest]

    def _call(args: dict) -> tuple[str, dict]:
        req = CallToolRequest(method="tools/call", params=CallToolRequestParams(name="read_file", arguments=args))
        text = asyncio.run(handler(req)).root.content[0].text
        return text, json.loads(text)

    yield _call
    _read_cache.reset()


def _write(tmp_path, name: str, text: str):
    p = tmp_path / name
    p.write_bytes(text.encode("utf-8"))
    return p


def test_an_escape_heavy_default_read_reaches_the_model_whole(call, tmp_path):
    """Quotes, backslashes and tabs grow when the result is JSON-encoded for the CLI, so the
    window must be bounded by the encoded text, not the raw characters."""
    p = _write(tmp_path, "quotes.txt", "".join('"q" \\ "q"\t"q" \\ "q"\t' * 4 + "\n" for _ in range(3_000)))
    text, out = call({"path": str(p)})
    assert "content" in out, out
    assert len(text) <= CLI_INLINE_CAP_CHARS, f"{len(text)} characters would be persisted, not shown"
    assert out["truncated"] is True and "note" in out


def test_a_one_line_megabyte_file_reaches_the_model_whole(call, tmp_path):
    p = _write(tmp_path, "one_line.txt", "x" * 1_000_000)
    text, out = call({"path": str(p)})
    assert len(text) <= CLI_INLINE_CAP_CHARS
    assert "cut short" in out["note"]


def test_control_characters_are_counted_as_their_escapes(call, tmp_path):
    """A control character becomes six characters (\\u0001) in the JSON the CLI receives."""
    p = _write(tmp_path, "ctrl.txt", ("\x01\x02\x03\x04" * 30 + "\n") * 3_000)
    text, out = call({"path": str(p)})
    assert len(text) <= CLI_INLINE_CAP_CHARS
    assert out["truncated"] is True


def test_a_95_kb_file_is_read_window_by_window_inside_one_turn(call, tmp_path):
    """#595's acceptance: each window is under the cap, and the following windows, read with
    offset/max_lines from the note, come back in turn inside the 90-second dedup window."""
    body = "".join(f'{i:05d} "a quoted line" with a \\ backslash and more words to fill it out\n' for i in range(1_300))
    assert 90_000 < len(body.encode("utf-8")) < 100_000
    p = _write(tmp_path, "doc.txt", body)

    got, offset, calls = [], 0, 0
    while True:
        calls += 1
        assert calls < 50
        text, out = call({"path": str(p), "offset": offset, "max_lines": 400})
        assert len(text) <= CLI_INLINE_CAP_CHARS
        assert not out.get("deduped"), out
        got.append(out["content"])
        shown = re.search(r"showing lines (\d+)-(\d+) of (\d+)", out.get("note", ""))
        end = int(shown.group(2)) if shown else out["total_lines"]
        if end >= out["total_lines"]:
            break
        assert end > offset
        offset = end
    assert "".join(got) == body
    assert calls >= 4  # escaping makes 400 of these lines too big for one window


def test_an_identical_read_is_deduped_without_claiming_where_the_content_is(call, tmp_path):
    """#595: the dedup note said the content was "above", which the tool cannot know (it may
    never have reached the model). It says only what the tool knows, and how to go on."""
    p = _write(tmp_path, "small.txt", "hello\n")
    call({"path": str(p)})
    _text, again = call({"path": str(p)})
    assert again.get("deduped") is True
    assert "above" not in again["note"]
    assert "offset" in again["note"]
