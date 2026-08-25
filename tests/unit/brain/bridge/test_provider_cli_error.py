"""#119 — a CLI failure reported *inside* a result frame is not a reply.

The Claude CLI can exit 0 and report the failure in the JSON payload:

    {"is_error": true, "result": "Not logged in · Please run /login"}

The non-zero-exit guard never sees those. `_cli_error_detail` was added for #92
and wired into `chat()`, `_run_chat_stream()` and `_chat_with_mcp_tools()` — but
two paths were missed, and each turns a system error into content:

* `generate()` — the Haiku path (attunement detector, emotion backfill). The
  error string is handed back as the call's output and parsed as if it were the
  detector's classification JSON.
* `_parse_stream_json_result()` — the image path's parser, and the single route
  `_chat_with_images()` takes. It special-cases only the over-budget frame; any
  other `is_error` frame falls through and is returned as the assistant reply,
  which the engine then persists as a turn the companion supposedly said.

The over-budget frame stays excluded on purpose: it carries real partial work
and has its own graceful path at each call site.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from brain.bridge.provider import ClaudeCliProvider


def test_generate_raises_on_an_is_error_frame_instead_of_returning_it():
    """The Haiku path must not hand an auth error back as its output."""
    mock_result = MagicMock()
    mock_result.returncode = 0  # the CLI exits 0 and reports failure in-band
    mock_result.stdout = json.dumps({
        "is_error": True,
        "result": "Not logged in · Please run /login",
    })
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        p = ClaudeCliProvider(model="claude-haiku-4-5-20251001")
        with pytest.raises(Exception) as exc_info:  # noqa: PT011 — see assert below
            p.generate("classify this", system="you are a detector")

    assert "Not logged in" in str(exc_info.value), (
        "the CLI error text must surface as a raised failure, not be returned "
        f"as the generation: {exc_info.value!r}"
    )


def test_stream_json_parser_raises_on_an_is_error_result_frame():
    """The image path's parser must not return an error frame as the reply.

    `_chat_with_images` routes its whole output through this function, so the
    check belongs here rather than at the call site — one guard covers the path.
    """
    from brain.bridge.provider import _parse_stream_json_result

    stdout = "\n".join([
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps({
            "type": "result",
            "is_error": True,
            "result": "API Error: 529 overloaded_error",
        }),
    ])

    with pytest.raises(Exception) as exc_info:  # noqa: PT011 — see assert below
        _parse_stream_json_result(stdout)

    assert "529" in str(exc_info.value), (
        "an is_error result frame was returned as the assistant reply, which "
        f"the engine then persists as a companion turn: {exc_info.value!r}"
    )
