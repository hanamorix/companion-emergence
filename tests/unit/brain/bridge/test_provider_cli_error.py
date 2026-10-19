"""#119 — a CLI failure reported *inside* a result frame is not a reply.

The Claude CLI can exit 0 and report the failure in the JSON payload:

    {"is_error": true, "result": "Not logged in · Please run /login"}

The non-zero-exit guard never sees those. `_cli_error_detail` was added for #92
and wired into `chat()`, `_run_chat_stream()` and `_chat_with_mcp_tools()` — but
`generate()` (the Haiku path: attunement detector, emotion backfill) was missed:
the error string was handed back as the call's output and parsed as if it were
the detector's classification JSON. (A second missed path, the image path's
stream-json parser, was deleted with the image transport in #401.)

The over-budget frame stays excluded on purpose: it is a stop, not a failure, and
has its own graceful path at each call site.
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
