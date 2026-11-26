"""#440 - PRE-FLIP check C-PF: does the CURRENT Claude CLI still reject an unknown model the way the brain expects?

Marked ``live`` and ``requires_claude_cli``: deselected by the default pre-check filter
(``-m "not live and not requires_claude_cli"``). The owner runs it once, on a machine with the current CLI,
before flipping PR #441:

    pytest tests/bridge/test_model_rejection_live.py -m "live" -s

It makes two real, token-free calls with an id that cannot exist (the CLI answers with a 404 and no cost). It
drives BOTH paths the feature depends on: ``generate()`` (the ``--output-format json`` argv the Apply-time check
uses) and ``chat_stream`` (the production streaming path), and compares the real sentence to the one pinned in the
committed fixture. A failure means the CLI reworded its rejection: update ``brain/bridge/model_error.py`` and the
fixture before flipping.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from brain.bridge import model_error
from brain.bridge.chat import ChatMessage, StreamError
from brain.bridge.provider import ClaudeCliProvider

pytestmark = [pytest.mark.live, pytest.mark.requires_claude_cli]

BOGUS = "claude-nonexistent-0"
FIXTURE = Path(__file__).parent / "fixtures" / "cli_2_1_219_model_not_found.ndjson"


def _pinned_sentence(model: str) -> str:
    frames = [json.loads(x) for x in FIXTURE.read_text(encoding="utf-8").splitlines() if x.strip()]
    template = next(f for f in frames if f.get("type") == "result")["result"]
    return template.replace("claude-sonet-5", model)


@pytest.fixture(scope="module", autouse=True)
def _cli_version():
    if shutil.which("claude") is None:
        pytest.skip("claude CLI not on PATH")
    out = subprocess.run(["claude", "--version"], capture_output=True, text=True, check=False).stdout.strip()
    print(f"\nclaude --version: {out}")
    return out


def test_generate_path_rejection_is_recognised_and_matches_the_pinned_sentence():
    with pytest.raises(RuntimeError) as err:
        ClaudeCliProvider(probe_model=BOGUS).generate("hi")
    text = str(err.value)
    assert model_error.is_model_rejection(text), text
    assert _pinned_sentence(BOGUS) in text, text
    assert model_error.rejected_model_name(text, None) == BOGUS
    # #448: the json argv carries the numeric status (it has no error code, only the stream-json frames do)
    assert "api_error_status=404" in text, text


def test_chat_stream_path_rejection_is_recognised_and_matches_the_pinned_sentence():
    events = list(ClaudeCliProvider(probe_model=BOGUS).chat_stream([ChatMessage(role="user", content="hi")]))
    errors = [e for e in events if isinstance(e, StreamError)]
    assert errors, events
    text = f"[{errors[0].stage}] {errors[0].detail}"
    assert model_error.is_model_rejection(text), text
    assert _pinned_sentence(BOGUS) in text, text
    assert model_error.rejected_model_name(text, None) == BOGUS
    # #448: the structured signals reach the detector on the streaming path too
    assert "api_error_status=404" in text and "error=model_not_found" in text, text


def test_save_time_probe_receives_the_model_not_found_code():
    """Owner decision D15: the picker's / `nell model set`'s check runs stream-json, so it gets the code."""
    with pytest.raises(Exception) as err:
        ClaudeCliProvider(probe_model=BOGUS).probe("hi")
    text = str(err.value)
    print(f"\nprobe() failure text: {text}")
    assert model_error.is_model_rejection(text), text
    assert "api_error_status=404" in text and "error=model_not_found" in text, text
    assert model_error.rejected_model_name(text, None) == BOGUS
