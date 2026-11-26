"""#467 applied to #450's `ClaudeCliProvider.probe`: a `null` (or non-string) `result` in the result frame is not text.

`probe` returned `str(payload.get("result", ""))`, so a frame whose `result` is JSON `null` made the probe answer the literal word
"None" (the defect #467 fixed in the streaming, `generate()` and blocking chat paths). It reads the field with `_text_only` now.
"""

from __future__ import annotations

import subprocess

import pytest

from brain.bridge.provider import ClaudeCliProvider


def _probe_with_result_frame(monkeypatch, frame: dict) -> str:
    done = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(ClaudeCliProvider, "_run_blocking_cli", lambda self, cmd, prompt, calls: (done, frame))
    return ClaudeCliProvider(probe_model="haiku", timeout_seconds=30).probe("ping")


@pytest.mark.parametrize("value", [None, 5, 1.5, ["a"], {"x": 1}, True])
def test_a_result_that_is_not_text_is_empty_not_the_word_none(monkeypatch, value):
    assert _probe_with_result_frame(monkeypatch, {"type": "result", "is_error": False, "result": value}) == ""


def test_a_missing_result_is_empty(monkeypatch):
    assert _probe_with_result_frame(monkeypatch, {"type": "result", "is_error": False}) == ""


def test_a_text_result_is_returned_unchanged(monkeypatch):
    assert _probe_with_result_frame(monkeypatch, {"type": "result", "is_error": False, "result": "pong"}) == "pong"
