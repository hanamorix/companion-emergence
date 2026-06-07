"""Tests for the history-window truncation helper in brain.chat.engine."""

from __future__ import annotations

from brain.bridge.chat import ChatMessage
from brain.chat.engine import _HISTORY_WINDOW_MSGS, _window_history


def test_history_truncated_to_window():
    msgs = [ChatMessage(role=("user" if i % 2 == 0 else "assistant"), content=f"m{i}") for i in range(200)]
    out = _window_history(msgs)
    assert len(out) == _HISTORY_WINDOW_MSGS
    assert out[-1].content == "m199"  # most-recent kept
    assert out[0].content == f"m{200 - _HISTORY_WINDOW_MSGS}"  # oldest kept is window-start


def test_history_under_window_unchanged():
    msgs = [ChatMessage(role="user", content=f"m{i}") for i in range(10)]
    assert _window_history(msgs) == msgs


def test_window_is_about_40_turns():
    # 80 messages ≈ 40 user+assistant turns
    assert _HISTORY_WINDOW_MSGS == 80


def test_window_strips_leading_assistant():
    # Build a list where the raw [-_HISTORY_WINDOW_MSGS:] slice starts with an
    # assistant message.  With n = window+1 and alternating assistant/user
    # starting at index 0, index (n - window) = 1 has role "user".  Shift by
    # one more: use n = window+2 with the same pattern so the slice-start index
    # (n - window) = 2 lands on an assistant turn (even index → assistant).
    n = _HISTORY_WINDOW_MSGS + 2
    msgs = [
        ChatMessage(role=("assistant" if i % 2 == 0 else "user"), content=f"m{i}")
        for i in range(n)
    ]
    raw_slice = msgs[-_HISTORY_WINDOW_MSGS:]
    assert raw_slice[0].role == "assistant", (
        "test construction error: raw slice must start on assistant for the strip to be exercised"
    )
    out = _window_history(msgs)
    assert out[0].role == "user"  # leading assistant stripped
