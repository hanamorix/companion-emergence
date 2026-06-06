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
