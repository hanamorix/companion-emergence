"""Tests for brain.utils.emotion."""

from __future__ import annotations

from brain.utils.emotion import format_emotion_summary


def test_format_emotion_summary_empty():
    assert format_emotion_summary({}) == ""


def test_format_emotion_summary_top_5_descending():
    emotions = {
        "love": 8.5,
        "tenderness": 7.1,
        "defiance": 3.0,
        "creative_hunger": 6.2,
        "grief": 5.0,
        "awe": 2.0,
    }
    result = format_emotion_summary(emotions)
    lines = result.split("\n")
    assert len(lines) == 5
    assert lines[0] == "- love: 8.5/10"
    assert lines[1] == "- tenderness: 7.1/10"


def test_format_emotion_summary_fewer_than_5():
    emotions = {"love": 6.0, "defiance": 3.0}
    result = format_emotion_summary(emotions)
    assert result == "- love: 6.0/10\n- defiance: 3.0/10"
