"""Tests for substantive-turn skip-list."""
from brain.attunement.detector import should_run_detector
from brain.attunement.store import BufferTurn


def test_skips_when_buffer_empty():
    assert should_run_detector(buffer_slice=[], user_message="hi there friend", reply_text="hi back") is False


def test_skips_when_user_message_under_five_words():
    assert should_run_detector(
        buffer_slice=[BufferTurn(id="t1", content="ok")],
        user_message="ok",
        reply_text="hi back",
    ) is False


def test_skips_when_user_message_is_tool_only_empty():
    assert should_run_detector(
        buffer_slice=[BufferTurn(id="t1", content="")],
        user_message="",
        reply_text="anything",
    ) is False


def test_runs_when_user_message_has_substance():
    assert should_run_detector(
        buffer_slice=[BufferTurn(id="t1", content="I had a long day today.")],
        user_message="I had a long day today.",
        reply_text="oh love.",
    ) is True


def test_runs_with_whitespace_only_message_treated_as_empty():
    assert should_run_detector(
        buffer_slice=[BufferTurn(id="t1", content="   ")],
        user_message="   ",
        reply_text="x",
    ) is False
