"""Tests for research_notes.py — per-interest markdown notes store."""
from datetime import UTC, datetime

from brain.engines.research_notes import (
    append_session_notes,
    read_notes_tail,
)

D1 = datetime(2026, 7, 11, tzinfo=UTC)
D2 = datetime(2026, 7, 13, tzinfo=UTC)


def test_append_creates_file_with_header(tmp_path):
    append_session_notes(tmp_path, "abc", "- fact one", now=datetime(2026, 7, 13, tzinfo=UTC))
    text = (tmp_path / "research" / "abc.md").read_text(encoding="utf-8")
    assert "## Session 2026-07-13" in text and "- fact one" in text


def test_append_accumulates(tmp_path):
    append_session_notes(tmp_path, "abc", "first", now=D1)
    append_session_notes(tmp_path, "abc", "second", now=D2)
    text = (tmp_path / "research" / "abc.md").read_text(encoding="utf-8")
    assert text.index("first") < text.index("second")


def test_read_tail_empty_when_missing(tmp_path):
    assert read_notes_tail(tmp_path, "nope") == ""


def test_read_tail_caps_and_aligns_to_session_header(tmp_path):
    append_session_notes(tmp_path, "abc", "x" * 5000, now=D1)
    append_session_notes(tmp_path, "abc", "recent", now=D2)
    tail = read_notes_tail(tmp_path, "abc", max_chars=200)
    assert len(tail) <= 200 and "recent" in tail
    assert tail.startswith("## Session")


def test_interest_id_is_sanitised(tmp_path):
    append_session_notes(tmp_path, "../evil", "n", now=D1)
    assert not (tmp_path.parent / "evil.md").exists()
    assert (tmp_path / "research").exists()
