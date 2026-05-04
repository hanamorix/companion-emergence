"""Tests for the four works MCP tools."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.tools.impls.list_works import list_works
from brain.tools.impls.read_work import read_work
from brain.tools.impls.save_work import save_work
from brain.tools.impls.search_works import search_works


def _persona(tmp_path: Path) -> Path:
    p = tmp_path / "persona"
    p.mkdir()
    return p


# ---------- save_work ----------


def test_save_work_creates_index_row_and_file(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    result = save_work(
        title="The Lighthouse Keeper's Daughter",
        type="story",
        content="Once upon a time...",
        summary="A short story.",
        persona_dir=persona_dir,
    )
    assert "id" in result
    work_id = result["id"]
    assert (persona_dir / "data" / "works" / f"{work_id}.md").exists()
    assert (persona_dir / "data" / "works.db").exists()


def test_save_work_rejects_invalid_type(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    result = save_work(
        title="Bad",
        type="not_a_real_type",
        content="content",
        persona_dir=persona_dir,
    )
    assert "error" in result
    assert "type" in result["error"].lower()


def test_save_work_rejects_empty_title(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    result = save_work(title="", type="story", content="content", persona_dir=persona_dir)
    assert "error" in result
    assert "title" in result["error"].lower()


def test_save_work_rejects_empty_content(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    result = save_work(title="A title", type="story", content="", persona_dir=persona_dir)
    assert "error" in result
    assert "content" in result["error"].lower()


def test_save_work_dedupes_same_content(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    r1 = save_work(title="t", type="idea", content="same content", persona_dir=persona_dir)
    r2 = save_work(title="t", type="idea", content="same content", persona_dir=persona_dir)
    assert r1["id"] == r2["id"]


def test_save_work_word_count_is_recorded(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    content = "one two three four five six"
    result = save_work(
        title="t", type="idea", content=content, persona_dir=persona_dir
    )
    listed = list_works(persona_dir=persona_dir)
    assert listed[0]["word_count"] == 6


# ---------- list_works ----------


def test_list_works_empty_returns_empty_list(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    assert list_works(persona_dir=persona_dir) == []


def test_list_works_returns_recent_first(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    save_work(title="first", type="idea", content="alpha", persona_dir=persona_dir)
    save_work(title="second", type="idea", content="beta", persona_dir=persona_dir)
    listed = list_works(persona_dir=persona_dir)
    assert listed[0]["title"] == "second"
    assert listed[1]["title"] == "first"


def test_list_works_filters_by_type(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    save_work(title="story1", type="story", content="alpha", persona_dir=persona_dir)
    save_work(title="code1", type="code", content="beta", persona_dir=persona_dir)
    stories = list_works(type="story", persona_dir=persona_dir)
    assert len(stories) == 1
    assert stories[0]["title"] == "story1"


# ---------- search_works ----------


def test_search_works_finds_by_content(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    save_work(
        title="A note", type="idea", content="chrysanthemums in autumn",
        persona_dir=persona_dir,
    )
    matches = search_works(query="chrysanthemums", persona_dir=persona_dir)
    assert len(matches) == 1
    assert matches[0]["title"] == "A note"


def test_search_works_returns_empty_on_no_match(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    save_work(title="t", type="idea", content="alpha", persona_dir=persona_dir)
    assert search_works(query="zzzz", persona_dir=persona_dir) == []


# ---------- read_work ----------


def test_read_work_returns_full_content(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    saved = save_work(
        title="t", type="story", content="full content body here",
        summary="my summary", persona_dir=persona_dir,
    )
    fetched = read_work(id=saved["id"], persona_dir=persona_dir)
    assert fetched["content"] == "full content body here"
    assert fetched["title"] == "t"
    assert fetched["summary"] == "my summary"


def test_read_work_missing_id_returns_error(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    result = read_work(id="zzzzzzzzzzzz", persona_dir=persona_dir)
    assert "error" in result
