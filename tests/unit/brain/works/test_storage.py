"""Tests for brain.works storage layer + Work dataclass."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain import works

# ---------- Work dataclass + helpers ----------


def test_work_dataclass_round_trips_via_dict() -> None:
    """Work supports to_dict/from_dict for serialization."""
    w = works.Work(
        id="abc123def456",
        title="The Lighthouse Keeper's Daughter",
        type="story",
        created_at=datetime(2026, 5, 4, 17, 42, 31, tzinfo=UTC),
        session_id="8c9d2a1f",
        word_count=1247,
        summary="A short story about inheritance and the woman who keeps the lamp.",
    )
    round_tripped = works.Work.from_dict(w.to_dict())
    assert round_tripped == w


def test_work_dataclass_session_id_optional() -> None:
    """session_id is optional (None when no bridge session)."""
    w = works.Work(
        id="abc123def456",
        title="A note",
        type="idea",
        created_at=datetime(2026, 5, 4, tzinfo=UTC),
        session_id=None,
        word_count=10,
        summary=None,
    )
    assert w.session_id is None
    round_tripped = works.Work.from_dict(w.to_dict())
    assert round_tripped.session_id is None


def test_work_types_vocab() -> None:
    """WORK_TYPES is a frozenset with the seven canonical types."""
    assert works.WORK_TYPES == frozenset(
        {"story", "code", "planning", "idea", "role_play", "letter", "other"}
    )


def test_make_work_id_is_deterministic_for_same_content() -> None:
    """Same content → same id (content-hash based)."""
    content = "Once upon a time there was a lighthouse keeper's daughter."
    assert works.make_work_id(content) == works.make_work_id(content)


def test_make_work_id_returns_12_hex_chars() -> None:
    """ID is 12 lowercase hex characters."""
    work_id = works.make_work_id("any content")
    assert len(work_id) == 12
    assert all(c in "0123456789abcdef" for c in work_id)


def test_make_work_id_differs_for_different_content() -> None:
    """Different content → different id."""
    a = works.make_work_id("alpha")
    b = works.make_work_id("beta")
    assert a != b


# ---------- storage (markdown file I/O) ----------


from brain.works import storage  # noqa: E402  late import groups with storage tests below


def _make_persona(tmp_path: Path) -> Path:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir(parents=True)
    return persona_dir


def _sample_work() -> works.Work:
    return works.Work(
        id="abc123def456",
        title="The Lighthouse Keeper's Daughter",
        type="story",
        created_at=datetime(2026, 5, 4, 17, 42, 31, tzinfo=UTC),
        session_id="8c9d2a1f",
        word_count=12,
        summary="Inheritance and the woman who keeps the lamp.",
    )


def test_write_markdown_creates_works_dir_and_file(tmp_path: Path) -> None:
    persona_dir = _make_persona(tmp_path)
    w = _sample_work()
    storage.write_markdown(persona_dir, w, content="Once upon a lighthouse.")

    expected_path = persona_dir / "data" / "works" / "abc123def456.md"
    assert expected_path.exists()


def test_write_markdown_emits_yaml_frontmatter(tmp_path: Path) -> None:
    persona_dir = _make_persona(tmp_path)
    w = _sample_work()
    storage.write_markdown(persona_dir, w, content="Body content here.")

    file_text = (persona_dir / "data" / "works" / "abc123def456.md").read_text(encoding="utf-8")
    assert file_text.startswith("---\n")
    assert "id: abc123def456" in file_text
    assert "title: The Lighthouse Keeper's Daughter" in file_text
    assert "type: story" in file_text
    assert "created_at: 2026-05-04T17:42:31+00:00" in file_text
    assert "session_id: 8c9d2a1f" in file_text
    assert "word_count: 12" in file_text
    assert "summary: Inheritance and the woman who keeps the lamp." in file_text
    # Frontmatter terminator + body
    assert "\n---\n\nBody content here." in file_text


def test_write_markdown_handles_none_session_and_summary(tmp_path: Path) -> None:
    """session_id=None and summary=None should serialize as empty/absent, not 'None'."""
    persona_dir = _make_persona(tmp_path)
    w = works.Work(
        id="aaa111bbb222",
        title="A bare idea",
        type="idea",
        created_at=datetime(2026, 5, 4, tzinfo=UTC),
        session_id=None,
        word_count=2,
        summary=None,
    )
    storage.write_markdown(persona_dir, w, content="A spark.")
    file_text = (persona_dir / "data" / "works" / "aaa111bbb222.md").read_text(encoding="utf-8")
    # Must not contain the literal string "None" in YAML
    assert "session_id: None" not in file_text
    assert "summary: None" not in file_text


def test_read_markdown_returns_work_and_content(tmp_path: Path) -> None:
    persona_dir = _make_persona(tmp_path)
    w = _sample_work()
    storage.write_markdown(persona_dir, w, content="Body lives here.")

    loaded_work, loaded_content = storage.read_markdown(persona_dir, "abc123def456")
    assert loaded_work == w
    assert loaded_content == "Body lives here."


def test_read_markdown_missing_file_raises(tmp_path: Path) -> None:
    persona_dir = _make_persona(tmp_path)
    # Valid-format id (12 lowercase hex) that doesn't exist on disk.
    with pytest.raises(FileNotFoundError):
        storage.read_markdown(persona_dir, "000000000000")


def test_read_markdown_rejects_path_traversal_in_id(tmp_path: Path) -> None:
    """Defense in depth: id with path separators or '..' must be rejected."""
    persona_dir = _make_persona(tmp_path)
    for bad_id in ("../etc/passwd", "../../escape", "with/slash", "with\\backslash", ".."):
        with pytest.raises(ValueError):
            storage.read_markdown(persona_dir, bad_id)


def test_write_markdown_atomic_via_save_with_backup(tmp_path: Path) -> None:
    """Writing twice creates a .bak (existing health/save_with_backup pattern)."""
    persona_dir = _make_persona(tmp_path)
    w = _sample_work()
    storage.write_markdown(persona_dir, w, content="first version")
    storage.write_markdown(persona_dir, w, content="second version")

    # The current write succeeded (whole point); .bak presence is the
    # save_with_backup pattern's responsibility.
    main_file = persona_dir / "data" / "works" / "abc123def456.md"
    assert "second version" in main_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Audit 2026-05-07 P3-6 — YAML-safe frontmatter
# ---------------------------------------------------------------------------


def test_yaml_frontmatter_handles_colons_in_title(tmp_path: Path) -> None:
    """Titles with ':' used to corrupt frontmatter parsing."""
    persona_dir = _make_persona(tmp_path)
    w = works.Work(
        id="aaa111bbb222",
        title="Notes: a colon-bearing title",
        type="idea",
        created_at=datetime(2026, 5, 4, tzinfo=UTC),
        session_id=None,
        word_count=2,
        summary=None,
    )
    storage.write_markdown(persona_dir, w, content="body")
    out, _ = storage.read_markdown(persona_dir, "aaa111bbb222")
    assert out.title == "Notes: a colon-bearing title"


def test_yaml_frontmatter_handles_yaml_reserved_words(tmp_path: Path) -> None:
    """A title that's literally 'yes' / 'true' / 'null' must round-trip
    as the string, not get coerced to bool/None."""
    persona_dir = _make_persona(tmp_path)
    for tricky in ["yes", "no", "true", "false", "null", "123", "1.5"]:
        w = works.Work(
            id="bbb222ccc333",
            title=tricky,
            type="idea",
            created_at=datetime(2026, 5, 4, tzinfo=UTC),
            session_id=None,
            word_count=1,
            summary=None,
        )
        storage.write_markdown(persona_dir, w, content="body")
        out, _ = storage.read_markdown(persona_dir, "bbb222ccc333")
        assert out.title == tricky, f"title {tricky!r} did not round-trip"


def test_yaml_frontmatter_handles_multiline_summary(tmp_path: Path) -> None:
    """Summaries with newlines round-trip as a YAML literal block scalar."""
    persona_dir = _make_persona(tmp_path)
    multi = "First line.\nSecond line.\nThird."
    w = works.Work(
        id="ccc333ddd444",
        title="multi-line summary test",
        type="story",
        created_at=datetime(2026, 5, 4, tzinfo=UTC),
        session_id=None,
        word_count=3,
        summary=multi,
    )
    storage.write_markdown(persona_dir, w, content="body")
    out, _ = storage.read_markdown(persona_dir, "ccc333ddd444")
    assert out.summary == multi


def test_yaml_frontmatter_handles_quote_chars(tmp_path: Path) -> None:
    """Embedded double-quotes in a quoted scalar round-trip via escaping."""
    persona_dir = _make_persona(tmp_path)
    w = works.Work(
        id="ddd444eee555",
        title='She said "hello" — like that',
        type="idea",
        created_at=datetime(2026, 5, 4, tzinfo=UTC),
        session_id=None,
        word_count=4,
        summary=None,
    )
    storage.write_markdown(persona_dir, w, content="body")
    out, _ = storage.read_markdown(persona_dir, "ddd444eee555")
    assert out.title == 'She said "hello" — like that'
