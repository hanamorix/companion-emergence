"""Tests for brain.initiate.draft — failed-to-promote routing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from brain.initiate.draft import (
    append_draft_fragment,
    compose_draft_fragment,
    has_new_drafts_since,
)


def test_append_draft_fragment_creates_file_and_appends(tmp_path: Path) -> None:
    append_draft_fragment(
        tmp_path,
        timestamp="2026-05-11T14:32:00+00:00",
        source="dream",
        body="The dream wasn't loud enough to bring up.",
    )
    content = (tmp_path / "draft_space.md").read_text()
    assert "## 2026-05-11 14:32 (dream)" in content
    assert "The dream wasn't loud enough" in content


def test_append_draft_fragment_idempotent_on_timestamp_source(tmp_path: Path) -> None:
    """Re-appending the same timestamp+source produces only one entry."""
    for _ in range(3):
        append_draft_fragment(
            tmp_path,
            timestamp="2026-05-11T14:32:00+00:00",
            source="dream",
            body="x",
        )
    content = (tmp_path / "draft_space.md").read_text()
    assert content.count("## 2026-05-11 14:32 (dream)") == 1


def test_compose_draft_fragment_calls_provider_once(tmp_path: Path) -> None:
    """The expensive composition happens with exactly one cheap LLM call."""
    provider = MagicMock(complete=MagicMock(return_value="composed fragment text"))
    result = compose_draft_fragment(
        provider,
        source="dream",
        source_id="dream_001",
        linked_memory_excerpts=["bench", "tools"],
    )
    assert provider.complete.call_count == 1
    assert result == "composed fragment text"


def test_compose_draft_fragment_falls_back_to_template_on_error(tmp_path: Path) -> None:
    """LLM failure produces a deterministic templated fragment."""
    provider = MagicMock(complete=MagicMock(side_effect=RuntimeError("boom")))
    result = compose_draft_fragment(
        provider,
        source="dream",
        source_id="dream_001",
        linked_memory_excerpts=["bench"],
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_has_new_drafts_since_returns_true_when_file_newer(tmp_path: Path) -> None:
    last_seen_iso = "2024-01-01T00:00:00+00:00"
    append_draft_fragment(
        tmp_path,
        timestamp="2026-05-11T14:32:00+00:00",
        source="dream",
        body="x",
    )
    assert has_new_drafts_since(tmp_path, last_seen_iso) is True


def test_has_new_drafts_since_returns_false_when_no_changes(tmp_path: Path) -> None:
    """If draft_space.md hasn't been modified since last_seen, return False."""
    assert has_new_drafts_since(tmp_path, "2024-01-01T00:00:00+00:00") is False


def test_compose_draft_fragment_uses_user_name_not_hana() -> None:
    """compose_draft_fragment prompt must not contain 'Hana' when user_name is set."""
    provider = MagicMock(complete=MagicMock(return_value="a quiet fragment"))
    compose_draft_fragment(
        provider,
        source="dream",
        source_id="dr_001",
        linked_memory_excerpts=["the workshop bench"],
        user_name="Henryk",
    )
    args, _ = provider.complete.call_args
    prompt_text = args[0]
    assert "Henryk" in prompt_text
    assert "Hana" not in prompt_text


def test_compose_draft_fragment_uses_companion_name_not_nell() -> None:
    """compose_draft_fragment prompt must not hardcode 'Nell'."""
    provider = MagicMock(complete=MagicMock(return_value="fragment"))
    compose_draft_fragment(
        provider,
        source="dream",
        source_id="dr_001",
        linked_memory_excerpts=["bench"],
        companion_name="Mira",
    )
    args, _ = provider.complete.call_args
    assert "Mira" in args[0]
    assert "Nell" not in args[0]


# ── read_drafts_since ─────────────────────────────────────────────────────────


def test_read_drafts_since_returns_fragments_after_cursor(tmp_path: Path) -> None:
    """Fragments strictly after the cutoff timestamp are returned; older ones excluded."""
    from brain.initiate.draft import read_drafts_since  # noqa: PLC0415

    append_draft_fragment(
        tmp_path,
        timestamp="2026-06-01T09:00:00",
        source="d_reflection",
        body="An older thought that didn't rise.",
    )
    append_draft_fragment(
        tmp_path,
        timestamp="2026-06-04T10:00:00",
        source="emotion_spike",
        body="A newer surge of feeling.",
    )

    result = read_drafts_since(tmp_path, "2026-06-02T00:00:00")

    assert len(result) == 1
    frag = result[0]
    assert frag.source == "emotion_spike"
    assert "newer surge" in frag.body


def test_read_drafts_since_empty_when_no_file(tmp_path: Path) -> None:
    """Returns empty list when draft_space.md does not exist."""
    from brain.initiate.draft import read_drafts_since  # noqa: PLC0415

    result = read_drafts_since(tmp_path, "2026-01-01T00:00:00")
    assert result == []


def test_read_drafts_since_multiline_body(tmp_path: Path) -> None:
    """Multi-line bodies are captured in full."""
    from brain.initiate.draft import read_drafts_since  # noqa: PLC0415

    append_draft_fragment(
        tmp_path,
        timestamp="2026-06-04T12:00:00",
        source="dream",
        body="Line one.\nLine two.\nLine three.",
    )
    result = read_drafts_since(tmp_path, "2026-06-01T00:00:00")
    assert len(result) == 1
    assert "Line two" in result[0].body


def test_read_drafts_since_tz_aware_cutoff(tmp_path: Path) -> None:
    """Tz-aware cutoff is coerced to naive for comparison without raising."""
    from brain.initiate.draft import read_drafts_since  # noqa: PLC0415

    append_draft_fragment(
        tmp_path,
        timestamp="2026-06-04T10:00:00",
        source="emotion_spike",
        body="Should surface.",
    )
    # tz-aware cutoff (before the fragment) — must not raise TypeError
    result = read_drafts_since(tmp_path, "2026-06-01T00:00:00+00:00")
    assert len(result) == 1


def test_load_draft_review_cursor_returns_empty_when_no_file(tmp_path: Path) -> None:
    """No cursor file → returns empty string."""
    from brain.initiate.draft import load_draft_review_cursor  # noqa: PLC0415

    assert load_draft_review_cursor(tmp_path) == ""


def test_save_and_load_draft_review_cursor_roundtrip(tmp_path: Path) -> None:
    """save_draft_review_cursor persists; load_draft_review_cursor retrieves it."""
    from brain.initiate.draft import (  # noqa: PLC0415
        load_draft_review_cursor,
        save_draft_review_cursor,
    )

    iso = "2026-06-04T10:30:00+00:00"
    save_draft_review_cursor(tmp_path, iso)
    assert load_draft_review_cursor(tmp_path) == iso
