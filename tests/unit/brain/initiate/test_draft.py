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
        tmp_path, timestamp="2026-05-11T14:32:00+00:00",
        source="dream", body="x",
    )
    assert has_new_drafts_since(tmp_path, last_seen_iso) is True


def test_has_new_drafts_since_returns_false_when_no_changes(tmp_path: Path) -> None:
    """If draft_space.md hasn't been modified since last_seen, return False."""
    assert has_new_drafts_since(tmp_path, "2024-01-01T00:00:00+00:00") is False
