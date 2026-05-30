"""Tests for the monologue source in the visible inner life feed."""
from __future__ import annotations

from pathlib import Path

from brain.bridge.feed import TYPE_OPENER, FeedEntryType, build_monologue_entries


def test_monologue_is_a_valid_feed_entry_type():
    from typing import get_args
    assert "monologue" in get_args(FeedEntryType)


def test_monologue_opener_present():
    assert "monologue" in TYPE_OPENER
    assert TYPE_OPENER["monologue"]


def test_no_file_yields_empty_list(tmp_path: Path):
    assert build_monologue_entries(tmp_path, limit=10) == []


def test_reads_recent_entries_newest_first(tmp_path: Path):
    import json

    log = tmp_path / "monologue_digest.jsonl"
    log.write_text(
        "\n".join(
            json.dumps({"ts": ts, "digest": digest})
            for ts, digest in [
                ("2026-05-30T10:00:00+00:00", "she dwelt on the morning"),
                ("2026-05-30T11:00:00+00:00", "she searched and didn't find"),
                ("2026-05-30T12:00:00+00:00", "she felt the gap"),
            ]
        )
    )
    entries = build_monologue_entries(tmp_path, limit=10)
    assert len(entries) == 3
    assert entries[0].body == "she felt the gap"
    assert entries[-1].body == "she dwelt on the morning"


def test_respects_limit(tmp_path: Path):
    import json

    log = tmp_path / "monologue_digest.jsonl"
    log.write_text(
        "\n".join(
            json.dumps({"ts": f"2026-05-30T{h:02}:00:00+00:00", "digest": f"d{h}"})
            for h in range(20)
        )
    )
    entries = build_monologue_entries(tmp_path, limit=5)
    assert len(entries) == 5


def test_skips_malformed_lines(tmp_path: Path):
    log = tmp_path / "monologue_digest.jsonl"
    log.write_text(
        '{"ts": "2026-05-30T10:00:00+00:00", "digest": "good"}\n'
        "not json\n"
        '{"missing_digest": true}\n'
    )
    entries = build_monologue_entries(tmp_path, limit=10)
    assert len(entries) == 1
    assert entries[0].body == "good"
