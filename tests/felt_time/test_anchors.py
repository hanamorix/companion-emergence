"""Tests for brain.felt_time.anchors — unified anchor stream."""

import json
from pathlib import Path

from brain.felt_time.anchors import extract_all, scan_since


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def test_extract_all_returns_empty_when_no_logs(tmp_path):
    assert extract_all(tmp_path) == []


def test_extract_all_pulls_dreams(tmp_path):
    _write_jsonl(
        tmp_path / "dreams.log.jsonl",
        [
            {"ts": "2026-05-17T20:00:00+00:00", "summary": "the boat one"},
            {"ts": "2026-05-17T22:00:00+00:00", "summary": "the kitchen one"},
        ],
    )
    anchors = extract_all(tmp_path)
    assert [a.type for a in anchors] == ["dream", "dream"]
    assert [a.label for a in anchors] == ["the boat one", "the kitchen one"]
    assert anchors[0].source_ref == "dreams.log.jsonl:1"


def test_extract_all_pulls_all_three_existing_sources(tmp_path):
    _write_jsonl(
        tmp_path / "dreams.log.jsonl", [{"ts": "2026-05-17T20:00:00+00:00", "summary": "d1"}]
    )
    _write_jsonl(
        tmp_path / "growth.log.jsonl", [{"ts": "2026-05-17T21:00:00+00:00", "title": "g1"}]
    )
    _write_jsonl(
        tmp_path / "soul.log.jsonl", [{"ts": "2026-05-17T22:00:00+00:00", "moment_label": "s1"}]
    )

    anchors = extract_all(tmp_path)
    assert [a.type for a in anchors] == ["dream", "growth", "soul"]
    assert [a.label for a in anchors] == ["d1", "g1", "s1"]


def test_extract_all_skips_entries_missing_ts_or_label(tmp_path):
    _write_jsonl(
        tmp_path / "dreams.log.jsonl",
        [
            {"ts": "2026-05-17T20:00:00+00:00"},  # missing summary
            {"summary": "labeled but no ts"},
            {"ts": "2026-05-17T21:00:00+00:00", "summary": "good"},
        ],
    )
    anchors = extract_all(tmp_path)
    assert len(anchors) == 1
    assert anchors[0].label == "good"


def test_extract_all_tolerates_corrupt_jsonl_line(tmp_path):
    (tmp_path / "dreams.log.jsonl").write_text(
        '{"ts": "2026-05-17T20:00:00+00:00", "summary": "before"}\n'
        "not-json\n"
        '{"ts": "2026-05-17T22:00:00+00:00", "summary": "after"}\n'
    )
    anchors = extract_all(tmp_path)
    assert [a.label for a in anchors] == ["before", "after"]


def test_scan_since_returns_only_newer_anchors(tmp_path):
    _write_jsonl(
        tmp_path / "dreams.log.jsonl",
        [
            {"ts": "2026-05-17T20:00:00+00:00", "summary": "old"},
            {"ts": "2026-05-17T22:00:00+00:00", "summary": "new"},
        ],
    )
    later = scan_since(tmp_path, "2026-05-17T21:00:00+00:00")
    assert [a.label for a in later] == ["new"]

    # since_ts=None returns all
    assert len(scan_since(tmp_path, None)) == 2
