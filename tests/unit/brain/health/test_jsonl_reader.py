"""Tests for brain.health.jsonl_reader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from brain.health.jsonl_reader import read_jsonl_skipping_corrupt


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_jsonl_skipping_corrupt(tmp_path / "missing.jsonl") == []


def test_well_formed_lines_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n", encoding="utf-8")
    out = read_jsonl_skipping_corrupt(p)
    assert out == [{"a": 1}, {"a": 2}]


def test_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n\n\n" + json.dumps({"a": 2}) + "\n", encoding="utf-8")
    assert read_jsonl_skipping_corrupt(p) == [{"a": 1}, {"a": 2}]


def test_skips_corrupt_lines_and_warns(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.WARNING)
    p = tmp_path / "log.jsonl"
    p.write_text(
        json.dumps({"good": 1}) + "\n{not valid\n" + json.dumps({"good": 2}) + "\n",
        encoding="utf-8",
    )
    out = read_jsonl_skipping_corrupt(p)
    assert out == [{"good": 1}, {"good": 2}]

    bad = [r for r in caplog.records if "malformed jsonl line" in r.getMessage()]
    assert len(bad) == 1
    msg = bad[0].getMessage()
    assert "line 2" in msg
    assert "{not valid" in msg


def test_warning_includes_path_and_truncates_long_content(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.WARNING)
    p = tmp_path / "log.jsonl"
    long_corrupt = "{" + ("x" * 500)
    p.write_text(long_corrupt + "\n", encoding="utf-8")
    read_jsonl_skipping_corrupt(p)
    msg = next(r.getMessage() for r in caplog.records if "malformed jsonl line" in r.getMessage())
    assert str(p) in msg
    assert "x" * 200 in msg  # 200-char preview
    assert "x" * 500 not in msg  # truncated
