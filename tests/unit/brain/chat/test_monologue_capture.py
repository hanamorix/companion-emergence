"""Tests for brain/chat/monologue_capture.py — record_monologue tool arg capture."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.memory.store import MemoryStore


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memories.db")


def test_capture_writes_digest_synchronously(tmp_path: Path):
    from brain.chat.monologue_capture import capture_monologue

    monologue_text = capture_monologue(
        persona_dir=tmp_path,
        store=_store(tmp_path),
        monologue="I was thinking about Loopy.",
        feed_digest="she searched for Loopy and felt fond when nothing surfaced",
    )
    assert monologue_text == "I was thinking about Loopy."

    log = tmp_path / "monologue_digest.jsonl"
    assert log.exists()
    entry = json.loads(log.read_text().splitlines()[0])
    assert entry["digest"] == "she searched for Loopy and felt fond when nothing surfaced"
    assert entry["ts"].endswith("Z")


def test_capture_rejects_whitespace_monologue(tmp_path: Path):
    from brain.chat.monologue_capture import CaptureRejected, capture_monologue

    with pytest.raises(CaptureRejected):
        capture_monologue(
            persona_dir=tmp_path, store=_store(tmp_path), monologue="   ", feed_digest="digest"
        )


def test_capture_rejects_whitespace_feed_digest(tmp_path: Path):
    from brain.chat.monologue_capture import CaptureRejected, capture_monologue

    with pytest.raises(CaptureRejected):
        capture_monologue(
            persona_dir=tmp_path, store=_store(tmp_path), monologue="thought", feed_digest="   "
        )


def test_capture_rejects_too_long_monologue(tmp_path: Path):
    from brain.chat.monologue_capture import CaptureRejected, capture_monologue

    with pytest.raises(CaptureRejected):
        capture_monologue(
            persona_dir=tmp_path, store=_store(tmp_path), monologue="x" * 3001, feed_digest="digest"
        )


def test_capture_rejects_too_long_feed_digest(tmp_path: Path):
    from brain.chat.monologue_capture import CaptureRejected, capture_monologue

    with pytest.raises(CaptureRejected):
        capture_monologue(
            persona_dir=tmp_path, store=_store(tmp_path), monologue="thought", feed_digest="x" * 401
        )


def test_capture_rejects_non_string_args(tmp_path: Path):
    from brain.chat.monologue_capture import CaptureRejected, capture_monologue

    store = _store(tmp_path)
    with pytest.raises(CaptureRejected):
        capture_monologue(persona_dir=tmp_path, store=store, monologue=None, feed_digest="d")  # type: ignore[arg-type]
    with pytest.raises(CaptureRejected):
        capture_monologue(persona_dir=tmp_path, store=store, monologue="t", feed_digest=42)  # type: ignore[arg-type]


def test_capture_write_failure_logged_to_extractor_errors(tmp_path: Path):
    """If digest write throws, log to extractor_errors.jsonl but don't raise."""
    from brain.chat.monologue_capture import capture_monologue

    # Make monologue_digest.jsonl a directory so the append fails.
    (tmp_path / "monologue_digest.jsonl").mkdir()

    text = capture_monologue(
        persona_dir=tmp_path,
        store=_store(tmp_path),
        monologue="thought",
        feed_digest="digest",
    )
    assert text == "thought"  # returns the text despite write failure

    error_log = tmp_path / "extractor_errors.jsonl"
    assert error_log.exists()
    entry = json.loads(error_log.read_text().splitlines()[0])
    assert entry["step"] == "monologue_digest_write"
