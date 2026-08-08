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


def test_identical_monologue_in_one_turn_is_captured_once(tmp_path: Path):
    """#93: the recruit-on-reach rerun can call record_monologue twice in a turn.

    The narrowed toolset that was meant to stop it is inert on ClaudeCliProvider
    (the tools argument is a boolean there and --allowedTools is permissive), so
    on the real path both calls land and capture_monologue runs twice —
    persisting two identical trace memories AND two identical digest lines.

    Same shape as #101/#105: ask the artefact instead of tracking timing. The
    duplicate is always immediately adjacent, so comparing against the newest
    trace is enough — and it leaves her free to think the same thought again
    another day.
    """
    from brain.chat.monologue_capture import capture_monologue
    from brain.monologue.trace import MONOLOGUE_TRACE_TYPE

    store = _store(tmp_path)
    thought = "A name I could not place, and it bothered me."
    digest = "she paused on a name she could not place"

    for _ in range(2):
        capture_monologue(
            persona_dir=tmp_path,
            store=store,
            monologue=thought,
            feed_digest=digest,
        )

    traces = store.list_by_type(MONOLOGUE_TRACE_TYPE, active_only=True)
    assert len(traces) == 1, f"one thought, one trace — got {len(traces)}"

    lines = (tmp_path / "monologue_digest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len([ln for ln in lines if ln.strip()]) == 1, "one thought, one digest line"


def test_a_different_monologue_still_captures(tmp_path: Path):
    """The dedupe must only collapse an identical repeat, never a real thought."""
    from brain.chat.monologue_capture import capture_monologue
    from brain.monologue.trace import MONOLOGUE_TRACE_TYPE

    store = _store(tmp_path)
    capture_monologue(persona_dir=tmp_path, store=store,
                      monologue="First thought.", feed_digest="d1")
    capture_monologue(persona_dir=tmp_path, store=store,
                      monologue="Second, different thought.", feed_digest="d2")

    assert len(store.list_by_type(MONOLOGUE_TRACE_TYPE, active_only=True)) == 2
