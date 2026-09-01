"""Conformance test for archive segmentation (C11).

Maps to changes/cascade-compaction/1.5-criteria.md:
  C11  archive segmentation + multi-segment reader + atomicity
"""

from __future__ import annotations

import json
from pathlib import Path

import brain.ingest.buffer as buffer_mod
from brain.health.jsonl_reader import read_jsonl_skipping_corrupt
from brain.ingest.buffer import (
    _archive_segments,
    _archived_conversations_dir,
    append_archive,
    read_archive,
)


def _rec(i: int, text: str) -> dict:
    return {
        "session_id": "sess_c11",
        "speaker": "user",
        "text": text,
        "ts": f"2026-01-01T00:{i:02d}:00+00:00",
    }


def test_c11_archive_segments_reader_crash(tmp_path: Path, monkeypatch) -> None:
    sid = "sess_c11"
    # A small cap forces many segment rolls from a modest record count.
    monkeypatch.setattr(buffer_mod, "_ARCHIVE_SEGMENT_MAX_BYTES", 200)

    records = [_rec(i, text=f"payload number {i} " * 3) for i in range(30)]
    total_written = 0
    for r in records:
        total_written += append_archive(tmp_path, sid, [r])
    assert total_written > 0

    # (a) more than one segment file exists.
    segments = _archive_segments(tmp_path, sid)
    assert len(segments) > 1

    # (b) reader returns the exact original sequence, numeric-ordered across segments.
    got = read_archive(tmp_path, sid)
    assert [g["text"] for g in got] == [r["text"] for r in records]

    # (c) provenance fields intact.
    for g, r in zip(got, records, strict=True):
        assert g["session_id"] == r["session_id"]
        assert g["speaker"] == r["speaker"]
        assert g["ts"] == r["ts"]

    # (d) append_archive still returns the bytes written (byte-count contract),
    # which is what the caller (compaction.py) uses to gate the buffer rewrite.
    marker_rec = {
        "session_id": sid, "speaker": "user",
        "text": "byte count check", "ts": "2026-01-01T01:00:00+00:00",
    }
    payload = json.dumps(marker_rec, ensure_ascii=False) + "\n"
    written = append_archive(tmp_path, sid, [marker_rec])
    assert written == len(payload.encode("utf-8"))

    # --- (e) simulate a crash mid-roll ---------------------------------
    d = _archived_conversations_dir(tmp_path)
    segs_before = _archive_segments(tmp_path, sid)
    last_num, last_path = segs_before[-1]

    # A zero-length trailing segment: created but never actually written
    # before the crash (the roll-to-new-segment step landed, the write didn't).
    empty_seg = d / f"{sid}.{last_num + 1:03d}.jsonl"
    empty_seg.write_text("")

    # A torn (non-JSON, no trailing newline) final line appended to the
    # active segment — the write was interrupted mid-record.
    with open(last_path, "a", encoding="utf-8") as fh:
        fh.write('{"session_id": "sess_c11", "speaker": "user", "tex')

    got_after_crash = read_archive(tmp_path, sid)
    expected_texts = [r["text"] for r in records] + ["byte count check"]
    assert [g["text"] for g in got_after_crash] == expected_texts

    # H6: a newest-segment-only reader would miss the earlier records —
    # confirm the full multi-segment reader recovers strictly more.
    newest_seg_num, newest_seg_path = _archive_segments(tmp_path, sid)[-1]
    newest_only = read_jsonl_skipping_corrupt(newest_seg_path)
    assert len(newest_only) < len(got_after_crash)

    # --- back-compat: a legacy single <sid>.jsonl still reads ----------
    legacy_sid = "sess_c11_legacy"
    legacy_path = d / f"{legacy_sid}.jsonl"
    legacy_records = [_rec(i, text=f"legacy {i}") for i in range(5)]
    with open(legacy_path, "w", encoding="utf-8") as fh:
        for r in legacy_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    got_legacy = read_archive(tmp_path, legacy_sid)
    assert [g["text"] for g in got_legacy] == [r["text"] for r in legacy_records]
