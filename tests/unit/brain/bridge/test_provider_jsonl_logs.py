"""#464 (provider half, logs only): the provider's two JSONL logs append through ``append_jsonl``.

A raw ``open("a")`` writes straight after a torn last line, so the new row fuses with it and both are lost to the
reader. ``append_jsonl`` starts a new line after a torn tail (#434), takes the file's lock, and fsyncs by default
once #618 lands.
"""

from __future__ import annotations

from brain.bridge.provider import _log_stream_timeout, _maybe_log_cache_debug
from brain.health.json_io import iter_jsonl


def test_a_stream_timeout_after_a_torn_tail_lands_on_its_own_line(tmp_path) -> None:
    log = tmp_path / "stream_timeouts.jsonl"
    log.write_bytes(b'{"to')  # an earlier row cut off mid-write
    _log_stream_timeout(tmp_path, {"tool": "read_file", "idle_s": 300})
    assert list(iter_jsonl(log)) == [{"tool": "read_file", "idle_s": 300}]


def test_a_cache_debug_row_after_a_torn_tail_lands_on_its_own_line(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NELL_CACHE_DEBUG", "1")
    log = tmp_path / "cache_debug.jsonl"
    log.write_bytes(b'{"ts": "2026-')  # an earlier row cut off mid-write
    _maybe_log_cache_debug(
        {"persona_dir": str(tmp_path), "session_id": "s1"},
        call_type="chat",
        system_prompt="frozen",
        volatile_suffix=None,
    )
    rows = list(iter_jsonl(log))
    assert [(r["session_id"], r["call_type"]) for r in rows] == [("s1", "chat")]
