"""Tests for `_maybe_log_cache_debug` — the prompt-caching byte-stability probe.

Part of the `prompt-caching-adopt` change (P1 instrumentation slice). The function is
off by default and fully fail-soft. See docs/guarded-change/prompt-caching-adopt/.
"""

from __future__ import annotations

import json

import pytest

from brain.bridge.provider import _maybe_log_cache_debug


def test_no_op_when_env_unset(tmp_path, monkeypatch):
    """Default (env unset): writes nothing, raises nothing."""
    monkeypatch.delenv("NELL_CACHE_DEBUG", raising=False)
    _maybe_log_cache_debug(
        {"persona_dir": str(tmp_path)},
        call_type="chat",
        system_prompt="hello",
        volatile_suffix=None,
    )
    assert not (tmp_path / "cache_debug.jsonl").exists()


def test_writes_row_when_enabled(tmp_path, monkeypatch):
    """Enabled + persona_dir: appends one JSONL row with the byte-stability fields."""
    monkeypatch.setenv("NELL_CACHE_DEBUG", "1")
    opts = {"persona_dir": str(tmp_path), "session_id": "sess-1"}
    _maybe_log_cache_debug(opts, call_type="chat", system_prompt="frozen", volatile_suffix=None)

    rows = [
        json.loads(line)
        for line in (tmp_path / "cache_debug.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["call_type"] == "chat"
    assert row["session_id"] == "sess-1"
    assert row["system_char_len"] == len("frozen")
    assert row["volatile_present"] is False
    assert len(row["system_sha256"]) == 64


def test_same_system_prompt_yields_same_hash(tmp_path, monkeypatch):
    """The whole point: identical system text → identical system_sha256 (the C1 probe)."""
    monkeypatch.setenv("NELL_CACHE_DEBUG", "1")
    opts = {"persona_dir": str(tmp_path)}
    for _ in range(3):
        _maybe_log_cache_debug(opts, call_type="chat", system_prompt="STATIC", volatile_suffix=None)
    rows = [
        json.loads(line)
        for line in (tmp_path / "cache_debug.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len({r["system_sha256"] for r in rows}) == 1


def test_no_op_when_no_persona_dir(monkeypatch):
    """Even enabled, a missing persona_dir is a silent no-op (no path to write)."""
    monkeypatch.setenv("NELL_CACHE_DEBUG", "1")
    _maybe_log_cache_debug(None, call_type="chat", system_prompt="x", volatile_suffix=None)
    _maybe_log_cache_debug({}, call_type="chat", system_prompt="x", volatile_suffix=None)


def test_fail_soft_on_bad_persona_dir(monkeypatch):
    """A write failure (persona_dir under a file path) must be swallowed, not raised."""
    monkeypatch.setenv("NELL_CACHE_DEBUG", "1")
    _maybe_log_cache_debug(
        {"persona_dir": "/dev/null/nope"},
        call_type="chat",
        system_prompt="x",
        volatile_suffix=None,
    )  # no exception = pass


def test_file_trigger_enables_cache_debug_without_env_var(tmp_path, monkeypatch):
    """touch <persona_dir>/cache_debug.on enables logging even without NELL_CACHE_DEBUG=1.

    This is the GUI-bridge use case: env vars don't survive a NellFace launch,
    so the marker file is the reliable way to enable the log on a live bridge.
    """
    monkeypatch.delenv("NELL_CACHE_DEBUG", raising=False)
    (tmp_path / "cache_debug.on").touch()
    opts = {"persona_dir": str(tmp_path), "session_id": "sess-file-trigger"}
    _maybe_log_cache_debug(opts, call_type="chat", system_prompt="static", volatile_suffix=None)

    log_path = tmp_path / "cache_debug.jsonl"
    assert log_path.exists(), "cache_debug.jsonl must be written when marker file is present"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["session_id"] == "sess-file-trigger"


def test_no_marker_file_and_no_env_var_is_no_op(tmp_path, monkeypatch):
    """Without env var AND without the marker file: writes nothing (double-guard check)."""
    monkeypatch.delenv("NELL_CACHE_DEBUG", raising=False)
    _maybe_log_cache_debug(
        {"persona_dir": str(tmp_path)},
        call_type="chat",
        system_prompt="volatile",
        volatile_suffix=None,
    )
    assert not (tmp_path / "cache_debug.jsonl").exists()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
