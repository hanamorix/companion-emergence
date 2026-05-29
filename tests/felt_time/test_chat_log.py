import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


def test_append_chat_tick_creates_file(tmp_path: Path) -> None:
    from brain.felt_time.chat_log import CHAT_TURNS_LOG_FILENAME, append_chat_tick

    ts = datetime(2026, 5, 29, 10, 0, 0, tzinfo=UTC)
    append_chat_tick(tmp_path, ts=ts, turns=3)

    log = tmp_path / CHAT_TURNS_LOG_FILENAME
    assert log.exists()
    row = json.loads(log.read_text().strip())
    assert row["turns"] == 3
    assert "2026-05-29" in row["ts"]


def test_append_chat_tick_zero_turns_is_written(tmp_path: Path) -> None:
    from brain.felt_time.chat_log import CHAT_TURNS_LOG_FILENAME, append_chat_tick

    append_chat_tick(tmp_path, ts=datetime.now(UTC), turns=0)
    log = tmp_path / CHAT_TURNS_LOG_FILENAME
    row = json.loads(log.read_text().strip())
    assert row["turns"] == 0


def test_append_chat_tick_appends_multiple_rows(tmp_path: Path) -> None:
    from brain.felt_time.chat_log import CHAT_TURNS_LOG_FILENAME, append_chat_tick

    now = datetime.now(UTC)
    append_chat_tick(tmp_path, ts=now, turns=1)
    append_chat_tick(tmp_path, ts=now, turns=2)
    lines = (tmp_path / CHAT_TURNS_LOG_FILENAME).read_text().strip().splitlines()
    assert len(lines) == 2


def test_trim_old_entries_removes_old(tmp_path: Path) -> None:
    from brain.felt_time.chat_log import _trim_old_entries

    now = datetime.now(UTC)
    cutoff = now - timedelta(days=30)
    entries = [
        {"ts": now - timedelta(days=40), "turns": 1.0},  # too old
        {"ts": now - timedelta(days=20), "turns": 2.0},  # keep
        {"ts": now - timedelta(days=1), "turns": 3.0},   # keep
    ]
    kept = _trim_old_entries(entries, cutoff=cutoff)
    assert len(kept) == 2
    assert all(e["ts"] >= cutoff for e in kept)


def test_load_recent_samples_returns_none_when_no_file(tmp_path: Path) -> None:
    from brain.felt_time.chat_log import load_recent_samples

    assert load_recent_samples(tmp_path) is None


def test_load_recent_samples_returns_none_below_cold_start(tmp_path: Path) -> None:
    from brain.felt_time.chat_log import append_chat_tick, load_recent_samples

    now = datetime.now(UTC)
    append_chat_tick(tmp_path, ts=now, turns=1)
    append_chat_tick(tmp_path, ts=now, turns=2)
    # Only 2 entries — below _COLD_START_MIN=3
    assert load_recent_samples(tmp_path) is None


def test_load_recent_samples_returns_window_entries(tmp_path: Path) -> None:
    from brain.felt_time.chat_log import append_chat_tick, load_recent_samples

    now = datetime.now(UTC)
    for i in range(5):
        append_chat_tick(tmp_path, ts=now - timedelta(days=i), turns=i + 1)

    samples = load_recent_samples(tmp_path, window_days=7)
    assert samples is not None
    assert len(samples) == 5
    # Sorted ascending
    assert samples[0][0] < samples[-1][0]


def test_load_recent_samples_excludes_entries_outside_window(tmp_path: Path) -> None:
    from brain.felt_time.chat_log import append_chat_tick, load_recent_samples

    now = datetime.now(UTC)
    # 3 entries in window, 3 outside
    for i in range(3):
        append_chat_tick(tmp_path, ts=now - timedelta(days=i), turns=1)
    for i in range(3):
        append_chat_tick(tmp_path, ts=now - timedelta(days=10 + i), turns=99)

    samples = load_recent_samples(tmp_path, window_days=7)
    assert samples is not None
    assert len(samples) == 3
    assert all(v == 1.0 for _, v in samples)


def test_load_recent_samples_lazy_trim_removes_old_from_file(tmp_path: Path) -> None:
    from brain.felt_time.chat_log import CHAT_TURNS_LOG_FILENAME, append_chat_tick, load_recent_samples

    now = datetime.now(UTC)
    # Write 15 entries older than _RETAIN_DAYS=30 (triggers rewrite threshold >10)
    for i in range(15):
        append_chat_tick(tmp_path, ts=now - timedelta(days=35 + i), turns=0)
    # Write 5 recent entries
    for i in range(5):
        append_chat_tick(tmp_path, ts=now - timedelta(days=i), turns=1)

    load_recent_samples(tmp_path)

    # After lazy trim, file should only contain recent entries
    lines = (tmp_path / CHAT_TURNS_LOG_FILENAME).read_text().strip().splitlines()
    assert len(lines) == 5
