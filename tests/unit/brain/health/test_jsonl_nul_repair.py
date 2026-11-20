"""The one-time NUL repair (#618): an unclean stop left runs of NUL bytes in JSONL files."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from brain.health import jsonl_nul_repair
from brain.health.json_io import append_jsonl, iter_jsonl
from brain.health.jsonl_nul_repair import (
    STATE_FILENAME,
    run_jsonl_nul_repair,
    should_run_jsonl_nul_repair,
)


def _write(persona: Path, name: str, data: bytes) -> Path:
    p = persona / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_a_nul_run_at_a_line_start_is_removed_and_the_swallowed_record_comes_back(tmp_path):
    original = b'{"a":1}\n' + b"\0" * 20 + b'{"b":2}\n'
    p = _write(tmp_path, "x.jsonl", original)

    report = run_jsonl_nul_repair(tmp_path)

    assert list(iter_jsonl(p)) == [{"a": 1}, {"b": 2}]
    assert report.repaired["x.jsonl"].runs == 1
    assert report.repaired["x.jsonl"].bytes_removed == 20
    assert (tmp_path / "x.jsonl.pre-618").read_bytes() == original


_TORN_THEN_RUN = b'{"a":1}\n{"to' + b"\0" * 8 + b'{"b":2}\n'
_RUN_THEN_NEWLINE = b'{"to' + b"\0" * 4 + b'\n{"b":2}\n'


@pytest.mark.parametrize("chunk_size", range(1, len(_TORN_THEN_RUN) + 1))
def test_a_run_after_a_torn_record_becomes_a_newline(tmp_path, chunk_size):
    p = _write(tmp_path, "x.jsonl", _TORN_THEN_RUN)
    run_jsonl_nul_repair(tmp_path, chunk_size=chunk_size)
    assert p.read_bytes() == b'{"a":1}\n{"to\n{"b":2}\n'


@pytest.mark.parametrize("chunk_size", range(1, len(_RUN_THEN_NEWLINE) + 1))
def test_a_run_before_a_newline_after_a_torn_record_adds_no_blank_line(tmp_path, chunk_size):
    p = _write(tmp_path, "x.jsonl", _RUN_THEN_NEWLINE)
    run_jsonl_nul_repair(tmp_path, chunk_size=chunk_size)
    assert p.read_bytes() == b'{"to\n{"b":2}\n'


def test_a_run_at_the_end_of_the_file_is_dropped(tmp_path):
    p = _write(tmp_path, "x.jsonl", b'{"a":1}\n' + b"\0" * 30)
    run_jsonl_nul_repair(tmp_path)
    assert p.read_bytes() == b'{"a":1}\n'


def test_a_file_of_only_nuls_becomes_empty_with_its_backup_kept(tmp_path):
    p = _write(tmp_path, "x.jsonl", b"\0" * 64)
    report = run_jsonl_nul_repair(tmp_path)
    assert p.read_bytes() == b""
    assert (tmp_path / "x.jsonl.pre-618").read_bytes() == b"\0" * 64
    assert report.repaired["x.jsonl"].bytes_removed == 64


def test_a_run_split_across_chunks_is_repaired_the_same(tmp_path):
    original = b'{"a":1}\n' + b"\0" * 20 + b'{"b":2}\n'
    p = _write(tmp_path, "x.jsonl", original)
    run_jsonl_nul_repair(tmp_path)
    expected = p.read_bytes()

    q = _write(tmp_path / "small", "x.jsonl", original)
    report = run_jsonl_nul_repair(tmp_path / "small", chunk_size=4)
    assert q.read_bytes() == expected
    assert report.repaired["x.jsonl"].runs == 1
    assert report.repaired["x.jsonl"].bytes_removed == 20


def test_a_second_run_changes_nothing(tmp_path):
    p = _write(tmp_path, "x.jsonl", b'{"a":1}\n' + b"\0" * 20 + b'{"b":2}\n')
    run_jsonl_nul_repair(tmp_path)
    after_first = p.read_bytes()

    report = run_jsonl_nul_repair(tmp_path)

    assert report.repaired == {}
    assert p.read_bytes() == after_first
    assert not (tmp_path / "x.jsonl.pre-618.1").exists()


def test_an_existing_backup_is_kept_and_the_new_one_is_numbered(tmp_path):
    original = b'{"a":1}\n' + b"\0" * 5 + b'{"b":2}\n'
    _write(tmp_path, "x.jsonl", original)
    _write(tmp_path, "x.jsonl.pre-618", b"somebody else's bytes")

    report = run_jsonl_nul_repair(tmp_path)

    assert (tmp_path / "x.jsonl.pre-618").read_bytes() == b"somebody else's bytes"
    assert (tmp_path / "x.jsonl.pre-618.1").read_bytes() == original
    assert report.repaired["x.jsonl"].backup == "x.jsonl.pre-618.1"


def test_rotated_files_are_repaired_and_gz_files_are_not(tmp_path):
    rotated = _write(tmp_path, "x.jsonl.1", b'{"a":1}\n' + b"\0" * 5)
    gz = _write(tmp_path, "x.jsonl.1.gz", b"\x1f\x8b\0\0\0\0")

    report = run_jsonl_nul_repair(tmp_path)

    assert rotated.read_bytes() == b'{"a":1}\n'
    assert "x.jsonl.1" in report.repaired
    assert gz.read_bytes() == b"\x1f\x8b\0\0\0\0"
    assert "x.jsonl.1.gz" not in report.repaired


def test_the_reader_warns_no_more_once_the_file_is_repaired(tmp_path, caplog):
    p = _write(tmp_path, "x.jsonl", b'{"a":1}\n' + b"\0" * 20 + b'{"b":2}\n')

    with caplog.at_level(logging.WARNING):
        list(iter_jsonl(p))
    assert caplog.records

    run_jsonl_nul_repair(tmp_path)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert list(iter_jsonl(p)) == [{"a": 1}, {"b": 2}]
    assert not caplog.records


def test_an_append_during_the_repair_waits_and_is_kept(tmp_path, monkeypatch):
    p = _write(tmp_path, "x.jsonl", b'{"a":1}\n' + b"\0" * 20 + b'{"b":2}\n')
    real = jsonl_nul_repair.save_stream_atomic
    during: list[object] = []

    def wrapped(path, write, **kw):
        if path == p:
            during.append(append_jsonl(p, {"c": 3}, lock="try"))
        return real(path, write, **kw)

    monkeypatch.setattr(jsonl_nul_repair, "save_stream_atomic", wrapped)
    run_jsonl_nul_repair(tmp_path)

    assert during == [None]
    append_jsonl(p, {"c": 3})
    assert list(iter_jsonl(p)) == [{"a": 1}, {"b": 2}, {"c": 3}]
    assert p.read_bytes().startswith(b'{"a":1}\n{"b":2}\n{"c"')


def test_an_unreadable_file_leaves_the_state_unwritten_and_the_next_run_repairs_it(tmp_path, monkeypatch):
    p = _write(tmp_path, "x.jsonl", b'{"a":1}\n' + b"\0" * 20 + b'{"b":2}\n')
    real = jsonl_nul_repair._open_for_read
    broken = {"on": True}

    def flaky(path):
        if broken["on"] and path == p:
            raise PermissionError("locked")
        return real(path)

    monkeypatch.setattr(jsonl_nul_repair, "_open_for_read", flaky)

    report = run_jsonl_nul_repair(tmp_path)
    assert report.done is False
    assert report.failed == ["x.jsonl"]
    assert not (tmp_path / STATE_FILENAME).exists()
    assert should_run_jsonl_nul_repair(tmp_path) is True

    broken["on"] = False
    report = run_jsonl_nul_repair(tmp_path)
    assert report.done is True
    assert list(iter_jsonl(p)) == [{"a": 1}, {"b": 2}]
    assert (tmp_path / STATE_FILENAME).exists()
    assert should_run_jsonl_nul_repair(tmp_path) is False


def test_an_unlistable_folder_leaves_the_state_unwritten(tmp_path):
    import os
    import sys

    import pytest

    if sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0):
        pytest.skip("needs POSIX permissions and a non-root user")
    p = _write(tmp_path, "sub/x.jsonl", b'{"a":1}\n' + b"\0" * 20 + b'{"b":2}\n')
    (tmp_path / "sub").chmod(0o000)
    try:
        report = run_jsonl_nul_repair(tmp_path)
        assert report.done is False
        assert "sub" in report.failed
        assert not (tmp_path / STATE_FILENAME).exists()
    finally:
        (tmp_path / "sub").chmod(0o755)

    report = run_jsonl_nul_repair(tmp_path)
    assert report.done is True
    assert list(iter_jsonl(p)) == [{"a": 1}, {"b": 2}]
    assert (tmp_path / STATE_FILENAME).exists()


def test_the_emotion_engines_state_folder_is_left_alone(tmp_path):
    original = b'{"a":1}\n' + b"\0" * 20 + b'{"b":2}\n'
    p = _write(tmp_path, "state/emotion_changes/2026-11-20.jsonl", original)

    report = run_jsonl_nul_repair(tmp_path)

    assert p.read_bytes() == original
    assert not list(p.parent.glob("*.pre-618*"))
    assert report.repaired == {}


def test_a_failed_rewrite_removes_its_backup(tmp_path, monkeypatch):
    original = b'{"a":1}\n' + b"\0" * 20 + b'{"b":2}\n'
    p = _write(tmp_path, "x.jsonl", original)
    real = jsonl_nul_repair.save_stream_atomic
    calls = {"n": 0}

    def second_raises(path, write, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real(path, write, **kwargs)

    monkeypatch.setattr(jsonl_nul_repair, "save_stream_atomic", second_raises)

    report = run_jsonl_nul_repair(tmp_path)

    assert p.read_bytes() == original
    assert not list(tmp_path.glob("*.pre-618*"))
    assert report.failed == ["x.jsonl"]


def test_a_symlinked_file_is_not_replaced(tmp_path):
    import os

    import pytest

    outside = tmp_path / "outside"
    persona = tmp_path / "persona"
    persona.mkdir()
    original = b'{"a":1}\n' + b"\0" * 20 + b'{"b":2}\n'
    target = _write(outside, "real.jsonl", original)
    link = persona / "link.jsonl"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here")

    run_jsonl_nul_repair(persona)

    assert os.path.islink(link)
    assert target.read_bytes() == original


def test_the_file_mode_is_kept(tmp_path):
    import stat
    import sys

    import pytest

    if sys.platform == "win32":
        pytest.skip("POSIX permission bits")
    p = _write(tmp_path, "x.jsonl", b'{"a":1}\n' + b"\0" * 20 + b'{"b":2}\n')
    p.chmod(0o600)

    run_jsonl_nul_repair(tmp_path)

    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "x.jsonl.pre-618").stat().st_mode) == 0o600


def test_many_short_runs_are_repaired_quickly(tmp_path):
    import time

    p = _write(tmp_path, "x.jsonl", b"a\0" * 100_000)  # 200 KB of alternating byte and NUL

    start = time.perf_counter()
    report = run_jsonl_nul_repair(tmp_path)
    elapsed = time.perf_counter() - start

    assert report.repaired["x.jsonl"].runs == 100_000
    assert p.read_bytes() == b"a\n" * 100_000
    assert elapsed < 2.0


def test_supervisor_startup_repairs_a_damaged_file(tmp_path):
    """The real repair, through the real run_folded: nothing is patched."""
    import json
    import threading

    from brain.bridge.events import EventBus
    from brain.bridge.provider import FakeProvider
    from brain.bridge.supervisor import run_folded

    persona_dir = tmp_path / "test-persona"
    persona_dir.mkdir()
    (persona_dir / "active_conversations").mkdir()
    (persona_dir / "persona_config.json").write_text('{"provider": "fake", "searcher": "noop"}')
    p = _write(
        persona_dir, "heartbeats.log.jsonl", b'{"a":1}\n' + b"\0" * 20 + b'{"b":2}\n'
    )

    stop = threading.Event()
    stop.set()
    run_folded(
        stop,
        persona_dir=persona_dir,
        provider=FakeProvider(),
        event_bus=EventBus(),
        tick_interval_s=0.1,
        heartbeat_interval_s=None,
    )

    assert list(iter_jsonl(p)) == [{"a": 1}, {"b": 2}]
    state = json.loads((persona_dir / STATE_FILENAME).read_text(encoding="utf-8"))
    assert state["done"] is True
