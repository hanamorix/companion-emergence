"""The one-time NUL repair (#618): an unclean stop left runs of NUL bytes in JSONL files."""

from __future__ import annotations

import logging
from pathlib import Path

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


def test_a_run_after_a_torn_record_becomes_a_newline(tmp_path):
    p = _write(tmp_path, "x.jsonl", b'{"a":1}\n{"to' + b"\0" * 8 + b'{"b":2}\n')
    run_jsonl_nul_repair(tmp_path)
    assert p.read_bytes() == b'{"a":1}\n{"to\n{"b":2}\n'


def test_a_run_before_a_newline_after_a_torn_record_adds_no_blank_line(tmp_path):
    p = _write(tmp_path, "x.jsonl", b'{"to' + b"\0" * 4 + b'\n{"b":2}\n')
    run_jsonl_nul_repair(tmp_path)
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
