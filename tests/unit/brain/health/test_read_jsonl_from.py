"""``json_io.read_jsonl_from``: read a JSONL file from a line position (#643).

The caller keeps the last ``end_line`` it was given and passes it back as ``start_line``: lines up to that number are
skipped, later complete lines come back with their 1-based line numbers, and a last line with no newline yet (a torn
write) is neither returned nor counted, so it is read whole the next time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from brain.health import json_io as jio


def _write(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "q.jsonl"
    path.write_bytes(content)
    return path


def _lines(read: jio.JsonlRead) -> list[tuple[int, dict | None]]:
    return [(r.line, r.data) for r in read.records]


def test_reads_every_complete_line_with_its_number(tmp_path: Path) -> None:
    path = _write(tmp_path, b'{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    read = jio.read_jsonl_from(path)
    assert _lines(read) == [(1, {"a": 1}), (2, {"a": 2}), (3, {"a": 3})]
    assert read.end_line == 3
    assert all(r.reason is None for r in read.records)


def test_starts_after_the_given_line(tmp_path: Path) -> None:
    path = _write(tmp_path, b'{"a": 1}\n{"a": 2}\n{"a": 3}\n{"a": 4}\n')
    read = jio.read_jsonl_from(path, 2)
    assert _lines(read) == [(3, {"a": 3}), (4, {"a": 4})] and read.end_line == 4
    assert jio.read_jsonl_from(path, 4) == jio.JsonlRead([], 4)


def test_a_start_line_past_the_end_gives_no_records_and_the_true_end(tmp_path: Path) -> None:
    path = _write(tmp_path, b'{"a": 1}\n{"a": 2}\n')
    assert jio.read_jsonl_from(path, 99) == jio.JsonlRead([], 2)


def test_a_negative_start_line_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, b'{"a": 1}\n')
    with pytest.raises(ValueError):
        jio.read_jsonl_from(path, -1)


def test_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert jio.read_jsonl_from(tmp_path / "nope") == jio.JsonlRead([], 0)
    assert jio.read_jsonl_from(tmp_path / "nope", 5) == jio.JsonlRead([], 0)


def test_an_empty_file_is_empty(tmp_path: Path) -> None:
    assert jio.read_jsonl_from(_write(tmp_path, b"")) == jio.JsonlRead([], 0)


def test_a_torn_last_line_is_neither_returned_nor_counted_and_is_read_whole_next_time(tmp_path: Path) -> None:
    path = _write(tmp_path, b'{"a": 1}\n{"a": 2}\n{"a": 3')
    first = jio.read_jsonl_from(path)
    assert _lines(first) == [(1, {"a": 1}), (2, {"a": 2})] and first.end_line == 2
    path.write_bytes(b'{"a": 1}\n{"a": 2}\n{"a": 3}\n')  # the writer finishes the line
    second = jio.read_jsonl_from(path, first.end_line)
    assert _lines(second) == [(3, {"a": 3})] and second.end_line == 3


def test_a_bad_line_is_a_record_with_a_reason_and_still_counts(tmp_path: Path) -> None:
    content = b'{"a": 1}\nnot json\n[1, 2]\n{"b": "caf\xff"}\n{"a": 5}\n'
    read = jio.read_jsonl_from(_write(tmp_path, content))
    assert [r.line for r in read.records] == [1, 2, 3, 4, 5] and read.end_line == 5
    assert read.records[0] == jio.JsonlRecord(1, {"a": 1}, None)
    assert read.records[4] == jio.JsonlRecord(5, {"a": 5}, None)
    reasons = {r.line: r.reason for r in read.records[1:4]}
    assert reasons[2].startswith("malformed JSON")
    assert reasons[3] == "not a JSON object (list)"
    assert reasons[4].startswith("invalid UTF-8")
    assert all(r.data is None for r in read.records[1:4])
    assert "not json" not in reasons[2]  # a reason never carries the line's content


def test_a_zero_filled_line_is_a_bad_line(tmp_path: Path) -> None:
    content = b'{"a": 1}\n' + b"\x00" * 64 + b'\n{"a": 3}\n'
    read = jio.read_jsonl_from(_write(tmp_path, content))
    assert read.end_line == 3
    assert read.records[1].line == 2 and read.records[1].data is None
    assert read.records[1].reason is not None and read.records[1].reason.startswith("malformed JSON")
    assert _lines(read)[2] == (3, {"a": 3})


def test_blank_lines_count_as_lines_and_give_no_record(tmp_path: Path) -> None:
    read = jio.read_jsonl_from(_write(tmp_path, b'{"a": 1}\n\n   \n{"a": 4}\n'))
    assert _lines(read) == [(1, {"a": 1}), (4, {"a": 4})] and read.end_line == 4


def test_an_empty_object_is_a_record(tmp_path: Path) -> None:
    assert _lines(jio.read_jsonl_from(_write(tmp_path, b"{}\n"))) == [(1, {})]


def test_a_byte_order_mark_and_carriage_returns_are_handled_like_the_other_readers(tmp_path: Path) -> None:
    path = _write(tmp_path, b'\xef\xbb\xbf{"a": 1}\r\n{"a": 2}\r\n')
    read = jio.read_jsonl_from(path)
    assert _lines(read) == [(1, {"a": 1}), (2, {"a": 2})] and read.end_line == 2
    assert [rec for _text, rec in jio.iter_jsonl_raw(path)] == [{"a": 1}, {"a": 2}]


def test_after_a_crash_tail_append_jsonl_makes_the_fragment_a_counted_bad_line(tmp_path: Path) -> None:
    path = _write(tmp_path, b'{"a": 1}\n{"a": 2')  # a crash cut the second line
    before = jio.read_jsonl_from(path)
    assert before.end_line == 1
    jio.append_jsonl(path, {"a": 3})  # writes a newline first, so the fragment becomes a complete (bad) line
    after = jio.read_jsonl_from(path, before.end_line)
    assert after.end_line == 3
    assert [(r.line, r.data) for r in after.records] == [(2, None), (3, {"a": 3})]
    assert after.records[0].reason is not None and after.records[0].reason.startswith("malformed JSON")


@pytest.mark.skipif(sys.get_int_max_str_digits() == 0, reason="the interpreter's integer digit limit is off")
def test_an_over_long_integer_is_a_bad_line_like_in_the_other_readers(tmp_path: Path) -> None:
    huge = b'{"n": ' + b"9" * (sys.get_int_max_str_digits() + 1) + b"}\n"
    path = _write(tmp_path, b'{"a": 1}\n' + huge + b'{"a": 3}\n')
    read = jio.read_jsonl_from(path)
    assert [(r.line, r.data is None) for r in read.records] == [(1, False), (2, True), (3, False)]
    assert [rec for _text, rec in jio.iter_jsonl_raw(path, bad_lines=True)][1] is None


@pytest.mark.parametrize(
    "bad",
    [b"nope\n", b"[1]\n", b'{"b": "caf\xff"}\n', b"\x00\x00\n"],
)
def test_a_bad_line_has_the_same_reason_as_iter_jsonl_raw_gives_in_strict_mode(tmp_path: Path, bad: bytes) -> None:
    path = _write(tmp_path, b'{"a": 1}\n' + bad)
    read = jio.read_jsonl_from(path)
    with pytest.raises(jio.JsonFileError) as exc:
        list(jio.iter_jsonl_raw(path, strict=True))
    assert read.records[1].reason == exc.value.detail and read.records[1].line == exc.value.line == 2
