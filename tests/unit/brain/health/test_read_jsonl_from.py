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
def test_an_over_long_integer_is_a_bad_line_like_in_iter_jsonl_raw(tmp_path: Path) -> None:
    huge = b'{"n": ' + b"9" * (sys.get_int_max_str_digits() + 1) + b"}\n"
    path = _write(tmp_path, b'{"a": 1}\n' + huge + b'{"a": 3}\n')
    read = jio.read_jsonl_from(path)
    assert [(r.line, r.data is None) for r in read.records] == [(1, False), (2, True), (3, False)]
    assert [rec for _text, rec in jio.iter_jsonl_raw(path, bad_lines=True)][1] is None


@pytest.mark.skipif(sys.getrecursionlimit() > 100000, reason="needs a depth the parser rejects")
def test_a_too_deeply_nested_line_is_a_bad_line_with_a_reason(tmp_path: Path) -> None:
    deep = b"[" * 200000 + b"]" * 200000 + b"\n"
    path = _write(tmp_path, b'{"a": 1}\n' + deep + b'{"a": 3}\n')
    read = jio.read_jsonl_from(path)
    assert [(r.line, r.data is None) for r in read.records] == [(1, False), (2, True), (3, False)]
    reason = read.records[1].reason
    assert reason is not None and reason.startswith("malformed JSON")
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


# --- streaming (hana review 1): no whole-file load, bounded memory, opened through the read retry ------------------


def _forbid_whole_file_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.utils import file_ops

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("read_jsonl_from loaded the whole file")

    for owner in (file_ops, jio):
        monkeypatch.setattr(owner, "read_bytes_stat", refuse, raising=False)
    monkeypatch.setattr(Path, "read_bytes", refuse)


def test_reads_without_loading_the_whole_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, b'{"a": 1}\n\nnot json\n{"a": 4}\n{"a": 5')
    _forbid_whole_file_reads(monkeypatch)
    read = jio.read_jsonl_from(path, 1)
    assert [(r.line, r.data is None) for r in read.records] == [(3, True), (4, False)] and read.end_line == 4


def test_memory_stays_bounded_on_a_large_file(tmp_path: Path) -> None:
    import tracemalloc

    line = b'{"k": "' + b"x" * 90 + b'"}\n'
    count = 60_000
    path = _write(tmp_path, line * count)
    size = path.stat().st_size
    assert size > 5_000_000
    tracemalloc.start()
    try:
        read = jio.read_jsonl_from(path, count)  # skips every line
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert read == jio.JsonlRead([], count)
    assert peak < size // 10
    tail = jio.read_jsonl_from(path, count - 2)
    assert [r.line for r in tail.records] == [count - 1, count] and tail.end_line == count


def test_opens_the_file_through_retry_transient_with_the_read_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brain import dev_constants
    from brain.utils import file_ops

    seen: list[tuple[object, dict]] = []
    real = file_ops.retry_transient

    def spy(fn, *args, **kwargs):
        seen.append((fn, kwargs))
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(file_ops, "retry_transient", spy)
    jio.read_jsonl_from(_write(tmp_path, b'{"a": 1}\n'))
    assert any(
        fn is open and kw == {"attempts": dev_constants.FILE_READ_ATTEMPTS, "windows_only": True} for fn, kw in seen
    )


@pytest.mark.no_base_failure(reason="pins the unchanged contract through the rewrite: a skipped line is counted, never parsed or reported")
def test_a_bad_line_before_the_start_line_is_counted_and_not_reported(tmp_path: Path) -> None:
    read = jio.read_jsonl_from(_write(tmp_path, b'not json\n{"a": 2}\n{"a": 3}\n'), 2)
    assert _lines(read) == [(3, {"a": 3})] and read.end_line == 3


@pytest.mark.no_base_failure(reason="pins the unchanged contract through the rewrite: a torn last line is not counted even when start_line is past the end")
def test_a_torn_last_line_is_not_counted_when_the_start_line_is_past_the_end(tmp_path: Path) -> None:
    path = _write(tmp_path, b'{"a": 1}\n{"a": 2}\n{"a": 3')
    assert jio.read_jsonl_from(path, 99) == jio.JsonlRead([], 2)
    assert jio.read_jsonl_from(path, 2) == jio.JsonlRead([], 2)


@pytest.mark.no_base_failure(reason="pins the unchanged contract through the rewrite: only an open that is refused on Windows is retried, and any other OSError is raised")
def test_open_errors_are_raised_and_only_a_windows_refusal_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.utils import file_ops

    path = _write(tmp_path, b'{"a": 1}\n')
    with pytest.raises(OSError):
        jio.read_jsonl_from(tmp_path)  # a folder: not an empty result
    tries: list[int] = []

    def flaky_open(file, *args, **kwargs):
        tries.append(1)
        if len(tries) < 3:
            raise PermissionError(13, "busy")
        return real_open(file, *args, **kwargs)

    real_open = open
    monkeypatch.setattr(file_ops.time, "sleep", lambda _s: None)
    monkeypatch.setattr(file_ops, "open", flaky_open, raising=False)
    monkeypatch.setattr(file_ops, "_IS_WINDOWS", True)
    assert jio.read_jsonl_from(path).end_line == 1 and len(tries) == 3
    tries.clear()
    monkeypatch.setattr(file_ops, "_IS_WINDOWS", False)
    with pytest.raises(PermissionError):
        jio.read_jsonl_from(path)
    assert len(tries) == 1
