"""The line parsers behind ``iter_jsonl``, ``parse_jsonl``, ``iter_jsonl_raw`` and ``iter_jsonl_skipping_corrupt`` keep their
warning texts, their strict-mode reasons and their exception chains when they share one line classifier (#643).

The first block pins what the parsers did before the change (it passes on the old code by design, so each test carries the
``no_base_failure`` mark). The tests after them are new behaviour: a line the JSON parser rejects with a plain ``ValueError`` (an
integer of more than the interpreter's digit limit) or ``RecursionError`` used to abort ``iter_jsonl``, ``parse_jsonl``, the
skipping reader and ``read_jsonl_tail``, and is now skipped with a warning by every parser, as ``iter_jsonl_raw`` always did; in strict
mode it raises ``JsonlLineError`` from the line parser and ``JsonFileError(kind="line")`` through ``json_io``'s readers.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pytest

from brain.health import json_io as jio
from brain.health.jsonl_reader import JsonlLineError, _parse_lines, iter_jsonl_skipping_corrupt

_PINNED = pytest.mark.no_base_failure(reason="pins the parsers' existing warning texts and strict reasons, which the old code already has")

BAD_UTF8 = b'{"b": "caf\xff"}\n'
BAD_JSON = b"nope\n"
NOT_OBJECT = b"[1]\n"
GOOD = b'{"a": 1}\n'


def _file(tmp_path: Path, *lines: bytes) -> Path:
    path = tmp_path / "x.jsonl"
    path.write_bytes(b"".join(lines))
    return path


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


@_PINNED
def test_iter_jsonl_warning_texts(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _file(tmp_path, GOOD, BAD_JSON, NOT_OBJECT, BAD_UTF8)
    with caplog.at_level(logging.WARNING):
        assert list(jio.iter_jsonl(path)) == [{"a": 1}]
    msgs = _messages(caplog)
    assert msgs == [
        f"skipping malformed jsonl line 2 in {path}: Expecting value: line 1 column 1 (char 0) | content: 'nope'",
        f"skipping non-dict jsonl line 3 in {path} (value type=list) | content: '[1]'",
        f"skipping malformed jsonl line 4 in {path}: invalid UTF-8 (surrogates not allowed) | content: '{{\"b\": \"caf�\"}}\\n'",
    ]


@_PINNED
def test_skipping_corrupt_reader_has_the_same_warning_texts(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _file(tmp_path, BAD_JSON, NOT_OBJECT, BAD_UTF8, GOOD)
    with caplog.at_level(logging.WARNING):
        assert list(iter_jsonl_skipping_corrupt(path)) == [{"a": 1}]
    assert len(_messages(caplog)) == 3
    assert "Expecting value" in _messages(caplog)[0] and "value type=list" in _messages(caplog)[1]
    assert "invalid UTF-8 (surrogates not allowed)" in _messages(caplog)[2]


@_PINNED
def test_iter_jsonl_raw_warning_texts(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _file(tmp_path, BAD_JSON, NOT_OBJECT, BAD_UTF8, GOOD)
    with caplog.at_level(logging.WARNING):
        assert list(jio.iter_jsonl_raw(path)) == [('{"a": 1}', {"a": 1})]
    assert _messages(caplog) == [
        f"skipping malformed jsonl line 1 in {path}: malformed JSON (Expecting value: line 1 column 1 (char 0)) | content: 'nope'",
        f"skipping malformed jsonl line 2 in {path}: not a JSON object (list) | content: '[1]'",
        f"skipping malformed jsonl line 3 in {path}: invalid UTF-8 (surrogates not allowed) | content: '{{\"b\": \"caf�\"}}'",
    ]


@_PINNED
@pytest.mark.parametrize(
    ("line", "reason"),
    [
        (BAD_JSON, "malformed JSON (Expecting value: line 1 column 1 (char 0))"),
        (NOT_OBJECT, "not a JSON object (list)"),
        (BAD_UTF8, "invalid UTF-8 (surrogates not allowed)"),
    ],
)
def test_strict_reasons_and_line_numbers(tmp_path: Path, line: bytes, reason: str) -> None:
    path = _file(tmp_path, GOOD, line)
    for reader in (jio.iter_jsonl, jio.iter_jsonl_raw):
        with pytest.raises(jio.JsonFileError) as exc:
            list(reader(path, strict=True))
        assert (exc.value.kind, exc.value.detail, exc.value.line) == ("line", reason, 2)
    with pytest.raises(jio.JsonFileError) as exc:
        list(jio.parse_jsonl(GOOD + line, strict=True))
    assert (exc.value.kind, exc.value.detail, exc.value.line) == ("line", reason, 2)


@_PINNED
def test_strict_error_keeps_the_parsers_own_exception_in_its_chain(tmp_path: Path) -> None:
    path = _file(tmp_path, BAD_JSON)
    with pytest.raises(jio.JsonFileError) as exc:
        list(jio.iter_jsonl(path, strict=True))
    chain = []
    err: BaseException | None = exc.value
    while err is not None:
        chain.append(type(err).__name__)
        err = err.__cause__
    assert "JSONDecodeError" in chain


@_PINNED
def test_blank_lines_count_for_line_numbers_and_give_no_record(tmp_path: Path) -> None:
    path = _file(tmp_path, GOOD, b"\n", b"   \n", BAD_JSON)
    with pytest.raises(jio.JsonFileError) as exc:
        list(jio.iter_jsonl(path, strict=True))
    assert exc.value.line == 4
    assert list(jio.iter_jsonl_raw(path)) == [('{"a": 1}', {"a": 1})]


@pytest.mark.skipif(sys.get_int_max_str_digits() == 0, reason="the interpreter's integer digit limit is off")
@pytest.mark.parametrize("reader_name", ["iter_jsonl", "parse_jsonl", "iter_jsonl_skipping_corrupt"])
def test_an_over_long_integer_line_is_skipped_with_a_warning_by_the_readers_that_aborted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, reader_name: str
) -> None:
    huge = b'{"n": ' + b"9" * (sys.get_int_max_str_digits() + 1) + b"}\n"
    path = _file(tmp_path, GOOD, huge, GOOD)
    with caplog.at_level(logging.WARNING):
        if reader_name == "parse_jsonl":
            rows = list(jio.parse_jsonl(path.read_bytes()))
        elif reader_name == "iter_jsonl_skipping_corrupt":
            rows = list(iter_jsonl_skipping_corrupt(path))
        else:
            rows = list(jio.iter_jsonl(path))
    assert rows == [{"a": 1}, {"a": 1}]
    assert len(_messages(caplog)) == 1


@_PINNED
@pytest.mark.skipif(sys.get_int_max_str_digits() == 0, reason="the interpreter's integer digit limit is off")
def test_iter_jsonl_raw_already_skipped_an_over_long_integer_line(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    huge = b'{"n": ' + b"9" * (sys.get_int_max_str_digits() + 1) + b"}\n"
    path = _file(tmp_path, GOOD, huge, GOOD)
    with caplog.at_level(logging.WARNING):
        assert [rec for _text, rec in jio.iter_jsonl_raw(path)] == [{"a": 1}, {"a": 1}]
    assert len(_messages(caplog)) == 1


@pytest.mark.skipif(sys.getrecursionlimit() > 100000, reason="needs a depth the parser rejects")
def test_a_too_deeply_nested_line_is_skipped_not_fatal(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    deep = b"[" * 200000 + b"]" * 200000 + b"\n"
    path = _file(tmp_path, GOOD, deep, GOOD)
    with caplog.at_level(logging.WARNING):
        assert list(jio.iter_jsonl(path)) == [{"a": 1}, {"a": 1}]
    assert len(_messages(caplog)) == 1


@_PINNED
def test_an_empty_object_line_is_a_record(tmp_path: Path) -> None:
    path = _file(tmp_path, b"{}\n", GOOD)
    assert list(jio.iter_jsonl(path)) == [{}, {"a": 1}]
    assert list(jio.iter_jsonl_raw(path)) == [("{}", {}), ('{"a": 1}', {"a": 1})]


# --- strict mode and the tail reader on a line the JSON parser rejects with a plain ValueError or RecursionError (#643) ------

_NEEDS_DIGIT_LIMIT = pytest.mark.skipif(sys.get_int_max_str_digits() == 0, reason="the interpreter's integer digit limit is off")
_NEEDS_DEPTH_LIMIT = pytest.mark.skipif(sys.getrecursionlimit() > 100000, reason="needs a depth the parser rejects")


def _huge_int_line() -> bytes:
    return b'{"n": ' + b"9" * (sys.get_int_max_str_digits() + 1) + b"}\n"


def _deep_line() -> bytes:
    return b"[" * 200000 + b"]" * 200000 + b"\n"


@_NEEDS_DIGIT_LIMIT
def test_strict_line_parser_raises_jsonl_line_error_for_an_over_long_integer() -> None:
    """A bare ``ValueError`` used to escape; the error is now the line error, which is a ``ValueError`` and has the line number."""
    stream = io.StringIO(_huge_int_line().decode())
    with pytest.raises(JsonlLineError) as exc:
        list(_parse_lines(stream, Path("x.jsonl"), strict=True))
    assert isinstance(exc.value, ValueError)
    assert exc.value.line == 1


@_NEEDS_DEPTH_LIMIT
def test_strict_line_parser_raises_jsonl_line_error_for_a_too_deeply_nested_line() -> None:
    stream = io.StringIO(_deep_line().decode())
    with pytest.raises(JsonlLineError) as exc:
        list(_parse_lines(stream, Path("x.jsonl"), strict=True))
    assert exc.value.line == 1


@_NEEDS_DIGIT_LIMIT
@pytest.mark.parametrize("reader_name", ["iter_jsonl", "parse_jsonl", "read_jsonl_tail"])
def test_strict_readers_raise_a_line_file_error_for_an_over_long_integer(tmp_path: Path, reader_name: str) -> None:
    """Through ``json_io`` the strict error is ``JsonFileError(kind="line")``: an ``OSError`` and a ``ValueError``."""
    path = _file(tmp_path, GOOD, _huge_int_line())
    with pytest.raises(jio.JsonFileError) as exc:
        if reader_name == "parse_jsonl":
            list(jio.parse_jsonl(path.read_bytes(), strict=True))
        elif reader_name == "read_jsonl_tail":
            jio.read_jsonl_tail(path, 2, strict=True, recover=jio.Recovery.RAISE)
        else:
            list(jio.iter_jsonl(path, strict=True))
    assert exc.value.kind == "line"
    assert isinstance(exc.value, OSError) and isinstance(exc.value, ValueError)
    if reader_name != "read_jsonl_tail":  # the tail reader numbers the lines of the window it read
        assert exc.value.line == 2


@_NEEDS_DEPTH_LIMIT
@pytest.mark.parametrize("reader_name", ["iter_jsonl", "read_jsonl_tail"])
def test_strict_readers_raise_a_line_file_error_for_a_too_deeply_nested_line(tmp_path: Path, reader_name: str) -> None:
    path = _file(tmp_path, GOOD, _deep_line())
    with pytest.raises(jio.JsonFileError) as exc:
        if reader_name == "read_jsonl_tail":
            jio.read_jsonl_tail(path, 2, strict=True, recover=jio.Recovery.RAISE)
        else:
            list(jio.iter_jsonl(path, strict=True))
    assert exc.value.kind == "line"


@_NEEDS_DIGIT_LIMIT
def test_the_tail_reader_skips_an_over_long_integer_line_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _file(tmp_path, GOOD, _huge_int_line(), GOOD)
    with caplog.at_level(logging.WARNING):
        assert jio.read_jsonl_tail(path, 3, recover=jio.Recovery.RAISE) == [{"a": 1}, {"a": 1}]
    assert len(_messages(caplog)) == 1
