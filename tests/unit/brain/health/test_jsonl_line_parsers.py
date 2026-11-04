"""The line parsers behind ``iter_jsonl``, ``parse_jsonl``, ``iter_jsonl_raw`` and ``iter_jsonl_skipping_corrupt`` keep their
warning texts, their strict-mode reasons and their exception chains when they share one line classifier (#643).

The first block pins what the parsers did before the change (it passes on the old code by design, so each test carries the
``no_base_failure`` mark). The last test is new behaviour: a line the JSON parser rejects with a plain ``ValueError`` (an
integer of more than the interpreter's digit limit) used to abort ``iter_jsonl`` and ``parse_jsonl`` and is now skipped
with a warning by every parser, as ``iter_jsonl_raw`` always did.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from brain.health import json_io as jio
from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt

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
@pytest.mark.parametrize("reader_name", ["iter_jsonl", "iter_jsonl_raw", "parse_jsonl", "iter_jsonl_skipping_corrupt"])
def test_an_over_long_integer_line_is_skipped_with_a_warning_by_every_parser(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, reader_name: str
) -> None:
    huge = b'{"n": ' + b"9" * (sys.get_int_max_str_digits() + 1) + b"}\n"
    path = _file(tmp_path, GOOD, huge, GOOD)
    with caplog.at_level(logging.WARNING):
        if reader_name == "parse_jsonl":
            rows = list(jio.parse_jsonl(path.read_bytes()))
        elif reader_name == "iter_jsonl_skipping_corrupt":
            rows = list(iter_jsonl_skipping_corrupt(path))
        elif reader_name == "iter_jsonl_raw":
            rows = [rec for _text, rec in jio.iter_jsonl_raw(path)]
        else:
            rows = list(jio.iter_jsonl(path))
    assert rows == [{"a": 1}, {"a": 1}]
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
