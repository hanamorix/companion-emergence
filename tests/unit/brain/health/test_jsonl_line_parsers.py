"""The line parsers behind ``iter_jsonl``, ``parse_jsonl``, ``iter_jsonl_raw`` and ``iter_jsonl_skipping_corrupt`` keep their
warning texts, their strict-mode reasons and their exception chains when they share one line classifier (#643).

Every test here pins what the parsers did before the change (it passes on the old code by design, so each carries the
``no_base_failure`` mark). That includes a known defect, kept by the owner's ruling and filed separately: a line the JSON parser
rejects with a plain ``ValueError`` (an integer of more than the interpreter's digit limit) or ``RecursionError`` (nesting too deep)
stops ``iter_jsonl``, ``read_jsonl``, ``parse_jsonl``, ``read_jsonl_tail`` and the skipping reader with the parser's own bare
exception, in strict mode too, while ``iter_jsonl_raw`` skips such a line (strict: ``JsonFileError(kind="line")``). A fix that makes
the first group skip it must change those tests on purpose.
"""

from __future__ import annotations

import gzip
import io
import logging
import sys
from pathlib import Path

import pytest

from brain.health import json_io as jio
from brain.health.jsonl_reader import JsonlLineError, _parse_lines, iter_jsonl_skipping_corrupt

_PINNED = pytest.mark.no_base_failure(reason="pins what the parsers already do on origin/main: their warning texts, strict reasons and what they do with a line the JSON parser cannot load")

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


@_PINNED
def test_an_empty_object_line_is_a_record(tmp_path: Path) -> None:
    path = _file(tmp_path, b"{}\n", GOOD)
    assert list(jio.iter_jsonl(path)) == [{}, {"a": 1}]
    assert list(jio.iter_jsonl_raw(path)) == [("{}", {}), ('{"a": 1}', {"a": 1})]


# --- a line the JSON parser rejects with a plain ValueError or RecursionError (the behaviour from before #643) ---------------

_NEEDS_DIGIT_LIMIT = pytest.mark.skipif(sys.get_int_max_str_digits() == 0, reason="the interpreter's integer digit limit is off")
_NEEDS_DEPTH_LIMIT = pytest.mark.skipif(sys.getrecursionlimit() > 100000, reason="needs a depth the parser rejects")


def _huge_int_line() -> bytes:
    return b'{"n": ' + b"9" * (sys.get_int_max_str_digits() + 1) + b"}\n"


def _deep_line() -> bytes:
    return b"[" * 200000 + b"]" * 200000 + b"\n"


# (line builder, the exact exception the JSON parser gives for it)
_UNLOADABLE = [
    pytest.param(_huge_int_line, ValueError, id="over-long-integer", marks=_NEEDS_DIGIT_LIMIT),
    pytest.param(_deep_line, RecursionError, id="too-deeply-nested", marks=_NEEDS_DEPTH_LIMIT),
]

# Readers that stop on such a line. The ones that return a list give back nothing when they stop.
_ABORTING_READERS = [
    "iter_jsonl",
    "iter_jsonl_gz",
    "read_jsonl",
    "parse_jsonl",
    "iter_jsonl_skipping_corrupt",
    "read_jsonl_tail",
    "read_jsonl_tail_gz",
]
_RETURNS_A_LIST = {"read_jsonl", "read_jsonl_tail", "read_jsonl_tail_gz"}
_HAVE_STRICT = [name for name in _ABORTING_READERS if name != "iter_jsonl_skipping_corrupt"]


def _run(reader_name: str, tmp_path: Path, content: bytes, *, strict: bool) -> tuple[list[dict], Exception | None]:
    """Run one reader over ``content``: (the rows it gave before it stopped, the exception that stopped it or None)."""
    path = _file(tmp_path, content)
    if reader_name.endswith("_gz"):
        path = tmp_path / "x.jsonl.gz"
        path.write_bytes(gzip.compress(content))
    recover = jio.Recovery.RAISE
    rows: list[dict] = []
    try:
        if reader_name in ("iter_jsonl", "iter_jsonl_gz"):
            stream = jio.iter_jsonl(path, strict=strict, recover=recover)
        elif reader_name == "read_jsonl":
            stream = iter(jio.read_jsonl(path, strict=strict, recover=recover))
        elif reader_name == "parse_jsonl":
            stream = jio.parse_jsonl(content, strict=strict)
        elif reader_name == "iter_jsonl_skipping_corrupt":
            stream = iter_jsonl_skipping_corrupt(path)
        else:
            stream = iter(jio.read_jsonl_tail(path, 3, strict=strict, recover=recover))
        for row in stream:
            rows.append(row)
    except Exception as exc:  # noqa: BLE001 - the test inspects exactly what came out
        return rows, exc
    return rows, None


def _assert_the_parsers_own_exception(exc: Exception | None, expected: type[Exception]) -> None:
    """Exactly the exception ``json.loads`` raised: not ``JsonlLineError`` or ``JsonFileError``, which are ``ValueError``s too."""
    assert exc is not None, "the read went on past the line"
    assert type(exc) is expected, f"got {type(exc).__name__}: {exc}"
    assert exc.__cause__ is None


@_PINNED
@pytest.mark.parametrize(("build", "expected"), _UNLOADABLE)
@pytest.mark.parametrize("reader_name", _ABORTING_READERS)
def test_a_line_the_json_parser_cannot_load_stops_the_read_with_its_own_exception(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, reader_name: str, build, expected: type[Exception]
) -> None:
    with caplog.at_level(logging.WARNING):
        rows, exc = _run(reader_name, tmp_path, GOOD + build() + GOOD, strict=False)
    _assert_the_parsers_own_exception(exc, expected)
    assert rows == ([] if reader_name in _RETURNS_A_LIST else [{"a": 1}])  # the line after it was never reached
    assert _messages(caplog) == []


@_PINNED
@pytest.mark.parametrize(("build", "expected"), _UNLOADABLE)
@pytest.mark.parametrize("reader_name", _HAVE_STRICT)
def test_strict_mode_gives_the_same_bare_exception_for_such_a_line(
    tmp_path: Path, reader_name: str, build, expected: type[Exception]
) -> None:
    rows, exc = _run(reader_name, tmp_path, GOOD + build(), strict=True)
    _assert_the_parsers_own_exception(exc, expected)
    assert rows == ([] if reader_name in _RETURNS_A_LIST else [{"a": 1}])


@_PINNED
@pytest.mark.parametrize(("build", "expected"), _UNLOADABLE)
@pytest.mark.parametrize("strict", [False, True])
def test_the_line_parser_itself_stops_on_such_a_line(build, expected: type[Exception], strict: bool) -> None:
    stream = io.StringIO(GOOD.decode() + build().decode())
    with pytest.raises(expected) as exc:
        list(_parse_lines(stream, Path("x.jsonl"), strict=strict))
    assert type(exc.value) is expected and not isinstance(exc.value, JsonlLineError)


@_PINNED
@pytest.mark.parametrize(("build", "expected"), _UNLOADABLE)
def test_iter_jsonl_raw_skips_such_a_line_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, build, expected: type[Exception]
) -> None:
    path = _file(tmp_path, GOOD, build(), GOOD)
    with caplog.at_level(logging.WARNING):
        assert [rec for _text, rec in jio.iter_jsonl_raw(path)] == [{"a": 1}, {"a": 1}]
    assert len(_messages(caplog)) == 1 and "malformed JSON" in _messages(caplog)[0]
    seen = [rec for _text, rec in jio.iter_jsonl_raw(path, bad_lines=True)]
    assert seen == [{"a": 1}, None, {"a": 1}]


@_PINNED
@pytest.mark.parametrize(("build", "expected"), _UNLOADABLE)
def test_iter_jsonl_raw_strict_raises_a_line_file_error_for_such_a_line(tmp_path: Path, build, expected: type[Exception]) -> None:
    path = _file(tmp_path, GOOD, build())
    with pytest.raises(jio.JsonFileError) as exc:
        list(jio.iter_jsonl_raw(path, strict=True))
    assert (exc.value.kind, exc.value.line) == ("line", 2)
