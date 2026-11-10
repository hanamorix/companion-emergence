"""The line parsers behind ``iter_jsonl``, ``parse_jsonl``, ``iter_jsonl_raw`` and ``iter_jsonl_skipping_corrupt`` keep their
warning texts, their strict-mode reasons and their exception chains when they share one line classifier (#643).

Most tests here pin what the parsers did before the change (they pass on the old code by design, so each carries the
``no_base_failure`` mark). The exception is a known defect, kept by the owner's ruling and filed as #646: a line the JSON parser
cannot load (an integer of more than the interpreter's digit limit, nesting too deep) stops ``iter_jsonl``, ``read_jsonl``,
``parse_jsonl``, ``read_jsonl_tail`` and the skipping reader, in strict mode too, with ``UnloadableJsonlLineError`` (a
``ValueError`` with a stable ``code``, the path and the line; the parser's own exception is its ``__cause__``). Those tests are new
behaviour and carry no mark. ``iter_jsonl_raw`` skips such a line (strict: ``JsonFileError(kind="line")``). A fix for #646 that
makes the first group skip it must change those tests on purpose.
"""

from __future__ import annotations

import copy
import gzip
import io
import logging
import pickle
import sys
from pathlib import Path

import pytest

from brain.health import json_io as jio
from brain.health import jsonl_reader
from brain.health.jsonl_reader import JsonlLineError, _parse_lines, iter_jsonl_skipping_corrupt

_PINNED = pytest.mark.no_base_failure(reason="pins what the parsers already do on origin/main: their warning texts, strict reasons and the skip of iter_jsonl_raw for a line the JSON parser cannot load")

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


# --- a line the JSON parser cannot load: the read stops with UnloadableJsonlLineError (#643, #646) ----------------------------

_NEEDS_DIGIT_LIMIT = pytest.mark.skipif(sys.get_int_max_str_digits() == 0, reason="the interpreter's integer digit limit is off")
_NEEDS_DEPTH_LIMIT = pytest.mark.skipif(sys.getrecursionlimit() > 100000, reason="needs a depth the parser rejects")


def _huge_int_line() -> bytes:
    return b'{"n": ' + b"9" * (sys.get_int_max_str_digits() + 1) + b"}\n"


def _deep_line() -> bytes:
    return b"[" * 200000 + b"]" * 200000 + b"\n"


# (line builder, the exact exception the JSON parser gives for it, which becomes the new error's __cause__)
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


def _layout(kind: str, bad: bytes) -> tuple[bytes, int, int]:
    """(file content, the bad line's 1-based line in the file, its position in the plain tail's last-3-lines window)."""
    if kind == "plain":
        return GOOD + bad + GOOD, 2, 2
    # a blank first line: the file line is 3, but the plain tail drops blank lines, so its window line is 2
    return b"\n" + GOOD + bad, 3, 2


_LAYOUTS = ["plain", "blank-first"]


def _run(reader_name: str, tmp_path: Path, content: bytes, *, strict: bool) -> tuple[list[dict], Exception | None, Path]:
    """Run one reader over ``content``: (the rows it gave before it stopped, the exception that stopped it or None, the path it read)."""
    path = _file(tmp_path, content)
    if reader_name.endswith("_gz"):
        path = tmp_path / "x.jsonl.gz"
        path.write_bytes(gzip.compress(content))
    if reader_name == "parse_jsonl":
        path = Path("<memory>")
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
        return rows, exc, path
    return rows, None, path


def _assert_the_unloadable_line_error(
    exc: BaseException | None, cause: type[Exception], line: int | None, path: Path, window_line: int | None = None
) -> None:
    """Exactly ``UnloadableJsonlLineError``: not a bare parser exception, not ``JsonlLineError`` or ``JsonFileError`` (``ValueError``s too).

    ``line`` is the file line, or None for the plain tail reader, which parses a window of the last lines: then ``reason`` says
    ``window_line``, the position in that window, and the message does not claim a line."""
    assert exc is not None, "the read went on past the line"
    assert type(exc).__name__ == "UnloadableJsonlLineError", f"got {type(exc).__name__}: {exc}"
    assert type(exc) is jsonl_reader.UnloadableJsonlLineError
    assert isinstance(exc, ValueError)
    assert not isinstance(exc, (JsonlLineError, jio.JsonFileError))
    assert exc.code == "jsonl_unloadable_line"
    assert exc.line == line
    assert exc.path == path
    if window_line is not None:
        assert line is None
        assert f"line {window_line} of the window of the last lines read, not a file line" in exc.reason
        assert "None" not in str(exc) and str(exc).startswith(f"{path}: ") and exc.reason in str(exc)
    else:
        assert "window" not in exc.reason
    assert type(exc.__cause__) is cause  # the parser's own exception stays reachable


def test_the_error_code_is_a_stable_class_attribute() -> None:
    assert jsonl_reader.UnloadableJsonlLineError.code == "jsonl_unloadable_line"
    assert issubclass(jsonl_reader.UnloadableJsonlLineError, ValueError)
    assert not issubclass(jsonl_reader.UnloadableJsonlLineError, JsonlLineError)


@pytest.mark.parametrize("clone", [copy.copy, lambda e: pickle.loads(pickle.dumps(e))], ids=["copy", "pickle"])
def test_the_error_survives_copy_and_pickle(clone) -> None:
    original = jsonl_reader.UnloadableJsonlLineError(7, "malformed JSON (x)", Path("a") / "b.jsonl")
    again = clone(original)
    assert type(again) is type(original)
    assert (again.line, again.reason, again.path, again.code, str(again)) == (
        original.line,
        original.reason,
        original.path,
        original.code,
        str(original),
    )


@pytest.mark.parametrize("clone", [copy.copy, lambda e: pickle.loads(pickle.dumps(e))], ids=["copy", "pickle"])
def test_the_error_without_a_line_survives_copy_and_pickle_and_its_message_claims_no_line(clone) -> None:
    original = jsonl_reader.UnloadableJsonlLineError(None, "malformed JSON (x) (line 2 of the window)", Path("a") / "b.jsonl")
    again = clone(original)
    assert type(again) is type(original) and again.line is None
    assert (again.reason, again.path, again.code, str(again)) == (original.reason, original.path, original.code, str(original))
    assert str(original) == f"{Path('a') / 'b.jsonl'}: malformed JSON (x) (line 2 of the window)"


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize(("build", "cause"), _UNLOADABLE)
@pytest.mark.parametrize("reader_name", _ABORTING_READERS)
def test_a_line_the_json_parser_cannot_load_stops_the_read_with_a_named_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, reader_name: str, build, cause: type[Exception], layout: str
) -> None:
    content, file_line, window_line = _layout(layout, build())
    with caplog.at_level(logging.WARNING):
        rows, exc, path = _run(reader_name, tmp_path, content, strict=False)
    if reader_name == "read_jsonl_tail":  # the plain tail parses a window of the file: no file line, the window position is in the reason
        _assert_the_unloadable_line_error(exc, cause, None, path, window_line)
    else:
        _assert_the_unloadable_line_error(exc, cause, file_line, path)
    assert rows == ([] if reader_name in _RETURNS_A_LIST else [{"a": 1}])
    assert _messages(caplog) == []


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize(("build", "cause"), _UNLOADABLE)
@pytest.mark.parametrize("reader_name", _HAVE_STRICT)
def test_strict_mode_gives_the_same_named_error_for_such_a_line(
    tmp_path: Path, reader_name: str, build, cause: type[Exception], layout: str
) -> None:
    content, file_line, window_line = _layout(layout, build())
    rows, exc, path = _run(reader_name, tmp_path, content, strict=True)
    if reader_name == "read_jsonl_tail":
        _assert_the_unloadable_line_error(exc, cause, None, path, window_line)
    else:
        _assert_the_unloadable_line_error(exc, cause, file_line, path)
    assert rows == ([] if reader_name in _RETURNS_A_LIST else [{"a": 1}])


@pytest.mark.parametrize(("build", "cause"), _UNLOADABLE)
@pytest.mark.parametrize("strict", [False, True])
def test_the_line_parser_itself_stops_on_such_a_line(build, cause: type[Exception], strict: bool) -> None:
    stream = io.StringIO(GOOD.decode() + build().decode())
    path = Path("x.jsonl")
    with pytest.raises(ValueError) as exc:
        list(_parse_lines(stream, path, strict=strict))
    _assert_the_unloadable_line_error(exc.value, cause, 2, path)


@_NEEDS_DIGIT_LIMIT
def test_the_message_names_the_file_and_line_and_never_the_lines_content(tmp_path: Path) -> None:
    _rows, exc, path = _run("iter_jsonl", tmp_path, GOOD + _huge_int_line(), strict=False)
    text = str(exc)
    assert text.startswith(f"{path}: line 2: malformed JSON (")
    assert "9" * 20 not in text


@_PINNED
@pytest.mark.parametrize(("build", "_expected"), _UNLOADABLE)
def test_iter_jsonl_raw_skips_such_a_line_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, build, _expected: type[Exception]
) -> None:
    path = _file(tmp_path, GOOD, build(), GOOD)
    with caplog.at_level(logging.WARNING):
        assert [rec for _text, rec in jio.iter_jsonl_raw(path)] == [{"a": 1}, {"a": 1}]
    assert len(_messages(caplog)) == 1 and "malformed JSON" in _messages(caplog)[0]
    seen = [rec for _text, rec in jio.iter_jsonl_raw(path, bad_lines=True)]
    assert seen == [{"a": 1}, None, {"a": 1}]


@_PINNED
@pytest.mark.parametrize(("build", "_expected"), _UNLOADABLE)
def test_iter_jsonl_raw_strict_raises_a_line_file_error_for_such_a_line(tmp_path: Path, build, _expected: type[Exception]) -> None:
    path = _file(tmp_path, GOOD, build())
    with pytest.raises(jio.JsonFileError) as exc:
        list(jio.iter_jsonl_raw(path, strict=True))
    assert (exc.value.kind, exc.value.line) == ("line", 2)
