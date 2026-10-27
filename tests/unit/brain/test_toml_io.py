"""Contract of ``brain/utils/toml_io.py`` (#529, criteria C51 and the C7 golden-file half).

The helper is the only place the product reads or writes TOML. These tests pin: the strict and the never-raising readers,
the line and column of a parse error (installed Python and a literal 3.12-style message), the BOM, the value writer's
round trip, the atomic byte writer (fsync before replace, no newline translation, temp file removed on failure), the
stat-signature cache, and ``TomlDocument`` (in-place single-line edits with every other byte untouched).
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from brain.utils import toml_io
from brain.utils.toml_io import TomlDocument, TomlFileError

POSIX = os.name == "posix"

# ---------------------------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------------------------


def test_read_toml_returns_the_parsed_dict(tmp_path: Path) -> None:
    p = tmp_path / "a.toml"
    p.write_text('x = 1\n[t]\ny = "z"\n', encoding="utf-8")
    assert toml_io.read_toml(p) == {"x": 1, "t": {"y": "z"}}


def test_read_toml_missing_kind(tmp_path: Path) -> None:
    with pytest.raises(TomlFileError) as err:
        toml_io.read_toml(tmp_path / "nope.toml")
    assert err.value.kind == "missing"
    assert isinstance(err.value, FileNotFoundError | OSError)


def test_read_toml_io_kind(tmp_path: Path) -> None:
    with pytest.raises(TomlFileError) as err:
        toml_io.read_toml(tmp_path)  # a directory: an I/O fault, not a content one
    assert err.value.kind == "io"


def test_read_toml_damaged_carries_line_column_and_reason(tmp_path: Path) -> None:
    p = tmp_path / "a.toml"
    p.write_text("a = 1\nb = 2\nc = \n", encoding="utf-8")
    with pytest.raises(TomlFileError) as err:
        toml_io.read_toml(p)
    e = err.value
    assert e.kind == "damaged" and e.path == p
    assert e.line == 3 and isinstance(e.column, int)
    assert e.reason and "line" not in e.reason  # the position is its own field, not repeated in the reason
    assert isinstance(e, OSError) and isinstance(e, ValueError)
    assert "line 3" in str(e) and str(p) in str(e)


def test_a_duplicate_key_is_damaged_with_its_line(tmp_path: Path) -> None:
    p = tmp_path / "a.toml"
    p.write_text('k = 1\n\nk = 2\n', encoding="utf-8")
    with pytest.raises(TomlFileError) as err:
        toml_io.read_toml(p)
    assert err.value.kind == "damaged" and err.value.line == 3


def test_error_position_on_a_python_312_style_message() -> None:
    """Python 3.12 and 3.13 give the position only in the message tail; 3.14 also has lineno/colno. Both are read."""
    line, col, reason = toml_io._split_position("Invalid statement (at line 7, column 12)")
    assert (line, col, reason) == (7, 12, "Invalid statement")


def test_error_position_from_an_exception_without_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    class OldParserError(Exception):
        pass

    exc = OldParserError("Expected '=' after a key in a key/value pair (at line 4, column 9)")
    assert toml_io._position_of(exc, "a = 1\n" * 5) == (4, 9, "Expected '=' after a key in a key/value pair")


def test_error_position_for_the_end_of_the_document() -> None:
    text = "a = 1\nb\n"
    with pytest.raises(TomlFileError) as err:
        toml_io.parse_toml(text)
    assert err.value.line is not None and err.value.line >= 2


def test_the_installed_python_gives_the_position(tmp_path: Path) -> None:
    with pytest.raises(TomlFileError) as err:
        toml_io.parse_toml("ok = 1\n\nbad = [\n")
    assert err.value.line is not None and err.value.column is not None


def test_a_leading_bom_is_stripped_and_the_file_reads(tmp_path: Path) -> None:
    p = tmp_path / "a.toml"
    p.write_bytes(b"\xef\xbb\xbfx = 1\r\ny = 2\r\n")
    assert toml_io.read_toml(p) == {"x": 1, "y": 2}
    res = toml_io.read_toml_result(p)
    assert res.outcome == "ok" and res.data == {"x": 1, "y": 2}


def test_a_non_utf8_file_is_damaged_with_the_line_of_the_first_bad_byte(tmp_path: Path) -> None:
    p = tmp_path / "a.toml"
    p.write_bytes(b'a = "ok"\nb = "caf\xe9"\n')  # cp1252 e-acute on line 2
    with pytest.raises(TomlFileError) as err:
        toml_io.read_toml(p)
    assert err.value.kind == "damaged" and err.value.line == 2
    assert "UTF-8" in err.value.reason


def test_read_toml_result_never_raises_and_reports_each_outcome(tmp_path: Path) -> None:
    ok = tmp_path / "ok.toml"
    ok.write_text("a = 1\n", encoding="utf-8")
    empty = tmp_path / "empty.toml"
    empty.write_bytes(b"")
    bad = tmp_path / "bad.toml"
    bad.write_text("a = \n", encoding="utf-8")
    assert toml_io.read_toml_result(ok).outcome == "ok"
    assert toml_io.read_toml_result(tmp_path / "missing.toml").outcome == "missing"
    r = toml_io.read_toml_result(empty)
    assert r.outcome == "empty" and r.data == {}
    r = toml_io.read_toml_result(bad)
    assert r.outcome == "damaged" and r.line == 1 and r.kind == "damaged" and r.reason
    assert toml_io.read_toml_result(tmp_path).outcome == "io"


def test_a_comment_only_file_is_ok_and_empty_data(tmp_path: Path) -> None:
    p = tmp_path / "a.toml"
    p.write_text("# nothing here\n", encoding="utf-8")
    r = toml_io.read_toml_result(p)
    assert r.outcome == "ok" and r.data == {}


def test_parse_toml_is_the_string_codec() -> None:
    assert toml_io.parse_toml("a = 1") == {"a": 1}
    with pytest.raises(TomlFileError):
        toml_io.parse_toml("a = ")


# ---------------------------------------------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------------------------------------------


def test_flatten_reads_nested_tables_and_dotted_keys_as_dotted_keys() -> None:
    data = toml_io.parse_toml('[model]\ncompaction = "x"\n[a.b]\nc = 1\nd.e = 2\n')
    assert toml_io.flatten_toml(data) == {"model.compaction": "x", "a.b.c": 1, "a.b.d.e": 2}


def test_flatten_prefers_the_shallowest_spelling() -> None:
    data = toml_io.parse_toml('"a.b" = 1\n[a]\nb = 2\n')
    assert toml_io.flatten_toml(data) == {"a.b": 1}


def test_flatten_keeps_lists_whole() -> None:
    assert toml_io.flatten_toml({"a": {"b": [1, 2]}}) == {"a.b": [1, 2]}


# ---------------------------------------------------------------------------------------------------------------
# dumps_value
# ---------------------------------------------------------------------------------------------------------------

ROUND_TRIP = [
    "plain",
    "",
    'a "quote" inside',
    "back\\slash",
    "hash # inside",
    "equals = sign",
    "new\nline",
    "tab\there",
    "carriage\rreturn",
    "del\x7fchar",
    "nul\x00char",
    "ctrl\x01\x1fchars",
    "café ünï 日本語 \U0001f600",
    0,
    -7,
    2**70,
    -(2**70),
    600.0,
    0.3,
    1e-05,
    1e22,
    -0.0,
    float("inf"),
    float("-inf"),
    True,
    False,
    [],
    [1, 2, 3],
    ["a", "b # c"],
    [1.5, "x", True],
    [[1, 2], ["a"]],
]


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b and math.copysign(1, a) == math.copysign(1, b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b


@pytest.mark.parametrize("value", ROUND_TRIP, ids=lambda v: repr(v)[:30])
def test_dumps_value_round_trips(value: Any) -> None:
    text = toml_io.dumps_value(value)
    assert "\n" not in text and "\r" not in text
    back = toml_io.parse_toml(f"x = {text}")["x"]
    assert _same(back, value)


def test_dumps_value_nan_round_trips_nan_aware() -> None:
    back = toml_io.parse_toml(f"x = {toml_io.dumps_value(float('nan'))}")["x"]
    assert isinstance(back, float) and math.isnan(back)


def test_dumps_value_tuples_render_as_arrays() -> None:
    assert toml_io.parse_toml(f"x = {toml_io.dumps_value(('a', 'b'))}")["x"] == ["a", "b"]


def test_dumps_value_bool_is_tested_before_int() -> None:
    assert toml_io.dumps_value(True) == "true"
    assert toml_io.dumps_value(1) == "1"


def test_dumps_value_floats_keep_their_float_look() -> None:
    assert toml_io.dumps_value(600.0) == "600.0"
    assert toml_io.dumps_value(-0.0) == "-0.0"


def test_dumps_value_refuses_what_it_cannot_render() -> None:
    for bad in ("lone\ud800surrogate", None, {"a": 1}, object(), b"bytes", ["ok", None]):
        with pytest.raises(TomlFileError):
            toml_io.dumps_value(bad)


# ---------------------------------------------------------------------------------------------------------------
# write_toml_text
# ---------------------------------------------------------------------------------------------------------------


def test_write_toml_text_writes_bytes_untranslated(tmp_path: Path) -> None:
    p = tmp_path / "a.toml"
    payload = b"\xef\xbb\xbfa = 1\r\nb = 2\n"
    toml_io.write_toml_text(p, payload)
    assert p.read_bytes() == payload


def test_write_toml_text_creates_the_parent_directory(tmp_path: Path) -> None:
    p = tmp_path / "deep" / "er" / "a.toml"
    toml_io.write_toml_text(p, b"a = 1\n")
    assert p.read_bytes() == b"a = 1\n"


def test_write_toml_text_fsyncs_before_the_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.utils import atomic_write

    order: list[str] = []
    real_fsync, real_replace = os.fsync, atomic_write._replace_with_retry
    monkeypatch.setattr(toml_io.os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(atomic_write, "_replace_with_retry", lambda s, d: (order.append("replace"), real_replace(s, d))[1])
    toml_io.write_toml_text(tmp_path / "a.toml", b"a = 1\n")
    assert order[:2] == ["fsync", "replace"]  # the data is on disk before the replace; the folder fsync comes after


def test_write_toml_text_can_skip_the_fsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(toml_io.os, "fsync", lambda fd: calls.append(fd))
    toml_io.write_toml_text(tmp_path / "a.toml", b"a = 1\n", fsync=False)
    assert calls == []


def test_the_write_is_the_shared_atomic_writer() -> None:
    from brain.utils import atomic_write

    assert toml_io.save_bytes_atomic is atomic_write.save_bytes_atomic


def test_write_toml_text_removes_the_temp_file_when_the_replace_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "a.toml"
    p.write_bytes(b"old = 1\n")

    def boom(src: Path, dst: Path) -> None:
        raise PermissionError("held open")

    from brain.utils import atomic_write

    monkeypatch.setattr(atomic_write, "_replace_with_retry", boom)
    with pytest.raises(PermissionError):
        toml_io.write_toml_text(p, b"new = 2\n")
    assert p.read_bytes() == b"old = 1\n"
    assert [c.name for c in tmp_path.iterdir()] == ["a.toml"]


def test_write_toml_text_writes_through_a_symlink_and_keeps_the_link(tmp_path: Path) -> None:
    """A dotfile manager may link the settings file to a file elsewhere: the write lands in the target and the link stays
    (rev-4 review bycatch; an atomic replace onto the link used to turn it into a regular file)."""
    target_dir = tmp_path / "dotfiles"
    target_dir.mkdir()
    target = target_dir / "real.toml"
    target.write_bytes(b"old = 1\n")
    link = tmp_path / "a.toml"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")
    toml_io.write_toml_text(link, b"new = 2\n")
    assert link.is_symlink() and link.resolve() == target.resolve()
    assert target.read_bytes() == b"new = 2\n"
    assert toml_io.read_toml(link) == {"new": 2}
    assert sorted(c.name for c in tmp_path.iterdir()) == ["a.toml", "dotfiles"]
    assert [c.name for c in target_dir.iterdir()] == ["real.toml"]  # no temp file left beside the target


def test_write_toml_text_replaces_a_dangling_link_as_before(tmp_path: Path) -> None:
    link = tmp_path / "a.toml"
    try:
        link.symlink_to(tmp_path / "nowhere" / "x.toml")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")
    toml_io.write_toml_text(link, b"new = 2\n")
    assert not link.is_symlink() and link.read_bytes() == b"new = 2\n"
    assert not (tmp_path / "nowhere").exists()


def test_write_toml_text_is_atomic_for_a_polling_reader(tmp_path: Path) -> None:
    p = tmp_path / "a.toml"
    toml_io.write_toml_text(p, b'v = "0"\n')
    stop = threading.Event()
    bad: list[str] = []

    def reader() -> None:
        while not stop.is_set():
            r = toml_io.read_toml_result(p)
            if r.outcome != "ok" or "v" not in r.data:
                bad.append(r.outcome)

    t = threading.Thread(target=reader)
    if POSIX:  # a reader that is always inside an open() blocks the replace on Windows by design (the writer's bounded retry)
        t.start()
    try:
        for i in range(200):
            toml_io.write_toml_text(p, f'v = "{i}"\n'.encode(), fsync=False)
    finally:
        stop.set()
        if POSIX:
            t.join()
    assert bad == []


# ---------------------------------------------------------------------------------------------------------------
# read_toml_cached
# ---------------------------------------------------------------------------------------------------------------


def test_read_toml_cached_parses_once_and_rereads_a_changed_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "a.toml"
    p.write_text("a = 1\n", encoding="utf-8")
    calls: list[int] = []
    real = toml_io._parse_text
    monkeypatch.setattr(toml_io, "_parse_text", lambda text: (calls.append(1), real(text))[1])
    toml_io._reset_cache_for_tests()
    assert toml_io.read_toml_cached(p) == {"a": 1}
    assert toml_io.read_toml_cached(p) == {"a": 1}
    assert len(calls) == 1  # the unchanged file is not parsed again
    p.write_text("a = 22\n", encoding="utf-8")  # a different size
    assert toml_io.read_toml_cached(p) == {"a": 22}
    assert len(calls) == 2
    # same size, different content, forced identical mtime: the inode differs after a replace-style write
    st = p.stat()
    other = tmp_path / "b.toml"
    other.write_text("a = 33\n", encoding="utf-8")
    os.utime(other, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.replace(other, p)
    assert toml_io.read_toml_cached(p) == {"a": 33}


def test_read_toml_cached_raises_for_a_missing_or_damaged_file(tmp_path: Path) -> None:
    toml_io._reset_cache_for_tests()
    with pytest.raises(TomlFileError) as err:
        toml_io.read_toml_cached(tmp_path / "x.toml")
    assert err.value.kind == "missing"
    bad = tmp_path / "bad.toml"
    bad.write_text("a = \n", encoding="utf-8")
    with pytest.raises(TomlFileError) as err2:
        toml_io.read_toml_cached(bad)
    assert err2.value.kind == "damaged"


def test_read_toml_cached_is_thread_safe(tmp_path: Path) -> None:
    p = tmp_path / "a.toml"
    p.write_text("a = 1\n", encoding="utf-8")
    toml_io._reset_cache_for_tests()
    out: list[dict] = []
    errs: list[BaseException] = []

    def go() -> None:
        try:
            for _ in range(50):
                out.append(toml_io.read_toml_cached(p))
        except BaseException as exc:  # noqa: BLE001
            errs.append(exc)

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errs == [] and all(d == {"a": 1} for d in out)


# ---------------------------------------------------------------------------------------------------------------
# TomlDocument: golden-file edits. One module on purpose (the highest-risk component).
# ---------------------------------------------------------------------------------------------------------------


def _doc(text: str | bytes) -> TomlDocument:
    return TomlDocument.from_bytes(text.encode() if isinstance(text, str) else text)


HAND_EDITED = (
    "# My own header, kept.\n"
    "\n"
    "# The live chat turn.\n"
    '# default: "claude-sonnet-5"\n'
    '"model.interactive_chat" = "claude-sonnet-5"  # my note stays\n'
    "\n"
    "# order was shuffled by me\n"
    "# default: 600.0\n"
    '"chat.idle_lull_seconds" = 600.0\n'
    "\n"
    '"some.unknown.key" = "keep me"\n'
    "unquoted.dotted = 5   # also kept\n"
    "\n"
    'multi = [\n'
    "  1,\n"
    "  2,\n"
    "]\n"
)


def test_set_value_changes_only_the_value_text() -> None:
    doc = _doc(HAND_EDITED)
    assert doc.set_value("model.interactive_chat", "claude-opus-5") is True
    expect = HAND_EDITED.replace('= "claude-sonnet-5"  # my note stays', '= "claude-opus-5"  # my note stays')
    assert doc.text() == expect
    assert doc.to_bytes() == expect.encode()


def test_set_value_keeps_every_other_line_byte_identical() -> None:
    doc = _doc(HAND_EDITED)
    doc.set_value("chat.idle_lull_seconds", 30.0)
    before, after = HAND_EDITED.splitlines(), doc.text().splitlines()
    assert len(before) == len(after)
    diff = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert diff == [8] and after[8] == '"chat.idle_lull_seconds" = 30.0'


def test_set_value_equal_text_reports_no_change() -> None:
    doc = _doc(HAND_EDITED)
    assert doc.set_value("chat.idle_lull_seconds", 600.0) is False
    assert doc.text() == HAND_EDITED


@pytest.mark.parametrize(
    ("line", "expect"),
    [
        ('"k" = "a # b"  # trailing', '"k" = "NEW"  # trailing'),
        ('"k" = "say \\"hi\\" # x\\\\"  # c', '"k" = "NEW"  # c'),
        ("\"k\" = 'lit # eral'   # c", '"k" = "NEW"   # c'),
        ('"k" = [1, 2, "x # y"]  # n', '"k" = "NEW"  # n'),
        ('"k" = [ [1, 2], [3] ]# n', '"k" = "NEW"# n'),
        ('"k"="a"', '"k"="NEW"'),
        ('  "k"   =   "a"   ', '  "k"   =   "NEW"   '),
        ('"k" = "a=b"  # x = y', '"k" = "NEW"  # x = y'),
        ("\"k\" = 'a\\b' # x", '"k" = "NEW" # x'),
        ('"k" = 5#c', '"k" = "NEW"#c'),
    ],
)
def test_value_comment_boundary_is_found_by_tokenising(line: str, expect: str) -> None:
    doc = _doc(f"{line}\n")
    assert doc.set_value("k", "NEW") is True
    assert doc.text() == f"{expect}\n"


@pytest.mark.parametrize(
    "line",
    [
        "'a.b' = 1",
        "a.b = 1",
        '"a"."b" = 1',
        '"a" . "b" = 1',
        '"a.b"=1',
        '  "a.b"  =  1  ',
        "a . b = 1",
    ],
)
def test_every_spelling_of_a_key_is_one_key_and_is_edited_in_place(line: str) -> None:
    doc = _doc(f"{line}\n")
    assert doc.state("a.b") == "single"
    assert doc.values() == {"a.b": 1}
    doc.set_value("a.b", 2)
    assert doc.text() == f"{line}\n".replace("1", "2")
    assert doc.values() == {"a.b": 2}


def test_an_entry_inside_a_user_table_is_found_and_edited_in_place() -> None:
    text = "# mine\n[a]\nb = 1  # c\n\n[x]\ny = 2\n"
    doc = _doc(text)
    assert doc.state("a.b") == "single" and doc.state("x.y") == "single"
    doc.set_value("a.b", 9)
    assert doc.text() == text.replace("b = 1", "b = 9")


def test_a_key_in_two_spellings_is_ambiguous_and_set_value_refuses() -> None:
    text = '"a.b" = 1\na.b = 2\n'
    doc = _doc(text)
    assert doc.state("a.b") == "ambiguous"
    assert doc.values() == {"a.b": 1}  # the flat quoted spelling wins for the reader
    with pytest.raises(TomlFileError) as err:
        doc.set_value("a.b", 3)
    assert "a.b" in str(err.value)
    assert doc.text() == text


def test_a_multi_line_value_is_read_but_not_edited() -> None:
    text = 'k = [\n  1,\n  2,\n]\nother = 1\n'
    doc = _doc(text)
    assert doc.values()["k"] == [1, 2]
    assert doc.state("k") == "unlocatable"
    with pytest.raises(TomlFileError) as err:
        doc.set_value("k", [3])
    assert "k" in str(err.value)
    assert doc.text() == text
    assert doc.state("other") == "single"
    doc.set_value("other", 2)
    assert doc.text() == text.replace("other = 1", "other = 2")


def test_an_inline_table_is_read_but_not_edited() -> None:
    doc = _doc("a = {b = 1}\n")
    assert doc.values() == {"a.b": 1}
    assert doc.state("a.b") == "unlocatable"


def test_a_text_line_that_looks_like_an_entry_inside_a_multi_line_string_is_never_edited() -> None:
    text = 'blurb = """\nk = 1\n"""\nk2 = 5\n'
    doc = _doc(text)
    assert "k" not in doc.values()
    assert doc.state("k") in ("absent", "unlocatable")
    doc.set_value("k2", 6)
    assert doc.text() == text.replace("k2 = 5", "k2 = 6")


def test_set_value_on_an_absent_key_raises() -> None:
    doc = _doc("a = 1\n")
    with pytest.raises(TomlFileError):
        doc.set_value("nope", 1)


def test_bom_and_crlf_are_kept_on_untouched_lines_and_used_for_added_ones() -> None:
    raw = b"\xef\xbb\xbf# head\r\n\r\n# d\r\n# default: 1\r\n\"a\" = 1 # x\r\nother = 2\r\n"
    doc = _doc(raw)
    assert doc.values() == {"a": 1, "other": 2}
    doc.set_value("a", 7)
    expect = raw.replace(b'"a" = 1 # x', b'"a" = 7 # x')
    assert doc.to_bytes() == expect
    doc.insert("new.key", "v", ["# adds one", "# default: \"v\""])
    out = doc.to_bytes()
    assert out.startswith(b"\xef\xbb\xbf")
    assert out.startswith(expect)  # nothing before the insertion changed
    added = out[len(expect):]
    assert b"\n" not in added.replace(b"\r\n", b"")  # every added line break is CRLF
    assert added.endswith(b"\r\n")
    assert toml_io.parse_toml(doc.text())["new.key"] == "v"


def test_an_lf_file_gets_lf_lines_and_a_missing_final_newline_is_supplied() -> None:
    doc = _doc("a = 1")  # no trailing newline
    doc.insert("b", 2, ["# two"])
    assert doc.text() == 'a = 1\n\n# two\n"b" = 2\n'


def test_insert_goes_before_the_first_table_header_not_inside_the_users_table() -> None:
    text = "a = 1\n\n[model]\ncompaction = \"x\"\n"
    doc = _doc(text)
    doc.insert("reranker.precision", "high", ["# d", '# default: "high"'])
    parsed = toml_io.parse_toml(doc.text())
    assert parsed["reranker.precision"] == "high"  # top level, not model."reranker.precision"
    assert parsed["model"] == {"compaction": "x"}
    assert doc.text().index("reranker.precision") < doc.text().index("[model]")


def test_insert_leaves_a_comment_directly_above_a_header_with_that_header() -> None:
    text = "a = 1\n\n# about the model table\n[model]\nx = 1\n"
    doc = _doc(text)
    doc.insert("new", 5, [])
    out = doc.text()
    assert out.index('"new"') < out.index("# about the model table") < out.index("[model]")
    assert "# about the model table\n[model]" in out


def test_several_inserts_keep_their_order() -> None:
    doc = _doc("[t]\nx = 1\n")
    doc.insert("a", 1, [])
    doc.insert("b", 2, [])
    out = doc.text()
    assert out.index('"a"') < out.index('"b"') < out.index("[t]")
    assert doc.values() == {"a": 1, "b": 2, "t.x": 1}


def test_insert_at_the_end_of_a_file_with_no_table() -> None:
    doc = _doc("a = 1\n")
    doc.insert("b", 2, ["# d"])
    assert doc.text() == 'a = 1\n\n# d\n"b" = 2\n'


def test_insert_into_an_empty_document() -> None:
    doc = TomlDocument.from_bytes(b"")
    doc.insert("a.b", "x", ["# first"])
    assert doc.text() == '# first\n"a.b" = "x"\n'
    assert doc.values() == {"a.b": "x"}


def test_insert_refuses_a_key_already_present() -> None:
    doc = _doc("a = 1\n")
    with pytest.raises(TomlFileError):
        doc.insert("a", 2, [])


def test_comment_above_is_the_contiguous_run_and_a_blank_line_breaks_it() -> None:
    doc = _doc("# far\n\n# d1\n# default: 1\n\"a\" = 1\n\n# lonely\n\n\"b\" = 2\n")
    assert doc.comment_above("a") == ["# d1", "# default: 1"]
    assert doc.comment_above("b") == []


def test_set_comment_above_replaces_the_run_and_can_create_one() -> None:
    doc = _doc("# d1\n# default: 1\n\"a\" = 1\n\"b\" = 2\n")
    assert doc.set_comment_above("a", ["# d1", "# default: 2"]) is True
    assert doc.set_comment_above("a", ["# d1", "# default: 2"]) is False  # same text: no change
    assert doc.set_comment_above("b", ["# default: 5"]) is True
    assert doc.text() == '# d1\n# default: 2\n"a" = 1\n# default: 5\n"b" = 2\n'


def test_set_comment_above_does_not_touch_a_run_separated_by_a_blank_line() -> None:
    doc = _doc("# orphan: default: 1\n\n\"a\" = 1\n")
    doc.set_comment_above("a", ["# default: 2"])
    assert doc.text() == '# orphan: default: 1\n\n# default: 2\n"a" = 1\n'


def test_a_hash_inside_a_value_is_never_taken_for_a_comment_when_reading_the_run() -> None:
    doc = _doc('# default: "x # y"\n"a" = "x # y"\n')
    assert doc.comment_above("a") == ['# default: "x # y"']


def test_verify_compares_the_parsed_text_nan_aware() -> None:
    doc = _doc("a = nan\nb = 1\n")
    assert doc.verify({"a": float("nan"), "b": 1}) is True
    assert doc.verify({"a": float("nan"), "b": 2}) is False
    assert doc.verify({"a": float("nan")}) is False


def test_the_document_reports_a_damaged_source() -> None:
    with pytest.raises(TomlFileError) as err:
        TomlDocument.from_bytes(b"a = \n")
    assert err.value.kind == "damaged" and err.value.line == 1


def test_a_non_utf8_source_is_damaged() -> None:
    with pytest.raises(TomlFileError) as err:
        TomlDocument.from_bytes(b'a = "\xe9"\n')
    assert err.value.kind == "damaged"


def test_the_edit_is_a_pure_function_of_the_text() -> None:
    a, b = _doc(HAND_EDITED), _doc(HAND_EDITED)
    a.set_value("model.interactive_chat", "x")
    b.set_value("model.interactive_chat", "x")
    assert a.to_bytes() == b.to_bytes()
