"""Phase-2 module additions to brain.health.json_io (criteria PC8, PC11, PC12 of 1.5-criteria-phase2.md).

Recovery.CONTENT, nonfinite=, iter_jsonl_raw. Nothing here touches a persona directory.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from brain.health import json_io as jio
from brain.health.json_io import JsonFileError, Recovery

# ---------------------------------------------------------------------------------------------
# Recovery.CONTENT: bad content takes the default, an I/O fault raises
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [b"", b"{not json", b"\xff\xfe\x00bad", b"[1, 2]", b"   \n"],
    ids=["empty", "malformed", "bad-utf8", "wrong-shape", "blank"],
)
def test_pc8_content_bad_content_returns_the_default(tmp_path: Path, payload: bytes) -> None:
    p = tmp_path / "state.json"
    p.write_bytes(payload)
    assert jio.read_json(p, default=lambda: {"d": 1}, recover=Recovery.CONTENT, expect=dict) == {"d": 1}
    assert p.read_bytes() == payload  # the file is left alone


def test_pc8_content_missing_file_returns_the_default(tmp_path: Path) -> None:
    assert jio.read_json(tmp_path / "nope.json", default=dict, recover=Recovery.CONTENT) == {}


def test_pc8_content_an_io_fault_raises_and_is_not_the_default(tmp_path: Path) -> None:
    d = tmp_path / "state.json"
    d.mkdir()  # a directory where the file should be: an I/O fault on every platform
    with pytest.raises(JsonFileError) as exc:
        jio.read_json(d, default=dict, recover=Recovery.CONTENT)
    assert exc.value.kind == "io"
    assert isinstance(exc.value, OSError)


def test_pc8_content_does_not_escape_a_wider_handler_at_the_site(tmp_path: Path) -> None:
    """A site that keeps `except OSError` still catches the I/O fault: CONTENT is not a bypass."""
    d = tmp_path / "state.json"
    d.mkdir()
    try:
        data = jio.read_json(d, default=lambda: "fallback", recover=Recovery.CONTENT)
    except OSError:
        data = "site handler"
    assert data == "site handler"


def test_pc8_content_needs_a_default(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        jio.read_json(tmp_path / "x.json", recover=Recovery.CONTENT)


def test_pc8_content_is_rejected_by_the_jsonl_readers(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        list(jio.iter_jsonl(tmp_path / "x.jsonl", recover=Recovery.CONTENT))


# ---------------------------------------------------------------------------------------------
# nonfinite="null"
# ---------------------------------------------------------------------------------------------


class _F(float):
    pass


def test_pc12_default_still_raises_before_touching_the_file(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    with pytest.raises(ValueError):
        jio.write_json(p, {"x": float("nan")})
    assert not p.exists()
    with pytest.raises(ValueError):
        jio.encode_jsonl({"x": float("inf")})


def test_pc12_null_maps_nested_nonfinite_values() -> None:
    rec = {
        "a": float("nan"),
        "b": [1.5, float("inf"), (float("-inf"), {"c": float("nan")})],
        "s": "NaN stays text",
        "f": _F("nan"),
        "ok": 2.0,
    }
    line = jio.encode_jsonl(rec, nonfinite="null")
    assert json.loads(line) == {
        "a": None,
        "b": [1.5, None, [None, {"c": None}]],
        "s": "NaN stays text",
        "f": None,
        "ok": 2.0,
    }
    assert line.endswith(b"\n") and line.isascii()


def test_pc12_null_covers_what_an_encode_default_hook_returns() -> None:
    class Thing:
        pass

    line = jio.encode_jsonl({"t": Thing()}, encode_default=lambda o: {"v": float("nan")}, nonfinite="null")
    assert json.loads(line) == {"t": {"v": None}}


def test_pc12_null_does_not_hide_an_unserialisable_object() -> None:
    with pytest.raises(TypeError):
        jio.encode_jsonl({"t": object()}, nonfinite="null")


def test_pc12_null_cycle_is_a_value_error() -> None:
    a: dict = {}
    a["self"] = a
    with pytest.raises(ValueError):
        jio.encode_jsonl(a, nonfinite="null")


def test_pc12_write_append_and_rewrite_take_the_parameter(tmp_path: Path) -> None:
    j = tmp_path / "a.json"
    assert jio.write_json(j, {"x": float("nan")}, nonfinite="null") is True
    assert json.loads(j.read_text()) == {"x": None}
    log = tmp_path / "a.jsonl"
    assert jio.append_jsonl(log, {"x": float("inf")}, nonfinite="null") > 0
    assert jio.write_jsonl(tmp_path / "b.jsonl", [{"x": float("nan")}], nonfinite="null") is True
    assert json.loads(log.read_text()) == {"x": None}
    assert json.loads((tmp_path / "b.jsonl").read_text()) == {"x": None}


# ---------------------------------------------------------------------------------------------
# iter_jsonl_raw
# ---------------------------------------------------------------------------------------------


def test_pc8_raw_yields_the_line_text_and_record(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_bytes(b'{"a": 1}\n\n{"b":  2}\r\n')
    got = list(jio.iter_jsonl_raw(p))
    assert got == [('{"a": 1}', {"a": 1}), ('{"b":  2}', {"b": 2})]  # text exactly as written


def test_pc8_raw_skips_bad_lines_unless_asked(tmp_path: Path, caplog) -> None:
    p = tmp_path / "a.jsonl"
    p.write_bytes(b'{"a": 1}\nnot json\n[1]\n\xff\xfe\n{"c": 3}\n')
    with caplog.at_level(logging.WARNING):
        assert [r for _, r in jio.iter_jsonl_raw(p)] == [{"a": 1}, {"c": 3}]
    got = list(jio.iter_jsonl_raw(p, bad_lines=True))
    assert [r for _, r in got] == [{"a": 1}, None, None, None, {"c": 3}]
    assert [t for t, _ in got][1:3] == ["not json", "[1]"]


def test_pc8_raw_strict_raises_on_the_first_bad_line(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_bytes(b'{"a": 1}\nbroken\n')
    it = jio.iter_jsonl_raw(p, strict=True)
    assert next(it)[1] == {"a": 1}
    with pytest.raises(JsonFileError) as exc:
        next(it)
    assert exc.value.kind == "line" and exc.value.line == 2


def test_pc8_raw_reads_gz_and_a_missing_file(tmp_path: Path) -> None:
    import gzip

    p = tmp_path / "a.jsonl.gz"
    with gzip.open(p, "wb") as fh:
        fh.write(b'{"a": 1}\n')
    assert list(jio.iter_jsonl_raw(p)) == [('{"a": 1}', {"a": 1})]
    assert list(jio.iter_jsonl_raw(tmp_path / "none.jsonl")) == []


def test_pc11_raw_recover_raise_raises_on_an_io_fault(tmp_path: Path) -> None:
    d = tmp_path / "a.jsonl"
    d.mkdir()
    with pytest.raises(JsonFileError) as exc:
        list(jio.iter_jsonl_raw(d, recover=Recovery.RAISE))
    assert exc.value.kind == "io"
    assert list(jio.iter_jsonl_raw(d)) == []  # DEFAULT stops with a warning


def test_pc8_raw_bom_on_the_first_line_is_skipped(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_bytes(b'\xef\xbb\xbf{"a": 1}\n')
    assert [r for _, r in jio.iter_jsonl_raw(p)] == [{"a": 1}]


def test_pc8_a_simulated_windows_os_name_does_not_make_the_module_build_a_windowspath(
    tmp_path: Path, monkeypatch
) -> None:
    """Tests that patch ``os.name`` to "nt" (daemon cmd_stop on Windows) call the migrated state reader: a
    ``Path(path)`` of a Path would pick WindowsPath there, which cannot be instantiated on POSIX."""
    p = tmp_path / "state.json"
    p.write_text('{"a": 1}')
    monkeypatch.setattr(os, "name", "nt")
    assert jio.read_json(p, default=dict) == {"a": 1}
    assert jio.read_json_result(p, default=dict, recover=Recovery.HEAL).data == {"a": 1}
    assert list(jio.iter_jsonl(tmp_path / "none.jsonl")) == []
