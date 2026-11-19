"""#735: read_file counts the lines outside its window from whole chunks, and can read past a cut line.

Every ranged read used to run each line of the whole file through a Python loop, only to count lines and to call the
file binary if any byte wasn't UTF-8: about 3 s for a 20 MiB file of bare newlines (2026-11-19, M3 Pro). The results
must stay exactly what they were; only the time changes.
"""

from __future__ import annotations

import time

import pytest

from brain.tools.impls import _read_cache
from brain.tools.impls.read_file import read_file


def test_a_deep_page_of_a_20mib_file_of_newlines_is_fast(tmp_path):
    p = tmp_path / "nl.txt"
    p.write_bytes(b"\n" * (20 * 1024 * 1024))

    start = time.process_time()
    out = read_file(str(p), persona_dir=tmp_path / "x", offset=10_000_000, max_lines=5)
    cpu = time.process_time() - start

    assert out["total_lines"] == 20 * 1024 * 1024
    assert out["content"] == "\n" * 5
    assert cpu < 0.75, f"a deep page took {cpu:.2f} s of CPU"


_CHUNK = 64 * 1024  # read_file._READ_CHUNK_BYTES
_BREAKS = "\n\r\x0b\x0c\x1c\x1d\x1e\x85  "


def _texts():
    """Texts whose line ends fall on and around the chunk boundaries."""
    mixed = "".join(f"line {i}{_BREAKS[i % len(_BREAKS)]}" for i in range(30_000))
    return {
        "empty": "",
        "no final break": "a\nb\nc",
        "crlf split at the chunk boundary": "x" * (_CHUNK - 1) + "\r\n" + "tail\n" * 10,
        "bare cr at the chunk boundary": "x" * (_CHUNK - 1) + "\rnext\n" * 3,
        "bare cr at the end": "a\r\nb\r",
        "every break character": mixed,
        "crlf everywhere": "row\r\n" * 40_000,
        "long lines across chunks": ("y" * 50_000 + "\n") * 6,
    }


@pytest.mark.no_base_failure(reason="pins that #735's chunk counting gives exactly the per-line results main gives")
@pytest.mark.parametrize("name", list(_texts()))
def test_every_window_matches_splitlines(tmp_path, name):
    text = _texts()[name]
    p = tmp_path / "f.txt"
    p.write_bytes(text.encode("utf-8"))
    lines = text.splitlines(keepends=True)
    n = len(lines)
    for offset, count in [(0, None), (0, 3), (1, 2), (n // 2, 4), (max(n - 2, 0), 5), (n, 1), (n + 3, 1)]:
        _read_cache.reset()  # each window is a fresh read, not a repeat within the dedup window
        out = read_file(str(p), persona_dir=tmp_path / "x", offset=offset, max_lines=count)
        assert out["total_lines"] == n, (name, offset)
        if "error" in out or out.get("note", "").startswith("offset"):
            continue
        want = lines[offset : offset + (400 if count is None else count)]
        if sum(len(x) for x in want) <= 30_000:  # well inside the window budget, so nothing is cut
            assert out["content"] == "".join(want), (name, offset, count)


@pytest.mark.no_base_failure(reason="pins that a bad byte after the window still makes the file binary (#735); main does too")
def test_a_bad_byte_after_the_window_still_makes_the_file_binary(tmp_path):
    p = tmp_path / "late.txt"
    p.write_bytes(b"fine line\n" * 200_000 + b"\xff\n")

    out = read_file(str(p), persona_dir=tmp_path / "x", max_lines=5)

    assert "content" not in out and "binary file" in out["note"]


# ---- #735 part 2: a line cut at the window budget can be read past the cut ----
def test_a_cut_line_can_be_read_to_its_end_with_char_offset(tmp_path):
    import re

    long_line = "".join(chr(ord("a") + i % 26) for i in range(100_000))
    p = tmp_path / "long.txt"
    p.write_bytes((long_line + "\nnext\n").encode("utf-8"))

    got, char_offset, reads = "", 0, 0
    while True:
        _read_cache.reset()
        out = read_file(str(p), persona_dir=tmp_path / "x", offset=0, max_lines=1, char_offset=char_offset)
        got += out["content"]
        reads += 1
        m = re.search(r"pass offset=0, char_offset=(\d+) to read on", out.get("note", ""))
        if m is None:
            break
        char_offset = int(m.group(1))
        assert reads < 10
    assert got == long_line + "\n"
    assert reads == 3  # 100,000 characters at 40,000 a window


def test_a_char_offset_past_the_line_end_gives_its_break_alone(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"short\nnext\n")
    out = read_file(str(p), persona_dir=tmp_path / "x", offset=0, max_lines=2, char_offset=999)
    assert out["content"] == "\nnext\n"
    assert "(line 0 from character 999)" in out["note"]


@pytest.mark.no_base_failure(reason="pins that a negative char_offset reads from the line start, not an error (#735); main ignores char_offset, so it passes there too")
def test_a_negative_char_offset_acts_as_zero(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"whole\n")
    out = read_file(str(p), persona_dir=tmp_path / "x", char_offset=-5)
    assert out["content"] == "whole\n" and "truncated" not in out


def test_two_char_offsets_in_one_turn_are_two_reads(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"abcdef\n")
    _read_cache.reset()
    first = read_file(str(p), persona_dir=tmp_path / "x", char_offset=1)
    second = read_file(str(p), persona_dir=tmp_path / "x", char_offset=3)
    assert first["content"] == "bcdef\n" and second["content"] == "def\n"


def test_a_pdf_line_is_read_from_char_offset(tmp_path):
    from tests.unit.brain.tools.test_read_file_pdf import _make_pdf

    f = tmp_path / "doc.pdf"
    f.write_bytes(_make_pdf(["hello world"]))
    out = read_file(path=str(f), persona_dir=tmp_path, offset=0, max_lines=1, char_offset=6)
    assert out["content"] == "world\n"


def test_the_schema_offers_char_offset():
    from brain.tools.schemas import SCHEMAS

    params = SCHEMAS["read_file"]["parameters"]
    assert params["properties"]["char_offset"]["type"] == "integer"
    assert "char_offset" not in params["required"]


def test_a_char_offset_past_an_unterminated_last_line_keeps_the_count(tmp_path):
    """Review of #735: dropping the whole of a last line with no break must still count it."""
    p = tmp_path / "f.txt"
    p.write_bytes(b"abc")
    out = read_file(str(p), persona_dir=tmp_path / "x", char_offset=3)
    assert out["total_lines"] == 1
    assert out["content"] == "" and out["note"].startswith("showing lines 0-1 of 1 (line 0 from character 3)")
    _read_cache.reset()
    p.write_bytes(b"l0\nabc")
    assert read_file(str(p), persona_dir=tmp_path / "x", offset=1, char_offset=5)["total_lines"] == 2


def test_the_count_after_the_window_stays_fast_when_chunks_end_mid_line(tmp_path):
    """Review of #735: an odd first byte puts every chunk boundary inside a line; the tail must still be counted."""
    p = tmp_path / "odd.txt"
    p.write_bytes(b"x" + b"a\n" * (10 * 1024 * 1024 - 1))
    start = time.process_time()
    out = read_file(str(p), persona_dir=tmp_path / "x", max_lines=5)
    cpu = time.process_time() - start
    assert out["total_lines"] == 10 * 1024 * 1024 - 1  # "xa\n", then "a\n" lines
    assert cpu < 0.5, f"the head read took {cpu:.2f} s of CPU"
