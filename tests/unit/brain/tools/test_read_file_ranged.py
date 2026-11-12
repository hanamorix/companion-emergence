import tracemalloc

import brain.tools.impls.read_file as rf
from brain.tools.impls.read_file import read_file


def _big(tmp_path, n):
    p = tmp_path / "big.txt"
    p.write_text("\n".join(f"line{i}" for i in range(n)), encoding="utf-8")
    return p


def test_max_lines_returns_slice(tmp_path):
    p = _big(tmp_path, 1000)
    out = read_file(str(p), persona_dir=tmp_path / "x", max_lines=10)
    assert out["content"].count("\n") <= 10
    assert out["truncated"] is True and out["total_lines"] == 1000


def test_offset_and_max_lines(tmp_path):
    p = _big(tmp_path, 100)
    out = read_file(str(p), persona_dir=tmp_path / "x", offset=50, max_lines=5)
    assert out["content"].startswith("line50")


def test_large_file_without_max_lines_is_head_capped(tmp_path):
    p = _big(tmp_path, 2000)
    out = read_file(str(p), persona_dir=tmp_path / "x")
    assert out["truncated"] is True
    assert out["total_lines"] == 2000
    assert out["content"].count("\n") <= 400  # _DEFAULT_HEAD_LINES


def test_small_file_returned_whole(tmp_path):
    p = tmp_path / "s.txt"
    # newline="\n": write_text otherwise translates \n → os.linesep, and
    # read_file is deliberately byte-faithful — on Windows the fixture would
    # become CRLF on disk and the assertion would compare the wrong bytes.
    p.write_text("a\nb\nc", encoding="utf-8", newline="\n")
    out = read_file(str(p), persona_dir=tmp_path / "x")
    assert out["content"] == "a\nb\nc" and out.get("truncated", False) is False


# ---------------------------------------------------------------------------
# #590 — a text file over the old 256 KiB cap is read a window at a time
# ---------------------------------------------------------------------------


def _numbered(tmp_path, n, *, name="big.txt"):
    """n lines "line0000000\\n" (12 bytes each), written byte-exact."""
    p = tmp_path / name
    p.write_bytes(b"".join(b"line%07d\n" % i for i in range(n)))
    return p


_ONE_MB_LINES = (1024 * 1024) // 12 + 1  # just over 1 MiB of 12-byte lines


def test_one_megabyte_text_file_reads_a_range(tmp_path):
    """#590 acceptance: a 1 MB text file is not refused whole; offset/max_lines reach into it."""
    p = _numbered(tmp_path, _ONE_MB_LINES)
    assert p.stat().st_size > 1024 * 1024
    out = read_file(str(p), persona_dir=tmp_path / "x", offset=50_000, max_lines=5)
    assert "error" not in out, out
    assert out["content"] == "".join(f"line{i:07d}\n" for i in range(50_000, 50_005))
    assert out["total_lines"] == _ONE_MB_LINES
    assert out["truncated"] is True
    assert out["note"].startswith(f"showing lines 50000-50005 of {_ONE_MB_LINES}")


def test_one_megabyte_text_file_without_range_shows_first_window(tmp_path):
    """#590 acceptance: no offset/max_lines -> the first window plus the "showing lines..." note."""
    p = _numbered(tmp_path, _ONE_MB_LINES)
    out = read_file(str(p), persona_dir=tmp_path / "x")
    assert "error" not in out, out
    assert out["content"] == "".join(f"line{i:07d}\n" for i in range(400))
    assert out["truncated"] is True and out["total_lines"] == _ONE_MB_LINES
    assert out["note"] == f"showing lines 0-400 of {_ONE_MB_LINES}; pass offset/max_lines to read more"


def test_offset_without_max_lines_windows_from_offset(tmp_path):
    """offset alone moves the default window. Before #590 a file over 400 lines ignored it and
    returned the head while the note claimed the window started at the offset."""
    p = _numbered(tmp_path, 2000)
    out = read_file(str(p), persona_dir=tmp_path / "x", offset=1000)
    assert out["content"] == "".join(f"line{i:07d}\n" for i in range(1000, 1400))
    assert out["note"] == "showing lines 1000-1400 of 2000; pass offset/max_lines to read more"


def test_file_over_the_cap_is_still_refused_with_the_same_error(tmp_path, monkeypatch):
    """The byte cap stays as a guard against pathological files: same refusal shape and audit."""
    monkeypatch.setattr(rf, "_file_read_max_bytes", lambda: 1024)
    p = _numbered(tmp_path, 200)  # 2400 bytes
    persona = tmp_path / "x"
    out = read_file(str(p), persona_dir=persona, offset=0, max_lines=5)
    assert out == {"error": "file too large (2400 bytes > 1024 cap) — not shown"}
    audit = (persona / "file_access.jsonl").read_text(encoding="utf-8")
    assert '"error": "too large"' in audit


def test_default_text_cap_matches_the_upload_cap():
    """#590: the cap guards against pathological files and is sized to the 20 MiB upload, image
    and PDF caps, so any file the user can share also opens as text."""
    assert rf._FILE_READ_MAX_BYTES == rf._FILE_PDF_MAX_BYTES == rf._FILE_IMAGE_MAX_BYTES == 20 * 1024 * 1024


def test_multibyte_character_split_across_read_chunks_decodes(tmp_path):
    """A UTF-8 character that straddles two read chunks is decoded, not reported as binary."""
    body = b"a" * (rf._READ_CHUNK_BYTES - 1) + "é".encode() + b"\nnext\n"
    p = tmp_path / "accent.txt"
    p.write_bytes(body)
    out = read_file(str(p), persona_dir=tmp_path / "x")
    assert out["content"] == body.decode("utf-8")
    assert out["total_lines"] == 2


def test_line_breaks_split_like_str_splitlines_across_read_chunks(tmp_path):
    """A \\r\\n straddling two read chunks is one break; \\r and form feed break lines too,
    exactly as the old whole-file str.splitlines() did."""
    body = b"a" * (rf._READ_CHUNK_BYTES - 1) + b"\r\nb\rc\x0cd"
    p = tmp_path / "breaks.txt"
    p.write_bytes(body)
    out = read_file(str(p), persona_dir=tmp_path / "x", offset=1, max_lines=10)
    assert out["total_lines"] == len(body.decode("utf-8").splitlines()) == 4
    assert out["content"] == "b\rc\x0cd"


def test_invalid_utf8_after_the_window_is_still_binary(tmp_path):
    """Binary detection covers the whole file, as the old whole-file decode did, not just the lines shown."""
    p = tmp_path / "late.bin"
    p.write_bytes(b"".join(b"line%07d\n" % i for i in range(20_000)) + b"\xff\xfe\x00")
    out = read_file(str(p), persona_dir=tmp_path / "x", max_lines=5)
    assert "content" not in out
    assert "binary" in out["note"]


def test_range_read_does_not_load_the_whole_file(tmp_path):
    """A range read streams to the offset: peak Python memory stays far below the file's size."""
    p = _numbered(tmp_path, 700_000)  # ~8 MiB
    size = p.stat().st_size
    tracemalloc.start()
    try:
        out = read_file(str(p), persona_dir=tmp_path / "x", offset=600_000, max_lines=10)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert out["content"].startswith("line0600000\n")
    assert out["total_lines"] == 700_000
    assert peak < size // 8, f"peak {peak} bytes for a {size}-byte file"


def test_a_huge_single_line_is_cut_to_the_window_budget(tmp_path):
    """A 1 MB file with no line breaks must not put 1 MB into the result."""
    p = tmp_path / "oneline.txt"
    p.write_bytes(b"x" * (1024 * 1024))
    out = read_file(str(p), persona_dir=tmp_path / "x")
    assert "error" not in out, out
    assert out["content"] == "x" * rf._WINDOW_MAX_CHARS
    assert out["total_lines"] == 1 and out["truncated"] is True
    assert out["note"] == (
        f"showing lines 0-1 of 1; line 0 is longer than {rf._WINDOW_MAX_CHARS} characters and was cut short; "
        "pass offset/max_lines to read more"
    )


def test_window_stops_at_the_character_budget(tmp_path, monkeypatch):
    """Long lines cannot add up past the budget: the window ends early and says where to resume."""
    monkeypatch.setattr(rf, "_WINDOW_MAX_CHARS", 100)
    p = tmp_path / "wide.txt"
    p.write_bytes(b"".join(b"%02d" % i + b"y" * 27 + b"\n" for i in range(50)))  # 30 chars per line
    out = read_file(str(p), persona_dir=tmp_path / "x")
    assert out["content"] == "".join(f"{i:02d}" + "y" * 27 + "\n" for i in range(3))
    assert out["note"] == "showing lines 0-3 of 50; stopped at 100 characters; pass offset/max_lines to read more"
    nxt = read_file(str(p), persona_dir=tmp_path / "y", offset=3, max_lines=2)
    assert nxt["content"] == "03" + "y" * 27 + "\n04" + "y" * 27 + "\n"
