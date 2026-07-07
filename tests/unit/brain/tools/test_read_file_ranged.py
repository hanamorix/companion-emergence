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
