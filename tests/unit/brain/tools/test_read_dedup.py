from brain.tools.impls import _read_cache
from brain.tools.impls.read_file import read_file


def test_repeat_read_same_turn_is_deduped(tmp_path, monkeypatch):
    _read_cache.reset()
    f = tmp_path / "Notes.md"
    f.write_text("secret content", encoding="utf-8")
    a = read_file(str(f), persona_dir=tmp_path / "p")
    assert "secret content" in a["content"]
    # second read of the SAME file (mixed case path) → dedup note, no re-emit
    b = read_file(str(tmp_path / "notes.md").replace("notes.md", "Notes.md"),
                  persona_dir=tmp_path / "p")
    assert b.get("deduped") is True
    assert "secret content" not in b.get("content", "")


def test_reset_clears_dedup(tmp_path):
    _read_cache.reset()
    f = tmp_path / "x.md"
    f.write_text("hi", encoding="utf-8")
    read_file(str(f), persona_dir=tmp_path / "p")
    _read_cache.reset()
    out = read_file(str(f), persona_dir=tmp_path / "p")
    assert "hi" in out["content"]  # fresh again after reset


def test_next_range_of_a_file_in_the_same_turn_is_not_deduped(tmp_path):
    """#590: a file is read a range at a time, so the next range in the same turn is new content;
    only a repeat of the same range is deduped."""
    _read_cache.reset()
    f = tmp_path / "long.txt"
    f.write_bytes(b"".join(b"row%d\n" % i for i in range(20)))
    first = read_file(str(f), persona_dir=tmp_path / "p", offset=0, max_lines=5)
    assert first["content"].startswith("row0\n")
    second = read_file(str(f), persona_dir=tmp_path / "p", offset=5, max_lines=5)
    assert second.get("deduped") is not True
    assert second["content"] == "row5\nrow6\nrow7\nrow8\nrow9\n"
    again = read_file(str(f), persona_dir=tmp_path / "p", offset=5, max_lines=5)
    assert again.get("deduped") is True
