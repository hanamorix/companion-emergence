import json

from brain.tools.impls.read_file import _FILE_READ_MAX_BYTES, read_file


def test_read_file_returns_text(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello from disk")
    out = read_file(path=str(f), persona_dir=tmp_path)
    assert out["content"] == "hello from disk"


def test_read_file_oversized_refused_not_truncated(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("x" * (_FILE_READ_MAX_BYTES + 1))
    out = read_file(path=str(f), persona_dir=tmp_path)
    assert "error" in out and "content" not in out


def test_read_file_missing_returns_clean_error(tmp_path):
    out = read_file(path=str(tmp_path / "nope.txt"), persona_dir=tmp_path)
    assert "error" in out


def test_read_file_directory_returns_clean_error(tmp_path):
    out = read_file(path=str(tmp_path), persona_dir=tmp_path)
    assert "error" in out


def test_read_file_writes_audit_line(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hi")
    read_file(path=str(f), persona_dir=tmp_path)
    line = json.loads((tmp_path / "file_access.jsonl").read_text().strip().splitlines()[-1])
    assert line["tool"] == "read_file" and line["ok"] is True


def test_read_file_binary_returns_note_not_bytes(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"\xff\xfe\x00\x01\x02\x80\x81")  # invalid UTF-8
    out = read_file(path=str(f), persona_dir=tmp_path)
    assert "content" not in out
    assert "note" in out and "binary" in out["note"].lower()


def test_read_file_through_dispatch(tmp_path):
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    from brain.tools import NELL_TOOL_NAMES, dispatch
    from brain.tools.schemas import SCHEMAS

    assert "read_file" in NELL_TOOL_NAMES and "read_file" in SCHEMAS
    f = tmp_path / "n.txt"
    f.write_text("ok")
    out = dispatch(
        "read_file",
        {"path": str(f)},
        store=MemoryStore(":memory:"),
        hebbian=HebbianMatrix(":memory:"),
        persona_dir=tmp_path,
    )
    assert out["content"] == "ok"  # through-path test (Organ-DoD)
