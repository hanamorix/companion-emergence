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


# ---------------------------------------------------------------------------
# P0 image-tool-route — read_file returns viewable images (cap-ordering)
# ---------------------------------------------------------------------------

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _png_of_size(n: int) -> bytes:
    """A byte blob that sniffs as image/png (magic prefix) padded to >= n bytes."""
    return _PNG_MAGIC + b"\x00" * max(0, n - len(_PNG_MAGIC))


def test_read_file_returns_image_result_for_png(tmp_path):
    """A png returns a structured image result (base64 + media_type), not text."""
    f = tmp_path / "pic.png"
    f.write_bytes(_png_of_size(64))
    out = read_file(path=str(f), persona_dir=tmp_path)
    assert "image" in out and "content" not in out
    assert out["image"]["media_type"] == "image/png"
    assert out["image"]["data_b64"]  # non-empty base64
    assert out["image"]["size_bytes"] == 64


def test_read_file_large_image_not_refused_by_text_cap(tmp_path):
    """C12 — an image between the 256KB text cap and image_max_bytes returns an
    image result, NOT 'file too large'. This is the cap-ordering guard: the
    image branch runs before the text cap."""
    size = _FILE_READ_MAX_BYTES + 50_000  # > 256KB text cap, << 20MB image cap
    f = tmp_path / "big.png"
    f.write_bytes(_png_of_size(size))
    out = read_file(path=str(f), persona_dir=tmp_path)
    assert "image" in out, out
    assert "error" not in out
    # Discriminator: the file IS over the text cap, so a text-cap-first ordering
    # would refuse it — proving the ordering is load-bearing.
    assert f.stat().st_size > _FILE_READ_MAX_BYTES


def test_read_file_large_image_refused_when_image_branch_disabled(tmp_path, monkeypatch):
    """C12 shown-able-to-fail (arm b) — reproduce the mis-ordered/pre-change
    behaviour: with the image branch effectively off, the same >256KB image
    hits the text cap and returns 'file too large'."""
    import brain.tools.impls.read_file as rf

    monkeypatch.setattr(rf, "_image_types", lambda: [])  # disable image detection
    size = _FILE_READ_MAX_BYTES + 50_000
    f = tmp_path / "big2.png"
    f.write_bytes(_png_of_size(size))
    out = read_file(path=str(f), persona_dir=tmp_path)
    assert "error" in out and "too large" in out["error"]
    assert "image" not in out


def test_read_file_image_over_image_max_bytes_refused(tmp_path, monkeypatch):
    """C12 shown-able-to-fail (arm a) — an image larger than image_max_bytes is
    refused with an image-specific 'image too large' error, proving the image
    cap is live."""
    import brain.tools.impls.read_file as rf

    monkeypatch.setattr(rf, "_image_max_bytes", lambda: 1024)
    f = tmp_path / "huge.png"
    f.write_bytes(_png_of_size(4096))
    out = read_file(path=str(f), persona_dir=tmp_path)
    assert "error" in out and "image too large" in out["error"]
    assert "image" not in out


def test_read_file_image_audit_has_no_base64(tmp_path):
    """C15 — the file_access audit line for an image read records the size, never
    the base64 payload."""
    f = tmp_path / "audited.png"
    f.write_bytes(_png_of_size(128))
    out = read_file(path=str(f), persona_dir=tmp_path)
    b64 = out["image"]["data_b64"]
    line = (tmp_path / "file_access.jsonl").read_text().strip().splitlines()[-1]
    assert b64 not in line
    rec = json.loads(line)
    assert rec["ok"] is True and rec["bytes"] == 128


def test_read_file_text_still_returns_text_after_image_branch(tmp_path):
    """C3 — a normal UTF-8 text file still returns its text content unchanged
    (the image branch falls through for non-image bytes)."""
    f = tmp_path / "note.md"
    f.write_text("UNIQUE-LINE-c3 hello")
    out = read_file(path=str(f), persona_dir=tmp_path)
    assert "UNIQUE-LINE-c3" in out["content"]
    assert "image" not in out
