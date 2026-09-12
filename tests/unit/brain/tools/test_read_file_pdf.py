"""#126: read_file extracts text from PDFs (pypdf), page-marked, windowed like text."""

from __future__ import annotations

from pathlib import Path

from brain import tunables
from brain.tools.impls.read_file import read_file


def _make_pdf(page_texts):
    """Minimal valid PDF, one page per entry; None => page with no text stream."""
    objs = []  # list of bytes bodies, 1-indexed
    def add(body):
        objs.append(body)
        return len(objs)
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    pages_placeholder = add(b"")  # will be replaced
    for text in page_texts:
        if text is None:
            content = None
        else:
            esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("latin-1")
            stream = b"BT /F1 12 Tf 72 720 Td (" + esc + b") Tj ET"
            content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        pg = b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 %d 0 R >> >>" % (pages_placeholder, font)
        if content is not None:
            pg += b" /Contents %d 0 R" % content
        pg += b" >>"
        page_ids.append(add(pg))
    kids = b" ".join(b"%d 0 R" % i for i in page_ids)
    objs[pages_placeholder - 1] = b"<< /Type /Pages /Kids [" + kids + b"] /Count %d >>" % len(page_ids)
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_placeholder)
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, catalog, xref)
    return bytes(out)


def test_read_file_pdf_returns_page_marked_text(tmp_path: Path) -> None:
    f = tmp_path / "doc.pdf"
    f.write_bytes(_make_pdf(["Hello PDF world", "Second page here"]))

    out = read_file(path=str(f), persona_dir=tmp_path)

    assert out["pages"] == 2
    assert "Hello PDF world" in out["content"]
    assert "Second page here" in out["content"]
    assert "--- page 2 ---" in out["content"]
    assert "binary" not in out.get("note", "")


def test_read_file_pdf_without_text_layer_says_so(tmp_path: Path) -> None:
    f = tmp_path / "scan.pdf"
    f.write_bytes(_make_pdf([None, None]))

    out = read_file(path=str(f), persona_dir=tmp_path)

    assert "content" not in out
    assert "no extractable text" in out["note"]
    assert out["pages"] == 2


def test_read_file_pdf_over_cap_is_refused(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "big.pdf"
    f.write_bytes(_make_pdf(["x"]))
    monkeypatch.setattr(tunables, "get_tunable", lambda key, default=None: 10 if key == "files.pdf_max_bytes" else default)

    out = read_file(path=str(f), persona_dir=tmp_path)

    assert "error" in out and "content" not in out


def test_read_file_pdf_ranged_read_windows_lines(tmp_path: Path) -> None:
    f = tmp_path / "two.pdf"
    f.write_bytes(_make_pdf(["alpha", "beta"]))

    out = read_file(path=str(f), persona_dir=tmp_path, max_lines=1, offset=0)

    assert out["truncated"] is True
    assert "alpha" in out["content"]
    assert "beta" not in out["content"]
    assert "%PDF" not in out["content"]  # extracted text, not the raw bytes


def test_read_file_corrupt_pdf_fails_soft(tmp_path: Path) -> None:
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"%PDF-1.4\ngarbage with no xref")

    out = read_file(path=str(f), persona_dir=tmp_path)

    assert "error" in out and "content" not in out
