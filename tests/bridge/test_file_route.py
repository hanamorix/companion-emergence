"""P0 image-tool-route — #43 end-to-end (C16) + upload path-traversal (C17).

C16: a non-image file POSTed to the widened /upload is stored on disk and its
stored path is readable via the read_file tool (returns the file's text).
Shown-able-to-fail (annotated): the PRE-change /upload hard-refused every
non-image at 415/422, so the chain never started — the widening is what closes
#43.

C17: a crafted upload filename containing traversal characters cannot escape
``<persona_dir>/files/`` — the on-disk name is the validated content sha only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app
from brain.tools.impls.read_file import read_file


def _client(persona_dir: Path) -> TestClient:
    app = build_app(persona_dir=persona_dir, client_origin="tests", auth_token=None)
    return TestClient(app)


def test_upload_then_read_file_end_to_end(persona_dir: Path) -> None:
    """C16 — POST a text file → stored → read_file returns the known line."""
    known = "UNIQUE-C16-LINE the kolinsky sable"
    payload = f"first line\n{known}\nlast line".encode()
    sha = hashlib.sha256(payload).hexdigest()
    client = _client(persona_dir)
    with client:
        r = client.post("/upload", files={"file": ("doc.txt", payload, "text/plain")})
    assert r.status_code == 200, r.text  # pre-change: 415/422 (image-only) — chain never started
    body = r.json()
    assert body["kind"] == "file"
    assert body["sha"] == sha
    # The stored path is derived from the sha; read it back via read_file.
    stored = persona_dir / "files" / sha
    assert stored.exists()
    out = read_file(path=str(stored), persona_dir=persona_dir)
    assert "content" in out
    assert known in out["content"]


def test_upload_traversal_filename_cannot_escape_store(persona_dir: Path) -> None:
    """C17 — a traversal upload filename does not create any path outside
    ``<persona_dir>/files/``; the on-disk name is the content sha only."""
    payload = b"traversal attempt payload"
    sha = hashlib.sha256(payload).hexdigest()
    client = _client(persona_dir)
    with client:
        r = client.post(
            "/upload",
            files={"file": ("../../../etc/pwned", payload, "application/octet-stream")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sha"] == sha
    # Stored strictly under files/<sha>; the filename is display-only metadata.
    stored = persona_dir / "files" / sha
    assert stored.exists()
    assert stored.resolve().parent == (persona_dir / "files").resolve()
    # No traversal dirs were created adjacent to the persona dir.
    assert not (persona_dir.parent / "etc").exists()
    # The files dir contains exactly the sha-named file, nothing path-shaped.
    entries = [p.name for p in (persona_dir / "files").iterdir()]
    assert entries == [sha]
    assert "pwned" not in "".join(entries)
