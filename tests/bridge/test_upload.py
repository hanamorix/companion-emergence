"""Bridge POST /upload — multimodal image upload endpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app

# Minimal 1×1 transparent PNG.
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63606060600000000400015e36b8c80000000049454e44ae426082"
)
_TINY_PNG_SHA = hashlib.sha256(_TINY_PNG).hexdigest()


def _client(persona_dir: Path, auth_token: str | None = None) -> TestClient:
    app = build_app(
        persona_dir=persona_dir,
        client_origin="tests",
        auth_token=auth_token,
    )
    return TestClient(app)


def test_upload_happy_path_returns_sha(persona_dir: Path) -> None:
    client = _client(persona_dir)
    with client:
        r = client.post(
            "/upload",
            files={"file": ("photo.png", _TINY_PNG, "image/png")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sha"] == _TINY_PNG_SHA
    assert body["media_type"] == "image/png"
    assert body["size_bytes"] == len(_TINY_PNG)
    saved = persona_dir / "images" / f"{_TINY_PNG_SHA}.png"
    assert saved.exists()
    assert saved.read_bytes() == _TINY_PNG


def test_upload_dedupes_same_content(persona_dir: Path) -> None:
    client = _client(persona_dir)
    with client:
        r1 = client.post(
            "/upload",
            files={"file": ("a.png", _TINY_PNG, "image/png")},
        )
        r2 = client.post(
            "/upload",
            files={"file": ("b.png", _TINY_PNG, "image/png")},
        )
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["sha"] == r2.json()["sha"]
    files = list((persona_dir / "images").iterdir())
    assert len(files) == 1


def test_upload_stores_non_image_file(persona_dir: Path) -> None:
    """#43 widening: a non-image file is stored (kind=file), no longer 415'd.

    The pre-change endpoint hard-refused every non-image at 415; the widened
    endpoint routes it to the general content-addressable store.
    """
    payload = b"%PDF-1.7 hello"
    sha = hashlib.sha256(payload).hexdigest()
    client = _client(persona_dir)
    with client:
        r = client.post(
            "/upload",
            files={"file": ("x.pdf", payload, "application/pdf")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "file"
    assert body["sha"] == sha
    assert body["filename"] == "x.pdf"
    # On-disk name is the sha ONLY (no client filename/extension in the path).
    saved = persona_dir / "files" / sha
    assert saved.exists()
    assert saved.read_bytes() == payload
    assert not (persona_dir / "files" / "x.pdf").exists()


def test_upload_rejects_oversized_file(persona_dir: Path) -> None:
    """Endpoint enforces a 20MB cap by default."""
    client = _client(persona_dir)
    huge = b"\x00" * (20 * 1024 * 1024 + 1)
    with client:
        r = client.post(
            "/upload",
            files={"file": ("big.png", huge, "image/png")},
        )
    assert r.status_code == 413


def test_upload_requires_auth_when_token_set(persona_dir: Path) -> None:
    client = _client(persona_dir, auth_token="secret-token")
    with client:
        # No auth header — denied.
        r = client.post(
            "/upload",
            files={"file": ("p.png", _TINY_PNG, "image/png")},
        )
    assert r.status_code == 401


def test_upload_passes_with_correct_bearer(persona_dir: Path) -> None:
    client = _client(persona_dir, auth_token="secret-token")
    with client:
        r = client.post(
            "/upload",
            files={"file": ("p.png", _TINY_PNG, "image/png")},
            headers={"Authorization": "Bearer secret-token"},
        )
    assert r.status_code == 200


def test_upload_rejects_wrong_bearer(persona_dir: Path) -> None:
    client = _client(persona_dir, auth_token="secret-token")
    with client:
        r = client.post(
            "/upload",
            files={"file": ("p.png", _TINY_PNG, "image/png")},
            headers={"Authorization": "Bearer wrong"},
        )
    assert r.status_code == 401


def test_upload_rejects_mismatched_magic_bytes(persona_dir: Path) -> None:
    """Client claims PNG but bytes don't have the PNG signature → 422."""
    client = _client(persona_dir)
    with client:
        r = client.post(
            "/upload",
            files={"file": ("fake.png", b"GIF89a" + b"\x00" * 20, "image/png")},
        )
    assert r.status_code == 422
    assert "look like" in r.json()["detail"]


def test_upload_stores_unrecognised_bytes_as_file(persona_dir: Path) -> None:
    """Bytes that sniff as no image type are stored in the general file store.

    Pre-change these were 422'd; the #43 widening treats any non-image bytes
    as a general file (the on-disk name is still the content sha only).
    """
    payload = b"this is not any image"
    sha = hashlib.sha256(payload).hexdigest()
    client = _client(persona_dir)
    with client:
        r = client.post(
            "/upload",
            files={"file": ("mystery.txt", payload, "text/plain")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "file"
    assert body["sha"] == sha
    assert (persona_dir / "files" / sha).exists()
