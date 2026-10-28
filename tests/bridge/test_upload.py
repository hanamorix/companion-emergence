"""Bridge POST /upload — multimodal image upload endpoint."""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import starlette.formparsers
from fastapi import UploadFile
from fastapi.testclient import TestClient
from starlette.formparsers import MultiPartParser

from brain import file_store, tunables
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


def test_upload_honours_a_lowered_upload_cap(persona_dir: Path) -> None:
    """`files.upload_max_bytes` is the one cap of /upload (#531): lowered below 20 MiB it binds images and files alike.

    The route used to compare against ``max(<its own 20 MiB literal>, upload_max_bytes())``, so a lowered setting
    was silently ignored.
    """
    tunables.set_values({"files.upload_max_bytes": 1024})
    assert file_store.upload_max_bytes() == 1024
    png_of_the_cap = _TINY_PNG + b"\x00" * (1024 - len(_TINY_PNG))
    client = _client(persona_dir)
    with client:
        too_big_file = client.post("/upload", files={"file": ("big.bin", b"a" * 1025, "application/octet-stream")})
        too_big_image = client.post("/upload", files={"file": ("big.png", png_of_the_cap + b"\x00", "image/png")})
        file_at_the_cap = client.post("/upload", files={"file": ("ok.bin", b"b" * 1024, "application/octet-stream")})
        image_at_the_cap = client.post("/upload", files={"file": ("ok.png", png_of_the_cap, "image/png")})
    assert too_big_file.status_code == 413, too_big_file.text
    assert too_big_image.status_code == 413, too_big_image.text
    assert file_at_the_cap.status_code == 200, file_at_the_cap.text
    assert image_at_the_cap.status_code == 200, image_at_the_cap.text


_KEY = "files.upload_max_bytes"


def test_upload_reads_a_cap_that_is_not_above_zero_as_the_registered_default(persona_dir: Path) -> None:
    """A hand-edited cap of 0 or below falls back to the registered default (#531), it does not refuse everything.

    With 0 the route would answer 413 to every non-empty body; with -2 it would read the whole body before refusing.
    """
    default = file_store._UPLOAD_MAX_BYTES  # noqa: SLF001
    tunables.set_values({_KEY: 1024})
    client = _client(persona_dir)
    with client:
        anchor = client.post("/upload", files={"file": ("big.bin", b"a" * 1025, "application/octet-stream")})
        assert anchor.status_code == 413, anchor.text  # the route follows the setting, so the phases below mean something
        for bad in (0, -2):
            tunables.set_values({_KEY: bad})
            assert file_store.upload_max_bytes() == default
            small_file = client.post("/upload", files={"file": ("x.bin", b"b" * 2048, "application/octet-stream")})
            small_image = client.post("/upload", files={"file": ("x.png", _TINY_PNG, "image/png")})
            assert small_file.status_code == 200, (bad, small_file.text)
            assert small_image.status_code == 200, (bad, small_image.text)
        too_big = client.post("/upload", files={"file": ("big.bin", b"\x00" * (default + 1), "application/octet-stream")})
        assert too_big.status_code == 413, too_big.text


def test_a_cap_that_is_not_above_zero_reads_as_the_default_and_is_logged_once(monkeypatch, caplog) -> None:
    default = file_store._UPLOAD_MAX_BYTES  # noqa: SLF001
    monkeypatch.setattr(tunables, "_warned_keys", set())
    with caplog.at_level(logging.WARNING, logger="brain.tunables"):
        for bad in (0, -2):
            tunables.set_values({_KEY: bad})
            caplog.clear()
            assert file_store.upload_max_bytes() == default
            assert file_store.upload_max_bytes() == default
            texts = [r.getMessage() for r in caplog.records]
            assert len(texts) == 1 and _KEY in texts[0] and repr(bad) in texts[0], (bad, texts)
        tunables.set_values({_KEY: 1024})
        caplog.clear()
        assert file_store.upload_max_bytes() == 1024
    assert caplog.records == []


# --- the read follows the body, not the cap (#531) ---------------------------------------------------------------------
# `file.read(n)` on a spooled upload takes `n` as a C `Py_ssize_t` (OverflowError above `sys.maxsize`) and, once the body is
# above the spool threshold (rolled to a disk file), allocates `n` bytes up front (MemoryError for a large `n`). The route
# used to read `cap + 1`, so a large cap made uploads answer 500 although the body was small. These tests use caps no read can
# honour, built from the constants (their values differ per platform); no test allocates a buffer of a cap's size, the reads
# are recorded instead.
_UNREADABLE_CAPS = (sys.maxsize - 1, sys.maxsize, sys.maxsize + 1)


def _record_reads(monkeypatch, *, forget_size: bool) -> list[int]:
    """Swap the UploadFile the multipart parser builds for one that records the size of every ``read`` (and can forget its size)."""
    sizes: list[int] = []

    class _RecordingUploadFile(UploadFile):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            if forget_size:
                self.size = None

        async def read(self, size: int = -1) -> bytes:
            sizes.append(size)
            return await super().read(size)

    monkeypatch.setattr(starlette.formparsers, "UploadFile", _RecordingUploadFile)
    return sizes


def _post_file(client: TestClient, body: bytes, name: str = "x.bin", media_type: str = "application/octet-stream"):
    return client.post("/upload", files={"file": (name, body, media_type)})


def _assert_the_cap_is_a_boundary(client: TestClient, sizes: list[int], cap: int) -> None:
    """A body one byte over the cap is refused, a body at the cap is accepted (in memory or rolled to disk)."""
    tunables.set_values({_KEY: cap})
    assert file_store.upload_max_bytes() == cap
    sizes.clear()
    over = _post_file(client, b"\x01" * (cap + 1))
    assert over.status_code == 413, (cap, over.text)
    sizes.clear()
    at = _post_file(client, b"\x02" * cap)
    assert at.status_code == 200, (cap, at.text)


def test_upload_reads_only_what_the_body_holds_whatever_the_cap(persona_dir: Path, monkeypatch) -> None:
    sizes = _record_reads(monkeypatch, forget_size=False)
    spool = MultiPartParser.spool_max_size
    client = _client(persona_dir)
    with client:
        # The route follows the setting, in memory and rolled to disk, so the phases below mean something.
        _assert_the_cap_is_a_boundary(client, sizes, 1024)
        _assert_the_cap_is_a_boundary(client, sizes, spool + 8)
        assert sizes == [spool + 8], sizes  # the body at the cap: one read, as large as the body
        bodies = (
            (b"b" * 2048, "x.bin", "application/octet-stream"),
            (_TINY_PNG, "x.png", "image/png"),
            (b"\x03" * (spool + 1), "big.bin", "application/octet-stream"),  # one byte above the spool threshold: rolled to disk
        )
        for cap in _UNREADABLE_CAPS:
            tunables.set_values({_KEY: cap})
            assert file_store.upload_max_bytes() == cap  # a cap no read can honour is still the cap in force
            for body, name, media_type in bodies:
                sizes.clear()
                response = _post_file(client, body, name, media_type)
                assert response.status_code == 200, (cap, name, response.text)
                assert sizes == [len(body)], (cap, name, sizes)  # asked for what the body holds, nothing more


def test_upload_with_an_unknown_size_reads_in_bounded_pieces(persona_dir: Path, monkeypatch) -> None:
    """An UploadFile with no size (starlette's multipart parser always sets one; other code may not) is read in bounded pieces."""
    sizes = _record_reads(monkeypatch, forget_size=True)
    spool = MultiPartParser.spool_max_size
    client = _client(persona_dir)
    with client:
        _assert_the_cap_is_a_boundary(client, sizes, 1024)
        _assert_the_cap_is_a_boundary(client, sizes, spool + 8)
        for cap in (sys.maxsize, sys.maxsize + 1):
            tunables.set_values({_KEY: cap})
            small = _post_file(client, b"b" * 2048)
            rolled = _post_file(client, b"\x03" * (spool + 1))
            assert small.status_code == 200, (cap, small.text)
            assert rolled.status_code == 200, (cap, rolled.text)
    assert sizes, "the route never read the file"
    assert max(sizes) <= shutil.COPY_BUFSIZE, max(sizes)  # no read is as large as a cap or a body


def _chunked_multipart(body: bytes, boundary: str) -> Iterator[bytes]:
    yield (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="x.bin"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    piece = 4096
    for start in range(0, len(body), piece):
        yield body[start : start + piece]
    yield f"\r\n--{boundary}--\r\n".encode()


def test_upload_in_a_chunked_transfer_is_bounded_the_same_way(persona_dir: Path) -> None:
    """A generator body is sent with Transfer-Encoding: chunked and no Content-Length; the answers do not change."""
    spool = MultiPartParser.spool_max_size
    boundary = "kindledboundary"
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}

    def post(body: bytes):
        return client.post("/upload", content=_chunked_multipart(body, boundary), headers=headers)

    client = _client(persona_dir)
    with client:
        tunables.set_values({_KEY: 1024})
        assert post(b"\x01" * 1025).status_code == 413
        assert post(b"\x02" * 1024).status_code == 200
        for cap in (sys.maxsize, sys.maxsize + 1):
            tunables.set_values({_KEY: cap})
            small = post(b"b" * 2048)
            rolled = post(b"\x03" * (spool + 1))
            assert small.status_code == 200, (cap, small.text)
            assert rolled.status_code == 200, (cap, rolled.text)


class _RecordingFile(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.sizes: list[int] = []

    def read(self, size: int | None = -1) -> bytes:
        self.sizes.append(-1 if size is None else size)
        return super().read(size)


def _read_capped(data: bytes, size: int | None, max_bytes: int) -> tuple[bytes, list[int]]:
    from brain.bridge import server

    file = _RecordingFile(data)
    raw = asyncio.run(server._read_upload_capped(UploadFile(file=file, size=size), max_bytes))  # noqa: SLF001
    return raw, file.sizes


@pytest.mark.parametrize("length", [0, 1, 1023, 1024, 1025, 4000])
def test_read_upload_capped_with_a_known_size_makes_one_read_no_larger_than_the_body(length: int) -> None:
    data = bytes(range(256)) * 16
    data = (data * 2)[:length]
    for cap in (1024, sys.maxsize - 1, sys.maxsize, sys.maxsize + 1):
        raw, sizes = _read_capped(data, length, cap)
        assert raw == data[: cap + 1], (cap, length)  # the whole body, or one byte over the cap
        assert sizes == [min(cap + 1, length)], (cap, length, sizes)


@pytest.mark.parametrize("length", [0, 1, 1024, 1025, 3 * shutil.COPY_BUFSIZE + 5])
def test_read_upload_capped_with_no_size_reads_bounded_pieces_and_returns_the_same_bytes(length: int) -> None:
    data = (bytes(range(256)) * (length // 256 + 1))[:length]
    for cap in (1024, shutil.COPY_BUFSIZE + 10, sys.maxsize - 1, sys.maxsize, sys.maxsize + 1):
        raw, sizes = _read_capped(data, None, cap)
        assert raw == data[: cap + 1], (cap, length)
        assert max(sizes) <= shutil.COPY_BUFSIZE, (cap, length, sizes)


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
