"""Tests for brain.images — sha-addressable image storage helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from brain.images import (
    ImageRecord,
    compute_sha,
    image_path,
    media_type_to_ext,
    read_image_bytes,
    save_image_bytes,
)

# A minimal 1x1 transparent PNG. Plenty for round-trip tests.
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63606060600000000400015e36b8c80000000049454e44ae426082"
)
_TINY_PNG_SHA = hashlib.sha256(_TINY_PNG).hexdigest()


# ---------------------------------------------------------------------------
# compute_sha
# ---------------------------------------------------------------------------


def test_compute_sha_returns_64_hex() -> None:
    sha = compute_sha(_TINY_PNG)
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)


def test_compute_sha_is_deterministic() -> None:
    assert compute_sha(_TINY_PNG) == compute_sha(_TINY_PNG)


def test_compute_sha_distinct_for_distinct_bytes() -> None:
    assert compute_sha(b"a") != compute_sha(b"b")


# ---------------------------------------------------------------------------
# media_type_to_ext
# ---------------------------------------------------------------------------


def test_media_type_to_ext_png() -> None:
    assert media_type_to_ext("image/png") == "png"


def test_media_type_to_ext_jpeg() -> None:
    assert media_type_to_ext("image/jpeg") == "jpg"


def test_media_type_to_ext_webp() -> None:
    assert media_type_to_ext("image/webp") == "webp"


def test_media_type_to_ext_gif() -> None:
    assert media_type_to_ext("image/gif") == "gif"


def test_media_type_to_ext_unknown_raises() -> None:
    with pytest.raises(ValueError, match="media_type"):
        media_type_to_ext("application/pdf")


# ---------------------------------------------------------------------------
# image_path
# ---------------------------------------------------------------------------


def test_image_path_uses_persona_dir(tmp_path: Path) -> None:
    p = image_path(tmp_path, _TINY_PNG_SHA, "image/png")
    assert p == tmp_path / "images" / f"{_TINY_PNG_SHA}.png"


def test_image_path_validates_sha(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="image_sha"):
        image_path(tmp_path, "../etc/passwd", "image/png")


def test_image_path_rejects_short_sha(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="image_sha"):
        image_path(tmp_path, "a" * 63, "image/png")


# ---------------------------------------------------------------------------
# save_image_bytes
# ---------------------------------------------------------------------------


def test_save_image_bytes_returns_record(tmp_path: Path) -> None:
    record = save_image_bytes(tmp_path, _TINY_PNG, "image/png")
    assert isinstance(record, ImageRecord)
    assert record.sha == _TINY_PNG_SHA
    assert record.media_type == "image/png"
    assert record.size_bytes == len(_TINY_PNG)


def test_save_image_bytes_writes_file(tmp_path: Path) -> None:
    record = save_image_bytes(tmp_path, _TINY_PNG, "image/png")
    target = tmp_path / "images" / f"{record.sha}.png"
    assert target.exists()
    assert target.read_bytes() == _TINY_PNG


def test_save_image_bytes_creates_images_dir(tmp_path: Path) -> None:
    save_image_bytes(tmp_path, _TINY_PNG, "image/png")
    assert (tmp_path / "images").is_dir()


def test_save_image_bytes_dedupes_identical_content(tmp_path: Path) -> None:
    """Same bytes saved twice — single file, single sha, no .new residue."""
    r1 = save_image_bytes(tmp_path, _TINY_PNG, "image/png")
    r2 = save_image_bytes(tmp_path, _TINY_PNG, "image/png")
    assert r1.sha == r2.sha
    files = list((tmp_path / "images").iterdir())
    assert len(files) == 1
    assert all(not f.name.endswith(".new") for f in files)


def test_save_image_bytes_distinct_content_distinct_files(tmp_path: Path) -> None:
    save_image_bytes(tmp_path, _TINY_PNG, "image/png")
    save_image_bytes(tmp_path, b"different bytes", "image/png")
    files = list((tmp_path / "images").iterdir())
    assert len(files) == 2


def test_save_image_bytes_rejects_unsupported_media_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="media_type"):
        save_image_bytes(tmp_path, _TINY_PNG, "application/pdf")


def test_save_image_bytes_no_new_residue_on_success(tmp_path: Path) -> None:
    save_image_bytes(tmp_path, _TINY_PNG, "image/png")
    new_files = list((tmp_path / "images").glob("*.new"))
    assert new_files == []


# ---------------------------------------------------------------------------
# read_image_bytes
# ---------------------------------------------------------------------------


def test_read_image_bytes_round_trip(tmp_path: Path) -> None:
    record = save_image_bytes(tmp_path, _TINY_PNG, "image/png")
    out = read_image_bytes(tmp_path, record.sha, record.media_type)
    assert out == _TINY_PNG


def test_read_image_bytes_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_image_bytes(tmp_path, _TINY_PNG_SHA, "image/png")


def test_read_image_bytes_validates_sha(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="image_sha"):
        read_image_bytes(tmp_path, "../escape", "image/png")
