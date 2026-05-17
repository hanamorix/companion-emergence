"""Image storage helpers for multimodal personas.

Content-addressable layout: ``<persona_dir>/images/<sha256>.<ext>``.

The bridge ``/upload`` endpoint hands bytes to :func:`save_image_bytes`;
the provider layer reads them back via :func:`image_path` /
:func:`read_image_bytes` when composing multimodal turns. Buffers,
memories, and soul records reference images by sha only — no bytes are
inlined anywhere off-disk.

Validation:

* sha values are 64 lowercase hex characters; anything else raises
  ``ValueError`` (defends against path traversal and untrusted input).
* media_type must be one of the four whitelisted MIME types — same set
  as :data:`brain.bridge.chat._ALLOWED_MEDIA_TYPES`.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

_ALLOWED_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
_MEDIA_TYPE_TO_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ImageRecord:
    """Result of a successful image save.

    Attributes
    ----------
    sha:
        64-character lowercase hex sha256 of the image bytes.
    media_type:
        Validated MIME type the file was saved as.
    size_bytes:
        Length of the saved file's bytes (== len(input)).
    """

    sha: str
    media_type: str
    size_bytes: int


def compute_sha(data: bytes) -> str:
    """Compute a 64-char lowercase hex sha256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def media_type_to_ext(media_type: str) -> str:
    """Map MIME type → filesystem extension.

    Raises ``ValueError`` on unsupported media types.
    """
    try:
        return _MEDIA_TYPE_TO_EXT[media_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported media_type {media_type!r}; must be one of {sorted(_ALLOWED_MEDIA_TYPES)}"
        ) from exc


def _validate_sha(sha: str) -> None:
    """Raise ``ValueError`` unless ``sha`` is 64 lowercase hex chars."""
    if not _SHA256_HEX.fullmatch(sha):
        raise ValueError(f"image_sha must be 64 lowercase hex chars, got {sha!r}")


def image_path(persona_dir: Path, sha: str, media_type: str) -> Path:
    """Return the canonical on-disk path for ``sha`` under ``persona_dir``.

    Does not check existence — use :func:`read_image_bytes` for that.
    """
    _validate_sha(sha)
    ext = media_type_to_ext(media_type)
    return persona_dir / "images" / f"{sha}.{ext}"


def save_image_bytes(
    persona_dir: Path,
    data: bytes,
    media_type: str,
) -> ImageRecord:
    """Save ``data`` content-addressably under ``persona_dir/images/``.

    Atomic via ``.new`` + ``os.replace``. If the target already exists with
    matching sha (deduplication), no write is performed and the existing
    record is returned.

    Raises
    ------
    ValueError
        If ``media_type`` is not in the supported set.
    """
    if media_type not in _ALLOWED_MEDIA_TYPES:
        raise ValueError(
            f"unsupported media_type {media_type!r}; must be one of {sorted(_ALLOWED_MEDIA_TYPES)}"
        )
    # Defense in depth: the bridge /upload endpoint already sniffs at the
    # network boundary, but library callers (migration tools, internal
    # workers, tests) shouldn't be trusted to honour the contract. If the
    # bytes don't match a known signature — or don't match the declared
    # media_type — refuse before writing anything to disk.
    sniffed = sniff_media_type(data)
    if sniffed is None:
        raise ValueError(
            "image bytes match no supported signature; "
            "expected one of PNG, JPEG, WebP, GIF magic bytes"
        )
    if sniffed != media_type:
        raise ValueError(f"declared media_type {media_type!r} but bytes' signature is {sniffed!r}")
    sha = compute_sha(data)
    target = image_path(persona_dir, sha, media_type)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return ImageRecord(sha=sha, media_type=media_type, size_bytes=len(data))
    # Unique tmp path per writer so identical concurrent uploads of the
    # same sha don't race on a shared `<sha>.<ext>.new` file. The pid +
    # uuid suffix is enough to disambiguate threads + processes; final
    # `os.replace` is atomic so the target reflects exactly one writer.
    import uuid

    tmp = target.with_suffix(f"{target.suffix}.{os.getpid()}.{uuid.uuid4().hex[:8]}.new")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)
    except (FileNotFoundError, OSError):
        # If a concurrent writer beat us to the punch, the target should
        # now exist with the same sha (content-addressed). Fall through
        # to a final-state check before re-raising.
        if target.exists():
            tmp.unlink(missing_ok=True)
            return ImageRecord(sha=sha, media_type=media_type, size_bytes=len(data))
        tmp.unlink(missing_ok=True)
        raise
    return ImageRecord(sha=sha, media_type=media_type, size_bytes=len(data))


# Magic-byte signatures for the four allowed image types. Trusting the
# multipart `Content-Type` header alone lets a local client store
# arbitrary bytes under an image extension; sniffing closes that gap.
_IMAGE_MAGIC_BYTES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def sniff_media_type(data: bytes) -> str | None:
    """Return the inferred media_type for ``data`` or None if unknown.

    Recognises PNG, JPEG, GIF, and WebP (the four allowed types).
    Unknown/corrupt bytes return None — the caller should treat that as
    "reject this upload at the boundary."
    """
    for prefix, media_type in _IMAGE_MAGIC_BYTES:
        if data.startswith(prefix):
            return media_type
    # WebP: 'RIFF' + 4 size bytes + 'WEBP'
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


_EXT_TO_MEDIA_TYPE = {ext: mt for mt, ext in _MEDIA_TYPE_TO_EXT.items()}


def media_type_for_sha(persona_dir: Path, sha: str) -> str:
    """Look up the on-disk file for ``sha`` and return its media_type.

    Walks the allowed extensions; first match wins. Useful when a caller
    only has the sha (e.g. from a chat request body) and needs to
    construct an ImageBlock without round-tripping the media_type.

    Raises
    ------
    ValueError
        If ``sha`` is malformed.
    FileNotFoundError
        If no file exists for the sha under any allowed extension.
    """
    _validate_sha(sha)
    images_dir = persona_dir / "images"
    for ext, media_type in _EXT_TO_MEDIA_TYPE.items():
        if (images_dir / f"{sha}.{ext}").exists():
            return media_type
    raise FileNotFoundError(f"no image found for sha {sha} in {images_dir}")


def read_image_bytes(persona_dir: Path, sha: str, media_type: str) -> bytes:
    """Read image bytes for ``sha`` from ``persona_dir/images/``.

    Raises
    ------
    ValueError
        If ``sha`` is malformed or ``media_type`` is unsupported.
    FileNotFoundError
        If the file does not exist on disk.
    """
    target = image_path(persona_dir, sha, media_type)
    if not target.exists():
        raise FileNotFoundError(f"no image at {target}")
    return target.read_bytes()
