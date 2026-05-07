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

_ALLOWED_MEDIA_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)
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
            f"unsupported media_type {media_type!r}; "
            f"must be one of {sorted(_ALLOWED_MEDIA_TYPES)}"
        ) from exc


def _validate_sha(sha: str) -> None:
    """Raise ``ValueError`` unless ``sha`` is 64 lowercase hex chars."""
    if not _SHA256_HEX.fullmatch(sha):
        raise ValueError(
            f"image_sha must be 64 lowercase hex chars, got {sha!r}"
        )


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
            f"unsupported media_type {media_type!r}; "
            f"must be one of {sorted(_ALLOWED_MEDIA_TYPES)}"
        )
    sha = compute_sha(data)
    target = image_path(persona_dir, sha, media_type)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return ImageRecord(sha=sha, media_type=media_type, size_bytes=len(data))
    tmp = target.with_suffix(target.suffix + ".new")
    tmp.write_bytes(data)
    os.replace(tmp, target)
    return ImageRecord(sha=sha, media_type=media_type, size_bytes=len(data))


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
