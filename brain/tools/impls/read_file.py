"""read_file tool — read-only, guarded, audited. Used only when the user asks."""
from __future__ import annotations

import base64
import difflib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from brain import tunables
from brain.images import media_type_to_ext, save_image_bytes, sniff_media_type
from brain.tools.impls import _read_cache

_FILE_READ_MAX_BYTES = tunables.register("files.read_max_bytes", 256 * 1024)
# Viewable-image support. The image branch runs BEFORE the 256KB text cap so
# a real photo (>256KB text cap, <= image_max_bytes) is returned as a viewable
# image rather than refused by the text cap (P0 red-team F-1 cap-ordering).
# image_max_bytes defaults >= the 20MB /upload cap so there is no
# upload-succeeds-but-read-refuses dead zone.
_FILE_IMAGE_TYPES = tunables.register(
    "files.image_types", ["image/png", "image/jpeg", "image/webp", "image/gif"]
)
_FILE_IMAGE_MAX_BYTES = tunables.register("files.image_max_bytes", 20 * 1024 * 1024)
# #126: PDFs are read as extracted text (pypdf, pure-Python). Cap matches /upload.
_FILE_PDF_MAX_BYTES = tunables.register("files.pdf_max_bytes", 20 * 1024 * 1024)
_PDF_MAGIC = b"%PDF-"
_SUGGEST_MAX = 10
_DEFAULT_HEAD_LINES = 400


def _file_read_max_bytes() -> int:
    return tunables.get_tunable("files.read_max_bytes", _FILE_READ_MAX_BYTES)


def _image_types() -> list:
    return tunables.get_tunable("files.image_types", _FILE_IMAGE_TYPES)


def _image_max_bytes() -> int:
    return tunables.get_tunable("files.image_max_bytes", _FILE_IMAGE_MAX_BYTES)


def _pdf_max_bytes() -> int:
    return tunables.get_tunable("files.pdf_max_bytes", _FILE_PDF_MAX_BYTES)


def _window_lines(
    lines: list[str], *, max_lines: int | None, offset: int
) -> tuple[str, int, int, bool]:
    """Apply the ranged-read / head-cap policy shared by text and PDF reads.

    Returns (sliced_text, start, window_len, truncated).
    """
    total = len(lines)
    start = max(0, int(offset or 0))
    if max_lines is not None:
        window = lines[start : start + max(0, int(max_lines))]
        truncated = (start + len(window)) < total or start > 0
    elif total > _DEFAULT_HEAD_LINES:
        window = lines[:_DEFAULT_HEAD_LINES]
        truncated = True
    else:
        window = lines[start:] if start else lines
        truncated = start > 0
    return "".join(window), start, len(window), truncated


def _read_pdf(
    p: Path, *, raw: str, persona_dir: Path, max_lines: int | None, offset: int
) -> dict:
    """#126: return a PDF's text layer, page-marked and windowed like a text file.

    No text layer (scan / image-only) -> an honest note with the page count.
    Any pypdf failure -> fail-soft error dict. Page rendering is out of scope.
    """
    size = p.stat().st_size
    cap = _pdf_max_bytes()
    if size > cap:
        _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=size, ok=False, error="too large")
        return {"error": f"PDF too large ({size} bytes > {cap} cap) — not shown"}

    _dedup_key = os.path.normcase(os.path.realpath(str(p)))
    if _read_cache.seen_recently(_dedup_key):
        _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=0, ok=True, error="deduped")
        return {
            "path": str(p),
            "deduped": True,
            "note": "you already read this file moments ago this turn — its content is above.",
        }
    _read_cache.mark(_dedup_key)

    try:
        from pypdf import PdfReader  # noqa: PLC0415 — only imported on the PDF path

        reader = PdfReader(str(p))
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # owner-password-only PDFs open with the empty user password
            except Exception:  # noqa: BLE001
                pass
        page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:  # noqa: BLE001
        _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=size, ok=False, error=f"pdf: {exc}")
        return {"error": f"could not read PDF: {exc}"}

    pages = len(page_texts)
    if not any(page_texts):
        _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=size, ok=True, error="pdf: no text layer")
        return {
            "path": str(p),
            "pages": pages,
            "note": f"PDF has no extractable text ({pages} page(s) — scanned or image-only); not shown",
        }

    joined = "".join(
        (f"--- page {i} ---\n" if i > 1 else "") + text + "\n" for i, text in enumerate(page_texts, 1)
    )
    lines = joined.splitlines(keepends=True)
    sliced, start, window_len, truncated = _window_lines(lines, max_lines=max_lines, offset=offset)
    _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=size, ok=True)
    out: dict = {"path": str(p), "content": sliced, "pages": pages, "total_lines": len(lines)}
    if truncated:
        out["truncated"] = True
        out["note"] = (
            f"showing lines {start}-{start + window_len} of {len(lines)} across {pages} page(s); "
            "pass offset/max_lines to read more"
        )
    return out


def _suggest(target: Path) -> list[str]:
    """Case-insensitive / fuzzy filename suggestions from the target's parent dir.
    Keeps the model from crawling parent dirs when it guesses a wrong name."""
    parent = target.parent
    try:
        if not parent.is_dir():
            return []
        names = [c.name for c in parent.iterdir() if c.is_file()]
    except OSError:
        return []
    stem = target.name.casefold()
    # substring matches first, then close fuzzy matches, deduped, capped.
    subs = [n for n in names if stem in n.casefold() or n.casefold() in stem]
    fuzzy = difflib.get_close_matches(target.name, names, n=_SUGGEST_MAX, cutoff=0.6)
    out: list[str] = []
    for n in [*subs, *fuzzy]:
        if n not in out:
            out.append(n)
    return out[:_SUGGEST_MAX]


def _audit(
    persona_dir: Path,
    *,
    tool: str,
    path: str,
    resolved: str,
    bytes_: int,
    ok: bool,
    error: str | None = None,
) -> None:
    try:
        persona_dir.mkdir(parents=True, exist_ok=True)
        with (persona_dir / "file_access.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "tool": tool,
                        "path": path,
                        "resolved_path": resolved,
                        "bytes": bytes_,
                        "ok": ok,
                        "error": error,
                    }
                )
                + "\n"
            )
    except Exception:  # noqa: BLE001
        pass


def read_file(path: str, *, persona_dir: Path, max_lines: int | None = None,
              offset: int = 0, **_) -> dict:
    """Read a text file's contents (read-only). Refuses files over the size cap.

    max_lines: optional — return at most this many lines (ranged read).
    offset: 0-based line to start reading from (used with max_lines).
    Large files without max_lines are head-capped at _DEFAULT_HEAD_LINES.
    """
    raw = path
    try:
        p = Path(os.path.expandvars(os.path.expanduser(path))).resolve()
    except Exception as exc:  # noqa: BLE001
        _audit(persona_dir, tool="read_file", path=raw, resolved="", bytes_=0, ok=False, error=str(exc))
        return {"error": f"bad path: {exc}"}

    if not p.exists() or not p.is_file():
        suggestions = _suggest(p)
        _audit(
            persona_dir,
            tool="read_file",
            path=raw,
            resolved=str(p),
            bytes_=0,
            ok=False,
            error="not a readable file",
        )
        return {"error": f"not a readable file: {p}", "did_you_mean": suggestions}

    # Image branch — MUST run BEFORE the text size-cap check below so a real
    # photo (larger than the 256KB text cap but within files.image_max_bytes)
    # is returned as a viewable image, not refused by the text cap (P0 red-team
    # F-1 cap-ordering). Sniff the media type from a small magic-byte prefix.
    try:
        with p.open("rb") as _fh:
            _prefix = _fh.read(32)
    except OSError:
        _prefix = b""
    sniffed = sniff_media_type(_prefix)
    if sniffed is not None and sniffed in _image_types():
        img_size = p.stat().st_size
        img_cap = _image_max_bytes()
        if img_size > img_cap:
            _audit(
                persona_dir,
                tool="read_file",
                path=raw,
                resolved=str(p),
                bytes_=img_size,
                ok=False,
                error="image too large",
            )
            return {"error": f"image too large ({img_size} bytes > {img_cap} cap) — not shown"}
        _dedup_key = os.path.normcase(os.path.realpath(str(p)))
        if _read_cache.seen_recently(_dedup_key):
            _audit(
                persona_dir, tool="read_file", path=raw, resolved=str(p),
                bytes_=0, ok=True, error="deduped",
            )
            return {
                "path": str(p),
                "deduped": True,
                "note": "you already read this file moments ago this turn — its content is above.",
            }
        _read_cache.mark(_dedup_key)
        try:
            data = p.read_bytes()
        except OSError as exc:
            _audit(
                persona_dir, tool="read_file", path=raw, resolved=str(p),
                bytes_=img_size, ok=False, error=str(exc),
            )
            return {"error": f"read failed: {exc}"}
        # Content-address the opened image into the persona's image store so a
        # durable, hash-bearing handle (images/<sha>.<ext>) exists for it. This
        # is what lets a normal memory later bind to the image by content hash:
        # the engine surfaces this rel_path into the durable buffer (see
        # engine._persist_turn + the stored_image_path invocation field). The
        # write is content-addressed, idempotent, dedup'd, and confined to
        # <persona_dir>/images/ (brain.images validates the sha — no traversal).
        # NEVER put base64 in the stored_image block (red-team G5 / C15).
        stored_image: dict | None = None
        try:
            rec = save_image_bytes(persona_dir, data, sniffed)
            ext = media_type_to_ext(rec.media_type)
            stored_image = {
                "sha": rec.sha,
                "media_type": rec.media_type,
                "rel_path": f"images/{rec.sha}.{ext}",
            }
        except (ValueError, OSError):
            # Content-addressing is best-effort: a store failure must never
            # break the viewable-image return (she must still SEE the pixels).
            stored_image = None
        # Audit records the size + the content-addressed rel_path, never the
        # base64 (red-team G5 / C15). The rel_path is a content hash (metadata),
        # not image bytes.
        _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=img_size, ok=True)
        result: dict = {
            "path": str(p),
            "image": {
                "media_type": sniffed,
                "data_b64": base64.b64encode(data).decode("ascii"),
                "size_bytes": img_size,
            },
        }
        if stored_image is not None:
            result["stored_image"] = stored_image
        return result

    # PDF branch (#126) — also BEFORE the text cap: a PDF is binary-large but
    # text-small, same cap-ordering reasoning as the image branch above.
    if _prefix.startswith(_PDF_MAGIC):
        return _read_pdf(p, raw=raw, persona_dir=persona_dir, max_lines=max_lines, offset=offset)

    cap = _file_read_max_bytes()
    size = p.stat().st_size
    if size > cap:
        _audit(
            persona_dir,
            tool="read_file",
            path=raw,
            resolved=str(p),
            bytes_=size,
            ok=False,
            error="too large",
        )
        return {"error": f"file too large ({size} bytes > {cap} cap) — not shown"}

    # Platform-correct case handling: normcase lowercases on Windows (case-
    # insensitive FS) and is a no-op on macOS/Linux, where realpath already
    # canonicalises case on the case-insensitive macOS FS and keeps genuinely
    # distinct files distinct on case-sensitive Linux. Do NOT casefold here — on
    # Linux that would collide two different files (Notes.md vs notes.md).
    _dedup_key = os.path.normcase(os.path.realpath(str(p)))
    if _read_cache.seen_recently(_dedup_key):
        _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=0, ok=True, error="deduped")
        return {
            "path": str(p),
            "deduped": True,
            "note": "you already read this file moments ago this turn — its content is above.",
        }
    _read_cache.mark(_dedup_key)

    try:
        data = p.read_bytes()
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            _audit(
                persona_dir,
                tool="read_file",
                path=raw,
                resolved=str(p),
                bytes_=size,
                ok=True,
                error="binary",
            )
            return {"path": str(p), "note": f"(binary file, {size} bytes — not shown)"}

        lines = content.splitlines(keepends=True)
        total = len(lines)
        sliced, start, window_len, truncated = _window_lines(
            lines, max_lines=max_lines, offset=offset
        )
        _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=size, ok=True)
        out: dict = {"path": str(p), "content": sliced, "total_lines": total}
        if truncated:
            out["truncated"] = True
            out["note"] = (
                f"showing lines {start}-{start + window_len} of {total}; "
                "pass offset/max_lines to read more"
            )
        return out

    except OSError as exc:
        _audit(
            persona_dir,
            tool="read_file",
            path=raw,
            resolved=str(p),
            bytes_=size,
            ok=False,
            error=str(exc),
        )
        return {"error": f"read failed: {exc}"}
