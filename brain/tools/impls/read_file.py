"""read_file tool — read-only, guarded, audited. Used only when the user asks."""
from __future__ import annotations

import base64
import codecs
import difflib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from brain import tunables
from brain.files.path_identity import unusable
from brain.files.read_guard import PROTECTED, is_read_denied
from brain.health.json_io import append_jsonl
from brain.images import media_type_to_ext, save_image_bytes, sniff_media_type
from brain.tools.impls import _read_cache

# #590: this cap no longer guards the context. A text file is read a window at a
# time (_DEFAULT_HEAD_LINES lines, at most _WINDOW_MAX_CHARS characters), streamed
# so a range read never holds the whole file, so the cap only stops pathological
# files (each read scans the whole file to count lines and check it is UTF-8). It
# matches the 20 MiB upload, image and PDF caps, so any file the user can share opens.
_FILE_READ_MAX_BYTES = tunables.register(
    "files.read_max_bytes", 20 * 1024 * 1024,
    description="Largest text file opened, in bytes; text is read 400 lines at a time, so this only stops huge files.",
)
# Viewable-image support. The image branch runs BEFORE the text cap so a real
# photo (over a text cap set lower than image_max_bytes) is returned as a viewable
# image rather than refused by the text cap (P0 red-team F-1 cap-ordering).
# image_max_bytes defaults to the same 20 MiB as files.upload_max_bytes (a
# separate setting), so an image uploaded at the default is not refused here.
_FILE_IMAGE_TYPES = tunables.register(
    "files.image_types", ["image/png", "image/jpeg", "image/webp", "image/gif"],
    description="Kinds of image the kindled can look at when it opens an image file.",
)
_FILE_IMAGE_MAX_BYTES = tunables.register(
    "files.image_max_bytes", 20 * 1024 * 1024,
    description="Largest image the kindled will look at, in bytes.",
)
# #126: PDFs are read as extracted text (pypdf, pure-Python). The cap defaults to the upload cap's 20 MiB.
_FILE_PDF_MAX_BYTES = tunables.register(
    "files.pdf_max_bytes", 20 * 1024 * 1024,
    description="Largest PDF the kindled will read, in bytes.",
)
_PDF_MAGIC = b"%PDF-"
_SUGGEST_MAX = 10
_DEFAULT_HEAD_LINES = 400
# #595: the most text one read returns, counted as the characters of its JSON-encoded form
# (what the MCP handler sends: quotes, backslashes and control characters grow when
# escaped). The Claude CLI passes a tool result to the model only up to 50,000 characters
# and persists anything longer behind a 2 KB preview that chat cannot open (measured on
# claude 2.1.321, docs/cli-provider-capabilities.md). 40,000 leaves room under that cap for
# the path, the note and the keys.
_WINDOW_MAX_CHARS = 40_000
_READ_CHUNK_BYTES = 64 * 1024
# #595: the tool knows only that it returned this recently, not that the result reached the
# model (an over-size result never did), so the note says no more than that.
_DEDUP_NOTE = (
    "this part of the file was already returned less than 90 seconds ago, so it is not sent again; "
    "read a different part with offset/max_lines, or ask again shortly to re-read it."
)
_DEDUP_NOTE_IMAGE = "this image was already returned less than 90 seconds ago, so it is not sent again."
# The characters str.splitlines() breaks on, so streamed lines match it exactly.
_LINE_BREAKS = "\n\r\x0b\x0c\x1c\x1d\x1e\x85  "


def _file_read_max_bytes() -> int:
    return tunables.get_tunable("files.read_max_bytes", _FILE_READ_MAX_BYTES)


def _image_types() -> list:
    return tunables.get_tunable("files.image_types", _FILE_IMAGE_TYPES)


def _image_max_bytes() -> int:
    return tunables.get_tunable("files.image_max_bytes", _FILE_IMAGE_MAX_BYTES)


def _pdf_max_bytes() -> int:
    return tunables.get_tunable("files.pdf_max_bytes", _FILE_PDF_MAX_BYTES)


def _encoded_len(text: str) -> int:
    """Characters ``text`` takes inside the JSON the MCP handler sends (ensure_ascii=False)."""
    return len(json.dumps(text, ensure_ascii=False)) - 2


def _encoded_prefix(text: str, room: int) -> str:
    """The longest prefix of ``text`` whose encoded form fits in ``room`` characters."""
    lo, hi = 0, min(len(text), room)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _encoded_len(text[:mid]) <= room:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


class _Window:
    """The ranged-read policy shared by text and PDF reads, fed one line at a time.

    Keeps lines [offset, offset + max_lines) — max_lines defaults to _DEFAULT_HEAD_LINES —
    up to _WINDOW_MAX_CHARS characters, and counts every line fed. A first line longer
    than the budget is cut to it; any later line that would pass it ends the window.
    """

    def __init__(self, *, max_lines: int | None, offset: int) -> None:
        self.start = max(0, int(offset or 0))
        count = _DEFAULT_HEAD_LINES if max_lines is None else max(0, int(max_lines))
        self.end = self.start + count
        self.budget = _WINDOW_MAX_CHARS
        self.parts: list[str] = []
        self.used = 0
        self.total = 0
        self.full = False
        self.cut_line: int | None = None

    def add(self, line: str, *, cut: bool = False) -> None:
        index = self.total
        self.total += 1
        if self.full or not self.start <= index < self.end:
            return
        room = self.budget - self.used
        size = _encoded_len(line)
        if size > room:
            self.full = True
            if self.parts:
                return
            line, cut = _encoded_prefix(line, room), True
            size = _encoded_len(line)
        self.parts.append(line)
        self.used += size
        if cut:
            self.cut_line = index
            self.full = True

    @property
    def dedup_suffix(self) -> str:
        """Appended to the read-dedup key, so reading the next range of a file in the same
        turn is not mistaken for re-reading the range already shown (#590)."""
        return f"#lines={self.start}:{self.end}"

    @property
    def truncated(self) -> bool:
        return self.start > 0 or self.start + len(self.parts) < self.total or self.cut_line is not None

    def note(self, *, pages: int | None = None) -> str:
        across = f" across {pages} page(s)" if pages is not None else ""
        if self.start >= self.total:
            return f"offset {self.start} is past the end: there are {self.total} lines{across}"
        where = f"of {self.total}{across}"
        text = f"showing lines {self.start}-{self.start + len(self.parts)} {where}"
        if self.cut_line is not None:
            text += f"; line {self.cut_line} is longer than {self.budget} characters and was cut short"
        elif self.full:
            text += f"; stopped at {self.budget} characters"
        return text + "; pass offset/max_lines to read more"


def _stream_lines(fh, limit: int):
    """Yield (line, cut) for each line of a UTF-8 byte stream, as str.splitlines(keepends=True)
    would split the decoded text, holding at most ``limit`` characters of any line.

    Reads _READ_CHUNK_BYTES at a time, so memory is bounded by the chunk and ``limit``,
    never the file. Invalid UTF-8 anywhere raises UnicodeDecodeError.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    line, cut, carry = "", False, ""
    while True:
        chunk = fh.read(_READ_CHUNK_BYTES)
        text = carry + decoder.decode(chunk, final=not chunk)
        carry = ""
        if chunk and text.endswith("\r"):
            carry, text = "\r", text[:-1]  # it may pair with a "\n" in the next chunk
        for part in text.splitlines(keepends=True):
            if not cut:
                room = limit - len(line)
                if len(part) > room:
                    line, cut = line + part[:room], True
                else:
                    line += part
            if part[-1] in _LINE_BREAKS:
                yield line, cut
                line, cut = "", False
        if not chunk:
            if line or cut:
                yield line, cut
            return


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

    window = _Window(max_lines=max_lines, offset=offset)
    _dedup_key = os.path.normcase(os.path.realpath(str(p))) + window.dedup_suffix
    if _read_cache.seen_recently(_dedup_key):
        _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=0, ok=True, error="deduped")
        return {
            "path": str(p),
            "deduped": True,
            "note": _DEDUP_NOTE,
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
    for line in joined.splitlines(keepends=True):
        window.add(line)
    _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=size, ok=True)
    out: dict = {"path": str(p), "content": "".join(window.parts), "pages": pages, "total_lines": window.total}
    if window.truncated:
        out["truncated"] = True
        out["note"] = window.note(pages=pages)
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
        append_jsonl(
            persona_dir / "file_access.jsonl",
            {
                "ts": datetime.now(UTC).isoformat(),
                "tool": tool,
                "path": path,
                "resolved_path": resolved,
                "bytes": bytes_,
                "ok": ok,
                "error": error,
            },
            on_error="log",
            nonfinite="null",
        )
    except Exception:  # noqa: BLE001
        pass


def read_file(path: str, *, persona_dir: Path, max_lines: int | None = None,
              offset: int = 0, **_) -> dict:
    """Read a text file's contents (read-only). Refuses files over the size cap.

    max_lines: optional — return at most this many lines (ranged read).
    offset: 0-based line to start reading from.
    Without max_lines a read shows _DEFAULT_HEAD_LINES lines from offset; any read
    shows at most _WINDOW_MAX_CHARS characters (#590).
    """
    raw = path
    try:
        p = Path(os.path.expandvars(os.path.expanduser(path))).resolve()
    except Exception as exc:  # noqa: BLE001
        _audit(persona_dir, tool="read_file", path=raw, resolved="", bytes_=0, ok=False, error=str(exc))
        return {"error": f"bad path: {exc}"}
    if (why := unusable(p)) is not None:  # #695: too long, no permission, a loop: refuse before anything stats it
        _audit(persona_dir, tool="read_file", path=raw, resolved="", bytes_=0, ok=False, error=why)
        return {"error": f"bad path: {why}"}

    if is_read_denied(p, persona_dir=persona_dir):
        _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=0, ok=False, error=PROTECTED)
        return {"error": f"refused: {p} is in a {PROTECTED}"}

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
    # photo (larger than the text cap but within files.image_max_bytes)
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
                "note": _DEDUP_NOTE_IMAGE,
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
    window = _Window(max_lines=max_lines, offset=offset)
    _dedup_key = os.path.normcase(os.path.realpath(str(p))) + window.dedup_suffix
    if _read_cache.seen_recently(_dedup_key):
        _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=0, ok=True, error="deduped")
        return {
            "path": str(p),
            "deduped": True,
            "note": _DEDUP_NOTE,
        }
    _read_cache.mark(_dedup_key)

    try:
        # #590: stream the file a chunk at a time; only the window's lines are kept.
        # The whole file is still scanned, to count its lines and to call it binary
        # if any of it is not UTF-8, as the old whole-file decode did.
        try:
            with p.open("rb") as fh:
                for line, cut in _stream_lines(fh, window.budget):
                    window.add(line, cut=cut)
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

        _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=size, ok=True)
        out: dict = {"path": str(p), "content": "".join(window.parts), "total_lines": window.total}
        if window.truncated:
            out["truncated"] = True
            out["note"] = window.note()
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
