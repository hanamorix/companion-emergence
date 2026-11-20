"""One-time repair of the NUL runs an unclean stop left in JSONL files (#618).

An append that was not fsynced could be cut short by a power loss or a hard kill, and the filesystem then handed back a
run of ``\\0`` bytes where the tail should be. Such a run swallows the record after it (the reader sees one garbage
line). This startup migration walks the persona folder, and for each ``.jsonl`` file (or rotated ``.jsonl.N``) that
holds a NUL byte it keeps a ``.pre-618`` copy of the original and rewrites the file without the NULs, putting one
newline where a run sat in the middle of a line so the record after it is whole again.

Everything streams in ``chunk_size`` pieces (these logs can be large); the rewrite and the copy are atomic saves done
under the file's own lock, so an append waits instead of landing in a file about to be replaced. A file that cannot be
read or written, or a folder that cannot be listed, is listed in ``failed`` and no state file is written, so the next
start tries again.

The file's ``file_lock(path)`` holds off the writers that go through ``append_jsonl``'s default lock. The emotion
engine's change logs and queue under the top-level ``<persona_dir>/state/`` are appended under the engine's own
``state_lock`` (and the queue is read by line position), so that folder is left to the engine and is not walked.
Symlinked files are skipped, and a rewritten file and its backup keep the original's permission bits. State file
``<persona_dir>/jsonl_nul_repair_state.json`` makes a finished repair a no-op.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from brain.health.json_io import Recovery, read_json, write_json
from brain.utils.atomic_write import save_stream_atomic
from brain.utils.file_lock import file_lock

logger = logging.getLogger(__name__)

STATE_FILENAME = "jsonl_nul_repair_state.json"
BACKUP_SUFFIX = ".pre-618"

_ROTATED = re.compile(r".+\.jsonl\.\d+")
_NL = 0x0A
_NUL = re.compile(rb"\x00")
_NOT_NUL = re.compile(rb"[^\x00]")


@dataclass(frozen=True)
class FileRepair:
    runs: int
    bytes_removed: int
    backup: str  # posix path relative to the persona folder


@dataclass(frozen=True)
class RepairReport:
    scanned: int
    repaired: dict[str, FileRepair]
    failed: list[str]
    done: bool


def _open_for_read(path: Path) -> BinaryIO:
    """The one place this module opens a file for reading (a seam for tests)."""
    return path.open("rb")


def _is_candidate(name: str) -> bool:
    if BACKUP_SUFFIX in name or name.endswith(".gz"):
        return False
    return name.endswith(".jsonl") or _ROTATED.fullmatch(name) is not None


def _has_nul(path: Path, chunk_size: int) -> bool:
    with _open_for_read(path) as f:
        while chunk := f.read(chunk_size):
            if b"\0" in chunk:
                return True
    return False


class _NulFilter:
    """Streams bytes through, dropping NUL runs; the state carries across chunks."""

    def __init__(self, out: BinaryIO) -> None:
        self._out = out
        self._last: int | None = None  # last byte written; None until something is
        self._before: int | None = None  # the byte before the current run
        self._in_run = False
        self._run_len = 0
        self.runs = 0
        self.bytes_removed = 0

    def _end_run(self, following: int | None, *, at_eof: bool) -> None:
        self.runs += 1
        self.bytes_removed += self._run_len
        self._in_run = False
        # A newline only where a run sat inside a line: not at the file start, not when a line break already borders it.
        if self._before is not None and self._before != _NL and (at_eof or following != _NL):
            self._out.write(b"\n")
            self._last = _NL

    def feed(self, chunk: bytes) -> None:
        pos, size = 0, len(chunk)
        while pos < size:
            if self._in_run:
                m = _NOT_NUL.search(chunk, pos)
                stop = size if m is None else m.start()
                self._run_len += stop - pos
                pos = stop
                if pos < size:
                    self._end_run(chunk[pos], at_eof=False)
                continue
            m = _NUL.search(chunk, pos)
            idx = -1 if m is None else m.start()
            end = size if m is None else idx
            if end > pos:
                self._out.write(chunk[pos:end])
                self._last = chunk[end - 1]
            if idx < 0:
                break
            self._in_run, self._run_len, self._before = True, 0, self._last
            pos = idx

    def finish(self) -> None:
        if self._in_run:
            self._end_run(None, at_eof=True)


def _backup_path(path: Path) -> Path:
    candidate = path.with_name(path.name + BACKUP_SUFFIX)
    n = 0
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{path.name}{BACKUP_SUFFIX}.{n}")
    return candidate


def _repair_file(path: Path, chunk_size: int) -> tuple[int, int, Path]:
    """Back up then rewrite ``path`` without its NULs, under its lock. Returns (runs, bytes_removed, backup)."""
    stats = {"runs": 0, "removed": 0}

    def copy_original(out: BinaryIO) -> None:
        with _open_for_read(path) as src:
            while chunk := src.read(chunk_size):
                out.write(chunk)

    def rewrite(out: BinaryIO) -> None:
        flt = _NulFilter(out)
        with _open_for_read(path) as src:
            while chunk := src.read(chunk_size):
                flt.feed(chunk)
        flt.finish()
        stats["runs"], stats["removed"] = flt.runs, flt.bytes_removed

    with file_lock(path):
        mode = stat.S_IMODE(os.stat(path).st_mode)
        backup = _backup_path(path)
        save_stream_atomic(backup, copy_original, mode=mode)
        try:
            save_stream_atomic(path, rewrite, mode=mode)
        except OSError:
            backup.unlink(missing_ok=True)  # a copy of a file that did not change
            raise
    return stats["runs"], stats["removed"], backup


def _state_path(persona_dir: Path) -> Path:
    return persona_dir / STATE_FILENAME


def should_run_jsonl_nul_repair(persona_dir: Path) -> bool:
    """False only when the state file reads as an object with ``"done": true``."""
    try:
        state = read_json(
            _state_path(persona_dir), default=lambda: None, recover=Recovery.CONTENT, expect=dict
        )
    except Exception:  # noqa: BLE001 - an unreadable state means the repair has not provably finished
        return True
    return not (isinstance(state, dict) and state.get("done") is True)


def run_jsonl_nul_repair(persona_dir: Path, *, chunk_size: int = 1 << 20) -> RepairReport:
    scanned = 0
    repaired: dict[str, FileRepair] = {}
    failed: list[str] = []

    def unlistable(exc: OSError) -> None:
        folder = Path(exc.filename) if exc.filename else persona_dir
        try:
            rel = folder.relative_to(persona_dir).as_posix()
        except ValueError:
            rel = folder.as_posix()
        logger.warning("jsonl_nul_repair: cannot list %s: %s", rel, exc)
        failed.append(rel)

    for root, dirs, files in os.walk(persona_dir, onerror=unlistable):
        dirs.sort()
        if Path(root) == persona_dir and "state" in dirs:
            dirs.remove("state")  # the emotion engine's own logs and queue: its state_lock, not ours
        for name in sorted(files):
            if not _is_candidate(name):
                continue
            path = Path(root) / name
            if os.path.islink(path):
                continue  # never follow or replace a link
            rel = path.relative_to(persona_dir).as_posix()
            scanned += 1
            try:
                if not _has_nul(path, chunk_size):
                    continue
                runs, removed, backup = _repair_file(path, chunk_size)
            except OSError as exc:
                logger.warning("jsonl_nul_repair: cannot repair %s: %s", rel, exc)
                failed.append(rel)
                continue
            backup_rel = backup.relative_to(persona_dir).as_posix()
            repaired[rel] = FileRepair(runs=runs, bytes_removed=removed, backup=backup_rel)
            logger.info(
                "jsonl_nul_repair: repaired %s (%d runs, %d bytes removed; original kept as %s)",
                rel, runs, removed, backup_rel,
            )
    done = not failed
    logger.info(
        "jsonl_nul_repair: scanned %d files, repaired %d, failed %d", scanned, len(repaired), len(failed)
    )
    if done:
        write_json(
            _state_path(persona_dir),
            {
                "done": True,
                "completed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "repaired": {
                    rel: {"runs": r.runs, "bytes_removed": r.bytes_removed, "backup": r.backup}
                    for rel, r in repaired.items()
                },
            },
        )
    return RepairReport(scanned=scanned, repaired=repaired, failed=failed, done=done)
