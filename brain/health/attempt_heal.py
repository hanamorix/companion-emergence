"""Core heal helpers: the plain-text heal and the legacy JSON save wrapper.

The atomic writers (the .bak rotation, the unique temp, the retried replace) live in ``brain.utils.atomic_write``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.health.anomaly import BrainAnomaly
from brain.utils.atomic_write import (
    save_with_backup_text,  # the one writer, used by save_with_backup below
)
from brain.utils.time import iso_utc

logger = logging.getLogger(__name__)


def _filename_safe_timestamp(now: datetime) -> str:
    """Build an ISO-like timestamp safe for filenames on every platform.

    `iso_utc(now)` produces strings with `:` (e.g. `2026-04-25T18:30:00.123456Z`),
    which Windows rejects in filenames. We swap colons for hyphens so the
    quarantine filename round-trips on POSIX + Windows.
    """
    return iso_utc(now).replace(":", "-")


def _classify_cause(path: Path) -> str:
    """Heuristic: hand-edit vs disk vs unknown.

    user_edit: mtime within 60s, size < 100KB, content starts with { or [.
    Otherwise unknown. (No specific disk-error heuristic in v1.)
    """
    try:
        st = path.stat()
        if time.time() - st.st_mtime < 60 and st.st_size < 100_000:
            head = path.read_bytes()[:1]
            if head and head[:1] in (b"{", b"["):
                return "user_edit"
    except OSError:
        pass
    return "unknown"


def attempt_heal_text(
    path: Path,
    default_factory: Callable[[], str],
) -> tuple[str, BrainAnomaly | None]:
    """Like the JSON heal (brain.health.json_io) but for plain-text files (no JSON parsing).

    Missing → (default_factory(), None) silently.
    Empty → quarantine, walk .bak1/.bak2/.bak3, restore freshest valid backup.
    All baks missing or empty → write default text + return anomaly.

    What counts as "corrupt" for plain text: empty file (len == 0 after strip).
    This is the practical failure mode after a disk hiccup on an identity file.
    """
    if not path.exists():
        default_text = default_factory()
        path.write_text(default_text, encoding="utf-8")
        return default_text, None

    content = path.read_text(encoding="utf-8")
    if content.strip():
        return content, None

    # Empty file — treat as corrupt; start the heal cycle.
    return _heal_text_from_baks(path, default_factory)


def _heal_text_from_baks(
    path: Path,
    default_factory: Callable[[], str],
) -> tuple[str, BrainAnomaly]:
    now = datetime.now(UTC)
    likely_cause = _classify_cause(path)
    quarantine = path.with_name(f"{path.name}.corrupt-{_filename_safe_timestamp(now)}")
    os.replace(path, quarantine)  # atomic-write: ok -- moves the damaged file aside (a rename of a finished file), not a write

    for bak_index in (1, 2, 3):
        bak = path.with_name(f"{path.name}.bak{bak_index}")
        if not bak.exists():
            continue
        bak_content = bak.read_text(encoding="utf-8")
        if not bak_content.strip():
            # This bak is also empty — quarantine it too.
            bak_quarantine = path.with_name(
                f"{path.name}.bak{bak_index}.corrupt-{_filename_safe_timestamp(now)}"
            )
            os.replace(bak, bak_quarantine)  # atomic-write: ok -- moves an empty backup aside (a rename of a finished file)
            continue

        # Found a valid bak — restore.
        os.replace(bak, path)  # atomic-write: ok -- restores a finished backup file over the live one (a rename, not a write)
        return (
            bak_content,
            BrainAnomaly(
                timestamp=now,
                file=path.name,
                kind="json_parse_error",  # closest kind; text "empty" maps here
                action=f"restored_from_bak{bak_index}",  # type: ignore[arg-type]
                quarantine_path=quarantine.name,
                likely_cause=likely_cause,
                detail="empty text file",
            ),
        )

    # All baks corrupt or missing — reset to default.
    default_text = default_factory()
    path.write_text(default_text, encoding="utf-8")
    return (
        default_text,
        BrainAnomaly(
            timestamp=now,
            file=path.name,
            kind="json_parse_error",
            action="reset_to_default",
            quarantine_path=quarantine.name,
            likely_cause=likely_cause,
            detail="empty text file, all baks missing or empty",
        ),
    )


def save_with_backup(path: Path, data: Any, backup_count: int = 3) -> None:
    """Atomic JSON save with .bak rotation. Delegates to save_with_backup_text."""
    save_with_backup_text(path, json.dumps(data, indent=2) + "\n", backup_count=backup_count)
