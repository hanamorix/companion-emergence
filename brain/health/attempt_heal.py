"""Core heal + save helpers — atomic .bak rotation + corruption recovery."""

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
from brain.utils.time import iso_utc

logger = logging.getLogger(__name__)


def _filename_safe_timestamp(now: datetime) -> str:
    """Build an ISO-like timestamp safe for filenames on every platform.

    `iso_utc(now)` produces strings with `:` (e.g. `2026-04-25T18:30:00.123456Z`),
    which Windows rejects in filenames. We swap colons for hyphens so the
    quarantine filename round-trips on POSIX + Windows.
    """
    return iso_utc(now).replace(":", "-")


def attempt_heal(
    path: Path,
    default_factory: Callable[[], Any],
    schema_validator: Callable[[Any], None] | None = None,
) -> tuple[Any, BrainAnomaly | None]:
    """Load `path`, healing from .bak rotation if corrupt.

    Returns (data, anomaly_or_None).
      - Missing file → (default_factory(), None)
      - Healthy file → (parsed_data, None)
      - Corrupt file → quarantine, walk .bak1/.bak2/.bak3, restore freshest
        valid backup; if all corrupt, write default_factory() output.
        Returns (data, BrainAnomaly).
    """
    if not path.exists():
        return default_factory(), None

    try:
        data = _load_and_validate(path, schema_validator)
        return data, None
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return _heal_from_baks(path, default_factory, schema_validator, exc)


def _load_and_validate(path: Path, validator: Callable[[Any], None] | None) -> Any:
    data = json.loads(path.read_text(encoding="utf-8"))
    if validator is not None:
        validator(data)
    return data


def _heal_from_baks(
    path: Path,
    default_factory: Callable[[], Any],
    schema_validator: Callable[[Any], None] | None,
    original_exc: Exception,
) -> tuple[Any, BrainAnomaly]:
    now = datetime.now(UTC)
    likely_cause = _classify_cause(path)
    quarantine = path.with_name(f"{path.name}.corrupt-{_filename_safe_timestamp(now)}")
    os.replace(path, quarantine)

    kind = (
        "json_parse_error" if isinstance(original_exc, json.JSONDecodeError) else "schema_mismatch"
    )

    for bak_index in (1, 2, 3):
        bak = path.with_name(f"{path.name}.bak{bak_index}")
        if not bak.exists():
            continue
        try:
            data = _load_and_validate(bak, schema_validator)
        except (json.JSONDecodeError, ValueError, TypeError):
            # This bak is also corrupt — quarantine it too.
            bak_quarantine = path.with_name(
                f"{path.name}.bak{bak_index}.corrupt-{_filename_safe_timestamp(now)}"
            )
            os.replace(bak, bak_quarantine)
            continue

        # Found a valid bak — restore.
        os.replace(bak, path)
        return (
            data,
            BrainAnomaly(
                timestamp=now,
                file=path.name,
                kind=kind,  # type: ignore[arg-type]
                action=f"restored_from_bak{bak_index}",  # type: ignore[arg-type]
                quarantine_path=quarantine.name,
                likely_cause=likely_cause,
                detail=str(original_exc)[:500],
            ),
        )

    # All baks corrupt or missing — reset to default.
    default_data = default_factory()
    path.write_text(json.dumps(default_data, indent=2) + "\n", encoding="utf-8")
    return (
        default_data,
        BrainAnomaly(
            timestamp=now,
            file=path.name,
            kind=kind,  # type: ignore[arg-type]
            action="reset_to_default",
            quarantine_path=quarantine.name,
            likely_cause=likely_cause,
            detail=str(original_exc)[:500],
        ),
    )


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


def save_with_backup(path: Path, data: Any, backup_count: int = 3) -> None:
    """Atomic save with .bak rotation.

    Writes <path>.new, rotates existing <path> → <path>.bak1 → ... → <path>.bak{N},
    drops oldest, then atomically replaces <path>. Stale <path>.new from a prior
    crash is unlinked before the new write begins.
    """
    new_path = path.with_name(path.name + ".new")
    if new_path.exists():
        new_path.unlink()  # stale from prior crash; always incomplete

    new_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    # Rotate: drop oldest first, then walk down toward live.
    oldest = path.with_name(f"{path.name}.bak{backup_count}")
    if oldest.exists():
        oldest.unlink()
    for i in range(backup_count - 1, 0, -1):
        src = path.with_name(f"{path.name}.bak{i}")
        dst = path.with_name(f"{path.name}.bak{i + 1}")
        if src.exists():
            os.replace(src, dst)
    if path.exists():
        os.replace(path, path.with_name(f"{path.name}.bak1"))

    os.replace(new_path, path)
