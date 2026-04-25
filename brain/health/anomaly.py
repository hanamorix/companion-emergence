"""BrainAnomaly + AlarmEntry — structured records of detection events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from brain.utils.time import iso_utc, parse_iso_utc

AnomalyKind = Literal["json_parse_error", "schema_mismatch", "sqlite_integrity_fail"]
AnomalyAction = Literal[
    "restored_from_bak1",
    "restored_from_bak2",
    "restored_from_bak3",
    "reset_to_default",
    "reconstructed_from_memories",
    "alarmed_unrecoverable",
    "verify_after_write_failed",
]
LikelyCause = Literal["user_edit", "disk", "unknown"]


@dataclass(frozen=True)
class BrainAnomaly:
    timestamp: datetime
    file: str
    kind: AnomalyKind
    action: AnomalyAction
    quarantine_path: str | None
    likely_cause: LikelyCause
    detail: str

    def to_dict(self) -> dict:
        return {
            "timestamp": iso_utc(self.timestamp),
            "file": self.file,
            "kind": self.kind,
            "action": self.action,
            "quarantine_path": self.quarantine_path,
            "likely_cause": self.likely_cause,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BrainAnomaly:
        return cls(
            timestamp=parse_iso_utc(data["timestamp"]),
            file=str(data["file"]),
            kind=data["kind"],
            action=data["action"],
            quarantine_path=data.get("quarantine_path"),
            likely_cause=data.get("likely_cause", "unknown"),
            detail=str(data.get("detail", "")),
        )


@dataclass(frozen=True)
class AlarmEntry:
    file: str
    kind: str
    first_seen_at: datetime
    occurrences_in_window: int


class BrainIntegrityError(Exception):
    """Raised when a SQLite database fails PRAGMA integrity_check.

    The brain's memory or hebbian graph is unrecoverable from this state
    in v1 — surfaces as a Layer 3 alarm in the audit log.
    """

    def __init__(self, db_path: str, detail: str) -> None:
        super().__init__(f"integrity check failed for {db_path}: {detail}")
        self.db_path = db_path
        self.detail = detail
