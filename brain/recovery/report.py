"""RecoveryReport — outcome of a `nell recover` run. Mirrors MigrationReport."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

RecoveryMode = Literal["source", "graveyard"]


@dataclass(frozen=True)
class RecoveryReport:
    persona: str
    mode: RecoveryMode
    source_dir: str | None
    memories_restored_full: int
    memories_restored_summary: int
    memories_unfaded: int
    edges_repaired: int
    edges_pruned_unrecoverable: int
    backup_path: str | None
    elapsed_seconds: float
    dry_run: bool

    def to_json(self) -> str:
        payload = asdict(self)
        payload["kind"] = "RecoveryReport"
        return json.dumps(payload, default=str)


def format_report(r: RecoveryReport) -> str:
    dry = " (dry-run — nothing written)" if r.dry_run else ""
    lines = [
        f"Recovery for {r.persona} — mode: {r.mode}{dry}",
        f"  memories restored (full):    {r.memories_restored_full}",
        f"  memories restored (summary): {r.memories_restored_summary}",
        f"  memories un-faded:           {r.memories_unfaded}",
        f"  edges repaired:              {r.edges_repaired}",
        f"  edges pruned (unrecoverable):{r.edges_pruned_unrecoverable}",
    ]
    if r.backup_path:
        lines.append(f"  backup: {r.backup_path}")
    return "\n".join(lines)
