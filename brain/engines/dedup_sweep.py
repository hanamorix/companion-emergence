"""dedup_sweep.py — one-time retroactive duplicate cleanup (P3 retention
rework, Change 4).

Reuses the consolidation gate's exact-normalize equivalence class
(``_normalize``: whitespace-collapse + casefold) over the whole active
corpus, so an existing near-duplicate pair created before the gate existed
(or that slipped past it) gets merged in one pass. Loss-preserving: every
removed row's full pre-image is archived to the same
``consolidation_archive.jsonl`` the gate's merge path writes to (reason
``"dedup_sweep"``) BEFORE the row is removed. Idempotent: re-running finds
no exact-normalize group with more than one surviving member.

An optional near-dup layer accepts an injectable ``judge`` (same
duplicate/distinct classifier shape as the gate's Haiku classifier),
default OFF — the sweep is deterministic and fully testable without a
provider unless a judge is explicitly supplied.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from brain.engines.consolidation import _ARCHIVE_FILENAME, _normalize
from brain.memory.store import Memory, MemoryStore
from brain.utils.file_lock import file_lock

logger = logging.getLogger(__name__)

_REPORT_FILENAME = "dedup_sweep_report.jsonl"

# (candidate, canonical) -> "duplicate" | "distinct". Same shape as the
# consolidation gate's Pass-2 classifier judgement, scoped to a binary call.
Judge = Callable[[Memory, Memory], str]


@dataclass
class MergeRecord:
    canonical_id: str
    removed_id: str
    content_preview: str


@dataclass
class Report:
    groups_merged: int = 0
    rows_removed: int = 0
    archive_path: str = ""
    merges: list[MergeRecord] = field(default_factory=list)

    def as_log(self) -> dict:
        return {
            "groups_merged": self.groups_merged,
            "rows_removed": self.rows_removed,
            "archive_path": self.archive_path,
            "merges": [
                {
                    "canonical_id": m.canonical_id,
                    "removed_id": m.removed_id,
                    "content_preview": m.content_preview,
                }
                for m in self.merges
            ],
        }


def _archive_preimage(persona_dir: Path, target: Memory) -> None:
    """Append the pre-removal memory to the plain consolidation archive
    (reason "dedup_sweep") BEFORE it is removed — loss-preserving (C4.1).
    Same archive file + shape as the gate's ``_archive_preimage``, a
    distinct reason so the two sources are distinguishable on read."""
    rec = {
        "archived_at": datetime.now(UTC).isoformat(),
        "reason": "dedup_sweep",
        "target": target.to_dict(),
    }
    path = persona_dir / _ARCHIVE_FILENAME
    with file_lock(path):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_report(persona_dir: Path, report: Report) -> None:
    path = persona_dir / _REPORT_FILENAME
    rec = {"run_at": datetime.now(UTC).isoformat(), **report.as_log()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _pick_canonical(members: list[Memory]) -> Memory:
    """Keep the highest-importance member; ties broken by oldest created_at
    (the longest-standing record wins a tie)."""
    ordered = sorted(members, key=lambda m: (-float(m.importance), m.created_at))
    return ordered[0]


def run_dedup_sweep(
    store: MemoryStore,
    persona_dir: str | Path,
    *,
    judge: Judge | None = None,
) -> Report:
    """One-time sweep merging existing near-duplicates. Run once over a
    persona's corpus (e.g. via the CLI entry).

    Primary layer (always on): exact-normalize grouping over ``store.list_active()``
    rows — zero false-merge risk, deterministic. For each group of size > 1:
    the highest-importance member (ties: oldest) is kept as canonical; every
    OTHER member is archived (full pre-image, reason "dedup_sweep") then
    hard-deleted; the canonical's importance is raised to the group max if a
    removed member scored higher.

    Optional near-dup layer: when ``judge`` is supplied, canonical survivors
    left after the exact-normalize pass are compared pairwise; a "duplicate"
    verdict merges the later-created of the pair into the earlier the same
    way (archive-then-delete, carry max importance). Off by default so the
    sweep needs no provider to be fully deterministic and testable.

    Returns a Report (groups merged, rows removed, archive path, per-merge
    canonical/removed ids) and appends it to ``dedup_sweep_report.jsonl``.
    """
    persona_dir = Path(persona_dir)
    memories = store.list_active()

    groups: dict[str, list[Memory]] = {}
    for mem in memories:
        norm = _normalize(mem.content)
        if not norm:
            continue
        groups.setdefault(norm, []).append(mem)

    report = Report(archive_path=str(persona_dir / _ARCHIVE_FILENAME))
    canonicals: list[Memory] = []

    for members in groups.values():
        if len(members) < 2:
            canonicals.append(members[0])
            continue
        canonical = _pick_canonical(members)
        max_importance = max(float(m.importance) for m in members)
        for dup in members:
            if dup.id == canonical.id:
                continue
            _archive_preimage(persona_dir, dup)
            store.hard_delete(dup.id)
            report.merges.append(
                MergeRecord(
                    canonical_id=canonical.id,
                    removed_id=dup.id,
                    content_preview=dup.content[:120],
                )
            )
            report.rows_removed += 1
        if max_importance > canonical.importance:
            store.update(canonical.id, importance=max_importance)
            canonical = store.get(canonical.id, bump=False) or canonical
        report.groups_merged += 1
        canonicals.append(canonical)

    if judge is not None:
        _near_dup_pass(store, persona_dir, canonicals, judge, report)

    _write_report(persona_dir, report)
    return report


def _near_dup_pass(
    store: MemoryStore,
    persona_dir: Path,
    canonicals: list[Memory],
    judge: Judge,
    report: Report,
) -> None:
    """Optional near-dup layer (off by default): pairwise-judge the exact-dedup
    survivors and merge any pair the judge calls "duplicate". Fail-open per
    pair (a judge fault skips that pair, never loses a row)."""
    removed: set[str] = set()
    ordered = sorted(canonicals, key=lambda m: m.created_at)
    for i, cand in enumerate(ordered):
        if cand.id in removed:
            continue
        for other in ordered[i + 1 :]:
            if other.id in removed:
                continue
            try:
                verdict = judge(other, cand)
            except Exception:  # noqa: BLE001 — a judge fault must not lose a row
                logger.exception("dedup_sweep: near-dup judge raised; skipping pair")
                continue
            if verdict != "duplicate":
                continue
            _archive_preimage(persona_dir, other)
            store.hard_delete(other.id)
            removed.add(other.id)
            report.merges.append(
                MergeRecord(
                    canonical_id=cand.id,
                    removed_id=other.id,
                    content_preview=other.content[:120],
                )
            )
            report.rows_removed += 1
            if float(other.importance) > float(cand.importance):
                store.update(cand.id, importance=float(other.importance))
