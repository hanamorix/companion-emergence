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

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from brain.engines.consolidation import _ARCHIVE_FILENAME, _normalize
from brain.health.json_io import append_jsonl
from brain.memory.store import Memory, MemoryStore

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
    append_jsonl(path, rec, nonfinite="null")


def _write_report(persona_dir: Path, report: Report) -> None:
    path = persona_dir / _REPORT_FILENAME
    rec = {"run_at": datetime.now(UTC).isoformat(), **report.as_log()}
    append_jsonl(path, rec)


def _pick_canonical(members: list[Memory], exempt: set[str]) -> Memory:
    """Keep a soul-linked member if there is one (#431), then the highest
    importance; ties broken by oldest created_at (the longest-standing record
    wins a tie)."""
    ordered = sorted(
        members, key=lambda m: (m.id not in exempt, -float(m.importance), m.created_at)
    )
    return ordered[0]


def _load_exempt_ids(persona_dir: Path) -> set[str]:
    """Memory ids the soul depends on (crystallised, under review, and their merge
    targets): the same sets forgetting exempts. The sweep never deletes one (#431).
    Raises ``SoulExemptionsUnavailableError`` when they cannot be read, before any
    row is touched: a failed read is not "nothing is exempt"."""
    from brain.forgetting import _load_soul_linked_ids  # lazy: forgetting is heavier

    # Fails closed (#431): an unreadable exemption set raises here, so nothing is deleted, and the
    # error reaches the CLI as a traceback.
    crystallised, under_review = _load_soul_linked_ids(persona_dir, caller="dedup_sweep")
    return crystallised | under_review


def run_dedup_sweep(
    store: MemoryStore,
    persona_dir: str | Path,
    *,
    judge: Judge | None = None,
) -> Report:
    """One-time sweep merging existing near-duplicates. Run once over a
    persona's corpus (e.g. via the CLI entry).

    Soul exemptions (#431): before any row is touched, the sweep loads the ids
    forgetting exempts (crystallised, under soul review, and the merge targets
    of those ids). A soul-linked row is never archived or deleted. If those
    sets cannot be read, ``SoulExemptionsUnavailableError`` propagates and
    nothing is archived, deleted or reported (#392: a failed read is not
    "nothing is exempt").

    Primary layer (always on): exact-normalize grouping over ``store.list_active()``
    rows — zero false-merge risk, deterministic. For each group of size > 1 the
    canonical is a soul-linked member if there is one, else the
    highest-importance member (ties: oldest). Every other member that is not
    soul-linked is archived (full pre-image, reason "dedup_sweep") then
    hard-deleted, and the canonical's importance is raised to the group max if
    a member scored higher. A group with nothing removable is not counted as
    merged.

    Optional near-dup layer: when ``judge`` is supplied, canonical survivors
    left after the exact-normalize pass are compared pairwise; a "duplicate"
    verdict merges the later-created of the pair into the earlier the same
    way (archive-then-delete, carry max importance). If the later one is
    soul-linked, the earlier one is removed instead; a pair of soul-linked
    rows is left alone. Off by default so the sweep needs no provider to be
    fully deterministic and testable.

    Returns a Report (groups merged, rows removed, archive path, per-merge
    canonical/removed ids) and appends it to ``dedup_sweep_report.jsonl``.
    """
    persona_dir = Path(persona_dir)
    exempt = _load_exempt_ids(persona_dir)
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
        canonical = _pick_canonical(members, exempt)
        # A soul-linked row always stays (#431); if every other member is one, nothing merges.
        # Skip such a group before counting it: counted, it reads as "merged" on every re-run.
        removable = [m for m in members if m.id != canonical.id and m.id not in exempt]
        if not removable:
            canonicals.append(canonical)
            continue
        max_importance = max(float(m.importance) for m in members)
        for dup in removable:
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
        _near_dup_pass(store, persona_dir, canonicals, judge, report, exempt)

    _write_report(persona_dir, report)
    return report


def _near_dup_pass(
    store: MemoryStore,
    persona_dir: Path,
    canonicals: list[Memory],
    judge: Judge,
    report: Report,
    exempt: set[str],
) -> None:
    """Optional near-dup layer (off by default): pairwise-judge the exact-dedup
    survivors and merge any pair the judge calls "duplicate". Fail-open per
    pair (a judge fault skips that pair, never loses a row). The later of a
    pair is removed, unless it is soul-linked: then the earlier goes instead,
    and a pair of two soul-linked rows is left alone (#431)."""
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
            if verdict != "duplicate" or (other.id in exempt and cand.id in exempt):
                continue
            keep, drop = (other, cand) if other.id in exempt else (cand, other)
            # Re-read both (#431): an earlier merge in this pass may have raised either one's
            # importance and the in-memory objects never see it. Stale, the carry-over below lowers
            # a raised canonical again, or drops its max with it, and the archive gets a stale pre-image.
            # A row gone since the pass listed it (a concurrent writer) is never archived from the
            # stale copy, deleted or updated: the pair is skipped and logged (#431).
            keep_now = store.get(keep.id, bump=False)
            drop_now = store.get(drop.id, bump=False)
            if keep_now is None or drop_now is None:
                logger.warning(
                    "dedup_sweep: skipping pair %s / %s: a row vanished mid-pass", keep.id, drop.id
                )
                continue
            keep, drop = keep_now, drop_now
            _archive_preimage(persona_dir, drop)
            store.hard_delete(drop.id)
            removed.add(drop.id)
            # Rows merged into ``drop`` earlier now live on in ``keep``; otherwise their records name a
            # deleted row (#431).
            for earlier in report.merges:
                if earlier.canonical_id == drop.id:
                    earlier.canonical_id = keep.id
            report.merges.append(
                MergeRecord(
                    canonical_id=keep.id,
                    removed_id=drop.id,
                    content_preview=drop.content[:120],
                )
            )
            report.rows_removed += 1
            if float(drop.importance) > float(keep.importance):
                store.update(keep.id, importance=float(drop.importance))
            if drop.id == cand.id:
                # ``cand`` is gone (#431): comparing it further would merge later rows into a deleted
                # row and lose them.
                break
