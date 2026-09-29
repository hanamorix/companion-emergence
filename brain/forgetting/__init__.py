"""brain.forgetting — composite salience + state machine + graveyard.

Spec: docs/superpowers/specs/2026-05-18-forgetting-design.md
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from brain.bridge import job_progress
from brain.felt_time.lived_age import IntensityDrivers
from brain.felt_time.state import load_or_recover as load_felt_time
from brain.forgetting import graveyard, policy, salience, tombstone
from brain.forgetting.policy import Transition
from brain.health.attempt_heal import save_with_backup
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore, _row_to_memory

log = logging.getLogger(__name__)

FORGETTING_STATE_FILENAME = "forgetting_state.json"


def _load_forgetting_state(persona_dir: Path) -> dict[str, int]:
    """Read consecutive_low_passes counters keyed by memory_id.
    Corrupt file → all-zero counters (defensive)."""
    p = persona_dir / FORGETTING_STATE_FILENAME
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            return {}
        return {k: int(v) for k, v in data.items() if isinstance(v, (int, float))}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def _persist_forgetting_state(persona_dir: Path, counters: dict[str, int]) -> None:
    """Atomic save via save_with_backup."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    save_with_backup(persona_dir / FORGETTING_STATE_FILENAME, counters)


def _load_soul_linked_ids(persona_dir: Path) -> tuple[set[str], set[str]]:
    """Returns (crystallised_ids, under_review_ids).

    Best-effort: if the soul subsystem isn't reachable, both sets are
    empty — memories that should have been exempt may fade, but that's
    safer than crashing the supervisor pass.
    """
    try:
        from brain.soul.audit import list_crystallised_memory_ids

        crystallised = set(list_crystallised_memory_ids(persona_dir))
    except Exception:
        crystallised = set()
    try:
        from brain.soul.candidates import list_under_review_memory_ids

        under_review = set(list_under_review_memory_ids(persona_dir))
    except Exception:
        under_review = set()
    return crystallised, under_review


def _load_migration_grace(persona_dir: Path) -> tuple[datetime | None, float]:
    """Return (migrated_at_utc, lived_age_hours_at_migration) from source-manifest.json.
    (None, 0.0) when no manifest or fields absent — i.e. no grace (back-compat)."""
    p = persona_dir / "source-manifest.json"
    if not p.exists():
        return None, 0.0
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None, 0.0
    raw = data.get("migrated_at_utc") or data.get("generated_at_utc")
    mig: datetime | None = None
    if isinstance(raw, str):
        try:
            mig = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            mig = None
    lived = data.get("lived_age_hours_at_migration", 0.0)
    try:
        lived = float(lived)
    except (TypeError, ValueError):
        lived = 0.0
    return mig, lived


_FORGETTING_PROGRESS_JOB = "forgetting"


def run_pass(
    persona_dir: Path,
    *,
    event_bus: Any,
    intensity_drivers: IntensityDrivers | None = None,
    should_pause: Callable[[], bool] | None = None,
    progress_out: dict[str, bool] | None = None,
) -> dict[str, int]:
    """Run one forgetting pass over all active+fading memories.

    Returns an aggregate summary dict with counts; also publishes a
    `forgetting_pass` event_bus event with the same payload.

    ``should_pause`` (ram-spike-fix INC-10, S14/S32/S41/S65): checked after
    each memory. The S32 table's item unit is "one memory"; this pass
    previously had no per-item resume point at all (counters were only
    persisted at pass end) — resume is now a NEW ``forgetting_progress.json``
    cursor (``brain.bridge.job_progress``, C31), a keyset position over
    memories ordered by ``id`` (the same collation SQLite already sorts
    strings by, so a Python string comparison on the resumed slice matches
    the SQL ``ORDER BY``). ``counters`` (consecutive-low-passes) are now
    ALSO persisted after every memory, not only at pass end, so a pause or a
    crash mid-pass never loses or re-applies a counter update. On a clean
    finish the cursor is cleared (a stale cursor would otherwise cause the
    NEXT pass to wrongly skip the memories before it).
    """
    start = time.monotonic()
    counters = _load_forgetting_state(persona_dir)
    felt_state, _recovered = load_felt_time(persona_dir)
    crystallised_ids, under_review_ids = _load_soul_linked_ids(persona_dir)
    soul_linked = crystallised_ids | under_review_ids
    migrated_at_utc, lived_at_migration = _load_migration_grace(persona_dir)

    summary: dict[str, int] = {"faded": 0, "unfaded": 0, "lost": 0, "exempt": 0, "total": 0}

    db_path = persona_dir / "memories.db"
    if not db_path.exists():
        summary["duration_ms"] = int((time.monotonic() - start) * 1000)
        event_bus.publish({"type": "forgetting_pass", **summary})
        return summary

    store = MemoryStore(db_path)
    hebbian_path = persona_dir / "hebbian.db"
    hebbian = (
        HebbianMatrix(str(hebbian_path)) if hebbian_path.exists() else HebbianMatrix(":memory:")
    )

    try:
        # Walk active + fading memories, ordered by id (INC-10: a stable,
        # deterministic order the resume cursor below can key off).
        # Use a direct SELECT (not store.get) so the forgetting pass does NOT
        # bump recall_count — the pass is an internal evaluation, not a user
        # recall. Bumping via store.get would inflate recall salience and prevent
        # the consecutive-low-passes counter from accumulating correctly.
        rows = store._conn.execute(
            "SELECT * FROM memories WHERE state IN ('active', 'fading') ORDER BY id"
        ).fetchall()
        memories = [_row_to_memory(r) for r in rows]
        summary["total"] = len(memories)

        resume_cursor = job_progress.load_progress(persona_dir, _FORGETTING_PROGRESS_JOB)
        last_id = resume_cursor.get("last_id") if isinstance(resume_cursor, dict) else None
        if isinstance(last_id, str):
            # Stage-6 red-team MAJOR, fixed: the cursor's anchor row may no
            # longer be IN `memories` on resume -- a LOSE transition
            # hard-deletes the row (store.hard_delete, below), so an exact
            # `m.id == last_id` search would never match and resume_idx
            # would silently stay 0, reprocessing the WHOLE backlog. Since
            # `memories` is a deterministic `ORDER BY id` scan, the correct
            # resume point is keyset-style: the first row whose id sorts
            # AFTER last_id — this is correct whether or not that exact row
            # still exists (deleted, or merely absent for any other reason).
            resume_idx = len(memories)
            for i, m in enumerate(memories):
                if m.id > last_id:
                    resume_idx = i
                    break
            memories = memories[resume_idx:]

        for memory in memories:
            memory_id = memory.id
            if policy.is_exempt(
                memory,
                soul_crystallised_ids=crystallised_ids,
                under_review_ids=under_review_ids,
                now_lived_age_hours=felt_state.lived_age_hours,
            ):
                summary["exempt"] += 1
                continue

            if policy.is_within_import_grace(
                memory,
                migrated_at_utc=migrated_at_utc,
                lived_age_hours_at_migration=lived_at_migration,
                current_lived_age_hours=felt_state.lived_age_hours,
            ):
                summary["exempt"] += 1
                continue

            s = salience.score(
                memory,
                store=store,
                hebbian=hebbian,
                felt_time_state=felt_state,
                soul_linked_ids=soul_linked,
            )
            prev_low = counters.get(memory_id, 0)
            # Update consecutive_low_passes for this pass.
            if s < policy.LOST_THRESHOLD:
                next_low = prev_low + 1
            else:
                next_low = 0
            nw = intensity_drivers.narrative_weight if intensity_drivers else 0.0
            transition = policy.next_state(
                memory,
                salience=s,
                consecutive_low_passes=next_low,
                narrative_weight=nw,
            )
            if transition == Transition.FADE:
                summary_text = tombstone.summarise(memory.content)
                store.fade(memory_id, summary=summary_text)
                summary["faded"] += 1
            elif transition == Transition.UNFADE:
                store.unfade(memory_id)
                summary["unfaded"] += 1
                next_low = 0  # reset on unfade
            elif transition == Transition.LOSE:
                inputs = salience.compute_inputs(
                    memory,
                    store=store,
                    hebbian=hebbian,
                    felt_time_state=felt_state,
                    soul_linked_ids=soul_linked,
                )
                neighbors_at_drop = hebbian.neighbors(memory_id)
                # Graveyard write BEFORE hard_delete (spec §4 order), now also
                # tombstoning the link structure so recovery can rebuild it.
                graveyard.append(
                    persona_dir,
                    memory=memory,
                    salience_at_drop=s,
                    inputs=inputs,
                    lived_age_hours=felt_state.lived_age_hours,
                    reason=f"salience<{policy.LOST_THRESHOLD} for {next_low} consecutive passes",
                    hebbian_neighbors=neighbors_at_drop,
                )
                store.hard_delete(memory_id)
                # Remove orphaned edges BEFORE grief so a grief failure can
                # never strand a dangling edge.
                hebbian.remove_memory(memory_id)
                summary["lost"] += 1
                try:
                    from brain import grief

                    grief.handle_drop(memory=memory, persona_dir=persona_dir, store=store)
                except Exception:
                    log.exception(
                        "grief.handle_drop failed inside forgetting pass for memory_id=%s",
                        memory_id,
                    )
                next_low = 0  # cleared; row gone

            if next_low > 0:
                counters[memory_id] = next_low
            else:
                counters.pop(memory_id, None)

            # INC-10 (S14/S32/S41/S65): persist progress after EVERY memory —
            # counters used to be saved only at pass end (module docstring's
            # own note) — so a between-items pause or a crash mid-pass loses
            # nothing and never re-applies a counter update on resume.
            # Stage-6 red-team MINOR, addressed: these are two separate
            # atomic writes, not one atomic pair — a crash in the narrow
            # window between them is possible. Cursor is saved FIRST so that
            # window's failure mode is "this item's counter update is lost"
            # (self-healing: consecutive_low_passes just takes one extra
            # pass to reach LOST_THRESHOLD, a soft heuristic already
            # tolerant of resets), never "double-applied" (which the
            # opposite order would risk: an already-saved cursor is what a
            # resume trusts to skip the item, so counters must not be
            # written to look "not yet done" behind a cursor that already
            # says it's done).
            job_progress.save_progress(
                persona_dir, _FORGETTING_PROGRESS_JOB, {"last_id": memory_id}
            )
            _persist_forgetting_state(persona_dir, counters)
            if should_pause is not None and memory is not memories[-1] and should_pause():
                log.info("forgetting pass: pausing between memories for chat (INC-10)")
                summary["duration_ms"] = int((time.monotonic() - start) * 1000)
                event_bus.publish({"type": "forgetting_pass", **summary})
                if progress_out is not None:
                    progress_out["paused"] = True
                return summary
    finally:
        store.close()
        hebbian.close()

    # A clean finish clears the cursor — a stale one would wrongly make the
    # NEXT pass skip memories that precede it (this pass already re-evaluated
    # the whole backlog by the time it gets here).
    job_progress.clear_progress(persona_dir, _FORGETTING_PROGRESS_JOB)
    _persist_forgetting_state(persona_dir, counters)
    summary["duration_ms"] = int((time.monotonic() - start) * 1000)
    event_bus.publish({"type": "forgetting_pass", **summary})
    return summary


__all__ = ["run_pass", "graveyard", "policy", "salience", "tombstone"]
