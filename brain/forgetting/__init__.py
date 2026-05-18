"""brain.forgetting — composite salience + state machine + graveyard.

Spec: docs/superpowers/specs/2026-05-18-forgetting-design.md
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from brain.felt_time.state import load_or_recover as load_felt_time
from brain.forgetting import graveyard, policy, salience, tombstone
from brain.forgetting.policy import Transition
from brain.health.attempt_heal import save_with_backup
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore, _row_to_memory

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


def run_pass(persona_dir: Path, *, event_bus: Any) -> dict[str, int]:
    """Run one forgetting pass over all active+fading memories.

    Returns an aggregate summary dict with counts; also publishes a
    `forgetting_pass` event_bus event with the same payload.
    """
    start = time.monotonic()
    counters = _load_forgetting_state(persona_dir)
    felt_state, _recovered = load_felt_time(persona_dir)
    crystallised_ids, under_review_ids = _load_soul_linked_ids(persona_dir)
    soul_linked = crystallised_ids | under_review_ids

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
        # Walk active + fading memories.
        # Use a direct SELECT (not store.get) so the forgetting pass does NOT
        # bump recall_count — the pass is an internal evaluation, not a user
        # recall. Bumping via store.get would inflate recall salience and prevent
        # the consecutive-low-passes counter from accumulating correctly.
        rows = store._conn.execute(
            "SELECT * FROM memories WHERE state IN ('active', 'fading')"
        ).fetchall()
        memories = [_row_to_memory(r) for r in rows]
        summary["total"] = len(memories)

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
            transition = policy.next_state(memory, salience=s, consecutive_low_passes=next_low)
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
                # Graveyard write BEFORE hard_delete (spec §4 order).
                graveyard.append(
                    persona_dir,
                    memory=memory,
                    salience_at_drop=s,
                    inputs=inputs,
                    lived_age_hours=felt_state.lived_age_hours,
                    reason=f"salience<{policy.LOST_THRESHOLD} for {next_low} consecutive passes",
                )
                store.hard_delete(memory_id)
                summary["lost"] += 1
                next_low = 0  # cleared; row gone

            if next_low > 0:
                counters[memory_id] = next_low
            else:
                counters.pop(memory_id, None)
    finally:
        store.close()
        hebbian.close()

    _persist_forgetting_state(persona_dir, counters)
    summary["duration_ms"] = int((time.monotonic() - start) * 1000)
    event_bus.publish({"type": "forgetting_pass", **summary})
    return summary


__all__ = ["run_pass", "graveyard", "policy", "salience", "tombstone"]
