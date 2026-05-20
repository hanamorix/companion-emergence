"""brain.narrative_memory — anchor-seeded narrative arcs over memory.

Third member of the Memory & time cluster (Tier 1 #1). Public surface:
    run_pass(persona_dir, *, event_bus, ...) — ArcUpdatePass orchestrator

Inherits substrate from felt-time (FeltTimeState anchors + lived_age) and
forgetting (policy.is_exempt + salience.score). See spec
docs/superpowers/specs/2026-05-19-narrative-memory-design.md.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from brain.narrative_memory.arc import Arc, ArcMember
from brain.narrative_memory.membership import (
    EmbeddingsView,
    HebbianView,
    is_candidate,
)
from brain.narrative_memory.policy import (
    MAX_ARC_MEMBERS,
    should_close,
    should_open,
)
from brain.narrative_memory.state import (
    RECENTLY_CLOSED_CAP,
    append_event,
    load_or_recover,
    save_state,
)

_LOG = logging.getLogger(__name__)


class _AnchorLike(Protocol):
    type: str
    ref: str
    label: str
    ts_iso: str
    lived_age_hours: float
    seed_memory_ids: tuple[str, ...]


class _MemoryLike(Protocol):
    id: str
    created_at_iso: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _make_arc_id(seed_anchor_ref: str, ts_iso: str) -> str:
    # arc_<YYYYMMDD>_<8hex>
    yyyymmdd = ts_iso[:10].replace("-", "") if len(ts_iso) >= 10 else "00000000"
    digest = hashlib.sha256(f"{seed_anchor_ref}:{ts_iso}:{uuid.uuid4()}".encode()).hexdigest()[:8]
    return f"arc_{yyyymmdd}_{digest}"


def run_pass(
    persona_dir: Path,
    *,
    event_bus: Any,
    anchor_sweep: Callable[[Path, str | None], list[_AnchorLike]],
    candidate_pool: Callable[..., list[_MemoryLike]],
    salience_score: Callable[..., float],
    is_exempt: Callable[[_MemoryLike], bool],
    hebbian: HebbianView,
    embeddings: EmbeddingsView,
    felt_time_state: Any,
) -> dict[str, int | float]:
    """Run one arc-update pass. Returns aggregate counts for telemetry.

    Per spec §5:
      1. Load state from disk
      2. Anchor sweep — open new arcs / extend existing arcs
      3. Candidate sweep — apply is_candidate against each open arc
      4. Member-cap enforcement — evict lowest-salience past MAX_ARC_MEMBERS
      5. Close sweep — staleness + all-members-lost
      6. Persist state.json + publish event_bus aggregate

    Args are injected so the orchestrator stays unit-testable without the
    full supervisor stack — the supervisor wrapper (Phase 8) wires up the
    real anchor extraction, salience scoring, exemption check, and stores.
    """
    t0 = time.monotonic()
    state = load_or_recover(persona_dir)

    counts: dict[str, int] = {"opened": 0, "extended": 0, "closed": 0, "evicted": 0}
    lived_age_now = getattr(felt_time_state, "lived_age_hours", None)
    centroid_cache: dict[str, np.ndarray | None] = {}
    # Arcs opened or extended in THIS pass — staleness check skips them
    # so a freshly-touched arc can't close in the same tick.
    touched_arc_ids: set[str] = set()

    # --- Anchor sweep ---
    last_pass_ts = state.last_pass_ts_iso
    for anchor in anchor_sweep(persona_dir, last_pass_ts):
        if not anchor.seed_memory_ids:
            _LOG.warning("anchor %s has no seed memory ids — skipping", anchor.ref)
            continue

        if should_open(tuple(anchor.seed_memory_ids), open_arcs=state.open):
            arc_id = _make_arc_id(anchor.ref, anchor.ts_iso)
            # Seed-member salience defaults to 0.0 because the seed memory
            # hasn't been independently scored at anchor-fire time. Cap
            # eviction can promote it on subsequent passes once its salience
            # is computed naturally via the candidate-sweep path.
            seed_members: list[ArcMember] = []
            for sid in anchor.seed_memory_ids:
                seed_members.append(
                    ArcMember(
                        memory_id=sid,
                        joined_at_iso=anchor.ts_iso,
                        lived_age_at_join=anchor.lived_age_hours,
                        salience_at_join=0.0,
                    )
                )
            new_arc = Arc(
                id=arc_id,
                state="open",
                seed_anchor_type=anchor.type,
                seed_anchor_ref=anchor.ref,
                seed_memory_ids=tuple(anchor.seed_memory_ids),
                title=anchor.label,
                opened_at_iso=anchor.ts_iso,
                lived_age_at_open=anchor.lived_age_hours,
                last_extended_at_iso=anchor.ts_iso,
                closed_at_iso=None,
                lived_age_at_close=None,
                members=tuple(seed_members),
            )
            state.open[arc_id] = new_arc
            touched_arc_ids.add(arc_id)
            append_event(
                persona_dir,
                {
                    "event": "arc_opened",
                    "arc_id": arc_id,
                    "seed_anchor_type": anchor.type,
                    "seed_anchor_ref": anchor.ref,
                    "seed_memory_ids": list(anchor.seed_memory_ids),
                    "title": anchor.label,
                    "ts_iso": anchor.ts_iso,
                    "lived_age_hours": anchor.lived_age_hours,
                },
            )
            for sm in seed_members:
                append_event(
                    persona_dir,
                    {
                        "event": "member_added",
                        "arc_id": arc_id,
                        "memory_id": sm.memory_id,
                        "ts_iso": anchor.ts_iso,
                        "lived_age_hours": anchor.lived_age_hours,
                        "salience_at_join": sm.salience_at_join,
                        "via": "seed",
                    },
                )
            counts["opened"] += 1
        else:
            # Extend the open arc whose member set intersects seed_memory_ids
            target_arc_id: str | None = None
            for arc_id, arc in state.open.items():
                existing = {m.memory_id for m in arc.members}
                if existing & set(anchor.seed_memory_ids):
                    target_arc_id = arc_id
                    break
            if target_arc_id is None:
                continue
            arc = state.open[target_arc_id]
            existing_ids = {m.memory_id for m in arc.members}
            new_members = list(arc.members)
            extended_now = False
            for sid in anchor.seed_memory_ids:
                if sid in existing_ids:
                    continue
                new_member = ArcMember(
                    memory_id=sid,
                    joined_at_iso=anchor.ts_iso,
                    lived_age_at_join=anchor.lived_age_hours,
                    salience_at_join=0.0,
                )
                new_members.append(new_member)
                existing_ids.add(sid)
                extended_now = True
                append_event(
                    persona_dir,
                    {
                        "event": "member_added",
                        "arc_id": target_arc_id,
                        "memory_id": sid,
                        "ts_iso": anchor.ts_iso,
                        "lived_age_hours": anchor.lived_age_hours,
                        "salience_at_join": 0.0,
                        "via": "seed",
                    },
                )
            if extended_now:
                state.open[target_arc_id] = _replace(
                    arc,
                    members=tuple(new_members),
                    last_extended_at_iso=anchor.ts_iso,
                )
                touched_arc_ids.add(target_arc_id)
                counts["extended"] += 1

    # --- Candidate sweep ---
    for arc_id, arc in list(state.open.items()):
        for memory in candidate_pool(persona_dir, opened_at_iso=arc.opened_at_iso):
            if any(m.memory_id == memory.id for m in arc.members):
                continue
            if is_exempt(memory):
                continue
            hit, via = is_candidate(
                memory, arc, hebbian=hebbian, embeddings=embeddings, centroid_cache=centroid_cache
            )
            if not hit:
                continue
            sal = salience_score(memory, ctx=None)
            new_member = ArcMember(
                memory_id=memory.id,
                joined_at_iso=_now_iso(),
                lived_age_at_join=lived_age_now if lived_age_now is not None else 0.0,
                salience_at_join=sal,
            )
            arc = _replace(
                arc, members=arc.members + (new_member,), last_extended_at_iso=_now_iso()
            )
            state.open[arc_id] = arc
            touched_arc_ids.add(arc_id)
            append_event(
                persona_dir,
                {
                    "event": "member_added",
                    "arc_id": arc_id,
                    "memory_id": memory.id,
                    "ts_iso": new_member.joined_at_iso,
                    "lived_age_hours": new_member.lived_age_at_join,
                    "salience_at_join": sal,
                    "via": via,
                },
            )
            counts["extended"] += 1

    # --- Member-cap enforcement ---
    for arc_id, arc in list(state.open.items()):
        if len(arc.members) <= MAX_ARC_MEMBERS:
            continue
        # Sort by salience ascending; drop lowest until at cap
        sorted_members = sorted(arc.members, key=lambda m: m.salience_at_join)
        evictions = len(arc.members) - MAX_ARC_MEMBERS
        evicted = sorted_members[:evictions]
        kept = sorted(sorted_members[evictions:], key=lambda m: m.joined_at_iso)
        for ev in evicted:
            append_event(
                persona_dir,
                {
                    "event": "member_evicted",
                    "arc_id": arc_id,
                    "memory_id": ev.memory_id,
                    "ts_iso": _now_iso(),
                    "reason": "max_members",
                },
            )
            counts["evicted"] += 1
        state.open[arc_id] = _replace(arc, members=tuple(kept))

    # --- Membership refresh: detect lost members in open arcs, fire grief touches ---
    # Per spec §6 Phase 9.2: when an arc member-id resolves to a graveyard entry
    # rather than a live memory, call handle_recall_touch with triggering_arc_id set
    # so the breadcrumb attributes the touch to both the lost memory and the open arc.
    db_path = persona_dir / "memories.db"
    store_for_grief = None
    if db_path.exists():
        from brain.memory.store import MemoryStore

        store_for_grief = MemoryStore(db_path)

    if store_for_grief is not None and state.open:
        try:
            from brain.forgetting import graveyard as _grave

            grave_entries = _grave.read_all(persona_dir)
            grave_by_id = {e.get("memory_id"): e for e in grave_entries if "memory_id" in e}
            if grave_by_id:
                for arc_id, arc in state.open.items():
                    lost_member_ids = []
                    for member in arc.members:
                        if member.memory_id in grave_by_id:
                            live = store_for_grief.get(member.memory_id)
                            if live is None:
                                lost_member_ids.append(member.memory_id)
                    if lost_member_ids:
                        try:
                            from brain import grief as _grief

                            _grief.handle_recall_touch(
                                touched_ids=sorted(lost_member_ids),
                                graveyard_entries=grave_entries,
                                persona_dir=persona_dir,
                                store=store_for_grief,
                                lived_age_hours_now=lived_age_now
                                if lived_age_now is not None
                                else 0.0,
                                triggering_arc_id=arc_id,
                            )
                        except Exception:  # noqa: BLE001
                            _LOG.exception(
                                "grief.handle_recall_touch (membership) failed for arc_id=%s",
                                arc_id,
                            )
        except Exception:  # noqa: BLE001
            _LOG.exception("narrative_memory membership refresh (grief) raised")

    # --- Close sweep ---
    try:
        for arc_id, arc in list(state.open.items()):
            # Use lived_age_at_join of the most recent member as the "last extended" lived age
            last_ext_lived_age = max(
                (m.lived_age_at_join for m in arc.members), default=arc.lived_age_at_open
            )
            reason: str | None = None
            if not arc.members:
                reason = "all_members_lost"
            elif arc_id in touched_arc_ids:
                # Freshly opened/extended this pass — cannot be stale by definition.
                reason = None
            elif should_close(
                arc, lived_age_now=lived_age_now, last_extended_lived_age=last_ext_lived_age
            ):
                reason = "stale_72h"
            if reason is None:
                continue

            max_emotion, dominant = (0.0, None)
            if store_for_grief is not None:
                max_emotion, dominant = _compute_arc_close_emotion_inputs(
                    arc=arc, store=store_for_grief, persona_dir=persona_dir
                )

            closed = _replace(
                arc,
                state="closed",
                closed_at_iso=_now_iso(),
                lived_age_at_close=lived_age_now,
                max_member_emotion_normalised=max_emotion,
                dominant_non_grief_emotion=dominant,
            )
            state.recently_closed.append(closed)
            del state.open[arc_id]
            append_event(
                persona_dir,
                {
                    "event": "arc_closed",
                    "arc_id": arc_id,
                    "ts_iso": closed.closed_at_iso,
                    "lived_age_hours": lived_age_now if lived_age_now is not None else 0.0,
                    "reason": reason,
                    "final_member_count": len(closed.members),
                },
            )
            counts["closed"] += 1

            # Fault-isolated grief write — after the state mutation + JSONL append.
            # handle_arc_close is internally fault-isolated; this outer try guards
            # only against import-time failures on brain.grief.
            try:
                from brain import grief as _grief

                if store_for_grief is not None:
                    _grief.handle_arc_close(
                        arc=closed, persona_dir=persona_dir, store=store_for_grief
                    )
            except Exception:
                _LOG.exception("grief.handle_arc_close failed for arc_id=%s", arc_id)
    finally:
        if store_for_grief is not None:
            store_for_grief.close()

    # Cap recently_closed
    if len(state.recently_closed) > RECENTLY_CLOSED_CAP:
        state.recently_closed = state.recently_closed[-RECENTLY_CLOSED_CAP:]

    # --- Persist + telemetry ---
    state.last_pass_ts_iso = _now_iso()
    state.replayed = False
    save_state(persona_dir, state)

    duration_ms = (time.monotonic() - t0) * 1000.0
    aggregate: dict[str, int | float | str] = {
        **counts,
        "total_open": len(state.open),
        "duration_ms": duration_ms,
        "type": "arc_update_pass",
    }
    try:
        event_bus.publish(aggregate)
    except Exception:  # noqa: BLE001
        _LOG.exception("arc-update pass event_bus publish raised")

    return aggregate  # type: ignore[return-value]


def _compute_arc_close_emotion_inputs(
    *,
    arc: Arc,
    store: Any,
    persona_dir: Path,
) -> tuple[float, tuple[str, float] | None]:
    """Compute the strongest emotional signal across an arc's members.

    For each member: if a live MemoryStore record exists, use its raw [0, 10]
    emotion values directly. Otherwise, fall back to the graveyard entry's
    `emotion_norm × 10` proxy (reconstructing the original 0-10 emotion at
    the same scale as a live lookup — no salience factor, per spec §3; see
    brain/forgetting/salience.SalienceInputs for the stored fields).

    Returns:
        (max_normalised_emotion, dominant_non_grief_emotion):
            max_normalised_emotion: max emotion across all members, normalised
                to [0, 1] for compute_arc_close_intensity.
            dominant_non_grief_emotion: (emotion_name, intensity_0_to_10) for
                the emotion that produced the max value, or None if no
                member had any non-grief emotion.

    Per spec §3 graveyard-proxy fallback: arcs whose members have largely been
    forgotten still produce honest arc-close grief.
    """
    from brain.forgetting import graveyard as _grave

    grave_entries = _grave.read_all(persona_dir)
    grave_by_id = {e.get("memory_id"): e for e in grave_entries}

    best_emotion_value = 0.0
    best_named: tuple[str, float] | None = None
    for member in arc.members:
        live = store.get(member.memory_id)
        if live is not None and live.emotions:
            for name, val in live.emotions.items():
                if name == "memory_grief":
                    continue
                v = float(val)
                if v > best_emotion_value:
                    best_emotion_value = v
                    best_named = (name, v)
            continue
        entry = grave_by_id.get(member.memory_id)
        if entry is None:
            continue
        inputs = entry.get("salience_inputs_at_drop") or {}
        emotion_norm = float(inputs.get("emotion") or 0.0)
        # Reconstruct the original 0-10 emotion at the same scale as a live lookup.
        # Spec §3 caller responsibility: no salience factor — salience_at_drop is
        # near zero by definition (LOST_THRESHOLD = 0.10 in forgetting/policy.py),
        # so multiplying suppressed grief for fully-forgotten arcs.
        proxy = emotion_norm * 10.0
        if proxy > best_emotion_value:
            best_emotion_value = proxy
            em_dict = entry.get("emotion_at_ingest") or {}
            non_grief = [
                (n, float(v)) for n, v in em_dict.items() if n != "memory_grief" and float(v) > 0.0
            ]
            if non_grief:
                best_named = max(non_grief, key=lambda kv: kv[1])

    return (max(0.0, min(1.0, best_emotion_value / 10.0)), best_named)


def _replace(arc: Arc, **changes: Any) -> Arc:
    """Frozen-Arc replace helper."""
    return Arc(
        id=changes.get("id", arc.id),
        state=changes.get("state", arc.state),
        seed_anchor_type=changes.get("seed_anchor_type", arc.seed_anchor_type),
        seed_anchor_ref=changes.get("seed_anchor_ref", arc.seed_anchor_ref),
        seed_memory_ids=changes.get("seed_memory_ids", arc.seed_memory_ids),
        title=changes.get("title", arc.title),
        opened_at_iso=changes.get("opened_at_iso", arc.opened_at_iso),
        lived_age_at_open=changes.get("lived_age_at_open", arc.lived_age_at_open),
        last_extended_at_iso=changes.get("last_extended_at_iso", arc.last_extended_at_iso),
        closed_at_iso=changes.get("closed_at_iso", arc.closed_at_iso),
        lived_age_at_close=changes.get("lived_age_at_close", arc.lived_age_at_close),
        members=changes.get("members", arc.members),
        max_member_emotion_normalised=changes.get(
            "max_member_emotion_normalised", arc.max_member_emotion_normalised
        ),
        dominant_non_grief_emotion=changes.get(
            "dominant_non_grief_emotion", arc.dominant_non_grief_emotion
        ),
    )


__all__ = ["run_pass"]
