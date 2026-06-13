"""Resolution tracking — sustained gap crystallisation (self-model §2, §6 R-E4).

`check_and_emit_resolution(prior, new, *, persona_dir, session_id)` is the
single entry point.  It is called by the cadence tick (Task 7) after the gap
has been recomputed.  In this task it is implemented as a standalone testable
function; Task 7 wires it into the supervisor.

Two resolution paths (R-E4 — not tool-hostage):
  PATH A — reconcile:   gap WAS open + sustained >= _SUSTAINED_TICKS and is
                        now ``acknowledged`` or ``dismissed`` (tool was called).
  PATH B — natural:     gap WAS open + sustained >= _SUSTAINED_TICKS and is
                        now effectively closed because the new gap's magnitude
                        collapsed to zero (declared re-matched derived with no
                        tool call needed).

Either path calls `_emit_resolution(persona_dir, resolved_gap, path)`:
  1. Commits a tiny ``self_model_resolved`` memory to get a memory_id.
  2. Queues a soul candidate via `brain.ingest.soul_queue.queue_soul_candidate`
     (existing pipeline — same mechanics as monologue crystallisation).
  3. Publishes a ``self_model_gap_resolved`` feed event via
     `brain.bridge.events.publish`.

Audit counter (dead-loop observability, §6 R-E4):
  A small JSON counter at <persona_dir>/self_model_audit.json tracks:
    gaps_surfaced  — incremented by the cadence tick when a gap is active
    reconciles_called — incremented by reconcile_self_read on any successful call
  This makes "she surfaces the gap repeatedly but never reconciles" observable
  without adding any new LLM spend.  The counter is intentionally cheap: no
  JSONL streaming reader, just a bounded 2-integer JSON object.

Const:
  _SUSTAINED_TICKS = 3  — a gap must be open for at least this many consecutive
  cadence ticks before a resolution crystallises into a soul candidate.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from brain.self_model.gap import Gap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUSTAINED_TICKS: int = 3
"""Gap must be open for this many consecutive cadence ticks before crystallising."""

_RESOLVED_STATUSES = frozenset({"acknowledged", "dismissed"})
"""Gap statuses produced by the reconcile tool (PATH A)."""

_AUDIT_FILENAME = "self_model_audit.json"

# ---------------------------------------------------------------------------
# Audit counter
# ---------------------------------------------------------------------------


def _audit_path(persona_dir: Path) -> Path:
    return persona_dir / _AUDIT_FILENAME


def load_audit(persona_dir: Path) -> dict:
    """Return {gaps_surfaced, reconciles_called}.  Missing file → zeros (fail-open)."""
    p = _audit_path(persona_dir)
    if not p.exists():
        return {"gaps_surfaced": 0, "reconciles_called": 0}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {
            "gaps_surfaced": int(raw.get("gaps_surfaced", 0)),
            "reconciles_called": int(raw.get("reconciles_called", 0)),
        }
    except Exception:  # noqa: BLE001
        return {"gaps_surfaced": 0, "reconciles_called": 0}


def _save_audit(persona_dir: Path, audit: dict) -> None:
    persona_dir.mkdir(parents=True, exist_ok=True)
    p = _audit_path(persona_dir)
    try:
        p.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        logger.warning("self_model: failed to save audit counter: %s", exc)


def increment_gaps_surfaced(persona_dir: Path) -> None:
    """Increment the gaps_surfaced counter.  Called by the cadence tick."""
    audit = load_audit(persona_dir)
    audit["gaps_surfaced"] = audit["gaps_surfaced"] + 1
    _save_audit(persona_dir, audit)


def increment_reconciles(persona_dir: Path) -> None:
    """Increment the reconciles_called counter.  Called by reconcile_self_read."""
    audit = load_audit(persona_dir)
    audit["reconciles_called"] = audit["reconciles_called"] + 1
    _save_audit(persona_dir, audit)


# ---------------------------------------------------------------------------
# Resolution emission
# ---------------------------------------------------------------------------


def _emit_resolution(
    persona_dir: Path,
    resolved_gap: Gap,
    resolution_path: str,
    session_id: str,
) -> None:
    """Queue a soul candidate + publish a feed event for a resolved gap.

    Step 1: commit a tiny ``self_model_resolved`` memory → get a memory_id.
    Step 2: build an ExtractedItem and call queue_soul_candidate.
    Step 3: publish ``self_model_gap_resolved`` via brain.bridge.events.

    Fail-soft at every step: failures are logged at WARN, never re-raised.
    """
    from brain.bridge import events
    from brain.ingest.soul_queue import DEFAULT_SOUL_THRESHOLD, queue_soul_candidate
    from brain.ingest.types import ExtractedItem
    from brain.memory.store import Memory, MemoryStore

    # Build human-readable text describing the resolved self-knowledge gap.
    if resolved_gap.per_channel:
        channels_str = ", ".join(
            f"{ch} (Δ{delta:+.2f})" for ch, delta in sorted(resolved_gap.per_channel.items())
        )
        gap_desc = f"channels: {channels_str}; magnitude {resolved_gap.magnitude:.2f}"
    elif resolved_gap.unnamed_pressure > 0:
        gap_desc = f"unnamed pressure {resolved_gap.unnamed_pressure:.2f}"
    else:
        gap_desc = "emotional gap resolved"

    path_label = "through self-reflection" if resolution_path == "reconcile" else "naturally"
    text = (
        f"[self-model] A sustained emotional gap resolved {path_label}. "
        f"Gap: {gap_desc}. "
        f"She came to see herself more clearly."
    )

    # Step 1 — commit a placeholder memory to obtain a memory_id.
    memory_id: str | None = None
    try:
        store = MemoryStore(persona_dir / "memories.db")
        try:
            mem = Memory.create_new(
                content=text,
                memory_type="self_model_resolved",
                domain="self",
                importance=9.0,  # sustained + resolved → high-importance self-knowledge
            )
            store.create(mem)
            memory_id = mem.id
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("self_model: resolve memory write failed: %s", exc)

    if memory_id is None:
        logger.warning("self_model: skipping soul candidate — no memory_id")
        return

    # Step 2 — queue the soul candidate.
    item = ExtractedItem(
        text=text,
        label="observation",
        importance=9,
    )
    if item.importance >= DEFAULT_SOUL_THRESHOLD:
        try:
            queue_soul_candidate(
                persona_dir,
                memory_id=memory_id,
                item=item,
                session_id=session_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("self_model: queue_soul_candidate failed: %s", exc)

    # Step 3 — publish feed event (fail-soft; never crash on a publish).
    try:
        events.publish(
            "self_model_gap_resolved",
            resolution_path=resolution_path,
            magnitude=resolved_gap.magnitude,
            channels=list(resolved_gap.per_channel.keys()),
            memory_id=memory_id,
            session_id=session_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("self_model: publish self_model_gap_resolved failed: %s", exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def check_and_emit_resolution(
    prior: Gap | None,
    new: Gap | None,
    *,
    persona_dir: Path,
    session_id: str,
) -> None:
    """Decide if a sustained gap just resolved, and if so emit the downstream effects.

    Called from the cadence tick (Task 7) after recomputing the gap.

    **The decision is made on ``prior``, not ``new``** — this is load-bearing.
    The resolution *evidence* (how long the gap sustained, and whether the
    reconcile tool acted on it) lives on the PRIOR gap:

      - ``prior.sustained_ticks`` is the count accumulated across the consecutive
        ticks the gap stayed open. ``new`` is recomputed fresh each tick; on a
        reconcile it resets (different channel set / non-open prior), and on
        natural reconvergence it collapses to a zero-magnitude fresh gap. So
        ``new`` never carries the sustained evidence.
      - The reconcile tool mutates the PERSISTED current_gap (status →
        acknowledged/dismissed). That persisted gap is loaded as ``prior`` on the
        NEXT tick. So a reconcile shows up as ``prior.status`` being resolved,
        not ``new.status``.

    Resolution logic:
      - ``prior`` must be non-None and sustained >= _SUSTAINED_TICKS.
      - PATH A (reconcile):  ``prior.status`` is in _RESOLVED_STATUSES
                             (the tool acted on it last tick).
      - PATH B (natural):    ``prior`` was still open, and the gap is now closed —
                             ``new`` is None, or ``new.magnitude == 0`` and
                             ``new.unnamed_pressure == 0`` (declared re-matched
                             derived with no tool call).

    A gap that resolves BEFORE reaching _SUSTAINED_TICKS is transient — it
    silently passes without crystallising.  This avoids noise from brief
    momentary divergences.

    Args:
        prior:       The gap state from the PREVIOUS cadence tick — the carrier
                     of the resolution evidence (sustained_ticks + reconciled
                     status).
        new:         The freshly recomputed gap for THIS tick (or None when no
                     gap is active). Used only to detect natural closure.
        persona_dir: Persona directory for all I/O.
        session_id:  Used in the soul candidate record.
    """
    if prior is None:
        return

    # The gap must have been sustained long enough to crystallise. The sustained
    # evidence lives on PRIOR (new is a fresh recompute and does not carry it).
    if prior.sustained_ticks < _SUSTAINED_TICKS:
        return

    # PATH A — reconcile tool was called last tick: PRIOR is now acknowledged
    # or dismissed (the tool mutated the persisted gap; it loaded back as prior).
    if prior.status in _RESOLVED_STATUSES:
        logger.info(
            "self_model: gap resolved via reconcile (sustained=%d, magnitude=%.3f)",
            prior.sustained_ticks,
            prior.magnitude,
        )
        _emit_resolution(persona_dir, prior, "reconcile", session_id)
        return

    # PATH B — natural reconvergence: prior was still open, and the gap has now
    # closed (no new gap, or its magnitude collapsed to zero) with no tool call.
    new_closed = new is None or (new.magnitude == 0.0 and new.unnamed_pressure == 0.0)
    if prior.status == "open" and new_closed:
        logger.info(
            "self_model: gap resolved naturally (sustained=%d, prior_magnitude=%.3f)",
            prior.sustained_ticks,
            prior.magnitude,
        )
        _emit_resolution(persona_dir, prior, "natural", session_id)
        return
