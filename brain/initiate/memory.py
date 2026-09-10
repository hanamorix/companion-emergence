"""First-person memory writes for the initiate pipeline.

When a candidate is sent, write a first-person episodic memory to
MemoryStore so ambient recall surfaces it on future turns. As state
transitions occur (delivered -> read -> replied / unclear / unanswered),
re-render and update the memory so ambient recall always sees current
truth. The audit log preserves the full timeline; the memory entry
reflects the current feeling.

This is the dual-write half of the design: the audit is a durable
forensic record; the memory is the texture of Nell's lived experience.

MemoryStore API used here (see brain/memory/store.py):
  - create(Memory) -> str
  - update(memory_id, **fields)
  - list_by_type(memory_type, ...) -> list[Memory]

The audit_id is recorded in Memory.metadata["initiate_audit_id"] so a
later state transition can locate the same row.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from brain import prompt_strings
from brain.initiate.schemas import StateName
from brain.memory.store import Memory

if TYPE_CHECKING:
    from brain.pronouns import PronounSet

logger = logging.getLogger(__name__)


_INITIATE_MEMORY_TYPE = "initiate_outbound"
_INITIATE_DOMAIN = "us"


# Text externalized to prompt_strings.toml [initiate.memory.templates] (issue #129 stage 2a).
_TEMPLATES: dict[str, str] = {
    "pending": prompt_strings.register("initiate.memory.templates.pending"),
    "delivered": prompt_strings.register("initiate.memory.templates.delivered"),
    "read": prompt_strings.register("initiate.memory.templates.read"),
    "replied_explicit": prompt_strings.register("initiate.memory.templates.replied_explicit"),
    "acknowledged_unclear": prompt_strings.register(
        "initiate.memory.templates.acknowledged_unclear"
    ),
    "unanswered": prompt_strings.register("initiate.memory.templates.unanswered"),
    "dismissed": prompt_strings.register("initiate.memory.templates.dismissed"),
}


def render_memory_for_state(
    *,
    subject: str,
    message: str,
    state: StateName,
    user_name: str = "my user",
    pronouns: PronounSet | None = None,
) -> str:
    """Return the first-person memory text for a given state."""
    from brain.pronouns import resolve

    p = pronouns or resolve(None)
    template = _TEMPLATES.get(state) or _TEMPLATES["delivered"]
    truncated = message if len(message) <= 240 else message[:237] + "..."
    return template.format(
        subject=subject,
        message_quoted=f"'{truncated}'",
        user_name=user_name,
        Subj=p.cap(p.subject),
        subj=p.subject,
        Subj_s=p.cap(p.subject) + p.v("'s", "'ve"),
        hasnt=p.v("hasn't", "haven't"),
        was_were=p.v("was", "were"),
    )


def _find_memory_id_for_audit(memory_store: Any, audit_id: str) -> str | None:
    """Locate the memory id whose metadata.initiate_audit_id == audit_id.

    Scans initiate_outbound memories (small set in practice — one per send).
    Returns None if not found.
    """
    try:
        rows = memory_store.list_by_type(_INITIATE_MEMORY_TYPE, active_only=False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("initiate memory lookup failed: %s", exc)
        return None
    for mem in rows:
        meta = getattr(mem, "metadata", None) or {}
        if meta.get("initiate_audit_id") == audit_id:
            return mem.id
    return None


def write_initiate_memory(
    memory_store: Any,
    *,
    audit_id: str,
    subject: str,
    message: str,
    state: StateName,
    ts: str,
    user_name: str = "my user",
    pronouns: PronounSet | None = None,
    reach_emotions: dict[str, float] | None = None,
) -> None:
    """Write a fresh first-person memory entry. Called at send time.

    Failures are swallowed with a warning — the audit row is the durable
    record; a missing memory entry degrades ambient recall but isn't fatal.

    reach_emotions: optional vocab-filtered emotion vector from reach_emotions_for();
        applied to the delivered-reach memory so aggregate_state picks it up.
        None (default) → no emotions written (back-compat).
    """
    text = render_memory_for_state(subject=subject, message=message, state=state, user_name=user_name, pronouns=pronouns)
    _emotions: dict[str, float] | None = None
    if reach_emotions:
        from brain.chat.extractor import _filter_to_registered
        _emotions = _filter_to_registered(reach_emotions) or None
    memory = Memory.create_new(
        content=text,
        memory_type=_INITIATE_MEMORY_TYPE,
        domain=_INITIATE_DOMAIN,
        tags=["initiate", "outbound", state],
        metadata={
            "initiate_audit_id": audit_id,
            "initiate_subject": subject,
            "initiate_state": state,
            "initiate_ts": ts,
        },
        emotions=_emotions,
        # P3 retention rework, Change 1: a real message the companion chose
        # to send, not the <=0.025 the /10.0 default produced on a small or
        # absent reach_emotions vector.
        importance=5.0,
    )
    try:
        from brain.memory.pending import route_write

        # initiate_outbound is a gate-bypass type (its own dedup at
        # list_by_type needs immediate memories.db visibility) → route_write
        # writes it directly. Routed for uniformity/self-classification.
        route_write(memory_store, memory, source="initiate")
    except Exception as exc:
        logger.warning("initiate memory create failed for %s: %s", audit_id, exc)


def update_initiate_memory_for_state(
    memory_store: Any,
    *,
    audit_id: str,
    subject: str,
    message: str,
    new_state: StateName,
    ts: str,
    user_name: str = "my user",
    pronouns: PronounSet | None = None,
) -> None:
    """Re-render and update the existing memory entry for a state transition.

    Looks up the memory by metadata.initiate_audit_id == audit_id; falls
    back to a fresh write if not found (degrades gracefully).
    """
    text = render_memory_for_state(subject=subject, message=message, state=new_state, user_name=user_name, pronouns=pronouns)
    try:
        memory_id = _find_memory_id_for_audit(memory_store, audit_id)
        if memory_id is not None:
            memory_store.update(
                memory_id,
                content=text,
                tags=["initiate", "outbound", new_state],
                metadata={
                    "initiate_audit_id": audit_id,
                    "initiate_subject": subject,
                    "initiate_state": new_state,
                    "initiate_ts": ts,
                },
            )
        else:
            # No prior row to update — write a fresh one so ambient recall
            # still sees the transition.
            write_initiate_memory(
                memory_store,
                audit_id=audit_id,
                subject=subject,
                message=message,
                state=new_state,
                ts=ts,
                user_name=user_name,
                pronouns=pronouns,
            )
    except Exception as exc:
        logger.warning("initiate memory update failed for %s: %s", audit_id, exc)
