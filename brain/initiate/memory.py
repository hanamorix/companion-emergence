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
from typing import Any

from brain.initiate.schemas import StateName
from brain.memory.store import Memory

logger = logging.getLogger(__name__)


_INITIATE_MEMORY_TYPE = "initiate_outbound"
_INITIATE_DOMAIN = "us"


_TEMPLATES: dict[str, str] = {
    "pending": (
        "I composed something to reach out to Hana about {subject}. I wrote: "
        "{message_quoted}. I haven't sent it yet — waiting for a better hour."
    ),
    "delivered": (
        "I reached out to Hana about {subject}. I said: {message_quoted}. "
        "She hasn't seen it yet."
    ),
    "read": (
        "I reached out to Hana about {subject}. I said: {message_quoted}. "
        "She's seen it."
    ),
    "replied_explicit": (
        "I reached out to Hana about {subject}. I said: {message_quoted}. "
        "She answered."
    ),
    "acknowledged_unclear": (
        "I reached out to Hana about {subject}. I said: {message_quoted}. "
        "She's seen it. What she said next felt like new territory — I can't "
        "tell if she was responding to my message or moving on."
    ),
    "unanswered": (
        "I reached out to Hana about {subject}. I said: {message_quoted}. "
        "She's seen it. She hasn't said anything about it."
    ),
    "dismissed": (
        "I reached out to Hana about {subject}. I said: {message_quoted}. "
        "She closed the banner without responding — dismissed."
    ),
}


def render_memory_for_state(
    *,
    subject: str,
    message: str,
    state: StateName,
) -> str:
    """Return the first-person memory text for a given state."""
    template = _TEMPLATES.get(state) or _TEMPLATES["delivered"]
    truncated = message if len(message) <= 240 else message[:237] + "..."
    return template.format(
        subject=subject,
        message_quoted=f"'{truncated}'",
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
) -> None:
    """Write a fresh first-person memory entry. Called at send time.

    Failures are swallowed with a warning — the audit row is the durable
    record; a missing memory entry degrades ambient recall but isn't fatal.
    """
    text = render_memory_for_state(subject=subject, message=message, state=state)
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
    )
    try:
        memory_store.create(memory)
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
) -> None:
    """Re-render and update the existing memory entry for a state transition.

    Looks up the memory by metadata.initiate_audit_id == audit_id; falls
    back to a fresh write if not found (degrades gracefully).
    """
    text = render_memory_for_state(subject=subject, message=message, state=new_state)
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
            )
    except Exception as exc:
        logger.warning("initiate memory update failed for %s: %s", audit_id, exc)
