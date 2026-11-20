"""Real implementation of the get_body_state tool.

Reads the emotion engine's short averages, gathers session inputs, projects
BodyState from them, returns its serialized form.

Stub previously returned loaded:False with placeholder fields; the real
impl returns loaded:True so the brain can distinguish "module is real"
from "module is pending". Spec §3.3, §7.3 (no silent failures via
deceptive stub-shape).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.body.state import body_state_from_short
from brain.body.words import count_words_in_session
from brain.emotion import engine
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.utils.memory import days_since_human


def get_body_state(
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,  # noqa: ARG001 — kept for dispatcher signature symmetry
    persona_dir: Path,
    session_hours: float = 0.0,
) -> dict[str, Any]:
    """Return the brain's current body state.

    `session_hours` is injected by the dispatcher when the caller doesn't
    provide it — the dispatcher computes it from the active conversation
    buffer's earliest timestamp (same signal the UI's /persona/state body
    block uses). Caller-provided values take precedence; explicit 0.0
    from a CLI / fresh-install path is honored and
    count_words_in_session falls back to a 1-hour window.
    """
    now = datetime.now(UTC)

    live_engine = engine.engine_for(persona_dir)
    # Readers show absent while a recovery rebuild is pending. Accepted edge: while a recovery rebuild is pending
    # or retrying, read() returns nothing and the body emotions are absent.
    short = engine.short_values(live_engine.read())

    days_since = days_since_human(store, now=now, persona_dir=persona_dir)
    words = count_words_in_session(
        store,
        persona_dir=persona_dir,
        session_hours=session_hours,
        now=now,
    )

    body = body_state_from_short(
        short,
        feels_high=live_engine.feels_high,
        session_hours=session_hours,
        words_written=words,
        days_since_contact=days_since,
        now=now,
    )
    return body.to_dict()
