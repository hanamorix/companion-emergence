"""Real implementation of the get_body_state tool.

Reads aggregated emotion state (with climax reset already applied by
aggregate_state), gathers session inputs, computes BodyState, returns
its serialized form.

Stub previously returned loaded:False with placeholder fields; the real
impl returns loaded:True so the brain can distinguish "module is real"
from "module is pending". Spec §3.3, §7.3 (no silent failures via
deceptive stub-shape).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.body.state import compute_body_state
from brain.body.words import count_words_in_session
from brain.emotion.aggregate import aggregate_state
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore, _row_to_memory
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

    # Aggregate emotion state from recent EMOTION-BEARING memories — matches
    # _build_emotion_summary and _build_body_block in chat/prompt.py, and
    # _build_emotions / _build_body in bridge/persona_state.py.
    #
    # #90: the comment here used to claim it matched _build_emotion_summary's
    # window, but that function had already been filtered to emotion-bearing
    # rows and this had not — so the two disagreed. Unfiltered, this returned {}
    # on a steady-state brain: measured on the real persona, 48 of the 50 most
    # recent memories were emotionless `heartbeat` rows.
    #
    # Filtering cannot change the aggregation — an emotionless memory is a no-op
    # in aggregate_state's max-pool — it only stops the window being spent on
    # rows that cannot contribute.
    rows = store._conn.execute(  # noqa: SLF001 — internal same-tier access
        "SELECT * FROM memories "
        "WHERE active = 1 "
        "AND emotions_json IS NOT NULL "
        "AND emotions_json != '{}' "
        "ORDER BY created_at DESC LIMIT 200"
    ).fetchall()
    memories = [_row_to_memory(row) for row in rows]
    state = aggregate_state(memories)  # already applies climax reset

    days_since = days_since_human(store, now=now, persona_dir=persona_dir)
    words = count_words_in_session(
        store,
        persona_dir=persona_dir,
        session_hours=session_hours,
        now=now,
    )

    body = compute_body_state(
        emotions=state.emotions,
        session_hours=session_hours,
        words_written=words,
        days_since_contact=days_since,
        now=now,
    )
    return body.to_dict()
