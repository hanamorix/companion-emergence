"""The incident-regime builder — an "aged persona" fixture via the REAL fold.

Generalizes ``t3_setup.build``. Reaches an authentic multiply-folded (generation >= 2) rolling
summary BY CONSTRUCTION, not by live-running a long session: seed a synthetic session under a known
sid, then run the REAL product fold (``brain.chat.compaction.compact_conversation``,
``fold_existing_summary=True``) REPEATEDLY in batches, advancing ``now`` so the aged middle drains
while the count-protected tail stays. Optionally seeds an interior-continuity block.

**Fidelity:** the summary is produced by the REAL fold, not a hand-written proxy. The fold makes a
provider call, so the caller INJECTS the provider — a fake/deterministic one keeps a test token-free;
a live run passes ``brain.chat.compaction.build_compaction_provider(persona_dir)``. Unlike the port,
this builder has NO pinned-claude-binary guard (stage-3 R3), so it runs on any CI box.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .config import SYNTHETIC_USER

MIN_KEEP_TAIL = 40


def _persona_user_name(persona_dir: Path) -> str:
    """The persona's configured ``user_name`` (falls back to the synthetic default)."""
    cfg = persona_dir / "persona_config.json"
    try:
        data = json.loads(cfg.read_text())
        name = data.get("user_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except (OSError, json.JSONDecodeError):
        pass
    return SYNTHETIC_USER


@dataclass
class IncidentSpec:
    """How to build the aged/compacted regime."""

    user_beats: list[str] = field(default_factory=list)
    assistant_beats: list[str] = field(default_factory=list)
    interior_traces: list[str] = field(default_factory=list)
    session_turns: int = 120
    fold_passes: int = 11
    fold_batch: int = 8          # turns folded per pass (max_compact_turns)
    min_keep_tail: int = MIN_KEEP_TAIL
    seed_interior: bool = True


@dataclass
class IncidentResult:
    sid: str
    generation: int
    fold_passes_run: int
    summary_rows: int
    tail_turns: int
    interior_block_chars: int
    folds: list[dict] = field(default_factory=list)


def _gen_session(spec: IncidentSpec, sid: str, base: datetime) -> list[dict]:
    """Alternating user/assistant records spanning several synthetic days so folds bite.

    The oldest ``session_turns - min_keep_tail`` turns are spread across many days (so they age past
    the 24h cutoff over folds); the final ``min_keep_tail`` pack into the last hours (protected tail).
    """
    n = spec.session_turns
    records: list[dict] = []
    old_n = max(0, n - spec.min_keep_tail)
    ub = spec.user_beats or ["(no user beats configured)"]
    ab = spec.assistant_beats or ["(no assistant beats configured)"]
    for i in range(n):
        speaker = "user" if i % 2 == 0 else "assistant"
        beats = ub if speaker == "user" else ab
        text = beats[(i // 2) % len(beats)]
        if i < old_n:
            ts = base - timedelta(hours=6 * (old_n - i) + 6)
        else:
            ts = base - timedelta(minutes=3 * (n - i))
        records.append({
            "session_id": sid, "speaker": speaker,
            "text": f"{text} (t{i})", "ts": ts.isoformat(timespec="seconds"),
        })
    return records


def build_compacted_state(persona_dir: Path, spec: IncidentSpec, provider: object) -> IncidentResult:
    """Build the incident-regime state in ``persona_dir`` using the REAL fold + injected provider."""
    from brain.chat.compaction import compact_conversation
    from brain.ingest.buffer import ingest_turn, read_session, write_cursor
    from brain.memory.store import Memory, MemoryStore
    from brain.monologue.ambient import build_interior_continuity_block
    from brain.monologue.trace import MONOLOGUE_TRACE_TYPE

    # L3: without a middle to fold, this would silently produce a gen-0 "incident" (no fold). Fail
    # loud so a misconfigured spec can't masquerade as an aged persona.
    if spec.session_turns <= spec.min_keep_tail:
        raise ValueError(
            f"session_turns ({spec.session_turns}) must exceed min_keep_tail ({spec.min_keep_tail}) "
            "so there is an aged middle to fold — otherwise no incident regime is built."
        )

    # A3: thread the persona's CONFIGURED user_name into the interior block instead of hardcoding.
    user_name = _persona_user_name(persona_dir)

    # sid MUST be a canonical UUID (the WS /stream/{sid} gate is UUID-only).
    sid = str(uuid.uuid4())
    base = datetime.now(UTC)
    records = _gen_session(spec, sid, base)
    for r in records:
        ingest_turn(persona_dir, r)
    newest_ts = max(r["ts"] for r in records)
    write_cursor(persona_dir, sid, newest_ts)

    # Batch-drain the aged middle: `now` fixed far ahead so the whole non-protected middle is aged
    # past the 24h cutoff each pass; each pass folds the oldest FOLD_BATCH aged turns and re-merges
    # the prior summary (fold=True) -> generation increments per pass to a multiply-folded state.
    fold_now = base + timedelta(days=400)
    folds: list[dict] = []
    for p in range(spec.fold_passes):
        res = compact_conversation(
            persona_dir, sid,
            older_than=timedelta(hours=24), fold_existing_summary=True,
            provider=provider, min_keep_tail=spec.min_keep_tail,
            max_compact_turns=spec.fold_batch, now=fold_now,
        )
        folds.append({
            "pass": p, "compacted": res.compacted, "gen": res.new_gen,
            "n": res.compacted_n, "fell_soft": res.fell_soft, "reason": res.reason,
        })
        if not res.compacted and res.reason == "nothing_aged":
            break

    turns = read_session(persona_dir, sid)
    summary_rows = [t for t in turns if t.get("speaker") == "summary"]
    tail = [t for t in turns if t.get("speaker") != "summary"]
    generation = max((f["gen"] for f in folds), default=0)

    interior_block = ""
    if spec.seed_interior and spec.interior_traces:
        store = MemoryStore(db_path=persona_dir / "memories.db")
        try:
            tbase = datetime.now(UTC)
            seeds = list(spec.interior_traces)
            for i, tx in enumerate(seeds):
                m = Memory.create_new(
                    content=tx, memory_type=MONOLOGUE_TRACE_TYPE, domain="interior",
                    emotions={"focus": 0.5, "warmth": 0.3}, tags=["interior", "seed"],
                    importance=5.0,
                )
                m.created_at = tbase - timedelta(minutes=len(seeds) - i)
                store.create(m)
            interior_block = build_interior_continuity_block(store, user_name=user_name)
        finally:
            store.close()

    return IncidentResult(
        sid=sid,
        generation=generation,
        fold_passes_run=len(folds),
        summary_rows=len(summary_rows),
        tail_turns=len(tail),
        interior_block_chars=len(interior_block),
        folds=folds,
    )
