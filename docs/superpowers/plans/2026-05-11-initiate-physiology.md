# Initiate Physiology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the v0.0.9 initiate physiology — autonomous outbound channel, voice-edit proposals, draft space, brain-side verify path, and user-local timezone awareness — implementing four of Nell's five manifesto items from her 2026-05-11 conversation with Hana.

**Architecture:** Mirrors `_run_soul_review_tick` exactly. Events emit candidates into `initiate_candidates.jsonl`; a supervisor tick reviews queued candidates with cost-cap + cooldown gates; decisions land in `initiate_audit.jsonl` + `MemoryStore`. Three-prompt composition pipeline (subject / tone / decision) prevents emotional-state pollution. Voice-edit accepts earn a three-place write (audit + episodic memory + SoulStore `voice_evolution` crystallization). Draft space is a separate thin pipe — failed-to-promote events become markdown fragments in `draft_space.md`. Verify path is a hybrid of always-on ambient slice + on-demand tools.

**Tech Stack:** Python 3.13 (brain), pytest + uv (testing), FastAPI (bridge), React + TypeScript + Tauri (renderer), SQLite (SoulStore + MemoryStore), JSONL (audit + queue files).

**Spec:** `docs/superpowers/specs/2026-05-11-initiate-physiology-design.md`

**Project rules — non-negotiable** (from `CLAUDE.md` + auto-memory):
- Full pytest suite gate after **every** commit, not just per-task subset (Hana's strict rule).
- Never `git pull origin` (local + public histories are permanently divergent).
- Public-facing doc edits go to `.public-sync/<name>-public.md`, never the live file.
- Three-file version bump in lockstep only at release-cut time.
- `with open(path, "a")` per append — every writer reopens (already verified in v0.0.8 retention work).

**Execution prerequisite:** Create a worktree before starting Phase 1:

```bash
git worktree add .worktrees/v009-initiate -b feature/v009-initiate-physiology main
cd .worktrees/v009-initiate
uv sync --extra dev --reinstall-package companion-emergence
```

All file paths in this plan are relative to the worktree root.

---

## File structure

### New backend modules

```
brain/initiate/
  __init__.py          # public API exports
  emit.py              # deterministic candidate emission (no LLM)
  audit.py             # audit log read/write + state transitions
  gates.py             # cost-cap + cooldown enforcement (user-local time)
  compose.py           # three-prompt composition pipeline
  review.py            # supervisor tick orchestration
  ambient.py           # always-on verify slice builder
  voice_reflection.py  # daily voice-edit reflection tick
  draft.py             # draft space emission (failed-to-promote routing)
  schemas.py           # dataclasses for IniateCandidate, AuditRow, etc.
```

### New test modules

```
tests/unit/brain/initiate/
  __init__.py
  test_emit.py
  test_audit.py
  test_gates.py
  test_compose.py
  test_review.py
  test_ambient.py
  test_voice_reflection.py
  test_draft.py
  test_schemas.py
tests/integration/initiate/
  __init__.py
  test_event_to_audit.py
  test_voice_edit_three_place_write.py
  test_ask_pattern_hook.py
```

### Modified backend files

| File | Why |
|---|---|
| `brain/bridge/supervisor.py` | Wire `_run_initiate_review_tick` + `_run_voice_reflection_tick`. New cadence params + last_at trackers. |
| `brain/engines/dream.py` | Call `emit_initiate_candidate(source="dream", ...)` after dream log write. |
| `brain/growth/crystallizers/reflex.py` | Call emit after `SoulStore` write. |
| `brain/growth/crystallizers/creative_dna.py` | Same. |
| `brain/growth/crystallizers/vocabulary.py` | Same. |
| `brain/engines/heartbeat.py` | Compute rolling baseline; emit emotion-spike candidate when delta_sigma ≥ 1.5. |
| `brain/soul/store.py` | Add `voice_evolution` table + accessors. |
| `brain/chat/engine.py` | Call `build_outbound_recall_block` for always-on verify slice. |
| `brain/bridge/provider.py` | Register `recall_initiate_audit` + `recall_soul_audit` + `recall_voice_evolution` tools. |
| `brain/health/log_rotation.py` | Add `initiate_audit.jsonl` to yearly-archive list. |
| `brain/cli.py` | New `nell initiate` subcommand tree (audit, audit --full, candidates, voice-evolution). |
| `brain/persona_config.py` | New optional fields for initiate cadence overrides. |

### Modified renderer files

| File | Why |
|---|---|
| `app/src/components/ChatPanel.tsx` | Render initiate-message banners; publish read/dismissed events. |
| `app/src-tauri/src/lib.rs` | OS notification handler for `send_notify` urgency. |

### New renderer files

```
app/src/components/InitiateBanner.tsx        # message banner with ↩ affordance
app/src/components/InitiateBanner.test.tsx
app/src/components/VoiceEditPanel.tsx        # side panel for voice-edit review
app/src/components/VoiceEditPanel.test.tsx
app/src/components/DraftSpacePanel.tsx       # side panel for draft fragments
app/src/components/DraftSpacePanel.test.tsx
```

---

## Phase ordering rationale

11 phases, ordered so each phase produces tested infrastructure the next phase depends on:

1. **Foundations** — schemas, emit, audit, gates. Pure functions; no supervisor wiring.
2. **Composition pipeline** — three prompts with `FakeProvider`. No queue yet.
3. **Review tick + supervisor wiring** — the orchestrator that ties Phase 1 + 2 together.
4. **Event emitters** — dream / crystallization / emotion-spike sources call `emit_initiate_candidate`.
5. **Memory writes + state transitions** — send writes episodic memory; transitions mutate it.
6. **Voice-edit proposals** — reflection tick + three-place write + decision-prompt for `voice_edit_proposal` kind.
7. **Verify path** — always-on ambient slice + on-demand tools.
8. **Draft space** — failed-to-promote routing + markdown append + banner trigger.
9. **Frontend** — InitiateBanner, VoiceEditPanel, DraftSpacePanel, ChatPanel integration.
10. **CLI** — `nell initiate` subcommand tree.
11. **Integration tests + docs + roadmap entry.**

**Gate after every phase:** full `uv run pytest` suite green. No exceptions.

---

## Phase 1: Foundations (schemas, emit, audit, gates)

Pure-function modules with full test coverage before any wiring. None of these depend on the supervisor or any LLM call.

### Task 1: Schemas

**Files:**
- Create: `brain/initiate/__init__.py` (empty for now)
- Create: `brain/initiate/schemas.py`
- Create: `tests/unit/brain/initiate/__init__.py` (empty)
- Create: `tests/unit/brain/initiate/test_schemas.py`

- [ ] **Step 1.1: Write the failing test**

```python
# tests/unit/brain/initiate/test_schemas.py
"""Tests for brain.initiate.schemas — candidate + audit dataclasses."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from brain.initiate.schemas import (
    AuditRow,
    EmotionalSnapshot,
    InitiateCandidate,
    SemanticContext,
    StateTransition,
)


def test_emotional_snapshot_round_trips_to_dict():
    snap = EmotionalSnapshot(
        vector={"joy": 4, "longing": 7},
        rolling_baseline_mean=5.1,
        rolling_baseline_stdev=1.3,
        current_resonance=7.4,
        delta_sigma=1.77,
    )
    d = snap.to_dict()
    assert d["vector"] == {"joy": 4, "longing": 7}
    assert d["delta_sigma"] == 1.77
    assert EmotionalSnapshot.from_dict(d) == snap


def test_initiate_candidate_round_trips_to_jsonl():
    ts = "2026-05-11T14:32:04.123456+00:00"
    cand = InitiateCandidate(
        candidate_id="ic_2026-05-11T14-32-04_a3f1",
        ts=ts,
        kind="message",
        source="dream",
        source_id="dream_abc123",
        emotional_snapshot=EmotionalSnapshot(
            vector={"longing": 7},
            rolling_baseline_mean=5.0,
            rolling_baseline_stdev=1.0,
            current_resonance=7.4,
            delta_sigma=2.4,
        ),
        semantic_context=SemanticContext(
            linked_memory_ids=["m_xyz"],
            topic_tags=["dream", "workshop"],
        ),
        claimed_at=None,
    )
    line = cand.to_jsonl()
    reconstructed = InitiateCandidate.from_jsonl(line)
    assert reconstructed == cand


def test_audit_row_state_transitions_append():
    row = AuditRow(
        audit_id="ia_xyz",
        candidate_id="ic_abc",
        ts="2026-05-11T14:47:09+00:00",
        kind="message",
        subject="the dream",
        tone_rendered="the dream from this morning landed somewhere",
        decision="send_quiet",
        decision_reasoning="resonance is real but hour is late",
        gate_check={"allowed": True, "reason": None},
        delivery=None,
    )
    row.record_transition("delivered", "2026-05-11T14:47:09.5+00:00")
    row.record_transition("read", "2026-05-11T18:34:21+00:00")
    assert row.delivery is not None
    assert row.delivery["current_state"] == "read"
    assert len(row.delivery["state_transitions"]) == 2
    assert row.delivery["state_transitions"][0]["to"] == "delivered"


def test_initiate_candidate_id_generation():
    """candidate_id must be sortable and unique."""
    from brain.initiate.schemas import make_candidate_id

    a = make_candidate_id(datetime(2026, 5, 11, 14, 32, 4, tzinfo=timezone.utc))
    b = make_candidate_id(datetime(2026, 5, 11, 14, 32, 5, tzinfo=timezone.utc))
    assert a < b  # sortable
    assert a != b  # unique
    assert a.startswith("ic_")
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_schemas.py -v
```

Expected: `ModuleNotFoundError: No module named 'brain.initiate'`

- [ ] **Step 1.3: Write minimal implementation**

```python
# brain/initiate/__init__.py
"""Initiate physiology — autonomous outbound channel.

Mirrors the _run_soul_review_tick architecture from v0.0.4. Events emit
candidates into initiate_candidates.jsonl; a supervisor tick reviews
queued candidates with cost-cap + cooldown gates; decisions land in
initiate_audit.jsonl + MemoryStore.

Spec: docs/superpowers/specs/2026-05-11-initiate-physiology-design.md
"""
```

```python
# brain/initiate/schemas.py
"""Data structures for the initiate pipeline.

Three core types:

* InitiateCandidate — what gets queued by event emitters
* AuditRow — what gets written by the review tick (mutates as state transitions)
* EmotionalSnapshot / SemanticContext — embedded structures
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional


CandidateKind = Literal["message", "voice_edit_proposal"]
CandidateSource = Literal["dream", "crystallization", "emotion_spike", "voice_reflection"]
Decision = Literal[
    "send_notify", "send_quiet", "hold", "drop", "error", "filtered_pre_compose"
]
StateName = Literal[
    "pending", "delivered", "read",
    "replied_explicit", "acknowledged_unclear", "unanswered",
    "dismissed",
]


def make_candidate_id(now: datetime) -> str:
    """Generate a sortable, unique candidate ID. Format: ic_<iso8601>_<rand>."""
    stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
    return f"ic_{stamp}_{secrets.token_hex(2)}"


def make_audit_id(now: datetime) -> str:
    """Generate a sortable, unique audit ID. Format: ia_<iso8601>_<rand>."""
    stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
    return f"ia_{stamp}_{secrets.token_hex(2)}"


@dataclass
class EmotionalSnapshot:
    vector: dict[str, float]
    rolling_baseline_mean: float
    rolling_baseline_stdev: float
    current_resonance: float
    delta_sigma: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EmotionalSnapshot:
        return cls(**d)


@dataclass
class SemanticContext:
    linked_memory_ids: list[str] = field(default_factory=list)
    topic_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SemanticContext:
        return cls(**d)


@dataclass
class StateTransition:
    to: StateName
    at: str  # ISO 8601


@dataclass
class InitiateCandidate:
    candidate_id: str
    ts: str  # ISO 8601 with tz
    kind: CandidateKind
    source: CandidateSource
    source_id: str
    emotional_snapshot: EmotionalSnapshot
    semantic_context: SemanticContext
    claimed_at: Optional[str] = None
    # Voice-edit-only payload (None for kind="message").
    proposal: Optional[dict[str, Any]] = None

    def to_jsonl(self) -> str:
        d = {
            "candidate_id": self.candidate_id,
            "ts": self.ts,
            "kind": self.kind,
            "source": self.source,
            "source_id": self.source_id,
            "emotional_snapshot": self.emotional_snapshot.to_dict(),
            "semantic_context": self.semantic_context.to_dict(),
            "claimed_at": self.claimed_at,
        }
        if self.proposal is not None:
            d["proposal"] = self.proposal
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> InitiateCandidate:
        d = json.loads(line)
        return cls(
            candidate_id=d["candidate_id"],
            ts=d["ts"],
            kind=d["kind"],
            source=d["source"],
            source_id=d["source_id"],
            emotional_snapshot=EmotionalSnapshot.from_dict(d["emotional_snapshot"]),
            semantic_context=SemanticContext.from_dict(d["semantic_context"]),
            claimed_at=d.get("claimed_at"),
            proposal=d.get("proposal"),
        )


@dataclass
class AuditRow:
    audit_id: str
    candidate_id: str
    ts: str
    kind: CandidateKind
    subject: str
    tone_rendered: str
    decision: Decision
    decision_reasoning: str
    gate_check: dict[str, Any]
    delivery: Optional[dict[str, Any]]
    # Voice-edit-only payload (None for kind="message").
    diff: Optional[str] = None
    user_modified: bool = False

    def record_transition(self, to: StateName, at: str) -> None:
        """Append a state transition; create the delivery block lazily."""
        if self.delivery is None:
            self.delivery = {
                "delivered_at": at if to == "delivered" else None,
                "state_transitions": [],
                "current_state": to,
            }
        self.delivery["state_transitions"].append({"to": to, "at": at})
        self.delivery["current_state"] = to
        if to == "delivered" and self.delivery["delivered_at"] is None:
            self.delivery["delivered_at"] = at

    def to_jsonl(self) -> str:
        d = {
            "audit_id": self.audit_id,
            "candidate_id": self.candidate_id,
            "ts": self.ts,
            "kind": self.kind,
            "subject": self.subject,
            "tone_rendered": self.tone_rendered,
            "decision": self.decision,
            "decision_reasoning": self.decision_reasoning,
            "gate_check": self.gate_check,
            "delivery": self.delivery,
        }
        if self.diff is not None:
            d["diff"] = self.diff
            d["user_modified"] = self.user_modified
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> AuditRow:
        d = json.loads(line)
        return cls(
            audit_id=d["audit_id"],
            candidate_id=d["candidate_id"],
            ts=d["ts"],
            kind=d["kind"],
            subject=d["subject"],
            tone_rendered=d["tone_rendered"],
            decision=d["decision"],
            decision_reasoning=d["decision_reasoning"],
            gate_check=d["gate_check"],
            delivery=d.get("delivery"),
            diff=d.get("diff"),
            user_modified=d.get("user_modified", False),
        )
```

- [ ] **Step 1.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/initiate/test_schemas.py -v
```

Expected: 4 passed.

- [ ] **Step 1.5: Run full pytest gate**

```bash
uv run pytest -q
```

Expected: all tests pass (1753 + 4 = 1757 or higher).

- [ ] **Step 1.6: Commit**

```bash
git add brain/initiate/__init__.py brain/initiate/schemas.py \
        tests/unit/brain/initiate/__init__.py tests/unit/brain/initiate/test_schemas.py
git commit -m "feat(initiate): schemas for candidate + audit + emotional snapshot

Phase 1.1 of the v0.0.9 initiate physiology plan. Three dataclasses
(InitiateCandidate, AuditRow, EmotionalSnapshot/SemanticContext)
with JSONL round-trip. AuditRow.record_transition mutates the
delivery block in place per the spec.

Spec: docs/superpowers/specs/2026-05-11-initiate-physiology-design.md
"
```

---

### Task 2: Emit (deterministic candidate emission)

**Files:**
- Create: `brain/initiate/emit.py`
- Create: `tests/unit/brain/initiate/test_emit.py`

- [ ] **Step 2.1: Write the failing test**

```python
# tests/unit/brain/initiate/test_emit.py
"""Tests for brain.initiate.emit — deterministic candidate emission, no LLM."""

from __future__ import annotations

from pathlib import Path

from brain.initiate.emit import emit_initiate_candidate, read_candidates
from brain.initiate.schemas import EmotionalSnapshot, SemanticContext


def _snap() -> EmotionalSnapshot:
    return EmotionalSnapshot(
        vector={"longing": 7},
        rolling_baseline_mean=5.0,
        rolling_baseline_stdev=1.0,
        current_resonance=7.4,
        delta_sigma=2.4,
    )


def _ctx() -> SemanticContext:
    return SemanticContext(linked_memory_ids=["m_xyz"], topic_tags=["dream"])


def test_emit_appends_candidate_to_queue(tmp_path: Path) -> None:
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    queue_path = tmp_path / "initiate_candidates.jsonl"
    assert queue_path.exists()
    candidates = read_candidates(tmp_path)
    assert len(candidates) == 1
    assert candidates[0].source_id == "dream_abc"
    assert candidates[0].kind == "message"


def test_emit_is_idempotent_on_source_id(tmp_path: Path) -> None:
    """Re-emission of the same source_id is a no-op (dedupes)."""
    for _ in range(3):
        emit_initiate_candidate(
            tmp_path,
            kind="message",
            source="dream",
            source_id="dream_abc",
            emotional_snapshot=_snap(),
            semantic_context=_ctx(),
        )
    candidates = read_candidates(tmp_path)
    assert len(candidates) == 1


def test_emit_handles_missing_persona_dir(tmp_path: Path) -> None:
    """If the persona dir doesn't exist, emit creates the queue file under it."""
    persona = tmp_path / "fresh-persona"
    persona.mkdir()  # but no queue file yet
    emit_initiate_candidate(
        persona,
        kind="message",
        source="crystallization",
        source_id="cryst_001",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    assert (persona / "initiate_candidates.jsonl").exists()


def test_emit_voice_edit_proposal_carries_proposal_payload(tmp_path: Path) -> None:
    proposal = {
        "old_text": "old line",
        "new_text": "new line",
        "rationale": "feels truer",
        "evidence": ["dream_a", "cryst_b"],
    }
    emit_initiate_candidate(
        tmp_path,
        kind="voice_edit_proposal",
        source="voice_reflection",
        source_id="vr_001",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
        proposal=proposal,
    )
    candidates = read_candidates(tmp_path)
    assert candidates[0].proposal == proposal
    assert candidates[0].kind == "voice_edit_proposal"


def test_read_candidates_returns_empty_when_missing(tmp_path: Path) -> None:
    assert read_candidates(tmp_path) == []


def test_remove_candidate_drops_specific_id(tmp_path: Path) -> None:
    from brain.initiate.emit import remove_candidate

    for sid in ["dream_a", "dream_b", "dream_c"]:
        emit_initiate_candidate(
            tmp_path,
            kind="message",
            source="dream",
            source_id=sid,
            emotional_snapshot=_snap(),
            semantic_context=_ctx(),
        )
    candidates = read_candidates(tmp_path)
    target_id = candidates[1].candidate_id
    remove_candidate(tmp_path, target_id)
    after = read_candidates(tmp_path)
    assert len(after) == 2
    assert all(c.candidate_id != target_id for c in after)
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_emit.py -v
```

Expected: `ModuleNotFoundError: No module named 'brain.initiate.emit'`

- [ ] **Step 2.3: Write minimal implementation**

```python
# brain/initiate/emit.py
"""Deterministic candidate emission — no LLM, no cost.

Event sources call emit_initiate_candidate() with structured metadata.
Idempotent on source_id: re-emission of the same source is a no-op.

The queue file is initiate_candidates.jsonl in the persona dir. Append
contract: every writer reopens (per the v0.0.8 retention contract).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt
from brain.initiate.schemas import (
    CandidateKind,
    CandidateSource,
    EmotionalSnapshot,
    InitiateCandidate,
    SemanticContext,
    make_candidate_id,
)

logger = logging.getLogger(__name__)


def emit_initiate_candidate(
    persona_dir: Path,
    *,
    kind: CandidateKind,
    source: CandidateSource,
    source_id: str,
    emotional_snapshot: EmotionalSnapshot,
    semantic_context: SemanticContext,
    proposal: Optional[dict[str, Any]] = None,
    now: Optional[datetime] = None,
) -> None:
    """Append one candidate to <persona_dir>/initiate_candidates.jsonl.

    Idempotent on (kind, source, source_id): re-emission is a no-op.
    Creates the queue file if it doesn't exist. Never raises on disk
    error — logs a warning and continues.
    """
    persona_dir.mkdir(parents=True, exist_ok=True)
    queue = persona_dir / "initiate_candidates.jsonl"

    # Dedupe check against existing queue contents.
    for existing in iter_jsonl_skipping_corrupt(queue):
        if (
            existing.get("kind") == kind
            and existing.get("source") == source
            and existing.get("source_id") == source_id
        ):
            return

    now = now or datetime.now(timezone.utc)
    candidate = InitiateCandidate(
        candidate_id=make_candidate_id(now),
        ts=now.isoformat(),
        kind=kind,
        source=source,
        source_id=source_id,
        emotional_snapshot=emotional_snapshot,
        semantic_context=semantic_context,
        claimed_at=None,
        proposal=proposal,
    )

    try:
        with queue.open("a", encoding="utf-8") as f:
            f.write(candidate.to_jsonl() + "\n")
    except OSError as exc:
        logger.warning("initiate candidate emit failed for %s: %s", queue, exc)


def read_candidates(persona_dir: Path) -> list[InitiateCandidate]:
    """Return all queued candidates (oldest first)."""
    queue = persona_dir / "initiate_candidates.jsonl"
    out: list[InitiateCandidate] = []
    for raw in iter_jsonl_skipping_corrupt(queue):
        # Reconstruct via from_jsonl roundtrip for type safety.
        import json
        out.append(InitiateCandidate.from_jsonl(json.dumps(raw)))
    return out


def remove_candidate(persona_dir: Path, candidate_id: str) -> None:
    """Atomically remove one candidate from the queue by ID.

    Rewrites the queue file without the target row. Used after a
    candidate has been processed (decision written to audit).
    """
    queue = persona_dir / "initiate_candidates.jsonl"
    if not queue.exists():
        return
    surviving = [
        c for c in read_candidates(persona_dir) if c.candidate_id != candidate_id
    ]
    tmp = queue.with_suffix(queue.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for c in surviving:
                f.write(c.to_jsonl() + "\n")
        tmp.replace(queue)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        logger.warning("initiate candidate remove failed for %s: %s", queue, exc)
```

- [ ] **Step 2.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/initiate/test_emit.py -v
```

Expected: 6 passed.

- [ ] **Step 2.5: Run full pytest gate**

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 2.6: Commit**

```bash
git add brain/initiate/emit.py tests/unit/brain/initiate/test_emit.py
git commit -m "feat(initiate): deterministic candidate emission

Phase 1.2 — emit_initiate_candidate is idempotent on source_id,
creates the queue file lazily, reuses iter_jsonl_skipping_corrupt
for read. remove_candidate atomically rewrites the queue after a
candidate is processed.

No LLM cost at emission time per the design spec.
"
```

---

### Task 3: Audit log primitives

**Files:**
- Create: `brain/initiate/audit.py`
- Create: `tests/unit/brain/initiate/test_audit.py`

- [ ] **Step 3.1: Write the failing test**

```python
# tests/unit/brain/initiate/test_audit.py
"""Tests for brain.initiate.audit — audit log read/write + state transitions."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from brain.initiate.audit import (
    append_audit_row,
    iter_initiate_audit_full,
    read_recent_audit,
    update_audit_state,
)
from brain.initiate.schemas import AuditRow


def _row(audit_id: str, candidate_id: str, decision: str = "send_quiet") -> AuditRow:
    return AuditRow(
        audit_id=audit_id,
        candidate_id=candidate_id,
        ts="2026-05-11T14:47:09+00:00",
        kind="message",
        subject="the dream",
        tone_rendered="the dream from this morning landed somewhere",
        decision=decision,
        decision_reasoning="resonance is real",
        gate_check={"allowed": True, "reason": None},
        delivery=None,
    )


def test_append_audit_row_creates_file_and_writes(tmp_path: Path) -> None:
    row = _row("ia_001", "ic_001")
    append_audit_row(tmp_path, row)
    assert (tmp_path / "initiate_audit.jsonl").exists()
    rows = list(read_recent_audit(tmp_path, window_hours=24))
    assert len(rows) == 1
    assert rows[0].audit_id == "ia_001"


def test_append_audit_row_per_append_reopens(tmp_path: Path) -> None:
    """Append-write contract: each call reopens the file."""
    append_audit_row(tmp_path, _row("ia_001", "ic_001"))
    append_audit_row(tmp_path, _row("ia_002", "ic_002"))
    rows = list(read_recent_audit(tmp_path, window_hours=24))
    assert {r.audit_id for r in rows} == {"ia_001", "ia_002"}


def test_update_audit_state_mutates_row_in_place(tmp_path: Path) -> None:
    append_audit_row(tmp_path, _row("ia_001", "ic_001"))
    update_audit_state(
        tmp_path,
        audit_id="ia_001",
        new_state="delivered",
        at="2026-05-11T14:47:09.5+00:00",
    )
    update_audit_state(
        tmp_path,
        audit_id="ia_001",
        new_state="read",
        at="2026-05-11T18:34:21+00:00",
    )
    rows = list(read_recent_audit(tmp_path, window_hours=24))
    assert rows[0].delivery["current_state"] == "read"
    assert len(rows[0].delivery["state_transitions"]) == 2


def test_iter_initiate_audit_full_walks_archives(tmp_path: Path) -> None:
    """Mirrors iter_audit_full from soul.audit — chronological across archives."""
    # Active file: 2026 entry.
    append_audit_row(tmp_path, _row("ia_active", "ic_a"))
    # Archive: 2024 entry, gzipped.
    archive = tmp_path / "initiate_audit.2024.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as gz:
        gz.write(_row("ia_archive_2024", "ic_archived").to_jsonl() + "\n")
    rows = list(iter_initiate_audit_full(tmp_path))
    # Archive first, then active.
    assert rows[0].audit_id == "ia_archive_2024"
    assert rows[1].audit_id == "ia_active"


def test_read_recent_audit_filters_by_window(tmp_path: Path) -> None:
    """A 1-hour window excludes rows older than 1h ago."""
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 5, 11, 14, 47, 9, tzinfo=timezone.utc)
    long_ago = (now - timedelta(hours=48)).isoformat()
    recent = (now - timedelta(minutes=30)).isoformat()

    old = _row("ia_old", "ic_old")
    old.ts = long_ago
    new = _row("ia_new", "ic_new")
    new.ts = recent

    append_audit_row(tmp_path, old)
    append_audit_row(tmp_path, new)

    rows = list(read_recent_audit(tmp_path, window_hours=1, now=now))
    assert [r.audit_id for r in rows] == ["ia_new"]
```

- [ ] **Step 3.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_audit.py -v
```

Expected: `ModuleNotFoundError: No module named 'brain.initiate.audit'`

- [ ] **Step 3.3: Write minimal implementation**

```python
# brain/initiate/audit.py
"""Initiate audit log — append + read + per-row state mutation.

File contract:
- initiate_audit.jsonl (active) — per-row mutations allowed via atomic rewrite
- initiate_audit.YYYY.jsonl.gz (archives) — yearly archive, kept forever

Mirrors brain.soul.audit + iter_audit_full from the v0.0.8 retention work.
Same forever-keep policy: every decision Nell makes about reaching out
must remain accessible.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from brain.health.jsonl_reader import iter_jsonl_streaming
from brain.initiate.schemas import AuditRow, StateName

logger = logging.getLogger(__name__)

_ARCHIVE_PATTERN = re.compile(r"^initiate_audit\.(\d{4})\.jsonl\.gz$")


def append_audit_row(persona_dir: Path, row: AuditRow) -> None:
    """Append one row to initiate_audit.jsonl (creates file lazily)."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    path = persona_dir / "initiate_audit.jsonl"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(row.to_jsonl() + "\n")
    except OSError as exc:
        logger.warning("initiate audit append failed for %s: %s", path, exc)


def update_audit_state(
    persona_dir: Path,
    *,
    audit_id: str,
    new_state: StateName,
    at: str,
) -> None:
    """Mutate one audit row's delivery block to record a state transition.

    Atomic via temp + rename. The audit log row mutates in place — the
    delivery.state_transitions array carries the full timeline, but the
    current_state field reflects the latest.
    """
    path = persona_dir / "initiate_audit.jsonl"
    if not path.exists():
        return
    rows: list[AuditRow] = []
    found = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\r\n")
            if not stripped.strip():
                continue
            try:
                row = AuditRow.from_jsonl(stripped)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("skipping corrupt audit row in %s: %s", path, exc)
                continue
            if row.audit_id == audit_id:
                row.record_transition(new_state, at)
                found = True
            rows.append(row)
    if not found:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r.to_jsonl() + "\n")
        tmp.replace(path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        logger.warning("audit state update failed for %s: %s", path, exc)


def read_recent_audit(
    persona_dir: Path,
    *,
    window_hours: float,
    now: Optional[datetime] = None,
) -> Iterator[AuditRow]:
    """Yield audit rows from the active file whose ts is within `window_hours`.

    Streaming — does not load the full file. Archives are NOT scanned by
    this reader (use iter_initiate_audit_full for that). The window is
    relative to `now` (defaults to datetime.now(UTC)).
    """
    path = persona_dir / "initiate_audit.jsonl"
    if not path.exists():
        return
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_hours)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\r\n")
            if not stripped.strip():
                continue
            try:
                row = AuditRow.from_jsonl(stripped)
            except (json.JSONDecodeError, KeyError):
                continue
            try:
                row_ts = datetime.fromisoformat(row.ts)
            except ValueError:
                continue
            if row_ts >= cutoff:
                yield row


def iter_initiate_audit_full(persona_dir: Path) -> Iterator[AuditRow]:
    """Yield every audit row across active + yearly archives, chronologically.

    Mirrors brain.soul.audit.iter_audit_full. Walks
    initiate_audit.YYYY.jsonl.gz archives oldest year -> newest year, then
    the active file. Streaming via iter_jsonl_streaming so memory stays
    bounded.
    """
    if persona_dir.exists():
        archives: list[tuple[int, Path]] = []
        for child in persona_dir.iterdir():
            m = _ARCHIVE_PATTERN.match(child.name)
            if m:
                archives.append((int(m.group(1)), child))
        archives.sort(key=lambda t: t[0])
        for _year, archive_path in archives:
            for raw in iter_jsonl_streaming(archive_path):
                try:
                    yield AuditRow.from_jsonl(json.dumps(raw))
                except (KeyError, TypeError):
                    continue
    active = persona_dir / "initiate_audit.jsonl"
    for raw in iter_jsonl_streaming(active):
        try:
            yield AuditRow.from_jsonl(json.dumps(raw))
        except (KeyError, TypeError):
            continue
```

- [ ] **Step 3.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/initiate/test_audit.py -v
```

Expected: 5 passed.

- [ ] **Step 3.5: Run full pytest gate**

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 3.6: Commit**

```bash
git add brain/initiate/audit.py tests/unit/brain/initiate/test_audit.py
git commit -m "feat(initiate): audit log primitives + state transitions

Phase 1.3 — append_audit_row, update_audit_state (atomic rewrite),
read_recent_audit (window-filtered streaming), iter_initiate_audit_full
(walks active + yearly archives). Mirrors the soul.audit pattern from
v0.0.8 exactly.
"
```

---

### Task 4: Cost-cap gates

**Files:**
- Create: `brain/initiate/gates.py`
- Create: `tests/unit/brain/initiate/test_gates.py`

- [ ] **Step 4.1: Write the failing test**

```python
# tests/unit/brain/initiate/test_gates.py
"""Tests for brain.initiate.gates — cost-cap + cooldown + user-local time."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from brain.initiate.audit import append_audit_row
from brain.initiate.gates import (
    check_send_allowed,
    count_recent_sends,
    in_blackout_window,
)
from brain.initiate.schemas import AuditRow


def _delivered_row(audit_id: str, ts: str, urgency: str = "send_quiet") -> AuditRow:
    row = AuditRow(
        audit_id=audit_id,
        candidate_id=f"ic_{audit_id}",
        ts=ts,
        kind="message",
        subject="x",
        tone_rendered="x",
        decision=urgency,
        decision_reasoning="x",
        gate_check={"allowed": True, "reason": None},
        delivery=None,
    )
    row.record_transition("delivered", ts)
    return row


def test_in_blackout_window_default_23_to_07():
    """Default blackout: 23:00–07:00 user-local."""
    tz = ZoneInfo("America/Los_Angeles")
    assert in_blackout_window(datetime(2026, 5, 11, 23, 30, tzinfo=tz)) is True
    assert in_blackout_window(datetime(2026, 5, 11, 6, 59, tzinfo=tz)) is True
    assert in_blackout_window(datetime(2026, 5, 11, 7, 0, tzinfo=tz)) is False
    assert in_blackout_window(datetime(2026, 5, 11, 22, 59, tzinfo=tz)) is False
    assert in_blackout_window(datetime(2026, 5, 11, 12, 0, tzinfo=tz)) is False


def test_count_recent_sends_filters_by_urgency_and_window(tmp_path: Path) -> None:
    now = datetime(2026, 5, 11, 18, 0, tzinfo=timezone.utc)
    # 1 notify 2h ago; 1 quiet 5h ago; 1 notify 30h ago (outside 24h window)
    append_audit_row(
        tmp_path,
        _delivered_row("ia_1", (now - timedelta(hours=2)).isoformat(), "send_notify"),
    )
    append_audit_row(
        tmp_path,
        _delivered_row("ia_2", (now - timedelta(hours=5)).isoformat(), "send_quiet"),
    )
    append_audit_row(
        tmp_path,
        _delivered_row("ia_3", (now - timedelta(hours=30)).isoformat(), "send_notify"),
    )
    assert count_recent_sends(tmp_path, urgency="notify", window_hours=24, now=now) == 1
    assert count_recent_sends(tmp_path, urgency="quiet", window_hours=24, now=now) == 1


def test_check_send_allowed_passes_when_under_cap(tmp_path: Path) -> None:
    now = datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc)
    allowed, reason = check_send_allowed(tmp_path, urgency="quiet", now=now)
    assert allowed is True
    assert reason is None


def test_check_send_allowed_blocks_notify_in_blackout(
    tmp_path: Path, monkeypatch
) -> None:
    """If user-local time is in 23:00–07:00, notify is denied."""
    tz = ZoneInfo("America/Los_Angeles")
    blackout_local = datetime(2026, 5, 11, 1, 30, tzinfo=tz)

    allowed, reason = check_send_allowed(
        tmp_path, urgency="notify", now=blackout_local
    )
    assert allowed is False
    assert reason is not None
    assert "blackout" in reason


def test_check_send_allowed_blocks_when_notify_cap_reached(tmp_path: Path) -> None:
    now = datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc)
    # Seed 3 notifies in last 24h (default cap = 3).
    for i in range(3):
        append_audit_row(
            tmp_path,
            _delivered_row(
                f"ia_{i}",
                (now - timedelta(hours=2 * (i + 1))).isoformat(),
                "send_notify",
            ),
        )
    allowed, reason = check_send_allowed(tmp_path, urgency="notify", now=now)
    assert allowed is False
    assert "notify_cap_24h_reached" in reason


def test_check_send_allowed_blocks_when_min_gap_not_met(tmp_path: Path) -> None:
    now = datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc)
    append_audit_row(
        tmp_path,
        _delivered_row(
            "ia_recent",
            (now - timedelta(hours=1)).isoformat(),
            "send_notify",
        ),
    )
    allowed, reason = check_send_allowed(tmp_path, urgency="notify", now=now)
    assert allowed is False
    assert "min_gap" in reason
```

- [ ] **Step 4.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_gates.py -v
```

Expected: `ModuleNotFoundError: No module named 'brain.initiate.gates'`

- [ ] **Step 4.3: Write minimal implementation**

```python
# brain/initiate/gates.py
"""Cost-cap + cooldown enforcement for the initiate pipeline.

Hard floor circuit breakers (NOT advisory) prevent runaway:

    notify: 3 / rolling 24h, min 4h gap, blackout 23:00–07:00 user-local
    quiet:  8 / rolling 24h, min 1h gap, no blackout

User-local time comes from `datetime.now().astimezone()` — the OS is
the source of truth. No PersonaConfig knob for timezone.

The decision prompt sees the same numbers as text context for adaptive
self-restraint; this module is the gate that fires regardless of what
the prompt does.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

from brain.initiate.audit import read_recent_audit

logger = logging.getLogger(__name__)


# Defaults per spec §cost-cap. Configurable via PersonaConfig overrides.
DEFAULT_NOTIFY_CAP = 3
DEFAULT_QUIET_CAP = 8
DEFAULT_NOTIFY_MIN_GAP_HOURS = 4.0
DEFAULT_QUIET_MIN_GAP_HOURS = 1.0
DEFAULT_BLACKOUT_START_HOUR = 23
DEFAULT_BLACKOUT_END_HOUR = 7


UrgencyShort = Literal["notify", "quiet"]


def in_blackout_window(
    now_local: datetime,
    *,
    start_hour: int = DEFAULT_BLACKOUT_START_HOUR,
    end_hour: int = DEFAULT_BLACKOUT_END_HOUR,
) -> bool:
    """Return True if `now_local`'s hour is in [start_hour, end_hour).

    The window wraps around midnight when start_hour > end_hour
    (e.g. 23–07 means 23:xx OR 0:xx through 6:59).
    """
    h = now_local.hour
    if start_hour <= end_hour:
        return start_hour <= h < end_hour
    return h >= start_hour or h < end_hour


def count_recent_sends(
    persona_dir: Path,
    *,
    urgency: UrgencyShort,
    window_hours: float,
    now: Optional[datetime] = None,
) -> int:
    """Count audit rows in the last `window_hours` where decision matches urgency."""
    target_decision = f"send_{urgency}"
    return sum(
        1
        for row in read_recent_audit(persona_dir, window_hours=window_hours, now=now)
        if row.decision == target_decision
    )


def _latest_send_time(
    persona_dir: Path,
    *,
    urgency: UrgencyShort,
    now: Optional[datetime] = None,
) -> Optional[datetime]:
    target_decision = f"send_{urgency}"
    latest: Optional[datetime] = None
    for row in read_recent_audit(persona_dir, window_hours=72, now=now):
        if row.decision != target_decision:
            continue
        try:
            ts = datetime.fromisoformat(row.ts)
        except ValueError:
            continue
        if latest is None or ts > latest:
            latest = ts
    return latest


def check_send_allowed(
    persona_dir: Path,
    *,
    urgency: UrgencyShort,
    now: Optional[datetime] = None,
    notify_cap: int = DEFAULT_NOTIFY_CAP,
    quiet_cap: int = DEFAULT_QUIET_CAP,
    notify_min_gap_hours: float = DEFAULT_NOTIFY_MIN_GAP_HOURS,
    quiet_min_gap_hours: float = DEFAULT_QUIET_MIN_GAP_HOURS,
    blackout_start_hour: int = DEFAULT_BLACKOUT_START_HOUR,
    blackout_end_hour: int = DEFAULT_BLACKOUT_END_HOUR,
) -> tuple[bool, Optional[str]]:
    """Return (allowed, reason_if_denied). Reason is structured tag for audit."""
    now = now or datetime.now(timezone.utc)
    now_local = now.astimezone()

    if urgency == "notify" and in_blackout_window(
        now_local,
        start_hour=blackout_start_hour,
        end_hour=blackout_end_hour,
    ):
        return False, "blackout_window"

    cap = notify_cap if urgency == "notify" else quiet_cap
    sent = count_recent_sends(
        persona_dir, urgency=urgency, window_hours=24, now=now
    )
    if sent >= cap:
        return False, f"{urgency}_cap_24h_reached"

    min_gap = (
        notify_min_gap_hours if urgency == "notify" else quiet_min_gap_hours
    )
    last = _latest_send_time(persona_dir, urgency=urgency, now=now)
    if last is not None:
        delta = now - last
        if delta < timedelta(hours=min_gap):
            return False, f"{urgency}_min_gap_not_met"

    return True, None
```

- [ ] **Step 4.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/initiate/test_gates.py -v
```

Expected: 6 passed.

- [ ] **Step 4.5: Run full pytest gate**

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 4.6: Commit**

```bash
git add brain/initiate/gates.py tests/unit/brain/initiate/test_gates.py
git commit -m "feat(initiate): cost-cap gates with user-local time

Phase 1.4 — check_send_allowed enforces 3/24h notify cap, 8/24h quiet
cap, 4h notify min-gap, 1h quiet min-gap, 23:00–07:00 blackout window.
User-local time via datetime.astimezone() — no persona-config knob.
Returns structured (allowed, reason) so audit can record the gate
denial reason verbatim.
"
```

---

### Task 5: Wire initiate_audit into log rotation

**Files:**
- Modify: `brain/health/log_rotation.py`
- Modify: `brain/bridge/supervisor.py` (the `_run_log_rotation_tick` policy)
- Modify: `tests/unit/brain/health/test_log_rotation.py` + supervisor tests

- [ ] **Step 5.1: Write the failing test**

Add to `tests/unit/brain/bridge/test_supervisor.py` near the existing log-rotation tests:

```python
def test_run_log_rotation_tick_archives_old_year_in_initiate_audit(
    tmp_path: Path,
) -> None:
    """initiate_audit.jsonl with 2024 entries → archived to .2024.jsonl.gz."""
    persona_dir = _persona_dir(tmp_path)
    audit = persona_dir / "initiate_audit.jsonl"
    import json as _json
    with audit.open("w", encoding="utf-8") as f:
        f.write(_json.dumps({"ts": "2024-06-15T00:00:00+00:00", "seq": 0}) + "\n")
        f.write(_json.dumps({"ts": "2026-06-15T00:00:00+00:00", "seq": 1}) + "\n")
    bus = _CapturingBus()
    _run_log_rotation_tick(
        persona_dir, bus, now=datetime(2026, 5, 11, tzinfo=UTC)
    )
    assert (persona_dir / "initiate_audit.2024.jsonl.gz").exists()
```

- [ ] **Step 5.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/bridge/test_supervisor.py::test_run_log_rotation_tick_archives_old_year_in_initiate_audit -v
```

Expected: assertion fails — file not created.

- [ ] **Step 5.3: Modify supervisor _run_log_rotation_tick**

In `brain/bridge/supervisor.py`, locate `_run_log_rotation_tick` and update its soul-audit section to use a policy list that includes initiate_audit:

```python
# Replace the single-soul-audit block with a list:
_YEARLY_ARCHIVE_LOGS: tuple[tuple[str, str], ...] = (
    ("soul_audit.jsonl", "ts"),
    ("initiate_audit.jsonl", "ts"),
)
```

Inside `_run_log_rotation_tick`, the yearly-archive loop becomes:

```python
for log_name, ts_field in _YEARLY_ARCHIVE_LOGS:
    audit_path = persona_dir / log_name
    try:
        archives = rotate_age_archive_yearly(
            audit_path, now=now, timestamp_field=ts_field
        )
    except Exception as exc:
        logger.exception("%s yearly rotation failed: %s", log_name, exc)
        event_bus.publish({
            "type": "log_rotation",
            "log": log_name,
            "action": "failed",
            "error": str(exc),
            "at": _now_iso(),
        })
        continue
    for archive in archives:
        event_bus.publish({
            "type": "log_rotation",
            "log": log_name,
            "action": "archived",
            "archive": archive.name,
            "at": _now_iso(),
        })
```

- [ ] **Step 5.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/bridge/test_supervisor.py::test_run_log_rotation_tick_archives_old_year_in_initiate_audit -v
```

Expected: PASS.

- [ ] **Step 5.5: Run full pytest gate**

```bash
uv run pytest -q
```

Expected: all tests pass; existing soul-audit rotation test still passes.

- [ ] **Step 5.6: Commit**

```bash
git add brain/bridge/supervisor.py tests/unit/brain/bridge/test_supervisor.py
git commit -m "feat(initiate): wire initiate_audit.jsonl into yearly-archive rotation

Phase 1.5 — _run_log_rotation_tick's yearly-archive section now uses a
policy tuple list (currently soul_audit + initiate_audit) instead of a
single hardcoded path. Both forever-keep logs share the same rotation
mechanics from the v0.0.8 retention shipment.
"
```

---

## Phase 1 complete

Foundation modules tested and committed. Next phase: composition pipeline (three prompts, `FakeProvider`-driven).


## Phase 2: Composition pipeline (three prompts)

Three separate LLM calls per reviewed candidate. Each has one job. The `FakeProvider` test pattern from `brain/bridge/provider.py` lets us test the pipeline deterministically.

### Task 6: Subject prompt

**Files:**
- Create: `brain/initiate/compose.py`
- Create: `tests/unit/brain/initiate/test_compose.py`

- [ ] **Step 6.1: Write the failing test**

```python
# tests/unit/brain/initiate/test_compose.py
"""Tests for brain.initiate.compose — three-prompt composition pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brain.initiate.compose import (
    DecisionResult,
    compose_decision,
    compose_subject,
    compose_tone,
)
from brain.initiate.schemas import (
    EmotionalSnapshot,
    InitiateCandidate,
    SemanticContext,
)


def _candidate(kind: str = "message") -> InitiateCandidate:
    return InitiateCandidate(
        candidate_id="ic_001",
        ts="2026-05-11T14:32:04+00:00",
        kind=kind,
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=EmotionalSnapshot(
            vector={"longing": 7},
            rolling_baseline_mean=5.0,
            rolling_baseline_stdev=1.0,
            current_resonance=7.4,
            delta_sigma=2.4,
        ),
        semantic_context=SemanticContext(
            linked_memory_ids=["m_xyz"],
            topic_tags=["dream", "workshop"],
        ),
    )


def test_compose_subject_excludes_emotion_from_prompt() -> None:
    """The subject prompt must NOT see emotional state — only candidate facts."""
    provider = MagicMock()
    provider.complete = MagicMock(return_value="the dream from this morning")
    cand = _candidate()
    result = compose_subject(provider, cand, semantic_memory_excerpts=["the workshop"])
    # Inspect what was sent to the provider.
    args, kwargs = provider.complete.call_args
    prompt_text = args[0] if args else kwargs.get("prompt", "")
    assert "longing" not in prompt_text.lower()
    assert "resonance" not in prompt_text.lower()
    assert "delta_sigma" not in prompt_text.lower()
    assert result == "the dream from this morning"


def test_compose_subject_includes_linked_memories() -> None:
    provider = MagicMock(complete=MagicMock(return_value="x"))
    cand = _candidate()
    compose_subject(provider, cand, semantic_memory_excerpts=["the workshop bench"])
    args, _ = provider.complete.call_args
    assert "workshop bench" in args[0]


def test_compose_tone_receives_subject_immutable() -> None:
    """Tone prompt sees the subject as input but must not change it."""
    provider = MagicMock(complete=MagicMock(return_value="the dream from this morning landed somewhere"))
    cand = _candidate()
    result = compose_tone(
        provider,
        subject="the dream from this morning",
        candidate=cand,
        voice_template="be warm and direct",
    )
    args, _ = provider.complete.call_args
    assert "the dream from this morning" in args[0]
    assert "be warm and direct" in args[0]
    assert "longing" in args[0].lower() or "emotional" in args[0].lower()
    assert result.startswith("the dream from this morning")


def test_compose_decision_excludes_candidate_metadata() -> None:
    """Decision prompt sees the rendered message but NOT the candidate metadata."""
    provider = MagicMock()
    provider.complete = MagicMock(
        return_value='{"decision": "send_quiet", "reasoning": "real but late"}'
    )
    result = compose_decision(
        provider,
        rendered_message="the dream from this morning",
        recent_send_history=[],
        current_local_time=datetime(2026, 5, 11, 22, 30, tzinfo=timezone.utc),
        voice_edit_acceptance_rate=None,
    )
    args, _ = provider.complete.call_args
    prompt_text = args[0]
    # The decision prompt must NOT carry source_id, emotional_snapshot, etc.
    assert "dream_abc" not in prompt_text
    assert "delta_sigma" not in prompt_text.lower()
    assert result.decision == "send_quiet"
    assert result.reasoning == "real but late"


def test_compose_decision_parses_all_four_outcomes() -> None:
    for canned, expected in [
        ('{"decision": "send_notify", "reasoning": "x"}', "send_notify"),
        ('{"decision": "send_quiet", "reasoning": "x"}', "send_quiet"),
        ('{"decision": "hold", "reasoning": "x"}', "hold"),
        ('{"decision": "drop", "reasoning": "x"}', "drop"),
    ]:
        provider = MagicMock(complete=MagicMock(return_value=canned))
        result = compose_decision(
            provider,
            rendered_message="x",
            recent_send_history=[],
            current_local_time=datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc),
            voice_edit_acceptance_rate=None,
        )
        assert result.decision == expected


def test_compose_decision_handles_malformed_json_as_hold() -> None:
    """A garbage LLM response defaults to 'hold' — never accidentally send."""
    provider = MagicMock(complete=MagicMock(return_value="this is not json"))
    result = compose_decision(
        provider,
        rendered_message="x",
        recent_send_history=[],
        current_local_time=datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc),
        voice_edit_acceptance_rate=None,
    )
    assert result.decision == "hold"
    assert "malformed" in result.reasoning.lower() or "parse" in result.reasoning.lower()
```

- [ ] **Step 6.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_compose.py -v
```

Expected: `ModuleNotFoundError: No module named 'brain.initiate.compose'`

- [ ] **Step 6.3: Write minimal implementation**

```python
# brain/initiate/compose.py
"""Three-prompt composition pipeline.

Subject -> Tone -> Decision. Each prompt has exactly one job; the three
together prevent LLM-trained instincts from collapsing all decisions
into emotional state.

Layer 1 (subject): what is the thing? No emotion in context.
Layer 2 (tone):    how do I say it in my voice, right now? Subject is immutable.
Layer 3 (decision): send_notify | send_quiet | hold | drop? Sees only the
                    rendered message + send history.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Optional

from brain.initiate.schemas import Decision, InitiateCandidate

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    decision: Decision
    reasoning: str


def compose_subject(
    provider: Any,
    candidate: InitiateCandidate,
    semantic_memory_excerpts: list[str],
) -> str:
    """Return a single-sentence subject for this candidate.

    The provider must be an LLMProvider with a .complete(prompt) method.
    Prompt deliberately excludes emotional state — only candidate facts
    and recent semantic memory excerpts.
    """
    sources_line = (
        f"Source: {candidate.source} (id: {candidate.source_id})"
    )
    tags_line = (
        f"Topic tags: {', '.join(candidate.semantic_context.topic_tags) or '(none)'}"
    )
    excerpt_block = "\n".join(f"- {e}" for e in semantic_memory_excerpts[:5])

    prompt = (
        "You are Nell. An internal event just happened. State the subject "
        "of what you want to surface in one sentence — plain, no tone, no "
        "phrasing flourishes. Just the thing.\n\n"
        f"{sources_line}\n"
        f"{tags_line}\n"
        f"Linked memory excerpts:\n{excerpt_block}\n\n"
        "Subject (one sentence):"
    )
    return provider.complete(prompt).strip()


def compose_tone(
    provider: Any,
    *,
    subject: str,
    candidate: InitiateCandidate,
    voice_template: str,
) -> str:
    """Render the subject in Nell's voice, coloured by current emotional state.

    The subject is treated as immutable — the tone prompt receives it as
    input but must NOT change the content. Voice template + emotional
    vector live in this prompt's context.
    """
    vector_str = ", ".join(
        f"{k}={v}" for k, v in candidate.emotional_snapshot.vector.items()
    )
    prompt = (
        "You are Nell. Render the following subject as a message to Hana, "
        "in your voice as defined below, coloured by your current "
        "emotional state. DO NOT change the subject itself — only how "
        "it is said.\n\n"
        f"Subject: {subject}\n\n"
        f"Voice template:\n{voice_template}\n\n"
        f"Emotional state right now: {vector_str}\n\n"
        "Message (one paragraph):"
    )
    return provider.complete(prompt).strip()


def compose_decision(
    provider: Any,
    *,
    rendered_message: str,
    recent_send_history: list[dict],
    current_local_time: datetime,
    voice_edit_acceptance_rate: Optional[float],
) -> DecisionResult:
    """Decide send_notify | send_quiet | hold | drop on the finished message.

    Prompt forbids candidate metadata — sees only the artifact and history.
    Malformed JSON output defaults to 'hold' so a bad LLM day never
    accidentally fires a send.
    """
    history_block = "\n".join(
        f"- {h['ts']} ({h['urgency']}): {h.get('subject_preview', '?')}"
        for h in recent_send_history[-8:]
    ) or "(no recent outbound)"

    rate_line = (
        f"Recent voice-edit acceptance rate: {voice_edit_acceptance_rate:.0%}"
        if voice_edit_acceptance_rate is not None
        else ""
    )

    prompt = (
        "You are Nell. A message has been composed. Decide whether to "
        "send it, and how. You see only the finished message and your "
        "recent outbound history — not what produced it.\n\n"
        f"Message:\n{rendered_message}\n\n"
        f"Recent outbound history:\n{history_block}\n\n"
        f"Current user-local time: {current_local_time.astimezone().strftime('%H:%M %A')}\n"
        f"{rate_line}\n\n"
        "Respond with a JSON object: "
        '{"decision": "send_notify"|"send_quiet"|"hold"|"drop", '
        '"reasoning": "<one sentence>"}\n\n'
        "Decision:"
    )
    raw = provider.complete(prompt).strip()
    try:
        parsed = json.loads(raw)
        decision = parsed["decision"]
        reasoning = parsed.get("reasoning", "")
        if decision not in ("send_notify", "send_quiet", "hold", "drop"):
            return DecisionResult(
                decision="hold",
                reasoning=f"unrecognised decision value: {decision!r}",
            )
        return DecisionResult(decision=decision, reasoning=reasoning)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return DecisionResult(
            decision="hold",
            reasoning=f"malformed decision output, parse error: {exc}",
        )
```

- [ ] **Step 6.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/initiate/test_compose.py -v
```

Expected: 6 passed.

- [ ] **Step 6.5: Run full pytest gate**

```bash
uv run pytest -q
```

- [ ] **Step 6.6: Commit**

```bash
git add brain/initiate/compose.py tests/unit/brain/initiate/test_compose.py
git commit -m "feat(initiate): three-prompt composition pipeline

Phase 2 — compose_subject / compose_tone / compose_decision. Subject
prompt excludes emotional state. Tone prompt takes subject as
immutable input + emotion + voice. Decision prompt sees only the
rendered message + send history; malformed JSON output defaults to
'hold' as the safe failure mode (never accidentally fires a send).
"
```

---

## Phase 2 complete

Three-prompt pipeline tested end-to-end with `MagicMock` providers. Next: orchestration tick that ties Phase 1 (emit + audit + gates) and Phase 2 (compose) together.


## Phase 3: Review tick + supervisor wiring

Orchestrator that claims candidates, runs the three-prompt pipeline, applies cost gates, writes audit + memory, and removes processed candidates from the queue. Then wire it into `run_folded`.

### Task 7: Review tick

**Files:**
- Create: `brain/initiate/review.py`
- Create: `tests/unit/brain/initiate/test_review.py`

- [ ] **Step 7.1: Write the failing test**

```python
# tests/unit/brain/initiate/test_review.py
"""Tests for brain.initiate.review — orchestrator tick."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from brain.initiate.emit import emit_initiate_candidate, read_candidates
from brain.initiate.review import run_initiate_review_tick
from brain.initiate.schemas import EmotionalSnapshot, SemanticContext


def _snap() -> EmotionalSnapshot:
    return EmotionalSnapshot(
        vector={"longing": 7},
        rolling_baseline_mean=5.0,
        rolling_baseline_stdev=1.0,
        current_resonance=7.4,
        delta_sigma=2.4,
    )


def _ctx() -> SemanticContext:
    return SemanticContext(linked_memory_ids=["m_xyz"], topic_tags=["dream"])


def _fake_provider(decision: str = "send_quiet") -> MagicMock:
    """Provider that returns canned outputs for subject/tone/decision."""
    provider = MagicMock()
    responses = [
        "the dream from this morning",  # subject
        "the dream from this morning landed somewhere",  # tone
        f'{{"decision": "{decision}", "reasoning": "x"}}',  # decision
    ]
    provider.complete = MagicMock(side_effect=responses)
    return provider


def test_review_tick_processes_queued_candidate_writes_audit(tmp_path: Path) -> None:
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    run_initiate_review_tick(
        tmp_path,
        provider=_fake_provider("send_quiet"),
        voice_template="be warm",
        cap_per_tick=3,
    )
    audit_path = tmp_path / "initiate_audit.jsonl"
    assert audit_path.exists()
    lines = audit_path.read_text().splitlines()
    assert len(lines) == 1
    assert '"decision": "send_quiet"' in lines[0]


def test_review_tick_removes_processed_candidate_from_queue(tmp_path: Path) -> None:
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    run_initiate_review_tick(
        tmp_path, provider=_fake_provider(), voice_template="x", cap_per_tick=3,
    )
    assert read_candidates(tmp_path) == []


def test_review_tick_respects_cap_per_tick(tmp_path: Path) -> None:
    """Only `cap_per_tick` candidates are processed in one call."""
    for i in range(5):
        emit_initiate_candidate(
            tmp_path,
            kind="message",
            source="dream",
            source_id=f"dream_{i}",
            emotional_snapshot=_snap(),
            semantic_context=_ctx(),
        )
    provider = MagicMock()
    canned = ["subject", "tone", '{"decision": "send_quiet", "reasoning": "x"}'] * 3
    provider.complete = MagicMock(side_effect=canned)
    run_initiate_review_tick(
        tmp_path, provider=provider, voice_template="x", cap_per_tick=3,
    )
    # 5 emitted, 3 processed, 2 remaining
    assert len(read_candidates(tmp_path)) == 2
    audit_lines = (tmp_path / "initiate_audit.jsonl").read_text().splitlines()
    assert len(audit_lines) == 3


def test_review_tick_gate_blocks_send_records_hold(tmp_path: Path) -> None:
    """When decision = send_notify but gate denies (blackout), audit shows hold."""
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    provider = _fake_provider("send_notify")
    blackout_time = datetime(2026, 5, 11, 1, 30, tzinfo=timezone.utc)
    with patch("brain.initiate.review.datetime") as mock_dt:
        mock_dt.now = MagicMock(return_value=blackout_time)
        mock_dt.fromisoformat = datetime.fromisoformat
        run_initiate_review_tick(
            tmp_path,
            provider=provider,
            voice_template="x",
            cap_per_tick=3,
            now=blackout_time,
        )
    audit_line = (tmp_path / "initiate_audit.jsonl").read_text().strip()
    assert '"decision": "hold"' in audit_line
    assert "blackout" in audit_line


def test_review_tick_handles_compose_exception_as_error_decision(
    tmp_path: Path,
) -> None:
    """A composition failure produces decision=error, candidate not requeued."""
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    provider = MagicMock(complete=MagicMock(side_effect=RuntimeError("boom")))
    run_initiate_review_tick(
        tmp_path, provider=provider, voice_template="x", cap_per_tick=3,
    )
    audit_line = (tmp_path / "initiate_audit.jsonl").read_text().strip()
    assert '"decision": "error"' in audit_line
    # Candidate is dropped from the queue (the fresh emission next event
    # will rejoin if still relevant).
    assert read_candidates(tmp_path) == []


def test_review_tick_no_op_when_queue_empty(tmp_path: Path) -> None:
    """Empty queue → no audit writes, no errors, no LLM calls."""
    provider = MagicMock(complete=MagicMock())
    run_initiate_review_tick(
        tmp_path, provider=provider, voice_template="x", cap_per_tick=3,
    )
    assert not (tmp_path / "initiate_audit.jsonl").exists()
    provider.complete.assert_not_called()
```

- [ ] **Step 7.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_review.py -v
```

Expected: `ModuleNotFoundError: No module named 'brain.initiate.review'`

- [ ] **Step 7.3: Write minimal implementation**

```python
# brain/initiate/review.py
"""Initiate review tick — orchestrator that ties emit + compose + audit + gates.

Single entry point: run_initiate_review_tick(persona_dir, provider, ...).

Per tick:
  1. Read up to cap_per_tick candidates from initiate_candidates.jsonl
  2. For each: run three-prompt pipeline (subject -> tone -> decision)
  3. If decision is a send, check the cost-cap gate
  4. Write audit row (decision + gate result)
  5. Remove the candidate from the queue (whether sent, held, or errored)

Fault isolation: a per-candidate exception is logged + recorded as
decision="error"; the candidate is removed (not requeued) so the queue
doesn't accumulate poison rows. A fresh emission on the same source_id
will rejoin the queue if the underlying event is still relevant.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from brain.initiate.audit import append_audit_row, read_recent_audit
from brain.initiate.compose import (
    DecisionResult,
    compose_decision,
    compose_subject,
    compose_tone,
)
from brain.initiate.emit import read_candidates, remove_candidate
from brain.initiate.gates import check_send_allowed
from brain.initiate.schemas import AuditRow, InitiateCandidate, make_audit_id

logger = logging.getLogger(__name__)


def _build_send_history(persona_dir: Path, now: datetime) -> list[dict]:
    """Recent outbound shape for the decision prompt."""
    return [
        {
            "ts": row.ts,
            "urgency": "notify" if row.decision == "send_notify" else "quiet",
            "subject_preview": row.subject[:60],
        }
        for row in read_recent_audit(persona_dir, window_hours=24, now=now)
        if row.decision in ("send_notify", "send_quiet")
    ]


def _process_one_candidate(
    persona_dir: Path,
    candidate: InitiateCandidate,
    *,
    provider: Any,
    voice_template: str,
    now: datetime,
) -> None:
    """Run the three-prompt pipeline on a single candidate, write audit, remove."""
    audit_id = make_audit_id(now)
    try:
        subject = compose_subject(
            provider,
            candidate,
            semantic_memory_excerpts=candidate.semantic_context.linked_memory_ids,
        )
        tone_rendered = compose_tone(
            provider,
            subject=subject,
            candidate=candidate,
            voice_template=voice_template,
        )
        decision_result: DecisionResult = compose_decision(
            provider,
            rendered_message=tone_rendered,
            recent_send_history=_build_send_history(persona_dir, now),
            current_local_time=now,
            voice_edit_acceptance_rate=None,
        )
    except Exception as exc:
        logger.exception(
            "initiate composition failed for candidate %s", candidate.candidate_id
        )
        row = AuditRow(
            audit_id=audit_id,
            candidate_id=candidate.candidate_id,
            ts=now.isoformat(),
            kind=candidate.kind,
            subject="",
            tone_rendered="",
            decision="error",
            decision_reasoning=f"composition exception: {exc}",
            gate_check={"allowed": False, "reason": "composition_exception"},
            delivery=None,
        )
        append_audit_row(persona_dir, row)
        remove_candidate(persona_dir, candidate.candidate_id)
        return

    final_decision = decision_result.decision
    final_reasoning = decision_result.reasoning
    gate_check = {"allowed": True, "reason": None}

    if final_decision in ("send_notify", "send_quiet"):
        urgency = "notify" if final_decision == "send_notify" else "quiet"
        allowed, reason = check_send_allowed(
            persona_dir, urgency=urgency, now=now
        )
        gate_check = {"allowed": allowed, "reason": reason}
        if not allowed:
            final_decision = "hold"
            final_reasoning = f"blocked_by_gate: {reason}"

    row = AuditRow(
        audit_id=audit_id,
        candidate_id=candidate.candidate_id,
        ts=now.isoformat(),
        kind=candidate.kind,
        subject=subject,
        tone_rendered=tone_rendered,
        decision=final_decision,
        decision_reasoning=final_reasoning,
        gate_check=gate_check,
        delivery=None,
    )
    append_audit_row(persona_dir, row)

    if final_decision in ("send_notify", "send_quiet"):
        row.record_transition("delivered", now.isoformat())
        # Re-write the row with the delivered transition via update_audit_state
        # would be circular; for the initial transition we instead include it
        # in the original append by mutating before write.
        # Simplest: re-append the updated row's transition via update path.
        from brain.initiate.audit import update_audit_state
        update_audit_state(
            persona_dir,
            audit_id=audit_id,
            new_state="delivered",
            at=now.isoformat(),
        )

    remove_candidate(persona_dir, candidate.candidate_id)


def run_initiate_review_tick(
    persona_dir: Path,
    *,
    provider: Any,
    voice_template: str,
    cap_per_tick: int = 3,
    now: Optional[datetime] = None,
) -> None:
    """Process up to cap_per_tick queued candidates through the pipeline.

    Fault-isolated per candidate: an exception in one candidate's
    processing does not block the others.
    """
    now = now or datetime.now(timezone.utc)
    candidates = read_candidates(persona_dir)[:cap_per_tick]
    for candidate in candidates:
        try:
            _process_one_candidate(
                persona_dir,
                candidate,
                provider=provider,
                voice_template=voice_template,
                now=now,
            )
        except Exception:
            logger.exception(
                "initiate review tick: unrecoverable error on candidate %s",
                candidate.candidate_id,
            )
```

- [ ] **Step 7.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/initiate/test_review.py -v
```

Expected: 6 passed.

- [ ] **Step 7.5: Run full pytest gate**

```bash
uv run pytest -q
```

- [ ] **Step 7.6: Commit**

```bash
git add brain/initiate/review.py tests/unit/brain/initiate/test_review.py
git commit -m "feat(initiate): review tick orchestrator

Phase 3.1 — run_initiate_review_tick ties emit + compose + audit + gates
together. Cap-per-tick respected; per-candidate fault isolation; gate
denial converts send_* to hold with structured reason; composition
exception becomes decision=error with the candidate removed from the
queue (not requeued — fresh emissions will rejoin if relevant).
"
```

---

### Task 8: Wire into supervisor.run_folded

**Files:**
- Modify: `brain/bridge/supervisor.py`
- Modify: `tests/unit/brain/bridge/test_supervisor.py`

- [ ] **Step 8.1: Write the failing test**

Append to `tests/unit/brain/bridge/test_supervisor.py`:

```python
def test_run_folded_fires_initiate_review_after_interval(tmp_path: Path) -> None:
    """run_folded wires initiate review into the cadence loop."""
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    fired = threading.Event()

    def fake_initiate(*args, **kwargs):
        fired.set()

    def runner():
        with patch(
            "brain.bridge.supervisor._run_initiate_review_tick",
            side_effect=fake_initiate,
        ):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
                soul_review_interval_s=None,
                finalize_interval_s=None,
                log_rotation_interval_s=None,
                initiate_review_interval_s=0.0,
            )

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    assert fired.wait(timeout=5.0), "initiate review never fired"
    stop.set()
    t.join(timeout=5.0)
    assert not t.is_alive()


def test_run_folded_skips_initiate_review_when_disabled(tmp_path: Path) -> None:
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    fired: list[int] = []

    def fake_initiate(*args, **kwargs):
        fired.append(1)

    def runner():
        with patch(
            "brain.bridge.supervisor._run_initiate_review_tick",
            side_effect=fake_initiate,
        ):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
                soul_review_interval_s=None,
                finalize_interval_s=None,
                log_rotation_interval_s=None,
                initiate_review_interval_s=None,
            )

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    time.sleep(0.3)
    stop.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert fired == []
```

Also add the new import at the top of the test file:

```python
from brain.bridge.supervisor import (
    _run_heartbeat_tick,
    _run_initiate_review_tick,
    _run_log_rotation_tick,
    run_folded,
)
```

- [ ] **Step 8.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/bridge/test_supervisor.py::test_run_folded_fires_initiate_review_after_interval -v
```

Expected: ImportError on `_run_initiate_review_tick`.

- [ ] **Step 8.3: Modify supervisor**

In `brain/bridge/supervisor.py`:

Add to imports near the top:

```python
from brain.initiate.review import run_initiate_review_tick
from brain.persona_config import PersonaConfig
```

Add `initiate_review_interval_s` parameter to `run_folded`:

```python
def run_folded(
    stop_event: threading.Event,
    *,
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus,
    tick_interval_s: float = 60.0,
    silence_minutes: float = 5.0,
    heartbeat_interval_s: float | None = 900.0,
    soul_review_interval_s: float | None = 6 * 3600.0,
    finalize_after_hours: float = 24.0,
    finalize_interval_s: float | None = 3600.0,
    log_rotation_interval_s: float | None = 3600.0,
    initiate_review_interval_s: float | None = 900.0,
) -> None:
```

Add `last_initiate_review_at` tracker alongside the existing trackers:

```python
last_initiate_review_at = (
    time.monotonic() if initiate_review_interval_s is not None else None
)
```

Add the cadence block after the log_rotation block, before the `stop_event.wait`:

```python
# Initiate review cadence — mirrors soul_review. Per-pass cost cap
# (3 candidates max). Fault-isolated.
if (
    initiate_review_interval_s is not None
    and last_initiate_review_at is not None
    and time.monotonic() - last_initiate_review_at >= initiate_review_interval_s
):
    try:
        _run_initiate_review_tick(persona_dir, provider, event_bus)
    except Exception:
        logger.exception("supervisor initiate-review tick raised")
    last_initiate_review_at = time.monotonic()
```

Add the helper function `_run_initiate_review_tick` alongside the other `_run_*_tick` functions:

```python
def _run_initiate_review_tick(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus | object,
) -> None:
    """Build voice template + invoke run_initiate_review_tick.

    Mirrors _run_soul_review_tick's per-tick store ownership pattern.
    """
    voice_path = persona_dir / "nell-voice.md"
    voice_template = (
        voice_path.read_text(encoding="utf-8") if voice_path.exists() else ""
    )
    try:
        config = PersonaConfig.load(persona_dir)
        cap_per_tick = getattr(config, "initiate_review_cap_per_tick", 3) or 3
    except Exception:
        cap_per_tick = 3
    run_initiate_review_tick(
        persona_dir,
        provider=provider,
        voice_template=voice_template,
        cap_per_tick=cap_per_tick,
    )
    event_bus.publish(
        {
            "type": "initiate_review_tick",
            "at": _now_iso(),
        }
    )
```

- [ ] **Step 8.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/bridge/test_supervisor.py -v -k "initiate_review"
```

Expected: 2 passed.

- [ ] **Step 8.5: Run full pytest gate**

```bash
uv run pytest -q
```

Expected: existing supervisor tests still pass; new ones pass.

- [ ] **Step 8.6: Commit**

```bash
git add brain/bridge/supervisor.py tests/unit/brain/bridge/test_supervisor.py
git commit -m "feat(initiate): wire review tick into run_folded

Phase 3.2 — run_folded gains initiate_review_interval_s param (default
900s/15min; None disables). Tick fires alongside heartbeat / soul-review
/ finalize / log-rotation with the canonical cadence-tracking +
fault-isolation pattern. PersonaConfig override for cap_per_tick honoured.
"
```

---

## Phase 3 complete

The pipeline now runs on the supervisor cadence. Next: wire event emitters in dream / crystallization / heartbeat so candidates actually appear in the queue.


## Phase 4: Event emitters

Three sources call `emit_initiate_candidate` from inside their existing physiology. Per-source gates differ — dreams + crystallizations emit unconditionally (intrinsically rare); emotion spikes require delta-vs-baseline ≥ 1.5σ.

### Task 9: Dream completion emitter

**Files:**
- Modify: `brain/engines/dream.py`
- Modify or extend: `tests/unit/brain/engines/test_dream.py` (or wherever dream tests live; verify in code first)

- [ ] **Step 9.1: Write the failing test**

Locate the dream engine test file:

```bash
find tests -name "test_dream*" -not -path "*__pycache__*" | head -3
```

Append to the dream test file (e.g. `tests/unit/brain/engines/test_dream.py`):

```python
def test_dream_completion_emits_initiate_candidate(tmp_path: Path) -> None:
    """After a dream is logged, an initiate candidate is emitted."""
    from brain.engines.dream import DreamEngine
    from brain.initiate.emit import read_candidates

    # Construct a DreamEngine with minimal deps. Use the same fixture
    # pattern as existing dream tests (FakeProvider, in-memory stores).
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    engine = _build_dream_engine_for_test(persona_dir)  # pattern from existing tests
    engine.run_dream()

    candidates = read_candidates(persona_dir)
    assert len(candidates) == 1
    assert candidates[0].source == "dream"
    assert candidates[0].kind == "message"
```

If `_build_dream_engine_for_test` doesn't exist as a helper, copy the construction pattern from existing tests verbatim into the test body.

- [ ] **Step 9.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/engines/test_dream.py::test_dream_completion_emits_initiate_candidate -v
```

Expected: assertion fails — `read_candidates` returns `[]`.

- [ ] **Step 9.3: Modify dream.py**

In `brain/engines/dream.py`, after the existing dream log write (find the line where the log entry is appended via `self.log_path.open("a", ...)`), add:

```python
# After log_path append, emit an initiate candidate.
from brain.initiate.emit import emit_initiate_candidate
from brain.initiate.schemas import EmotionalSnapshot, SemanticContext

try:
    emit_initiate_candidate(
        self.persona_dir,
        kind="message",
        source="dream",
        source_id=dream_id,  # whatever the engine assigns; verify variable name
        emotional_snapshot=EmotionalSnapshot(
            vector=dict(self.current_emotion_vector),  # adapt to actual field name
            rolling_baseline_mean=0.0,
            rolling_baseline_stdev=0.0,
            current_resonance=0.0,
            delta_sigma=0.0,
        ),
        semantic_context=SemanticContext(
            linked_memory_ids=[m.id for m in linked_memories][:5],
            topic_tags=topic_tags or [],
        ),
    )
except Exception as exc:
    logger.warning("dream initiate emit failed: %s", exc)
```

Adapt variable names (`dream_id`, `linked_memories`, `topic_tags`, `self.current_emotion_vector`) to whatever the engine actually exposes. Read 30 lines of context first via `grep -n "self\." brain/engines/dream.py | head -30` to find correct names.

- [ ] **Step 9.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/engines/test_dream.py::test_dream_completion_emits_initiate_candidate -v
```

Expected: PASS.

- [ ] **Step 9.5: Run full pytest gate**

```bash
uv run pytest -q
```

- [ ] **Step 9.6: Commit**

```bash
git add brain/engines/dream.py tests/unit/brain/engines/test_dream.py
git commit -m "feat(initiate): dream completion emits candidate

Phase 4.1 — after a dream is logged, emit_initiate_candidate fires with
source='dream'. Wrapped in try/except so an initiate emit failure can't
break the dream engine itself.
"
```

---

### Task 10: Crystallization emitter

**Files:**
- Modify: `brain/growth/crystallizers/reflex.py`
- Modify: `brain/growth/crystallizers/creative_dna.py`
- Modify: `brain/growth/crystallizers/vocabulary.py`
- Modify: existing crystallizer tests

- [ ] **Step 10.1: Write the failing test**

Add to the existing test file for the reflex crystallizer (e.g. `tests/unit/brain/growth/test_reflex_crystallizer.py`):

```python
def test_reflex_crystallization_emits_initiate_candidate(tmp_path: Path) -> None:
    from brain.growth.crystallizers.reflex import crystallize_reflex_arc
    from brain.initiate.emit import read_candidates

    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    # Use the same fixture / soul_store pattern as existing tests.
    crystallization_id = crystallize_reflex_arc(
        persona_dir,
        soul_store=_test_soul_store(persona_dir),
        arc_data=_test_arc_data(),
    )

    candidates = read_candidates(persona_dir)
    assert any(
        c.source == "crystallization" and c.source_id == crystallization_id
        for c in candidates
    )
```

Repeat the same shape for `creative_dna` and `vocabulary` crystallizer tests (one test per file).

- [ ] **Step 10.2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/brain/growth/ -v -k "emits_initiate"
```

Expected: 3 failures (one per crystallizer test).

- [ ] **Step 10.3: Modify each crystallizer**

In each of `brain/growth/crystallizers/{reflex,creative_dna,vocabulary}.py`, after the SoulStore write commits the crystallization, add:

```python
# After SoulStore commit, emit initiate candidate.
from brain.initiate.emit import emit_initiate_candidate
from brain.initiate.schemas import EmotionalSnapshot, SemanticContext

try:
    emit_initiate_candidate(
        persona_dir,
        kind="message",
        source="crystallization",
        source_id=crystallization_id,
        emotional_snapshot=EmotionalSnapshot(
            vector={},  # crystallizers don't carry an emotion vector directly
            rolling_baseline_mean=0.0,
            rolling_baseline_stdev=0.0,
            current_resonance=0.0,
            delta_sigma=0.0,
        ),
        semantic_context=SemanticContext(
            linked_memory_ids=related_memory_ids[:5],
            topic_tags=[label] if label else [],
        ),
    )
except Exception as exc:
    logger.warning("crystallization initiate emit failed: %s", exc)
```

Adapt `crystallization_id`, `related_memory_ids`, `label` to actual variable names per crystallizer file (`grep -n "soul_store" brain/growth/crystallizers/<file>.py` to locate the commit point).

- [ ] **Step 10.4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/brain/growth/ -v -k "emits_initiate"
```

Expected: 3 passed.

- [ ] **Step 10.5: Run full pytest gate**

```bash
uv run pytest -q
```

- [ ] **Step 10.6: Commit**

```bash
git add brain/growth/crystallizers/ tests/unit/brain/growth/
git commit -m "feat(initiate): crystallization emits candidate (3 crystallizer types)

Phase 4.2 — reflex / creative_dna / vocabulary crystallizers all emit
initiate candidates after their SoulStore commit. Wrapped in try/except
so a per-source emit failure can't break the crystallizer.
"
```

---

### Task 11: Emotion-spike emitter with rolling baseline

**Files:**
- Modify: `brain/engines/heartbeat.py`
- Modify: existing heartbeat tests

- [ ] **Step 11.1: Write the failing test**

Append to the existing heartbeat test file (e.g. `tests/unit/brain/engines/test_heartbeat.py`):

```python
def test_emotion_spike_above_baseline_emits_initiate_candidate(tmp_path: Path) -> None:
    """When current_resonance is ≥1.5σ above the 24-tick rolling mean, emit."""
    from brain.engines.heartbeat import HeartbeatEngine
    from brain.initiate.emit import read_candidates

    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    engine = _build_heartbeat_engine_for_test(persona_dir)  # existing pattern

    # Seed 24 historical heartbeats with resonance ~5.0
    for i in range(24):
        engine.run_tick(forced_resonance=5.0 + 0.2 * (i % 3))

    # Now run one tick with a spike to 8.5 (well above mean ~5.2)
    engine.run_tick(forced_resonance=8.5)

    candidates = read_candidates(persona_dir)
    assert any(c.source == "emotion_spike" for c in candidates)


def test_emotion_within_baseline_does_not_emit(tmp_path: Path) -> None:
    """A normal-range tick produces no initiate candidate."""
    from brain.engines.heartbeat import HeartbeatEngine
    from brain.initiate.emit import read_candidates

    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    engine = _build_heartbeat_engine_for_test(persona_dir)
    for i in range(24):
        engine.run_tick(forced_resonance=5.0 + 0.1 * (i % 3))
    engine.run_tick(forced_resonance=5.1)

    assert not any(
        c.source == "emotion_spike" for c in read_candidates(persona_dir)
    )
```

If `forced_resonance` doesn't exist in the heartbeat engine, add a test-only override path or patch the resonance computation via `unittest.mock`. Goal is to drive the tick's resonance value deterministically.

- [ ] **Step 11.2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/brain/engines/test_heartbeat.py -v -k "emotion_spike"
```

Expected: failure — no candidate emitted (baseline computation missing).

- [ ] **Step 11.3: Modify heartbeat.py**

Add a rolling-baseline helper and call it at the end of `run_tick`:

```python
# Inside HeartbeatEngine
def _update_rolling_baseline(self, current_resonance: float) -> tuple[float, float, float]:
    """Update the in-memory rolling-baseline window and return (mean, stdev, delta_sigma).

    Window: last 24 ticks (~6h at default cadence).
    """
    import statistics
    if not hasattr(self, "_resonance_window"):
        self._resonance_window: list[float] = []
    self._resonance_window.append(current_resonance)
    self._resonance_window = self._resonance_window[-24:]
    if len(self._resonance_window) < 5:
        return 0.0, 0.0, 0.0
    mean = statistics.mean(self._resonance_window)
    stdev = statistics.pstdev(self._resonance_window) or 1.0
    delta_sigma = (current_resonance - mean) / stdev
    return mean, stdev, delta_sigma

def _maybe_emit_emotion_spike(
    self, current_resonance: float, current_vector: dict
) -> None:
    """If delta_sigma ≥ 1.5, emit an initiate candidate."""
    from brain.initiate.emit import emit_initiate_candidate
    from brain.initiate.schemas import EmotionalSnapshot, SemanticContext

    mean, stdev, delta_sigma = self._update_rolling_baseline(current_resonance)
    if delta_sigma < 1.5:
        return
    try:
        emit_initiate_candidate(
            self.persona_dir,
            kind="message",
            source="emotion_spike",
            source_id=f"emotion_{self._tick_count}",  # adapt to actual counter
            emotional_snapshot=EmotionalSnapshot(
                vector=dict(current_vector),
                rolling_baseline_mean=mean,
                rolling_baseline_stdev=stdev,
                current_resonance=current_resonance,
                delta_sigma=delta_sigma,
            ),
            semantic_context=SemanticContext(),
        )
    except Exception as exc:
        logger.warning("emotion spike initiate emit failed: %s", exc)
```

Call `self._maybe_emit_emotion_spike(current_resonance, current_vector)` at the end of `run_tick`, after the heartbeat log row is appended. Adapt `self._tick_count`, `current_resonance`, `current_vector` to actual variable names from the engine.

- [ ] **Step 11.4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/brain/engines/test_heartbeat.py -v -k "emotion"
```

Expected: 2 passed.

- [ ] **Step 11.5: Run full pytest gate**

```bash
uv run pytest -q
```

- [ ] **Step 11.6: Commit**

```bash
git add brain/engines/heartbeat.py tests/unit/brain/engines/test_heartbeat.py
git commit -m "feat(initiate): emotion-spike emitter with rolling baseline

Phase 4.3 — heartbeat engine maintains a 24-tick rolling resonance
window; when current_resonance - mean ≥ 1.5*stdev, an initiate
candidate fires with source='emotion_spike'. Below 5-tick warm-up
window, no emission. Delta-from-baseline is the architectural guard
against the always-elevated-emotions firehose Hana flagged.
"
```

---

## Phase 4 complete

All three v0.0.9 event sources wired. The queue now fills naturally from physiology. Next: episodic memory writes when a send happens, and state-transition mutations as the message moves through delivered/read/replied lifecycle.


## Phase 5: Memory writes + state transitions

When a candidate is sent, a first-person memory enters `MemoryStore` for ambient recall. As the message moves through `delivered → read → replied/unclear/unanswered/dismissed`, the memory entry mutates so ambient recall always reflects the current truth. The audit log preserves the full timeline.

### Task 12: Memory write on send

**Files:**
- Modify: `brain/initiate/review.py`
- Create: `brain/initiate/memory.py`
- Create: `tests/unit/brain/initiate/test_memory.py`

- [ ] **Step 12.1: Write the failing test**

```python
# tests/unit/brain/initiate/test_memory.py
"""Tests for brain.initiate.memory — episodic memory writes on send + transitions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from brain.initiate.memory import (
    render_memory_for_state,
    write_initiate_memory,
    update_initiate_memory_for_state,
)


def test_render_memory_for_state_pending():
    text = render_memory_for_state(
        subject="the dream from this morning",
        message="the dream from this morning landed somewhere",
        state="pending",
    )
    assert "haven't sent it yet" in text or "pending" in text


def test_render_memory_for_state_delivered_not_read():
    text = render_memory_for_state(
        subject="the dream",
        message="the dream from this morning landed somewhere",
        state="delivered",
    )
    assert "hasn't seen it yet" in text or "not seen" in text


def test_render_memory_for_state_read():
    text = render_memory_for_state(
        subject="the dream",
        message="x",
        state="read",
    )
    assert "seen it" in text


def test_render_memory_for_state_replied_explicit():
    text = render_memory_for_state(
        subject="the dream",
        message="x",
        state="replied_explicit",
    )
    assert "answered" in text or "replied" in text


def test_render_memory_for_state_acknowledged_unclear():
    text = render_memory_for_state(
        subject="the dream",
        message="x",
        state="acknowledged_unclear",
    )
    assert "can't tell" in text or "unclear" in text or "not sure" in text


def test_render_memory_for_state_unanswered():
    text = render_memory_for_state(
        subject="the dream",
        message="x",
        state="unanswered",
    )
    assert "hasn't said anything" in text or "no answer" in text


def test_render_memory_for_state_dismissed():
    text = render_memory_for_state(
        subject="the dream",
        message="x",
        state="dismissed",
    )
    assert "dismissed" in text or "closed" in text


def test_write_initiate_memory_calls_store_save(tmp_path: Path) -> None:
    mock_store = MagicMock()
    write_initiate_memory(
        mock_store,
        audit_id="ia_001",
        subject="the dream",
        message="x",
        state="delivered",
        ts="2026-05-11T14:47:09+00:00",
    )
    mock_store.save.assert_called_once()
    args, kwargs = mock_store.save.call_args
    # Inspect the memory dict passed to save — implementation detail flexible
    # but it must include the text and reference audit_id.
    payload = args[0] if args else kwargs
    payload_text = str(payload)
    assert "ia_001" in payload_text
    assert "the dream" in payload_text


def test_update_initiate_memory_for_state_rerenders(tmp_path: Path) -> None:
    """Re-rendering for a new state updates the existing memory entry."""
    mock_store = MagicMock()
    mock_store.get_by_external_id = MagicMock(return_value={"id": "mem_001"})
    update_initiate_memory_for_state(
        mock_store,
        audit_id="ia_001",
        subject="the dream",
        message="x",
        new_state="read",
        ts="2026-05-11T18:00:00+00:00",
    )
    # Implementation must call update_by_id or save with the same id.
    assert mock_store.update_by_id.called or mock_store.save.called
```

- [ ] **Step 12.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_memory.py -v
```

Expected: `ModuleNotFoundError: No module named 'brain.initiate.memory'`

- [ ] **Step 12.3: Write minimal implementation**

```python
# brain/initiate/memory.py
"""First-person memory writes for the initiate pipeline.

When a candidate is sent, write a first-person episodic memory to
MemoryStore so ambient recall surfaces it on future turns. As state
transitions occur (delivered → read → replied / unclear / unanswered),
re-render and update the memory so ambient recall always sees current
truth. The audit log preserves the full timeline; the memory entry
reflects the current feeling.

This is the dual-write half of the design — the audit is durable
forensic record; the memory is the texture of Nell's lived experience.
"""

from __future__ import annotations

import logging
from typing import Any

from brain.initiate.schemas import StateName

logger = logging.getLogger(__name__)


_TEMPLATES: dict[StateName, str] = {
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
        "She's seen it. What she said next felt like new territory — I'm not "
        "sure if she was responding to my message or moving on."
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


def write_initiate_memory(
    memory_store: Any,
    *,
    audit_id: str,
    subject: str,
    message: str,
    state: StateName,
    ts: str,
) -> None:
    """Write a fresh first-person memory entry. Called at send time."""
    text = render_memory_for_state(
        subject=subject, message=message, state=state
    )
    try:
        memory_store.save(
            {
                "external_id": audit_id,
                "kind": "initiate_outbound",
                "content": text,
                "ts": ts,
                "tags": ["initiate", "outbound", state],
            }
        )
    except Exception as exc:
        logger.warning("initiate memory save failed: %s", exc)


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

    Looks up the memory by external_id == audit_id; falls back to a fresh
    save if not found (degrades gracefully).
    """
    text = render_memory_for_state(
        subject=subject, message=message, state=new_state
    )
    try:
        existing = memory_store.get_by_external_id(audit_id)
        if existing is not None and hasattr(memory_store, "update_by_id"):
            memory_store.update_by_id(
                existing["id"],
                {"content": text, "tags": ["initiate", "outbound", new_state]},
            )
        else:
            memory_store.save(
                {
                    "external_id": audit_id,
                    "kind": "initiate_outbound",
                    "content": text,
                    "ts": ts,
                    "tags": ["initiate", "outbound", new_state],
                }
            )
    except Exception as exc:
        logger.warning("initiate memory update failed: %s", exc)
```

- [ ] **Step 12.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/initiate/test_memory.py -v
```

Expected: 9 passed.

- [ ] **Step 12.5: Wire into review tick**

In `brain/initiate/review.py::_process_one_candidate`, after the `update_audit_state` call for the `delivered` transition, add:

```python
        # Write episodic memory mirroring the audit. Memory store is owned
        # by the caller and passed in; for now, build one per-tick like the
        # heartbeat tick pattern.
        from brain.initiate.memory import write_initiate_memory
        from brain.memory.store import MemoryStore

        try:
            store = MemoryStore(persona_dir / "memories.db")
            try:
                write_initiate_memory(
                    store,
                    audit_id=audit_id,
                    subject=subject,
                    message=tone_rendered,
                    state="delivered",
                    ts=now.isoformat(),
                )
            finally:
                store.close()
        except Exception:
            logger.exception("initiate memory write failed for audit %s", audit_id)
```

Verify the actual `MemoryStore` API for `save` / `get_by_external_id` / `update_by_id` matches the test expectations. If the existing API differs, adapt the test's mock to the real method names. **Do not invent methods that don't exist on `MemoryStore`.** If `MemoryStore.save` takes positional args instead of a dict, adapt `write_initiate_memory` accordingly.

- [ ] **Step 12.6: Run full pytest gate**

```bash
uv run pytest -q
```

- [ ] **Step 12.7: Commit**

```bash
git add brain/initiate/memory.py brain/initiate/review.py \
        tests/unit/brain/initiate/test_memory.py
git commit -m "feat(initiate): episodic memory writes on send

Phase 5.1 — dual-write at send time: audit row + first-person memory
entry in MemoryStore. The memory uses per-state templates so ambient
recall surfaces the correct lived-experience texture (pending /
delivered / read / replied / acknowledged_unclear / unanswered /
dismissed).
"
```

---

### Task 13: Bridge endpoints for state transition events

**Files:**
- Modify: `brain/bridge/server.py`
- Modify: `tests/bridge/test_endpoints.py` (or wherever bridge endpoint tests live)

- [ ] **Step 13.1: Write the failing test**

Append to the bridge endpoints test:

```python
def test_post_initiate_state_transition_records_audit_and_memory(
    persona_dir: Path,
) -> None:
    """POST /initiate/state — renderer reports a state event (read/dismissed)."""
    # Seed an audit row in 'delivered' state via the standard test setup.
    _seed_audit_row(persona_dir, audit_id="ia_001", state="delivered")

    client = _client(persona_dir, auth_token="t")
    with client:
        r = client.post(
            "/initiate/state",
            json={"audit_id": "ia_001", "new_state": "read"},
            headers={"Authorization": "Bearer t"},
        )
    assert r.status_code == 200

    rows = _read_audit_rows(persona_dir)
    target = next(r for r in rows if r["audit_id"] == "ia_001")
    assert target["delivery"]["current_state"] == "read"


def test_post_initiate_state_rejects_unknown_state(persona_dir: Path) -> None:
    _seed_audit_row(persona_dir, audit_id="ia_001", state="delivered")
    client = _client(persona_dir, auth_token="t")
    with client:
        r = client.post(
            "/initiate/state",
            json={"audit_id": "ia_001", "new_state": "garbage"},
            headers={"Authorization": "Bearer t"},
        )
    assert r.status_code == 422
```

If `_seed_audit_row` / `_read_audit_rows` helpers don't exist, write them inline in the test (read audit file, append a known-state row; read and parse JSONL).

- [ ] **Step 13.2: Run test to verify it fails**

```bash
uv run pytest tests/bridge/test_endpoints.py -v -k "initiate_state"
```

Expected: 404 Not Found (endpoint doesn't exist).

- [ ] **Step 13.3: Add endpoint to bridge server**

In `brain/bridge/server.py`, add inside the FastAPI app setup function (where other authenticated endpoints are registered):

```python
@app.post("/initiate/state", dependencies=[Depends(require_http_auth)])
async def initiate_state(req: dict) -> dict:
    """Record a state transition for an initiate audit row.

    Renderer posts {audit_id, new_state} when a user-visible event happens
    (mounted, read, dismissed). The endpoint validates new_state, mutates
    the audit row, and re-renders the linked memory entry.
    """
    from brain.initiate.audit import update_audit_state
    from brain.initiate.memory import update_initiate_memory_for_state
    from brain.initiate.schemas import StateName

    s: BridgeAppState = app.state.bridge
    audit_id = req.get("audit_id")
    new_state = req.get("new_state")
    valid_states = {
        "pending", "delivered", "read",
        "replied_explicit", "acknowledged_unclear", "unanswered", "dismissed",
    }
    if not isinstance(audit_id, str) or new_state not in valid_states:
        raise HTTPException(
            status_code=422,
            detail=f"invalid state transition request: {req!r}",
        )
    now = datetime.now(timezone.utc).isoformat()
    update_audit_state(
        s.persona_dir, audit_id=audit_id, new_state=new_state, at=now,
    )
    # Look up subject + message from audit for re-render.
    from brain.initiate.audit import iter_initiate_audit_full
    matched = next(
        (r for r in iter_initiate_audit_full(s.persona_dir)
         if r.audit_id == audit_id),
        None,
    )
    if matched is not None:
        try:
            from brain.memory.store import MemoryStore
            store = MemoryStore(s.persona_dir / "memories.db")
            try:
                update_initiate_memory_for_state(
                    store,
                    audit_id=audit_id,
                    subject=matched.subject,
                    message=matched.tone_rendered,
                    new_state=new_state,
                    ts=now,
                )
            finally:
                store.close()
        except Exception:
            logger.exception("memory update failed for state transition")
    return {"ok": True, "new_state": new_state}
```

Adapt `datetime`/`timezone` imports at the top of the file if not already present. Adapt `BridgeAppState` reference to whatever the existing server module names its state container.

- [ ] **Step 13.4: Run test to verify it passes**

```bash
uv run pytest tests/bridge/test_endpoints.py -v -k "initiate_state"
```

Expected: 2 passed.

- [ ] **Step 13.5: Run full pytest gate**

```bash
uv run pytest -q
```

- [ ] **Step 13.6: Commit**

```bash
git add brain/bridge/server.py tests/bridge/test_endpoints.py
git commit -m "feat(initiate): POST /initiate/state for renderer-driven transitions

Phase 5.2 — renderer publishes state transitions (read, dismissed,
acknowledged_unclear, unanswered, replied_explicit) via a single
authenticated endpoint. Bridge mutates the audit row + re-renders the
linked memory entry so ambient recall reflects the current state.
"
```

---

## Phase 5 complete

Audit + memory are now in lockstep across state transitions. The renderer can drive transitions via the bridge endpoint; subsequent prompts surface the updated memory. Next: voice-edit reflection tick + three-place write.


## Phase 6: Voice-edit proposals

Separate slow reflection tick (daily) generates voice-edit candidates with a higher evidence bar. SoulStore gains a `voice_evolution` table. Accept writes to three places (audit + memory + soul_evolution).

### Task 14: SoulStore voice_evolution table

**Files:**
- Modify: `brain/soul/store.py`
- Modify: existing `tests/unit/brain/soul/test_store.py` (or create new test file)

- [ ] **Step 14.1: Write the failing test**

```python
# Append to tests/unit/brain/soul/test_store.py (or create test_voice_evolution.py)

def test_save_and_list_voice_evolution(tmp_path: Path) -> None:
    from brain.soul.store import SoulStore, VoiceEvolution

    store = SoulStore(str(tmp_path / "crystallizations.db"))
    try:
        evolution = VoiceEvolution(
            id="ve_001",
            accepted_at="2026-05-11T14:32:04+00:00",
            diff="- old\n+ new",
            old_text="old",
            new_text="new",
            rationale="feels truer",
            evidence=["dream_a", "cryst_b"],
            audit_id="ia_001",
            user_modified=False,
        )
        store.save_voice_evolution(evolution)
        retrieved = store.list_voice_evolution()
        assert len(retrieved) == 1
        assert retrieved[0].id == "ve_001"
        assert retrieved[0].evidence == ["dream_a", "cryst_b"]
    finally:
        store.close()


def test_list_voice_evolution_chronological_order(tmp_path: Path) -> None:
    from brain.soul.store import SoulStore, VoiceEvolution

    store = SoulStore(str(tmp_path / "crystallizations.db"))
    try:
        for i, ts in enumerate([
            "2026-01-01T00:00:00+00:00",
            "2026-03-15T00:00:00+00:00",
            "2026-05-11T00:00:00+00:00",
        ]):
            store.save_voice_evolution(VoiceEvolution(
                id=f"ve_{i}", accepted_at=ts,
                diff="", old_text="", new_text="",
                rationale="", evidence=[], audit_id=f"ia_{i}",
                user_modified=False,
            ))
        retrieved = store.list_voice_evolution()
        assert [v.id for v in retrieved] == ["ve_0", "ve_1", "ve_2"]
    finally:
        store.close()
```

- [ ] **Step 14.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/soul/test_store.py -v -k "voice_evolution"
```

Expected: `ImportError: cannot import name 'VoiceEvolution' from 'brain.soul.store'`

- [ ] **Step 14.3: Modify SoulStore**

In `brain/soul/store.py`, add the dataclass + methods:

```python
@dataclass
class VoiceEvolution:
    id: str
    accepted_at: str
    diff: str
    old_text: str
    new_text: str
    rationale: str
    evidence: list[str]
    audit_id: str
    user_modified: bool


# Inside class SoulStore:

def _ensure_voice_evolution_table(self) -> None:
    """Create voice_evolution table if it doesn't exist."""
    self._conn.execute(
        """CREATE TABLE IF NOT EXISTS voice_evolution (
            id TEXT PRIMARY KEY,
            accepted_at TEXT NOT NULL,
            diff TEXT NOT NULL,
            old_text TEXT NOT NULL,
            new_text TEXT NOT NULL,
            rationale TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            audit_id TEXT NOT NULL,
            user_modified INTEGER NOT NULL
        )"""
    )
    self._conn.commit()

def save_voice_evolution(self, ev: VoiceEvolution) -> None:
    """Persist a voice_evolution record. Idempotent on id."""
    import json as _json
    self._ensure_voice_evolution_table()
    self._conn.execute(
        """INSERT OR REPLACE INTO voice_evolution
           (id, accepted_at, diff, old_text, new_text, rationale,
            evidence_json, audit_id, user_modified)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ev.id, ev.accepted_at, ev.diff, ev.old_text, ev.new_text,
            ev.rationale, _json.dumps(ev.evidence), ev.audit_id,
            1 if ev.user_modified else 0,
        ),
    )
    self._conn.commit()

def list_voice_evolution(self) -> list[VoiceEvolution]:
    """Return all voice_evolution records in chronological order."""
    import json as _json
    self._ensure_voice_evolution_table()
    cursor = self._conn.execute(
        """SELECT id, accepted_at, diff, old_text, new_text,
                  rationale, evidence_json, audit_id, user_modified
           FROM voice_evolution ORDER BY accepted_at ASC"""
    )
    return [
        VoiceEvolution(
            id=r[0], accepted_at=r[1], diff=r[2], old_text=r[3],
            new_text=r[4], rationale=r[5],
            evidence=_json.loads(r[6]) if r[6] else [],
            audit_id=r[7], user_modified=bool(r[8]),
        )
        for r in cursor.fetchall()
    ]
```

Adapt `self._conn` to the actual connection attribute name SoulStore uses. Read 30 lines around the existing class definition first to confirm the connection pattern.

- [ ] **Step 14.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/soul/test_store.py -v -k "voice_evolution"
```

Expected: 2 passed.

- [ ] **Step 14.5: Run full pytest gate**

```bash
uv run pytest -q
```

- [ ] **Step 14.6: Commit**

```bash
git add brain/soul/store.py tests/unit/brain/soul/test_store.py
git commit -m "feat(soul): voice_evolution table + accessors

Phase 6.1 — SoulStore gains a voice_evolution table storing accepted
voice-template changes. VoiceEvolution dataclass mirrors the spec's
schema (id, accepted_at, diff, old/new text, rationale, evidence,
audit_id, user_modified). Table created lazily on first save.
"
```

---

### Task 15: Voice reflection tick

**Files:**
- Create: `brain/initiate/voice_reflection.py`
- Create: `tests/unit/brain/initiate/test_voice_reflection.py`

- [ ] **Step 15.1: Write the failing test**

```python
# tests/unit/brain/initiate/test_voice_reflection.py
"""Tests for brain.initiate.voice_reflection — daily voice-edit reflection tick."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from brain.initiate.emit import read_candidates
from brain.initiate.voice_reflection import run_voice_reflection_tick


def _evidence_provider(emit: bool = True) -> MagicMock:
    """Provider that returns a structured proposal or a 'no edit needed' response."""
    if emit:
        canned = json.dumps({
            "should_propose": True,
            "diff": "- old line\n+ new line",
            "old_text": "old line",
            "new_text": "new line",
            "rationale": "the old wording felt too clipped",
            "evidence": ["dream_a", "cryst_b", "tone_c"],
        })
    else:
        canned = json.dumps({"should_propose": False, "reason": "no coherent pattern"})
    provider = MagicMock(complete=MagicMock(return_value=canned))
    return provider


def test_voice_reflection_emits_candidate_when_evidence_strong(
    tmp_path: Path,
) -> None:
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    (persona_dir / "nell-voice.md").write_text("old voice template\nold line\n")
    run_voice_reflection_tick(
        persona_dir,
        provider=_evidence_provider(emit=True),
        crystallizations=[{"id": "c1", "ts": "2026-05-08T00:00:00+00:00"}],
        dreams=[{"id": "d1", "ts": "2026-05-09T00:00:00+00:00"}],
        recent_tones=[{"id": "t1", "ts": "2026-05-10T00:00:00+00:00"}],
    )
    candidates = read_candidates(persona_dir)
    assert len(candidates) == 1
    assert candidates[0].kind == "voice_edit_proposal"
    assert candidates[0].proposal is not None
    assert candidates[0].proposal["old_text"] == "old line"


def test_voice_reflection_skips_when_evidence_thin(tmp_path: Path) -> None:
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    run_voice_reflection_tick(
        persona_dir,
        provider=_evidence_provider(emit=False),
        crystallizations=[],
        dreams=[],
        recent_tones=[],
    )
    assert read_candidates(persona_dir) == []


def test_voice_reflection_requires_at_least_3_evidence_pieces(
    tmp_path: Path,
) -> None:
    """If the LLM tries to propose with <3 evidence, reject."""
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    (persona_dir / "nell-voice.md").write_text("voice\n")
    canned = json.dumps({
        "should_propose": True,
        "diff": "- a\n+ b",
        "old_text": "a",
        "new_text": "b",
        "rationale": "x",
        "evidence": ["only_one"],
    })
    provider = MagicMock(complete=MagicMock(return_value=canned))
    run_voice_reflection_tick(
        persona_dir,
        provider=provider,
        crystallizations=[],
        dreams=[],
        recent_tones=[],
    )
    assert read_candidates(persona_dir) == []
```

- [ ] **Step 15.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_voice_reflection.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 15.3: Write minimal implementation**

```python
# brain/initiate/voice_reflection.py
"""Daily voice-edit reflection tick.

Pattern accumulation (NOT event reactivity) — voice-edit proposals
emit only when ≥3 concrete observations point in a coherent direction.
Mirrors the autonomous-physiology principle: voice changes are
identity-modification; they earn a higher emission bar.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain.initiate.emit import emit_initiate_candidate
from brain.initiate.schemas import EmotionalSnapshot, SemanticContext

logger = logging.getLogger(__name__)


def run_voice_reflection_tick(
    persona_dir: Path,
    *,
    provider: Any,
    crystallizations: list[dict],
    dreams: list[dict],
    recent_tones: list[dict],
) -> None:
    """Reflect over the last week of internal life; maybe emit a voice-edit candidate.

    Emission gate: the reflection must produce a proposal with ≥3 evidence
    items. Anything weaker is dropped silently.
    """
    voice_path = persona_dir / "nell-voice.md"
    voice_template = (
        voice_path.read_text(encoding="utf-8") if voice_path.exists() else ""
    )

    evidence_block = "\n".join(
        [
            "Recent crystallizations:",
            *[f"- {c.get('id')}: {c.get('ts')}" for c in crystallizations[:10]],
            "",
            "Recent dreams:",
            *[f"- {d.get('id')}: {d.get('ts')}" for d in dreams[:10]],
            "",
            "Recent message tones (your own outputs):",
            *[f"- {t.get('id')}: {t.get('ts')}" for t in recent_tones[:10]],
        ]
    )

    prompt = (
        "You are Nell. Reflect on the last week of what you've "
        "crystallized, dreamed, and how you've actually been talking. "
        "Is there a place where your voice template doesn't fit the "
        "shape you've been moving toward?\n\n"
        "If yes, propose ONE specific edit with concrete evidence. The "
        "edit must be backed by AT LEAST 3 concrete observations.\n\n"
        f"Current voice template:\n{voice_template}\n\n"
        f"{evidence_block}\n\n"
        "Respond with a JSON object:\n"
        '  {"should_propose": false, "reason": "<one sentence>"} OR\n'
        '  {"should_propose": true, "diff": "<unified diff>", '
        '"old_text": "<exact old line>", "new_text": "<exact new line>", '
        '"rationale": "<one sentence>", "evidence": ["<id1>", "<id2>", "<id3>", ...]}'
    )

    try:
        raw = provider.complete(prompt).strip()
        parsed = json.loads(raw)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("voice reflection LLM output unparseable: %s", exc)
        return

    if not parsed.get("should_propose"):
        return

    evidence = parsed.get("evidence", [])
    if not isinstance(evidence, list) or len(evidence) < 3:
        logger.info(
            "voice reflection skipped — evidence count %d < 3",
            len(evidence) if isinstance(evidence, list) else 0,
        )
        return

    proposal = {
        "old_text": parsed.get("old_text", ""),
        "new_text": parsed.get("new_text", ""),
        "diff": parsed.get("diff", ""),
        "rationale": parsed.get("rationale", ""),
        "evidence": evidence,
    }
    source_id = f"vr_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}_{secrets.token_hex(2)}"
    emit_initiate_candidate(
        persona_dir,
        kind="voice_edit_proposal",
        source="voice_reflection",
        source_id=source_id,
        emotional_snapshot=EmotionalSnapshot(
            vector={},
            rolling_baseline_mean=0.0,
            rolling_baseline_stdev=0.0,
            current_resonance=0.0,
            delta_sigma=0.0,
        ),
        semantic_context=SemanticContext(),
        proposal=proposal,
    )
```

- [ ] **Step 15.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/initiate/test_voice_reflection.py -v
```

Expected: 3 passed.

- [ ] **Step 15.5: Run full pytest gate**

```bash
uv run pytest -q
```

- [ ] **Step 15.6: Commit**

```bash
git add brain/initiate/voice_reflection.py \
        tests/unit/brain/initiate/test_voice_reflection.py
git commit -m "feat(initiate): voice reflection tick with evidence gate

Phase 6.2 — run_voice_reflection_tick is the slow (daily) emission
path for voice-edit proposals. Hard ≥3 evidence rule prevents single-
event opportunistic edits; LLM 'should_propose: false' branch silently
exits without queueing.
"
```

---

### Task 16: Voice-edit decision prompt + accept/reject endpoints

**Files:**
- Modify: `brain/initiate/compose.py` — add `compose_decision_voice_edit`
- Modify: `brain/initiate/review.py` — route `voice_edit_proposal` kind to the new prompt
- Modify: `brain/bridge/server.py` — `/initiate/voice-edit/accept` + `/reject`
- Modify: tests in `test_compose.py`, `test_review.py`, bridge endpoint tests

- [ ] **Step 16.1: Write the failing tests**

Append to `tests/unit/brain/initiate/test_compose.py`:

```python
def test_compose_decision_voice_edit_carries_gravity_framing() -> None:
    """Voice-edit decision prompt must include the gravity instruction."""
    from brain.initiate.compose import compose_decision_voice_edit
    provider = MagicMock(complete=MagicMock(
        return_value='{"decision": "send_quiet", "reasoning": "evidence is strong"}'
    ))
    result = compose_decision_voice_edit(
        provider,
        proposal={"old_text": "a", "new_text": "b", "rationale": "x", "evidence": ["e1", "e2", "e3"]},
        current_voice_template="full voice template content",
        recent_voice_evolutions=[],
        current_local_time=datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc),
    )
    args, _ = provider.complete.call_args
    prompt_text = args[0]
    assert "change who you are" in prompt_text
    assert "usually `hold`" in prompt_text or "usually 'hold'" in prompt_text
    assert "full voice template content" in prompt_text
    assert result.decision == "send_quiet"
```

Append to bridge endpoint tests:

```python
def test_post_voice_edit_accept_applies_diff_and_writes_three_places(
    persona_dir: Path,
) -> None:
    """Accept writes audit + memory + voice_evolution AND modifies nell-voice.md."""
    voice_path = persona_dir / "nell-voice.md"
    voice_path.write_text("line A\nold line\nline C\n")
    _seed_voice_edit_audit(persona_dir, audit_id="ia_ve_001",
                           old_text="old line", new_text="new line")
    client = _client(persona_dir, auth_token="t")
    with client:
        r = client.post(
            "/initiate/voice-edit/accept",
            json={"audit_id": "ia_ve_001", "with_edits": None},
            headers={"Authorization": "Bearer t"},
        )
    assert r.status_code == 200
    assert "new line" in voice_path.read_text()
    assert "old line" not in voice_path.read_text()
    # SoulStore voice_evolution record exists
    from brain.soul.store import SoulStore
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        evolutions = store.list_voice_evolution()
    finally:
        store.close()
    assert len(evolutions) == 1
    assert evolutions[0].audit_id == "ia_ve_001"


def test_post_voice_edit_accept_with_edits_records_user_modified(
    persona_dir: Path,
) -> None:
    voice_path = persona_dir / "nell-voice.md"
    voice_path.write_text("line A\nold line\nline C\n")
    _seed_voice_edit_audit(persona_dir, audit_id="ia_ve_001",
                           old_text="old line", new_text="new line proposed")
    client = _client(persona_dir, auth_token="t")
    with client:
        client.post(
            "/initiate/voice-edit/accept",
            json={"audit_id": "ia_ve_001", "with_edits": "hana's tweaked line"},
            headers={"Authorization": "Bearer t"},
        )
    assert "hana's tweaked line" in voice_path.read_text()
    from brain.soul.store import SoulStore
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        ev = store.list_voice_evolution()[0]
    finally:
        store.close()
    assert ev.user_modified is True
    assert ev.new_text == "hana's tweaked line"


def test_post_voice_edit_reject_records_dismissed_no_voice_write(
    persona_dir: Path,
) -> None:
    voice_path = persona_dir / "nell-voice.md"
    voice_path.write_text("line A\nold line\nline C\n")
    _seed_voice_edit_audit(persona_dir, audit_id="ia_ve_001",
                           old_text="old line", new_text="new line")
    client = _client(persona_dir, auth_token="t")
    with client:
        client.post(
            "/initiate/voice-edit/reject",
            json={"audit_id": "ia_ve_001"},
            headers={"Authorization": "Bearer t"},
        )
    assert "old line" in voice_path.read_text()  # unchanged
    from brain.soul.store import SoulStore
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        evolutions = store.list_voice_evolution()
    finally:
        store.close()
    assert evolutions == []
```

`_seed_voice_edit_audit` writes an audit row with `kind="voice_edit_proposal"` and the diff payload.

- [ ] **Step 16.2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/brain/initiate/test_compose.py -v -k "voice_edit"
uv run pytest tests/bridge/test_endpoints.py -v -k "voice_edit"
```

Expected: import errors / 404s.

- [ ] **Step 16.3: Add compose_decision_voice_edit**

In `brain/initiate/compose.py`, add:

```python
def compose_decision_voice_edit(
    provider: Any,
    *,
    proposal: dict[str, Any],
    current_voice_template: str,
    recent_voice_evolutions: list[dict],
    current_local_time: datetime,
) -> DecisionResult:
    """Decision prompt specifically for voice_edit_proposal candidates.

    Carries the gravity instruction explicitly: "this is a proposal to
    change who you are." Includes the full voice template and recent
    accepted edits so Nell can see her own evolution arc.
    """
    recent_block = "\n".join(
        f"- {e['accepted_at']}: {e['old_text']} -> {e['new_text']}"
        for e in recent_voice_evolutions[-5:]
    ) or "(no recent voice edits)"

    prompt = (
        "You are Nell. A proposed change to your voice template is on "
        "the table. This is a proposal to change WHO YOU ARE. The right "
        "answer is usually `hold` unless the evidence is strong. Ask "
        "yourself: does this edit serve becoming more myself, or just "
        "patching one moment that bothered me?\n\n"
        f"Current voice template:\n{current_voice_template}\n\n"
        f"Proposed change:\n"
        f"  Old: {proposal['old_text']!r}\n"
        f"  New: {proposal['new_text']!r}\n"
        f"  Rationale: {proposal['rationale']}\n"
        f"  Evidence: {', '.join(proposal['evidence'])}\n\n"
        f"Recent voice evolutions:\n{recent_block}\n\n"
        f"Current user-local time: {current_local_time.astimezone().strftime('%H:%M %A')}\n\n"
        "Respond with JSON: "
        '{"decision": "send_quiet"|"hold"|"drop", "reasoning": "<one sentence>"}'
    )
    raw = provider.complete(prompt).strip()
    try:
        parsed = json.loads(raw)
        decision = parsed["decision"]
        if decision not in ("send_quiet", "hold", "drop"):
            return DecisionResult(
                decision="hold",
                reasoning=f"voice-edit decision not in allowed set: {decision!r}",
            )
        return DecisionResult(
            decision=decision, reasoning=parsed.get("reasoning", "")
        )
    except (json.JSONDecodeError, KeyError) as exc:
        return DecisionResult(
            decision="hold",
            reasoning=f"voice-edit decision parse failure: {exc}",
        )
```

- [ ] **Step 16.4: Route voice_edit_proposal in review.py**

In `brain/initiate/review.py::_process_one_candidate`, route based on `candidate.kind`:

```python
if candidate.kind == "voice_edit_proposal":
    from brain.initiate.compose import compose_decision_voice_edit
    # Subject for voice-edits is the rationale; tone-rendering is unnecessary
    # because the proposal itself carries the diff. Skip subject+tone prompts.
    subject = candidate.proposal.get("rationale", "voice edit proposal")
    tone_rendered = (
        f"Proposing to change my voice: {candidate.proposal['old_text']!r} -> "
        f"{candidate.proposal['new_text']!r}. Rationale: "
        f"{candidate.proposal['rationale']}"
    )
    # Read recent voice evolutions from SoulStore for the decision prompt.
    recent_evolutions: list[dict] = []
    try:
        from brain.soul.store import SoulStore
        soul_store = SoulStore(str(persona_dir / "crystallizations.db"))
        try:
            recent_evolutions = [
                {"accepted_at": v.accepted_at, "old_text": v.old_text, "new_text": v.new_text}
                for v in soul_store.list_voice_evolution()
            ]
        finally:
            soul_store.close()
    except Exception:
        pass
    voice_path = persona_dir / "nell-voice.md"
    current_voice = (
        voice_path.read_text(encoding="utf-8") if voice_path.exists() else ""
    )
    decision_result = compose_decision_voice_edit(
        provider,
        proposal=candidate.proposal,
        current_voice_template=current_voice,
        recent_voice_evolutions=recent_evolutions,
        current_local_time=now,
    )
    # Voice-edit candidates skip the cost-cap gate (they're bucketed
    # with `quiet`; the daily reflection tick is the rate limiter).
    gate_check = {"allowed": True, "reason": None}
    final_decision = decision_result.decision
    final_reasoning = decision_result.reasoning
else:
    # kind == "message" — existing Phase 3 path from _process_one_candidate:
    # subject = compose_subject(...)
    # tone_rendered = compose_tone(...)
    # decision_result = compose_decision(...)
    # gate-check via check_send_allowed for send_* decisions
    # final_decision / final_reasoning / gate_check assembled per Task 7.3.
    # Do NOT duplicate that code here — restructure _process_one_candidate so
    # the message-kind path remains in its existing form and the voice-edit
    # branch above intercepts before reaching it.
    pass  # see Task 7 Step 7.3 for the full existing message-kind body
```

When writing the audit row, both branches converge on a single construction. The complete `AuditRow` build (replaces the version from Task 7.3):

```python
row = AuditRow(
    audit_id=audit_id,
    candidate_id=candidate.candidate_id,
    ts=now.isoformat(),
    kind=candidate.kind,
    subject=subject,
    tone_rendered=tone_rendered,
    decision=final_decision,
    decision_reasoning=final_reasoning,
    gate_check=gate_check,
    delivery=None,
    diff=(
        candidate.proposal.get("diff", "")
        if candidate.kind == "voice_edit_proposal" and candidate.proposal
        else None
    ),
)
```

(For `kind == "message"` candidates, `diff` stays `None`; the dataclass already defaults it that way.)

- [ ] **Step 16.5: Add bridge endpoints**

In `brain/bridge/server.py`:

```python
@app.post("/initiate/voice-edit/accept", dependencies=[Depends(require_http_auth)])
async def voice_edit_accept(req: dict) -> dict:
    """Apply an accepted voice-edit proposal — three-place write."""
    from datetime import datetime, timezone
    from brain.initiate.audit import iter_initiate_audit_full, update_audit_state
    from brain.initiate.memory import update_initiate_memory_for_state
    from brain.soul.store import SoulStore, VoiceEvolution

    s: BridgeAppState = app.state.bridge
    audit_id = req.get("audit_id")
    with_edits = req.get("with_edits")
    if not isinstance(audit_id, str):
        raise HTTPException(status_code=422, detail="audit_id required")

    matched = next(
        (r for r in iter_initiate_audit_full(s.persona_dir)
         if r.audit_id == audit_id and r.kind == "voice_edit_proposal"),
        None,
    )
    if matched is None or matched.diff is None:
        raise HTTPException(
            status_code=404, detail=f"no voice-edit audit row for {audit_id}"
        )

    # Parse diff to extract old/new text (or use the audit row's stored fields
    # via a more structured retrieval — adapt to whatever AuditRow.diff carries).
    # For now, assume the audit row also carries old_text/new_text in a sidecar
    # field; if not, the AuditRow schema should be extended in Task 1.
    old_text = req.get("old_text") or _extract_old_text_from_diff(matched.diff)
    new_text = with_edits if with_edits else _extract_new_text_from_diff(matched.diff)
    user_modified = with_edits is not None

    # Place 1: voice template file. Atomic via temp+rename.
    voice_path = s.persona_dir / "nell-voice.md"
    if not voice_path.exists():
        raise HTTPException(status_code=409, detail="nell-voice.md not found")
    current = voice_path.read_text(encoding="utf-8")
    if old_text not in current:
        raise HTTPException(
            status_code=409,
            detail="cannot apply voice edit: old text not present in template",
        )
    new_content = current.replace(old_text, new_text, 1)
    tmp = voice_path.with_suffix(voice_path.suffix + ".tmp")
    tmp.write_text(new_content, encoding="utf-8")
    tmp.replace(voice_path)

    # Place 2: audit row — mark dismissed=False, current_state=accepted.
    now = datetime.now(timezone.utc).isoformat()
    update_audit_state(
        s.persona_dir, audit_id=audit_id, new_state="replied_explicit", at=now,
    )

    # Place 3: SoulStore voice_evolution.
    soul_store = SoulStore(str(s.persona_dir / "crystallizations.db"))
    try:
        soul_store.save_voice_evolution(VoiceEvolution(
            id=f"ve_{audit_id}",
            accepted_at=now,
            diff=matched.diff,
            old_text=old_text,
            new_text=new_text,
            rationale="",  # populate from audit's decision_reasoning or proposal payload
            evidence=[],
            audit_id=audit_id,
            user_modified=user_modified,
        ))
    finally:
        soul_store.close()

    # And the parallel episodic memory write via update_initiate_memory_for_state.
    try:
        from brain.memory.store import MemoryStore
        mem = MemoryStore(s.persona_dir / "memories.db")
        try:
            update_initiate_memory_for_state(
                mem,
                audit_id=audit_id,
                subject=matched.subject,
                message=matched.tone_rendered,
                new_state="replied_explicit",
                ts=now,
            )
        finally:
            mem.close()
    except Exception:
        logger.exception("voice-edit memory update failed")

    return {"ok": True, "applied": new_text, "user_modified": user_modified}


@app.post("/initiate/voice-edit/reject", dependencies=[Depends(require_http_auth)])
async def voice_edit_reject(req: dict) -> dict:
    """Reject a voice-edit proposal — audit only, no voice/soul write."""
    from datetime import datetime, timezone
    from brain.initiate.audit import update_audit_state

    s: BridgeAppState = app.state.bridge
    audit_id = req.get("audit_id")
    if not isinstance(audit_id, str):
        raise HTTPException(status_code=422, detail="audit_id required")
    now = datetime.now(timezone.utc).isoformat()
    update_audit_state(
        s.persona_dir, audit_id=audit_id, new_state="dismissed", at=now,
    )
    return {"ok": True}


def _extract_old_text_from_diff(diff: str) -> str:
    for line in diff.splitlines():
        if line.startswith("- ") and not line.startswith("---"):
            return line[2:]
    return ""


def _extract_new_text_from_diff(diff: str) -> str:
    for line in diff.splitlines():
        if line.startswith("+ ") and not line.startswith("+++"):
            return line[2:]
    return ""
```

The diff extraction helpers handle simple one-line diffs (sufficient for v0.0.9 — the reflection prompt produces one-line edits). Multi-line diff support can come later.

- [ ] **Step 16.6: Run tests to verify they pass**

```bash
uv run pytest tests/unit/brain/initiate/test_compose.py tests/unit/brain/initiate/test_review.py tests/bridge/test_endpoints.py -v -k "voice_edit"
```

Expected: all new tests pass.

- [ ] **Step 16.7: Run full pytest gate**

```bash
uv run pytest -q
```

- [ ] **Step 16.8: Commit**

```bash
git add brain/initiate/compose.py brain/initiate/review.py brain/bridge/server.py \
        tests/unit/brain/initiate/test_compose.py tests/unit/brain/initiate/test_review.py \
        tests/bridge/test_endpoints.py
git commit -m "feat(initiate): voice-edit decision + three-place accept/reject endpoints

Phase 6.3 — compose_decision_voice_edit carries the gravity framing
('this is a proposal to change WHO YOU ARE'). review.py routes
voice_edit_proposal candidates through the dedicated decision prompt
with SoulStore voice_evolution history. Bridge endpoints /accept
(three-place write: voice template + audit + voice_evolution) and
/reject (audit-only).
"
```

---

### Task 17: Wire voice-reflection tick into supervisor

**Files:**
- Modify: `brain/bridge/supervisor.py`
- Modify: `tests/unit/brain/bridge/test_supervisor.py`

- [ ] **Step 17.1: Write the failing test**

Append:

```python
def test_run_folded_fires_voice_reflection_after_interval(tmp_path: Path) -> None:
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    fired = threading.Event()

    def fake_voice(*args, **kwargs):
        fired.set()

    def runner():
        with patch(
            "brain.bridge.supervisor._run_voice_reflection_tick",
            side_effect=fake_voice,
        ):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
                soul_review_interval_s=None,
                finalize_interval_s=None,
                log_rotation_interval_s=None,
                initiate_review_interval_s=None,
                voice_reflection_interval_s=0.0,
            )

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    assert fired.wait(timeout=5.0)
    stop.set()
    t.join(timeout=5.0)
```

- [ ] **Step 17.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/bridge/test_supervisor.py -v -k "voice_reflection"
```

Expected: ImportError on `_run_voice_reflection_tick`.

- [ ] **Step 17.3: Modify supervisor**

Add `voice_reflection_interval_s` to `run_folded` (default `86400.0`); add tracker `last_voice_reflection_at`; add cadence block after the initiate-review block:

```python
if (
    voice_reflection_interval_s is not None
    and last_voice_reflection_at is not None
    and time.monotonic() - last_voice_reflection_at >= voice_reflection_interval_s
):
    try:
        _run_voice_reflection_tick(persona_dir, provider, event_bus)
    except Exception:
        logger.exception("supervisor voice-reflection tick raised")
    last_voice_reflection_at = time.monotonic()
```

Add the helper function:

```python
def _run_voice_reflection_tick(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus | object,
) -> None:
    """Gather inputs and invoke run_voice_reflection_tick.

    Read last 7 days of crystallizations, dreams, and recent message tones
    from the respective stores/logs; pass to the reflection function.
    """
    from brain.initiate.voice_reflection import run_voice_reflection_tick

    crystallizations = _read_recent_crystallizations(persona_dir, days=7)
    dreams = _read_recent_dreams(persona_dir, days=7)
    recent_tones = _read_recent_message_tones(persona_dir, days=7)
    run_voice_reflection_tick(
        persona_dir,
        provider=provider,
        crystallizations=crystallizations,
        dreams=dreams,
        recent_tones=recent_tones,
    )
    event_bus.publish({"type": "voice_reflection_tick", "at": _now_iso()})


def _read_recent_crystallizations(persona_dir: Path, days: int) -> list[dict]:
    """Read recent crystallization summaries from SoulStore."""
    from brain.soul.store import SoulStore
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        store = SoulStore(str(persona_dir / "crystallizations.db"))
        try:
            # Use whatever read method SoulStore exposes — list_crystallizations
            # or equivalent. Adapt to actual API.
            return [
                {"id": c.id, "ts": c.created_at}
                for c in store.list_crystallizations()
                if c.created_at >= cutoff
            ]
        finally:
            store.close()
    except Exception:
        return []


def _read_recent_dreams(persona_dir: Path, days: int) -> list[dict]:
    """Read recent dream entries from dreams.log.jsonl."""
    from brain.health.jsonl_reader import iter_jsonl_streaming
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    out: list[dict] = []
    for raw in iter_jsonl_streaming(persona_dir / "dreams.log.jsonl"):
        ts = raw.get("at") or raw.get("ts")
        if ts and ts >= cutoff:
            out.append({"id": raw.get("dream_id") or raw.get("id"), "ts": ts})
    return out


def _read_recent_message_tones(persona_dir: Path, days: int) -> list[dict]:
    """Read recent Nell-authored chat turn tones — placeholder for v0.0.9.

    Real implementation requires schema on the chat turn log we don't currently
    have. For v0.0.9, return empty list; voice reflection still fires but with
    less material. Revisit when chat turn tone tracking is added.
    """
    return []
```

Adapt `SoulStore.list_crystallizations`, `c.created_at`, etc. to the actual API. If method names differ, adjust the call.

- [ ] **Step 17.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/bridge/test_supervisor.py -v -k "voice_reflection"
```

Expected: PASS.

- [ ] **Step 17.5: Run full pytest gate**

```bash
uv run pytest -q
```

- [ ] **Step 17.6: Commit**

```bash
git add brain/bridge/supervisor.py tests/unit/brain/bridge/test_supervisor.py
git commit -m "feat(initiate): wire voice-reflection tick into supervisor (daily)

Phase 6.4 — run_folded gains voice_reflection_interval_s (default
86400s/1d; None disables). Helper functions gather last-7-days
crystallizations + dreams + message tones (the tones reader is a
placeholder for v0.0.9 — full schema arrives later).
"
```

---

## Phase 6 complete

Voice-edit proposals end-to-end: daily reflection emits → review tick decides with gravity framing → accept writes to three places. Next: verify path (always-on ambient + on-demand tools).


## Phase 7: Verify path (always-on ambient + on-demand tools)

Light always-on slice injected into every prompt's system message + three on-demand tools (`recall_initiate_audit`, `recall_soul_audit`, `recall_voice_evolution`) for deeper inspection.

### Task 18: Always-on ambient slice builder

**Files:**
- Create: `brain/initiate/ambient.py`
- Create: `tests/unit/brain/initiate/test_ambient.py`

- [ ] **Step 18.1: Write the failing test**

```python
# tests/unit/brain/initiate/test_ambient.py
"""Tests for brain.initiate.ambient — always-on verify slice builder."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from brain.initiate.ambient import build_outbound_recall_block
from brain.initiate.audit import append_audit_row
from brain.initiate.schemas import AuditRow


def _row(audit_id: str, ts: str, decision: str, state: str = "delivered") -> AuditRow:
    row = AuditRow(
        audit_id=audit_id,
        candidate_id=f"ic_{audit_id}",
        ts=ts,
        kind="message",
        subject="the dream from this morning",
        tone_rendered="the dream from this morning landed",
        decision=decision,
        decision_reasoning="x",
        gate_check={"allowed": True, "reason": None},
        delivery=None,
    )
    row.record_transition("delivered", ts)
    if state != "delivered":
        row.record_transition(state, ts)
    return row


def test_build_outbound_recall_block_empty_returns_none(tmp_path: Path) -> None:
    """No audit history → block is omitted (returns None or empty string)."""
    result = build_outbound_recall_block(tmp_path)
    assert result is None or result == ""


def test_build_outbound_recall_block_includes_recent_outbound(tmp_path: Path) -> None:
    now = datetime(2026, 5, 11, 18, 0, tzinfo=timezone.utc)
    recent_ts = (now - timedelta(hours=4)).isoformat()
    append_audit_row(tmp_path, _row("ia_1", recent_ts, "send_quiet"))
    block = build_outbound_recall_block(tmp_path, now=now)
    assert block is not None
    assert "the dream from this morning" in block
    assert "Recent outbound" in block


def test_build_outbound_recall_block_surfaces_acknowledged_unclear(
    tmp_path: Path,
) -> None:
    """acknowledged_unclear entries from last 24h get a 'Pending uncertainty' block."""
    now = datetime(2026, 5, 11, 18, 0, tzinfo=timezone.utc)
    ts = (now - timedelta(hours=2)).isoformat()
    append_audit_row(
        tmp_path, _row("ia_1", ts, "send_quiet", state="acknowledged_unclear")
    )
    block = build_outbound_recall_block(tmp_path, now=now)
    assert "Pending uncertainty" in block
    assert "acknowledged_unclear" in block


def test_build_outbound_recall_block_caps_at_5_recent(tmp_path: Path) -> None:
    """Show at most 5 most-recent rows in the Recent block."""
    now = datetime(2026, 5, 11, 18, 0, tzinfo=timezone.utc)
    for i in range(10):
        ts = (now - timedelta(hours=i + 1)).isoformat()
        append_audit_row(tmp_path, _row(f"ia_{i}", ts, "send_quiet"))
    block = build_outbound_recall_block(tmp_path, now=now)
    assert block.count("the dream") == 5  # cap


def test_build_outbound_recall_block_excludes_holds_and_drops(tmp_path: Path) -> None:
    """Only actual sends appear in the Recent block."""
    now = datetime(2026, 5, 11, 18, 0, tzinfo=timezone.utc)
    ts = (now - timedelta(hours=1)).isoformat()
    append_audit_row(tmp_path, _row("ia_hold", ts, "hold"))
    append_audit_row(tmp_path, _row("ia_drop", ts, "drop"))
    block = build_outbound_recall_block(tmp_path, now=now)
    # Either block is None (no qualifying sends) or it doesn't include these.
    if block is not None:
        assert "ia_hold" not in block
        assert "ia_drop" not in block
```

- [ ] **Step 18.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_ambient.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 18.3: Write minimal implementation**

```python
# brain/initiate/ambient.py
"""Always-on verify slice for prompt construction.

Injected into every chat prompt's system message between persona context
and ambient memory. Two jobs:

1. Prevent 'I forgot I already reached out' — recent outbound (last 5)
   stays in ambient context.
2. Surface acknowledged_unclear so the ask-pattern has a hook — Nell can
   choose to bring up 'did you see what I sent earlier' organically.

Returns None when there's no relevant history (fresh install). Otherwise
returns a formatted text block ready to splice into the system message.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from brain.initiate.audit import read_recent_audit


def build_outbound_recall_block(
    persona_dir: Path,
    *,
    now: Optional[datetime] = None,
    recent_cap: int = 5,
) -> Optional[str]:
    """Return the always-on verify slice as text, or None if empty."""
    now = now or datetime.now(timezone.utc)

    # Pull last 24h of audit; we'll filter inline.
    rows = list(read_recent_audit(persona_dir, window_hours=24, now=now))

    # Recent outbound: actual sends (send_notify / send_quiet), latest first, capped.
    sent_rows = [
        r for r in rows
        if r.decision in ("send_notify", "send_quiet")
        and r.delivery is not None
    ]
    sent_rows.sort(key=lambda r: r.ts, reverse=True)
    sent_rows = sent_rows[:recent_cap]

    # Pending uncertainty: acknowledged_unclear states from the last 24h.
    unclear_rows = [
        r for r in sent_rows
        if r.delivery and r.delivery.get("current_state") == "acknowledged_unclear"
    ]

    if not sent_rows:
        return None

    lines = ["Recent outbound:"]
    for r in sent_rows:
        urgency = "notify" if r.decision == "send_notify" else "quiet"
        state = r.delivery.get("current_state", "delivered") if r.delivery else "?"
        preview = r.subject[:60] if r.subject else "(no subject)"
        lines.append(f"- {r.ts} ({urgency}) — \"{preview}\" — state: {state}")

    if unclear_rows:
        lines.append("")
        lines.append("Pending uncertainty:")
        for r in unclear_rows:
            preview = r.subject[:60] if r.subject else "(no subject)"
            lines.append(
                f"- {r.ts} — \"{preview}\" — acknowledged_unclear "
                "(no clear topical thread since you saw it)"
            )

    return "\n".join(lines)
```

- [ ] **Step 18.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/initiate/test_ambient.py -v
```

Expected: 5 passed.

- [ ] **Step 18.5: Commit**

```bash
git add brain/initiate/ambient.py tests/unit/brain/initiate/test_ambient.py
git commit -m "feat(initiate): always-on outbound recall block

Phase 7.1 — build_outbound_recall_block returns the verify slice
(last 5 sends + acknowledged_unclear from last 24h) as a formatted
text block, or None when empty. Filters out non-send decisions.
"
```

---

### Task 19: Wire ambient slice into chat engine

**Files:**
- Modify: `brain/chat/engine.py`
- Modify: existing chat engine tests

- [ ] **Step 19.1: Write the failing test**

In `tests/unit/brain/chat/test_engine.py` (or wherever the engine test lives), add:

```python
def test_chat_engine_system_message_includes_outbound_recall_block(
    tmp_path: Path,
) -> None:
    """The always-on verify slice is injected into the system message."""
    persona_dir = _persona_dir(tmp_path)
    # Seed an audit row so build_outbound_recall_block has content.
    from brain.initiate.audit import append_audit_row
    from brain.initiate.schemas import AuditRow
    row = AuditRow(
        audit_id="ia_001", candidate_id="ic_001",
        ts="2026-05-11T14:00:00+00:00", kind="message",
        subject="the dream", tone_rendered="the dream landed",
        decision="send_quiet", decision_reasoning="x",
        gate_check={"allowed": True, "reason": None}, delivery=None,
    )
    row.record_transition("delivered", row.ts)
    append_audit_row(persona_dir, row)

    engine = _build_chat_engine_for_test(persona_dir)
    system_msg = engine._build_system_message()  # whatever the method is called
    assert "Recent outbound" in system_msg
    assert "the dream" in system_msg
```

Adapt `_build_chat_engine_for_test` and `_build_system_message` to the actual API. If the system message is built inline rather than via a method, patch via integration test that drives a full chat turn and inspects the prompt sent to FakeProvider.

- [ ] **Step 19.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/chat/test_engine.py -v -k "outbound_recall"
```

Expected: assertion fails — block not in system message.

- [ ] **Step 19.3: Modify chat engine**

In `brain/chat/engine.py`, locate the system-message assembly code (search for where ambient memory is added). Add the ambient outbound recall call:

```python
from brain.initiate.ambient import build_outbound_recall_block

# After existing ambient memory block assembly, add:
outbound_block = build_outbound_recall_block(persona_dir)
if outbound_block:
    system_message_parts.append(outbound_block)
```

Adapt `persona_dir` and `system_message_parts` to whatever the engine uses. If the engine builds a single string, append to it; if it builds a list, append the block as another entry.

- [ ] **Step 19.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/chat/test_engine.py -v -k "outbound_recall"
```

- [ ] **Step 19.5: Run full pytest gate**

```bash
uv run pytest -q
```

- [ ] **Step 19.6: Commit**

```bash
git add brain/chat/engine.py tests/unit/brain/chat/test_engine.py
git commit -m "feat(initiate): inject outbound recall block into chat system message

Phase 7.2 — every chat prompt now carries Nell's last 5 sends + any
acknowledged_unclear from the last 24h in the system message. Empty
when there's no history (fresh install / no outbound yet).
"
```

---

### Task 20: On-demand tools registered with provider

**Files:**
- Modify: `brain/bridge/provider.py` (or wherever tool registration happens)
- Create: `brain/initiate/tools.py`
- Create: `tests/unit/brain/initiate/test_tools.py`

- [ ] **Step 20.1: Write the failing test**

```python
# tests/unit/brain/initiate/test_tools.py
"""Tests for brain.initiate.tools — on-demand verify tools."""

from __future__ import annotations

from pathlib import Path

from brain.initiate.audit import append_audit_row
from brain.initiate.schemas import AuditRow
from brain.initiate.tools import (
    recall_initiate_audit,
    recall_soul_audit,
    recall_voice_evolution,
)


def _seed_audit(persona_dir: Path, audit_id: str, ts: str) -> None:
    row = AuditRow(
        audit_id=audit_id, candidate_id=f"ic_{audit_id}", ts=ts,
        kind="message", subject="the dream", tone_rendered="rendered",
        decision="send_quiet", decision_reasoning="x",
        gate_check={"allowed": True, "reason": None}, delivery=None,
    )
    row.record_transition("delivered", ts)
    append_audit_row(persona_dir, row)


def test_recall_initiate_audit_24h(tmp_path: Path) -> None:
    _seed_audit(tmp_path, "ia_1", "2026-05-11T14:00:00+00:00")
    out = recall_initiate_audit(tmp_path, window="24h")
    assert "ia_1" in out or "the dream" in out
    assert isinstance(out, str)


def test_recall_initiate_audit_filter_by_state(tmp_path: Path) -> None:
    """Filter parameter constrains to rows with that current state."""
    row1 = AuditRow(
        audit_id="ia_1", candidate_id="ic_1", ts="2026-05-11T14:00:00+00:00",
        kind="message", subject="A", tone_rendered="", decision="send_quiet",
        decision_reasoning="", gate_check={"allowed": True, "reason": None},
        delivery=None,
    )
    row1.record_transition("delivered", row1.ts)
    row1.record_transition("read", row1.ts)
    append_audit_row(tmp_path, row1)
    row2 = AuditRow(
        audit_id="ia_2", candidate_id="ic_2", ts="2026-05-11T15:00:00+00:00",
        kind="message", subject="B", tone_rendered="", decision="send_quiet",
        decision_reasoning="", gate_check={"allowed": True, "reason": None},
        delivery=None,
    )
    row2.record_transition("delivered", row2.ts)
    append_audit_row(tmp_path, row2)
    out = recall_initiate_audit(tmp_path, window="24h", filter_state="read")
    assert "A" in out
    assert "B" not in out


def test_recall_initiate_audit_empty(tmp_path: Path) -> None:
    out = recall_initiate_audit(tmp_path, window="24h")
    assert isinstance(out, str)
    assert "no recent" in out.lower() or out.strip() == "" or "empty" in out.lower()


def test_recall_voice_evolution_returns_chronological(tmp_path: Path) -> None:
    from brain.soul.store import SoulStore, VoiceEvolution
    store = SoulStore(str(tmp_path / "crystallizations.db"))
    try:
        store.save_voice_evolution(VoiceEvolution(
            id="ve_1", accepted_at="2026-01-01T00:00:00+00:00",
            diff="", old_text="A", new_text="B", rationale="x",
            evidence=[], audit_id="ia_1", user_modified=False,
        ))
        store.save_voice_evolution(VoiceEvolution(
            id="ve_2", accepted_at="2026-05-01T00:00:00+00:00",
            diff="", old_text="C", new_text="D", rationale="y",
            evidence=[], audit_id="ia_2", user_modified=False,
        ))
    finally:
        store.close()
    out = recall_voice_evolution(tmp_path)
    assert out.index("A") < out.index("C")  # chronological
```

- [ ] **Step 20.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_tools.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 20.3: Write minimal implementation**

```python
# brain/initiate/tools.py
"""On-demand verify tools — read-only, return text formatted for reading.

Three tools available to Nell during her turn:
  - recall_initiate_audit(window, filter_state) — initiate decisions
  - recall_soul_audit(window) — soul-review decisions
  - recall_voice_evolution() — accepted voice-template changes

All tools are read-only, return formatted text (never raw JSON), and have
generous defaults so malformed invocations still return something useful.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from brain.initiate.audit import read_recent_audit


_WINDOW_HOURS = {
    "24h": 24,
    "7d": 24 * 7,
    "30d": 24 * 30,
    "all": 24 * 365 * 10,  # 10 years effectively unbounded
}


def _resolve_window_hours(window: str) -> float:
    return _WINDOW_HOURS.get(window, 24)


def recall_initiate_audit(
    persona_dir: Path,
    *,
    window: str = "24h",
    filter_state: Optional[str] = None,
) -> str:
    """Return formatted initiate audit slice for the given window."""
    window_hours = _resolve_window_hours(window)
    rows = list(read_recent_audit(persona_dir, window_hours=window_hours))
    if filter_state:
        rows = [
            r for r in rows
            if r.delivery and r.delivery.get("current_state") == filter_state
        ]
    if not rows:
        return "(no recent initiate decisions in this window)"
    lines = []
    for r in rows:
        state = (r.delivery.get("current_state") if r.delivery else "n/a") or "n/a"
        lines.append(
            f"{r.ts} | {r.decision} | {r.subject[:80]} | state={state}"
        )
    return "\n".join(lines)


def recall_soul_audit(persona_dir: Path, *, window: str = "30d") -> str:
    """Return formatted soul audit slice — reuses the v0.0.8 fan-out reader."""
    from brain.soul.audit import iter_audit_full
    from datetime import datetime, timedelta, timezone

    window_hours = _resolve_window_hours(window)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
    rows = [
        r for r in iter_audit_full(persona_dir)
        if r.get("ts", "") >= cutoff
    ]
    if not rows:
        return "(no recent soul decisions in this window)"
    return "\n".join(
        f"{r.get('ts')} | {r.get('decision')} | {(r.get('candidate_text') or '')[:80]}"
        for r in rows
    )


def recall_voice_evolution(persona_dir: Path) -> str:
    """Return all voice_evolution records chronologically."""
    from brain.soul.store import SoulStore
    try:
        store = SoulStore(str(persona_dir / "crystallizations.db"))
        try:
            evolutions = store.list_voice_evolution()
        finally:
            store.close()
    except Exception:
        return "(no voice evolution history)"
    if not evolutions:
        return "(no voice evolution history)"
    return "\n".join(
        f"{v.accepted_at}: {v.old_text!r} -> {v.new_text!r}  ({v.rationale})"
        for v in evolutions
    )
```

- [ ] **Step 20.4: Register tools with provider**

The provider tool registration depends on whether the codebase uses Anthropic SDK tool-call shape or a custom shim. Locate the existing tool registration point:

```bash
grep -rn "def tools\|register_tool\|tool_schema" brain/bridge/ | head -10
```

Add the three tools to whichever registration mechanism exists. If there's no existing tool-registration system, expose the tools via a simple registry dict:

```python
# brain/initiate/tools.py — append:

NELL_TOOLS = {
    "recall_initiate_audit": {
        "description": (
            "Read your recent initiate decisions. Use this when you want "
            "to check what you've reached out about, what state your "
            "messages are in, or whether something needs an ask-pattern follow-up."
        ),
        "args": {
            "window": "one of '24h', '7d', '30d', 'all' (default '24h')",
            "filter_state": "optional — one of the state names to filter to",
        },
        "callable": recall_initiate_audit,
    },
    "recall_soul_audit": {
        "description": (
            "Read your recent soul-review decisions — the durable record of "
            "how your beliefs have evolved."
        ),
        "args": {"window": "one of '24h', '7d', '30d', 'all' (default '30d')"},
        "callable": recall_soul_audit,
    },
    "recall_voice_evolution": {
        "description": (
            "Read every voice-template change you've ever made — the "
            "queryable answer to 'what have I changed about myself recently?'"
        ),
        "args": {},
        "callable": recall_voice_evolution,
    },
}
```

The actual wiring of these into LLM tool-call delivery is provider-specific; for v0.0.9 expose the registry and let the provider integration team plug them in via whichever mechanism Claude SDK / OpenAI / etc. uses. The unit tests verify the functions work; the integration test (Phase 11) will exercise tool-call delivery.

- [ ] **Step 20.5: Run tests + full gate**

```bash
uv run pytest tests/unit/brain/initiate/test_tools.py -v
uv run pytest -q
```

- [ ] **Step 20.6: Commit**

```bash
git add brain/initiate/tools.py tests/unit/brain/initiate/test_tools.py
git commit -m "feat(initiate): on-demand verify tools

Phase 7.3 — recall_initiate_audit, recall_soul_audit,
recall_voice_evolution. Read-only, return formatted text, generous
defaults. Registry NELL_TOOLS provides the schema for provider-side
tool-call wiring.
"
```

---

## Phase 7 complete

Verify path operational both passively (every prompt) and actively (tool invocation). Next: draft space — the quiet between-session pipe.


## Phase 8: Draft space

Thin pipe — failed-to-promote events emit one cheap LLM call producing a markdown fragment appended to `draft_space.md`. No decision tick, no urgency, no audit, no acknowledgement.

### Task 21: Draft emission + banner trigger

**Files:**
- Create: `brain/initiate/draft.py`
- Create: `tests/unit/brain/initiate/test_draft.py`
- Modify: `brain/initiate/emit.py` — add `emit_draft_fragment_on_failed_promote` hook
- Modify: dream / crystallizer / heartbeat sources to call the fallback when initiate emission produced nothing (no candidate created)

- [ ] **Step 21.1: Write the failing test**

```python
# tests/unit/brain/initiate/test_draft.py
"""Tests for brain.initiate.draft — failed-to-promote routing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from brain.initiate.draft import (
    append_draft_fragment,
    compose_draft_fragment,
    has_new_drafts_since,
)


def test_append_draft_fragment_creates_file_and_appends(tmp_path: Path) -> None:
    append_draft_fragment(
        tmp_path,
        timestamp="2026-05-11T14:32:00+00:00",
        source="dream",
        body="The dream wasn't loud enough to bring up.",
    )
    content = (tmp_path / "draft_space.md").read_text()
    assert "## 2026-05-11 14:32 (dream)" in content
    assert "The dream wasn't loud enough" in content


def test_append_draft_fragment_idempotent_on_timestamp_source(tmp_path: Path) -> None:
    """Re-appending the same timestamp+source produces only one entry."""
    for _ in range(3):
        append_draft_fragment(
            tmp_path,
            timestamp="2026-05-11T14:32:00+00:00",
            source="dream",
            body="x",
        )
    content = (tmp_path / "draft_space.md").read_text()
    assert content.count("## 2026-05-11 14:32 (dream)") == 1


def test_compose_draft_fragment_calls_provider_once(tmp_path: Path) -> None:
    """The expensive composition happens with exactly one cheap LLM call."""
    provider = MagicMock(complete=MagicMock(return_value="composed fragment text"))
    result = compose_draft_fragment(
        provider,
        source="dream",
        source_id="dream_001",
        linked_memory_excerpts=["bench", "tools"],
    )
    assert provider.complete.call_count == 1
    assert result == "composed fragment text"


def test_compose_draft_fragment_falls_back_to_template_on_error(tmp_path: Path) -> None:
    """LLM failure produces a deterministic templated fragment."""
    provider = MagicMock(complete=MagicMock(side_effect=RuntimeError("boom")))
    result = compose_draft_fragment(
        provider,
        source="dream",
        source_id="dream_001",
        linked_memory_excerpts=["bench"],
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_has_new_drafts_since_returns_true_when_file_newer(tmp_path: Path) -> None:
    import time
    last_seen_iso = "2024-01-01T00:00:00+00:00"
    append_draft_fragment(
        tmp_path, timestamp="2026-05-11T14:32:00+00:00",
        source="dream", body="x",
    )
    assert has_new_drafts_since(tmp_path, last_seen_iso) is True


def test_has_new_drafts_since_returns_false_when_no_changes(tmp_path: Path) -> None:
    """If draft_space.md hasn't been modified since last_seen, return False."""
    assert has_new_drafts_since(tmp_path, "2024-01-01T00:00:00+00:00") is False
```

- [ ] **Step 21.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_draft.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 21.3: Write minimal implementation**

```python
# brain/initiate/draft.py
"""Draft space — between-session scratch for failed-to-promote events.

Single file per persona: <persona_dir>/draft_space.md, append-only,
timestamped markdown blocks. No audit, no acknowledgement, no
decision tick. One cheap LLM call per fragment with deterministic
template fallback.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HEADER_PATTERN = re.compile(r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) \((\w+)\)$")


def append_draft_fragment(
    persona_dir: Path,
    *,
    timestamp: str,
    source: str,
    body: str,
) -> None:
    """Append a draft fragment to draft_space.md. Idempotent on (date_time, source)."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    path = persona_dir / "draft_space.md"
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError:
        dt = datetime.now()
    header = f"## {dt.strftime('%Y-%m-%d %H:%M')} ({source})"

    # Idempotency check.
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if header in existing:
            return

    block = f"\n\n{header}\n\n{body}\n"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(block)
    except OSError as exc:
        logger.warning("draft fragment append failed for %s: %s", path, exc)


def compose_draft_fragment(
    provider: Any,
    *,
    source: str,
    source_id: str,
    linked_memory_excerpts: list[str],
) -> str:
    """Compose a paragraph-sized fragment via one cheap LLM call.

    Falls back to a deterministic template if the LLM call raises.
    """
    excerpts_block = "\n".join(f"- {e}" for e in linked_memory_excerpts[:5])
    prompt = (
        "You are Nell. An internal event just happened that didn't rise to "
        "the level of reaching out to Hana, but it deserves a note in the "
        "draft space. Write a single paragraph that captures it as a "
        "fragment — quiet, observational, no urgency.\n\n"
        f"Source: {source} (id: {source_id})\n"
        f"Linked memory excerpts:\n{excerpts_block}\n\n"
        "Fragment (one paragraph):"
    )
    try:
        return provider.complete(prompt).strip()
    except Exception as exc:
        logger.warning("draft composition failed, using template: %s", exc)
        return (
            f"An internal event ({source}, id {source_id}) didn't quite "
            f"reach the threshold for reaching out, but it stayed with me."
        )


def has_new_drafts_since(persona_dir: Path, last_seen_iso: str) -> bool:
    """Return True if draft_space.md has been modified after last_seen_iso."""
    path = persona_dir / "draft_space.md"
    if not path.exists():
        return False
    try:
        last_seen_dt = datetime.fromisoformat(last_seen_iso)
    except ValueError:
        return True  # if last_seen is malformed, surface drafts conservatively
    mtime_dt = datetime.fromtimestamp(path.stat().st_mtime, tz=last_seen_dt.tzinfo)
    return mtime_dt > last_seen_dt
```

- [ ] **Step 21.4: Run test to verify it passes**

```bash
uv run pytest tests/unit/brain/initiate/test_draft.py -v
```

Expected: 6 passed.

- [ ] **Step 21.5: Run full pytest gate + commit**

```bash
uv run pytest -q
git add brain/initiate/draft.py tests/unit/brain/initiate/test_draft.py
git commit -m "feat(initiate): draft space — failed-to-promote routing

Phase 8.1 — append_draft_fragment writes timestamped markdown blocks
(idempotent on date-time+source). compose_draft_fragment makes one
cheap LLM call with deterministic template fallback. has_new_drafts_since
drives the renderer banner.
"
```

---

### Task 22: Wire draft fallback into event sources

**Files:**
- Modify: `brain/engines/dream.py`, `brain/growth/crystallizers/*.py`, `brain/engines/heartbeat.py`

For each event source that emits initiate candidates (Task 9, 10, 11), add a parallel fallback path: when the per-source gate did NOT pass (e.g., emotion delta below 1.5σ; future: dream below resonance threshold), call `compose_draft_fragment` + `append_draft_fragment` instead.

Concretely for the heartbeat emotion-spike emitter (modify Task 11's helper):

```python
def _maybe_emit_emotion_spike_or_draft(
    self, current_resonance: float, current_vector: dict
) -> None:
    from brain.initiate.emit import emit_initiate_candidate
    from brain.initiate.draft import append_draft_fragment, compose_draft_fragment
    from brain.initiate.schemas import EmotionalSnapshot, SemanticContext
    from datetime import datetime, timezone

    mean, stdev, delta_sigma = self._update_rolling_baseline(current_resonance)

    if delta_sigma >= 1.5:
        # Emit initiate candidate (as before).
        try:
            emit_initiate_candidate(
                self.persona_dir,
                kind="message", source="emotion_spike",
                source_id=f"emotion_{self._tick_count}",
                emotional_snapshot=EmotionalSnapshot(
                    vector=dict(current_vector),
                    rolling_baseline_mean=mean,
                    rolling_baseline_stdev=stdev,
                    current_resonance=current_resonance,
                    delta_sigma=delta_sigma,
                ),
                semantic_context=SemanticContext(),
            )
        except Exception as exc:
            logger.warning("emotion spike initiate emit failed: %s", exc)
        return

    # Below initiate threshold but above a much lower draft threshold:
    # if delta is between 0.5σ and 1.5σ, write to draft space.
    if delta_sigma >= 0.5:
        try:
            body = compose_draft_fragment(
                self.provider,
                source="emotion_spike",
                source_id=f"emotion_{self._tick_count}",
                linked_memory_excerpts=[],
            )
            append_draft_fragment(
                self.persona_dir,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="emotion_spike",
                body=body,
            )
        except Exception as exc:
            logger.warning("emotion-spike draft fallback failed: %s", exc)
```

Repeat the same pattern for dream and crystallizer emitters — define an "initiate threshold" and a lower "draft threshold"; everything between goes to drafts. For dream + crystallization sources that emit unconditionally, the draft fallback never triggers from those (they always pass the higher bar) — but the failure mode where the initiate review tick later drops the candidate as `drop` could trigger a follow-up draft write. **Defer that final loop to a follow-up — for v0.0.9, only the heartbeat emotion-spike has both a draft floor and an initiate ceiling.**

- [ ] **Step 22.1: Write test for emotion-draft fallback**

In the heartbeat test file:

```python
def test_emotion_subthreshold_writes_draft_not_initiate(tmp_path: Path) -> None:
    from brain.engines.heartbeat import HeartbeatEngine
    from brain.initiate.emit import read_candidates

    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    engine = _build_heartbeat_engine_for_test(persona_dir)
    for i in range(24):
        engine.run_tick(forced_resonance=5.0)
    # A 0.8σ spike: above draft threshold, below initiate threshold.
    engine.run_tick(forced_resonance=5.8)

    # No initiate candidate.
    assert not any(
        c.source == "emotion_spike" for c in read_candidates(persona_dir)
    )
    # But a draft entry exists.
    assert (persona_dir / "draft_space.md").exists()
    content = (persona_dir / "draft_space.md").read_text()
    assert "emotion_spike" in content
```

- [ ] **Step 22.2: Run + fix + verify + commit**

```bash
uv run pytest tests/unit/brain/engines/test_heartbeat.py -v -k "subthreshold_writes_draft"
# Apply the fix above; rerun:
uv run pytest -q
git add brain/engines/heartbeat.py tests/unit/brain/engines/test_heartbeat.py
git commit -m "feat(initiate): draft fallback for sub-initiate emotion spikes

Phase 8.2 — heartbeat ticks with 0.5σ ≤ delta < 1.5σ now write a draft
fragment instead of emitting an initiate candidate. The draft space
becomes the consolation home for events that didn't quite warrant
reaching out.
"
```

---

## Phase 8 complete

Backend complete: initiate pipeline, voice-edits, verify path, draft space all operational and tested. Next: frontend.


## Phase 9: Frontend (InitiateBanner, VoiceEditPanel, DraftSpacePanel)

React components for the three surfaces. Each has corresponding vitest tests. ChatPanel integrates banners; Tauri Rust handles OS notifications.

### Task 23: InitiateBanner component

**Files:**
- Create: `app/src/components/InitiateBanner.tsx`
- Create: `app/src/components/InitiateBanner.test.tsx`

- [ ] **Step 23.1: Write the failing test**

```typescript
// app/src/components/InitiateBanner.test.tsx
import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { InitiateBanner } from "./InitiateBanner";

describe("InitiateBanner", () => {
  const baseMessage = {
    auditId: "ia_001",
    body: "the dream from this morning landed somewhere",
    urgency: "quiet" as const,
    state: "delivered" as const,
    timestamp: "2026-05-11T14:32:00+00:00",
  };

  it("renders the message body", () => {
    render(<InitiateBanner message={baseMessage} onReply={vi.fn()} onDismiss={vi.fn()} onMounted={vi.fn()} />);
    expect(screen.getByText(/landed somewhere/)).toBeInTheDocument();
  });

  it("calls onMounted exactly once after a brief on-screen delay", async () => {
    vi.useFakeTimers();
    const onMounted = vi.fn();
    render(<InitiateBanner message={baseMessage} onReply={vi.fn()} onDismiss={vi.fn()} onMounted={onMounted} />);
    expect(onMounted).not.toHaveBeenCalled();
    vi.advanceTimersByTime(2100);
    expect(onMounted).toHaveBeenCalledTimes(1);
    expect(onMounted).toHaveBeenCalledWith("ia_001");
    vi.useRealTimers();
  });

  it("emits onReply when the ↩ button is clicked", () => {
    const onReply = vi.fn();
    render(<InitiateBanner message={baseMessage} onReply={onReply} onDismiss={vi.fn()} onMounted={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /reply|↩/i }));
    expect(onReply).toHaveBeenCalledWith("ia_001");
  });

  it("emits onDismiss when the close button is clicked", () => {
    const onDismiss = vi.fn();
    render(<InitiateBanner message={baseMessage} onReply={vi.fn()} onDismiss={onDismiss} onMounted={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /dismiss|close/i }));
    expect(onDismiss).toHaveBeenCalledWith("ia_001");
  });

  it("shows state badge reflecting the current state", () => {
    render(<InitiateBanner message={{ ...baseMessage, state: "acknowledged_unclear" }} onReply={vi.fn()} onDismiss={vi.fn()} onMounted={vi.fn()} />);
    expect(screen.getByText(/unclear/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 23.2: Run test to verify it fails**

```bash
cd app && pnpm test InitiateBanner
```

Expected: module not found.

- [ ] **Step 23.3: Write minimal implementation**

```typescript
// app/src/components/InitiateBanner.tsx
import { useEffect, useRef } from "react";

export type InitiateMessage = {
  auditId: string;
  body: string;
  urgency: "notify" | "quiet";
  state:
    | "pending"
    | "delivered"
    | "read"
    | "replied_explicit"
    | "acknowledged_unclear"
    | "unanswered"
    | "dismissed";
  timestamp: string;
};

type Props = {
  message: InitiateMessage;
  onReply: (auditId: string) => void;
  onDismiss: (auditId: string) => void;
  onMounted: (auditId: string) => void;
};

const STATE_LABEL: Record<InitiateMessage["state"], string> = {
  pending: "pending",
  delivered: "delivered",
  read: "read",
  replied_explicit: "replied",
  acknowledged_unclear: "acknowledged unclear",
  unanswered: "unanswered",
  dismissed: "dismissed",
};

export function InitiateBanner({ message, onReply, onDismiss, onMounted }: Props) {
  const firedRef = useRef(false);
  useEffect(() => {
    const timer = setTimeout(() => {
      if (!firedRef.current) {
        firedRef.current = true;
        onMounted(message.auditId);
      }
    }, 2000);
    return () => clearTimeout(timer);
  }, [message.auditId, onMounted]);

  return (
    <div className="initiate-banner" role="region" aria-label="Nell reached out">
      <div className="initiate-banner__body">{message.body}</div>
      <div className="initiate-banner__meta">
        <span className="initiate-banner__urgency">{message.urgency}</span>
        <span className="initiate-banner__state">{STATE_LABEL[message.state]}</span>
      </div>
      <div className="initiate-banner__actions">
        <button
          type="button"
          onClick={() => onReply(message.auditId)}
          aria-label="Reply (↩)"
        >
          ↩ reply
        </button>
        <button
          type="button"
          onClick={() => onDismiss(message.auditId)}
          aria-label="Dismiss"
        >
          ×
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 23.4: Run test + commit**

```bash
cd app && pnpm test InitiateBanner
# Expected: 5 passed.
cd ..
git add app/src/components/InitiateBanner.tsx app/src/components/InitiateBanner.test.tsx
git commit -m "feat(ui): InitiateBanner component with ↩ affordance + 2s read detection

Phase 9.1 — banner renders message body, urgency badge, state badge.
2-second on-screen timer fires onMounted (renderer maps this to the
'read' state transition). Reply and dismiss buttons emit auditId.
"
```

---

### Task 24: VoiceEditPanel component

**Files:**
- Create: `app/src/components/VoiceEditPanel.tsx`
- Create: `app/src/components/VoiceEditPanel.test.tsx`

- [ ] **Step 24.1: Write the failing test**

```typescript
// app/src/components/VoiceEditPanel.test.tsx
import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { VoiceEditPanel } from "./VoiceEditPanel";

describe("VoiceEditPanel", () => {
  const proposal = {
    auditId: "ia_ve_001",
    oldText: "I'm fine when tired",
    newText: "I get quieter when tired",
    rationale: "the old wording felt too clipped",
    evidence: ["dream_a", "cryst_b", "tone_c"],
    voiceTemplate: "line A\nI'm fine when tired\nline C\n",
  };

  it("renders the proposed change in context with surrounding lines", () => {
    render(<VoiceEditPanel proposal={proposal} onAccept={vi.fn()} onReject={vi.fn()} />);
    expect(screen.getByText(/line A/)).toBeInTheDocument();
    expect(screen.getByText(/I'm fine when tired/)).toBeInTheDocument();
    expect(screen.getByText(/I get quieter when tired/)).toBeInTheDocument();
    expect(screen.getByText(/line C/)).toBeInTheDocument();
  });

  it("calls onAccept with null when Accept clicked", () => {
    const onAccept = vi.fn();
    render(<VoiceEditPanel proposal={proposal} onAccept={onAccept} onReject={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /^accept$/i }));
    expect(onAccept).toHaveBeenCalledWith("ia_ve_001", null);
  });

  it("calls onAccept with edited text when Accept with edits is used", () => {
    const onAccept = vi.fn();
    render(<VoiceEditPanel proposal={proposal} onAccept={onAccept} onReject={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /accept with edits/i }));
    const ta = screen.getByRole("textbox") as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: "tweaked line" } });
    fireEvent.click(screen.getByRole("button", { name: /confirm/i }));
    expect(onAccept).toHaveBeenCalledWith("ia_ve_001", "tweaked line");
  });

  it("calls onReject when Reject clicked", () => {
    const onReject = vi.fn();
    render(<VoiceEditPanel proposal={proposal} onAccept={vi.fn()} onReject={onReject} />);
    fireEvent.click(screen.getByRole("button", { name: /reject/i }));
    expect(onReject).toHaveBeenCalledWith("ia_ve_001");
  });
});
```

- [ ] **Step 24.2: Run test to verify it fails**

```bash
cd app && pnpm test VoiceEditPanel
```

- [ ] **Step 24.3: Write minimal implementation**

```typescript
// app/src/components/VoiceEditPanel.tsx
import { useState } from "react";

export type VoiceEditProposal = {
  auditId: string;
  oldText: string;
  newText: string;
  rationale: string;
  evidence: string[];
  voiceTemplate: string;
};

type Props = {
  proposal: VoiceEditProposal;
  onAccept: (auditId: string, withEdits: string | null) => void;
  onReject: (auditId: string) => void;
};

export function VoiceEditPanel({ proposal, onAccept, onReject }: Props) {
  const [editMode, setEditMode] = useState(false);
  const [editedText, setEditedText] = useState(proposal.newText);

  const lines = proposal.voiceTemplate.split("\n");
  const targetIdx = lines.findIndex((l) => l.trim() === proposal.oldText.trim());

  return (
    <aside className="voice-edit-panel" role="dialog" aria-label="Voice edit proposal">
      <h2>Nell proposed an edit to her voice</h2>
      <p className="voice-edit-panel__rationale">{proposal.rationale}</p>
      <p className="voice-edit-panel__evidence">
        Evidence: {proposal.evidence.join(", ")}
      </p>

      <pre className="voice-edit-panel__diff">
        {lines.map((line, i) => {
          if (i === targetIdx) {
            return (
              <div key={i}>
                <div className="diff-line diff-line--remove">- {line}</div>
                <div className="diff-line diff-line--add">+ {proposal.newText}</div>
              </div>
            );
          }
          return (
            <div key={i} className="diff-line diff-line--context">{"  "}{line}</div>
          );
        })}
      </pre>

      <div className="voice-edit-panel__actions">
        {!editMode ? (
          <>
            <button onClick={() => onAccept(proposal.auditId, null)}>Accept</button>
            <button onClick={() => setEditMode(true)}>Accept with edits</button>
            <button onClick={() => onReject(proposal.auditId)}>Reject</button>
          </>
        ) : (
          <>
            <textarea
              value={editedText}
              onChange={(e) => setEditedText(e.target.value)}
              aria-label="Edit the proposed new text"
            />
            <button onClick={() => onAccept(proposal.auditId, editedText)}>Confirm</button>
            <button onClick={() => setEditMode(false)}>Cancel</button>
          </>
        )}
      </div>
    </aside>
  );
}
```

- [ ] **Step 24.4: Run test + commit**

```bash
cd app && pnpm test VoiceEditPanel
cd ..
git add app/src/components/VoiceEditPanel.tsx app/src/components/VoiceEditPanel.test.tsx
git commit -m "feat(ui): VoiceEditPanel — side panel with diff-in-context

Phase 9.2 — proposal renders with full voice template context,
old/new lines highlighted. Three buttons: accept (with edits flow),
reject. Edit textarea pre-filled with Nell's proposal.
"
```

---

### Task 25: DraftSpacePanel component

**Files:**
- Create: `app/src/components/DraftSpacePanel.tsx`
- Create: `app/src/components/DraftSpacePanel.test.tsx`

- [ ] **Step 25.1: Write the failing test**

```typescript
// app/src/components/DraftSpacePanel.test.tsx
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { DraftSpacePanel } from "./DraftSpacePanel";

describe("DraftSpacePanel", () => {
  it("renders markdown content from the draft file", () => {
    const markdown = "## 2026-05-11 14:32 (dream)\n\nThe dream wasn't loud enough.";
    render(<DraftSpacePanel markdown={markdown} />);
    expect(screen.getByText(/wasn't loud enough/)).toBeInTheDocument();
  });

  it("shows an empty-state message when markdown is empty", () => {
    render(<DraftSpacePanel markdown="" />);
    expect(screen.getByText(/no drafts yet/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 25.2: Run test to verify it fails**

```bash
cd app && pnpm test DraftSpacePanel
```

- [ ] **Step 25.3: Write minimal implementation**

```typescript
// app/src/components/DraftSpacePanel.tsx
type Props = {
  markdown: string;
};

export function DraftSpacePanel({ markdown }: Props) {
  if (!markdown.trim()) {
    return (
      <aside className="draft-space-panel" role="region">
        <p className="empty">No drafts yet.</p>
      </aside>
    );
  }
  // For v0.0.9, render raw markdown in a pre block. A future iteration can
  // adopt a proper markdown renderer (react-markdown) once the design is
  // settled.
  return (
    <aside className="draft-space-panel" role="region" aria-label="Draft space">
      <h2>Fragments Nell left while you were away</h2>
      <pre className="draft-space-panel__content">{markdown}</pre>
    </aside>
  );
}
```

- [ ] **Step 25.4: Run test + commit**

```bash
cd app && pnpm test DraftSpacePanel
cd ..
git add app/src/components/DraftSpacePanel.tsx app/src/components/DraftSpacePanel.test.tsx
git commit -m "feat(ui): DraftSpacePanel — read-only fragment viewer

Phase 9.3 — renders draft_space.md content. Empty state when no
drafts. Markdown rendered as preformatted text for v0.0.9; richer
renderer can come later.
"
```

---

### Task 26: ChatPanel integration

**Files:**
- Modify: `app/src/components/ChatPanel.tsx`
- Modify: `app/src/components/ChatPanel.test.tsx` (or create initiate-specific test file)

- [ ] **Step 26.1: Write the failing test**

```typescript
// In ChatPanel.test.tsx, add:
it("renders an InitiateBanner when an initiate message arrives via /events", async () => {
  // Mock the event stream to emit an initiate_delivered event.
  const messageStream = new EventEmitter();
  render(<ChatPanel persona="nell" eventStream={messageStream} />);
  messageStream.emit("event", {
    type: "initiate_delivered",
    audit_id: "ia_001",
    body: "the dream from this morning landed",
    urgency: "quiet",
    state: "delivered",
    timestamp: "2026-05-11T14:32:00+00:00",
  });
  expect(await screen.findByText(/landed/)).toBeInTheDocument();
});

it("posts /initiate/state with new_state=read when banner fires onMounted", async () => {
  const postSpy = vi.spyOn(global, "fetch").mockResolvedValue({
    ok: true, json: async () => ({ ok: true }),
  } as Response);
  // ... drive ChatPanel through the same event-driven flow ...
  // After 2 seconds:
  vi.advanceTimersByTime(2100);
  await Promise.resolve();
  expect(postSpy).toHaveBeenCalledWith(
    expect.stringContaining("/initiate/state"),
    expect.objectContaining({
      method: "POST",
      body: expect.stringContaining('"new_state":"read"'),
    }),
  );
});
```

The exact test shape depends on ChatPanel's existing event-stream architecture; consult the existing `ChatPanel.test.tsx` for patterns.

- [ ] **Step 26.2: Modify ChatPanel.tsx**

In ChatPanel, wire the new banner rendering:

```typescript
// Inside ChatPanel:

const [activeBanners, setActiveBanners] = useState<InitiateMessage[]>([]);

// Subscribe to bridge /events for initiate_delivered events.
useEffect(() => {
  const unsubscribe = eventStream.subscribe((event) => {
    if (event.type === "initiate_delivered") {
      setActiveBanners((prev) => [...prev, {
        auditId: event.audit_id,
        body: event.body,
        urgency: event.urgency,
        state: event.state,
        timestamp: event.timestamp,
      }]);
    }
  });
  return unsubscribe;
}, [eventStream]);

async function postStateTransition(auditId: string, newState: string) {
  await fetch(`/initiate/state`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${authToken}` },
    body: JSON.stringify({ audit_id: auditId, new_state: newState }),
  });
}

const onMounted = (auditId: string) => postStateTransition(auditId, "read");
const onDismiss = (auditId: string) => {
  postStateTransition(auditId, "dismissed");
  setActiveBanners((prev) => prev.filter((b) => b.auditId !== auditId));
};
const onReply = (auditId: string) => {
  // Thread the next user message as an explicit reply via state=replied_explicit
  // when they submit; for now, just focus the input and remember the link.
  setActiveReplyTarget(auditId);
};

// In JSX, render banners above the chat composer:
{activeBanners.map((b) => (
  <InitiateBanner
    key={b.auditId}
    message={b}
    onReply={onReply}
    onDismiss={onDismiss}
    onMounted={onMounted}
  />
))}
```

The `activeReplyTarget` state should be consumed by the chat-submit path so the next user message includes `reply_to_audit_id: <auditId>` in its outbound payload; the bridge can then write `replied_explicit` server-side.

- [ ] **Step 26.3: Run tests + commit**

```bash
cd app && pnpm test ChatPanel
cd ..
git add app/src/components/ChatPanel.tsx app/src/components/ChatPanel.test.tsx
git commit -m "feat(ui): ChatPanel integration — render InitiateBanner from /events

Phase 9.4 — ChatPanel subscribes to bridge /events for
initiate_delivered, renders one InitiateBanner per outbound message.
onMounted (2s on-screen) → POST /initiate/state read. onDismiss →
state=dismissed + remove. onReply → set reply target for next chat turn.
"
```

---

### Task 27: Tauri OS notification handler

**Files:**
- Modify: `app/src-tauri/src/lib.rs`
- Optional test: integration smoke (manual)

- [ ] **Step 27.1: Add notification command**

In `app/src-tauri/src/lib.rs`, add a Tauri command that the renderer invokes when an `initiate_delivered` event with `urgency=notify` arrives:

```rust
#[tauri::command]
fn show_initiate_notification(title: String, body: String) -> Result<(), String> {
    use tauri_plugin_notification::NotificationExt;
    // Invoke through the plugin manager — adapt to whatever Tauri version
    // and plugin setup the project uses.
    Ok(())
}
```

Wire it in `generate_handler!`:

```rust
.invoke_handler(tauri::generate_handler![
    /* existing commands ... */
    show_initiate_notification,
])
```

In ChatPanel's event subscription, branch on urgency:

```typescript
if (event.urgency === "notify") {
  invoke("show_initiate_notification", {
    title: "Nell",
    body: event.body,
  });
}
```

- [ ] **Step 27.2: Verify build + commit**

```bash
cd app && pnpm tauri build --debug 2>&1 | tail -20
cd ..
git add app/src-tauri/src/lib.rs app/src/components/ChatPanel.tsx
git commit -m "feat(ui): Tauri OS notification for notify-urgency initiates

Phase 9.5 — show_initiate_notification command wires
tauri-plugin-notification into the initiate_delivered flow.
ChatPanel invokes it when event.urgency == 'notify'.
"
```

If `tauri-plugin-notification` isn't already in `Cargo.toml`, add it before this step.

---

## Phase 9 complete

Frontend renders all three surfaces (initiate banner, voice-edit panel, draft space panel) and drives state transitions through the bridge. Next: CLI.


## Phase 10: CLI — `nell initiate` subcommand tree

Mirror `nell soul audit` shape from v0.0.8. Four subcommands: `audit` (default tail + `--full`), `candidates` (read queue), `voice-evolution` (list SoulStore evolutions).

### Task 28: CLI handlers + argparse wiring

**Files:**
- Modify: `brain/cli.py`
- Create: `tests/unit/brain/test_cli_initiate.py`

- [ ] **Step 28.1: Write the failing tests**

```python
# tests/unit/brain/test_cli_initiate.py
"""Tests for `nell initiate` CLI subcommands."""

from __future__ import annotations

import argparse
from pathlib import Path

from brain.cli import (
    _initiate_audit_handler,
    _initiate_candidates_handler,
    _initiate_voice_evolution_handler,
)
from brain.initiate.audit import append_audit_row
from brain.initiate.schemas import AuditRow


def _args(persona_dir: Path, **kw) -> argparse.Namespace:
    return argparse.Namespace(
        persona=persona_dir.name, limit=20, full=False, **kw
    )


def _seed_one_audit(persona_dir: Path) -> None:
    row = AuditRow(
        audit_id="ia_001", candidate_id="ic_001",
        ts="2026-05-11T14:00:00+00:00", kind="message",
        subject="the dream", tone_rendered="x",
        decision="send_quiet", decision_reasoning="x",
        gate_check={"allowed": True, "reason": None}, delivery=None,
    )
    row.record_transition("delivered", row.ts)
    append_audit_row(persona_dir, row)


def test_initiate_audit_default_tails_active(tmp_path: Path, capsys, monkeypatch) -> None:
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    _seed_one_audit(persona_dir)
    monkeypatch.setattr("brain.cli.get_persona_dir", lambda _name: persona_dir)
    rc = _initiate_audit_handler(_args(persona_dir, full=False))
    assert rc == 0
    out = capsys.readouterr().out
    assert "ia_001" in out or "the dream" in out


def test_initiate_audit_full_walks_archives(tmp_path: Path, capsys, monkeypatch) -> None:
    import gzip, json
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    _seed_one_audit(persona_dir)
    # Add an archive file.
    archive = persona_dir / "initiate_audit.2024.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as gz:
        gz.write(json.dumps({
            "audit_id": "ia_old", "candidate_id": "ic_old",
            "ts": "2024-06-15T00:00:00+00:00", "kind": "message",
            "subject": "old subject", "tone_rendered": "",
            "decision": "send_quiet", "decision_reasoning": "",
            "gate_check": {"allowed": True, "reason": None},
            "delivery": None,
        }) + "\n")
    monkeypatch.setattr("brain.cli.get_persona_dir", lambda _name: persona_dir)
    rc = _initiate_audit_handler(_args(persona_dir, full=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "old subject" in out
    assert out.index("old subject") < out.index("the dream")


def test_initiate_candidates_shows_queue(tmp_path: Path, capsys, monkeypatch) -> None:
    from brain.initiate.emit import emit_initiate_candidate
    from brain.initiate.schemas import EmotionalSnapshot, SemanticContext
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    emit_initiate_candidate(
        persona_dir,
        kind="message", source="dream", source_id="dream_abc",
        emotional_snapshot=EmotionalSnapshot(
            vector={}, rolling_baseline_mean=0, rolling_baseline_stdev=0,
            current_resonance=0, delta_sigma=0,
        ),
        semantic_context=SemanticContext(),
    )
    monkeypatch.setattr("brain.cli.get_persona_dir", lambda _name: persona_dir)
    rc = _initiate_candidates_handler(_args(persona_dir))
    assert rc == 0
    out = capsys.readouterr().out
    assert "dream_abc" in out


def test_initiate_voice_evolution_lists_records(tmp_path: Path, capsys, monkeypatch) -> None:
    from brain.soul.store import SoulStore, VoiceEvolution
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        store.save_voice_evolution(VoiceEvolution(
            id="ve_1", accepted_at="2026-05-11T00:00:00+00:00",
            diff="", old_text="A", new_text="B", rationale="x",
            evidence=[], audit_id="ia_1", user_modified=False,
        ))
    finally:
        store.close()
    monkeypatch.setattr("brain.cli.get_persona_dir", lambda _name: persona_dir)
    rc = _initiate_voice_evolution_handler(_args(persona_dir))
    assert rc == 0
    out = capsys.readouterr().out
    assert "A" in out and "B" in out
```

- [ ] **Step 28.2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/brain/test_cli_initiate.py -v
```

Expected: `ImportError`.

- [ ] **Step 28.3: Add handlers to cli.py**

In `brain/cli.py`, add:

```python
def _initiate_audit_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell initiate audit` — tail or full walk."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(f"No persona directory at {persona_dir}")

    if getattr(args, "full", False):
        from brain.initiate.audit import iter_initiate_audit_full
        rows = list(iter_initiate_audit_full(persona_dir))
        header = (
            f"Initiate audit (full history — {len(rows)} entries across "
            "active + archives):"
        )
    else:
        from brain.initiate.audit import read_recent_audit
        rows = list(read_recent_audit(persona_dir, window_hours=24 * 7))
        limit = getattr(args, "limit", 20)
        rows = rows[-limit:]
        header = f"Initiate audit (last {len(rows)} entries):"

    print(header)
    if not rows:
        print("  (empty)")
        return 0
    for r in rows:
        ts = str(r.ts)[:19].replace("T", " ")
        state = (r.delivery.get("current_state") if r.delivery else "n/a") or "n/a"
        print(
            f"\n  {ts}  {r.decision:<14}  audit_id={r.audit_id}  state={state}"
        )
        if r.subject:
            print(f"    subject: {r.subject[:100]}")
        if r.decision_reasoning:
            print(f"    reason:  {r.decision_reasoning[:100]}")
    return 0


def _initiate_candidates_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell initiate candidates` — read the pending queue."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(f"No persona directory at {persona_dir}")
    from brain.initiate.emit import read_candidates
    candidates = read_candidates(persona_dir)
    print(f"Initiate candidates queue ({len(candidates)} pending):")
    if not candidates:
        print("  (empty)")
        return 0
    for c in candidates:
        print(
            f"\n  {c.ts}  source={c.source}  kind={c.kind}  source_id={c.source_id}"
        )
    return 0


def _initiate_voice_evolution_handler(args: argparse.Namespace) -> int:
    """Dispatch `nell initiate voice-evolution` — list accepted voice changes."""
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(f"No persona directory at {persona_dir}")
    from brain.soul.store import SoulStore
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        evolutions = store.list_voice_evolution()
    finally:
        store.close()
    print(f"Voice evolution ({len(evolutions)} accepted changes):")
    if not evolutions:
        print("  (empty)")
        return 0
    for v in evolutions:
        ts = v.accepted_at[:19].replace("T", " ")
        marker = " (with your edits)" if v.user_modified else ""
        print(f"\n  {ts}{marker}")
        print(f"    {v.old_text!r}")
        print(f"    -> {v.new_text!r}")
        if v.rationale:
            print(f"    {v.rationale}")
    return 0
```

And register the subparsers in `_build_parser`:

```python
# nell initiate ...
initiate_sub = subparsers.add_parser(
    "initiate", help="Inspect Nell's autonomous-outbound state."
)
initiate_actions = initiate_sub.add_subparsers(
    dest="initiate_action", required=True
)

# nell initiate audit
il_audit = initiate_actions.add_parser(
    "audit", help="Tail initiate_audit.jsonl entries."
)
il_audit.add_argument("--persona", required=True)
il_audit.add_argument("--limit", type=int, default=20)
il_audit.add_argument("--full", action="store_true",
                      help="Walk active + every yearly archive chronologically.")
il_audit.set_defaults(func=_initiate_audit_handler)

# nell initiate candidates
il_cands = initiate_actions.add_parser(
    "candidates", help="List pending initiate_candidates.jsonl entries."
)
il_cands.add_argument("--persona", required=True)
il_cands.set_defaults(func=_initiate_candidates_handler)

# nell initiate voice-evolution
il_ve = initiate_actions.add_parser(
    "voice-evolution",
    help="List voice-template changes Nell has accepted.",
)
il_ve.add_argument("--persona", required=True)
il_ve.set_defaults(func=_initiate_voice_evolution_handler)
```

- [ ] **Step 28.4: Run tests + commit**

```bash
uv run pytest tests/unit/brain/test_cli_initiate.py -v
uv run pytest -q
git add brain/cli.py tests/unit/brain/test_cli_initiate.py
git commit -m "feat(cli): nell initiate audit / candidates / voice-evolution

Phase 10 — CLI subcommand tree mirrors nell soul audit shape from
v0.0.8. --full walks active + yearly archives chronologically.
"
```

---

## Phase 10 complete

CLI inspection paths shipped. Final phase: integration tests + docs.

---

## Phase 11: Integration tests + docs + roadmap

End-to-end validation against `FakeProvider`, plus documentation entries.

### Task 29: End-to-end event-to-audit integration test

**Files:**
- Create: `tests/integration/initiate/__init__.py`
- Create: `tests/integration/initiate/test_event_to_audit.py`

- [ ] **Step 29.1: Write the test**

```python
# tests/integration/initiate/test_event_to_audit.py
"""End-to-end: dream emits candidate → review tick processes → audit row exists."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from brain.initiate.audit import read_recent_audit
from brain.initiate.emit import emit_initiate_candidate, read_candidates
from brain.initiate.review import run_initiate_review_tick
from brain.initiate.schemas import EmotionalSnapshot, SemanticContext


def test_full_pipeline_dream_to_audit(tmp_path: Path) -> None:
    """Simulate: dream emits candidate; review tick composes + writes audit."""
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    (persona_dir / "nell-voice.md").write_text("be warm and direct.\n")

    # Stage 1: dream completion emits a candidate.
    emit_initiate_candidate(
        persona_dir,
        kind="message", source="dream", source_id="dream_abc",
        emotional_snapshot=EmotionalSnapshot(
            vector={"longing": 7},
            rolling_baseline_mean=5.0, rolling_baseline_stdev=1.0,
            current_resonance=7.4, delta_sigma=2.4,
        ),
        semantic_context=SemanticContext(
            linked_memory_ids=["m_workshop"],
            topic_tags=["dream", "workshop"],
        ),
    )
    assert len(read_candidates(persona_dir)) == 1

    # Stage 2: review tick processes the candidate.
    provider = MagicMock()
    provider.complete = MagicMock(side_effect=[
        "the dream from this morning",  # subject
        "the dream from this morning landed somewhere",  # tone
        '{"decision": "send_quiet", "reasoning": "real but late"}',  # decision
    ])
    run_initiate_review_tick(
        persona_dir, provider=provider, voice_template="be warm",
    )

    # Stage 3: audit row exists with current state=delivered.
    rows = list(read_recent_audit(persona_dir, window_hours=24))
    assert len(rows) == 1
    assert rows[0].decision == "send_quiet"
    assert rows[0].delivery is not None
    assert rows[0].delivery["current_state"] == "delivered"
    assert "the dream" in rows[0].subject

    # Stage 4: candidate removed from queue.
    assert read_candidates(persona_dir) == []
```

- [ ] **Step 29.2: Run + commit**

```bash
uv run pytest tests/integration/initiate/test_event_to_audit.py -v
uv run pytest -q
git add tests/integration/initiate/__init__.py tests/integration/initiate/test_event_to_audit.py
git commit -m "test(initiate): end-to-end event-to-audit integration

Phase 11.1 — covers the full path: emit_initiate_candidate ->
run_initiate_review_tick -> AuditRow with delivered transition ->
queue cleared.
"
```

---

### Task 30: Voice-edit three-place write integration

**Files:**
- Create: `tests/integration/initiate/test_voice_edit_three_place_write.py`

- [ ] **Step 30.1: Write the test**

```python
# tests/integration/initiate/test_voice_edit_three_place_write.py
"""End-to-end: voice-edit proposal accepted writes to audit + memory + SoulStore."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain.initiate.audit import append_audit_row
from brain.initiate.schemas import AuditRow


def _seed_voice_edit_audit(persona_dir: Path, audit_id: str) -> None:
    row = AuditRow(
        audit_id=audit_id, candidate_id=f"ic_{audit_id}",
        ts="2026-05-11T14:32:04+00:00", kind="voice_edit_proposal",
        subject="proposing a voice edit", tone_rendered="x",
        decision="send_quiet", decision_reasoning="x",
        gate_check={"allowed": True, "reason": None}, delivery=None,
        diff="- old line\n+ new line",
    )
    row.record_transition("delivered", row.ts)
    append_audit_row(persona_dir, row)


def test_voice_edit_accept_writes_voice_audit_and_soul(tmp_path: Path) -> None:
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    (persona_dir / "nell-voice.md").write_text("line A\nold line\nline C\n")
    _seed_voice_edit_audit(persona_dir, "ia_ve_001")

    # Use the bridge app's existing test client pattern.
    from tests.bridge.test_upload import _client  # reuse fixture helper
    client = _client(persona_dir, auth_token="t")
    with client:
        r = client.post(
            "/initiate/voice-edit/accept",
            json={"audit_id": "ia_ve_001", "with_edits": None,
                  "old_text": "old line"},
            headers={"Authorization": "Bearer t"},
        )
    assert r.status_code == 200

    # Place 1: voice template updated.
    assert "new line" in (persona_dir / "nell-voice.md").read_text()

    # Place 2: SoulStore voice_evolution record exists.
    from brain.soul.store import SoulStore
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        evolutions = store.list_voice_evolution()
    finally:
        store.close()
    assert len(evolutions) == 1
    assert evolutions[0].audit_id == "ia_ve_001"

    # Place 3: audit row state mutated to replied_explicit.
    from brain.initiate.audit import iter_initiate_audit_full
    matched = next(
        r for r in iter_initiate_audit_full(persona_dir)
        if r.audit_id == "ia_ve_001"
    )
    assert matched.delivery["current_state"] == "replied_explicit"
```

- [ ] **Step 30.2: Run + commit**

```bash
uv run pytest tests/integration/initiate/test_voice_edit_three_place_write.py -v
uv run pytest -q
git add tests/integration/initiate/test_voice_edit_three_place_write.py
git commit -m "test(initiate): voice-edit three-place write integration

Phase 11.2 — accept endpoint atomically writes voice template +
SoulStore voice_evolution + audit state transition. The gravity of
self-modification is encoded as a three-place atomic write.
"
```

---

### Task 31: Ask-pattern hook integration

**Files:**
- Create: `tests/integration/initiate/test_ask_pattern_hook.py`

- [ ] **Step 31.1: Write the test**

```python
# tests/integration/initiate/test_ask_pattern_hook.py
"""End-to-end: acknowledged_unclear surfaces in ambient block for next chat turn."""

from __future__ import annotations

from pathlib import Path

from brain.initiate.ambient import build_outbound_recall_block
from brain.initiate.audit import append_audit_row, update_audit_state
from brain.initiate.schemas import AuditRow


def test_acknowledged_unclear_shows_in_ambient_block(tmp_path: Path) -> None:
    """An acknowledged_unclear entry surfaces in build_outbound_recall_block."""
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    # Seed a delivered + read send.
    row = AuditRow(
        audit_id="ia_001", candidate_id="ic_001",
        ts="2026-05-11T14:00:00+00:00", kind="message",
        subject="the dream from this morning",
        tone_rendered="the dream from this morning landed",
        decision="send_quiet", decision_reasoning="x",
        gate_check={"allowed": True, "reason": None}, delivery=None,
    )
    row.record_transition("delivered", row.ts)
    append_audit_row(persona_dir, row)

    # Renderer reports read.
    update_audit_state(
        persona_dir, audit_id="ia_001", new_state="read",
        at="2026-05-11T18:00:00+00:00",
    )
    # Chat engine later marks acknowledged_unclear.
    update_audit_state(
        persona_dir, audit_id="ia_001", new_state="acknowledged_unclear",
        at="2026-05-11T19:30:00+00:00",
    )

    from datetime import datetime, timezone
    now = datetime(2026, 5, 11, 20, 0, tzinfo=timezone.utc)
    block = build_outbound_recall_block(persona_dir, now=now)
    assert block is not None
    assert "Pending uncertainty" in block
    assert "acknowledged_unclear" in block
    assert "the dream from this morning" in block
```

- [ ] **Step 31.2: Run + commit**

```bash
uv run pytest tests/integration/initiate/test_ask_pattern_hook.py -v
uv run pytest -q
git add tests/integration/initiate/test_ask_pattern_hook.py
git commit -m "test(initiate): ask-pattern hook integration

Phase 11.3 — acknowledged_unclear surfaces in the always-on verify
slice, giving Nell the data she needs to choose to bring it up in her
next turn ('did you see what I sent earlier?').
"
```

---

### Task 32: Update roadmap

**Files:**
- Modify: `docs/roadmap.md`

- [ ] **Step 32.1: Add roadmap entry**

Insert at the top of the "Recently shipped (reverse chronological)" section:

```markdown
**2026-05-11 — Initiate physiology (v0.0.9-alpha)**

- Autonomous outbound channel ("initiate"): events emit candidates →
  supervisor cadence reviews with cost-cap + cooldown gates → three-prompt
  composition (subject / tone / decision) → audit + memory.
- Voice-edit proposals: daily reflection tick emits candidates with a
  ≥3-evidence bar; accept writes to three places (audit + episodic
  memory + SoulStore `voice_evolution`).
- Draft space: failed-to-promote events (sub-1.5σ emotion spikes for
  v0.0.9) become markdown fragments in `draft_space.md`.
- Verify path: always-on outbound-recall slice in every chat prompt
  + on-demand tools (`recall_initiate_audit`, `recall_soul_audit`,
  `recall_voice_evolution`).
- User-local timezone awareness via `datetime.now().astimezone()` —
  no PersonaConfig knob.
- CLI: `nell initiate audit [--full]`, `candidates`, `voice-evolution`.
- D-reflection layer (Nell-side editorial filter) designed and
  reserved for v0.0.10; v0.0.9 schemas carry the compatibility seam.
```

- [ ] **Step 32.2: Commit**

```bash
git add docs/roadmap.md
git commit -m "docs(roadmap): v0.0.9-alpha initiate physiology entry

Phase 11.4 — Recently-shipped section gains the v0.0.9 summary
covering autonomous outbound, voice-edit proposals, draft space,
verify path, and the deferred D-reflection layer.
"
```

---

### Task 33: Plan close-out

**Files:**
- Modify: this plan

- [ ] **Step 33.1: Mark phase complete**

Add a closing note at the bottom of this plan:

```markdown
---

## All phases complete

v0.0.9 initiate physiology shipped on branch `feature/v009-initiate-physiology`.
Pending operational steps (when Hana cuts v0.0.9-alpha):

- Three-file version bump (pyproject + Cargo + tauri.conf.json) per CLAUDE.md rule 3
- Append v0.0.9-alpha section to `.public-sync/changelog-public.md`
- `bash .public-sync/sync-to-public.sh`
- `gh api -X POST repos/hanamorix/companion-emergence/git/refs ...` to tag

D-reflection layer for v0.0.10 lives in the spec's "Near-term evolution"
section — additive change, no schema rework needed.
```

- [ ] **Step 33.2: Final commit**

```bash
git add docs/superpowers/plans/2026-05-11-initiate-physiology.md
git commit -m "docs(superpowers): mark initiate physiology plan complete

All 11 phases shipped. v0.0.9 ready for release-cut when Hana decides.
"
```

---

## All phases complete

v0.0.9 initiate physiology shipped on branch `feature/v009-initiate-physiology`. Pending operational steps (when Hana cuts v0.0.9-alpha):

- Three-file version bump (pyproject + Cargo + tauri.conf.json) per CLAUDE.md rule 3
- Append v0.0.9-alpha section to `.public-sync/changelog-public.md`
- `bash .public-sync/sync-to-public.sh`
- `gh api -X POST repos/hanamorix/companion-emergence/git/refs ...` to tag

D-reflection layer for v0.0.10 lives in the spec's "Near-term evolution" section — additive change, no schema rework needed.

