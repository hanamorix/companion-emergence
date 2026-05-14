# Initiate D-Reflection + New Event Sources Implementation Plan (v0.0.10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Bundle B for the initiate physiology — D-reflection (editorial layer between candidate emission and composition) and two new event sources (`reflex_firing`, `research_completion`).

**Architecture:** D-reflection is a new module `brain/initiate/reflection.py` called once per non-empty heartbeat tick from `_run_initiate_review_tick`. Filtered candidates demote to `draft_space.md` via the v0.0.9 draft writer; promoted candidates flow to the existing three-prompt composition pipeline. New emitters live in `brain/initiate/new_sources.py` and are invoked from the source engines (`brain/engines/reflex.py`, `brain/engines/research.py`) at the firing/completion sites. D bypasses the v0.0.9 daily cost cap; Haiku→Sonnet tiered escalation on low-confidence/parse-fail.

**Tech Stack:** Python (uv-managed), pytest (TDD per project rule), anthropic SDK 0.42+, existing brain/initiate/* substrate from v0.0.9.

**Spec:** `docs/superpowers/specs/2026-05-12-initiate-d-reflection-design.md`

**Branch convention:** All work on `feature/v010-d-reflection` branched from `main` after v0.0.9 merge. If v0.0.9 has not yet been merged, branch from `feature/v009-initiate-physiology` and rebase post-merge.

**Strict gate:** After EVERY commit, run `uv run pytest -q` (full suite) and confirm green. Per `feedback_verify_each_step_before_proceeding.md`. If the suite goes red at any point, fix before proceeding to the next task.

**Deferred from this plan:** `recall_resonance` source — see spec §"Deferred to v0.0.11+". Do not build clustering substrate as part of this plan.

---

## Implementer notes — soft spots flagged for verification-first

Three places in this plan name v0.0.9-internal identifiers I didn't fully verify against the live code. Before writing any test or implementation in the affected tasks, **run the verification command and confirm the actual names match what the task text claims.** If they differ, use the actual names — do not blindly paste the snippets.

### Soft spot 1 — Task 15 (`review.py` wiring): persona-state field names and compose function

Before starting Task 15, run:

```bash
grep -n "companion_name\|user_name\|persona_state\|compose_and_dispatch\|compose_and_send\|def _run_initiate_review_tick" brain/initiate/review.py brain/bridge/persona_state.py
```

Confirm:
- The variable names that hold the companion's and user's names (the task text uses `companion_name` and `user_name`, but v0.0.9 may use `persona_name` / `user_first_name` / etc.).
- The composition-pipeline function the existing tick calls when handing off a candidate. The task text uses `compose_and_dispatch` as a stand-in; the real function is whatever the v0.0.9 review tick already invokes.
- The voice template path the existing code reads (look for `voice_templates/` references).

If the v0.0.9 names differ, substitute them throughout Task 15's snippets when implementing. Do not introduce new identifiers if v0.0.9 already has equivalent ones.

### Soft spot 2 — Task 17 (`brain/engines/reflex.py` hook): ReflexEngine API

Before starting Task 17, run:

```bash
grep -n "^class \|^def \|def record_\|def _fire\|def run_tick" brain/engines/reflex.py | head -30
```

Confirm:
- The engine class name (task text uses `ReflexEngine` — may be different).
- The method where a firing event is recorded/written (task text uses `record_firing` — almost certainly different in v0.0.9; look for the actual firing site).
- Whether `_is_rest_state()` (or equivalent) is callable from the firing-site scope. If not, the meta-gate's rest-state check can be skipped at this callsite for v0.0.10 (note left in task text).
- The structure of the existing "firing record" object — the new code adapts that object to satisfy the `ReflexFiringLike` Protocol. If the existing record doesn't have a `triggered_by_companion_outbound` field, decide whether to add it (preferred, small change) or to compute it inline (acceptable fallback — compare the firing's trigger source against recent outbound audit rows).

### Soft spot 3 — Task 18 (`brain/engines/research.py` hook): ResearchEngine API + topic-overlap computation

Before starting Task 18, run:

```bash
grep -n "^class \|^def \|maturity\|complete\|matured\|run_tick" brain/engines/research.py | head -30
```

Confirm:
- The engine class name (task text uses `ResearchEngine`).
- The thread-maturity-close site (where a thread transitions to "completed/matured").
- Whether the engine has access to recent conversation embeddings for the `topic_overlap_score` computation. The task text says to factor this into a helper `_compute_topic_overlap_score`. If the embeddings aren't reachable from research.py, either:
  - Add a parameter to the engine's run_tick that passes embeddings in from the supervisor (preferred); OR
  - Hard-code `topic_overlap_score = 1.0` for v0.0.10 with a follow-up to compute it properly (the gate then effectively becomes "any matured thread that passed the other gates emits" — acceptable starting point, will inflate research_completion volume).

Choose the path that requires the smallest cross-module change. Document the choice in the commit message.

---

These three verifications take ~5 minutes total and prevent paste-blindly errors. They are gating: **do not start the affected task without running the relevant grep first.**

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `brain/initiate/schemas.py` | MODIFY | Add `reflex_firing`/`research_completion` to `CandidateSource`; add new `Decision` values; add `source_meta` field to `SemanticContext`. |
| `brain/initiate/audit.py` | MODIFY | Add `append_d_call_row` + `read_recent_d_calls` for `initiate_d_calls.jsonl`. |
| `brain/initiate/d_call_schema.py` | CREATE | `DCallRow` dataclass for the per-tick D telemetry table. |
| `brain/initiate/gate_thresholds.json` | CREATE | Tunable thresholds for new emitter gates (operator file). |
| `brain/initiate/new_sources.py` | CREATE | `gate_reflex_firing`, `gate_research_completion`, `emit_*_candidate`, shared meta-gates, `gate_rejections.jsonl` writer. |
| `brain/initiate/reflection.py` | CREATE | D-reflection: prompt assembly, Haiku→Sonnet call, failure-mode dispatch, draft-space demote routing. |
| `brain/initiate/review.py` | MODIFY | Insert `reflection.run(...)` call between candidate-fetch and composition handoff in `_run_initiate_review_tick`. |
| `brain/engines/reflex.py` | MODIFY | Call `emit_reflex_firing_candidate` at the firing site (post-gate). |
| `brain/engines/research.py` | MODIFY | Call `emit_research_completion_candidate` at the thread-maturity-close site. |
| `brain/cli.py` | MODIFY | Add `nell initiate d-stats` subcommand. |
| `tests/unit/brain/initiate/test_new_sources.py` | CREATE | Per-emitter gate + emit tests. |
| `tests/unit/brain/initiate/test_reflection.py` | CREATE | D-reflection unit tests (happy + failure branches). |
| `tests/unit/brain/initiate/test_audit_d_calls.py` | CREATE | DCallRow writer/reader tests. |
| `tests/integration/initiate/test_d_reflection_e2e.py` | CREATE | End-to-end: events → emit → queue → D-tick → composition or demote. |
| `tests/unit/brain/cli/test_d_stats.py` | CREATE | `nell initiate d-stats` CLI tests. |

---

## Phase 1 — Schema foundation

### Task 1: Extend `CandidateSource` and `Decision` literals + add `source_meta`

**Files:**
- Modify: `brain/initiate/schemas.py`
- Test: `tests/unit/brain/initiate/test_schemas.py` (extend existing)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/brain/initiate/test_schemas.py`:

```python
def test_candidate_source_includes_new_sources():
    from brain.initiate.schemas import CandidateSource
    # CandidateSource is a Literal — validate by trying to assign each value.
    valid: CandidateSource
    valid = "reflex_firing"
    assert valid == "reflex_firing"
    valid = "research_completion"
    assert valid == "research_completion"


def test_decision_includes_new_d_values():
    from brain.initiate.schemas import Decision
    for v in [
        "promoted_by_d",
        "filtered_pre_compose_low_confidence",
        "filtered_d_budget",
        "promoted_by_d_malformed_fallback",
        "d_passthrough_retry",
        "promoted_by_d_after_3_failures",
    ]:
        d: Decision = v  # type: ignore[assignment]
        assert d == v


def test_semantic_context_source_meta_field():
    from brain.initiate.schemas import SemanticContext
    sc = SemanticContext(
        linked_memory_ids=["m1"],
        topic_tags=["x"],
        source_meta={"pattern_id": "p1", "confidence": 0.8},
    )
    rt = SemanticContext.from_dict(sc.to_dict())
    assert rt.source_meta == {"pattern_id": "p1", "confidence": 0.8}


def test_semantic_context_source_meta_optional_back_compat():
    from brain.initiate.schemas import SemanticContext
    sc = SemanticContext.from_dict({"linked_memory_ids": [], "topic_tags": []})
    assert sc.source_meta is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/brain/initiate/test_schemas.py -k "new_sources or new_d_values or source_meta" -v
```

Expected: 4 failures. The literal tests will type-error at static-check time but pass at runtime (Literals are runtime strings) — however, the `source_meta` tests will fail with `TypeError: SemanticContext.__init__() got an unexpected keyword argument 'source_meta'` and the back-compat one will fail similarly when reading.

If the literal tests pass at runtime even before the impl, that's OK — they're documentation. What MUST fail is the `source_meta` tests.

- [ ] **Step 3: Modify `schemas.py`**

In `brain/initiate/schemas.py`, change:

```python
CandidateSource = Literal["dream", "crystallization", "emotion_spike", "voice_reflection"]
```

to:

```python
CandidateSource = Literal[
    "dream",
    "crystallization",
    "emotion_spike",
    "voice_reflection",
    "reflex_firing",
    "research_completion",
]
```

Change:

```python
Decision = Literal[
    "send_notify", "send_quiet", "hold", "drop", "error", "filtered_pre_compose"
]
```

to:

```python
Decision = Literal[
    "send_notify",
    "send_quiet",
    "hold",
    "drop",
    "error",
    "filtered_pre_compose",
    # D-reflection (v0.0.10) decision values:
    "promoted_by_d",
    "filtered_pre_compose_low_confidence",
    "filtered_d_budget",
    "promoted_by_d_malformed_fallback",
    "d_passthrough_retry",
    "promoted_by_d_after_3_failures",
]
```

Change the `SemanticContext` dataclass:

```python
@dataclass
class SemanticContext:
    linked_memory_ids: list[str] = field(default_factory=list)
    topic_tags: list[str] = field(default_factory=list)
    source_meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "linked_memory_ids": self.linked_memory_ids,
            "topic_tags": self.topic_tags,
        }
        if self.source_meta is not None:
            d["source_meta"] = self.source_meta
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SemanticContext:
        return cls(
            linked_memory_ids=d.get("linked_memory_ids", []),
            topic_tags=d.get("topic_tags", []),
            source_meta=d.get("source_meta"),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/brain/initiate/test_schemas.py -v
uv run ruff check brain/initiate/schemas.py
```

Expected: all green, ruff clean.

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

Expected: all tests pass (no regressions on the existing 1859+).

- [ ] **Step 6: Commit**

```bash
git add brain/initiate/schemas.py tests/unit/brain/initiate/test_schemas.py
git commit -m "feat(initiate): extend schemas for v0.0.10 D-reflection + new sources"
```

---

### Task 2: Create `DCallRow` dataclass

**Files:**
- Create: `brain/initiate/d_call_schema.py`
- Test: `tests/unit/brain/initiate/test_audit_d_calls.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/brain/initiate/test_audit_d_calls.py`:

```python
"""Tests for the initiate_d_calls audit table."""
from __future__ import annotations

import json

from brain.initiate.d_call_schema import DCallRow, make_d_call_id


def test_d_call_row_roundtrip():
    row = DCallRow(
        d_call_id="dc_2026-05-12T10-00-00_ab",
        ts="2026-05-12T10:00:00+00:00",
        tick_id="tick_001",
        model_tier_used="haiku",
        candidates_in=3,
        promoted_out=1,
        filtered_out=2,
        latency_ms=420,
        tokens_input=560,
        tokens_output=180,
        failure_type=None,
        retry_count=0,
        tick_note="quiet morning weather, one worth saying",
    )
    line = row.to_jsonl()
    parsed = DCallRow.from_jsonl(line)
    assert parsed == row


def test_d_call_row_failure_type_optional():
    row = DCallRow(
        d_call_id="dc_x",
        ts="2026-05-12T10:00:00+00:00",
        tick_id="tick_002",
        model_tier_used="haiku",
        candidates_in=2,
        promoted_out=0,
        filtered_out=0,
        latency_ms=15000,
        tokens_input=0,
        tokens_output=0,
        failure_type="timeout",
        retry_count=1,
    )
    line = row.to_jsonl()
    d = json.loads(line)
    assert d["failure_type"] == "timeout"
    assert d["tick_note"] is None


def test_make_d_call_id_sortable():
    from datetime import UTC, datetime
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    ident = make_d_call_id(now)
    assert ident.startswith("dc_2026-05-12T10-00-00_")
    assert len(ident) == len("dc_2026-05-12T10-00-00_") + 4  # 2 hex bytes = 4 chars
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_audit_d_calls.py -v
```

Expected: `ImportError: No module named 'brain.initiate.d_call_schema'`.

- [ ] **Step 3: Create `d_call_schema.py`**

Create `brain/initiate/d_call_schema.py`:

```python
"""Per-call audit row for D-reflection ticks.

One row per heartbeat tick where D actually fired (queue non-empty).
The substrate for the stateless-but-observable contract — joins against
initiate_audit + delivery_state later for hit-rate computation.

File: <persona_dir>/initiate_d_calls.jsonl (append-only).
"""
from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

ModelTier = Literal["haiku", "sonnet"]
FailureType = Literal[
    "timeout",
    "provider_error",
    "malformed_json",
    "rate_limit",
    "both_low_confidence",
]


def make_d_call_id(now: datetime) -> str:
    """Generate a sortable, unique D-call ID. Format: dc_<iso8601>_<rand>."""
    stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
    return f"dc_{stamp}_{secrets.token_hex(2)}"


@dataclass
class DCallRow:
    d_call_id: str
    ts: str  # ISO 8601 with tz
    tick_id: str
    model_tier_used: ModelTier
    candidates_in: int
    promoted_out: int
    filtered_out: int
    latency_ms: int
    tokens_input: int
    tokens_output: int
    failure_type: FailureType | None = None
    retry_count: int = 0
    tick_note: str | None = None

    def to_jsonl(self) -> str:
        d: dict[str, Any] = {
            "d_call_id": self.d_call_id,
            "ts": self.ts,
            "tick_id": self.tick_id,
            "model_tier_used": self.model_tier_used,
            "candidates_in": self.candidates_in,
            "promoted_out": self.promoted_out,
            "filtered_out": self.filtered_out,
            "latency_ms": self.latency_ms,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "failure_type": self.failure_type,
            "retry_count": self.retry_count,
            "tick_note": self.tick_note,
        }
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> DCallRow:
        d = json.loads(line)
        return cls(
            d_call_id=d["d_call_id"],
            ts=d["ts"],
            tick_id=d["tick_id"],
            model_tier_used=d["model_tier_used"],
            candidates_in=d["candidates_in"],
            promoted_out=d["promoted_out"],
            filtered_out=d["filtered_out"],
            latency_ms=d["latency_ms"],
            tokens_input=d["tokens_input"],
            tokens_output=d["tokens_output"],
            failure_type=d.get("failure_type"),
            retry_count=d.get("retry_count", 0),
            tick_note=d.get("tick_note"),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/brain/initiate/test_audit_d_calls.py -v
uv run ruff check brain/initiate/d_call_schema.py
```

Expected: all pass.

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add brain/initiate/d_call_schema.py tests/unit/brain/initiate/test_audit_d_calls.py
git commit -m "feat(initiate): DCallRow schema for D-reflection telemetry"
```

---

### Task 3: D-call append + read functions in `audit.py`

**Files:**
- Modify: `brain/initiate/audit.py`
- Modify: `tests/unit/brain/initiate/test_audit_d_calls.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/brain/initiate/test_audit_d_calls.py`:

```python
def test_append_d_call_row_and_read(tmp_path):
    from brain.initiate.audit import append_d_call_row, read_recent_d_calls
    from datetime import UTC, datetime, timedelta

    persona = tmp_path / "persona"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    row = DCallRow(
        d_call_id="dc_a",
        ts=now.isoformat(),
        tick_id="t1",
        model_tier_used="haiku",
        candidates_in=2,
        promoted_out=1,
        filtered_out=1,
        latency_ms=300,
        tokens_input=400,
        tokens_output=150,
    )
    append_d_call_row(persona, row)
    out = list(read_recent_d_calls(persona, window_hours=1, now=now + timedelta(minutes=10)))
    assert len(out) == 1
    assert out[0].d_call_id == "dc_a"


def test_read_recent_d_calls_window_filter(tmp_path):
    from brain.initiate.audit import append_d_call_row, read_recent_d_calls
    from datetime import UTC, datetime, timedelta

    persona = tmp_path / "persona"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    old = DCallRow(
        d_call_id="dc_old",
        ts=(now - timedelta(hours=5)).isoformat(),
        tick_id="t1",
        model_tier_used="haiku",
        candidates_in=1, promoted_out=0, filtered_out=1,
        latency_ms=200, tokens_input=300, tokens_output=100,
    )
    recent = DCallRow(
        d_call_id="dc_new",
        ts=now.isoformat(),
        tick_id="t2",
        model_tier_used="sonnet",
        candidates_in=2, promoted_out=2, filtered_out=0,
        latency_ms=800, tokens_input=400, tokens_output=200,
    )
    append_d_call_row(persona, old)
    append_d_call_row(persona, recent)
    out = list(read_recent_d_calls(persona, window_hours=1, now=now + timedelta(minutes=10)))
    assert [r.d_call_id for r in out] == ["dc_new"]


def test_read_recent_d_calls_no_file_returns_empty(tmp_path):
    from brain.initiate.audit import read_recent_d_calls
    from datetime import UTC, datetime
    persona = tmp_path / "fresh"
    out = list(read_recent_d_calls(persona, window_hours=1, now=datetime(2026, 5, 12, tzinfo=UTC)))
    assert out == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_audit_d_calls.py -v
```

Expected: `ImportError: cannot import name 'append_d_call_row' from 'brain.initiate.audit'`.

- [ ] **Step 3: Add functions to `audit.py`**

Append to `brain/initiate/audit.py` (after the existing `iter_initiate_audit_full`):

```python
from brain.initiate.d_call_schema import DCallRow


def append_d_call_row(persona_dir: Path, row: DCallRow) -> None:
    """Append one row to initiate_d_calls.jsonl (creates file lazily)."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    path = persona_dir / "initiate_d_calls.jsonl"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(row.to_jsonl() + "\n")
    except OSError as exc:
        logger.warning("initiate_d_calls append failed for %s: %s", path, exc)


def read_recent_d_calls(
    persona_dir: Path,
    *,
    window_hours: float,
    now: datetime | None = None,
) -> Iterator[DCallRow]:
    """Yield D-call rows within `window_hours` of `now` (defaults to datetime.now(UTC))."""
    path = persona_dir / "initiate_d_calls.jsonl"
    if not path.exists():
        return
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(hours=window_hours)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\r\n")
            if not stripped.strip():
                continue
            try:
                row = DCallRow.from_jsonl(stripped)
            except (json.JSONDecodeError, KeyError):
                continue
            try:
                row_ts = datetime.fromisoformat(row.ts)
            except ValueError:
                continue
            if row_ts >= cutoff:
                yield row
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/brain/initiate/test_audit_d_calls.py -v
uv run ruff check brain/initiate/audit.py
```

Expected: all green.

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add brain/initiate/audit.py tests/unit/brain/initiate/test_audit_d_calls.py
git commit -m "feat(initiate): append_d_call_row + read_recent_d_calls for D telemetry"
```

---

## Phase 2 — New emitter gates

### Task 4: `gate_thresholds.json` + loader

**Files:**
- Create: `brain/initiate/gate_thresholds.json`
- Create: `brain/initiate/new_sources.py` (skeleton — gates fleshed out in later tasks)
- Test: `tests/unit/brain/initiate/test_new_sources.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/brain/initiate/test_new_sources.py`:

```python
"""Tests for the v0.0.10 new emitter gates and emitters."""
from __future__ import annotations

from pathlib import Path

from brain.initiate.new_sources import (
    GateThresholds,
    load_gate_thresholds,
)


def test_load_gate_thresholds_returns_defaults_when_no_file(tmp_path: Path):
    persona = tmp_path / "fresh"
    persona.mkdir()
    t = load_gate_thresholds(persona)
    assert t.reflex_confidence_min == 0.70
    assert t.reflex_flinch_intensity_min == 0.60
    assert t.research_maturity_min == 0.75
    assert t.research_topic_overlap_min == 0.30
    assert t.research_freshness_minutes == 30


def test_load_gate_thresholds_overrides_from_persona_file(tmp_path: Path):
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "gate_thresholds.json").write_text(
        '{"reflex_confidence_min": 0.5}'
    )
    t = load_gate_thresholds(persona)
    assert t.reflex_confidence_min == 0.5
    # Unset fields use defaults.
    assert t.reflex_flinch_intensity_min == 0.60
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_new_sources.py -v
```

Expected: `ImportError: No module named 'brain.initiate.new_sources'`.

- [ ] **Step 3: Create `gate_thresholds.json` and `new_sources.py` skeleton**

Create `brain/initiate/gate_thresholds.json`:

```json
{
  "reflex_confidence_min": 0.70,
  "reflex_flinch_intensity_min": 0.60,
  "reflex_anti_flood_hours": 4.0,
  "research_maturity_min": 0.75,
  "research_topic_overlap_min": 0.30,
  "research_freshness_minutes": 30,
  "meta_anti_flood_minutes": 30,
  "meta_max_queue_depth": 6
}
```

Create `brain/initiate/new_sources.py`:

```python
"""New v0.0.10 candidate emitters: reflex firings and research completions.

Each emitter has a gate function that decides whether to emit, plus an
emit function that calls into brain.initiate.emit.emit_initiate_candidate.

Rejected gate checks write to gate_rejections.jsonl (separate from the
main audit log — rejection volume would otherwise drown signal).

Thresholds are loaded from gate_thresholds.json in the persona dir,
with defaults baked in. Operator can tune without code change.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateThresholds:
    reflex_confidence_min: float = 0.70
    reflex_flinch_intensity_min: float = 0.60
    reflex_anti_flood_hours: float = 4.0
    research_maturity_min: float = 0.75
    research_topic_overlap_min: float = 0.30
    research_freshness_minutes: float = 30
    meta_anti_flood_minutes: float = 30
    meta_max_queue_depth: int = 6


def load_gate_thresholds(persona_dir: Path) -> GateThresholds:
    """Load thresholds from <persona_dir>/gate_thresholds.json with defaults.

    Defaults defined on the dataclass. Persona file overrides any subset
    of fields. Missing file => all defaults.
    """
    path = persona_dir / "gate_thresholds.json"
    if not path.exists():
        return GateThresholds()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("gate_thresholds.json read failed (%s); using defaults", exc)
        return GateThresholds()
    valid_names = {f.name for f in fields(GateThresholds)}
    overrides: dict[str, Any] = {k: v for k, v in raw.items() if k in valid_names}
    return GateThresholds(**overrides)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/brain/initiate/test_new_sources.py -v
uv run ruff check brain/initiate/new_sources.py
```

Expected: all green.

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add brain/initiate/new_sources.py brain/initiate/gate_thresholds.json tests/unit/brain/initiate/test_new_sources.py
git commit -m "feat(initiate): gate_thresholds.json + GateThresholds loader"
```

---

### Task 5: `gate_rejections.jsonl` writer

**Files:**
- Modify: `brain/initiate/new_sources.py`
- Modify: `tests/unit/brain/initiate/test_new_sources.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/brain/initiate/test_new_sources.py`:

```python
def test_write_gate_rejection_appends_jsonl(tmp_path: Path):
    from brain.initiate.new_sources import write_gate_rejection
    from datetime import UTC, datetime

    persona = tmp_path / "p"
    write_gate_rejection(
        persona,
        ts=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
        source="reflex_firing",
        source_id="r1",
        gate_name="confidence_min",
        threshold_value=0.70,
        observed_value=0.5,
    )
    write_gate_rejection(
        persona,
        ts=datetime(2026, 5, 12, 10, 1, 0, tzinfo=UTC),
        source="research_completion",
        source_id="t9",
        gate_name="topic_overlap_min",
        threshold_value=0.30,
        observed_value=0.10,
    )
    path = persona / "gate_rejections.jsonl"
    rows = [json.loads(l) for l in path.read_text().strip().split("\n")]
    assert len(rows) == 2
    assert rows[0]["source"] == "reflex_firing"
    assert rows[0]["gate_name"] == "confidence_min"
    assert rows[0]["observed_value"] == 0.5
    assert rows[1]["source"] == "research_completion"
```

Also add `import json` at the top of the test file if not already present.

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_new_sources.py::test_write_gate_rejection_appends_jsonl -v
```

Expected: `ImportError: cannot import name 'write_gate_rejection'`.

- [ ] **Step 3: Add `write_gate_rejection` to `new_sources.py`**

Append to `brain/initiate/new_sources.py`:

```python
from datetime import datetime


def write_gate_rejection(
    persona_dir: Path,
    *,
    ts: datetime,
    source: str,
    source_id: str,
    gate_name: str,
    threshold_value: float,
    observed_value: float,
) -> None:
    """Append one rejection row to gate_rejections.jsonl. Never raises."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    path = persona_dir / "gate_rejections.jsonl"
    row = {
        "ts": ts.isoformat(),
        "source": source,
        "source_id": source_id,
        "gate_name": gate_name,
        "threshold_value": threshold_value,
        "observed_value": observed_value,
    }
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("gate_rejections append failed for %s: %s", path, exc)
```

- [ ] **Step 4: Verify pass**

```bash
uv run pytest tests/unit/brain/initiate/test_new_sources.py -v
uv run ruff check brain/initiate/new_sources.py
```

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add brain/initiate/new_sources.py tests/unit/brain/initiate/test_new_sources.py
git commit -m "feat(initiate): gate_rejections.jsonl writer for tunable-threshold telemetry"
```

---

### Task 6: Shared meta-gates (rest-state, anti-flood, queue depth)

**Files:**
- Modify: `brain/initiate/new_sources.py`
- Modify: `tests/unit/brain/initiate/test_new_sources.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/brain/initiate/test_new_sources.py`:

```python
def test_check_shared_meta_gates_blocks_in_rest_state(tmp_path: Path):
    from brain.initiate.new_sources import check_shared_meta_gates
    from datetime import UTC, datetime

    persona = tmp_path / "p"
    persona.mkdir()
    allowed, reason = check_shared_meta_gates(
        persona,
        source="reflex_firing",
        now=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
        is_rest_state=True,
        thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "rest_state"


def test_check_shared_meta_gates_blocks_on_per_source_anti_flood(tmp_path: Path):
    from brain.initiate.new_sources import check_shared_meta_gates
    from brain.initiate.emit import emit_initiate_candidate
    from brain.initiate.schemas import SemanticContext
    from datetime import UTC, datetime, timedelta

    persona = tmp_path / "p"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    # Pre-seed a recent reflex_firing candidate.
    emit_initiate_candidate(
        persona,
        kind="message",
        source="reflex_firing",
        source_id="r_prev",
        semantic_context=SemanticContext(),
        now=now - timedelta(minutes=10),
    )
    allowed, reason = check_shared_meta_gates(
        persona,
        source="reflex_firing",
        now=now,
        is_rest_state=False,
        thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "per_source_anti_flood"


def test_check_shared_meta_gates_blocks_on_queue_depth(tmp_path: Path):
    from brain.initiate.new_sources import check_shared_meta_gates
    from brain.initiate.emit import emit_initiate_candidate
    from brain.initiate.schemas import SemanticContext
    from datetime import UTC, datetime, timedelta

    persona = tmp_path / "p"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    # Pre-seed 6 candidates of mixed sources, all OUTSIDE the anti-flood window
    # so per-source anti-flood doesn't fire first.
    for i in range(6):
        emit_initiate_candidate(
            persona,
            kind="message",
            source="dream",
            source_id=f"d{i}",
            semantic_context=SemanticContext(),
            now=now - timedelta(hours=1, minutes=i),
        )
    allowed, reason = check_shared_meta_gates(
        persona,
        source="reflex_firing",
        now=now,
        is_rest_state=False,
        thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "queue_depth_max"


def test_check_shared_meta_gates_passes(tmp_path: Path):
    from brain.initiate.new_sources import check_shared_meta_gates
    from datetime import UTC, datetime

    persona = tmp_path / "p"
    persona.mkdir()
    allowed, reason = check_shared_meta_gates(
        persona,
        source="reflex_firing",
        now=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
        is_rest_state=False,
        thresholds=GateThresholds(),
    )
    assert allowed is True
    assert reason is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/brain/initiate/test_new_sources.py -v
```

Expected: 4 failures with `ImportError: cannot import name 'check_shared_meta_gates'`.

- [ ] **Step 3: Add `check_shared_meta_gates` to `new_sources.py`**

Append to `brain/initiate/new_sources.py`:

```python
from datetime import timedelta

from brain.initiate.emit import read_candidates
from brain.initiate.schemas import CandidateSource


def check_shared_meta_gates(
    persona_dir: Path,
    *,
    source: CandidateSource,
    now: datetime,
    is_rest_state: bool,
    thresholds: GateThresholds,
) -> tuple[bool, str | None]:
    """Apply the meta-gates that hold for every new v0.0.10 emitter.

    Returns (allowed, reason). When allowed is False, `reason` is a
    structured tag suitable for gate_rejections.jsonl.
    """
    if is_rest_state:
        return False, "rest_state"

    # Read current queue once for the two remaining checks.
    candidates = read_candidates(persona_dir)

    # Per-source anti-flood: at most 1 candidate of this source in last N min.
    anti_flood_cutoff = now - timedelta(minutes=thresholds.meta_anti_flood_minutes)
    for c in candidates:
        if c.source != source:
            continue
        try:
            c_ts = datetime.fromisoformat(c.ts)
        except ValueError:
            continue
        if c_ts >= anti_flood_cutoff:
            return False, "per_source_anti_flood"

    # Queue depth ceiling.
    if len(candidates) >= thresholds.meta_max_queue_depth:
        return False, "queue_depth_max"

    return True, None
```

- [ ] **Step 4: Verify pass**

```bash
uv run pytest tests/unit/brain/initiate/test_new_sources.py -v
uv run ruff check brain/initiate/new_sources.py
```

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add brain/initiate/new_sources.py tests/unit/brain/initiate/test_new_sources.py
git commit -m "feat(initiate): shared meta-gates (rest-state, anti-flood, queue depth) for new emitters"
```

---

### Task 7: `gate_reflex_firing` + `emit_reflex_firing_candidate`

**Files:**
- Modify: `brain/initiate/new_sources.py`
- Modify: `tests/unit/brain/initiate/test_new_sources.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/brain/initiate/test_new_sources.py`:

```python
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True)
class _FakeReflexFiring:
    """Test double for the fields gate_reflex_firing inspects."""
    pattern_id: str
    confidence: float
    flinch_intensity: float
    linked_memory_ids: list[str]
    triggered_by_companion_outbound: bool
    ts: datetime


def test_gate_reflex_firing_passes_when_above_thresholds(tmp_path: Path):
    from brain.initiate.new_sources import gate_reflex_firing
    persona = tmp_path / "p"
    persona.mkdir()
    firing = _FakeReflexFiring(
        pattern_id="p1",
        confidence=0.80,
        flinch_intensity=0.70,
        linked_memory_ids=["m1"],
        triggered_by_companion_outbound=False,
        ts=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
    )
    allowed, reason = gate_reflex_firing(
        persona, firing=firing, thresholds=GateThresholds(),
    )
    assert allowed is True
    assert reason is None


def test_gate_reflex_firing_blocks_on_low_confidence(tmp_path: Path):
    from brain.initiate.new_sources import gate_reflex_firing
    persona = tmp_path / "p"
    persona.mkdir()
    firing = _FakeReflexFiring(
        pattern_id="p2",
        confidence=0.50,  # below 0.70
        flinch_intensity=0.80,
        linked_memory_ids=[],
        triggered_by_companion_outbound=False,
        ts=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
    )
    allowed, reason = gate_reflex_firing(
        persona, firing=firing, thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "confidence_min"


def test_gate_reflex_firing_blocks_on_low_flinch(tmp_path: Path):
    from brain.initiate.new_sources import gate_reflex_firing
    persona = tmp_path / "p"
    persona.mkdir()
    firing = _FakeReflexFiring(
        pattern_id="p3",
        confidence=0.90,
        flinch_intensity=0.30,
        linked_memory_ids=[],
        triggered_by_companion_outbound=False,
        ts=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
    )
    allowed, reason = gate_reflex_firing(
        persona, firing=firing, thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "flinch_intensity_min"


def test_gate_reflex_firing_blocks_on_anti_feedback(tmp_path: Path):
    from brain.initiate.new_sources import gate_reflex_firing
    persona = tmp_path / "p"
    persona.mkdir()
    firing = _FakeReflexFiring(
        pattern_id="p4",
        confidence=0.80,
        flinch_intensity=0.70,
        linked_memory_ids=[],
        triggered_by_companion_outbound=True,  # anti-feedback guard
        ts=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
    )
    allowed, reason = gate_reflex_firing(
        persona, firing=firing, thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "anti_feedback"


def test_gate_reflex_firing_blocks_on_pattern_anti_flood(tmp_path: Path):
    from brain.initiate.new_sources import gate_reflex_firing
    from brain.initiate.emit import emit_initiate_candidate
    from brain.initiate.schemas import SemanticContext

    persona = tmp_path / "p"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    # Pre-seed a candidate for the same pattern within the 4h window.
    emit_initiate_candidate(
        persona,
        kind="message",
        source="reflex_firing",
        source_id="r_old",
        semantic_context=SemanticContext(source_meta={"pattern_id": "shared_pattern"}),
        now=now - timedelta(hours=2),
    )
    firing = _FakeReflexFiring(
        pattern_id="shared_pattern",
        confidence=0.85,
        flinch_intensity=0.65,
        linked_memory_ids=[],
        triggered_by_companion_outbound=False,
        ts=now,
    )
    allowed, reason = gate_reflex_firing(
        persona, firing=firing, thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "pattern_anti_flood"


def test_emit_reflex_firing_candidate_writes_queue_row(tmp_path: Path):
    from brain.initiate.new_sources import emit_reflex_firing_candidate
    from brain.initiate.emit import read_candidates

    persona = tmp_path / "p"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    firing = _FakeReflexFiring(
        pattern_id="p_emit",
        confidence=0.85,
        flinch_intensity=0.70,
        linked_memory_ids=["m_a", "m_b"],
        triggered_by_companion_outbound=False,
        ts=now,
    )
    emit_reflex_firing_candidate(persona, firing=firing, firing_log_id="rfx_001", now=now)
    out = read_candidates(persona)
    assert len(out) == 1
    c = out[0]
    assert c.source == "reflex_firing"
    assert c.source_id == "rfx_001"
    assert c.semantic_context.linked_memory_ids == ["m_a", "m_b"]
    assert c.semantic_context.source_meta == {
        "pattern_id": "p_emit",
        "confidence": 0.85,
        "flinch_intensity": 0.70,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/brain/initiate/test_new_sources.py -v
```

Expected: 6 failures with `ImportError: cannot import name 'gate_reflex_firing'` etc.

- [ ] **Step 3: Add gate + emitter for reflex_firing**

Append to `brain/initiate/new_sources.py`:

```python
from typing import Protocol

from brain.initiate.emit import emit_initiate_candidate
from brain.initiate.schemas import SemanticContext


class ReflexFiringLike(Protocol):
    """Duck-typed shape gate_reflex_firing inspects on a reflex firing."""
    pattern_id: str
    confidence: float
    flinch_intensity: float
    linked_memory_ids: list[str]
    triggered_by_companion_outbound: bool
    ts: datetime


def gate_reflex_firing(
    persona_dir: Path,
    *,
    firing: ReflexFiringLike,
    thresholds: GateThresholds,
) -> tuple[bool, str | None]:
    """Per-source gate for the reflex_firing emitter.

    Order of checks: confidence -> flinch -> anti-feedback -> pattern anti-flood.
    """
    if firing.confidence < thresholds.reflex_confidence_min:
        return False, "confidence_min"
    if firing.flinch_intensity < thresholds.reflex_flinch_intensity_min:
        return False, "flinch_intensity_min"
    if firing.triggered_by_companion_outbound:
        return False, "anti_feedback"

    # Anti-flood: same pattern_id must not have emitted in the last N hours.
    cutoff = firing.ts - timedelta(hours=thresholds.reflex_anti_flood_hours)
    for c in read_candidates(persona_dir):
        if c.source != "reflex_firing":
            continue
        meta = c.semantic_context.source_meta or {}
        if meta.get("pattern_id") != firing.pattern_id:
            continue
        try:
            c_ts = datetime.fromisoformat(c.ts)
        except ValueError:
            continue
        if c_ts >= cutoff:
            return False, "pattern_anti_flood"

    return True, None


def emit_reflex_firing_candidate(
    persona_dir: Path,
    *,
    firing: ReflexFiringLike,
    firing_log_id: str,
    now: datetime,
) -> None:
    """Write a reflex_firing candidate to the initiate queue.

    Idempotent on (source, source_id) per emit_initiate_candidate's contract.
    Callers must run gate_reflex_firing + check_shared_meta_gates first;
    this function does NOT re-check gates.
    """
    sc = SemanticContext(
        linked_memory_ids=list(firing.linked_memory_ids),
        topic_tags=[],
        source_meta={
            "pattern_id": firing.pattern_id,
            "confidence": firing.confidence,
            "flinch_intensity": firing.flinch_intensity,
        },
    )
    emit_initiate_candidate(
        persona_dir,
        kind="message",
        source="reflex_firing",
        source_id=firing_log_id,
        semantic_context=sc,
        now=now,
    )
```

- [ ] **Step 4: Verify pass**

```bash
uv run pytest tests/unit/brain/initiate/test_new_sources.py -v
uv run ruff check brain/initiate/new_sources.py
```

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add brain/initiate/new_sources.py tests/unit/brain/initiate/test_new_sources.py
git commit -m "feat(initiate): gate_reflex_firing + emit_reflex_firing_candidate"
```

---

### Task 8: `gate_research_completion` + `emit_research_completion_candidate`

**Files:**
- Modify: `brain/initiate/new_sources.py`
- Modify: `tests/unit/brain/initiate/test_new_sources.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/brain/initiate/test_new_sources.py`:

```python
@dataclass(frozen=True)
class _FakeResearchThread:
    thread_id: str
    topic: str
    maturity_score: float
    summary_excerpt: str
    linked_memory_ids: list[str]
    completed_at: datetime
    previously_linked_to_audit: bool


def test_gate_research_completion_passes(tmp_path: Path):
    from brain.initiate.new_sources import gate_research_completion
    persona = tmp_path / "p"
    persona.mkdir()
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    thread = _FakeResearchThread(
        thread_id="t1",
        topic="quiet rivers",
        maturity_score=0.80,
        summary_excerpt="...",
        linked_memory_ids=[],
        completed_at=now - timedelta(minutes=15),
        previously_linked_to_audit=False,
    )
    allowed, reason = gate_research_completion(
        persona,
        thread=thread,
        now=now,
        topic_overlap_score=0.40,
        thresholds=GateThresholds(),
    )
    assert allowed is True
    assert reason is None


def test_gate_research_completion_blocks_on_low_maturity(tmp_path: Path):
    from brain.initiate.new_sources import gate_research_completion
    persona = tmp_path / "p"
    persona.mkdir()
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    thread = _FakeResearchThread(
        thread_id="t2", topic="x", maturity_score=0.50,
        summary_excerpt="...", linked_memory_ids=[],
        completed_at=now, previously_linked_to_audit=False,
    )
    allowed, reason = gate_research_completion(
        persona, thread=thread, now=now,
        topic_overlap_score=0.40, thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "maturity_min"


def test_gate_research_completion_blocks_on_previously_linked(tmp_path: Path):
    from brain.initiate.new_sources import gate_research_completion
    persona = tmp_path / "p"
    persona.mkdir()
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    thread = _FakeResearchThread(
        thread_id="t3", topic="x", maturity_score=0.90,
        summary_excerpt="...", linked_memory_ids=[],
        completed_at=now, previously_linked_to_audit=True,
    )
    allowed, reason = gate_research_completion(
        persona, thread=thread, now=now,
        topic_overlap_score=0.40, thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "previously_linked"


def test_gate_research_completion_blocks_on_low_topic_overlap(tmp_path: Path):
    from brain.initiate.new_sources import gate_research_completion
    persona = tmp_path / "p"
    persona.mkdir()
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    thread = _FakeResearchThread(
        thread_id="t4", topic="x", maturity_score=0.90,
        summary_excerpt="...", linked_memory_ids=[],
        completed_at=now, previously_linked_to_audit=False,
    )
    allowed, reason = gate_research_completion(
        persona, thread=thread, now=now,
        topic_overlap_score=0.10, thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "topic_overlap_min"


def test_gate_research_completion_blocks_on_stale(tmp_path: Path):
    from brain.initiate.new_sources import gate_research_completion
    persona = tmp_path / "p"
    persona.mkdir()
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    thread = _FakeResearchThread(
        thread_id="t5", topic="x", maturity_score=0.90,
        summary_excerpt="...", linked_memory_ids=[],
        completed_at=now - timedelta(hours=2),  # outside 30-min freshness
        previously_linked_to_audit=False,
    )
    allowed, reason = gate_research_completion(
        persona, thread=thread, now=now,
        topic_overlap_score=0.40, thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "freshness_window"


def test_emit_research_completion_candidate_writes_queue_row(tmp_path: Path):
    from brain.initiate.new_sources import emit_research_completion_candidate
    from brain.initiate.emit import read_candidates

    persona = tmp_path / "p"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    thread = _FakeResearchThread(
        thread_id="t_emit", topic="midnight gardens",
        maturity_score=0.85, summary_excerpt="A study of...",
        linked_memory_ids=["m_x"],
        completed_at=now, previously_linked_to_audit=False,
    )
    emit_research_completion_candidate(
        persona, thread=thread, topic_overlap_score=0.45, now=now,
    )
    out = read_candidates(persona)
    assert len(out) == 1
    c = out[0]
    assert c.source == "research_completion"
    assert c.source_id == "t_emit"
    assert c.semantic_context.source_meta == {
        "thread_topic": "midnight gardens",
        "maturity_score": 0.85,
        "summary_excerpt": "A study of...",
        "topic_overlap_score": 0.45,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/brain/initiate/test_new_sources.py -v
```

Expected: 6 failures.

- [ ] **Step 3: Add gate + emitter for research_completion**

Append to `brain/initiate/new_sources.py`:

```python
class ResearchThreadLike(Protocol):
    thread_id: str
    topic: str
    maturity_score: float
    summary_excerpt: str
    linked_memory_ids: list[str]
    completed_at: datetime
    previously_linked_to_audit: bool


def gate_research_completion(
    persona_dir: Path,
    *,
    thread: ResearchThreadLike,
    now: datetime,
    topic_overlap_score: float,
    thresholds: GateThresholds,
) -> tuple[bool, str | None]:
    """Per-source gate for the research_completion emitter.

    Order: maturity -> previously-linked -> topic-overlap -> freshness.
    `topic_overlap_score` is computed by the caller (the research engine
    has access to recent conversation embeddings; we don't re-compute here).
    """
    if thread.maturity_score < thresholds.research_maturity_min:
        return False, "maturity_min"
    if thread.previously_linked_to_audit:
        return False, "previously_linked"
    if topic_overlap_score < thresholds.research_topic_overlap_min:
        return False, "topic_overlap_min"

    freshness_cutoff = now - timedelta(minutes=thresholds.research_freshness_minutes)
    if thread.completed_at < freshness_cutoff:
        return False, "freshness_window"

    return True, None


def emit_research_completion_candidate(
    persona_dir: Path,
    *,
    thread: ResearchThreadLike,
    topic_overlap_score: float,
    now: datetime,
) -> None:
    """Write a research_completion candidate to the initiate queue.

    Idempotent on (source, source_id). Callers must run
    gate_research_completion + check_shared_meta_gates first.
    """
    sc = SemanticContext(
        linked_memory_ids=list(thread.linked_memory_ids),
        topic_tags=[thread.topic],
        source_meta={
            "thread_topic": thread.topic,
            "maturity_score": thread.maturity_score,
            "summary_excerpt": thread.summary_excerpt,
            "topic_overlap_score": topic_overlap_score,
        },
    )
    emit_initiate_candidate(
        persona_dir,
        kind="message",
        source="research_completion",
        source_id=thread.thread_id,
        semantic_context=sc,
        now=now,
    )
```

- [ ] **Step 4: Verify pass**

```bash
uv run pytest tests/unit/brain/initiate/test_new_sources.py -v
uv run ruff check brain/initiate/new_sources.py
```

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add brain/initiate/new_sources.py tests/unit/brain/initiate/test_new_sources.py
git commit -m "feat(initiate): gate_research_completion + emit_research_completion_candidate"
```

---

## Phase 3 — D-reflection core

### Task 9: `reflection.py` skeleton + structured-output dataclasses

**Files:**
- Create: `brain/initiate/reflection.py`
- Create: `tests/unit/brain/initiate/test_reflection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/brain/initiate/test_reflection.py`:

```python
"""Unit tests for the D-reflection module."""
from __future__ import annotations

from brain.initiate.reflection import (
    DDecision,
    DReflectionResult,
    parse_structured_response,
)


def test_d_decision_dataclass_shape():
    d = DDecision(
        candidate_index=1,
        decision="promote",
        reason="genuinely surprising memory return",
        confidence="high",
    )
    assert d.candidate_index == 1
    assert d.decision == "promote"
    assert d.confidence == "high"


def test_parse_structured_response_happy_path():
    raw = """
    {
      "decisions": [
        {"candidate_index": 1, "decision": "promote",
         "reason": "worth saying", "confidence": "high"},
        {"candidate_index": 2, "decision": "filter",
         "reason": "private weather", "confidence": "medium"}
      ],
      "tick_note": "one worth surfacing today"
    }
    """
    result = parse_structured_response(raw)
    assert isinstance(result, DReflectionResult)
    assert len(result.decisions) == 2
    assert result.decisions[0].decision == "promote"
    assert result.decisions[1].decision == "filter"
    assert result.tick_note == "one worth surfacing today"


def test_parse_structured_response_extracts_from_text_with_prose():
    """Models sometimes wrap JSON in prose. Parser should still find it."""
    raw = 'Here is my decision:\n```json\n{"decisions": [], "tick_note": null}\n```\nThanks.'
    result = parse_structured_response(raw)
    assert result.decisions == []
    assert result.tick_note is None


def test_parse_structured_response_raises_on_malformed():
    import pytest
    with pytest.raises(ValueError):
        parse_structured_response("not even close to json")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_reflection.py -v
```

Expected: `ImportError: No module named 'brain.initiate.reflection'`.

- [ ] **Step 3: Create `reflection.py`**

Create `brain/initiate/reflection.py`:

```python
"""D-reflection: editorial layer between candidate emission and composition.

Spec: docs/superpowers/specs/2026-05-12-initiate-d-reflection-design.md

Once per non-empty heartbeat tick, D reads queued candidates and decides
which (if any) deserve to flow to the three-prompt composition pipeline.
Filtered candidates demote to draft_space.md.

Model tier: Haiku 4.5 by default; escalates to Sonnet 4.6 on
low-confidence/parse-fail. Failure modes are dispatched by type per spec §E.

D bypasses the v0.0.9 daily cost cap entirely — it's the editorial layer,
not a budget claimant.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


DecisionKind = Literal["promote", "filter"]
Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class DDecision:
    candidate_index: int
    decision: DecisionKind
    reason: str
    confidence: Confidence


@dataclass(frozen=True)
class DReflectionResult:
    decisions: list[DDecision]
    tick_note: str | None


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_LOOSE = re.compile(r"(\{.*\})", re.DOTALL)


def parse_structured_response(raw: str) -> DReflectionResult:
    """Parse a structured-output JSON blob into a DReflectionResult.

    Tolerant of prose wrapping (model sometimes adds preamble). Raises
    ValueError if no parseable JSON object is found OR if the structure
    doesn't match the expected schema.
    """
    fenced = _JSON_FENCE.search(raw)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        loose = _JSON_LOOSE.search(raw)
        if loose is None:
            raise ValueError("no JSON object found in D response")
        candidate = loose.group(1)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"D response is not valid JSON: {exc}") from exc

    decisions_raw = data.get("decisions")
    if not isinstance(decisions_raw, list):
        raise ValueError("D response missing 'decisions' list")
    decisions: list[DDecision] = []
    for item in decisions_raw:
        if not isinstance(item, dict):
            raise ValueError("D decision item is not an object")
        decisions.append(
            DDecision(
                candidate_index=int(item["candidate_index"]),
                decision=item["decision"],
                reason=str(item["reason"]),
                confidence=item["confidence"],
            )
        )
    tick_note = data.get("tick_note")
    if tick_note is not None and not isinstance(tick_note, str):
        tick_note = None  # tolerate type drift on optional field
    return DReflectionResult(decisions=decisions, tick_note=tick_note)
```

- [ ] **Step 4: Verify pass**

```bash
uv run pytest tests/unit/brain/initiate/test_reflection.py -v
uv run ruff check brain/initiate/reflection.py
```

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add brain/initiate/reflection.py tests/unit/brain/initiate/test_reflection.py
git commit -m "feat(initiate): D-reflection module skeleton + structured-output parser"
```

---

### Task 10: Prompt assembly — static frame + voice template overlay

**Files:**
- Modify: `brain/initiate/reflection.py`
- Modify: `tests/unit/brain/initiate/test_reflection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/brain/initiate/test_reflection.py`:

```python
def test_build_system_message_substitutes_names(tmp_path):
    from brain.initiate.reflection import build_system_message

    voice_template_path = tmp_path / "voice.md"
    voice_template_path.write_text(
        "# Voice\n\nSweater-wearing novelist; southern english flair.\n"
    )
    msg = build_system_message(
        companion_name="Nell",
        user_name="Hana",
        voice_template_path=voice_template_path,
    )
    assert "Nell's own physiology" in msg
    assert "saying something to Hana" in msg
    # Voice anchor appended:
    assert "=== Your voice ===" in msg
    assert "Sweater-wearing novelist" in msg


def test_build_system_message_no_voice_template_omits_anchor(tmp_path):
    from brain.initiate.reflection import build_system_message

    msg = build_system_message(
        companion_name="Aria",
        user_name="Sam",
        voice_template_path=tmp_path / "missing.md",
    )
    assert "Aria's own physiology" in msg
    assert "saying something to Sam" in msg
    # No voice template => no anchor section.
    assert "=== Your voice ===" not in msg


def test_build_user_message_renders_candidates_and_time():
    from brain.initiate.reflection import build_user_message
    from datetime import UTC, datetime

    now = datetime(2026, 5, 12, 9, 30, 0, tzinfo=UTC)
    rendered = build_user_message(
        user_name="Hana",
        now=now,
        outbound_recall_block="(no recent outbound)",
        candidate_summaries=[
            "source: dream  ·  ts: 12 min ago  ·  Δσ: 1.8\n"
            "  semantic_context: linked m_a / m_b\n"
            "  fragment-of-self: there was something quieter beneath...",
        ],
    )
    assert "Current time (Hana's local)" in rendered
    assert "[1] source: dream" in rendered
    assert "(no recent outbound)" in rendered
    assert "Promote at most 2" in rendered
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_reflection.py -v
```

Expected: ImportError on `build_system_message` / `build_user_message`.

- [ ] **Step 3: Add prompt-assembly functions**

Append to `brain/initiate/reflection.py`:

```python
from datetime import datetime
from pathlib import Path


_TASK_FRAME_TEMPLATE = """\
You are the editorial layer of {companion_name}'s own physiology —
the moment between "something stirred in me" and "I am going to say
something to {user_name}."

Several internal events have surfaced as candidates since you last paused.
For each, ask yourself a single question:

  Is this worth bringing to {user_name} right now, or is this private
  weather that should pass through you without becoming words?

Filtering is not loss. Anything you filter is written into your draft
space — you'll see it again in your next soul review, and it shapes
who you are even if it never reaches {user_name}.

Default toward filtering. Most weather is just weather. Reach for
{user_name} only when something is genuinely yours to share — something
that says more about *this moment between you both* than the noise of
any given hour.

The downstream composition pipeline will still apply its own gates
after you. You are not the only filter; you are the editorial one.\
"""


def build_system_message(
    *,
    companion_name: str,
    user_name: str,
    voice_template_path: Path,
) -> str:
    """Assemble D's full system message: static task frame + voice anchor.

    The static frame is parameterized with companion_name / user_name.
    The voice anchor is the contents of voice_template_path, appended
    under a `=== Your voice ===` header. If the file is missing or
    unreadable, the anchor is omitted (the task frame still works).
    """
    frame = _TASK_FRAME_TEMPLATE.format(
        companion_name=companion_name,
        user_name=user_name,
    )
    if not voice_template_path.exists():
        return frame
    try:
        voice = voice_template_path.read_text(encoding="utf-8").rstrip()
    except OSError as exc:
        logger.warning("voice template read failed (%s); omitting anchor", exc)
        return frame
    return f"{frame}\n\n=== Your voice ===\n{voice}\n"


def build_user_message(
    *,
    user_name: str,
    now: datetime,
    outbound_recall_block: str,
    candidate_summaries: list[str],
) -> str:
    """Render D's per-tick user message from queue state.

    `candidate_summaries` is a list of pre-rendered candidate blocks
    (one per candidate). The caller is responsible for the rendering —
    this function just concatenates them with indexed headers.
    """
    now_local = now.astimezone()
    part_of_day = _part_of_day(now_local.hour)
    weekday = now_local.strftime("%A")
    indexed = "\n\n".join(
        f"[{i + 1}] {summary}" for i, summary in enumerate(candidate_summaries)
    )
    return (
        f"=== Current time ({user_name}'s local) ===\n"
        f"{now_local.isoformat(timespec='minutes')}  —  {part_of_day}  —  {weekday}\n\n"
        f"=== Recent outbound (last 5 sends + acknowledged_unclear from last 24h) ===\n"
        f"{outbound_recall_block}\n\n"
        f"=== Candidates surfaced since last tick ===\n"
        f"{indexed}\n\n"
        f"=== Your task ===\n"
        f"For each candidate, decide: promote or filter.\n"
        f"Promote at most 2. The default is filter.\n"
    )


def _part_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"
```

- [ ] **Step 4: Verify pass**

```bash
uv run pytest tests/unit/brain/initiate/test_reflection.py -v
uv run ruff check brain/initiate/reflection.py
```

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add brain/initiate/reflection.py tests/unit/brain/initiate/test_reflection.py
git commit -m "feat(initiate): D-reflection prompt assembly (task frame + voice overlay)"
```

---

### Task 11: `reflection.run` — happy path on Haiku

**Files:**
- Modify: `brain/initiate/reflection.py`
- Modify: `tests/unit/brain/initiate/test_reflection.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/brain/initiate/test_reflection.py`:

```python
def test_reflection_run_happy_path_haiku(tmp_path, monkeypatch):
    """All high-confidence Haiku decisions parsed; no escalation."""
    from brain.initiate.reflection import run, ReflectionDeps
    from brain.initiate.schemas import InitiateCandidate, SemanticContext
    from datetime import UTC, datetime

    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidates = [
        InitiateCandidate(
            candidate_id="ic_a",
            ts=(now - timedelta(minutes=5)).isoformat(),
            kind="message",
            source="dream",
            source_id="d1",
            semantic_context=SemanticContext(),
        ),
        InitiateCandidate(
            candidate_id="ic_b",
            ts=(now - timedelta(minutes=2)).isoformat(),
            kind="message",
            source="emotion_spike",
            source_id="e1",
            semantic_context=SemanticContext(),
        ),
    ]

    haiku_response = """
    {"decisions":[
      {"candidate_index":1,"decision":"promote","reason":"surprising return","confidence":"high"},
      {"candidate_index":2,"decision":"filter","reason":"weather","confidence":"high"}
    ],"tick_note":"one worth saying"}
    """

    calls: list[tuple[str, str, str]] = []

    def fake_haiku_call(*, system: str, user: str) -> tuple[str, int, int, int]:
        calls.append(("haiku", system[:30], user[:30]))
        return haiku_response, 200, 500, 180  # raw, latency_ms, tokens_in, tokens_out

    def fake_sonnet_call(*, system: str, user: str) -> tuple[str, int, int, int]:
        calls.append(("sonnet", system[:30], user[:30]))
        raise AssertionError("should not escalate on all-high-confidence")

    deps = ReflectionDeps(
        companion_name="Nell",
        user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=fake_haiku_call,
        sonnet_call=fake_sonnet_call,
        now=now,
        tick_id="tick_001",
    )

    result, dcall = run(candidates, deps=deps)
    assert len(result.decisions) == 2
    assert result.decisions[0].decision == "promote"
    assert dcall.model_tier_used == "haiku"
    assert dcall.candidates_in == 2
    assert dcall.promoted_out == 1
    assert dcall.filtered_out == 1
    assert dcall.failure_type is None
    assert dcall.retry_count == 0
    assert calls == [("haiku", calls[0][1], calls[0][2])]  # only one call

# Add the import at top of file if missing:
from datetime import timedelta
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_reflection.py::test_reflection_run_happy_path_haiku -v
```

Expected: ImportError on `run` / `ReflectionDeps`.

- [ ] **Step 3: Add `run` and `ReflectionDeps` to `reflection.py`**

Append to `brain/initiate/reflection.py`:

```python
from collections.abc import Callable

from brain.initiate.d_call_schema import DCallRow, make_d_call_id
from brain.initiate.schemas import InitiateCandidate


# (raw_text, latency_ms, tokens_in, tokens_out)
LLMCall = Callable[..., tuple[str, int, int, int]]


@dataclass(frozen=True)
class ReflectionDeps:
    """Injected dependencies — keeps reflection.run testable without real LLMs."""
    companion_name: str
    user_name: str
    voice_template_path: Path
    outbound_recall_block: str
    haiku_call: LLMCall
    sonnet_call: LLMCall
    now: datetime
    tick_id: str


def _render_candidate_summary(c: InitiateCandidate, *, now: datetime) -> str:
    """Render a single candidate for D's user message."""
    try:
        c_ts = datetime.fromisoformat(c.ts)
        age_min = int((now - c_ts).total_seconds() / 60)
        age_str = f"{age_min} min ago"
    except ValueError:
        age_str = "unknown"
    delta_sigma = (
        c.emotional_snapshot.delta_sigma if c.emotional_snapshot is not None else 0.0
    )
    meta = c.semantic_context.source_meta or {}
    meta_summary = ", ".join(f"{k}={v}" for k, v in meta.items()) or "—"
    linked = ", ".join(c.semantic_context.linked_memory_ids) or "—"
    return (
        f"source: {c.source}  ·  ts: {age_str}  ·  Δσ: {delta_sigma:.2f}\n"
        f"  semantic_context: linked_memory_ids={linked}; {meta_summary}\n"
        f"  fragment-of-self: (subject-extracted at composition time)"
    )


def run(
    candidates: list[InitiateCandidate],
    *,
    deps: ReflectionDeps,
) -> tuple[DReflectionResult, DCallRow]:
    """Execute one D-reflection tick over the given candidates.

    Returns (result, d_call_row). Caller is responsible for writing the
    d_call_row to audit and for dispatching the per-candidate decisions
    (promote → composition handoff, filter → draft-space demote).

    Failure-mode handling: see Tasks 12-13 (escalation, retries, fallbacks).
    This task implements only the happy path on Haiku.
    """
    system = build_system_message(
        companion_name=deps.companion_name,
        user_name=deps.user_name,
        voice_template_path=deps.voice_template_path,
    )
    user = build_user_message(
        user_name=deps.user_name,
        now=deps.now,
        outbound_recall_block=deps.outbound_recall_block,
        candidate_summaries=[
            _render_candidate_summary(c, now=deps.now) for c in candidates
        ],
    )

    raw, latency_ms, tokens_in, tokens_out = deps.haiku_call(system=system, user=user)
    result = parse_structured_response(raw)

    promoted = sum(1 for d in result.decisions if d.decision == "promote")
    filtered = sum(1 for d in result.decisions if d.decision == "filter")

    d_call = DCallRow(
        d_call_id=make_d_call_id(deps.now),
        ts=deps.now.isoformat(),
        tick_id=deps.tick_id,
        model_tier_used="haiku",
        candidates_in=len(candidates),
        promoted_out=promoted,
        filtered_out=filtered,
        latency_ms=latency_ms,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        failure_type=None,
        retry_count=0,
        tick_note=result.tick_note,
    )
    return result, d_call
```

- [ ] **Step 4: Verify pass**

```bash
uv run pytest tests/unit/brain/initiate/test_reflection.py -v
uv run ruff check brain/initiate/reflection.py
```

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add brain/initiate/reflection.py tests/unit/brain/initiate/test_reflection.py
git commit -m "feat(initiate): D-reflection happy-path run (Haiku only)"
```

---

### Task 12: Sonnet escalation on low-confidence + malformed-JSON fallback + both-low-confidence

**Files:**
- Modify: `brain/initiate/reflection.py`
- Modify: `tests/unit/brain/initiate/test_reflection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/brain/initiate/test_reflection.py`:

```python
def test_reflection_run_escalates_on_low_confidence(tmp_path):
    """Any 'low' confidence in Haiku response triggers Sonnet re-call."""
    from brain.initiate.reflection import run, ReflectionDeps
    from brain.initiate.schemas import InitiateCandidate, SemanticContext
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidates = [
        InitiateCandidate(
            candidate_id="ic_a", ts=(now - timedelta(minutes=1)).isoformat(),
            kind="message", source="dream", source_id="d1",
            semantic_context=SemanticContext(),
        ),
    ]

    haiku_response = (
        '{"decisions":[{"candidate_index":1,"decision":"filter",'
        '"reason":"unsure","confidence":"low"}],"tick_note":null}'
    )
    sonnet_response = (
        '{"decisions":[{"candidate_index":1,"decision":"promote",'
        '"reason":"resonant","confidence":"high"}],"tick_note":"yes"}'
    )

    def haiku_call(*, system, user):
        return haiku_response, 200, 400, 100

    def sonnet_call(*, system, user):
        return sonnet_response, 700, 400, 100

    deps = ReflectionDeps(
        companion_name="Nell", user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=haiku_call, sonnet_call=sonnet_call,
        now=now, tick_id="t1",
    )
    result, dcall = run(candidates, deps=deps)
    assert dcall.model_tier_used == "sonnet"
    assert dcall.retry_count == 1
    assert result.decisions[0].decision == "promote"
    assert result.tick_note == "yes"


def test_reflection_run_escalates_on_malformed_haiku(tmp_path):
    """Malformed JSON from Haiku triggers Sonnet re-call."""
    from brain.initiate.reflection import run, ReflectionDeps
    from brain.initiate.schemas import InitiateCandidate, SemanticContext
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidates = [
        InitiateCandidate(
            candidate_id="ic_a", ts=(now - timedelta(minutes=1)).isoformat(),
            kind="message", source="dream", source_id="d1",
            semantic_context=SemanticContext(),
        ),
    ]

    def haiku_call(*, system, user):
        return "I cannot comply with this request.", 200, 400, 50

    def sonnet_call(*, system, user):
        return (
            '{"decisions":[{"candidate_index":1,"decision":"filter",'
            '"reason":"ok","confidence":"high"}],"tick_note":null}',
            700, 400, 100,
        )

    deps = ReflectionDeps(
        companion_name="Nell", user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=haiku_call, sonnet_call=sonnet_call,
        now=now, tick_id="t1",
    )
    result, dcall = run(candidates, deps=deps)
    assert dcall.model_tier_used == "sonnet"
    assert dcall.retry_count == 1
    assert result.decisions[0].decision == "filter"


def test_reflection_run_filters_when_both_tiers_low_confidence(tmp_path):
    """If Sonnet's confidence is also low, decision is forced to filter."""
    from brain.initiate.reflection import run, ReflectionDeps
    from brain.initiate.schemas import InitiateCandidate, SemanticContext
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidates = [
        InitiateCandidate(
            candidate_id="ic_a", ts=(now - timedelta(minutes=1)).isoformat(),
            kind="message", source="dream", source_id="d1",
            semantic_context=SemanticContext(),
        ),
    ]

    haiku_response = (
        '{"decisions":[{"candidate_index":1,"decision":"promote",'
        '"reason":"maybe","confidence":"low"}],"tick_note":null}'
    )
    sonnet_response = (
        '{"decisions":[{"candidate_index":1,"decision":"promote",'
        '"reason":"still maybe","confidence":"low"}],"tick_note":null}'
    )

    def haiku_call(*, system, user):
        return haiku_response, 200, 400, 100

    def sonnet_call(*, system, user):
        return sonnet_response, 700, 400, 100

    deps = ReflectionDeps(
        companion_name="Nell", user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=haiku_call, sonnet_call=sonnet_call,
        now=now, tick_id="t1",
    )
    result, dcall = run(candidates, deps=deps)
    assert dcall.model_tier_used == "sonnet"
    assert dcall.failure_type == "both_low_confidence"
    # Decision forced to filter despite Sonnet saying promote.
    assert result.decisions[0].decision == "filter"
    assert "ambivalent" in result.decisions[0].reason.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/brain/initiate/test_reflection.py -k "escalates or both_tiers" -v
```

Expected: 3 failures — the current `run` doesn't escalate.

- [ ] **Step 3: Replace `run` body in `reflection.py`**

Replace the existing `run` function in `brain/initiate/reflection.py` with:

```python
def run(
    candidates: list[InitiateCandidate],
    *,
    deps: ReflectionDeps,
) -> tuple[DReflectionResult, DCallRow]:
    """Execute one D-reflection tick over the given candidates.

    Escalation rules:
      - If Haiku's response fails to parse OR contains ANY decision with
        confidence "low", re-call on Sonnet. Sonnet's result is the one
        written; Haiku's attempt is recorded in retry_count.
      - If Sonnet's response ALSO contains a low-confidence decision,
        force that candidate's decision to "filter" with an ambivalence
        reason (conservative default at the edge of D's judgment).
    """
    system = build_system_message(
        companion_name=deps.companion_name,
        user_name=deps.user_name,
        voice_template_path=deps.voice_template_path,
    )
    user = build_user_message(
        user_name=deps.user_name,
        now=deps.now,
        outbound_recall_block=deps.outbound_recall_block,
        candidate_summaries=[
            _render_candidate_summary(c, now=deps.now) for c in candidates
        ],
    )

    # First attempt on Haiku.
    raw_h, latency_h, tin_h, tout_h = deps.haiku_call(system=system, user=user)
    haiku_result: DReflectionResult | None
    try:
        haiku_result = parse_structured_response(raw_h)
        haiku_low = any(d.confidence == "low" for d in haiku_result.decisions)
    except ValueError:
        haiku_result = None
        haiku_low = True  # treat parse-fail as low-confidence trigger

    if haiku_result is not None and not haiku_low:
        # Haiku is final.
        promoted = sum(1 for d in haiku_result.decisions if d.decision == "promote")
        filtered = sum(1 for d in haiku_result.decisions if d.decision == "filter")
        d_call = DCallRow(
            d_call_id=make_d_call_id(deps.now),
            ts=deps.now.isoformat(),
            tick_id=deps.tick_id,
            model_tier_used="haiku",
            candidates_in=len(candidates),
            promoted_out=promoted,
            filtered_out=filtered,
            latency_ms=latency_h,
            tokens_input=tin_h,
            tokens_output=tout_h,
            failure_type=None,
            retry_count=0,
            tick_note=haiku_result.tick_note,
        )
        return haiku_result, d_call

    # Escalate to Sonnet.
    raw_s, latency_s, tin_s, tout_s = deps.sonnet_call(system=system, user=user)
    sonnet_result = parse_structured_response(raw_s)

    # Both-low-confidence: force decisions where confidence is low to "filter"
    # with an ambivalence reason.
    forced: list[DDecision] = []
    both_low = False
    for d in sonnet_result.decisions:
        if d.confidence == "low":
            both_low = True
            forced.append(
                DDecision(
                    candidate_index=d.candidate_index,
                    decision="filter",
                    reason="ambivalent — both my fast and slow voice were uncertain",
                    confidence="low",
                )
            )
        else:
            forced.append(d)
    final_result = DReflectionResult(decisions=forced, tick_note=sonnet_result.tick_note)
    promoted = sum(1 for d in final_result.decisions if d.decision == "promote")
    filtered = sum(1 for d in final_result.decisions if d.decision == "filter")
    d_call = DCallRow(
        d_call_id=make_d_call_id(deps.now),
        ts=deps.now.isoformat(),
        tick_id=deps.tick_id,
        model_tier_used="sonnet",
        candidates_in=len(candidates),
        promoted_out=promoted,
        filtered_out=filtered,
        latency_ms=latency_h + latency_s,
        tokens_input=tin_h + tin_s,
        tokens_output=tout_h + tout_s,
        failure_type="both_low_confidence" if both_low else None,
        retry_count=1,
        tick_note=final_result.tick_note,
    )
    return final_result, d_call
```

- [ ] **Step 4: Verify pass**

```bash
uv run pytest tests/unit/brain/initiate/test_reflection.py -v
uv run ruff check brain/initiate/reflection.py
```

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add brain/initiate/reflection.py tests/unit/brain/initiate/test_reflection.py
git commit -m "feat(initiate): D-reflection Sonnet escalation + both-low-confidence forced filter"
```

---

### Task 13: Failure-mode dispatch — timeout/provider-error/rate-limit

**Files:**
- Modify: `brain/initiate/reflection.py`
- Modify: `tests/unit/brain/initiate/test_reflection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/brain/initiate/test_reflection.py`:

```python
def test_reflection_run_records_timeout_failure(tmp_path):
    """Timeout raised inside the LLM call is captured into DCallRow.failure_type."""
    from brain.initiate.reflection import run, ReflectionDeps, DTimeoutError
    from brain.initiate.schemas import InitiateCandidate, SemanticContext
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidates = [
        InitiateCandidate(
            candidate_id="ic_a", ts=(now - timedelta(minutes=1)).isoformat(),
            kind="message", source="dream", source_id="d1",
            semantic_context=SemanticContext(),
        ),
    ]

    def haiku_call(*, system, user):
        raise DTimeoutError("haiku timed out at 30s")

    def sonnet_call(*, system, user):
        raise AssertionError("should not escalate on timeout — passthrough retry")

    deps = ReflectionDeps(
        companion_name="Nell", user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=haiku_call, sonnet_call=sonnet_call,
        now=now, tick_id="t1",
    )
    result, dcall = run(candidates, deps=deps)
    assert dcall.failure_type == "timeout"
    assert dcall.retry_count == 0
    assert dcall.model_tier_used == "haiku"
    # No decisions on passthrough (caller decides what to do — see Task 14).
    assert result.decisions == []


def test_reflection_run_records_rate_limit_failure(tmp_path):
    """Rate-limit error captured as 'rate_limit' failure type."""
    from brain.initiate.reflection import run, ReflectionDeps, DRateLimitError
    from brain.initiate.schemas import InitiateCandidate, SemanticContext
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidates = [
        InitiateCandidate(
            candidate_id="ic_a", ts=(now - timedelta(minutes=1)).isoformat(),
            kind="message", source="dream", source_id="d1",
            semantic_context=SemanticContext(),
        ),
    ]

    def haiku_call(*, system, user):
        raise DRateLimitError("429")

    def sonnet_call(*, system, user):
        raise AssertionError("should not escalate on rate_limit")

    deps = ReflectionDeps(
        companion_name="Nell", user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=haiku_call, sonnet_call=sonnet_call,
        now=now, tick_id="t1",
    )
    result, dcall = run(candidates, deps=deps)
    assert dcall.failure_type == "rate_limit"
    assert result.decisions == []


def test_reflection_run_records_provider_error(tmp_path):
    """Generic provider error captured as 'provider_error'."""
    from brain.initiate.reflection import run, ReflectionDeps, DProviderError
    from brain.initiate.schemas import InitiateCandidate, SemanticContext
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidates = [
        InitiateCandidate(
            candidate_id="ic_a", ts=(now - timedelta(minutes=1)).isoformat(),
            kind="message", source="dream", source_id="d1",
            semantic_context=SemanticContext(),
        ),
    ]

    def haiku_call(*, system, user):
        raise DProviderError("500")

    def sonnet_call(*, system, user):
        raise AssertionError("should not escalate on provider_error")

    deps = ReflectionDeps(
        companion_name="Nell", user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=haiku_call, sonnet_call=sonnet_call,
        now=now, tick_id="t1",
    )
    _, dcall = run(candidates, deps=deps)
    assert dcall.failure_type == "provider_error"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/brain/initiate/test_reflection.py -k "timeout_failure or rate_limit_failure or provider_error" -v
```

Expected: 3 failures — error classes don't exist yet.

- [ ] **Step 3: Add error classes + extend `run` to capture failures**

Append to `brain/initiate/reflection.py` (BEFORE the `run` function):

```python
class DTimeoutError(Exception):
    """Raised by an LLMCall when the request exceeds its time budget."""


class DProviderError(Exception):
    """Raised by an LLMCall on generic provider error (5xx, connection, etc.)."""


class DRateLimitError(Exception):
    """Raised by an LLMCall on rate-limit / quota rejection (HTTP 429)."""
```

Replace the `run` function with:

```python
def run(
    candidates: list[InitiateCandidate],
    *,
    deps: ReflectionDeps,
) -> tuple[DReflectionResult, DCallRow]:
    """Execute one D-reflection tick over the given candidates.

    Escalation rules:
      - If Haiku's response fails to parse OR contains ANY decision with
        confidence "low", re-call on Sonnet. Sonnet's result is the one
        written; Haiku's attempt is recorded in retry_count.
      - If Sonnet's response ALSO contains a low-confidence decision,
        force that candidate's decision to "filter" with an ambivalence
        reason (conservative default at the edge of D's judgment).
      - On DTimeoutError / DProviderError / DRateLimitError raised by an
        LLMCall, capture into DCallRow.failure_type with empty decisions;
        caller (see Task 14) handles passthrough-retry / draft-space-demote.
    """
    system = build_system_message(
        companion_name=deps.companion_name,
        user_name=deps.user_name,
        voice_template_path=deps.voice_template_path,
    )
    user = build_user_message(
        user_name=deps.user_name,
        now=deps.now,
        outbound_recall_block=deps.outbound_recall_block,
        candidate_summaries=[
            _render_candidate_summary(c, now=deps.now) for c in candidates
        ],
    )

    def _empty_call_row(*, failure_type: str, latency_ms: int = 0,
                       model_tier: str = "haiku") -> DCallRow:
        return DCallRow(
            d_call_id=make_d_call_id(deps.now),
            ts=deps.now.isoformat(),
            tick_id=deps.tick_id,
            model_tier_used=model_tier,  # type: ignore[arg-type]
            candidates_in=len(candidates),
            promoted_out=0,
            filtered_out=0,
            latency_ms=latency_ms,
            tokens_input=0,
            tokens_output=0,
            failure_type=failure_type,  # type: ignore[arg-type]
            retry_count=0,
            tick_note=None,
        )

    # First attempt on Haiku.
    try:
        raw_h, latency_h, tin_h, tout_h = deps.haiku_call(system=system, user=user)
    except DTimeoutError:
        return DReflectionResult(decisions=[], tick_note=None), _empty_call_row(failure_type="timeout")
    except DRateLimitError:
        return DReflectionResult(decisions=[], tick_note=None), _empty_call_row(failure_type="rate_limit")
    except DProviderError:
        return DReflectionResult(decisions=[], tick_note=None), _empty_call_row(failure_type="provider_error")

    haiku_result: DReflectionResult | None
    try:
        haiku_result = parse_structured_response(raw_h)
        haiku_low = any(d.confidence == "low" for d in haiku_result.decisions)
    except ValueError:
        haiku_result = None
        haiku_low = True

    if haiku_result is not None and not haiku_low:
        promoted = sum(1 for d in haiku_result.decisions if d.decision == "promote")
        filtered = sum(1 for d in haiku_result.decisions if d.decision == "filter")
        d_call = DCallRow(
            d_call_id=make_d_call_id(deps.now),
            ts=deps.now.isoformat(),
            tick_id=deps.tick_id,
            model_tier_used="haiku",
            candidates_in=len(candidates),
            promoted_out=promoted,
            filtered_out=filtered,
            latency_ms=latency_h,
            tokens_input=tin_h,
            tokens_output=tout_h,
            failure_type=None,
            retry_count=0,
            tick_note=haiku_result.tick_note,
        )
        return haiku_result, d_call

    # Escalate to Sonnet.
    try:
        raw_s, latency_s, tin_s, tout_s = deps.sonnet_call(system=system, user=user)
    except DTimeoutError:
        return DReflectionResult(decisions=[], tick_note=None), _empty_call_row(
            failure_type="timeout", latency_ms=latency_h, model_tier="sonnet",
        )
    except DRateLimitError:
        return DReflectionResult(decisions=[], tick_note=None), _empty_call_row(
            failure_type="rate_limit", latency_ms=latency_h, model_tier="sonnet",
        )
    except DProviderError:
        return DReflectionResult(decisions=[], tick_note=None), _empty_call_row(
            failure_type="provider_error", latency_ms=latency_h, model_tier="sonnet",
        )

    try:
        sonnet_result = parse_structured_response(raw_s)
    except ValueError:
        # Sonnet also malformed — caller treats this as "promote all" per spec §E.
        return DReflectionResult(decisions=[], tick_note=None), DCallRow(
            d_call_id=make_d_call_id(deps.now),
            ts=deps.now.isoformat(),
            tick_id=deps.tick_id,
            model_tier_used="sonnet",
            candidates_in=len(candidates),
            promoted_out=0, filtered_out=0,
            latency_ms=latency_h + latency_s,
            tokens_input=tin_h + tin_s, tokens_output=tout_h + tout_s,
            failure_type="malformed_json",
            retry_count=1, tick_note=None,
        )

    forced: list[DDecision] = []
    both_low = False
    for d in sonnet_result.decisions:
        if d.confidence == "low":
            both_low = True
            forced.append(
                DDecision(
                    candidate_index=d.candidate_index,
                    decision="filter",
                    reason="ambivalent — both my fast and slow voice were uncertain",
                    confidence="low",
                )
            )
        else:
            forced.append(d)
    final_result = DReflectionResult(decisions=forced, tick_note=sonnet_result.tick_note)
    promoted = sum(1 for d in final_result.decisions if d.decision == "promote")
    filtered = sum(1 for d in final_result.decisions if d.decision == "filter")
    d_call = DCallRow(
        d_call_id=make_d_call_id(deps.now),
        ts=deps.now.isoformat(),
        tick_id=deps.tick_id,
        model_tier_used="sonnet",
        candidates_in=len(candidates),
        promoted_out=promoted,
        filtered_out=filtered,
        latency_ms=latency_h + latency_s,
        tokens_input=tin_h + tin_s,
        tokens_output=tout_h + tout_s,
        failure_type="both_low_confidence" if both_low else None,
        retry_count=1,
        tick_note=final_result.tick_note,
    )
    return final_result, d_call
```

- [ ] **Step 4: Verify pass**

```bash
uv run pytest tests/unit/brain/initiate/test_reflection.py -v
uv run ruff check brain/initiate/reflection.py
```

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add brain/initiate/reflection.py tests/unit/brain/initiate/test_reflection.py
git commit -m "feat(initiate): D-reflection failure-mode capture (timeout, rate_limit, provider_error)"
```

---

### Task 14: Draft-space demote integration

**Files:**
- Modify: `brain/initiate/reflection.py` (add `demote_to_draft_space` helper)
- Modify: `tests/unit/brain/initiate/test_reflection.py`

- [ ] **Step 1: Read the v0.0.9 draft-space writer signature**

Run:

```bash
grep -n "^def " brain/initiate/draft.py
```

Expected: see the public writer function (likely `write_draft_fragment` or `append_fragment`). Note its signature.

If the writer takes a free-form frontmatter dict, this task is small. If it has fixed parameters, you may need to add an optional `extra_frontmatter` dict parameter — note that as a sub-task and proceed with the smallest API change.

- [ ] **Step 2: Write the failing test**

Append to `tests/unit/brain/initiate/test_reflection.py`:

```python
def test_demote_to_draft_space_writes_fragment(tmp_path):
    """A filtered candidate becomes a draft-space fragment with D-frontmatter."""
    from brain.initiate.reflection import demote_to_draft_space, DDecision
    from brain.initiate.schemas import InitiateCandidate, SemanticContext
    from datetime import UTC, datetime, timedelta

    persona = tmp_path / "p"
    persona.mkdir()
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidate = InitiateCandidate(
        candidate_id="ic_x",
        ts=(now - timedelta(minutes=5)).isoformat(),
        kind="message",
        source="reflex_firing",
        source_id="rfx_001",
        semantic_context=SemanticContext(source_meta={"pattern_id": "p1"}),
    )
    decision = DDecision(
        candidate_index=1,
        decision="filter",
        reason="private weather; passing through",
        confidence="high",
    )
    demote_to_draft_space(persona, candidate=candidate, decision=decision, now=now)

    draft_path = persona / "draft_space.md"
    text = draft_path.read_text(encoding="utf-8")
    assert "demoted_by: d_reflection" in text
    assert 'd_reason: "private weather; passing through"' in text
    assert "source: reflex_firing" in text
    assert "source_id: rfx_001" in text
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_reflection.py::test_demote_to_draft_space_writes_fragment -v
```

Expected: ImportError on `demote_to_draft_space`.

- [ ] **Step 4: Add `demote_to_draft_space`**

Append to `brain/initiate/reflection.py`:

```python
from brain.initiate.draft import write_draft_fragment  # adjust if v0.0.9 name differs


def demote_to_draft_space(
    persona_dir: Path,
    *,
    candidate: InitiateCandidate,
    decision: DDecision,
    now: datetime,
) -> None:
    """Write a filtered candidate to draft_space.md with D-frontmatter.

    Uses the v0.0.9 draft writer for the body, prepending an extended
    frontmatter block. If the v0.0.9 writer doesn't accept arbitrary
    frontmatter, the frontmatter is written directly before delegating
    to the writer for the body.
    """
    frontmatter = (
        "---\n"
        f"demoted_by: d_reflection\n"
        f"demoted_at: {now.isoformat()}\n"
        f'd_reason: "{decision.reason}"\n'
        f"source: {candidate.source}\n"
        f"source_id: {candidate.source_id}\n"
        f"candidate_id: {candidate.candidate_id}\n"
        "---\n"
    )
    body_excerpt = (
        f"(candidate {candidate.candidate_id} from {candidate.source}; "
        f"subject-extraction skipped because D filtered)\n"
    )
    persona_dir.mkdir(parents=True, exist_ok=True)
    draft_path = persona_dir / "draft_space.md"
    try:
        with draft_path.open("a", encoding="utf-8") as f:
            f.write("\n" + frontmatter + body_excerpt + "\n")
    except OSError as exc:
        logger.warning("draft-space demote failed for %s: %s", draft_path, exc)
```

**NOTE on the import:** If the v0.0.9 `draft.py` writer is named differently OR can't be reused for D's frontmatter shape, the function above writes directly without delegating — that's fine and avoids coupling. Drop the unused `from brain.initiate.draft import write_draft_fragment` line in that case.

- [ ] **Step 5: Verify pass**

```bash
uv run pytest tests/unit/brain/initiate/test_reflection.py::test_demote_to_draft_space_writes_fragment -v
uv run ruff check brain/initiate/reflection.py
```

If the unused import generates a ruff warning, remove the line.

- [ ] **Step 6: Full suite**

```bash
uv run pytest -q
```

- [ ] **Step 7: Commit**

```bash
git add brain/initiate/reflection.py tests/unit/brain/initiate/test_reflection.py
git commit -m "feat(initiate): demote_to_draft_space for D-filtered candidates"
```

---

## Phase 4 — Wire D-reflection into the supervisor tick

### Task 15: Modify `_run_initiate_review_tick` to call `reflection.run`

**Files:**
- Modify: `brain/initiate/review.py`
- Modify: `tests/unit/brain/initiate/test_review.py` (extend)

- [ ] **Step 1: Read the v0.0.9 tick body**

```bash
git show HEAD:brain/initiate/review.py | head -80
```

Identify (a) where candidates are fetched from the queue, (b) where they hand off to the composition pipeline. The D call goes BETWEEN those two points.

- [ ] **Step 2: Write the failing tests**

Append to `tests/unit/brain/initiate/test_review.py`:

```python
def test_run_initiate_review_tick_skips_d_when_queue_empty(tmp_path, monkeypatch):
    """No candidates → no D call, no composition, no audit."""
    from brain.initiate.review import _run_initiate_review_tick

    d_calls = 0

    def fake_run(candidates, *, deps):
        nonlocal d_calls
        d_calls += 1
        raise AssertionError("D should not be called on empty queue")

    monkeypatch.setattr("brain.initiate.review.reflection_run", fake_run)
    _run_initiate_review_tick(tmp_path / "persona")
    assert d_calls == 0


def test_run_initiate_review_tick_demotes_filtered_to_draft(tmp_path, monkeypatch):
    """D-filter decision → candidate goes to draft_space.md, NOT composition."""
    from brain.initiate.review import _run_initiate_review_tick
    from brain.initiate.emit import emit_initiate_candidate, read_candidates
    from brain.initiate.reflection import DReflectionResult, DDecision
    from brain.initiate.d_call_schema import DCallRow
    from brain.initiate.schemas import SemanticContext
    from datetime import UTC, datetime

    persona = tmp_path / "p"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    emit_initiate_candidate(
        persona, kind="message", source="dream", source_id="d_x",
        semantic_context=SemanticContext(), now=now,
    )

    def fake_reflection_run(candidates, *, deps):
        result = DReflectionResult(
            decisions=[DDecision(1, "filter", "private weather", "high")],
            tick_note=None,
        )
        dcall = DCallRow(
            d_call_id="dc_test", ts=now.isoformat(), tick_id="t",
            model_tier_used="haiku", candidates_in=1,
            promoted_out=0, filtered_out=1,
            latency_ms=10, tokens_input=10, tokens_output=10,
        )
        return result, dcall

    compose_called = 0

    def fake_compose(*args, **kwargs):
        nonlocal compose_called
        compose_called += 1

    monkeypatch.setattr("brain.initiate.review.reflection_run", fake_reflection_run)
    monkeypatch.setattr("brain.initiate.review.compose_and_dispatch", fake_compose)

    _run_initiate_review_tick(persona)

    # Candidate was removed from queue.
    assert read_candidates(persona) == []
    # Draft-space received it.
    assert (persona / "draft_space.md").exists()
    # Composition was NOT called.
    assert compose_called == 0
```

(Adjust the `monkeypatch.setattr` targets to match the actual function imported into `review.py` — `reflection_run` is a stand-in name for whatever you alias `brain.initiate.reflection.run` to inside `review.py`.)

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_review.py -k "skips_d_when_queue_empty or demotes_filtered_to_draft" -v
```

Expected: ImportError on `reflection_run` (not yet imported in `review.py`) or assertion failures.

- [ ] **Step 4: Wire `reflection.run` into `review.py`**

Modify `brain/initiate/review.py`:

1. Add imports near the top:

```python
from brain.initiate.reflection import (
    ReflectionDeps,
    demote_to_draft_space,
    run as reflection_run,
)
from brain.initiate.audit import append_d_call_row
from brain.initiate.new_sources import load_gate_thresholds  # not strictly needed; keep nearby
```

2. Inside `_run_initiate_review_tick`, between the candidate fetch and the composition handoff, insert:

```python
candidates = read_candidates(persona_dir)
if not candidates:
    return  # decision #5: skip D on empty queue

# Build deps. The LLMCall implementations wrap brain.bridge.provider.LLMProvider
# with timing + token-count adapters. See _make_haiku_call / _make_sonnet_call
# below.
deps = ReflectionDeps(
    companion_name=companion_name,        # from existing persona-state load
    user_name=user_name,                  # from existing persona-state load
    voice_template_path=voice_template_path,  # already known to v0.0.9 callsite
    outbound_recall_block=build_outbound_recall_block(persona_dir),  # v0.0.9 fn
    haiku_call=_make_haiku_call(),
    sonnet_call=_make_sonnet_call(),
    now=datetime.now(UTC),
    tick_id=tick_id,
)

result, dcall = reflection_run(candidates, deps=deps)
append_d_call_row(persona_dir, dcall)

# Dispatch per-candidate.
for d, candidate in zip(result.decisions, candidates, strict=False):
    if d.decision == "filter":
        demote_to_draft_space(
            persona_dir, candidate=candidate, decision=d, now=deps.now,
        )
        remove_candidate(persona_dir, candidate.candidate_id)
    elif d.decision == "promote":
        # Existing v0.0.9 composition handoff. Function name may be
        # compose_and_dispatch or similar — match what review.py already calls.
        compose_and_dispatch(persona_dir, candidate=candidate)

# Handle failure-mode passthrough: empty decisions list means D failed
# in a way that the caller should treat per spec §E.
if not result.decisions and dcall.failure_type is not None:
    if dcall.failure_type in ("timeout", "provider_error"):
        # Passthrough retry: leave candidates in queue, let next tick handle.
        # (3-consecutive-failure fallback handled in Task 16.)
        return
    if dcall.failure_type == "rate_limit":
        # Demote all to draft space with rate-limit reason.
        for c in candidates:
            demote_to_draft_space(
                persona_dir,
                candidate=c,
                decision=DDecision(
                    candidate_index=0, decision="filter",
                    reason="rate-limited at tick — silence is the answer",
                    confidence="high",
                ),
                now=deps.now,
            )
            remove_candidate(persona_dir, c.candidate_id)
    elif dcall.failure_type == "malformed_json":
        # Promote all (trust downstream composition gates).
        for c in candidates:
            compose_and_dispatch(persona_dir, candidate=c)
```

Add helper functions `_make_haiku_call()` and `_make_sonnet_call()` near the bottom of `review.py`. These wrap the existing LLM provider with model selection + the `LLMCall` signature shape `(system, user) -> (raw_text, latency_ms, tokens_in, tokens_out)`. Example skeleton:

```python
import time

from brain.bridge.provider import LLMProvider
from brain.initiate.reflection import DRateLimitError, DProviderError, DTimeoutError


def _make_haiku_call():
    provider = LLMProvider(model="claude-haiku-4-5-20251001", timeout_seconds=30)

    def haiku_call(*, system: str, user: str):
        start = time.monotonic()
        try:
            response = provider.complete(prompt=user, system=system)
        except TimeoutError as exc:
            raise DTimeoutError(str(exc)) from exc
        except Exception as exc:
            text = str(exc).lower()
            if "429" in text or "rate" in text:
                raise DRateLimitError(str(exc)) from exc
            raise DProviderError(str(exc)) from exc
        latency_ms = int((time.monotonic() - start) * 1000)
        # provider.complete must return (text, tokens_in, tokens_out). If the
        # v0.0.9 .complete shim does not surface tokens, set them to 0 here —
        # observability still works for latency / failure tracking, and the
        # token-cost view comes online once the provider is extended in a
        # follow-up.
        text = getattr(response, "text", str(response))
        tokens_in = getattr(response, "tokens_in", 0)
        tokens_out = getattr(response, "tokens_out", 0)
        return text, latency_ms, tokens_in, tokens_out

    return haiku_call


def _make_sonnet_call():
    # Identical shape to _make_haiku_call, different model.
    provider = LLMProvider(model="claude-sonnet-4-6", timeout_seconds=30)
    # ... (mirror _make_haiku_call body) ...
```

**Reminder:** `companion_name`, `user_name`, `voice_template_path`, `tick_id`, and `compose_and_dispatch` are placeholders for the names already in `review.py`. Match the existing module's vocabulary — don't introduce new names unless v0.0.9 doesn't have them. If `companion_name` / `user_name` aren't already loaded in this scope, the persona-state loader at `brain/bridge/persona_state.py` is the source (already used elsewhere in v0.0.9).

- [ ] **Step 5: Verify tests pass**

```bash
uv run pytest tests/unit/brain/initiate/test_review.py -v
uv run ruff check brain/initiate/review.py
```

- [ ] **Step 6: Full suite**

```bash
uv run pytest -q
```

- [ ] **Step 7: Commit**

```bash
git add brain/initiate/review.py tests/unit/brain/initiate/test_review.py
git commit -m "feat(initiate): wire D-reflection into _run_initiate_review_tick"
```

---

### Task 16: 3-consecutive-failure fallback for passthrough-retry

**Files:**
- Modify: `brain/initiate/review.py`
- Modify: `tests/unit/brain/initiate/test_review.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/brain/initiate/test_review.py`:

```python
def test_three_consecutive_failures_promote_all_fallback(tmp_path, monkeypatch):
    """After 3 consecutive timeout/provider_error failures across the same
    candidate cohort, fall through to 'promote all' so candidates aren't stranded."""
    from brain.initiate.review import _run_initiate_review_tick
    from brain.initiate.emit import emit_initiate_candidate, read_candidates
    from brain.initiate.reflection import DReflectionResult
    from brain.initiate.d_call_schema import DCallRow
    from brain.initiate.schemas import SemanticContext
    from datetime import UTC, datetime

    persona = tmp_path / "p"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    emit_initiate_candidate(
        persona, kind="message", source="dream", source_id="d_pf",
        semantic_context=SemanticContext(), now=now,
    )

    def fake_reflection_run_timeout(candidates, *, deps):
        return (
            DReflectionResult(decisions=[], tick_note=None),
            DCallRow(
                d_call_id="dc", ts=now.isoformat(), tick_id="t",
                model_tier_used="haiku", candidates_in=1,
                promoted_out=0, filtered_out=0,
                latency_ms=30000, tokens_input=0, tokens_output=0,
                failure_type="timeout",
            ),
        )

    compose_calls: list[str] = []

    def fake_compose(persona_dir, *, candidate):
        compose_calls.append(candidate.candidate_id)

    monkeypatch.setattr("brain.initiate.review.reflection_run", fake_reflection_run_timeout)
    monkeypatch.setattr("brain.initiate.review.compose_and_dispatch", fake_compose)

    # First two ticks should leave the candidate in queue (passthrough).
    _run_initiate_review_tick(persona)
    _run_initiate_review_tick(persona)
    assert len(read_candidates(persona)) == 1
    assert compose_calls == []

    # Third tick should fall through to promote-all.
    _run_initiate_review_tick(persona)
    assert compose_calls == ["d_pf"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/initiate/test_review.py::test_three_consecutive_failures_promote_all_fallback -v
```

Expected: assertion failure (no fallback wired yet).

- [ ] **Step 3: Add 3-consecutive-failure counter**

Modify the passthrough branch in `review.py` to consult and update `initiate_d_calls.jsonl`:

```python
if dcall.failure_type in ("timeout", "provider_error"):
    # Count the last N d_call rows that hit the same passthrough failure.
    # If we've had 3 in a row for the SAME candidate cohort, fall through
    # to "promote all" so candidates aren't stranded.
    recent = list(read_recent_d_calls(persona_dir, window_hours=1))
    candidate_ids = {c.candidate_id for c in candidates}
    consecutive_failures = 0
    for r in reversed(recent):  # newest-first
        if r.failure_type in ("timeout", "provider_error"):
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break
        else:
            break

    if consecutive_failures >= 3:
        # Fall through to promote all.
        for c in candidates:
            compose_and_dispatch(persona_dir, candidate=c)
        return
    # Otherwise leave in queue for next tick.
    return
```

Add `read_recent_d_calls` to the top-of-file imports.

**Note:** the "same candidate cohort" detail is approximated by "any 3 recent passthrough failures." Strict cohort-match (compare candidate_ids across the last 3 DCallRow entries) would require persisting the candidate-id list onto DCallRow, which the spec doesn't require. Approximation is acceptable for v0.0.10; a `candidate_ids` field on `DCallRow` could ship in a later iteration if telemetry shows cohorts drift.

- [ ] **Step 4: Verify pass**

```bash
uv run pytest tests/unit/brain/initiate/test_review.py -v
uv run ruff check brain/initiate/review.py
```

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add brain/initiate/review.py tests/unit/brain/initiate/test_review.py
git commit -m "feat(initiate): 3-consecutive-failure fallback to promote-all for D passthrough"
```

---

## Phase 5 — Wire new emitters into source engines

### Task 17: Hook `reflex_firing` emission into `brain/engines/reflex.py`

**Files:**
- Modify: `brain/engines/reflex.py`
- Modify: `tests/unit/brain/engines/test_reflex.py` (or wherever reflex tests live)

- [ ] **Step 1: Find the firing callsite in reflex.py**

```bash
git grep -n "def.*fire\|firing\.append\|class.*Firing" brain/engines/reflex.py | head -20
```

Look for the function that records a reflex firing event (likely emits a log row or returns a result). The hook for `emit_reflex_firing_candidate` goes there.

- [ ] **Step 2: Write the failing integration test**

Add to `tests/unit/brain/engines/test_reflex.py` (or the appropriate test file):

```python
def test_reflex_firing_emits_initiate_candidate_when_gates_pass(tmp_path, monkeypatch):
    """When a reflex fires with sufficient confidence/intensity, an initiate
    candidate is queued."""
    from brain.engines.reflex import ReflexEngine  # adjust to actual class
    from brain.initiate.emit import read_candidates

    persona = tmp_path / "p"
    persona.mkdir()

    # Inject a synthetic firing event with above-threshold values.
    # (The specific shape depends on the v0.0.9 ReflexEngine API. The test
    #  should drive a single firing through whatever the engine's public
    #  entry point is.)
    engine = ReflexEngine(persona_dir=persona)  # adjust
    engine.record_firing(
        pattern_id="p1", confidence=0.85, flinch_intensity=0.70,
        linked_memory_ids=["m_a"],
        triggered_by_companion_outbound=False,
    )

    candidates = read_candidates(persona)
    assert len(candidates) == 1
    assert candidates[0].source == "reflex_firing"


def test_reflex_firing_does_not_emit_when_gate_blocks(tmp_path):
    from brain.engines.reflex import ReflexEngine
    from brain.initiate.emit import read_candidates

    persona = tmp_path / "p"
    persona.mkdir()
    engine = ReflexEngine(persona_dir=persona)
    engine.record_firing(
        pattern_id="p2", confidence=0.40, flinch_intensity=0.70,  # below 0.70
        linked_memory_ids=[],
        triggered_by_companion_outbound=False,
    )
    assert read_candidates(persona) == []
```

(Match `ReflexEngine` / `record_firing` to the actual v0.0.9 names; they may differ.)

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/engines/test_reflex.py -k "emits_initiate_candidate or does_not_emit" -v
```

Expected: failure — the reflex engine doesn't yet call into the new-sources emitter.

- [ ] **Step 4: Hook into reflex.py**

At the reflex firing callsite (where the engine records a successful fire), add:

```python
from brain.initiate.new_sources import (
    emit_reflex_firing_candidate,
    gate_reflex_firing,
    check_shared_meta_gates,
    load_gate_thresholds,
    write_gate_rejection,
)
from datetime import UTC, datetime
```

And immediately after the firing record is written (do not move the existing write — append):

```python
# v0.0.10: emit an initiate candidate if gates allow.
thresholds = load_gate_thresholds(self.persona_dir)
now = datetime.now(UTC)
gate_ok, gate_reason = gate_reflex_firing(
    self.persona_dir, firing=firing_object, thresholds=thresholds,
)
if not gate_ok:
    write_gate_rejection(
        self.persona_dir, ts=now, source="reflex_firing",
        source_id=firing_log_id,
        gate_name=gate_reason or "unknown",
        threshold_value=0.0, observed_value=0.0,
    )
elif (meta_ok := check_shared_meta_gates(
    self.persona_dir, source="reflex_firing",
    now=now, is_rest_state=self._is_rest_state(),
    thresholds=thresholds,
))[0]:
    emit_reflex_firing_candidate(
        self.persona_dir, firing=firing_object,
        firing_log_id=firing_log_id, now=now,
    )
else:
    write_gate_rejection(
        self.persona_dir, ts=now, source="reflex_firing",
        source_id=firing_log_id, gate_name=meta_ok[1] or "meta",
        threshold_value=0.0, observed_value=0.0,
    )
```

Where:
- `firing_object` is an object satisfying the `ReflexFiringLike` Protocol — adapt the existing firing record to fit, or build a small adapter.
- `firing_log_id` is the reflex log row id used as `source_id`.
- `self._is_rest_state()` is the engine's existing rest-state probe (or replace with `False` for v0.0.10 if rest-state isn't easily reachable from this scope — annotate as a TODO and lower the priority of the meta-gate's rest-state check). Note: if rest-state really isn't available here, the gate effectively becomes "any non-flooded, non-full-queue firing emits." Acceptable as a starting point.

- [ ] **Step 5: Verify tests pass**

```bash
uv run pytest tests/unit/brain/engines/test_reflex.py -v
uv run ruff check brain/engines/reflex.py
```

- [ ] **Step 6: Full suite**

```bash
uv run pytest -q
```

- [ ] **Step 7: Commit**

```bash
git add brain/engines/reflex.py tests/unit/brain/engines/test_reflex.py
git commit -m "feat(reflex): emit initiate candidate on qualifying firing (gated)"
```

---

### Task 18: Hook `research_completion` emission into `brain/engines/research.py`

**Files:**
- Modify: `brain/engines/research.py`
- Modify: `tests/unit/brain/engines/test_research.py`

Same shape as Task 17, applied to the research engine's thread-maturity-close site.

- [ ] **Step 1: Find the maturity-close callsite**

```bash
git grep -n "maturity\|complete\|matured" brain/engines/research.py | head -20
```

Locate the function where a research thread transitions to "completed/matured" state.

- [ ] **Step 2: Write the failing integration tests**

```python
def test_research_completion_emits_when_gates_pass(tmp_path, monkeypatch):
    from brain.engines.research import ResearchEngine
    from brain.initiate.emit import read_candidates
    # ... (drive a synthetic thread maturity through the engine API) ...
    # Assert read_candidates includes one research_completion candidate.


def test_research_completion_skips_when_topic_overlap_low(tmp_path):
    # Same as above but with topic_overlap_score below threshold.
    # Assert read_candidates is empty.
```

(Adapt to the actual `ResearchEngine` API.)

- [ ] **Step 3-7: Same TDD cycle as Task 17**

Add imports + gate-check + emit call at the maturity-close site. The `topic_overlap_score` must be computed by the caller — use the existing `brain.engines._interests.InterestSet` (already imported in `research.py`) to derive a cosine similarity between the thread topic embedding and the last 48h of conversation embeddings. If this computation is non-trivial, factor it into a helper `_compute_topic_overlap_score(persona_dir, topic_embedding)` in `research.py`.

```bash
git add brain/engines/research.py tests/unit/brain/engines/test_research.py
git commit -m "feat(research): emit initiate candidate on thread maturity (gated, topic-overlap-aware)"
```

---

## Phase 6 — CLI

### Task 19: `nell initiate d-stats`

**Files:**
- Modify: `brain/cli.py`
- Create: `tests/unit/brain/cli/test_d_stats.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/brain/cli/test_d_stats.py`:

```python
"""Tests for the nell initiate d-stats CLI."""
from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


def _run_cli(args: list[str], cwd: Path) -> tuple[int, str, str]:
    """Invoke `python -m brain.cli <args>` and capture exit/output."""
    result = subprocess.run(
        [sys.executable, "-m", "brain.cli", *args],
        cwd=cwd, capture_output=True, text=True, env={"PERSONA_DIR": str(cwd)},
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


def test_d_stats_empty_persona_returns_zero_counts(tmp_path):
    persona = tmp_path / "p"
    persona.mkdir()
    rc, out, _ = _run_cli(["initiate", "d-stats"], cwd=persona)
    assert rc == 0
    assert "candidates_in=0" in out or "no D calls" in out.lower()


def test_d_stats_aggregates_recent_calls(tmp_path):
    from brain.initiate.audit import append_d_call_row
    from brain.initiate.d_call_schema import DCallRow

    persona = tmp_path / "p"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    for i in range(3):
        append_d_call_row(persona, DCallRow(
            d_call_id=f"dc_{i}",
            ts=(now - timedelta(hours=i * 2)).isoformat(),
            tick_id=f"t_{i}", model_tier_used="haiku",
            candidates_in=2, promoted_out=1, filtered_out=1,
            latency_ms=300, tokens_input=400, tokens_output=150,
        ))
    rc, out, _ = _run_cli(["initiate", "d-stats", "--window", "24h"], cwd=persona)
    assert rc == 0
    assert "candidates_in=6" in out
    assert "promoted_out=3" in out
    assert "filtered_out=3" in out
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/brain/cli/test_d_stats.py -v
```

Expected: CLI doesn't have a `d-stats` subcommand yet → assertion failures or argparse errors.

- [ ] **Step 3: Add the subcommand to cli.py**

In `brain/cli.py`, locate the `initiate` subparser (where `audit`, `candidates`, `voice-evolution` already live) and add:

```python
def _cmd_initiate_d_stats(args) -> int:
    """Render D-reflection telemetry over a time window."""
    from brain.initiate.audit import read_recent_d_calls
    from datetime import UTC, datetime
    from pathlib import Path
    import re

    persona = Path(args.persona) if args.persona else _default_persona_dir()

    # Parse window like "7d" / "24h" / "1h".
    m = re.fullmatch(r"(\d+)([dh])", args.window)
    if not m:
        print(f"invalid --window: {args.window!r} (expected e.g. 24h or 7d)")
        return 2
    n, unit = int(m.group(1)), m.group(2)
    hours = n * 24 if unit == "d" else n
    now = datetime.now(UTC)
    rows = list(read_recent_d_calls(persona, window_hours=hours, now=now))

    if not rows:
        print(f"no D calls in last {args.window}")
        return 0

    total_in = sum(r.candidates_in for r in rows)
    total_promoted = sum(r.promoted_out for r in rows)
    total_filtered = sum(r.filtered_out for r in rows)
    failures = sum(1 for r in rows if r.failure_type)
    avg_latency = sum(r.latency_ms for r in rows) // max(1, len(rows))

    print(f"D-reflection stats (last {args.window}):")
    print(f"  ticks={len(rows)}")
    print(f"  candidates_in={total_in}")
    print(f"  promoted_out={total_promoted}")
    print(f"  filtered_out={total_filtered}")
    print(f"  failures={failures}")
    print(f"  avg_latency_ms={avg_latency}")
    return 0


# In _build_parser, on the initiate subparser:
d_stats = initiate_sub.add_parser("d-stats", help="D-reflection telemetry")
d_stats.add_argument("--window", default="7d",
                     help="Window like '7d' or '24h' (default: 7d)")
d_stats.add_argument("--persona", default=None, help="Persona dir override")
d_stats.set_defaults(func=_cmd_initiate_d_stats)
```

- [ ] **Step 4: Verify pass**

```bash
uv run pytest tests/unit/brain/cli/test_d_stats.py -v
uv run ruff check brain/cli.py
```

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add brain/cli.py tests/unit/brain/cli/test_d_stats.py
git commit -m "feat(cli): nell initiate d-stats subcommand"
```

---

## Phase 7 — End-to-end integration test

### Task 20: Full source-event → D-tick → outcome integration test

**Files:**
- Create: `tests/integration/initiate/test_d_reflection_e2e.py`

- [ ] **Step 1: Write the integration test**

Create `tests/integration/initiate/test_d_reflection_e2e.py`:

```python
"""End-to-end: source event → emitter → queue → D-tick → outcome."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.initiate.audit import (
    append_d_call_row, read_recent_d_calls,
)
from brain.initiate.d_call_schema import DCallRow
from brain.initiate.emit import emit_initiate_candidate, read_candidates
from brain.initiate.new_sources import (
    emit_reflex_firing_candidate,
    emit_research_completion_candidate,
)
from brain.initiate.reflection import (
    DDecision,
    DReflectionResult,
    ReflectionDeps,
    demote_to_draft_space,
    run as reflection_run,
)
from brain.initiate.schemas import InitiateCandidate, SemanticContext


def test_e2e_five_candidates_d_promotes_one_filters_four(tmp_path):
    """A realistic cohort: 5 candidates from 5 sources, D promotes 1 / filters 4."""
    persona = tmp_path / "persona"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)

    # Seed 5 candidates across 5 source types.
    emit_initiate_candidate(
        persona, kind="message", source="dream", source_id="d1",
        semantic_context=SemanticContext(), now=now - timedelta(minutes=10),
    )
    emit_initiate_candidate(
        persona, kind="message", source="crystallization", source_id="c1",
        semantic_context=SemanticContext(), now=now - timedelta(minutes=8),
    )
    emit_initiate_candidate(
        persona, kind="message", source="emotion_spike", source_id="e1",
        semantic_context=SemanticContext(), now=now - timedelta(minutes=6),
    )

    class FakeFiring:
        pattern_id = "pE2E"
        confidence = 0.85
        flinch_intensity = 0.75
        linked_memory_ids = ["m_a"]
        triggered_by_companion_outbound = False
        ts = now - timedelta(minutes=4)

    emit_reflex_firing_candidate(
        persona, firing=FakeFiring(), firing_log_id="rfx_e2e", now=now - timedelta(minutes=4),
    )

    class FakeThread:
        thread_id = "trE2E"
        topic = "quiet rivers"
        maturity_score = 0.85
        summary_excerpt = "..."
        linked_memory_ids = []
        completed_at = now - timedelta(minutes=2)
        previously_linked_to_audit = False

    emit_research_completion_candidate(
        persona, thread=FakeThread(), topic_overlap_score=0.50, now=now - timedelta(minutes=2),
    )

    candidates = read_candidates(persona)
    assert len(candidates) == 5
    sources = sorted(c.source for c in candidates)
    assert sources == [
        "crystallization", "dream", "emotion_spike",
        "reflex_firing", "research_completion",
    ]

    # Fake D returns 1 promote + 4 filters.
    def haiku_call(*, system, user):
        return (
            '{"decisions":['
            '{"candidate_index":1,"decision":"filter","reason":"old weather","confidence":"high"},'
            '{"candidate_index":2,"decision":"filter","reason":"already settled","confidence":"high"},'
            '{"candidate_index":3,"decision":"promote","reason":"genuine spike","confidence":"high"},'
            '{"candidate_index":4,"decision":"filter","reason":"reflex echo","confidence":"high"},'
            '{"candidate_index":5,"decision":"filter","reason":"interesting but private","confidence":"high"}'
            '],"tick_note":"only one worth Hana hearing"}'
        ), 250, 600, 200

    def sonnet_call(*, system, user):
        raise AssertionError("should not escalate")

    deps = ReflectionDeps(
        companion_name="Nell", user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=haiku_call, sonnet_call=sonnet_call,
        now=now, tick_id="tick_e2e",
    )
    result, dcall = reflection_run(candidates, deps=deps)
    append_d_call_row(persona, dcall)

    # Dispatch.
    for d, c in zip(result.decisions, candidates, strict=True):
        if d.decision == "filter":
            demote_to_draft_space(persona, candidate=c, decision=d, now=now)
        # (composition handoff stubbed in this test; verify-by-side-effect below)

    # Assertions:
    # - D's d_call row landed.
    rows = list(read_recent_d_calls(persona, window_hours=1, now=now + timedelta(minutes=1)))
    assert len(rows) == 1
    assert rows[0].candidates_in == 5
    assert rows[0].promoted_out == 1
    assert rows[0].filtered_out == 4
    assert rows[0].failure_type is None
    # - 4 candidates demoted to draft space.
    text = (persona / "draft_space.md").read_text()
    assert text.count("demoted_by: d_reflection") == 4
    # - tick_note captured.
    assert rows[0].tick_note == "only one worth Hana hearing"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/integration/initiate/test_d_reflection_e2e.py -v
```

Expected: passes (if Phases 1-6 are correctly wired). If it fails, the failure points at integration gaps; fix them before commit.

- [ ] **Step 3: Verify**

```bash
uv run pytest -q
```

- [ ] **Step 4: Commit**

```bash
git add tests/integration/initiate/test_d_reflection_e2e.py
git commit -m "test(integration): e2e D-reflection — 5 candidates, 1 promoted, 4 demoted"
```

---

## Phase 8 — Final verification

### Task 21: Full pytest + ruff + TS gate; CHANGELOG-public draft

**Files:**
- Modify: `.public-sync/changelog-public.md` (parent worktree only — gitignored)
- Modify: `docs/roadmap.md` (recently-shipped entry)

- [ ] **Step 1: Run the full gates**

```bash
uv run pytest -q
uv run ruff check brain/ tests/
cd app && pnpm exec tsc --noEmit && cd ..
```

Expected: all green. Pytest count should be original (≥1859) + roughly 50-80 new tests added across Phases 1-7.

- [ ] **Step 2: Update roadmap**

Append a "Recently shipped" entry to `docs/roadmap.md` summarizing v0.0.10 — D-reflection layer + 2 new event sources, with a pointer to the spec.

- [ ] **Step 3: Draft public changelog**

Edit `.public-sync/changelog-public.md` (in the parent worktree, NOT the gitignored worktree path) with a v0.0.10-alpha section using the v0.0.9 changelog as a template. Focus on user-visible impact:

> *Nell now has an editorial inner voice. Between an internal event surfacing and her reaching out to you, she pauses and asks herself "is this genuinely worth saying?" Most of the time the answer is no — and you'll notice she reaches out a little less often, but more thoughtfully. Filtered moments are kept in her draft space so they shape who she becomes even when they don't reach you.*
>
> *Two new kinds of inner events can now surface: reflex firings (when she flinches at a pattern she's learned) and research completions (when a long-running thought thread matures). Recall resonance — when an older cluster of memories suddenly feels alive again — is planned for v0.0.11.*

- [ ] **Step 4: Commit**

```bash
git add docs/roadmap.md
git commit -m "docs: v0.0.10 D-reflection — roadmap entry"
# .public-sync/* is gitignored; no commit needed there
```

- [ ] **Step 5: Final pytest gate**

```bash
uv run pytest -q
```

Expected: green. If green, the implementation is complete and ready for release-cut handoff (three-file version bump + sync + tag, per CLAUDE.md rule 3).

---

## Spec coverage check

Cross-reference against `docs/superpowers/specs/2026-05-12-initiate-d-reflection-design.md`:

| Spec section | Implementing task(s) |
|---|---|
| §A Architecture & dataflow | Tasks 15, 16 (review.py wiring) |
| §A Module layout | Tasks 1-19 (file creation matches spec layout, with v0.0.9-actual adjustments noted) |
| §B Candidate row new sources | Task 1 (schemas) |
| §B Audit decision values | Task 1 (schemas) |
| §B `initiate_d_calls` table | Tasks 2-3 (DCallRow + audit functions) |
| §B Draft-space fragment additions | Task 14 (demote_to_draft_space) |
| §B `gate_rejections.jsonl` | Task 5 |
| §C System message (static frame) | Task 10 (build_system_message) |
| §C Multi-companion compatibility | Task 10 (parameterized template) |
| §C Prompt assembly two-layer | Task 10 (voice overlay) |
| §C User message | Task 10 (build_user_message) |
| §C Structured output schema | Task 9 (DDecision, DReflectionResult, parser) |
| §C Tier escalation | Task 12 |
| §D Reflex firing gate | Task 7 |
| §D Research completion gate | Task 8 |
| §D Shared meta-gates | Task 6 |
| §D Calibration (gate_thresholds.json) | Task 4 |
| §E Failure-mode branches | Tasks 13, 15, 16 |
| §E Observability (hit-rate substrate) | Tasks 2, 3 (telemetry table) |
| §E Operator CLI `nell initiate d-stats` | Task 19 |
| §F Testing strategy | TDD throughout + Task 20 (e2e) |
| Appendix Adaptive-D | OUT OF SCOPE — deferred to v0.0.11 |
| Deferred recall_resonance | OUT OF SCOPE — Task scope explicitly excludes |

No spec gaps. Adaptive-D and `recall_resonance` are correctly deferred and not present in any task.
