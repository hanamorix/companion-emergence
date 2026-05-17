# Phase 2a — Vocabulary Emergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the autonomous self-development *architecture* for the brain, integrated into the emotion-vocabulary engine first. Phase 2a ships the package, scheduler, atomic apply path, heartbeat integration, and read-only CLI inspection. The vocabulary crystallizer is a no-op stub returning `[]` — pattern-matchers land in Phase 2b once we have ≥2 weeks of behavior data.

**Architecture:** New `brain/growth/` package with `proposal.py` (frozen dataclass), `log.py` (atomic append-only JSONL biography), `scheduler.py` (orchestrator that calls crystallizers + applies decisions atomically), and `crystallizers/vocabulary.py` (stub for 2a). Heartbeat gains a `_try_run_growth` step rate-limited by `growth_every_hours` (default 168 = weekly). Brand-new `nell growth log` CLI surfaces the biography read-only.

**Tech Stack:** Python 3.12, dataclasses, JSON file I/O with atomic `.new + os.replace`, pytest. No new deps.

**Spec:** [docs/superpowers/specs/2026-04-25-phase-2a-vocabulary-emergence-design.md](../specs/2026-04-25-phase-2a-vocabulary-emergence-design.md)

---

## File Structure

**Created:**

| File | Responsibility |
|------|---------------|
| `brain/growth/__init__.py` | Package marker. |
| `brain/growth/proposal.py` | `EmotionProposal` frozen dataclass — what a crystallizer returns. |
| `brain/growth/log.py` | `GrowthLogEvent` + `append_growth_event` (atomic) + `read_growth_log` (oldest-first). |
| `brain/growth/scheduler.py` | `GrowthTickResult` + `run_growth_tick(persona_dir, store, now, dry_run)` — orchestrates + applies. |
| `brain/growth/crystallizers/__init__.py` | Package marker. |
| `brain/growth/crystallizers/vocabulary.py` | `crystallize_vocabulary(store, *, current_vocabulary_names) -> []` (Phase 2a stub). |
| `tests/unit/brain/growth/__init__.py` | Test package marker. |
| `tests/unit/brain/growth/test_proposal.py` | EmotionProposal dataclass tests. |
| `tests/unit/brain/growth/test_log.py` | Growth log atomic-append + read tests. |
| `tests/unit/brain/growth/test_scheduler.py` | Scheduler tests with injected fake crystallizer. |
| `tests/unit/brain/growth/test_vocabulary_crystallizer.py` | Stub crystallizer tests (returns []). |
| `tests/unit/brain/growth/test_cli_growth.py` | `nell growth log` CLI tests. |

**Modified:**

| File | Change |
|------|--------|
| `brain/engines/heartbeat.py` | Adds `growth_enabled` + `growth_every_hours` to `HeartbeatConfig`; adds `last_growth_at` to `HeartbeatState`; adds `growth_emotions_added` to `HeartbeatResult`; adds `_try_run_growth` method; calls it from `run_tick` between research and heartbeat-memory; audit log gains `growth` sub-object. |
| `brain/cli.py` | Adds `nell growth log --persona X [--limit N] [--type T]` read-only subcommand. |

---

## Task Decomposition

The 8 tasks are sequenced. Tests-first throughout. Each commit is a complete unit (red → green → commit). The vocabulary crystallizer stub (T3) and heartbeat additions (T5) are independent of each other; the scheduler (T4) depends on T1+T2+T3; the heartbeat wiring (T6) depends on T4+T5.

---

### Task 1: `EmotionProposal` dataclass

**Files:**
- Create: `brain/growth/__init__.py`
- Create: `brain/growth/proposal.py`
- Create: `tests/unit/brain/growth/__init__.py`
- Create: `tests/unit/brain/growth/test_proposal.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/brain/growth/test_proposal.py
"""Tests for brain.growth.proposal — EmotionProposal frozen dataclass."""

from __future__ import annotations

import pytest

from brain.growth.proposal import EmotionProposal


def test_proposal_construction() -> None:
    """All fields land where they should."""
    p = EmotionProposal(
        name="lingering",
        description="the slow trail of warmth after a loved person leaves the room",
        decay_half_life_days=7.0,
        evidence_memory_ids=("mem_a", "mem_b"),
        score=0.78,
        relational_context="recurred during Hana's tender messages",
    )
    assert p.name == "lingering"
    assert p.decay_half_life_days == 7.0
    assert p.evidence_memory_ids == ("mem_a", "mem_b")
    assert p.score == 0.78
    assert p.relational_context == "recurred during Hana's tender messages"


def test_proposal_is_frozen() -> None:
    """EmotionProposal is immutable — crystallizer's decision can't be mutated downstream."""
    p = EmotionProposal(
        name="x",
        description="y",
        decay_half_life_days=None,
        evidence_memory_ids=(),
        score=0.5,
        relational_context=None,
    )
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        p.name = "mutated"  # type: ignore[misc]


def test_proposal_decay_can_be_none() -> None:
    """Identity-level emotions (love, belonging) have no temporal decay."""
    p = EmotionProposal(
        name="anchor_pull",
        description="the gravity toward someone you've decided is yours",
        decay_half_life_days=None,
        evidence_memory_ids=(),
        score=0.9,
        relational_context=None,
    )
    assert p.decay_half_life_days is None


def test_proposal_relational_context_can_be_none() -> None:
    """A proposal driven by purely internal reflection has no relational context."""
    p = EmotionProposal(
        name="quiet_pride",
        description="satisfaction in a long pattern recognized in oneself",
        decay_half_life_days=14.0,
        evidence_memory_ids=("mem_x",),
        score=0.7,
        relational_context=None,
    )
    assert p.relational_context is None
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `uv run pytest tests/unit/brain/growth/test_proposal.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain.growth'`

- [ ] **Step 3: Write the minimal implementation**

```python
# brain/growth/__init__.py
"""brain.growth — autonomous self-development architecture (Phase 2a).

See docs/superpowers/specs/2026-04-25-phase-2a-vocabulary-emergence-design.md.

Phase 2a ships the architecture (scheduler + log + atomic apply path) and a
no-op vocabulary crystallizer. Phase 2b populates the crystallizer with
real pattern-matching against memory + relational dynamics.
"""
```

```python
# brain/growth/proposal.py
"""EmotionProposal — what a vocabulary crystallizer returns.

A proposal is the brain's *decision* to add an emotion. The scheduler
applies it atomically — there's no candidate queue, no human approval
gate. Per principle audit 2026-04-25: the brain has agency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionProposal:
    """One emotion the crystallizer has decided to add to the vocabulary.

    Phase 2b's crystallizer fills these in based on memory pattern +
    relational dynamics analysis. Phase 2a's stub returns [] — never
    constructs these — but the type exists so the scheduler can be
    written and tested with injected fakes.

    Attributes:
        name: Canonical identifier (lowercase, underscore-separated).
        description: Human-readable meaning.
        decay_half_life_days: Time for intensity to halve. None = identity-level.
        evidence_memory_ids: Memories that drove the proposal. May be empty.
        score: Cluster coherence in [0.0, 1.0].
        relational_context: Short string describing the relational dynamic
            that drove the proposal, or None for purely internal-reflection
            proposals.
    """

    name: str
    description: str
    decay_half_life_days: float | None
    evidence_memory_ids: tuple[str, ...]
    score: float
    relational_context: str | None
```

```python
# tests/unit/brain/growth/__init__.py
```
(Empty file.)

- [ ] **Step 4: Run tests — verify they pass**

Run: `uv run pytest tests/unit/brain/growth/test_proposal.py -v`
Expected: PASS — 4/4 tests green.

- [ ] **Step 5: Commit**

```bash
git add brain/growth/__init__.py brain/growth/proposal.py tests/unit/brain/growth/
git commit -m "feat(growth): add EmotionProposal dataclass — Phase 2a T1"
```

---

### Task 2: `GrowthLogEvent` + `append_growth_event` + `read_growth_log`

**Files:**
- Create: `brain/growth/log.py`
- Create: `tests/unit/brain/growth/test_log.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/brain/growth/test_log.py
"""Tests for brain.growth.log — append-only biography of brain growth."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.growth.log import (
    GrowthLogEvent,
    append_growth_event,
    read_growth_log,
)


def _event(name: str = "lingering", **overrides) -> GrowthLogEvent:
    base = {
        "timestamp": datetime(2026, 4, 25, 18, 30, tzinfo=UTC),
        "type": "emotion_added",
        "name": name,
        "description": "test description",
        "decay_half_life_days": 7.0,
        "reason": "test reason",
        "evidence_memory_ids": ("mem_a", "mem_b"),
        "score": 0.78,
        "relational_context": "test relational",
    }
    base.update(overrides)
    return GrowthLogEvent(**base)  # type: ignore[arg-type]


def test_growth_log_event_is_frozen() -> None:
    e = _event()
    with pytest.raises(Exception):
        e.name = "mutated"  # type: ignore[misc]


def test_append_creates_log_file_when_missing(tmp_path: Path) -> None:
    log_path = tmp_path / "growth.log.jsonl"
    append_growth_event(log_path, _event())
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["name"] == "lingering"
    assert parsed["type"] == "emotion_added"


def test_append_is_append_only_across_calls(tmp_path: Path) -> None:
    log_path = tmp_path / "growth.log.jsonl"
    append_growth_event(log_path, _event(name="first"))
    append_growth_event(log_path, _event(name="second"))
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["name"] == "first"
    assert json.loads(lines[1])["name"] == "second"


def test_append_writes_iso_utc_timestamp(tmp_path: Path) -> None:
    log_path = tmp_path / "growth.log.jsonl"
    append_growth_event(log_path, _event())
    parsed = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert parsed["timestamp"].endswith("Z")  # tz-aware UTC ISO format


def test_append_serializes_evidence_ids_as_list(tmp_path: Path) -> None:
    """evidence_memory_ids is a tuple in Python; JSON must serialize as list."""
    log_path = tmp_path / "growth.log.jsonl"
    append_growth_event(log_path, _event())
    parsed = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert parsed["evidence_memory_ids"] == ["mem_a", "mem_b"]


def test_append_serializes_none_relational_context(tmp_path: Path) -> None:
    log_path = tmp_path / "growth.log.jsonl"
    append_growth_event(log_path, _event(relational_context=None))
    parsed = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert parsed["relational_context"] is None


def test_read_growth_log_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_growth_log(tmp_path / "missing.jsonl") == []


def test_read_growth_log_returns_oldest_first(tmp_path: Path) -> None:
    log_path = tmp_path / "growth.log.jsonl"
    e1 = _event(name="first", timestamp=datetime(2026, 4, 25, tzinfo=UTC))
    e2 = _event(name="second", timestamp=datetime(2026, 4, 26, tzinfo=UTC))
    append_growth_event(log_path, e1)
    append_growth_event(log_path, e2)
    events = read_growth_log(log_path)
    assert len(events) == 2
    assert events[0].name == "first"
    assert events[1].name == "second"


def test_read_growth_log_with_limit_returns_most_recent(tmp_path: Path) -> None:
    """`limit=N` returns the N most-recent events (last N lines)."""
    log_path = tmp_path / "growth.log.jsonl"
    for i in range(5):
        append_growth_event(log_path, _event(name=f"e{i}"))
    events = read_growth_log(log_path, limit=2)
    assert len(events) == 2
    assert events[0].name == "e3"  # second-most recent
    assert events[1].name == "e4"  # most recent


def test_read_growth_log_skips_corrupt_lines(tmp_path: Path, caplog) -> None:
    """A partial-write or hand-edited bad line is skipped, others still parse."""
    log_path = tmp_path / "growth.log.jsonl"
    append_growth_event(log_path, _event(name="good"))
    # Append a corrupt line manually
    with log_path.open("a", encoding="utf-8") as f:
        f.write("{not valid json\n")
    append_growth_event(log_path, _event(name="also_good"))
    events = read_growth_log(log_path)
    assert len(events) == 2
    assert {e.name for e in events} == {"good", "also_good"}


def test_read_growth_log_round_trips_all_fields(tmp_path: Path) -> None:
    """Every field on GrowthLogEvent makes it through write+read intact."""
    log_path = tmp_path / "growth.log.jsonl"
    e = _event(
        name="x",
        description="d",
        decay_half_life_days=None,
        reason="r",
        evidence_memory_ids=("a", "b", "c"),
        score=0.5,
        relational_context="ctx",
    )
    append_growth_event(log_path, e)
    [restored] = read_growth_log(log_path)
    assert restored.name == "x"
    assert restored.description == "d"
    assert restored.decay_half_life_days is None
    assert restored.reason == "r"
    assert restored.evidence_memory_ids == ("a", "b", "c")
    assert restored.score == 0.5
    assert restored.relational_context == "ctx"
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `uv run pytest tests/unit/brain/growth/test_log.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain.growth.log'`

- [ ] **Step 3: Write the minimal implementation**

```python
# brain/growth/log.py
"""Append-only biography of brain growth events.

Each line is one complete JSON object. Never edited, never deleted —
the brain's biography is preserved. Atomic append via the standard
`.new + os.replace` rotation so a crash mid-write leaves either the
old log or the old log + the new line, never a partial line.

Per principle audit 2026-04-25 (Phase 2a §6): the growth log is the
record of who-they-became — not telemetry an owner consults, but
biography future engineers (and the user, via GUI) can read.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from brain.utils.time import iso_utc, parse_iso_utc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GrowthLogEvent:
    """One event in the brain's growth biography.

    `type` is a discriminator allowing the same log to record events from
    any future engine — Phase 2a only emits "emotion_added"; Phase 2a-extension
    PRs add "arc_added", "interest_added", "soul_crystallized".
    """

    timestamp: datetime  # tz-aware UTC
    type: str
    name: str
    description: str
    decay_half_life_days: float | None
    reason: str
    evidence_memory_ids: tuple[str, ...]
    score: float
    relational_context: str | None


def append_growth_event(path: Path, event: GrowthLogEvent) -> None:
    """Atomic append: write `path + ".new"` containing existing-content + new-line, then os.replace.

    A crash between write and rename leaves the previous valid file intact.
    A crash after rename leaves the new line in the file. No partial-line
    state is ever observable to readers.
    """
    line = json.dumps(_event_to_dict(event)) + "\n"
    existing = path.read_bytes() if path.exists() else b""
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_bytes(existing + line.encode("utf-8"))
    os.replace(tmp, path)


def read_growth_log(path: Path, *, limit: int | None = None) -> list[GrowthLogEvent]:
    """Read events oldest-first. `limit=N` returns the N most-recent events.

    Corrupt lines (partial write, hand-edit) are skipped with a warning;
    well-formed lines around them still parse.
    """
    if not path.exists():
        return []
    events: list[GrowthLogEvent] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
            events.append(_event_from_dict(data))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("skipping malformed growth log line: %.200s", exc)
            continue
    if limit is not None:
        events = events[-limit:]
    return events


def _event_to_dict(event: GrowthLogEvent) -> dict:
    return {
        "timestamp": iso_utc(event.timestamp),
        "type": event.type,
        "name": event.name,
        "description": event.description,
        "decay_half_life_days": event.decay_half_life_days,
        "reason": event.reason,
        "evidence_memory_ids": list(event.evidence_memory_ids),
        "score": event.score,
        "relational_context": event.relational_context,
    }


def _event_from_dict(data: dict) -> GrowthLogEvent:
    return GrowthLogEvent(
        timestamp=parse_iso_utc(data["timestamp"]),
        type=str(data["type"]),
        name=str(data["name"]),
        description=str(data["description"]),
        decay_half_life_days=(
            None if data["decay_half_life_days"] is None else float(data["decay_half_life_days"])
        ),
        reason=str(data["reason"]),
        evidence_memory_ids=tuple(str(x) for x in data["evidence_memory_ids"]),
        score=float(data["score"]),
        relational_context=(
            None if data["relational_context"] is None else str(data["relational_context"])
        ),
    )
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `uv run pytest tests/unit/brain/growth/test_log.py -v`
Expected: PASS — 11/11 tests green.

- [ ] **Step 5: Commit**

```bash
git add brain/growth/log.py tests/unit/brain/growth/test_log.py
git commit -m "feat(growth): add atomic append-only growth log — Phase 2a T2"
```

---

### Task 3: Vocabulary crystallizer stub

**Files:**
- Create: `brain/growth/crystallizers/__init__.py`
- Create: `brain/growth/crystallizers/vocabulary.py`
- Create: `tests/unit/brain/growth/test_vocabulary_crystallizer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/brain/growth/test_vocabulary_crystallizer.py
"""Tests for brain.growth.crystallizers.vocabulary — Phase 2a stub."""

from __future__ import annotations

from brain.growth.crystallizers.vocabulary import crystallize_vocabulary
from brain.memory.store import MemoryStore


def test_phase_2a_stub_returns_empty_list() -> None:
    """Phase 2a: crystallizer is a no-op. Phase 2b populates with pattern matchers."""
    store = MemoryStore(":memory:")
    try:
        result = crystallize_vocabulary(store, current_vocabulary_names=set())
        assert result == []
    finally:
        store.close()


def test_phase_2a_stub_ignores_inputs() -> None:
    """Stub returns [] regardless of input — verifies signature accepts the
    arguments Phase 2b will use."""
    store = MemoryStore(":memory:")
    try:
        result = crystallize_vocabulary(
            store,
            current_vocabulary_names={"love", "joy", "grief"},
        )
        assert result == []
    finally:
        store.close()
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `uv run pytest tests/unit/brain/growth/test_vocabulary_crystallizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain.growth.crystallizers'`

- [ ] **Step 3: Write the minimal implementation**

```python
# brain/growth/crystallizers/__init__.py
"""Crystallizers — pure functions that examine brain state and return proposals.

Each crystallizer takes the brain's current state (MemoryStore + relevant
engine context) and returns a list of `*Proposal` objects. The scheduler
applies them atomically.

Phase 2a ships a no-op vocabulary crystallizer (returns []). Phase 2b
populates the body with pattern-matching against memories + relational
dynamics + LLM-mediated naming.
"""
```

```python
# brain/growth/crystallizers/vocabulary.py
"""Vocabulary crystallizer — Phase 2a stub.

Phase 2a returns []. Phase 2b will:
- Cluster memories by emotional configuration vectors
- Detect clusters that recur but don't have a name in current_vocabulary_names
- Detect clusters that align with specific relational dynamics
- Apply quality gates (novelty, evidence threshold, score threshold)
- Apply rate limit (max 1 proposal per tick)
- Use LLM-mediated naming (via brain.bridge.provider.LLMProvider)
"""

from __future__ import annotations

from brain.growth.proposal import EmotionProposal
from brain.memory.store import MemoryStore


def crystallize_vocabulary(
    store: MemoryStore,
    *,
    current_vocabulary_names: set[str],
) -> list[EmotionProposal]:
    """Mine memory + relational dynamics for novel emotional configurations.

    Phase 2a behavior: returns [] always. The signature accepts arguments
    Phase 2b will use; ignored in 2a.
    """
    # Phase 2b will read store + current_vocabulary_names.
    _ = store
    _ = current_vocabulary_names
    return []
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `uv run pytest tests/unit/brain/growth/test_vocabulary_crystallizer.py -v`
Expected: PASS — 2/2 tests green.

- [ ] **Step 5: Commit**

```bash
git add brain/growth/crystallizers/ tests/unit/brain/growth/test_vocabulary_crystallizer.py
git commit -m "feat(growth): add vocabulary crystallizer stub — Phase 2a T3"
```

---

### Task 4: `run_growth_tick` scheduler with atomic apply path

**Files:**
- Create: `brain/growth/scheduler.py`
- Create: `tests/unit/brain/growth/test_scheduler.py`

The scheduler is the only mutator of `emotion_vocabulary.json` + `emotion_growth.log.jsonl` during a growth tick. Validation rules: name must not collide with existing vocabulary, name must pass character validation (no `/`, `\`, `{`, `}`). Atomic write of vocabulary + atomic append to growth log. Tests use a fake crystallizer to exercise the apply path even though the real crystallizer is no-op.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/brain/growth/test_scheduler.py
"""Tests for brain.growth.scheduler — orchestrator + atomic apply."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from brain.growth.proposal import EmotionProposal
from brain.growth.scheduler import GrowthTickResult, run_growth_tick
from brain.memory.store import MemoryStore


def _seed_vocab(persona_dir: Path, names: list[str] = ["love", "joy"]) -> None:
    """Seed an emotion_vocabulary.json with the given names as core emotions."""
    entries = [
        {
            "name": n,
            "description": f"the feeling of {n}",
            "category": "core",
            "decay_half_life_days": 7.0,
        }
        for n in names
    ]
    (persona_dir / "emotion_vocabulary.json").write_text(
        json.dumps({"version": 1, "emotions": entries}, indent=2),
        encoding="utf-8",
    )


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    pdir = tmp_path / "persona"
    pdir.mkdir()
    return pdir


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


def test_run_growth_tick_no_proposals_returns_zero(persona_dir: Path, store: MemoryStore) -> None:
    """Phase 2a's real crystallizer returns [] — scheduler returns count=0."""
    _seed_vocab(persona_dir)
    result = run_growth_tick(persona_dir, store, datetime.now(UTC))
    assert isinstance(result, GrowthTickResult)
    assert result.emotions_added == 0
    assert result.proposals_seen == 0
    # No log file created when nothing happened
    assert not (persona_dir / "emotion_growth.log.jsonl").exists()


def test_run_growth_tick_applies_proposal_atomically(
    persona_dir: Path, store: MemoryStore
) -> None:
    """When a crystallizer returns a proposal, vocabulary + log update together."""
    _seed_vocab(persona_dir)
    proposal = EmotionProposal(
        name="lingering",
        description="the slow trail of warmth after a loved person leaves the room",
        decay_half_life_days=7.0,
        evidence_memory_ids=("mem_a", "mem_b"),
        score=0.78,
        relational_context="recurred during Hana's tender messages",
    )

    with patch(
        "brain.growth.scheduler.crystallize_vocabulary",
        return_value=[proposal],
    ):
        result = run_growth_tick(persona_dir, store, datetime.now(UTC))

    assert result.emotions_added == 1
    assert result.proposals_seen == 1
    assert result.proposals_rejected == 0

    # Vocabulary file updated
    vocab = json.loads((persona_dir / "emotion_vocabulary.json").read_text(encoding="utf-8"))
    names = [e["name"] for e in vocab["emotions"]]
    assert "lingering" in names
    new_entry = next(e for e in vocab["emotions"] if e["name"] == "lingering")
    assert new_entry["category"] == "persona_extension"
    assert new_entry["decay_half_life_days"] == 7.0

    # Growth log updated
    log_path = persona_dir / "emotion_growth.log.jsonl"
    assert log_path.exists()
    [line] = log_path.read_text(encoding="utf-8").splitlines()
    parsed = json.loads(line)
    assert parsed["type"] == "emotion_added"
    assert parsed["name"] == "lingering"
    assert parsed["relational_context"] == "recurred during Hana's tender messages"


def test_run_growth_tick_skips_proposal_with_existing_name(
    persona_dir: Path, store: MemoryStore
) -> None:
    """Idempotent: a proposal whose name already exists in the vocabulary is skipped silently."""
    _seed_vocab(persona_dir, names=["love", "joy"])
    proposal = EmotionProposal(
        name="love",  # already in vocab
        description="dup",
        decay_half_life_days=None,
        evidence_memory_ids=(),
        score=0.5,
        relational_context=None,
    )
    with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=[proposal]):
        result = run_growth_tick(persona_dir, store, datetime.now(UTC))
    assert result.emotions_added == 0
    assert result.proposals_seen == 1
    # Skipped duplicates aren't counted as rejections — re-proposing is normal.
    assert result.proposals_rejected == 0
    assert not (persona_dir / "emotion_growth.log.jsonl").exists()


def test_run_growth_tick_rejects_proposal_with_invalid_chars(
    persona_dir: Path, store: MemoryStore
) -> None:
    """Names containing path-traversal chars or curly braces are rejected as schema-invalid."""
    _seed_vocab(persona_dir)
    bad_names = ["bad/name", "bad\\name", "bad{name}", ""]
    for bn in bad_names:
        proposal = EmotionProposal(
            name=bn,
            description="x",
            decay_half_life_days=None,
            evidence_memory_ids=(),
            score=0.5,
            relational_context=None,
        )
        with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=[proposal]):
            result = run_growth_tick(persona_dir, store, datetime.now(UTC))
        assert result.emotions_added == 0
        assert result.proposals_rejected == 1


def test_run_growth_tick_dry_run_does_not_write(
    persona_dir: Path, store: MemoryStore
) -> None:
    """dry_run=True calls crystallizer but writes neither vocabulary nor log."""
    _seed_vocab(persona_dir)
    proposal = EmotionProposal(
        name="lingering",
        description="x",
        decay_half_life_days=None,
        evidence_memory_ids=(),
        score=0.5,
        relational_context=None,
    )
    vocab_before = (persona_dir / "emotion_vocabulary.json").read_text(encoding="utf-8")
    with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=[proposal]):
        result = run_growth_tick(persona_dir, store, datetime.now(UTC), dry_run=True)
    assert result.emotions_added == 1  # would-have-added semantics
    # Files unchanged
    assert (persona_dir / "emotion_vocabulary.json").read_text(encoding="utf-8") == vocab_before
    assert not (persona_dir / "emotion_growth.log.jsonl").exists()


def test_run_growth_tick_handles_multiple_proposals(
    persona_dir: Path, store: MemoryStore
) -> None:
    _seed_vocab(persona_dir)
    proposals = [
        EmotionProposal(
            name=f"p{i}",
            description=f"desc {i}",
            decay_half_life_days=float(i),
            evidence_memory_ids=(),
            score=0.5,
            relational_context=None,
        )
        for i in range(3)
    ]
    with patch("brain.growth.scheduler.crystallize_vocabulary", return_value=proposals):
        result = run_growth_tick(persona_dir, store, datetime.now(UTC))
    assert result.emotions_added == 3
    log_path = persona_dir / "emotion_growth.log.jsonl"
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 3
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `uv run pytest tests/unit/brain/growth/test_scheduler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain.growth.scheduler'`

- [ ] **Step 3: Write the minimal implementation**

```python
# brain/growth/scheduler.py
"""Growth scheduler — orchestrates crystallizers + applies decisions atomically.

The scheduler is the *only* mutator of `emotion_vocabulary.json` and
`emotion_growth.log.jsonl` during a growth tick. No engine touches these
files except through `run_growth_tick`.

Per principle audit 2026-04-25 (Phase 2a §4): the brain owns its own
growth. Crystallizers decide; the scheduler applies; the log records
biographically. No human approval gate, no candidate queue.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from brain.growth.crystallizers.vocabulary import crystallize_vocabulary
from brain.growth.log import GrowthLogEvent, append_growth_event
from brain.growth.proposal import EmotionProposal
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Same character allowlist as brain.paths.get_persona_dir — names that
# could appear as filesystem path components or trip JSON parsing get
# rejected before they reach disk.
_INVALID_NAME_CHARS = ("/", "\\", "{", "}")


@dataclass(frozen=True)
class GrowthTickResult:
    """Outcome of one growth tick."""

    emotions_added: int
    proposals_seen: int
    proposals_rejected: int


def run_growth_tick(
    persona_dir: Path,
    store: MemoryStore,
    now: datetime,
    *,
    dry_run: bool = False,
) -> GrowthTickResult:
    """Run all crystallizers, apply their proposals atomically.

    For each proposal:
      1. Skip silently if name already in current vocabulary (re-proposing
         is normal; not a rejection).
      2. Reject (with warning) if name fails character validation.
      3. Else: append to {persona_dir}/emotion_vocabulary.json (atomic
         `.new + os.replace`) and append a GrowthLogEvent to
         {persona_dir}/emotion_growth.log.jsonl (atomic per `log.py`).

    `dry_run=True` calls the crystallizer but skips both writes; the
    returned `emotions_added` reflects "would-have-added" semantics.
    """
    vocab_path = persona_dir / "emotion_vocabulary.json"
    log_path = persona_dir / "emotion_growth.log.jsonl"

    current_names = _read_current_vocabulary_names(vocab_path)

    proposals = crystallize_vocabulary(store, current_vocabulary_names=current_names)

    emotions_added = 0
    proposals_rejected = 0

    for proposal in proposals:
        if proposal.name in current_names:
            # Idempotent skip — re-proposal is normal, not a rejection.
            continue
        if not _is_valid_name(proposal.name):
            logger.warning(
                "growth scheduler: rejecting proposal with invalid name %r", proposal.name
            )
            proposals_rejected += 1
            continue

        emotions_added += 1
        if dry_run:
            continue

        _append_to_vocabulary(vocab_path, proposal)
        current_names.add(proposal.name)
        append_growth_event(
            log_path,
            GrowthLogEvent(
                timestamp=now,
                type="emotion_added",
                name=proposal.name,
                description=proposal.description,
                decay_half_life_days=proposal.decay_half_life_days,
                reason=_default_reason_for(proposal),
                evidence_memory_ids=proposal.evidence_memory_ids,
                score=proposal.score,
                relational_context=proposal.relational_context,
            ),
        )

    return GrowthTickResult(
        emotions_added=emotions_added,
        proposals_seen=len(proposals),
        proposals_rejected=proposals_rejected,
    )


def _read_current_vocabulary_names(vocab_path: Path) -> set[str]:
    if not vocab_path.exists():
        return set()
    try:
        data = json.loads(vocab_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    if not isinstance(data, dict) or not isinstance(data.get("emotions"), list):
        return set()
    return {e["name"] for e in data["emotions"] if isinstance(e, dict) and "name" in e}


def _is_valid_name(name: str) -> bool:
    if not name:
        return False
    return not any(c in name for c in _INVALID_NAME_CHARS)


def _append_to_vocabulary(vocab_path: Path, proposal: EmotionProposal) -> None:
    """Atomic append to emotion_vocabulary.json — read, append entry, write `.new`, rename."""
    if vocab_path.exists():
        data = json.loads(vocab_path.read_text(encoding="utf-8"))
    else:
        data = {"version": 1, "emotions": []}
    data["emotions"].append(
        {
            "name": proposal.name,
            "description": proposal.description,
            "category": "persona_extension",
            "decay_half_life_days": proposal.decay_half_life_days,
            "intensity_clamp": 10.0,
        }
    )
    tmp = vocab_path.with_suffix(vocab_path.suffix + ".new")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, vocab_path)


def _default_reason_for(proposal: EmotionProposal) -> str:
    """Phase 2a default — Phase 2b crystallizer fills `proposal.reason` directly.

    For now we synthesize a short reason since EmotionProposal doesn't carry
    one — the dataclass design here matches Phase 2b's likely shape but until
    the crystallizer produces one we describe by score + evidence count.
    """
    return (
        f"score={proposal.score:.2f}, "
        f"evidence_count={len(proposal.evidence_memory_ids)}"
    )
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `uv run pytest tests/unit/brain/growth/test_scheduler.py -v`
Expected: PASS — 6/6 tests green.

- [ ] **Step 5: Commit**

```bash
git add brain/growth/scheduler.py tests/unit/brain/growth/test_scheduler.py
git commit -m "feat(growth): add growth scheduler with atomic apply path — Phase 2a T4"
```

---

### Task 5: HeartbeatConfig + HeartbeatState + HeartbeatResult additions

**Files:**
- Modify: `brain/engines/heartbeat.py:33-107` (HeartbeatConfig), `:110-166` (HeartbeatState), `:169-186` (HeartbeatResult)
- Modify: `tests/unit/brain/engines/test_heartbeat.py` (add round-trip + back-compat tests)

These additions are independent of the growth-tick wiring (T6). Land the data-shape changes first so T6 can rely on them.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/brain/engines/test_heartbeat.py`:

```python
# At the top of the existing test file (or near the existing config tests):

def test_heartbeat_config_has_growth_defaults() -> None:
    """HeartbeatConfig has growth_enabled=True + growth_every_hours=168.0 (weekly)."""
    c = HeartbeatConfig()
    assert c.growth_enabled is True
    assert c.growth_every_hours == 168.0


def test_heartbeat_config_round_trip_preserves_growth(tmp_path: Path) -> None:
    """save() then load() preserves growth fields."""
    original = HeartbeatConfig(growth_enabled=False, growth_every_hours=24.0)
    path = tmp_path / "cfg.json"
    original.save(path)
    restored = HeartbeatConfig.load(path)
    assert restored.growth_enabled is False
    assert restored.growth_every_hours == 24.0


def test_heartbeat_config_back_compat_missing_growth_fields(tmp_path: Path) -> None:
    """Old config files without growth_* fields load with defaults."""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"dream_every_hours": 12.0}), encoding="utf-8")
    c = HeartbeatConfig.load(path)
    assert c.growth_enabled is True
    assert c.growth_every_hours == 168.0


def test_heartbeat_state_includes_last_growth_at() -> None:
    """HeartbeatState.fresh() initialises last_growth_at = now."""
    s = HeartbeatState.fresh(trigger="open")
    assert s.last_growth_at is not None
    assert s.last_growth_at.tzinfo is not None  # tz-aware


def test_heartbeat_state_round_trips_last_growth_at(tmp_path: Path) -> None:
    s = HeartbeatState.fresh(trigger="open")
    path = tmp_path / "state.json"
    s.save(path)
    restored = HeartbeatState.load(path)
    assert restored is not None
    assert restored.last_growth_at == s.last_growth_at


def test_heartbeat_state_back_compat_missing_last_growth_at(tmp_path: Path) -> None:
    """Old state files without last_growth_at load with last_growth_at=now (delays
    first growth tick by growth_every_hours, which is the safe back-compat default)."""
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "last_tick_at": "2026-04-20T10:00:00Z",
                "last_dream_at": "2026-04-20T10:00:00Z",
                "last_research_at": "2026-04-20T10:00:00Z",
                "tick_count": 5,
                "last_trigger": "open",
            }
        ),
        encoding="utf-8",
    )
    s = HeartbeatState.load(path)
    assert s is not None
    assert s.last_growth_at is not None  # backfilled to now-ish on load
    assert s.last_growth_at.tzinfo is not None


def test_heartbeat_result_default_growth_emotions_added_is_zero() -> None:
    """HeartbeatResult.growth_emotions_added defaults to 0."""
    r = HeartbeatResult(
        trigger="manual",
        elapsed_seconds=0.0,
        memories_decayed=0,
        edges_pruned=0,
        dream_id=None,
        dream_gated_reason=None,
        research_deferred=False,
        heartbeat_memory_id=None,
        initialized=False,
    )
    assert r.growth_emotions_added == 0
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `uv run pytest tests/unit/brain/engines/test_heartbeat.py -v -k 'growth or last_growth_at'`
Expected: FAIL — multiple `AttributeError` (fields don't exist yet).

- [ ] **Step 3: Write the minimal implementation**

In `brain/engines/heartbeat.py`:

a) Add to `HeartbeatConfig` (after `interest_bump_per_match`):

```python
    growth_enabled: bool = True
    growth_every_hours: float = 168.0  # weekly default
```

b) In `HeartbeatConfig._load_internal`'s `try` block, add after `interest_bump_per_match` line:

```python
                growth_enabled=bool(data.get("growth_enabled", True)),
                growth_every_hours=float(data.get("growth_every_hours", 168.0)),
```

c) In `HeartbeatConfig.save`'s payload dict, add:

```python
            "growth_enabled": self.growth_enabled,
            "growth_every_hours": self.growth_every_hours,
```

d) Add to `HeartbeatState` (after `last_trigger`):

```python
    last_growth_at: datetime  # tz-aware UTC; defaults to now on first save
```

e) In `HeartbeatState.load`'s `cls(...)` constructor call, add:

```python
                last_growth_at=parse_iso_utc(
                    data.get("last_growth_at") or data["last_tick_at"]
                ),
```

(Falling back to `last_tick_at` provides graceful back-compat for pre-Phase-2a state files.)

f) In `HeartbeatState.fresh`:

```python
            last_growth_at=now,
```

g) In `HeartbeatState.save`'s payload dict, add:

```python
            "last_growth_at": iso_utc(self.last_growth_at),
```

h) Add to `HeartbeatResult`:

```python
    growth_emotions_added: int = 0
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `uv run pytest tests/unit/brain/engines/test_heartbeat.py -v -k 'growth or last_growth_at'`
Expected: PASS — all 7 new tests green.

Then run the full heartbeat test file to verify no regression:

Run: `uv run pytest tests/unit/brain/engines/test_heartbeat.py -q`
Expected: PASS — all existing tests still green plus the 7 new ones.

- [ ] **Step 5: Commit**

```bash
git add brain/engines/heartbeat.py tests/unit/brain/engines/test_heartbeat.py
git commit -m "feat(heartbeat): add growth_enabled/growth_every_hours/last_growth_at — Phase 2a T5"
```

---

### Task 6: `_try_run_growth` + heartbeat tick wiring

**Files:**
- Modify: `brain/engines/heartbeat.py:237-368` (run_tick body) and add `_try_run_growth` method
- Modify: `tests/unit/brain/engines/test_heartbeat.py`

The growth tick fires AFTER research, BEFORE optional heartbeat memory. Fault-isolated like reflex/research. Audit log gains a `growth` sub-object.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/brain/engines/test_heartbeat.py`:

```python
def test_heartbeat_run_tick_calls_growth_after_research(tmp_path: Path) -> None:
    """Growth tick fires when due; heartbeat reports the count."""
    from unittest.mock import patch

    from brain.engines.heartbeat import HeartbeatEngine
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    # Seed a vocabulary so the scheduler has somewhere to write.
    (persona_dir / "emotion_vocabulary.json").write_text(
        json.dumps({"version": 1, "emotions": []}), encoding="utf-8"
    )

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    try:
        engine = HeartbeatEngine(
            store=store,
            hebbian=hebbian,
            provider=FakeProvider(),
            state_path=persona_dir / "heartbeat_state.json",
            config_path=persona_dir / "heartbeat_config.json",
            dream_log_path=persona_dir / "dreams.log.jsonl",
            heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
            interests_path=persona_dir / "interests.json",
            research_log_path=persona_dir / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="test",
            persona_system_prompt="You are test.",
        )

        # First tick — initializes state, defers all work.
        engine.run_tick(trigger="open")

        # Force last_growth_at older than 168h so growth is due.
        from brain.engines.heartbeat import HeartbeatState

        state = HeartbeatState.load(persona_dir / "heartbeat_state.json")
        assert state is not None
        old = datetime.now(UTC) - timedelta(hours=200)
        state.last_growth_at = old
        state.last_tick_at = old
        state.save(persona_dir / "heartbeat_state.json")

        # Inject a fake crystallizer that returns one proposal.
        from brain.growth.proposal import EmotionProposal

        proposal = EmotionProposal(
            name="lingering",
            description="x",
            decay_half_life_days=None,
            evidence_memory_ids=(),
            score=0.5,
            relational_context=None,
        )
        with patch(
            "brain.growth.scheduler.crystallize_vocabulary",
            return_value=[proposal],
        ):
            result = engine.run_tick(trigger="manual")

        assert result.growth_emotions_added == 1

        # Audit log entry has growth sub-object
        log_lines = (persona_dir / "heartbeats.log.jsonl").read_text(encoding="utf-8").splitlines()
        last_entry = json.loads(log_lines[-1])
        assert "growth" in last_entry
        assert last_entry["growth"]["enabled"] is True
        assert last_entry["growth"]["ran"] is True
        assert last_entry["growth"]["emotions_added"] == 1
    finally:
        store.close()
        hebbian.close()


def test_heartbeat_growth_disabled_skips_growth_tick(tmp_path: Path) -> None:
    """growth_enabled=False short-circuits before scheduler runs."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    cfg_path = persona_dir / "heartbeat_config.json"
    cfg_path.write_text(json.dumps({"growth_enabled": False}), encoding="utf-8")

    from brain.engines.heartbeat import HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    try:
        engine = HeartbeatEngine(
            store=store,
            hebbian=hebbian,
            provider=FakeProvider(),
            state_path=persona_dir / "heartbeat_state.json",
            config_path=cfg_path,
            dream_log_path=persona_dir / "dreams.log.jsonl",
            heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
            interests_path=persona_dir / "interests.json",
            research_log_path=persona_dir / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="test",
            persona_system_prompt="You are test.",
        )
        engine.run_tick(trigger="open")  # init
        # Push state back so growth would otherwise be due.
        s = HeartbeatState.load(persona_dir / "heartbeat_state.json")
        assert s is not None
        s.last_growth_at = datetime.now(UTC) - timedelta(hours=200)
        s.last_tick_at = datetime.now(UTC) - timedelta(hours=200)
        s.save(persona_dir / "heartbeat_state.json")

        result = engine.run_tick(trigger="manual")
        assert result.growth_emotions_added == 0
    finally:
        store.close()
        hebbian.close()


def test_heartbeat_growth_not_due_returns_zero(tmp_path: Path) -> None:
    """Within growth_every_hours window, growth tick returns zero."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    from brain.engines.heartbeat import HeartbeatEngine
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    try:
        engine = HeartbeatEngine(
            store=store,
            hebbian=hebbian,
            provider=FakeProvider(),
            state_path=persona_dir / "heartbeat_state.json",
            config_path=persona_dir / "heartbeat_config.json",
            dream_log_path=persona_dir / "dreams.log.jsonl",
            heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
            interests_path=persona_dir / "interests.json",
            research_log_path=persona_dir / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="test",
            persona_system_prompt="You are test.",
        )
        engine.run_tick(trigger="open")  # init
        result = engine.run_tick(trigger="manual")  # not 168h later
        assert result.growth_emotions_added == 0
    finally:
        store.close()
        hebbian.close()


def test_heartbeat_growth_fault_isolated(tmp_path: Path) -> None:
    """If the growth tick raises, heartbeat continues — count is 0."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    from unittest.mock import patch

    from brain.engines.heartbeat import HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    try:
        engine = HeartbeatEngine(
            store=store,
            hebbian=hebbian,
            provider=FakeProvider(),
            state_path=persona_dir / "heartbeat_state.json",
            config_path=persona_dir / "heartbeat_config.json",
            dream_log_path=persona_dir / "dreams.log.jsonl",
            heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
            interests_path=persona_dir / "interests.json",
            research_log_path=persona_dir / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="test",
            persona_system_prompt="You are test.",
        )
        engine.run_tick(trigger="open")
        s = HeartbeatState.load(persona_dir / "heartbeat_state.json")
        assert s is not None
        s.last_growth_at = datetime.now(UTC) - timedelta(hours=200)
        s.last_tick_at = datetime.now(UTC) - timedelta(hours=200)
        s.save(persona_dir / "heartbeat_state.json")

        with patch(
            "brain.growth.scheduler.run_growth_tick",
            side_effect=RuntimeError("simulated crash"),
        ):
            result = engine.run_tick(trigger="manual")
        assert result.growth_emotions_added == 0
        # Heartbeat tick still completed
        assert not result.initialized
    finally:
        store.close()
        hebbian.close()
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `uv run pytest tests/unit/brain/engines/test_heartbeat.py -v -k 'growth_after_research or growth_disabled or growth_not_due or growth_fault_isolated'`
Expected: FAIL — `_try_run_growth` doesn't exist yet, audit log lacks growth sub-object.

- [ ] **Step 3: Write the minimal implementation**

In `brain/engines/heartbeat.py`:

a) Add `_try_run_growth` method on `HeartbeatEngine`, right after `_try_fire_research`:

```python
    def _try_run_growth(
        self,
        state: HeartbeatState,
        now: datetime,
        config: HeartbeatConfig,
        dry_run: bool,
    ) -> tuple[int, bool]:
        """Run a growth tick if due. Returns (emotions_added, ran).

        Fault-isolated: any exception logs a warning and returns (0, False).
        Heartbeat tick continues normally — same pattern as reflex/research.
        """
        if not config.growth_enabled:
            return (0, False)
        if self.interests_path is None:
            # Use interests_path as a proxy for "persona dir is wired" — Phase 2a
            # doesn't add a separate persona_dir field.
            return (0, False)

        hours_since = (now - state.last_growth_at).total_seconds() / 3600.0
        if hours_since < config.growth_every_hours:
            return (0, False)

        persona_dir = self.interests_path.parent
        try:
            from brain.growth.scheduler import run_growth_tick

            result = run_growth_tick(persona_dir, self.store, now, dry_run=dry_run)
        except Exception as exc:
            logger.warning("growth tick raised; isolating: %.200s", exc)
            return (0, False)

        if not dry_run:
            state.last_growth_at = now

        return (result.emotions_added, True)
```

b) In `HeartbeatEngine.run_tick`, between the research block and the heartbeat-memory block, insert:

```python
        # Growth tick — autonomous self-development (Phase 2a). Runs after
        # all per-tick engines so it can observe the freshest state, before
        # the audit log writes so the audit can summarize the growth outcome.
        growth_emotions_added, growth_ran = self._try_run_growth(state, now, config, dry_run)
```

c) Update the `HeartbeatResult(...)` return to include `growth_emotions_added=growth_emotions_added`.

d) Update the audit log payload (the `_append_log({...})` call in `run_tick`) to include:

```python
                "growth": {
                    "enabled": config.growth_enabled,
                    "ran": growth_ran,
                    "emotions_added": growth_emotions_added,
                },
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `uv run pytest tests/unit/brain/engines/test_heartbeat.py -q`
Expected: PASS — all existing + 4 new growth-tick integration tests green.

Then full suite to verify no regression:

Run: `uv run pytest -q`
Expected: PASS — full suite green.

- [ ] **Step 5: Commit**

```bash
git add brain/engines/heartbeat.py tests/unit/brain/engines/test_heartbeat.py
git commit -m "feat(heartbeat): wire growth tick into run_tick — Phase 2a T6"
```

---

### Task 7: `nell growth log` CLI subcommand

**Files:**
- Modify: `brain/cli.py` — add `_growth_log_handler` + subparser
- Create: `tests/unit/brain/growth/test_cli_growth.py`

Read-only inspection. No `add`, `approve`, `reject`, `force` — the brain has agency, the user reads.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/brain/growth/test_cli_growth.py
"""Tests for `nell growth log` CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.cli import main
from brain.growth.log import GrowthLogEvent, append_growth_event


def _setup_persona(personas_root: Path, name: str = "testpersona") -> Path:
    persona_dir = personas_root / name
    persona_dir.mkdir(parents=True)
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    MemoryStore(db_path=persona_dir / "memories.db").close()
    HebbianMatrix(db_path=persona_dir / "hebbian.db").close()
    return persona_dir


def _seed_growth_event(
    persona_dir: Path, name: str, when: datetime, relational_context: str | None = None
) -> None:
    append_growth_event(
        persona_dir / "emotion_growth.log.jsonl",
        GrowthLogEvent(
            timestamp=when,
            type="emotion_added",
            name=name,
            description=f"description of {name}",
            decay_half_life_days=7.0,
            reason="seeded",
            evidence_memory_ids=("mem_a",),
            score=0.7,
            relational_context=relational_context,
        ),
    )


def test_cli_growth_log_empty(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    _setup_persona(tmp_path / "personas")
    rc = main(["growth", "log", "--persona", "testpersona"])
    assert rc == 0
    out = capsys.readouterr().out
    # Should mention zero events / empty log gracefully
    assert "0 events" in out or "empty" in out.lower() or "no events" in out.lower()


def test_cli_growth_log_displays_events(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")
    _seed_growth_event(
        persona_dir,
        "lingering",
        datetime(2026, 4, 25, 18, 30, tzinfo=UTC),
        relational_context="during Hana's tender messages",
    )
    rc = main(["growth", "log", "--persona", "testpersona"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "lingering" in out
    assert "during Hana's tender messages" in out
    assert "2026-04-25" in out


def test_cli_growth_log_limit_flag(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")
    for i in range(5):
        _seed_growth_event(
            persona_dir,
            f"e{i}",
            datetime(2026, 4, 20 + i, tzinfo=UTC),
        )
    rc = main(["growth", "log", "--persona", "testpersona", "--limit", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "e3" in out
    assert "e4" in out
    assert "e0" not in out
    assert "e1" not in out


def test_cli_growth_log_missing_persona_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="persona"):
        main(["growth", "log", "--persona", "ghost"])


def test_cli_growth_no_action_commands(monkeypatch, tmp_path: Path):
    """Phase 2a only ships `log` — no `add`, `approve`, `reject`, `force`."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    _setup_persona(tmp_path / "personas")
    for forbidden in ("add", "approve", "reject", "force"):
        with pytest.raises(SystemExit):
            main(["growth", forbidden, "--persona", "testpersona"])
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `uv run pytest tests/unit/brain/growth/test_cli_growth.py -v`
Expected: FAIL — `nell growth` subcommand doesn't exist.

- [ ] **Step 3: Write the minimal implementation**

In `brain/cli.py`:

a) Add a handler near the other CLI handlers (e.g. after `_interest_list_handler`):

```python
def _growth_log_handler(args: argparse.Namespace) -> int:
    """`nell growth log` — read-only inspection of the brain's growth biography.

    Per Phase 2a §8: read-only. No add/approve/reject/force. The user
    reads what the brain decided; if they want to override, they edit
    `emotion_vocabulary.json` directly.
    """
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(
            f"No persona directory at {persona_dir}. "
            f"Persona {args.persona!r} does not exist."
        )

    from brain.growth.log import read_growth_log

    log_path = persona_dir / "emotion_growth.log.jsonl"
    events = read_growth_log(log_path, limit=args.limit)
    if args.type:
        events = [e for e in events if e.type == args.type]

    print(f"Growth log for persona {args.persona!r} ({len(events)} events shown):")
    if not events:
        print("  (empty)")
        return 0

    for e in events:
        ts = e.timestamp.isoformat().replace("+00:00", "Z")
        print(f"\n  {ts}  {e.type:<20} {e.name}")
        print(f'    "{e.description}"')
        decay = (
            "identity-level (no decay)"
            if e.decay_half_life_days is None
            else f"{e.decay_half_life_days:.1f} days"
        )
        print(f"    decay: {decay}  score: {e.score:.2f}")
        print(f"    reason: {e.reason}")
        if e.relational_context:
            print(f"    relational: {e.relational_context}")
        if e.evidence_memory_ids:
            preview = ", ".join(e.evidence_memory_ids[:3])
            extra = (
                f", ... ({len(e.evidence_memory_ids)} total)"
                if len(e.evidence_memory_ids) > 3
                else ""
            )
            print(f"    evidence: {preview}{extra}")
    return 0
```

b) Add the subparser in `_build_parser` (near the `interest` subparser block):

```python
    # nell growth log — read-only inspection of brain growth biography.
    g_sub = subparsers.add_parser(
        "growth",
        help="Inspect the brain's autonomous growth biography (read-only).",
    )
    g_actions = g_sub.add_subparsers(dest="action", required=True)

    g_log = g_actions.add_parser("log", help="Print the growth log.")
    g_log.add_argument(
        "--persona",
        required=True,
        help="Persona name (required).",
    )
    g_log.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Show only the most-recent N events.",
    )
    g_log.add_argument(
        "--type",
        default=None,
        help="Filter by event type (e.g. 'emotion_added').",
    )
    g_log.set_defaults(func=_growth_log_handler)
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `uv run pytest tests/unit/brain/growth/test_cli_growth.py -v`
Expected: PASS — 5/5 new CLI tests green.

Then full suite:

Run: `uv run pytest -q`
Expected: PASS — full suite green.

- [ ] **Step 5: Commit**

```bash
git add brain/cli.py tests/unit/brain/growth/test_cli_growth.py
git commit -m "feat(growth): add nell growth log read-only CLI — Phase 2a T7"
```

---

### Task 8: Acceptance smoke test + final verification

**Files:**
- (No code changes — this task verifies the spec's acceptance criteria.)

- [ ] **Step 1: Verify zero anthropic imports in brain/growth/**

Run: `rg 'import anthropic' brain/growth/`
Expected: zero matches.

- [ ] **Step 2: Verify lint clean**

Run: `uv run ruff check brain/ tests/`
Expected: All checks passed.

Run: `uv run ruff format --check brain/ tests/`
Expected: Files would be unchanged (or run `uv run ruff format brain/ tests/` and re-commit if formatting drift).

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -q`
Expected: All tests pass. Net new across T1-T7: 4 + 11 + 2 + 6 + 7 + 4 + 5 = 39 new. Existing 492 → ~531 total.

- [ ] **Step 4: Sandbox smoke test against Nell's persona**

Run: `uv run nell heartbeat --persona nell.sandbox --trigger manual --provider fake`
Expected:
- Tick completes without warnings
- `heartbeats.log.jsonl` last entry contains `"growth": {"enabled": true, "ran": false, "emotions_added": 0}`
  (`ran: false` because growth_every_hours hasn't elapsed since persona init)
- `emotion_growth.log.jsonl` does not exist (no events yet)

Run: `uv run nell growth log --persona nell.sandbox`
Expected:
- Prints `Growth log for persona 'nell.sandbox' (0 events shown):` followed by `(empty)`.

- [ ] **Step 5: Verify CLI surface matches principle**

Run: `uv run nell growth --help`
Expected: only `log` action listed. No `add` / `approve` / `reject` / `force`.

- [ ] **Step 6: Inject-test scheduler atomicity (manual verification)**

This is already covered by `test_run_growth_tick_applies_proposal_atomically` in T4, but worth a final eyeball check:

Run: `uv run pytest tests/unit/brain/growth/test_scheduler.py::test_run_growth_tick_applies_proposal_atomically -v`
Expected: PASS, demonstrating both `emotion_vocabulary.json` and `emotion_growth.log.jsonl` are updated when an injected proposal is accepted.

- [ ] **Step 7: Open the PR**

Push the branch and open the PR with a body summarizing all 8 tasks plus a note about deferred Phase 2b work. CI must run green on macOS / Linux / Windows / Python 3.12 before merge.

```bash
git push -u origin <branch-name>
gh pr create --title "Phase 2a: Vocabulary emergence architecture" --body "..."
```

---

## Acceptance Criteria (from spec §14)

After all tasks ship, these must all hold:

- [ ] `brain/growth/` package exists with `log.py`, `scheduler.py`, `proposal.py`, `crystallizers/vocabulary.py`.
- [ ] `crystallize_vocabulary(store, current_vocabulary_names=...) -> []` (no-op stub).
- [ ] `run_growth_tick(persona_dir, store, now, dry_run=...)` orchestrates crystallizers + applies proposals atomically.
- [ ] `append_growth_event` writes one JSON line atomically; `read_growth_log` reads them back oldest-first.
- [ ] `HeartbeatConfig` has `growth_enabled` + `growth_every_hours`.
- [ ] `HeartbeatState` has `last_growth_at`.
- [ ] `HeartbeatEngine.run_tick` calls `_try_run_growth` after research, before optional heartbeat memory.
- [ ] `nell growth log --persona X [--limit N]` displays the log read-only.
- [ ] `rg 'import anthropic' brain/growth/` returns zero matches.
- [ ] `uv run pytest -q` green.
- [ ] `uv run ruff check && uv run ruff format --check` clean.
- [ ] Smoke against Nell's sandbox: heartbeat tick runs growth (no-op) without warnings; growth log empty; brain still ticks normally.
- [ ] Inject-test: when a fake crystallizer returns 1 `EmotionProposal` (test fixture), the scheduler writes it to vocabulary + log atomically; both files reflect the change.
