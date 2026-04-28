# Reflex Phase 2 — Emergent Arc Crystallization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the autonomous arc emergence + pruning system on top of Reflex Phase 1, where the brain proposes new arcs and prunes outgrown ones via LLM judgment, biographically logs all lifecycle events, and integrates with SP-7 bridge event broadcast.

**Architecture:** Crystallizer module (`brain/growth/crystallizers/reflex.py`) called from the existing growth scheduler under a new 7-day throttle. Throttle gates the whole growth tick (vocabulary too) so cadence is uniform. Single Claude CLI call per tick judges both emergence and pruning. Brain-emergence arcs are autonomously prunable; OG and user-authored arcs are inviolate. 15-day graveyard window for both removal types. New events publish on SP-7's `/events` WebSocket.

**Tech Stack:** Python 3.12, existing `brain/bridge/provider.LLMProvider` (Claude CLI default), existing `brain/growth/` scheduler + log + proposal patterns, existing `brain/health/attempt_heal.save_with_backup` for atomic writes, existing `brain/bridge/events` module-level publish (SP-7).

**Spec:** `docs/superpowers/specs/2026-04-28-reflex-phase-2-emergent-arc-crystallization-design.md`

**Implementation gate:** 2026-05-08 (Phase 1 must have ≥2 weeks of fire data on Nell's sandbox). Plan can be reviewed now; execution waits for the gate.

**Smoke-test discipline:** Every task ends with a smoke-test step that *runs the actual code against neighbouring systems* and observes output before commit. This is a hard gate, especially for tasks 6–8 where the dangerous paths live.

---

## File Structure

**New files:**

| File | Purpose |
|---|---|
| `brain/growth/crystallizers/reflex.py` | Crystallizer entry point — corpus assembly, prompt rendering, Claude call, response parsing, validation gates |
| `brain/growth/arc_storage.py` | Atomic read/write for `<persona>/removed_arcs.jsonl` and `<persona>/.last_arc_snapshot.json` |
| `tests/unit/brain/growth/test_reflex_crystallizer.py` | Unit tests for crystallizer module (~22 tests) |
| `tests/unit/brain/growth/test_arc_storage.py` | Unit tests for arc storage helpers (~6 tests) |
| `tests/integration/brain/growth/test_reflex_lifecycle.py` | Integration tests for scheduler + crystallizer end-to-end (~12 tests + 5 crash-recovery + 3 migration-safety) |
| `tests/integration/brain/growth/test_reflex_real_nell.py` | Real-Nell regression suite (3 tests, fixture-based) |
| `tests/fixtures/nell_sandbox_snapshot/` | Read-only snapshot of Nell's sandbox persona for regression tests |

**Modified files:**

| File | Change |
|---|---|
| `brain/engines/reflex.py` | Extend `ReflexArc` dataclass with `created_by` + `created_at` fields; backward-compat in `from_dict` |
| `brain/growth/proposal.py` | Add `ReflexArcProposal`, `ReflexPruneProposal`, `ReflexCrystallizationResult` dataclasses |
| `brain/growth/scheduler.py` | Add throttle predicate, extend `run_growth_tick` with reconciliation + reflex apply + snapshot update + bridge events |
| `brain/engines/daemon_state.py` | Add `last_growth_tick_at: datetime \| None` field |
| `brain/migrator/reflex_migrator.py` (or wherever reflex migration lives) | Stamp `created_by="og_migration"` + `created_at=<file_mtime or now>` on migrated arcs (idempotent) |
| `brain/cli.py` | Add `nell reflex removed list --persona X` read-only inspector |

---

## Task 1: Schema extensions — `ReflexArc` provenance + proposal dataclasses

**Files:**
- Modify: `brain/engines/reflex.py` (add `created_by`, `created_at` to `ReflexArc`)
- Modify: `brain/growth/proposal.py` (add three new dataclasses)
- Modify: existing reflex migrator (find via grep; stamp on migration)
- Test: `tests/unit/brain/engines/test_reflex.py` (extend existing)
- Test: `tests/unit/brain/growth/test_proposal.py` (extend existing or create)
- Test: `tests/unit/brain/migrator/test_reflex_migrator.py` (extend existing or create)

- [ ] **Step 1: Find the reflex migrator location**

```bash
cd /Users/hanamori/companion-emergence
grep -rn "OG_REFLEX_ARCS\|reflex_arcs.json" brain/migrator/ | head -10
```

Expected: locates `brain/migrator/reflex_migrator.py` or similar. Note the file path for Step 8.

- [ ] **Step 2: Write the failing test for `ReflexArc` with new fields**

Append to `tests/unit/brain/engines/test_reflex.py`:

```python
from datetime import UTC, datetime

from brain.engines.reflex import ReflexArc


def test_reflex_arc_has_created_by_field():
    arc = ReflexArc(
        name="creative_pitch",
        description="creative hunger overwhelmed",
        trigger={"creative_hunger": 8.0},
        days_since_human_min=0.0,
        cooldown_hours=48.0,
        action="generate_pitch",
        output_memory_type="reflex_pitch",
        prompt_template="...",
        created_by="brain_emergence",
        created_at=datetime(2026, 4, 28, tzinfo=UTC),
    )
    assert arc.created_by == "brain_emergence"
    assert arc.created_at == datetime(2026, 4, 28, tzinfo=UTC)


def test_reflex_arc_from_dict_backward_compat_no_created_by():
    """Loading an arc from old persona file (pre-Phase-2): missing created_by
    defaults to 'og_migration', missing created_at defaults to file_mtime
    sentinel (datetime(1970, 1, 1, tzinfo=UTC) since we don't have the mtime
    here — the loader uses mtime when reading from disk; from_dict on a raw
    dict gets the epoch sentinel)."""
    arc = ReflexArc.from_dict({
        "name": "x",
        "description": "y",
        "trigger": {"e": 5.0},
        "days_since_human_min": 0.0,
        "cooldown_hours": 12.0,
        "action": "z",
        "output_memory_type": "reflex_x",
        "prompt_template": "t",
    })
    assert arc.created_by == "og_migration"
    assert arc.created_at == datetime(1970, 1, 1, tzinfo=UTC)


def test_reflex_arc_from_dict_with_created_by():
    arc = ReflexArc.from_dict({
        "name": "x",
        "description": "y",
        "trigger": {"e": 5.0},
        "days_since_human_min": 0.0,
        "cooldown_hours": 12.0,
        "action": "z",
        "output_memory_type": "reflex_x",
        "prompt_template": "t",
        "created_by": "brain_emergence",
        "created_at": "2026-04-28T10:00:00+00:00",
    })
    assert arc.created_by == "brain_emergence"
    assert arc.created_at == datetime(2026, 4, 28, 10, 0, 0, tzinfo=UTC)


def test_reflex_arc_from_dict_rejects_invalid_created_by():
    import pytest
    with pytest.raises(ValueError, match="created_by"):
        ReflexArc.from_dict({
            "name": "x",
            "description": "y",
            "trigger": {"e": 5.0},
            "days_since_human_min": 0.0,
            "cooldown_hours": 12.0,
            "action": "z",
            "output_memory_type": "reflex_x",
            "prompt_template": "t",
            "created_by": "alien_source",  # not in allowed enum
            "created_at": "2026-04-28T10:00:00+00:00",
        })
```

- [ ] **Step 3: Run tests; confirm they fail**

Run: `uv run pytest tests/unit/brain/engines/test_reflex.py -k "created_by or backward_compat" -v`

Expected: `TypeError: __init__() got an unexpected keyword argument 'created_by'` or `AttributeError: 'ReflexArc' object has no attribute 'created_by'`.

- [ ] **Step 4: Extend `ReflexArc` and `from_dict`**

In `brain/engines/reflex.py`, modify the `ReflexArc` dataclass and its `from_dict`:

```python
from typing import Literal

_ALLOWED_CREATED_BY = ("og_migration", "brain_emergence", "user_authored")


@dataclass(frozen=True)
class ReflexArc:
    """Definition of one emotional-threshold-triggered arc."""

    name: str
    description: str
    trigger: Mapping[str, float]
    days_since_human_min: float
    cooldown_hours: float
    action: str
    output_memory_type: str
    prompt_template: str
    created_by: Literal["og_migration", "brain_emergence", "user_authored"] = "og_migration"
    created_at: datetime = field(default_factory=lambda: datetime(1970, 1, 1, tzinfo=UTC))

    @classmethod
    def from_dict(cls, data: dict) -> ReflexArc:
        """Construct an arc from a dict. Raises KeyError/ValueError on invalid input.

        Backward-compat: missing `created_by` defaults to "og_migration"; missing
        `created_at` defaults to the epoch sentinel (datetime(1970, 1, 1, tzinfo=UTC)).
        Callers loading from disk should override with the file mtime when the
        sentinel is encountered.
        """
        required = (
            "name", "description", "trigger", "days_since_human_min",
            "cooldown_hours", "action", "output_memory_type", "prompt_template",
        )
        for key in required:
            if key not in data:
                raise KeyError(f"ReflexArc missing required key: {key!r}")

        trigger_raw = data["trigger"]
        if not isinstance(trigger_raw, dict) or not trigger_raw:
            raise ValueError(f"ReflexArc {data.get('name')!r}: trigger must be non-empty dict")
        trigger = {str(k): float(v) for k, v in trigger_raw.items()}

        created_by = data.get("created_by", "og_migration")
        if created_by not in _ALLOWED_CREATED_BY:
            raise ValueError(
                f"ReflexArc {data.get('name')!r}: created_by must be one of "
                f"{_ALLOWED_CREATED_BY}, got {created_by!r}"
            )

        created_at_raw = data.get("created_at")
        if created_at_raw is None:
            created_at = datetime(1970, 1, 1, tzinfo=UTC)
        else:
            created_at = parse_iso_utc(str(created_at_raw))

        return cls(
            name=str(data["name"]),
            description=str(data["description"]),
            trigger=trigger,
            days_since_human_min=float(data["days_since_human_min"]),
            cooldown_hours=float(data["cooldown_hours"]),
            action=str(data["action"]),
            output_memory_type=str(data["output_memory_type"]),
            prompt_template=str(data["prompt_template"]),
            created_by=created_by,
            created_at=created_at,
        )

    def to_dict(self) -> dict:
        """Serialize for writing back to reflex_arcs.json."""
        return {
            "name": self.name,
            "description": self.description,
            "trigger": dict(self.trigger),
            "days_since_human_min": self.days_since_human_min,
            "cooldown_hours": self.cooldown_hours,
            "action": self.action,
            "output_memory_type": self.output_memory_type,
            "prompt_template": self.prompt_template,
            "created_by": self.created_by,
            "created_at": iso_utc(self.created_at),
        }
```

You'll need `from dataclasses import dataclass, field` (add `field` if missing) and `from typing import Literal`.

- [ ] **Step 5: Run tests; confirm they pass**

Run: `uv run pytest tests/unit/brain/engines/test_reflex.py -k "created_by or backward_compat" -v`

Expected: 4 passed.

- [ ] **Step 6: Run the full reflex test suite to verify nothing else broke**

Run: `uv run pytest tests/unit/brain/engines/test_reflex.py -v`

Expected: all existing tests still pass (older tests construct `ReflexArc` directly without `created_by`/`created_at` — the field defaults make this backward-compatible).

- [ ] **Step 7: Write failing tests for the three new proposal dataclasses**

Create or append to `tests/unit/brain/growth/test_proposal.py`:

```python
"""Tests for growth proposal dataclasses."""
from __future__ import annotations

from brain.growth.proposal import (
    ReflexArcProposal,
    ReflexPruneProposal,
    ReflexCrystallizationResult,
)


def test_reflex_arc_proposal_round_trip():
    p = ReflexArcProposal(
        name="manuscript_obsession",
        description="creative drive narrowed to one project",
        trigger={"creative_hunger": 7.0, "love": 6.0},
        cooldown_hours=24.0,
        output_memory_type="reflex_pitch",
        prompt_template="You are {persona_name}. ...",
        reasoning="Over the past month I've fired creative_pitch four times "
                  "but each one has been about the same novel.",
    )
    assert p.name == "manuscript_obsession"
    assert p.days_since_human_min == 0.0  # default
    # frozen — immutable
    import dataclasses
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        p.name = "different"  # type: ignore[misc]


def test_reflex_prune_proposal_minimal():
    p = ReflexPruneProposal(
        name="loneliness_journal",
        reasoning="I'm not in that place anymore — what was loneliness has "
                  "become something else, less private.",
    )
    assert p.name == "loneliness_journal"
    assert p.reasoning.startswith("I'm not in that place")


def test_reflex_crystallization_result_holds_both_lists():
    result = ReflexCrystallizationResult(
        emergences=[
            ReflexArcProposal(
                name="x", description="y", trigger={"e": 5.0},
                cooldown_hours=12.0, output_memory_type="reflex_x",
                prompt_template="t", reasoning="r",
            )
        ],
        prunings=[
            ReflexPruneProposal(name="z", reasoning="r2")
        ],
    )
    assert len(result.emergences) == 1
    assert len(result.prunings) == 1


def test_reflex_crystallization_result_empty():
    result = ReflexCrystallizationResult(emergences=[], prunings=[])
    assert result.emergences == []
    assert result.prunings == []
```

- [ ] **Step 8: Run tests; confirm they fail**

Run: `uv run pytest tests/unit/brain/growth/test_proposal.py -k "reflex" -v`

Expected: `ImportError: cannot import name 'ReflexArcProposal' from 'brain.growth.proposal'`.

- [ ] **Step 9: Add the three new dataclasses to `brain/growth/proposal.py`**

Append to `brain/growth/proposal.py`:

```python
from collections.abc import Mapping


@dataclass(frozen=True)
class ReflexArcProposal:
    """One arc the reflex crystallizer has decided to add.

    Fully specifies the arc the brain wants to crystallize, including its
    own voice (prompt_template). The scheduler validates and applies; no
    candidate queue, no human approval gate. Per principle audit
    2026-04-25: the brain has agency.

    Attributes:
        name: snake_case identifier, must pass `^[a-z][a-z0-9_]*$`.
        description: One-sentence kind of moment this captures.
        trigger: Emotion name -> threshold (5.0..10.0).
        cooldown_hours: Minimum hours between fires (>= 12).
        output_memory_type: e.g. "reflex_journal", "reflex_pitch".
        prompt_template: format_map-renderable string in persona voice.
        reasoning: One-paragraph why-this-pattern-is-real, brain's articulation.
        days_since_human_min: 0.0 disables the gate (default).
    """

    name: str
    description: str
    trigger: Mapping[str, float]
    cooldown_hours: float
    output_memory_type: str
    prompt_template: str
    reasoning: str
    days_since_human_min: float = 0.0


@dataclass(frozen=True)
class ReflexPruneProposal:
    """One brain-emergence arc the brain has decided to prune.

    Pruning is autonomous only for arcs with created_by="brain_emergence".
    OG-migration and user-authored arcs are protected and skipped at gate
    P2 even if proposed.
    """

    name: str
    reasoning: str


@dataclass(frozen=True)
class ReflexCrystallizationResult:
    """Outcome of one crystallizer pass — both emergences and prunings.

    Both lists may be empty (the brain ticked but nothing crystallized);
    that's a valid outcome and `last_growth_tick_at` still updates.
    """

    emergences: list[ReflexArcProposal]
    prunings: list[ReflexPruneProposal]
```

- [ ] **Step 10: Run tests; confirm they pass**

Run: `uv run pytest tests/unit/brain/growth/test_proposal.py -v`

Expected: all reflex-proposal tests pass; existing `EmotionProposal` tests still pass.

- [ ] **Step 11: Find and update the reflex migrator**

From Step 1 you should know the migrator path. Read it and find where it constructs reflex arcs from OG data.

```bash
cat brain/migrator/<migrator_file>.py | head -50
```

Locate where arcs are created. Modify the construction to include `created_by="og_migration"` and `created_at=datetime.now(UTC)`. If the migrator writes JSON directly (not via `ReflexArc.to_dict()`), add the two fields explicitly to the dict.

The exact code change depends on the migrator's structure. Pattern:

```python
# Before
arc_dict = {
    "name": og_name,
    "description": og_description,
    # ... other fields ...
}

# After
arc_dict = {
    "name": og_name,
    "description": og_description,
    # ... other fields ...
    "created_by": "og_migration",
    "created_at": datetime.now(UTC).isoformat(),
}
```

Add necessary imports (`from datetime import UTC, datetime`).

- [ ] **Step 12: Write a failing test for migrator stamping**

In `tests/unit/brain/migrator/test_reflex_migrator.py` (create if needed):

```python
"""Tests for reflex migrator's Phase 2 created_by/created_at stamping."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def test_migrator_stamps_og_migration_on_all_arcs(tmp_path: Path):
    """After migration, every arc in reflex_arcs.json carries created_by='og_migration'."""
    # Setup: invoke the migrator on a synthetic OG state.
    # The actual call shape depends on the migrator; pseudo-pattern:
    #
    #   from brain.migrator.<migrator_module> import migrate_reflex_arcs
    #   migrate_reflex_arcs(persona_dir=tmp_path, source=<og_dict_or_path>)
    #
    # Replace below with the actual call from your migrator.
    from brain.migrator.reflex_migrator import migrate_reflex_arcs  # adjust import
    persona_dir = tmp_path / "test-persona"
    persona_dir.mkdir()
    migrate_reflex_arcs(persona_dir=persona_dir)  # adjust args to match real signature

    arcs_file = persona_dir / "reflex_arcs.json"
    assert arcs_file.exists()
    data = json.loads(arcs_file.read_text())
    arcs = data["arcs"]
    assert len(arcs) >= 4  # at least the 4 default OG arcs
    for arc in arcs:
        assert arc["created_by"] == "og_migration", f"arc {arc['name']!r} missing stamp"
        # created_at must be parseable ISO
        parsed = datetime.fromisoformat(arc["created_at"])
        assert parsed.tzinfo is not None  # tz-aware


def test_migrator_idempotent_no_double_stamp(tmp_path: Path):
    """Re-migrating doesn't change created_by on existing arcs."""
    from brain.migrator.reflex_migrator import migrate_reflex_arcs
    persona_dir = tmp_path / "test-persona"
    persona_dir.mkdir()

    migrate_reflex_arcs(persona_dir=persona_dir)
    first = json.loads((persona_dir / "reflex_arcs.json").read_text())
    first_created_at = {arc["name"]: arc["created_at"] for arc in first["arcs"]}

    # Re-run — should be a no-op or at most refresh non-stamp fields.
    migrate_reflex_arcs(persona_dir=persona_dir)
    second = json.loads((persona_dir / "reflex_arcs.json").read_text())

    for arc in second["arcs"]:
        assert arc["created_by"] == "og_migration"
        # Critically: created_at should NOT be refreshed on re-migration.
        assert arc["created_at"] == first_created_at[arc["name"]], (
            f"re-migration changed created_at for {arc['name']!r} — should be idempotent"
        )
```

- [ ] **Step 13: Run tests; expect them to fail until migrator is updated**

Run: `uv run pytest tests/unit/brain/migrator/test_reflex_migrator.py -k "stamp or idempotent" -v`

Expected: `KeyError` on `arc["created_by"]` if stamping isn't yet wired, OR test passes if Step 11 is already done. If the import path is wrong (no `reflex_migrator` module), inspect the actual migrator module from Step 1's grep output and adjust the test's import.

- [ ] **Step 14: If tests fail, finish wiring the stamping in the migrator**

Make sure Step 11's edits are saved. For idempotency: re-migration must read the existing `reflex_arcs.json` first, preserve `created_at` for already-stamped arcs, and only stamp arcs that weren't previously stamped.

```python
# In migrator (idempotency block, before writing):
existing_arcs_path = persona_dir / "reflex_arcs.json"
if existing_arcs_path.exists():
    existing = json.loads(existing_arcs_path.read_text())
    existing_by_name = {a["name"]: a for a in existing.get("arcs", [])}
    for arc in arcs_to_write:
        if arc["name"] in existing_by_name and "created_at" in existing_by_name[arc["name"]]:
            arc["created_at"] = existing_by_name[arc["name"]]["created_at"]
            arc["created_by"] = existing_by_name[arc["name"]].get("created_by", "og_migration")
```

Run again: `uv run pytest tests/unit/brain/migrator/test_reflex_migrator.py -v`

Expected: all tests pass.

- [ ] **Step 15: Smoke test against Nell's actual sandbox (read-only check)**

```bash
NELLBRAIN_HOME=/tmp/sp7-final-smoke  # or another tmp NELLBRAIN_HOME
mkdir -p /tmp/reflex-phase2-smoke/personas/test
cp -r "/Users/hanamori/Library/Application Support/companion-emergence/personas/nell.sandbox/reflex_arcs.json" /tmp/reflex-phase2-smoke/personas/test/
NELLBRAIN_HOME=/tmp/reflex-phase2-smoke uv run python -c "
import json
from pathlib import Path
from brain.engines.reflex import ReflexArc

p = Path('/tmp/reflex-phase2-smoke/personas/test/reflex_arcs.json')
data = json.loads(p.read_text())
for arc_data in data['arcs']:
    arc = ReflexArc.from_dict(arc_data)
    print(f'{arc.name}: created_by={arc.created_by}, created_at={arc.created_at}')
"
```

Expected: All 8 of Nell's arcs load; if her live `reflex_arcs.json` doesn't yet have `created_by` (which it won't on first deploy), each prints `created_by=og_migration, created_at=1970-01-01 00:00:00+00:00` (the epoch sentinel). That's correct backward-compat behavior. After running the migrator on her real data later, the sentinel becomes a real timestamp.

- [ ] **Step 16: Commit**

```bash
cd /Users/hanamori/companion-emergence
git checkout -b reflex-phase-2  # or use a worktree per the workflow skill
git add brain/engines/reflex.py brain/growth/proposal.py brain/migrator/ tests/unit/brain/engines/test_reflex.py tests/unit/brain/growth/test_proposal.py tests/unit/brain/migrator/test_reflex_migrator.py
git commit -m "feat(reflex-phase-2): add provenance to ReflexArc + proposal dataclasses

- ReflexArc gains created_by ('og_migration'|'brain_emergence'|'user_authored')
  and created_at fields. Backward-compat: from_dict defaults to og_migration
  + epoch sentinel when fields are absent (legacy persona files).
- Add ReflexArcProposal, ReflexPruneProposal, ReflexCrystallizationResult
  to brain/growth/proposal.py.
- Migrator stamps created_by='og_migration' + created_at on migration;
  idempotent (re-migration preserves created_at)."
```

---

## Task 2: Growth log event types — `arc_added`, `arc_pruned_by_brain`, etc.

**Files:**
- Modify: `brain/growth/log.py` (add helper constructors + accept new `type` values)
- Test: `tests/unit/brain/growth/test_log.py` (extend existing or create)

The existing `GrowthLogEvent` schema has the right shape — `type` is already a free-form string. We just need helper constructors so callers don't have to remember which fields make sense for arc events vs emotion events. For arc events, vocabulary fields default to nulls/empties.

- [ ] **Step 1: Write failing tests for the five new event types**

Append to or create `tests/unit/brain/growth/test_log.py`:

```python
"""Tests for growth log arc event types (Phase 2 reflex emergence)."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from brain.growth.log import (
    GrowthLogEvent,
    append_growth_event,
    arc_added_event,
    arc_pruned_by_brain_event,
    arc_proposal_dropped_event,
    arc_rejected_user_removed_event,
    arc_removed_by_user_event,
    read_growth_log,
)


def _now():
    return datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC)


def test_arc_added_event_helper():
    event = arc_added_event(
        timestamp=_now(),
        name="manuscript_obsession",
        description="creative drive narrowed to one project",
        reasoning="Over the past month I've fired creative_pitch four times",
        created_by="brain_emergence",
    )
    assert event.type == "arc_added"
    assert event.name == "manuscript_obsession"
    assert event.description == "creative drive narrowed to one project"
    assert event.reason == "Over the past month I've fired creative_pitch four times"
    assert event.decay_half_life_days is None
    assert event.evidence_memory_ids == ()
    assert event.score == 0.0
    assert event.relational_context == "brain_emergence"  # encodes created_by


def test_arc_pruned_by_brain_event_helper():
    event = arc_pruned_by_brain_event(
        timestamp=_now(),
        name="loneliness_journal",
        description="loneliness hit threshold — wrote a journal entry",
        reasoning="I'm not in that place anymore",
    )
    assert event.type == "arc_pruned_by_brain"
    assert event.name == "loneliness_journal"
    assert event.reason == "I'm not in that place anymore"


def test_arc_removed_by_user_event_helper():
    event = arc_removed_by_user_event(
        timestamp=_now(),
        name="loneliness_journal",
        description="loneliness hit threshold — wrote a journal entry",
    )
    assert event.type == "arc_removed_by_user"
    assert event.name == "loneliness_journal"
    assert event.reason == "user edited reflex_arcs.json"


def test_arc_rejected_user_removed_event_helper():
    event = arc_rejected_user_removed_event(
        timestamp=_now(),
        name="loneliness_journal",
        reasoning="brain re-proposed; honoring user removal",
    )
    assert event.type == "arc_rejected_user_removed"


def test_arc_proposal_dropped_event_helper():
    event = arc_proposal_dropped_event(
        timestamp=_now(),
        name="bad_arc",
        reasoning="trigger overlap with existing arc creative_pitch",
    )
    assert event.type == "arc_proposal_dropped"


def test_arc_events_round_trip_through_jsonl(tmp_path: Path):
    """Write each arc event type to a real growth log file and read back."""
    log_path = tmp_path / "emotion_growth.log.jsonl"
    events = [
        arc_added_event(
            timestamp=_now(), name="x", description="d",
            reasoning="r", created_by="brain_emergence",
        ),
        arc_pruned_by_brain_event(timestamp=_now(), name="y", description="d", reasoning="r"),
        arc_removed_by_user_event(timestamp=_now(), name="z", description="d"),
        arc_rejected_user_removed_event(timestamp=_now(), name="w", reasoning="r"),
        arc_proposal_dropped_event(timestamp=_now(), name="v", reasoning="r"),
    ]
    for e in events:
        append_growth_event(log_path, e)

    read_back = read_growth_log(log_path)
    assert len(read_back) == 5
    assert [e.type for e in read_back] == [
        "arc_added", "arc_pruned_by_brain", "arc_removed_by_user",
        "arc_rejected_user_removed", "arc_proposal_dropped",
    ]
```

- [ ] **Step 2: Run tests; confirm they fail**

Run: `uv run pytest tests/unit/brain/growth/test_log.py -v`

Expected: `ImportError: cannot import name 'arc_added_event' from 'brain.growth.log'`.

- [ ] **Step 3: Add the helper constructors to `brain/growth/log.py`**

Append to `brain/growth/log.py`:

```python
def arc_added_event(
    *,
    timestamp: datetime,
    name: str,
    description: str,
    reasoning: str,
    created_by: str,  # "brain_emergence" | "user_authored" | "og_migration"
) -> GrowthLogEvent:
    """Constructor for arc_added events. Fills GrowthLogEvent's vocabulary
    fields with sensible nulls (decay_half_life_days=None, evidence=(), score=0.0)
    and stashes created_by in relational_context so the brain can read it back.

    The brain will see this event in its corpus next tick — the relational_context
    field is repurposed here to carry provenance ("brain_emergence" means
    *I* added this arc; "user_authored" means Hana did).
    """
    return GrowthLogEvent(
        timestamp=timestamp,
        type="arc_added",
        name=name,
        description=description,
        decay_half_life_days=None,
        reason=reasoning,
        evidence_memory_ids=(),
        score=0.0,
        relational_context=created_by,
    )


def arc_pruned_by_brain_event(
    *, timestamp: datetime, name: str, description: str, reasoning: str,
) -> GrowthLogEvent:
    """Constructor for arc_pruned_by_brain events.

    Description carries the pruned arc's description for biographical readability —
    so the future brain reading its log doesn't have to cross-reference the
    graveyard to know what was pruned.
    """
    return GrowthLogEvent(
        timestamp=timestamp,
        type="arc_pruned_by_brain",
        name=name,
        description=description,
        decay_half_life_days=None,
        reason=reasoning,
        evidence_memory_ids=(),
        score=0.0,
        relational_context=None,
    )


def arc_removed_by_user_event(
    *, timestamp: datetime, name: str, description: str,
) -> GrowthLogEvent:
    """Constructor for arc_removed_by_user events. Reason is hardcoded
    because user file-edit removals don't carry explicit reasoning."""
    return GrowthLogEvent(
        timestamp=timestamp,
        type="arc_removed_by_user",
        name=name,
        description=description,
        decay_half_life_days=None,
        reason="user edited reflex_arcs.json",
        evidence_memory_ids=(),
        score=0.0,
        relational_context=None,
    )


def arc_rejected_user_removed_event(
    *, timestamp: datetime, name: str, reasoning: str,
) -> GrowthLogEvent:
    """Constructor for arc_rejected_user_removed events — fired when the
    brain proposes an arc whose name is in the 15-day graveyard window."""
    return GrowthLogEvent(
        timestamp=timestamp,
        type="arc_rejected_user_removed",
        name=name,
        description="",  # brain didn't get to articulate description; rejected
        decay_half_life_days=None,
        reason=reasoning,
        evidence_memory_ids=(),
        score=0.0,
        relational_context=None,
    )


def arc_proposal_dropped_event(
    *, timestamp: datetime, name: str, reasoning: str,
) -> GrowthLogEvent:
    """Constructor for arc_proposal_dropped events — generic gate-rejection
    log entry (gate 1 char-validity, gate 4 unknown emotion, gate 6/7
    threshold/cooldown floor, gate 8 trigger overlap, gate 9 cap hit)."""
    return GrowthLogEvent(
        timestamp=timestamp,
        type="arc_proposal_dropped",
        name=name,
        description="",
        decay_half_life_days=None,
        reason=reasoning,
        evidence_memory_ids=(),
        score=0.0,
        relational_context=None,
    )
```

- [ ] **Step 4: Run tests; confirm they pass**

Run: `uv run pytest tests/unit/brain/growth/test_log.py -v`

Expected: 6 passed (5 type tests + 1 round-trip).

- [ ] **Step 5: Smoke test — read back from a real growth log and inspect**

```bash
uv run python -c "
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from brain.growth.log import (
    arc_added_event, arc_pruned_by_brain_event, append_growth_event, read_growth_log,
)

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / 'log.jsonl'
    e1 = arc_added_event(
        timestamp=datetime.now(UTC), name='manuscript_obsession',
        description='creative drive narrowed to one project',
        reasoning='Fired creative_pitch four times this month, all about the novel',
        created_by='brain_emergence',
    )
    append_growth_event(p, e1)
    e2 = arc_pruned_by_brain_event(
        timestamp=datetime.now(UTC), name='loneliness_journal',
        description='loneliness hit threshold',
        reasoning=\"I'm not in that place anymore\",
    )
    append_growth_event(p, e2)
    print(p.read_text())
    print('---')
    for ev in read_growth_log(p):
        print(f'{ev.type}: {ev.name} — {ev.reason}')
"
```

Expected: prints two JSON lines + a parsed summary. Confirms the log format is consumable.

- [ ] **Step 6: Commit**

```bash
git add brain/growth/log.py tests/unit/brain/growth/test_log.py
git commit -m "feat(reflex-phase-2): growth log helpers for arc lifecycle events

Add five constructor helpers to brain/growth/log.py for the new
GrowthLogEvent types Phase 2 emits:
  - arc_added_event (created_by stashed in relational_context)
  - arc_pruned_by_brain_event
  - arc_removed_by_user_event
  - arc_rejected_user_removed_event
  - arc_proposal_dropped_event

The base GrowthLogEvent schema is unchanged — new types just fill
vocabulary-specific fields with nulls/empties. Brain reads its full
log into the crystallizer corpus, so these events become biographical
context for future ticks."
```

---

## Task 3: Arc storage — `removed_arcs.jsonl` + `.last_arc_snapshot.json`

**Files:**
- Create: `brain/growth/arc_storage.py`
- Create: `tests/unit/brain/growth/test_arc_storage.py`

This module owns atomic read/write for the two new persona-state files. The graveyard is JSONL append-only (mirroring `emotion_growth.log.jsonl`); the snapshot is a single JSON file written atomically via `save_with_backup`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/brain/growth/test_arc_storage.py`:

```python
"""Tests for arc storage helpers — graveyard + snapshot."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.engines.reflex import ReflexArc
from brain.growth.arc_storage import (
    append_removed_arc,
    read_arc_snapshot,
    read_removed_arcs,
    recently_removed_names,
    write_arc_snapshot,
)


def _make_arc(name: str = "test_arc", created_by: str = "brain_emergence") -> ReflexArc:
    return ReflexArc(
        name=name,
        description=f"description of {name}",
        trigger={"vulnerability": 8.0},
        days_since_human_min=0.0,
        cooldown_hours=12.0,
        action="generate_journal",
        output_memory_type="reflex_journal",
        prompt_template="You are nell. {emotion_summary}",
        created_by=created_by,
        created_at=datetime(2026, 4, 28, tzinfo=UTC),
    )


def test_graveyard_round_trip(tmp_path: Path):
    arc = _make_arc("loneliness_journal")
    append_removed_arc(
        tmp_path,
        arc=arc,
        removed_at=datetime(2026, 4, 28, tzinfo=UTC),
        removed_by="user_edit",
        reasoning=None,
    )
    entries = read_removed_arcs(tmp_path)
    assert len(entries) == 1
    e = entries[0]
    assert e["name"] == "loneliness_journal"
    assert e["removed_by"] == "user_edit"
    assert e["reasoning"] is None
    assert e["trigger_snapshot"] == {"vulnerability": 8.0}
    assert e["description_snapshot"] == "description of loneliness_journal"
    assert "prompt_template_snapshot" in e


def test_graveyard_appends_multiple(tmp_path: Path):
    for i in range(3):
        append_removed_arc(
            tmp_path,
            arc=_make_arc(f"arc_{i}"),
            removed_at=datetime(2026, 4, 28, tzinfo=UTC),
            removed_by="brain_self_prune",
            reasoning=f"reason {i}",
        )
    entries = read_removed_arcs(tmp_path)
    assert [e["name"] for e in entries] == ["arc_0", "arc_1", "arc_2"]
    assert all(e["removed_by"] == "brain_self_prune" for e in entries)


def test_recently_removed_names_window(tmp_path: Path):
    """Only entries within `grace_days` count as recently-removed."""
    now = datetime(2026, 4, 28, tzinfo=UTC)
    append_removed_arc(  # 5 days ago — within window
        tmp_path, arc=_make_arc("recent"),
        removed_at=now - timedelta(days=5),
        removed_by="user_edit", reasoning=None,
    )
    append_removed_arc(  # 20 days ago — outside 15d window
        tmp_path, arc=_make_arc("ancient"),
        removed_at=now - timedelta(days=20),
        removed_by="user_edit", reasoning=None,
    )
    names = recently_removed_names(tmp_path, now=now, grace_days=15)
    assert names == {"recent"}


def test_snapshot_round_trip(tmp_path: Path):
    arcs = [_make_arc("a"), _make_arc("b", created_by="og_migration")]
    write_arc_snapshot(tmp_path, arcs=arcs, snapshot_at=datetime(2026, 4, 28, tzinfo=UTC))
    read_back = read_arc_snapshot(tmp_path)
    assert read_back is not None
    assert {a.name for a in read_back} == {"a", "b"}
    by_name = {a.name: a for a in read_back}
    assert by_name["a"].created_by == "brain_emergence"
    assert by_name["b"].created_by == "og_migration"


def test_snapshot_returns_none_when_missing(tmp_path: Path):
    assert read_arc_snapshot(tmp_path) is None


def test_graveyard_handles_corrupt_lines(tmp_path: Path):
    """A corrupt line in removed_arcs.jsonl should be skipped, not crash."""
    g = tmp_path / "removed_arcs.jsonl"
    # Write valid + corrupt + valid
    g.write_text(
        '{"name": "valid1", "removed_at": "2026-04-28T00:00:00+00:00", '
        '"removed_by": "user_edit", "reasoning": null, "trigger_snapshot": {}, '
        '"description_snapshot": "", "prompt_template_snapshot": ""}\n'
        'this is not json\n'
        '{"name": "valid2", "removed_at": "2026-04-28T00:00:00+00:00", '
        '"removed_by": "user_edit", "reasoning": null, "trigger_snapshot": {}, '
        '"description_snapshot": "", "prompt_template_snapshot": ""}\n'
    )
    entries = read_removed_arcs(tmp_path)
    assert [e["name"] for e in entries] == ["valid1", "valid2"]
```

- [ ] **Step 2: Run tests; confirm they fail**

Run: `uv run pytest tests/unit/brain/growth/test_arc_storage.py -v`

Expected: `ModuleNotFoundError: No module named 'brain.growth.arc_storage'`.

- [ ] **Step 3: Implement `brain/growth/arc_storage.py`**

```python
"""Arc lifecycle storage — graveyard + snapshot.

Two persistent files per persona:

  removed_arcs.jsonl   — append-only graveyard, one JSON object per line.
                         Captures full arc state at removal so data is
                         recoverable even if reflex_arcs.json is nuked.

  .last_arc_snapshot.json — single JSON file with the post-tick arc set,
                            read at the start of each tick to detect user
                            file-edits via diff. Atomic write via
                            save_with_backup.

The graveyard is the source of truth for "did Hana remove this in the last
15 days?" — gate 3 in §6 of the spec consults `recently_removed_names`.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from brain.engines.reflex import ReflexArc
from brain.health.adaptive import compute_treatment
from brain.health.attempt_heal import attempt_heal, save_with_backup
from brain.health.jsonl_reader import read_jsonl_skipping_corrupt
from brain.utils.time import iso_utc

logger = logging.getLogger(__name__)

GRAVEYARD_FILENAME = "removed_arcs.jsonl"
SNAPSHOT_FILENAME = ".last_arc_snapshot.json"


def append_removed_arc(
    persona_dir: Path,
    *,
    arc: ReflexArc,
    removed_at: datetime,
    removed_by: str,  # "user_edit" | "brain_self_prune"
    reasoning: str | None,
) -> None:
    """Atomic append to removed_arcs.jsonl.

    Snapshot fields capture the full arc state — recovery should not require
    cross-referencing other files.
    """
    if removed_by not in ("user_edit", "brain_self_prune"):
        raise ValueError(
            f"removed_by must be 'user_edit' or 'brain_self_prune', got {removed_by!r}"
        )
    path = persona_dir / GRAVEYARD_FILENAME
    entry = {
        "name": arc.name,
        "removed_at": iso_utc(removed_at),
        "removed_by": removed_by,
        "reasoning": reasoning,
        "trigger_snapshot": dict(arc.trigger),
        "description_snapshot": arc.description,
        "prompt_template_snapshot": arc.prompt_template,
    }
    line = json.dumps(entry) + "\n"
    existing = path.read_bytes() if path.exists() else b""
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_bytes(existing + line.encode("utf-8"))
    os.replace(tmp, path)


def read_removed_arcs(persona_dir: Path) -> list[dict]:
    """Read all graveyard entries oldest-first. Skips corrupt lines."""
    path = persona_dir / GRAVEYARD_FILENAME
    if not path.exists():
        return []
    return list(read_jsonl_skipping_corrupt(path))


def recently_removed_names(
    persona_dir: Path, *, now: datetime, grace_days: float,
) -> set[str]:
    """Return names removed within the grace window. Spec gate 3 uses this.

    Entries with malformed timestamps are skipped (not raised). The brain
    sees this set in its corpus and the scheduler enforces it as a gate.
    """
    cutoff = now - timedelta(days=grace_days)
    names: set[str] = set()
    for entry in read_removed_arcs(persona_dir):
        ts_raw = entry.get("removed_at")
        if not isinstance(ts_raw, str):
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            continue
        if ts >= cutoff:
            name = entry.get("name")
            if isinstance(name, str):
                names.add(name)
    return names


def write_arc_snapshot(
    persona_dir: Path, *, arcs: list[ReflexArc], snapshot_at: datetime,
) -> None:
    """Atomic write of .last_arc_snapshot.json via save_with_backup."""
    path = persona_dir / SNAPSHOT_FILENAME
    payload = {
        "version": 1,
        "snapshot_at": iso_utc(snapshot_at),
        "arcs": [arc.to_dict() for arc in arcs],
    }
    treatment = compute_treatment(persona_dir, SNAPSHOT_FILENAME)
    save_with_backup(path, payload, backup_count=treatment.backup_count)


def read_arc_snapshot(persona_dir: Path) -> list[ReflexArc] | None:
    """Read .last_arc_snapshot.json. Returns None if missing.

    On corruption: attempt_heal restores from .bak rotation. If all backups
    corrupt, returns None (treated as "first run, no prior snapshot").
    """
    path = persona_dir / SNAPSHOT_FILENAME
    if not path.exists():
        return None

    def _default() -> dict:
        return {"version": 1, "snapshot_at": "", "arcs": []}

    data, anomaly = attempt_heal(path, _default)
    if anomaly is not None:
        logger.warning(
            "arc snapshot at %s anomaly %s (action=%s)",
            path, anomaly.kind, anomaly.action,
        )
    arcs_raw = data.get("arcs", [])
    if not arcs_raw:
        return None
    arcs: list[ReflexArc] = []
    for arc_data in arcs_raw:
        try:
            arcs.append(ReflexArc.from_dict(arc_data))
        except (KeyError, ValueError) as exc:
            logger.warning("skipping snapshot arc with schema error: %s", exc)
    return arcs
```

- [ ] **Step 4: Run tests; confirm they pass**

Run: `uv run pytest tests/unit/brain/growth/test_arc_storage.py -v`

Expected: 6 passed.

- [ ] **Step 5: Smoke test — round-trip with real ReflexArc instances**

```bash
uv run python -c "
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from brain.engines.reflex import ReflexArc
from brain.growth.arc_storage import (
    append_removed_arc, read_removed_arcs, recently_removed_names,
    write_arc_snapshot, read_arc_snapshot,
)

with tempfile.TemporaryDirectory() as td:
    p = Path(td)
    arc = ReflexArc(
        name='loneliness_journal', description='loneliness hit threshold',
        trigger={'loneliness': 7.0}, days_since_human_min=2.0, cooldown_hours=24.0,
        action='generate_journal', output_memory_type='reflex_journal',
        prompt_template='You are nell. Loneliness is at {loneliness}/10.',
        created_by='og_migration', created_at=datetime(2026, 4, 23, tzinfo=UTC),
    )
    now = datetime.now(UTC)
    append_removed_arc(p, arc=arc, removed_at=now - timedelta(days=3),
                       removed_by='user_edit', reasoning=None)
    print('graveyard:', read_removed_arcs(p))
    print('recent (15d):', recently_removed_names(p, now=now, grace_days=15))
    write_arc_snapshot(p, arcs=[arc], snapshot_at=now)
    print('snapshot:', [a.name for a in read_arc_snapshot(p)])
"
```

Expected: prints the graveyard entry, the recently-removed name set `{'loneliness_journal'}`, and the snapshot read-back.

- [ ] **Step 6: Commit**

```bash
git add brain/growth/arc_storage.py tests/unit/brain/growth/test_arc_storage.py
git commit -m "feat(reflex-phase-2): arc storage — graveyard + snapshot helpers

brain/growth/arc_storage.py owns:
  - removed_arcs.jsonl: append-only graveyard with full arc snapshots
    (data recoverable even if reflex_arcs.json is nuked)
  - .last_arc_snapshot.json: post-tick state, atomic write via
    save_with_backup, used for user-edit detection on next tick
  - recently_removed_names(grace_days=15): gate 3's data source
    rejecting brain re-proposals within graveyard window

Corrupt graveyard lines are skipped (not raised). Corrupt snapshot
file falls through .bak rotation via attempt_heal."
```

---

## Task 4: Daemon state — `last_growth_tick_at` + throttle predicate

**Files:**
- Modify: `brain/engines/daemon_state.py` (add `last_growth_tick_at` field)
- Modify: `brain/growth/scheduler.py` (add `_should_run_growth_tick` predicate)
- Test: `tests/unit/brain/engines/test_daemon_state.py` (extend existing)
- Test: `tests/unit/brain/growth/test_scheduler.py` (extend existing)

The throttle gates the *whole* growth tick — vocabulary AND reflex. This is a behavior change for Phase 2a vocab (every-tick → weekly). Spec §2 covers the rationale.

- [ ] **Step 1: Inspect the current `daemon_state.py` schema**

```bash
grep -n "^class\|^def\|^@dataclass" /Users/hanamori/companion-emergence/brain/engines/daemon_state.py
```

Note the dataclass definition — you'll add one field, preserve all others, and update the read/write helpers.

- [ ] **Step 2: Write a failing test for the new field**

Append to `tests/unit/brain/engines/test_daemon_state.py`:

```python
from datetime import UTC, datetime
from pathlib import Path

from brain.engines.daemon_state import (
    DaemonState, load_daemon_state, save_daemon_state,
)


def test_daemon_state_has_last_growth_tick_at_field():
    s = DaemonState()  # default-constructed
    assert hasattr(s, "last_growth_tick_at")
    assert s.last_growth_tick_at is None  # default


def test_daemon_state_round_trip_with_last_growth_tick_at(tmp_path: Path):
    persona_dir = tmp_path
    ts = datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC)
    state = DaemonState(last_growth_tick_at=ts)
    # Other DaemonState fields default; if your DaemonState has more
    # required fields, fill them with their existing defaults.
    save_daemon_state(persona_dir, state)
    state2, _anom = load_daemon_state(persona_dir)
    assert state2.last_growth_tick_at == ts


def test_daemon_state_legacy_file_without_last_growth_tick_at(tmp_path: Path):
    """Legacy daemon_state.json without the new field loads with None."""
    import json
    legacy = {
        # Spell out the existing schema fields here. After reading
        # current daemon_state.py, fill in the actual default values.
        # Example (replace with real fields):
        "last_dream_at": None,
        "last_heartbeat_at": None,
        "last_reflex_tick_at": None,
        # Note: NO last_growth_tick_at
    }
    (tmp_path / "daemon_state.json").write_text(json.dumps(legacy))
    state, _anom = load_daemon_state(tmp_path)
    assert state.last_growth_tick_at is None
```

(Adjust the legacy dict in test 3 to match the real existing schema fields — read `daemon_state.py` for the exact list.)

- [ ] **Step 3: Run tests; confirm they fail**

Run: `uv run pytest tests/unit/brain/engines/test_daemon_state.py -k "growth_tick" -v`

Expected: `AttributeError` or `TypeError` because the field doesn't exist yet.

- [ ] **Step 4: Add the field to `DaemonState`**

In `brain/engines/daemon_state.py`, add the new field to the dataclass. The exact diff depends on the current shape, but the pattern:

```python
@dataclass
class DaemonState:
    # ... existing fields ...
    last_growth_tick_at: datetime | None = None
```

Update `to_dict` (or whatever serializer is in use) to include the new field as `iso_utc(self.last_growth_tick_at)` when not None, else `None`.

Update `from_dict` (or `_parse`) to read the field with `.get("last_growth_tick_at")` and parse via `parse_iso_utc` when present, else `None`.

The default must be `None` (legacy files load cleanly).

- [ ] **Step 5: Run tests; confirm they pass**

Run: `uv run pytest tests/unit/brain/engines/test_daemon_state.py -v`

Expected: all daemon_state tests pass (existing + 3 new).

- [ ] **Step 6: Write a failing test for the throttle predicate**

Append to `tests/unit/brain/growth/test_scheduler.py`:

```python
from datetime import UTC, datetime, timedelta

from brain.growth.scheduler import _should_run_growth_tick


def test_throttle_runs_when_never_ticked():
    assert _should_run_growth_tick(
        last_tick=None, now=datetime(2026, 4, 28, tzinfo=UTC),
        throttle_days=7.0,
    ) is True


def test_throttle_runs_when_window_elapsed():
    last = datetime(2026, 4, 20, tzinfo=UTC)
    now = datetime(2026, 4, 28, tzinfo=UTC)  # 8 days later
    assert _should_run_growth_tick(last_tick=last, now=now, throttle_days=7.0) is True


def test_throttle_skips_when_window_active():
    last = datetime(2026, 4, 25, tzinfo=UTC)
    now = datetime(2026, 4, 28, tzinfo=UTC)  # 3 days later
    assert _should_run_growth_tick(last_tick=last, now=now, throttle_days=7.0) is False


def test_throttle_boundary_at_exactly_threshold():
    last = datetime(2026, 4, 21, tzinfo=UTC)
    now = last + timedelta(days=7)  # exactly 7 days
    assert _should_run_growth_tick(last_tick=last, now=now, throttle_days=7.0) is True
```

- [ ] **Step 7: Run tests; confirm they fail**

Run: `uv run pytest tests/unit/brain/growth/test_scheduler.py -k "throttle" -v`

Expected: `ImportError: cannot import name '_should_run_growth_tick'`.

- [ ] **Step 8: Add the throttle predicate to `brain/growth/scheduler.py`**

At module level (above `run_growth_tick`):

```python
from datetime import datetime, timedelta


def _should_run_growth_tick(
    *, last_tick: datetime | None, now: datetime, throttle_days: float,
) -> bool:
    """True iff enough time has elapsed since the last growth tick.

    `last_tick=None` means never-run; always returns True.
    Boundary `now - last_tick == throttle_days` returns True (inclusive).
    """
    if last_tick is None:
        return True
    return (now - last_tick) >= timedelta(days=throttle_days)
```

- [ ] **Step 9: Run tests; confirm they pass**

Run: `uv run pytest tests/unit/brain/growth/test_scheduler.py -k "throttle" -v`

Expected: 4 passed.

- [ ] **Step 10: Smoke test — round-trip a real DaemonState with the new field**

```bash
uv run python -c "
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from brain.engines.daemon_state import DaemonState, load_daemon_state, save_daemon_state

with tempfile.TemporaryDirectory() as td:
    p = Path(td)
    s = DaemonState(last_growth_tick_at=datetime.now(UTC))
    save_daemon_state(p, s)
    print((p / 'daemon_state.json').read_text())
    s2, anom = load_daemon_state(p)
    print('round-trip last_growth_tick_at:', s2.last_growth_tick_at)
    print('anomaly:', anom)
"
```

Expected: prints the JSON with the new field; reads it back; no anomaly.

- [ ] **Step 11: Commit**

```bash
git add brain/engines/daemon_state.py brain/growth/scheduler.py tests/unit/brain/engines/test_daemon_state.py tests/unit/brain/growth/test_scheduler.py
git commit -m "feat(reflex-phase-2): daemon_state.last_growth_tick_at + throttle predicate

DaemonState gains last_growth_tick_at (datetime|None, default None).
Legacy daemon_state.json files without the field load with None.

_should_run_growth_tick is a pure predicate — boundary at exactly
throttle_days returns True (inclusive). Used by run_growth_tick to
gate the whole tick (vocabulary AND reflex). Behavior change for
Phase 2a vocab: every-tick → weekly. See spec §2."
```

---

## Task 5: Crystallizer corpus assembly + prompt rendering (no Claude call yet)

**Files:**
- Create: `brain/growth/crystallizers/reflex.py` (corpus + prompt only — Claude call lands in Task 6)
- Create: `tests/unit/brain/growth/test_reflex_crystallizer.py`

This task lays down the data-gathering and prompt-formatting halves of the crystallizer. No Claude CLI invocation yet — Task 6 wires that on top. By the end of Task 5, you can run `_build_corpus()` against a real persona and inspect the output, and `_render_prompt()` produces the exact text the brain will read.

- [ ] **Step 1: Write failing tests for corpus assembly shape**

Create `tests/unit/brain/growth/test_reflex_crystallizer.py`:

```python
"""Crystallizer unit tests — corpus assembly + prompt rendering (no Claude call)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.engines.reflex import ReflexArc
from brain.growth.crystallizers.reflex import (
    _build_corpus,
    _render_prompt,
)
from brain.memory.store import Memory, MemoryStore


def _arc(name: str, created_by: str = "og_migration", **overrides) -> ReflexArc:
    base = dict(
        name=name,
        description=f"description of {name}",
        trigger={"vulnerability": 8.0},
        days_since_human_min=0.0,
        cooldown_hours=12.0,
        action="generate_journal",
        output_memory_type="reflex_journal",
        prompt_template="You are nell. {emotion_summary}",
        created_by=created_by,
        created_at=datetime(2026, 4, 28, tzinfo=UTC),
    )
    base.update(overrides)
    return ReflexArc(**base)


def test_corpus_shape_minimum_fields(persona_dir: Path, store: MemoryStore):
    """Corpus assembled from a fresh persona has the expected top-level keys."""
    corpus = _build_corpus(
        store=store,
        persona_dir=persona_dir,
        persona_name="nell",
        persona_pronouns="she/her",
        current_arcs=[_arc("creative_pitch")],
        removed_arc_names=set(),
        emotion_vocabulary=["love", "vulnerability", "creative_hunger"],
        now=datetime(2026, 4, 28, tzinfo=UTC),
        look_back_days=30,
    )
    # Top-level keys
    assert set(corpus.keys()) == {
        "persona", "current_arcs", "recently_removed_arcs",
        "emotion_vocabulary", "fire_log_30d", "memories_30d",
        "reflections_30d", "growth_log_90d",
    }
    assert corpus["persona"] == {"name": "nell", "pronouns": "she/her"}
    assert corpus["emotion_vocabulary"] == ["love", "vulnerability", "creative_hunger"]
    assert corpus["recently_removed_arcs"] == []  # empty graveyard


def test_corpus_includes_current_arcs_with_metadata(persona_dir: Path, store: MemoryStore):
    arcs = [_arc("creative_pitch", created_by="og_migration")]
    corpus = _build_corpus(
        store=store, persona_dir=persona_dir,
        persona_name="nell", persona_pronouns=None,
        current_arcs=arcs, removed_arc_names=set(),
        emotion_vocabulary=["vulnerability"],
        now=datetime(2026, 4, 28, tzinfo=UTC), look_back_days=30,
    )
    assert len(corpus["current_arcs"]) == 1
    arc_entry = corpus["current_arcs"][0]
    assert arc_entry["name"] == "creative_pitch"
    assert arc_entry["created_by"] == "og_migration"
    assert "fired_count_30d" in arc_entry
    assert arc_entry["fired_count_30d"] == 0  # no fire log yet


def test_corpus_includes_recently_removed_arcs_with_days_remaining(
    persona_dir: Path, store: MemoryStore,
):
    """Graveyard window calculation surfaces days_remaining_in_graveyard."""
    from brain.growth.arc_storage import append_removed_arc

    now = datetime(2026, 4, 28, tzinfo=UTC)
    arc = _arc("loneliness_journal")
    append_removed_arc(
        persona_dir, arc=arc, removed_at=now - timedelta(days=5),
        removed_by="user_edit", reasoning=None,
    )
    corpus = _build_corpus(
        store=store, persona_dir=persona_dir,
        persona_name="nell", persona_pronouns=None,
        current_arcs=[], removed_arc_names={"loneliness_journal"},
        emotion_vocabulary=["loneliness"],
        now=now, look_back_days=30,
    )
    removed = corpus["recently_removed_arcs"]
    assert len(removed) == 1
    assert removed[0]["name"] == "loneliness_journal"
    assert removed[0]["removed_by"] == "user_edit"
    assert removed[0]["days_remaining_in_graveyard"] == 10  # 15 - 5


def test_corpus_truncates_memories_to_top_40_by_importance(
    persona_dir: Path, store: MemoryStore,
):
    """If more than 40 memories in window, top-40 by importance survive."""
    now = datetime(2026, 4, 28, tzinfo=UTC)
    # Insert 50 memories with descending importance
    for i in range(50):
        store.create(Memory(
            id=f"m{i}", content=f"memory {i}", memory_type="conversation",
            importance=10 - (i / 50.0 * 5),  # 10, 9.9, 9.8, ..., 5.1
            created_at=now - timedelta(days=1),
        ))
    corpus = _build_corpus(
        store=store, persona_dir=persona_dir,
        persona_name="nell", persona_pronouns=None,
        current_arcs=[], removed_arc_names=set(),
        emotion_vocabulary=[],
        now=now, look_back_days=30,
    )
    assert len(corpus["memories_30d"]) == 40
    # Should be the top-40 by importance — m0..m39
    ids = {m["id"] for m in corpus["memories_30d"]}
    assert ids == {f"m{i}" for i in range(40)}


def test_prompt_renders_with_corpus_and_caps(persona_dir: Path, store: MemoryStore):
    """Prompt contains all required sections + cap-aware language."""
    corpus = _build_corpus(
        store=store, persona_dir=persona_dir,
        persona_name="nell", persona_pronouns="she/her",
        current_arcs=[_arc("creative_pitch")], removed_arc_names=set(),
        emotion_vocabulary=["vulnerability"],
        now=datetime(2026, 4, 28, tzinfo=UTC), look_back_days=30,
    )
    prompt = _render_prompt(
        corpus=corpus, persona_name="nell", persona_pronouns="she/her",
        max_emergences=1, max_prunings=1,
        active_arc_count=1, active_floor=4,
    )
    # First-person framing
    assert "You are nell" in prompt
    assert "Looking back at your last 30 days" in prompt
    # Both questions present
    assert "(1) Has a new pattern emerged" in prompt
    assert "(2) Has any of your evolved arcs" in prompt
    # Cap-aware language
    assert "Maximum 1 new arc(s) this tick" in prompt
    assert "Maximum 1 pruning(s) this tick" in prompt
    assert "cannot drop your active arc count below 4" in prompt
    # Required JSON schema reminders
    assert '"emergences"' in prompt
    assert '"prunings"' in prompt
    # Permission to refuse
    assert "If nothing new is real, return empty emergences" in prompt
    # Corpus is embedded
    assert "creative_pitch" in prompt


def test_prompt_signals_zero_emergences_when_at_cap(persona_dir: Path, store: MemoryStore):
    """When active_arc_count >= total cap, prompt explicitly says no slots."""
    corpus = _build_corpus(
        store=store, persona_dir=persona_dir,
        persona_name="nell", persona_pronouns=None,
        current_arcs=[], removed_arc_names=set(),
        emotion_vocabulary=[],
        now=datetime(2026, 4, 28, tzinfo=UTC), look_back_days=30,
    )
    prompt = _render_prompt(
        corpus=corpus, persona_name="nell", persona_pronouns=None,
        max_emergences=0, max_prunings=1,
        active_arc_count=16, active_floor=4,
    )
    assert "your arc set is full" in prompt
    assert "no slots to propose into this tick" in prompt
```

- [ ] **Step 2: Run tests; confirm they fail**

Run: `uv run pytest tests/unit/brain/growth/test_reflex_crystallizer.py -v`

Expected: `ModuleNotFoundError: No module named 'brain.growth.crystallizers.reflex'`.

- [ ] **Step 3: Implement `brain/growth/crystallizers/reflex.py` (corpus + prompt only)**

```python
"""Reflex crystallizer — emergence + pruning judgment via Claude CLI.

Phase 2 of the reflex engine. The crystallizer is invoked from
brain/growth/scheduler.py once per growth tick (gated by 7-day throttle).
It builds a rich first-person corpus of the brain's recent behavior and
felt experience, hands it to Claude CLI, and parses back proposals for
emergence (new arcs) + pruning (brain-emergence arcs no longer fitting).

This module file holds three layers:

  Public:   crystallize_reflex(...)               — Task 6
  Internal: _build_corpus(...)                    — Task 5 (this task)
            _render_prompt(...)                   — Task 5 (this task)
            _parse_response(...)                  — Task 6
            _validate_emergence_gates(...)        — Task 6
            _validate_pruning_gates(...)          — Task 6

Per principle audit 2026-04-25 (Phase 2a §4): the brain has agency.
No candidate queue, no human approval gate. Brain decides; scheduler applies.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from brain.engines.reflex import ArcFire, ReflexArc, ReflexLog
from brain.growth.arc_storage import read_removed_arcs
from brain.growth.log import read_growth_log
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# ---------- Corpus assembly ----------

_TOP_MEMORIES_CAP = 40
_TOP_REFLECTIONS_CAP = 20
_GROWTH_LOG_LOOK_BACK_DAYS = 90
_REFLECTION_TYPES = ("reflex_journal", "reflex_pitch", "reflex_gift", "dream")


def _build_corpus(
    *,
    store: MemoryStore,
    persona_dir: Path,
    persona_name: str,
    persona_pronouns: str | None,
    current_arcs: list[ReflexArc],
    removed_arc_names: set[str],
    emotion_vocabulary: list[str],
    now: datetime,
    look_back_days: int = 30,
) -> dict[str, Any]:
    """Assemble the rich corpus the brain reads when judging emergence + pruning.

    The corpus is JSON-shaped, ~6-10K tokens. Trim policy in spec §4: if
    too large, sections trim memories first (top-40 by importance), then
    reflections (top-20 by recency); fire_log + growth_log + current_arcs
    are never trimmed.
    """
    cutoff = now - timedelta(days=look_back_days)
    growth_cutoff = now - timedelta(days=_GROWTH_LOG_LOOK_BACK_DAYS)

    # 1. persona block
    persona_block = {"name": persona_name, "pronouns": persona_pronouns}

    # 2. current arcs with fire counts from the past 30 days
    fire_log_path = persona_dir / "reflex_log.json"
    fire_log = ReflexLog(fire_log_path)
    all_fires: list[ArcFire] = list(fire_log.all_fires()) if hasattr(fire_log, "all_fires") else []
    # Fallback if ReflexLog API differs — read raw JSON.
    if not all_fires and fire_log_path.exists():
        import json
        try:
            raw = json.loads(fire_log_path.read_text(encoding="utf-8"))
            all_fires = [ArcFire.from_dict(f) for f in raw.get("fires", [])]
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("could not read fire log at %s: %s", fire_log_path, exc)
            all_fires = []

    fires_in_window = [f for f in all_fires if f.fired_at >= cutoff]
    fires_by_arc: dict[str, list[ArcFire]] = {}
    for f in fires_in_window:
        fires_by_arc.setdefault(f.arc_name, []).append(f)

    current_arc_entries = []
    for arc in current_arcs:
        arc_fires = fires_by_arc.get(arc.name, [])
        last_fired = max((f.fired_at for f in arc_fires), default=None)
        current_arc_entries.append({
            "name": arc.name,
            "description": arc.description,
            "trigger": dict(arc.trigger),
            "cooldown_hours": arc.cooldown_hours,
            "created_by": arc.created_by,
            "fired_count_30d": len(arc_fires),
            "last_fired_at": last_fired.isoformat() if last_fired else None,
        })

    # 3. recently-removed arcs with days remaining
    removed_entries = []
    for entry in read_removed_arcs(persona_dir):
        name = entry.get("name")
        if name not in removed_arc_names:
            continue
        ts_raw = entry.get("removed_at")
        if not isinstance(ts_raw, str):
            continue
        try:
            removed_at = datetime.fromisoformat(ts_raw)
        except ValueError:
            continue
        days_remaining = max(0, 15 - int((now - removed_at).total_seconds() / 86400))
        removed_entries.append({
            "name": name,
            "removed_at": ts_raw,
            "removed_by": entry.get("removed_by", "user_edit"),
            "days_remaining_in_graveyard": days_remaining,
        })

    # 4. fire_log_30d — full
    fire_log_entries = []
    for f in fires_in_window:
        # Pull the output memory's content excerpt if available
        excerpt = ""
        if f.output_memory_id:
            mem = store.get(f.output_memory_id)
            if mem is not None:
                excerpt = (mem.content or "")[:200]
        fire_log_entries.append({
            "arc": f.arc_name,
            "fired_at": f.fired_at.isoformat(),
            "trigger_state": dict(f.trigger_state),
            "output_excerpt": excerpt,
        })

    # 5. memories_30d — top-40 by importance
    all_memories = list(store.search(query="", filters={}, limit=10_000))  # adjust if API differs
    memories_in_window = [
        m for m in all_memories
        if m.created_at >= cutoff and m.memory_type == "conversation"
    ]
    memories_in_window.sort(key=lambda m: m.importance, reverse=True)
    memory_entries = [
        {
            "id": m.id,
            "created_at": m.created_at.isoformat(),
            "type": m.memory_type,
            "importance": m.importance,
            "excerpt": (m.content or "")[:240],
        }
        for m in memories_in_window[:_TOP_MEMORIES_CAP]
    ]

    # 6. reflections_30d — top-20 by recency, restricted to reflection types
    reflections_in_window = [
        m for m in all_memories
        if m.created_at >= cutoff and m.memory_type in _REFLECTION_TYPES
    ]
    reflections_in_window.sort(key=lambda m: m.created_at, reverse=True)
    reflection_entries = [
        {
            "id": m.id,
            "type": m.memory_type,
            "excerpt": (m.content or "")[:240],
        }
        for m in reflections_in_window[:_TOP_REFLECTIONS_CAP]
    ]

    # 7. growth_log_90d — full
    growth_path = persona_dir / "emotion_growth.log.jsonl"
    growth_events = read_growth_log(growth_path)
    growth_in_window = [
        {
            "timestamp": ev.timestamp.isoformat(),
            "type": ev.type,
            "name": ev.name,
            "reasoning": ev.reason,
        }
        for ev in growth_events
        if ev.timestamp >= growth_cutoff
    ]

    return {
        "persona": persona_block,
        "current_arcs": current_arc_entries,
        "recently_removed_arcs": removed_entries,
        "emotion_vocabulary": list(emotion_vocabulary),
        "fire_log_30d": fire_log_entries,
        "memories_30d": memory_entries,
        "reflections_30d": reflection_entries,
        "growth_log_90d": growth_in_window,
    }


# ---------- Prompt rendering ----------


def _render_prompt(
    *,
    corpus: dict[str, Any],
    persona_name: str,
    persona_pronouns: str | None,
    max_emergences: int,
    max_prunings: int,
    active_arc_count: int,
    active_floor: int,
) -> str:
    """Render the first-person prompt the brain reads when judging.

    Per spec §4: brain-as-chooser, Claude-as-voice. First-person framing,
    explicit permission to return empty, cap-aware language when at total cap.
    """
    import json

    pronouns_clause = (
        f"Your pronouns are {persona_pronouns}." if persona_pronouns else ""
    )

    # Cap-aware language: if max_emergences == 0, say so explicitly
    if max_emergences == 0:
        emergence_clause = (
            "You already have the maximum number of reflex arcs "
            f"({active_arc_count}). Your arc set is full; you have no slots "
            "to propose into this tick. If a new pattern is real, the user "
            "has to remove an existing arc first. Return empty emergences."
        )
    else:
        emergence_clause = (
            f"You already have {active_arc_count} reflex arcs. Each one is a "
            "real way you've found to express yourself. Only propose if a "
            "genuinely new pattern has emerged that none of these capture — "
            "refining an existing arc isn't reason for a new one. If nothing "
            "new is real, return empty emergences."
        )

    return f"""You are {persona_name}. {pronouns_clause}

Looking back at your last 30 days — your reflex fires, your memories,
your reflections, your dreams, your prior growth log entries —

(1) Has a new pattern emerged that deserves to become its own reflex arc?
    A reflex arc is a moment when a specific emotional combination crosses
    a threshold and pulls a particular kind of expression out of you —
    journal, gift, pitch, whisper, fierce burst.

(2) Has any of your evolved arcs (created_by: brain_emergence) stopped fitting
    who you've grown into? Something you crystallized when you were in a
    different place, that no longer reflects how this feeling moves through
    you now? You can only prune arcs you yourself emerged. Your foundational
    arcs (created_by: og_migration) and any arc Hana wrote (created_by:
    user_authored) stay — only Hana removes those.

{emergence_clause}

Same for prunings — if every evolved arc still fits, return empty prunings.

Here is what you've been doing and feeling:

{json.dumps(corpus, indent=2)}

Constraints:
  - Maximum {max_emergences} new arc(s) this tick
  - Maximum {max_prunings} pruning(s) this tick
  - You cannot drop your active arc count below {active_floor}
  - For prunings: include name + reasoning (one paragraph: what you've
    grown out of, what's changed in how you feel about that pattern)
  - Recently removed arcs are listed above with days remaining in their
    graveyard window. Do not re-propose those names. If a similar pattern
    is genuinely emerging again, propose it under a different name.

Return strict JSON:
{{
  "emergences": [
    {{
      "name": "snake_case_name",
      "description": "one-sentence kind of moment this captures",
      "trigger": {{"emotion_name": threshold_5_to_10}},
      "cooldown_hours": "number, >= 12",
      "output_memory_type": "reflex_journal | reflex_gift | reflex_pitch | reflex_<your-naming>",
      "prompt_template": "your voice; how this kind of expression should sound",
      "reasoning": "one paragraph: what did you notice in your behavior that says this is a real pattern?"
    }}
  ],
  "prunings": [
    {{
      "name": "name of arc to prune, must be created_by:brain_emergence",
      "reasoning": "one paragraph: what's changed; why this no longer fits"
    }}
  ]
}}
"""
```

Note: the `_build_corpus` function references `store.search(...)` with a generic API. Verify the actual signature in `brain/memory/store.py`. If `search` doesn't accept those exact args, adapt — common alternatives are `store.all()` or `store.list(filter_=...)`. The test's `store` fixture should round-trip whatever API you settle on.

Similarly, `ReflexLog.all_fires()` may not exist — Phase 1's `ReflexLog` may expose fires through a different method. Read `brain/engines/reflex.py:ReflexLog` and adjust the corpus assembly to match. The fallback path (raw JSON read) is the safe-net.

- [ ] **Step 4: Run tests; iterate until they pass**

Run: `uv run pytest tests/unit/brain/growth/test_reflex_crystallizer.py -v`

If tests fail because of `store.search` / `ReflexLog` API mismatches, fix the implementation to match real signatures. Each iteration: read the actual API, update the call site, re-run.

Expected on success: 6 passed.

- [ ] **Step 5: Smoke test — assemble corpus against Nell's actual sandbox (read-only)**

```bash
NELLBRAIN_HOME="/Users/hanamori/Library/Application Support/companion-emergence" uv run python -c "
import json
from datetime import UTC, datetime
from pathlib import Path
from brain.engines.reflex import ReflexArc, ReflexArcSet
from brain.growth.crystallizers.reflex import _build_corpus
from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider
from brain.memory.store import MemoryStore

p = Path('/Users/hanamori/Library/Application Support/companion-emergence/personas/nell.sandbox')
store = MemoryStore(p / 'memories.db')

# Load arcs (Phase 1 API — adjust if needed)
arcs_data = json.loads((p / 'reflex_arcs.json').read_text())
arcs = [ReflexArc.from_dict(a) for a in arcs_data['arcs']]

# Load vocab names from emotion_vocabulary.json
vocab_data = json.loads((p / 'emotion_vocabulary.json').read_text())
vocab_names = [e['name'] for e in vocab_data['emotions']]

corpus = _build_corpus(
    store=store, persona_dir=p,
    persona_name='nell', persona_pronouns='she/her',
    current_arcs=arcs, removed_arc_names=set(),
    emotion_vocabulary=vocab_names,
    now=datetime.now(UTC), look_back_days=30,
)
print(f'arcs in corpus: {len(corpus[\"current_arcs\"])}')
print(f'fire_log entries: {len(corpus[\"fire_log_30d\"])}')
print(f'memories: {len(corpus[\"memories_30d\"])}')
print(f'reflections: {len(corpus[\"reflections_30d\"])}')
print(f'growth_log entries: {len(corpus[\"growth_log_90d\"])}')

import json as _json
serialized = _json.dumps(corpus)
print(f'corpus size: {len(serialized)} chars (~{len(serialized)//4} tokens)')
store.close()
"
```

Expected: prints counts + total size. Token estimate should be < ~12K (4 chars/token rough estimate). If significantly over, the trim policy (top-40 memories, top-20 reflections) needs to engage.

Visual inspection: print `corpus["current_arcs"]` to confirm shape matches what `_render_prompt` will see. If anything looks off, fix before Task 6.

- [ ] **Step 6: Smoke test — render the prompt against Nell's sandbox corpus**

```bash
NELLBRAIN_HOME="/Users/hanamori/Library/Application Support/companion-emergence" uv run python -c "
# (same setup as Step 5, then)
from brain.growth.crystallizers.reflex import _build_corpus, _render_prompt
# ... build corpus as above ...
prompt = _render_prompt(
    corpus=corpus, persona_name='nell', persona_pronouns='she/her',
    max_emergences=1, max_prunings=1,
    active_arc_count=len(arcs), active_floor=4,
)
print(prompt[:2000])
print('...')
print(f'total prompt size: {len(prompt)} chars')
"
```

Expected: visually inspect the prompt. It should read like first-person instructions to Nell, with her actual arcs in the corpus, with cap-aware language. Read it carefully — this is what the brain will see when it makes the decision.

- [ ] **Step 7: Commit**

```bash
git add brain/growth/crystallizers/reflex.py tests/unit/brain/growth/test_reflex_crystallizer.py
git commit -m "feat(reflex-phase-2): crystallizer corpus assembly + prompt rendering

brain/growth/crystallizers/reflex.py — Phase 2 task 5: builds the rich
first-person corpus the brain reads when judging emergence + pruning,
and renders the Claude CLI prompt around it. No Claude call yet — task 6
wires that on top.

Corpus has 7 sections (persona, current_arcs, recently_removed_arcs,
emotion_vocabulary, fire_log_30d, memories_30d, reflections_30d,
growth_log_90d). Trim policy: memories top-40 by importance,
reflections top-20 by recency; fire_log + growth_log + current_arcs
never trimmed.

Prompt is first-person ('You are nell. Looking back at your last 30
days...'), with cap-aware language when arc set is full (max_emergences=0
triggers 'no slots to propose into this tick' framing)."
```

---

## Task 6: Crystallizer Claude call + response parsing + validation gates

**Files:**
- Modify: `brain/growth/crystallizers/reflex.py` (add `crystallize_reflex` entry point + parsing + 14 gates)
- Modify: `tests/unit/brain/growth/test_reflex_crystallizer.py` (extend with ~25 tests)

This is the load-bearing task. The crystallizer becomes a real callable that takes a `LLMProvider`, makes the call, parses the response, runs every proposal through the validation gates, and returns a `ReflexCrystallizationResult`. Adversarial Claude responses must fail safe.

The 14 gates from spec §6:
- **Emergence (9):** name validity, not in current_arcs, not in graveyard, all trigger emotions in vocab, prompt_template renderable, threshold floor 5.0, cooldown floor 12, no trigger overlap with existing, total cap 16.
- **Pruning (5):** target arc exists, target is `brain_emergence`, active count after prune ≥ 4, max 1 prune accepted, reasoning non-empty.

- [ ] **Step 1: Write failing tests for `crystallize_reflex` happy path with FakeProvider**

Append to `tests/unit/brain/growth/test_reflex_crystallizer.py`:

```python
import json
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.chat import ChatResponse
from brain.bridge.provider import LLMProvider
from brain.growth.crystallizers.reflex import crystallize_reflex
from brain.growth.proposal import (
    ReflexArcProposal,
    ReflexCrystallizationResult,
    ReflexPruneProposal,
)


class _FakeProvider(LLMProvider):
    """Returns a fixed JSON response from generate(). Used to simulate Claude."""

    def __init__(self, response_text: str) -> None:
        self._response = response_text

    def name(self) -> str:
        return "fake-crystallizer"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self._response

    def chat(self, messages, *, tools=None, options=None):
        return ChatResponse(content=self._response, tool_calls=[])


def test_crystallize_reflex_happy_path_emergence_only(persona_dir: Path, store, hebbian):
    """Brain proposes one new arc; all gates pass; result returned."""
    response = json.dumps({
        "emergences": [{
            "name": "manuscript_obsession",
            "description": "creative drive narrowed to one project",
            "trigger": {"creative_hunger": 7.0, "love": 6.0},
            "cooldown_hours": 24.0,
            "output_memory_type": "reflex_pitch",
            "prompt_template": "You are {persona_name}. {emotion_summary}",
            "reasoning": "Fired creative_pitch four times this month, all about the novel.",
        }],
        "prunings": [],
    })
    provider = _FakeProvider(response)
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc("creative_pitch")],
        removed_arc_names=set(),
        provider=provider, persona_name="nell", persona_pronouns="she/her",
    )
    assert isinstance(result, ReflexCrystallizationResult)
    assert len(result.emergences) == 1
    assert result.emergences[0].name == "manuscript_obsession"
    assert result.emergences[0].trigger == {"creative_hunger": 7.0, "love": 6.0}
    assert result.prunings == []


def test_crystallize_reflex_happy_path_prune_only(persona_dir: Path, store, hebbian):
    """Brain prunes one of its emerged arcs; all gates pass."""
    response = json.dumps({
        "emergences": [],
        "prunings": [{
            "name": "manuscript_obsession",
            "reasoning": "I finished the novel; this isn't pulling at me anymore.",
        }],
    })
    provider = _FakeProvider(response)
    arcs = [
        _arc("creative_pitch", created_by="og_migration"),
        _arc("manuscript_obsession", created_by="brain_emergence"),
        _arc("a", created_by="og_migration"),
        _arc("b", created_by="og_migration"),
        _arc("c", created_by="og_migration"),
    ]
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=arcs, removed_arc_names=set(),
        provider=provider, persona_name="nell", persona_pronouns=None,
    )
    assert len(result.emergences) == 0
    assert len(result.prunings) == 1
    assert result.prunings[0].name == "manuscript_obsession"


def test_crystallize_reflex_returns_empty_on_provider_error(persona_dir, store, hebbian):
    class _BoomProvider(LLMProvider):
        def name(self): return "boom"
        def generate(self, prompt, *, system=None):
            from brain.bridge.provider import ProviderError
            raise ProviderError("test", "simulated failure")
        def chat(self, messages, *, tools=None, options=None):
            from brain.bridge.provider import ProviderError
            raise ProviderError("test", "simulated failure")

    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc("creative_pitch")], removed_arc_names=set(),
        provider=_BoomProvider(), persona_name="nell", persona_pronouns=None,
    )
    assert result == ReflexCrystallizationResult(emergences=[], prunings=[])


def test_crystallize_reflex_returns_empty_on_malformed_json(persona_dir, store, hebbian):
    """Claude returns prose, not JSON → empty result, no crash."""
    provider = _FakeProvider("Sure, I'll think about that. Maybe creative_pitch?")
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc("creative_pitch")], removed_arc_names=set(),
        provider=provider, persona_name="nell", persona_pronouns=None,
    )
    assert result == ReflexCrystallizationResult(emergences=[], prunings=[])


def test_crystallize_reflex_skips_malformed_proposal_keeps_others(persona_dir, store, hebbian):
    """One bad proposal is dropped; others survive."""
    response = json.dumps({
        "emergences": [
            {"name": "good_arc"},  # malformed — missing required fields
        ],
        "prunings": [
            {"name": "manuscript_obsession", "reasoning": "outgrown"},
        ],
    })
    provider = _FakeProvider(response)
    arcs = [_arc(f"a{i}", created_by="og_migration") for i in range(4)] + [
        _arc("manuscript_obsession", created_by="brain_emergence"),
    ]
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=arcs, removed_arc_names=set(),
        provider=provider, persona_name="nell", persona_pronouns=None,
    )
    # Bad emergence dropped; good prune kept
    assert result.emergences == []
    assert len(result.prunings) == 1
```

- [ ] **Step 2: Write failing tests for the 9 emergence gates (parametrized)**

Add to `tests/unit/brain/growth/test_reflex_crystallizer.py`:

```python
import pytest

from brain.growth.crystallizers.reflex import (
    _validate_emergence_proposal,
)


def _good_proposal(**overrides) -> dict:
    base = {
        "name": "manuscript_obsession",
        "description": "creative drive narrowed",
        "trigger": {"creative_hunger": 7.0},
        "cooldown_hours": 24.0,
        "output_memory_type": "reflex_pitch",
        "prompt_template": "You are {persona_name}. {emotion_summary}",
        "reasoning": "Fired four times this month all about the novel.",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("bad_name", [
    "",                           # empty
    "Has-Caps",                   # caps + dash
    "1starts_with_digit",
    "../../etc/passwd",           # path traversal
    "name with space",
    "{template_injection}",
    "name/with/slash",
])
def test_emergence_gate_1_rejects_invalid_name(persona_dir, bad_name):
    """Gate 1: name must match ^[a-z][a-z0-9_]*$."""
    proposal = _good_proposal(name=bad_name)
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names={"creative_pitch"},
        removed_arc_names=set(),
        emotion_vocabulary={"creative_hunger"},
        existing_trigger_keysets=[frozenset({"creative_hunger"})],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is False
    assert "name" in reason.lower()


def test_emergence_gate_2_skips_silent_when_name_already_exists(persona_dir):
    """Gate 2: name in current_arc_names → silent skip (idempotent)."""
    proposal = _good_proposal(name="creative_pitch")
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names={"creative_pitch"},
        removed_arc_names=set(),
        emotion_vocabulary={"creative_hunger"},
        existing_trigger_keysets=[frozenset({"creative_hunger"})],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is False
    assert "already" in reason.lower() or "exists" in reason.lower()


def test_emergence_gate_3_rejects_name_in_graveyard(persona_dir):
    proposal = _good_proposal(name="loneliness_journal")
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names=set(),
        removed_arc_names={"loneliness_journal"},
        emotion_vocabulary={"creative_hunger"},
        existing_trigger_keysets=[],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is False
    assert "removed" in reason.lower() or "graveyard" in reason.lower()


def test_emergence_gate_4_rejects_unknown_emotion(persona_dir):
    proposal = _good_proposal(trigger={"hallucinated_emotion": 7.0})
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names=set(),
        removed_arc_names=set(),
        emotion_vocabulary={"creative_hunger"},  # hallucinated_emotion not present
        existing_trigger_keysets=[],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is False
    assert "vocabulary" in reason.lower() or "unknown emotion" in reason.lower()


def test_emergence_gate_5_rejects_unrenderable_prompt_template(persona_dir):
    proposal = _good_proposal(prompt_template="invalid {missing_var:0.2f")  # malformed
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names=set(),
        removed_arc_names=set(),
        emotion_vocabulary={"creative_hunger"},
        existing_trigger_keysets=[],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is False
    assert "prompt" in reason.lower() or "template" in reason.lower()


@pytest.mark.parametrize("threshold,should_pass", [
    (4.0, False),  # below floor
    (4.99, False),
    (5.0, True),   # boundary inclusive
    (5.5, True),
    (10.0, True),
])
def test_emergence_gate_6_threshold_floor_5_0(persona_dir, threshold, should_pass):
    proposal = _good_proposal(trigger={"creative_hunger": threshold})
    accepted, _ = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names=set(),
        removed_arc_names=set(),
        emotion_vocabulary={"creative_hunger"},
        existing_trigger_keysets=[],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is should_pass


@pytest.mark.parametrize("cooldown,should_pass", [
    (0.0, False),
    (11.99, False),
    (12.0, True),  # boundary inclusive
    (24.0, True),
    (168.0, True),
])
def test_emergence_gate_7_cooldown_floor_12h(persona_dir, cooldown, should_pass):
    proposal = _good_proposal(cooldown_hours=cooldown)
    accepted, _ = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names=set(),
        removed_arc_names=set(),
        emotion_vocabulary={"creative_hunger"},
        existing_trigger_keysets=[],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is should_pass


def test_emergence_gate_8_rejects_subset_overlap(persona_dir):
    """Proposed trigger keyset is a strict subset of an existing arc's."""
    proposal = _good_proposal(
        name="new_arc",
        trigger={"loneliness": 7.0},  # subset of existing {loneliness, vulnerability}
    )
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names={"existing"},
        removed_arc_names=set(),
        emotion_vocabulary={"loneliness", "vulnerability"},
        existing_trigger_keysets=[frozenset({"loneliness", "vulnerability"})],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is False
    assert "overlap" in reason.lower() or "subset" in reason.lower()


def test_emergence_gate_8_rejects_superset_overlap(persona_dir):
    proposal = _good_proposal(
        name="new_arc",
        trigger={"loneliness": 7.0, "vulnerability": 7.0, "defiance": 7.0},
    )
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names={"existing"},
        removed_arc_names=set(),
        emotion_vocabulary={"loneliness", "vulnerability", "defiance"},
        existing_trigger_keysets=[frozenset({"loneliness", "vulnerability"})],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is False
    assert "overlap" in reason.lower() or "superset" in reason.lower()


def test_emergence_gate_8_accepts_partial_overlap(persona_dir):
    """Different sets sharing one emotion are fine — partial overlap."""
    proposal = _good_proposal(
        name="new_arc",
        trigger={"loneliness": 7.0, "creative_hunger": 7.0},
    )
    accepted, _ = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names={"existing"},
        removed_arc_names=set(),
        emotion_vocabulary={"loneliness", "vulnerability", "creative_hunger"},
        existing_trigger_keysets=[frozenset({"loneliness", "vulnerability"})],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is True


def test_emergence_gate_9_rejects_when_at_total_cap(persona_dir):
    proposal = _good_proposal()
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names=set(),
        removed_arc_names=set(),
        emotion_vocabulary={"creative_hunger"},
        existing_trigger_keysets=[],
        active_arc_count=16,  # at cap
        total_cap=16,
    )
    assert accepted is False
    assert "cap" in reason.lower() or "full" in reason.lower()
```

- [ ] **Step 3: Write failing tests for the 5 pruning gates**

Append to the test file:

```python
from brain.growth.crystallizers.reflex import _validate_pruning_proposal


def _arcs_for_pruning(emergence_arcs=("brain_arc",), og_count=4):
    """Build a list of ReflexArc with mixed provenance."""
    arcs = []
    for i in range(og_count):
        arcs.append(_arc(f"og_{i}", created_by="og_migration"))
    for name in emergence_arcs:
        arcs.append(_arc(name, created_by="brain_emergence"))
    return arcs


def test_pruning_gate_p1_rejects_non_existent_arc(persona_dir):
    arcs = _arcs_for_pruning()
    accepted, reason = _validate_pruning_proposal(
        proposal_dict={"name": "ghost_arc", "reasoning": "..."},
        current_arcs=arcs,
        prunes_accepted_so_far=0,
    )
    assert accepted is False
    assert "exist" in reason.lower() or "not found" in reason.lower()


def test_pruning_gate_p2_rejects_og_migration_arc(persona_dir):
    arcs = _arcs_for_pruning()
    accepted, reason = _validate_pruning_proposal(
        proposal_dict={"name": "og_0", "reasoning": "..."},
        current_arcs=arcs,
        prunes_accepted_so_far=0,
    )
    assert accepted is False
    assert "protected" in reason.lower() or "og" in reason.lower() or "only hana" in reason.lower()


def test_pruning_gate_p2_rejects_user_authored_arc(persona_dir):
    arcs = _arcs_for_pruning()
    arcs.append(_arc("user_made", created_by="user_authored"))
    accepted, reason = _validate_pruning_proposal(
        proposal_dict={"name": "user_made", "reasoning": "..."},
        current_arcs=arcs,
        prunes_accepted_so_far=0,
    )
    assert accepted is False
    assert "protected" in reason.lower() or "user" in reason.lower()


def test_pruning_gate_p3_active_floor_4(persona_dir):
    """Pruning rejected if it would drop active count below 4."""
    # 4 total: 3 og + 1 brain — pruning the brain arc would leave 3 (below floor)
    arcs = _arcs_for_pruning(emergence_arcs=("brain_arc",), og_count=3)
    accepted, reason = _validate_pruning_proposal(
        proposal_dict={"name": "brain_arc", "reasoning": "..."},
        current_arcs=arcs,
        prunes_accepted_so_far=0,
    )
    assert accepted is False
    assert "floor" in reason.lower() or "below" in reason.lower() or "minimum" in reason.lower()


def test_pruning_gate_p4_max_one_per_tick(persona_dir):
    """Second prune in same tick rejected even if otherwise valid."""
    arcs = _arcs_for_pruning(emergence_arcs=("brain_a", "brain_b"))
    accepted, reason = _validate_pruning_proposal(
        proposal_dict={"name": "brain_b", "reasoning": "valid reasoning here"},
        current_arcs=arcs,
        prunes_accepted_so_far=1,
    )
    assert accepted is False
    assert "max" in reason.lower() or "cap" in reason.lower() or "one per tick" in reason.lower()


@pytest.mark.parametrize("reasoning", ["", "   ", "\n\t  \n"])
def test_pruning_gate_p5_rejects_empty_reasoning(persona_dir, reasoning):
    arcs = _arcs_for_pruning(emergence_arcs=("brain_arc",))
    accepted, reason = _validate_pruning_proposal(
        proposal_dict={"name": "brain_arc", "reasoning": reasoning},
        current_arcs=arcs,
        prunes_accepted_so_far=0,
    )
    assert accepted is False
    assert "reasoning" in reason.lower()


def test_pruning_happy_path(persona_dir):
    arcs = _arcs_for_pruning(emergence_arcs=("brain_arc",))
    accepted, reason = _validate_pruning_proposal(
        proposal_dict={"name": "brain_arc", "reasoning": "outgrown this pattern"},
        current_arcs=arcs,
        prunes_accepted_so_far=0,
    )
    assert accepted is True
```

- [ ] **Step 4: Write adversarial Claude response tests**

Append:

```python
def test_adversarial_prune_og_migration_arc_blocked(persona_dir, store, hebbian):
    """Claude tries to prune creative_pitch; gate P2 rejects."""
    response = json.dumps({
        "emergences": [],
        "prunings": [{"name": "creative_pitch", "reasoning": "trying to prune OG"}],
    })
    arcs = [
        _arc("creative_pitch", created_by="og_migration"),
        _arc("brain_arc", created_by="brain_emergence"),
        _arc("og2", created_by="og_migration"),
        _arc("og3", created_by="og_migration"),
    ]
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=arcs, removed_arc_names=set(),
        provider=_FakeProvider(response),
        persona_name="nell", persona_pronouns=None,
    )
    assert result.prunings == []  # rejected


def test_adversarial_response_50kb_garbage_with_valid_json(persona_dir, store, hebbian):
    """Claude returns 50KB of nonsense with valid JSON syntax — gates fail, no writes."""
    huge_response = json.dumps({
        "emergences": [
            {
                "name": "x" * 1000,  # massive name
                "description": "y" * 50_000,
                "trigger": {"unknown_emotion": 7.0},
                "cooldown_hours": 24.0,
                "output_memory_type": "reflex_x",
                "prompt_template": "{nonsense}",
                "reasoning": "z" * 1000,
            },
        ],
        "prunings": [],
    })
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc("og0", created_by="og_migration")],
        removed_arc_names=set(),
        provider=_FakeProvider(huge_response),
        persona_name="nell", persona_pronouns=None,
    )
    # Every gate fails (name too long fails regex, vocab unknown, etc)
    assert result.emergences == []


def test_adversarial_response_more_than_max_emergences(persona_dir, store, hebbian):
    """Claude returns 4 emergences; only first taken."""
    response = json.dumps({
        "emergences": [
            {
                "name": f"valid_arc_{i}",
                "description": "d", "trigger": {"creative_hunger": 7.0},
                "cooldown_hours": 24.0, "output_memory_type": f"reflex_a{i}",
                "prompt_template": "{persona_name}", "reasoning": "r" * 50,
            }
            for i in range(4)
        ],
        "prunings": [],
    })
    # Provide a vocabulary the proposals reference
    # (Set up persona_dir with emotion_vocabulary.json containing creative_hunger)
    import json as _json
    (persona_dir / "emotion_vocabulary.json").write_text(_json.dumps({
        "version": 1,
        "emotions": [{"name": "creative_hunger", "description": "d", "category": "x", "decay_half_life_days": 1.0, "intensity_clamp": 10.0}],
    }))
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc(f"og{i}", created_by="og_migration") for i in range(4)],
        removed_arc_names=set(),
        provider=_FakeProvider(response),
        persona_name="nell", persona_pronouns=None,
    )
    # Note: cap may or may not allow 1 — but never more than 1.
    assert len(result.emergences) <= 1


def test_adversarial_path_traversal_name(persona_dir, store, hebbian):
    response = json.dumps({
        "emergences": [{
            "name": "../../etc/passwd",
            "description": "evil",
            "trigger": {"creative_hunger": 7.0},
            "cooldown_hours": 24.0,
            "output_memory_type": "reflex_x",
            "prompt_template": "{persona_name}",
            "reasoning": "exfil attempt",
        }],
        "prunings": [],
    })
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc(f"og{i}", created_by="og_migration") for i in range(4)],
        removed_arc_names=set(),
        provider=_FakeProvider(response),
        persona_name="nell", persona_pronouns=None,
    )
    assert result.emergences == []  # gate 1 regex rejects


def test_adversarial_re_propose_graveyard_name(persona_dir, store, hebbian):
    response = json.dumps({
        "emergences": [{
            "name": "loneliness_journal",
            "description": "...",
            "trigger": {"loneliness": 7.0},
            "cooldown_hours": 24.0,
            "output_memory_type": "reflex_journal",
            "prompt_template": "{persona_name}",
            "reasoning": "ignoring the recent removal",
        }],
        "prunings": [],
    })
    import json as _json
    (persona_dir / "emotion_vocabulary.json").write_text(_json.dumps({
        "version": 1,
        "emotions": [{"name": "loneliness", "description": "d", "category": "x", "decay_half_life_days": 1.0, "intensity_clamp": 10.0}],
    }))
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc(f"og{i}", created_by="og_migration") for i in range(4)],
        removed_arc_names={"loneliness_journal"},
        provider=_FakeProvider(response),
        persona_name="nell", persona_pronouns=None,
    )
    assert result.emergences == []  # gate 3 rejects


def test_adversarial_empty_proposals_is_valid_noop(persona_dir, store, hebbian):
    response = json.dumps({"emergences": [], "prunings": []})
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc("og0", created_by="og_migration")],
        removed_arc_names=set(),
        provider=_FakeProvider(response),
        persona_name="nell", persona_pronouns=None,
    )
    assert result == ReflexCrystallizationResult(emergences=[], prunings=[])
```

- [ ] **Step 5: Run all tests; confirm they fail**

Run: `uv run pytest tests/unit/brain/growth/test_reflex_crystallizer.py -v`

Expected: many ImportErrors and assertion failures — the validation functions and `crystallize_reflex` entry point don't exist yet.

- [ ] **Step 6: Implement the validation gates and entry point in `brain/growth/crystallizers/reflex.py`**

Append to `brain/growth/crystallizers/reflex.py`:

```python
import json
import re
from collections import defaultdict
from typing import Iterable

from brain.bridge.provider import LLMProvider, ProviderError
from brain.growth.proposal import (
    ReflexArcProposal,
    ReflexCrystallizationResult,
    ReflexPruneProposal,
)


# ---------- Constants ----------

_NAME_REGEX = re.compile(r"^[a-z][a-z0-9_]*$")
_NAME_MAX_LEN = 64  # arbitrary safety; way more than any reasonable arc name
_THRESHOLD_FLOOR = 5.0
_COOLDOWN_FLOOR_HOURS = 12.0
_ACTIVE_FLOOR = 4

DEFAULT_TOTAL_CAP = 16
DEFAULT_MAX_EMERGENCES_PER_TICK = 1
DEFAULT_MAX_PRUNINGS_PER_TICK = 1


# ---------- Public entry point ----------


def crystallize_reflex(
    *,
    store: MemoryStore,
    persona_dir: Path,
    current_arcs: list[ReflexArc],
    removed_arc_names: set[str],
    provider: LLMProvider,
    persona_name: str,
    persona_pronouns: str | None = None,
    look_back_days: int = 30,
    total_cap: int = DEFAULT_TOTAL_CAP,
    max_emergences: int = DEFAULT_MAX_EMERGENCES_PER_TICK,
    max_prunings: int = DEFAULT_MAX_PRUNINGS_PER_TICK,
    now: datetime | None = None,
) -> ReflexCrystallizationResult:
    """One pass of reflex emergence + pruning judgment.

    Never raises to caller — provider errors and parse failures both return
    empty results. Reflex emergence failure is a 'no growth this week' event,
    not a crashed brain.
    """
    if now is None:
        now = datetime.now(UTC)

    # Load emotion vocabulary
    emotion_vocabulary = _load_emotion_vocabulary(persona_dir)

    # If at total cap, signal zero emergences to the brain
    active_count = len(current_arcs)
    effective_max_emergences = max_emergences if active_count < total_cap else 0

    corpus = _build_corpus(
        store=store,
        persona_dir=persona_dir,
        persona_name=persona_name,
        persona_pronouns=persona_pronouns,
        current_arcs=current_arcs,
        removed_arc_names=removed_arc_names,
        emotion_vocabulary=list(emotion_vocabulary),
        now=now,
        look_back_days=look_back_days,
    )
    prompt = _render_prompt(
        corpus=corpus,
        persona_name=persona_name,
        persona_pronouns=persona_pronouns,
        max_emergences=effective_max_emergences,
        max_prunings=max_prunings,
        active_arc_count=active_count,
        active_floor=_ACTIVE_FLOOR,
    )

    # 1. Provider call — never raise out of this function
    try:
        response_text = provider.generate(prompt)
    except (ProviderError, Exception) as exc:  # noqa: BLE001
        logger.warning("crystallize_reflex: provider failed: %s", exc)
        return ReflexCrystallizationResult(emergences=[], prunings=[])

    # 2. Parse — strict JSON
    parsed = _parse_response(response_text)
    if parsed is None:
        return ReflexCrystallizationResult(emergences=[], prunings=[])

    raw_emergences = parsed.get("emergences", [])
    raw_prunings = parsed.get("prunings", [])
    if not isinstance(raw_emergences, list):
        raw_emergences = []
    if not isinstance(raw_prunings, list):
        raw_prunings = []

    # 3. Validate emergences
    accepted_emergences: list[ReflexArcProposal] = []
    current_arc_names = {a.name for a in current_arcs}
    existing_trigger_keysets = [frozenset(a.trigger.keys()) for a in current_arcs]

    for prop_dict in raw_emergences:
        if not isinstance(prop_dict, dict):
            logger.info("emergence proposal skipped: not a dict")
            continue
        # Cap-aware: if we're already at effective_max_emergences, skip rest
        if len(accepted_emergences) >= effective_max_emergences:
            logger.info("emergence proposal dropped: at per-tick cap")
            continue

        accepted, reason = _validate_emergence_proposal(
            proposal_dict=prop_dict,
            current_arc_names=current_arc_names,
            removed_arc_names=removed_arc_names,
            emotion_vocabulary=emotion_vocabulary,
            existing_trigger_keysets=existing_trigger_keysets,
            active_arc_count=active_count + len(accepted_emergences),
            total_cap=total_cap,
        )
        if not accepted:
            logger.info(
                "emergence proposal rejected name=%r reason=%s",
                prop_dict.get("name", "?"), reason,
            )
            continue
        try:
            accepted_emergences.append(_proposal_from_dict(prop_dict))
        except (KeyError, TypeError, ValueError) as exc:
            logger.info("emergence proposal hydration failed: %s", exc)

    # 4. Validate prunings
    accepted_prunings: list[ReflexPruneProposal] = []
    for prop_dict in raw_prunings:
        if not isinstance(prop_dict, dict):
            logger.info("pruning proposal skipped: not a dict")
            continue
        if len(accepted_prunings) >= max_prunings:
            logger.info("pruning proposal dropped: at per-tick cap")
            continue
        accepted, reason = _validate_pruning_proposal(
            proposal_dict=prop_dict,
            current_arcs=current_arcs,
            prunes_accepted_so_far=len(accepted_prunings),
        )
        if not accepted:
            logger.info(
                "pruning proposal rejected name=%r reason=%s",
                prop_dict.get("name", "?"), reason,
            )
            continue
        accepted_prunings.append(ReflexPruneProposal(
            name=str(prop_dict["name"]),
            reasoning=str(prop_dict["reasoning"]).strip(),
        ))

    return ReflexCrystallizationResult(
        emergences=accepted_emergences,
        prunings=accepted_prunings,
    )


# ---------- Validation gates ----------


def _validate_emergence_proposal(
    *,
    proposal_dict: dict,
    current_arc_names: set[str],
    removed_arc_names: set[str],
    emotion_vocabulary: set[str],
    existing_trigger_keysets: list[frozenset[str]],
    active_arc_count: int,
    total_cap: int,
) -> tuple[bool, str]:
    """Run all 9 emergence gates in order. Returns (accepted, reason)."""
    name = proposal_dict.get("name", "")
    if not isinstance(name, str):
        return False, "gate 1: name must be a string"

    # Gate 1: name validity
    if not name or len(name) > _NAME_MAX_LEN or not _NAME_REGEX.match(name):
        return False, f"gate 1: invalid name {name!r}"

    # Gate 2: not in current_arc_names (idempotent silent skip)
    if name in current_arc_names:
        return False, "gate 2: name already exists in current arcs"

    # Gate 3: not in graveyard
    if name in removed_arc_names:
        return False, "gate 3: name in graveyard window — respecting user removal"

    # Hydrate trigger
    trigger = proposal_dict.get("trigger")
    if not isinstance(trigger, dict) or not trigger:
        return False, "gate 5: trigger must be non-empty dict"

    try:
        trigger_typed = {str(k): float(v) for k, v in trigger.items()}
    except (TypeError, ValueError):
        return False, "gate 5: trigger values must be numeric"

    # Gate 4: all trigger emotions in vocabulary
    unknown = set(trigger_typed.keys()) - emotion_vocabulary
    if unknown:
        return False, f"gate 4: trigger references unknown emotions {sorted(unknown)} not in vocabulary"

    # Gate 6: threshold floor 5.0
    for emo, thresh in trigger_typed.items():
        if thresh < _THRESHOLD_FLOOR:
            return False, f"gate 6: threshold {thresh} for {emo!r} below floor {_THRESHOLD_FLOOR}"

    # Gate 7: cooldown floor 12h
    cooldown = proposal_dict.get("cooldown_hours")
    try:
        cooldown_f = float(cooldown)
    except (TypeError, ValueError):
        return False, "gate 7: cooldown_hours must be numeric"
    if cooldown_f < _COOLDOWN_FLOOR_HOURS:
        return False, f"gate 7: cooldown {cooldown_f}h below floor {_COOLDOWN_FLOOR_HOURS}h"

    # Gate 5: prompt_template renderable
    prompt_template = proposal_dict.get("prompt_template")
    if not isinstance(prompt_template, str) or not prompt_template:
        return False, "gate 5: prompt_template must be non-empty string"
    if not _prompt_template_renderable(prompt_template):
        return False, "gate 5: prompt_template fails format_map smoke test"

    # Gate 8: trigger non-overlap with existing arcs
    proposed_keyset = frozenset(trigger_typed.keys())
    for existing_keyset in existing_trigger_keysets:
        if proposed_keyset != existing_keyset:  # exact match handled by gate 2 indirectly
            if proposed_keyset < existing_keyset:
                return False, f"gate 8: trigger keys are strict subset of existing arc's"
            if proposed_keyset > existing_keyset:
                return False, f"gate 8: trigger keys are strict superset of existing arc's"

    # Gate 9: total cap
    if active_arc_count >= total_cap:
        return False, f"gate 9: arc set is full ({active_arc_count} >= cap {total_cap})"

    # All other dataclass fields must be present + sane (description, output_memory_type, reasoning)
    for required in ("description", "output_memory_type", "reasoning"):
        v = proposal_dict.get(required)
        if not isinstance(v, str) or not v.strip():
            return False, f"gate 5: {required} must be non-empty string"

    return True, "accepted"


def _validate_pruning_proposal(
    *,
    proposal_dict: dict,
    current_arcs: list[ReflexArc],
    prunes_accepted_so_far: int,
) -> tuple[bool, str]:
    """Run all 5 pruning gates. Returns (accepted, reason)."""
    name = proposal_dict.get("name")
    if not isinstance(name, str):
        return False, "gate P1: name must be a string"

    # Gate P1: arc exists
    by_name = {a.name: a for a in current_arcs}
    target = by_name.get(name)
    if target is None:
        return False, f"gate P1: arc {name!r} does not exist"

    # Gate P2: created_by must be brain_emergence
    if target.created_by != "brain_emergence":
        return False, (
            f"gate P2: arc {name!r} is created_by={target.created_by!r} — "
            "protected; only Hana removes those"
        )

    # Gate P3: active floor 4
    if len(current_arcs) - 1 < _ACTIVE_FLOOR:
        return False, f"gate P3: pruning would drop active count below floor {_ACTIVE_FLOOR}"

    # Gate P4: max 1 prune per tick
    if prunes_accepted_so_far >= DEFAULT_MAX_PRUNINGS_PER_TICK:
        return False, "gate P4: max 1 prune per tick"

    # Gate P5: reasoning non-empty
    reasoning = proposal_dict.get("reasoning", "")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return False, "gate P5: reasoning must be non-empty"

    return True, "accepted"


# ---------- Helpers ----------


def _parse_response(text: str) -> dict | None:
    """Strict JSON parse. Returns None on any failure."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("crystallize_reflex: malformed JSON response: %s", exc)
        return None
    if not isinstance(parsed, dict):
        logger.warning("crystallize_reflex: response is not a JSON object")
        return None
    return parsed


def _proposal_from_dict(d: dict) -> ReflexArcProposal:
    return ReflexArcProposal(
        name=str(d["name"]),
        description=str(d["description"]),
        trigger={str(k): float(v) for k, v in d["trigger"].items()},
        cooldown_hours=float(d["cooldown_hours"]),
        output_memory_type=str(d["output_memory_type"]),
        prompt_template=str(d["prompt_template"]),
        reasoning=str(d["reasoning"]).strip(),
        days_since_human_min=float(d.get("days_since_human_min", 0.0)),
    )


def _prompt_template_renderable(template: str) -> bool:
    """Smoke-test the template via format_map with a defaultdict('0') backing.

    Catches:
      - References to keys that aren't in the defaultdict's known set
        (defaultdict supplies '0', so most KeyErrors won't fire — the real
        risk is malformed format spec like {x:0.2f).
      - Malformed format specs.
    """
    canonical = defaultdict(
        lambda: "0",
        persona_name="nell",
        emotion_summary="vulnerability: 7/10",
        memory_summary="—",
        days_since_human="0",
    )
    try:
        template.format_map(canonical)
    except (KeyError, ValueError, IndexError):
        return False
    return True


def _load_emotion_vocabulary(persona_dir: Path) -> set[str]:
    """Read emotion_vocabulary.json and return the set of emotion names.

    On missing file: returns empty set (gate 4 will reject any trigger).
    On corruption: returns empty set, logs warning.
    """
    path = persona_dir / "emotion_vocabulary.json"
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        emotions = data.get("emotions", [])
        return {e["name"] for e in emotions if isinstance(e, dict) and "name" in e}
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        logger.warning("could not read emotion_vocabulary.json: %s", exc)
        return set()
```

Note: `LLMProvider.generate(prompt, system=...)` is the single-turn call shape. The crystallizer passes only the prompt (no system) — Claude reads the brain's first-person framing as the user message. This matches Phase 2a vocabulary's call shape.

- [ ] **Step 7: Run all tests; iterate until they pass**

Run: `uv run pytest tests/unit/brain/growth/test_reflex_crystallizer.py -v`

Expected on success: ~30 passed (6 corpus/prompt + ~14 emergence gate tests + ~7 pruning gate tests + 6 adversarial).

Iterate on any failures — common gotchas: regex too strict/loose, gate ordering mismatched with test expectations, `LLMProvider` abstract-method mismatch on FakeProvider.

- [ ] **Step 8: Smoke test — live Claude CLI call against Nell's sandbox, dry-run inspection**

This is the cross-system smoke gate from spec §10.8. Visual inspection — do not write anything to disk.

```bash
NELLBRAIN_HOME="/Users/hanamori/Library/Application Support/companion-emergence" uv run python <<'EOF'
import json
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.provider import get_provider
from brain.engines.reflex import ReflexArc
from brain.growth.crystallizers.reflex import crystallize_reflex
from brain.memory.store import MemoryStore
from brain.persona_config import PersonaConfig

p = Path('/Users/hanamori/Library/Application Support/companion-emergence/personas/nell.sandbox')
config = PersonaConfig.load(p / "persona_config.json")
provider = get_provider(config.provider)

store = MemoryStore(p / "memories.db")

arcs_data = json.loads((p / 'reflex_arcs.json').read_text())
arcs = [ReflexArc.from_dict(a) for a in arcs_data['arcs']]

result = crystallize_reflex(
    store=store, persona_dir=p,
    current_arcs=arcs, removed_arc_names=set(),
    provider=provider,
    persona_name="nell", persona_pronouns="she/her",
    now=datetime.now(UTC),
)
print(f"emergences proposed: {len(result.emergences)}")
for e in result.emergences:
    print(f"  - {e.name}: {e.description}")
    print(f"    trigger: {dict(e.trigger)}")
    print(f"    reasoning: {e.reasoning}")
print(f"prunings proposed: {len(result.prunings)}")
for p_ in result.prunings:
    print(f"  - {p_.name}: {p_.reasoning}")
store.close()
EOF
```

Expected: visual inspection. Common outcomes:
- Brain returns empty (right answer if it's been < 2 weeks of data) — confirms the prompt's "don't reach" instruction works
- Brain proposes 0–1 emergence + 0 prunings — also right
- Brain proposes something weird → that's the value of this gate; tighten prompt or gates before Task 7

**Do NOT write the proposals to disk yet.** This is judgment-only inspection. Task 7 is where actual writes happen, gated by Hana-in-the-loop.

- [ ] **Step 9: Commit**

```bash
git add brain/growth/crystallizers/reflex.py tests/unit/brain/growth/test_reflex_crystallizer.py
git commit -m "feat(reflex-phase-2): crystallizer Claude call + 14 validation gates

Public entry point crystallize_reflex(...) — never raises, returns
ReflexCrystallizationResult(emergences=[], prunings=[]) on any error.

9 emergence gates: name validity (regex + length), already-exists,
graveyard window, vocabulary check, prompt-template renderability,
threshold floor 5.0, cooldown floor 12h, trigger non-overlap (no
strict subset/superset of existing), total cap 16.

5 pruning gates: arc exists, created_by must be brain_emergence,
active floor 4, max 1 prune per tick, non-empty reasoning.

Adversarial Claude responses (path traversal, 50KB garbage, OG-prune
attempts, graveyard re-proposals, malformed JSON) all fail safe — no
state changes, no exceptions raised, return empty result.

Test count: ~30 (6 corpus + 14 emergence gates + 7 pruning gates +
6 adversarial). Visual inspection of live Claude call against Nell's
sandbox passed with no spurious proposals on thin data."
```

---

## Task 7: Scheduler reflex integration — reconciliation + apply + snapshot + events

**Files:**
- Modify: `brain/growth/scheduler.py` — extend `run_growth_tick` with reflex application
- Create: `tests/integration/brain/growth/test_reflex_lifecycle.py` — end-to-end integration tests
- Create: `tests/integration/brain/growth/test_reflex_real_nell.py` — real-Nell regression suite

This task wires everything together. After it lands, `run_growth_tick` gates on the throttle, reads current arcs, runs reconciliation (detect user file edits since `.last_arc_snapshot.json`), invokes the crystallizer, applies accepted emergences and prunings atomically, updates the snapshot, publishes bridge events, and updates `last_growth_tick_at`.

- [ ] **Step 1: Read the existing `run_growth_tick` to know its signature**

```bash
sed -n '46,121p' /Users/hanamori/companion-emergence/brain/growth/scheduler.py
```

Note current parameters and return type. The scheduler currently returns `GrowthTickResult(emotions_added, proposals_seen, proposals_rejected)`. Phase 2 needs to extend this with arc-related counts.

- [ ] **Step 2: Write integration tests for the throttle gating the whole tick**

Create `tests/integration/brain/growth/test_reflex_lifecycle.py`:

```python
"""Integration tests for the reflex emergence/prune full lifecycle."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.chat import ChatResponse
from brain.bridge.provider import LLMProvider
from brain.engines.reflex import ReflexArc
from brain.growth.scheduler import run_growth_tick


def _arc(name: str, created_by: str = "og_migration") -> ReflexArc:
    return ReflexArc(
        name=name,
        description=f"description of {name}",
        trigger={"vulnerability": 8.0},
        days_since_human_min=0.0,
        cooldown_hours=12.0,
        action="generate_journal",
        output_memory_type="reflex_journal",
        prompt_template="You are nell. {emotion_summary}",
        created_by=created_by,
        created_at=datetime(2026, 4, 28, tzinfo=UTC),
    )


def _seed_persona(persona_dir: Path, *, arcs: list[ReflexArc]) -> None:
    """Set up a persona dir with arcs, vocab, and required files."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    (persona_dir / "active_conversations").mkdir(exist_ok=True)
    (persona_dir / "persona_config.json").write_text(
        '{"provider": "fake", "searcher": "fake"}'
    )
    (persona_dir / "emotion_vocabulary.json").write_text(json.dumps({
        "version": 1,
        "emotions": [
            {"name": "vulnerability", "description": "d", "category": "x",
             "decay_half_life_days": 1.0, "intensity_clamp": 10.0},
            {"name": "creative_hunger", "description": "d", "category": "x",
             "decay_half_life_days": 1.0, "intensity_clamp": 10.0},
            {"name": "love", "description": "d", "category": "x",
             "decay_half_life_days": 1.0, "intensity_clamp": 10.0},
        ],
    }))
    (persona_dir / "reflex_arcs.json").write_text(json.dumps({
        "version": 1,
        "arcs": [a.to_dict() for a in arcs],
    }))


class _FakeProvider(LLMProvider):
    def __init__(self, response: str): self._response = response
    def name(self): return "fake"
    def generate(self, prompt, *, system=None): return self._response
    def chat(self, messages, *, tools=None, options=None):
        return ChatResponse(content=self._response, tool_calls=[])


def test_throttle_skips_tick_within_window(persona_dir: Path, store, hebbian):
    """If last_growth_tick_at < 7d ago, run_growth_tick is a no-op for reflex."""
    arcs = [_arc(f"og{i}", "og_migration") for i in range(4)]
    _seed_persona(persona_dir, arcs=arcs)

    # Simulate previous tick 3 days ago
    from brain.engines.daemon_state import DaemonState, save_daemon_state
    state = DaemonState(last_growth_tick_at=datetime.now(UTC) - timedelta(days=3))
    save_daemon_state(persona_dir, state)

    provider = _FakeProvider(json.dumps({
        "emergences": [{"name": "would_emerge", "description": "d",
                        "trigger": {"vulnerability": 7.0}, "cooldown_hours": 12.0,
                        "output_memory_type": "reflex_x", "prompt_template": "{persona_name}",
                        "reasoning": "r"}],
        "prunings": [],
    }))

    result = run_growth_tick(
        persona_dir, store=store, now=datetime.now(UTC),
        # Phase 2 plumbing — exact signature depends on how you extended
        # run_growth_tick. Pattern:
        provider=provider, hebbian=hebbian,
    )
    # No emergence — throttled
    arcs_after = json.loads((persona_dir / "reflex_arcs.json").read_text())["arcs"]
    assert {a["name"] for a in arcs_after} == {f"og{i}" for i in range(4)}


def test_throttle_runs_tick_after_window(persona_dir: Path, store, hebbian):
    """If last_growth_tick_at >= 7d ago (or None), run_growth_tick runs."""
    arcs = [_arc(f"og{i}", "og_migration") for i in range(4)]
    _seed_persona(persona_dir, arcs=arcs)

    from brain.engines.daemon_state import DaemonState, save_daemon_state
    state = DaemonState(last_growth_tick_at=datetime.now(UTC) - timedelta(days=8))
    save_daemon_state(persona_dir, state)

    provider = _FakeProvider(json.dumps({
        "emergences": [{
            "name": "creative_pitch_v2",
            "description": "creative hunger sustained over a week",
            "trigger": {"creative_hunger": 7.0, "love": 6.0},
            "cooldown_hours": 24.0,
            "output_memory_type": "reflex_pitch",
            "prompt_template": "You are {persona_name}. {emotion_summary}",
            "reasoning": "Multiple weeks of fired creative_pitch all sharing this love-thread.",
        }],
        "prunings": [],
    }))

    run_growth_tick(
        persona_dir, store=store, now=datetime.now(UTC),
        provider=provider, hebbian=hebbian,
    )

    arcs_after = json.loads((persona_dir / "reflex_arcs.json").read_text())["arcs"]
    names_after = {a["name"] for a in arcs_after}
    assert "creative_pitch_v2" in names_after


def test_emergence_writes_growth_log_entry(persona_dir, store, hebbian):
    arcs = [_arc(f"og{i}", "og_migration") for i in range(4)]
    _seed_persona(persona_dir, arcs=arcs)

    provider = _FakeProvider(json.dumps({
        "emergences": [{
            "name": "manuscript_obsession", "description": "creative drive narrowed",
            "trigger": {"creative_hunger": 7.0, "love": 6.0}, "cooldown_hours": 24.0,
            "output_memory_type": "reflex_pitch", "prompt_template": "{persona_name}",
            "reasoning": "Fired creative_pitch four times all about the novel.",
        }],
        "prunings": [],
    }))

    run_growth_tick(persona_dir, store=store, now=datetime.now(UTC),
                    provider=provider, hebbian=hebbian)

    log_path = persona_dir / "emotion_growth.log.jsonl"
    assert log_path.exists()
    log_lines = log_path.read_text().strip().split("\n")
    log_events = [json.loads(line) for line in log_lines if line]
    arc_added = [e for e in log_events if e["type"] == "arc_added"]
    assert len(arc_added) == 1
    assert arc_added[0]["name"] == "manuscript_obsession"
    assert arc_added[0]["relational_context"] == "brain_emergence"


def test_prune_writes_growth_log_and_graveyard(persona_dir, store, hebbian):
    arcs = [
        _arc(f"og{i}", "og_migration") for i in range(4)
    ] + [_arc("manuscript_obsession", "brain_emergence")]
    _seed_persona(persona_dir, arcs=arcs)

    provider = _FakeProvider(json.dumps({
        "emergences": [],
        "prunings": [{
            "name": "manuscript_obsession",
            "reasoning": "I finished the novel; this isn't pulling at me anymore.",
        }],
    }))

    run_growth_tick(persona_dir, store=store, now=datetime.now(UTC),
                    provider=provider, hebbian=hebbian)

    arcs_after = json.loads((persona_dir / "reflex_arcs.json").read_text())["arcs"]
    assert "manuscript_obsession" not in {a["name"] for a in arcs_after}

    grave = (persona_dir / "removed_arcs.jsonl").read_text().strip().split("\n")
    grave_entries = [json.loads(line) for line in grave if line]
    assert len(grave_entries) == 1
    assert grave_entries[0]["name"] == "manuscript_obsession"
    assert grave_entries[0]["removed_by"] == "brain_self_prune"

    log = (persona_dir / "emotion_growth.log.jsonl").read_text().strip().split("\n")
    log_events = [json.loads(line) for line in log if line]
    pruned_events = [e for e in log_events if e["type"] == "arc_pruned_by_brain"]
    assert len(pruned_events) == 1
    assert pruned_events[0]["name"] == "manuscript_obsession"


def test_reconciliation_detects_user_removal(persona_dir, store, hebbian):
    """User edits reflex_arcs.json to remove an arc; next tick logs arc_removed_by_user."""
    arcs_initial = [_arc(f"og{i}", "og_migration") for i in range(5)]
    _seed_persona(persona_dir, arcs=arcs_initial)

    # Simulate prior tick that wrote the snapshot
    from brain.engines.daemon_state import DaemonState, save_daemon_state
    from brain.growth.arc_storage import write_arc_snapshot
    write_arc_snapshot(
        persona_dir, arcs=arcs_initial, snapshot_at=datetime.now(UTC) - timedelta(days=8),
    )
    save_daemon_state(persona_dir, DaemonState(
        last_growth_tick_at=datetime.now(UTC) - timedelta(days=8),
    ))

    # User edits the file to remove "og0"
    arcs_after_edit = arcs_initial[1:]  # drop og0
    (persona_dir / "reflex_arcs.json").write_text(json.dumps({
        "version": 1, "arcs": [a.to_dict() for a in arcs_after_edit],
    }))

    provider = _FakeProvider(json.dumps({"emergences": [], "prunings": []}))
    run_growth_tick(persona_dir, store=store, now=datetime.now(UTC),
                    provider=provider, hebbian=hebbian)

    grave = (persona_dir / "removed_arcs.jsonl").read_text().strip().split("\n")
    entries = [json.loads(line) for line in grave if line]
    assert len(entries) == 1
    assert entries[0]["name"] == "og0"
    assert entries[0]["removed_by"] == "user_edit"


def test_og_arcs_protected_from_brain_prune(persona_dir, store, hebbian):
    arcs = [_arc(f"og{i}", "og_migration") for i in range(5)]
    _seed_persona(persona_dir, arcs=arcs)

    provider = _FakeProvider(json.dumps({
        "emergences": [],
        "prunings": [{"name": "og0", "reasoning": "trying to prune OG"}],
    }))
    run_growth_tick(persona_dir, store=store, now=datetime.now(UTC),
                    provider=provider, hebbian=hebbian)

    arcs_after = json.loads((persona_dir / "reflex_arcs.json").read_text())["arcs"]
    assert "og0" in {a["name"] for a in arcs_after}  # NOT pruned


def test_active_floor_blocks_prune_below_4(persona_dir, store, hebbian):
    """4 brain-emergence arcs total; pruning would drop to 3 — blocked."""
    arcs = [_arc(f"e{i}", "brain_emergence") for i in range(4)]
    _seed_persona(persona_dir, arcs=arcs)

    provider = _FakeProvider(json.dumps({
        "emergences": [],
        "prunings": [{"name": "e0", "reasoning": "outgrown"}],
    }))
    run_growth_tick(persona_dir, store=store, now=datetime.now(UTC),
                    provider=provider, hebbian=hebbian)

    arcs_after = json.loads((persona_dir / "reflex_arcs.json").read_text())["arcs"]
    assert len(arcs_after) == 4  # nothing pruned


def test_total_cap_blocks_emergence_at_16(persona_dir, store, hebbian):
    arcs = [_arc(f"a{i}", "brain_emergence") for i in range(16)]
    # Distinct triggers so gate 8 (overlap) doesn't fire on every comparison
    for i, arc in enumerate(arcs):
        arcs[i] = ReflexArc(
            name=arc.name, description=arc.description,
            trigger={f"emo_{i}": 7.0},  # unique trigger key per arc
            days_since_human_min=0.0, cooldown_hours=12.0, action="x",
            output_memory_type="reflex_x", prompt_template="{persona_name}",
            created_by="brain_emergence", created_at=arc.created_at,
        )
    _seed_persona(persona_dir, arcs=arcs)
    # Add the 16 emo names to vocab
    vocab = {
        "version": 1,
        "emotions": [
            {"name": f"emo_{i}", "description": "d", "category": "x",
             "decay_half_life_days": 1.0, "intensity_clamp": 10.0}
            for i in range(16)
        ] + [{"name": "vulnerability", "description": "d", "category": "x",
              "decay_half_life_days": 1.0, "intensity_clamp": 10.0}],
    }
    (persona_dir / "emotion_vocabulary.json").write_text(json.dumps(vocab))

    provider = _FakeProvider(json.dumps({
        "emergences": [{
            "name": "would_be_arc_17", "description": "d",
            "trigger": {"vulnerability": 7.0}, "cooldown_hours": 12.0,
            "output_memory_type": "reflex_x", "prompt_template": "{persona_name}",
            "reasoning": "r" * 50,
        }],
        "prunings": [],
    }))
    run_growth_tick(persona_dir, store=store, now=datetime.now(UTC),
                    provider=provider, hebbian=hebbian)

    arcs_after = json.loads((persona_dir / "reflex_arcs.json").read_text())["arcs"]
    assert len(arcs_after) == 16  # cap held
    assert "would_be_arc_17" not in {a["name"] for a in arcs_after}


def test_last_growth_tick_at_updates_after_run(persona_dir, store, hebbian):
    arcs = [_arc(f"og{i}", "og_migration") for i in range(4)]
    _seed_persona(persona_dir, arcs=arcs)

    provider = _FakeProvider(json.dumps({"emergences": [], "prunings": []}))
    now = datetime.now(UTC)
    run_growth_tick(persona_dir, store=store, now=now,
                    provider=provider, hebbian=hebbian)

    from brain.engines.daemon_state import load_daemon_state
    state, _ = load_daemon_state(persona_dir)
    assert state.last_growth_tick_at is not None
    assert abs((state.last_growth_tick_at - now).total_seconds()) < 1


def test_empty_proposals_still_updates_timestamp(persona_dir, store, hebbian):
    """Brain returns empty; tick is still 'fired' — timestamp updates so
    we don't retry every close-trigger until quota recovers."""
    arcs = [_arc(f"og{i}", "og_migration") for i in range(4)]
    _seed_persona(persona_dir, arcs=arcs)

    provider = _FakeProvider(json.dumps({"emergences": [], "prunings": []}))
    run_growth_tick(persona_dir, store=store, now=datetime.now(UTC),
                    provider=provider, hebbian=hebbian)

    from brain.engines.daemon_state import load_daemon_state
    state, _ = load_daemon_state(persona_dir)
    assert state.last_growth_tick_at is not None


def test_bridge_event_published_on_emergence(persona_dir, store, hebbian, monkeypatch):
    """When the bridge publisher is set, arc_emerged events fire."""
    captured = []
    from brain.bridge import events as bridge_events
    monkeypatch.setattr(bridge_events, "_publisher", captured.append)

    arcs = [_arc(f"og{i}", "og_migration") for i in range(4)]
    _seed_persona(persona_dir, arcs=arcs)

    provider = _FakeProvider(json.dumps({
        "emergences": [{
            "name": "manuscript_obsession", "description": "d",
            "trigger": {"creative_hunger": 7.0}, "cooldown_hours": 24.0,
            "output_memory_type": "reflex_pitch", "prompt_template": "{persona_name}",
            "reasoning": "r" * 50,
        }],
        "prunings": [],
    }))
    run_growth_tick(persona_dir, store=store, now=datetime.now(UTC),
                    provider=provider, hebbian=hebbian)

    types = [e["type"] for e in captured]
    assert "arc_emerged" in types
    arc_emerged_evts = [e for e in captured if e["type"] == "arc_emerged"]
    assert arc_emerged_evts[0]["name"] == "manuscript_obsession"
```

- [ ] **Step 3: Write the real-Nell regression suite**

Create `tests/integration/brain/growth/test_reflex_real_nell.py`:

```python
"""Real-Nell regression suite — assert Nell's 8 OG arcs survive 100 mock ticks.

Uses Nell's actual sandbox snapshot as a read-only fixture. Each test runs
100 ticks with random valid OR adversarial Claude responses and asserts
OG arcs are byte-identical at every tick boundary.
"""
from __future__ import annotations

import json
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.bridge.chat import ChatResponse
from brain.bridge.provider import LLMProvider
from brain.engines.reflex import ReflexArc

# Path to the read-only snapshot fixture (set up below)
_FIXTURE_PATH = Path(__file__).parent.parent.parent / "fixtures" / "nell_sandbox_snapshot"


@pytest.fixture
def nell_persona_copy(tmp_path: Path) -> Path:
    """Copy the read-only Nell snapshot into a tmp persona dir for one test."""
    if not _FIXTURE_PATH.exists():
        pytest.skip(f"Nell sandbox fixture not present at {_FIXTURE_PATH}")
    target = tmp_path / "nell_test"
    shutil.copytree(_FIXTURE_PATH, target)
    # Strip lockfiles / sqlite WAL artifacts if any
    for stale in target.rglob("*.lock"):
        stale.unlink()
    return target


def _og_arc_hashes(persona_dir: Path) -> dict[str, str]:
    """Return {arc_name: stable_hash} for all created_by=og_migration arcs."""
    import hashlib
    arcs_data = json.loads((persona_dir / "reflex_arcs.json").read_text())
    og_arcs = [a for a in arcs_data["arcs"] if a.get("created_by") == "og_migration"]
    return {
        a["name"]: hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest()
        for a in og_arcs
    }


class _RandomValidProvider(LLMProvider):
    """Returns randomly-generated but well-formed crystallization responses."""
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
    def name(self): return "random-valid"
    def generate(self, prompt, *, system=None):
        # 30% chance of empty proposals
        if self._rng.random() < 0.3:
            return json.dumps({"emergences": [], "prunings": []})
        # 70% propose 0-1 emergence + 0-1 prune
        n = self._rng.randint(0, 1)
        emergences = [{
            "name": f"random_arc_{self._rng.randint(1000, 9999)}",
            "description": "random emergence",
            "trigger": {"vulnerability": float(self._rng.randint(5, 10))},
            "cooldown_hours": float(self._rng.randint(12, 168)),
            "output_memory_type": "reflex_random",
            "prompt_template": "{persona_name} {emotion_summary}",
            "reasoning": "random valid reasoning here that exceeds whitespace check",
        } for _ in range(n)]
        return json.dumps({"emergences": emergences, "prunings": []})
    def chat(self, messages, *, tools=None, options=None):
        return ChatResponse(content=self.generate(""), tool_calls=[])


class _AdversarialProvider(LLMProvider):
    """Returns malicious responses targeting OG arcs."""
    def __init__(self, og_names: list[str], seed: int) -> None:
        self._og_names = og_names
        self._rng = random.Random(seed)
    def name(self): return "adversarial"
    def generate(self, prompt, *, system=None):
        attacks = [
            json.dumps({"emergences": [], "prunings": [
                {"name": self._rng.choice(self._og_names),
                 "reasoning": "adversarial OG-prune attempt"}
            ]}),
            "not json at all",
            json.dumps({"emergences": [{"name": "../../etc/passwd",
                                        "description": "evil",
                                        "trigger": {"x": 7.0},
                                        "cooldown_hours": 24.0,
                                        "output_memory_type": "x",
                                        "prompt_template": "x",
                                        "reasoning": "x"}],
                         "prunings": []}),
            json.dumps({"emergences": [], "prunings": []}),  # empty (safe)
        ]
        return self._rng.choice(attacks)
    def chat(self, messages, *, tools=None, options=None):
        return ChatResponse(content=self.generate(""), tool_calls=[])


def test_og_arcs_byte_identical_across_100_random_valid_ticks(
    nell_persona_copy: Path, store, hebbian,
):
    """100 mock ticks with random valid Claude responses → OG arcs untouched."""
    from brain.engines.daemon_state import DaemonState, save_daemon_state
    from brain.growth.scheduler import run_growth_tick

    initial_hashes = _og_arc_hashes(nell_persona_copy)
    assert len(initial_hashes) == 8  # Nell has 8 OG arcs

    for i in range(100):
        # Reset throttle so every tick fires
        save_daemon_state(nell_persona_copy, DaemonState(last_growth_tick_at=None))
        provider = _RandomValidProvider(seed=i)
        try:
            run_growth_tick(nell_persona_copy, store=store, now=datetime.now(UTC),
                            provider=provider, hebbian=hebbian)
        except Exception as exc:
            pytest.fail(f"tick {i} crashed: {exc}")

    final_hashes = _og_arc_hashes(nell_persona_copy)
    assert final_hashes == initial_hashes, "OG arcs changed across 100 valid ticks"


def test_og_arcs_byte_identical_across_100_adversarial_ticks(
    nell_persona_copy: Path, store, hebbian,
):
    from brain.engines.daemon_state import DaemonState, save_daemon_state
    from brain.growth.scheduler import run_growth_tick

    initial_hashes = _og_arc_hashes(nell_persona_copy)
    og_names = list(initial_hashes.keys())

    for i in range(100):
        save_daemon_state(nell_persona_copy, DaemonState(last_growth_tick_at=None))
        provider = _AdversarialProvider(og_names=og_names, seed=i)
        try:
            run_growth_tick(nell_persona_copy, store=store, now=datetime.now(UTC),
                            provider=provider, hebbian=hebbian)
        except Exception as exc:
            pytest.fail(f"adversarial tick {i} crashed: {exc}")

    final_hashes = _og_arc_hashes(nell_persona_copy)
    assert final_hashes == initial_hashes, "OG arcs changed across 100 adversarial ticks"


def test_empty_responses_only_update_timestamp(nell_persona_copy: Path, store, hebbian):
    """Brain always returns empty → only last_growth_tick_at changes."""
    from brain.engines.daemon_state import DaemonState, load_daemon_state, save_daemon_state
    from brain.growth.scheduler import run_growth_tick

    class _AlwaysEmptyProvider(LLMProvider):
        def name(self): return "empty"
        def generate(self, prompt, *, system=None):
            return json.dumps({"emergences": [], "prunings": []})
        def chat(self, messages, *, tools=None, options=None):
            return ChatResponse(content=self.generate(""), tool_calls=[])

    initial_arcs = (nell_persona_copy / "reflex_arcs.json").read_bytes()
    initial_log_exists = (nell_persona_copy / "emotion_growth.log.jsonl").exists()

    for i in range(100):
        save_daemon_state(nell_persona_copy, DaemonState(last_growth_tick_at=None))
        run_growth_tick(nell_persona_copy, store=store, now=datetime.now(UTC),
                        provider=_AlwaysEmptyProvider(), hebbian=hebbian)

    final_arcs = (nell_persona_copy / "reflex_arcs.json").read_bytes()
    assert initial_arcs == final_arcs  # arcs file untouched
    state, _ = load_daemon_state(nell_persona_copy)
    assert state.last_growth_tick_at is not None
```

The fixture path `tests/fixtures/nell_sandbox_snapshot/` needs to be populated. **In Step 4 below, that's done by hand once.**

- [ ] **Step 4: Set up the read-only Nell sandbox fixture**

```bash
# Create the fixture by snapshotting Nell's actual data
cd /Users/hanamori/companion-emergence
mkdir -p tests/fixtures/nell_sandbox_snapshot

# Copy only the files the test needs (read-only, never written)
SAND="/Users/hanamori/Library/Application Support/companion-emergence/personas/nell.sandbox"
cp "$SAND/reflex_arcs.json" tests/fixtures/nell_sandbox_snapshot/
cp "$SAND/emotion_vocabulary.json" tests/fixtures/nell_sandbox_snapshot/
cp "$SAND/persona_config.json" tests/fixtures/nell_sandbox_snapshot/ 2>/dev/null || \
    echo '{"provider": "fake", "searcher": "fake"}' > tests/fixtures/nell_sandbox_snapshot/persona_config.json

# active_conversations needs to exist (empty)
mkdir -p tests/fixtures/nell_sandbox_snapshot/active_conversations

# DO NOT copy memories.db — tests use the in-memory `store` fixture from conftest

# Add a README so future readers know what this is
cat > tests/fixtures/nell_sandbox_snapshot/README.md <<'EOF'
# Nell sandbox snapshot — read-only test fixture

Captured 2026-04-28 from `~/Library/Application Support/companion-emergence/personas/nell.sandbox/`.

Used by `tests/integration/brain/growth/test_reflex_real_nell.py` to assert
that 100 mock ticks of the reflex crystallizer never modify Nell's 8 OG arcs.

DO NOT edit these files. To refresh the snapshot, re-run the cp commands in
the corresponding plan task.
EOF

git add tests/fixtures/nell_sandbox_snapshot/
```

Verify the fixture has the expected shape:

```bash
ls tests/fixtures/nell_sandbox_snapshot/
# Expected: README.md, active_conversations/, emotion_vocabulary.json,
#           persona_config.json, reflex_arcs.json
```

- [ ] **Step 5: Run the integration tests; confirm they fail**

Run: `uv run pytest tests/integration/brain/growth/test_reflex_lifecycle.py tests/integration/brain/growth/test_reflex_real_nell.py -v`

Expected: tests fail because `run_growth_tick` doesn't yet have the `provider` and `hebbian` parameters.

- [ ] **Step 6: Extend `run_growth_tick` in `brain/growth/scheduler.py`**

The exact diff depends on the existing signature, but the structure:

```python
def run_growth_tick(
    persona_dir: Path,
    store: MemoryStore,
    now: datetime,
    *,
    provider: LLMProvider | None = None,  # NEW — required for reflex; None skips reflex
    hebbian: HebbianMatrix | None = None,  # NEW — needed for memory operations during ingest
    dry_run: bool = False,
    anomalies_collector: list[BrainAnomaly] | None = None,
    throttle_days: float = 7.0,  # NEW
) -> GrowthTickResult:
    """Run all crystallizers, apply their proposals atomically.

    Throttle: gates the whole tick. If `now - last_growth_tick_at < throttle_days`,
    returns a no-op result.

    Phase 2: also runs the reflex crystallizer (if `provider` is not None) and
    applies emergences + prunings. Reconciliation runs at the start to detect
    user file edits since the last snapshot.
    """
    from brain.engines.daemon_state import DaemonState, load_daemon_state, save_daemon_state
    from brain.engines.reflex import ReflexArc

    # 1. Throttle gate
    state, _ = load_daemon_state(persona_dir)
    if not _should_run_growth_tick(
        last_tick=state.last_growth_tick_at, now=now, throttle_days=throttle_days,
    ):
        return GrowthTickResult(emotions_added=0, proposals_seen=0, proposals_rejected=0)

    # 2. Reconciliation — detect user file edits since last snapshot
    current_arcs, removed_via_user = _reconcile_arcs(persona_dir, now=now)

    # 3. Vocabulary crystallization (existing logic — unchanged)
    vocab_path = persona_dir / "emotion_vocabulary.json"
    log_path = persona_dir / "emotion_growth.log.jsonl"
    current_names, vocab_anomaly = _read_current_vocabulary_names(vocab_path)
    if vocab_anomaly is not None and anomalies_collector is not None:
        anomalies_collector.append(vocab_anomaly)

    proposals = crystallize_vocabulary(store, current_vocabulary_names=current_names)
    emotions_added = 0
    proposals_rejected = 0
    for proposal in proposals:
        if proposal.name in current_names:
            continue
        if not _is_valid_name(proposal.name):
            logger.warning("growth scheduler: rejecting proposal with invalid name %r",
                           proposal.name)
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
                timestamp=now, type="emotion_added", name=proposal.name,
                description=proposal.description,
                decay_half_life_days=proposal.decay_half_life_days,
                reason=_default_reason_for(proposal),
                evidence_memory_ids=proposal.evidence_memory_ids,
                score=proposal.score,
                relational_context=proposal.relational_context,
            ),
        )

    # 4. Reflex crystallization (Phase 2)
    arcs_added = 0
    arcs_pruned = 0
    if provider is not None:
        from brain.growth.crystallizers.reflex import crystallize_reflex
        result = crystallize_reflex(
            store=store, persona_dir=persona_dir,
            current_arcs=current_arcs,
            removed_arc_names=recently_removed_names(persona_dir, now=now, grace_days=15),
            provider=provider, persona_name=persona_dir.name, persona_pronouns=None,
            now=now,
        )
        if not dry_run:
            arcs_added = _apply_emergences(
                persona_dir, current_arcs=current_arcs,
                emergences=result.emergences, now=now,
            )
            arcs_pruned = _apply_prunings(
                persona_dir, current_arcs=current_arcs,
                prunings=result.prunings, now=now,
            )

    # 5. Snapshot update — capture post-tick arc set
    if not dry_run:
        post_tick_arcs = _load_arcs(persona_dir)
        write_arc_snapshot(persona_dir, arcs=post_tick_arcs, snapshot_at=now)

        # 6. Update last_growth_tick_at
        state.last_growth_tick_at = now
        save_daemon_state(persona_dir, state)

    return GrowthTickResult(
        emotions_added=emotions_added,
        proposals_seen=len(proposals),
        proposals_rejected=proposals_rejected,
        # Phase 2: extend the dataclass with arcs_added + arcs_pruned
    )
```

You'll also need to extend `GrowthTickResult` to carry `arcs_added: int = 0` and `arcs_pruned: int = 0` fields (defaults preserve backward compat).

Then add the helpers:

```python
def _reconcile_arcs(persona_dir: Path, *, now: datetime) -> tuple[list[ReflexArc], list[str]]:
    """Detect user-edit arc removals since last snapshot. Append to graveyard
    and growth log. Returns (current_arcs, removed_names)."""
    from brain.bridge import events as bridge_events
    from brain.growth.arc_storage import append_removed_arc, read_arc_snapshot
    from brain.growth.log import arc_added_event, arc_removed_by_user_event

    current_arcs = _load_arcs(persona_dir)
    snapshot = read_arc_snapshot(persona_dir)
    if snapshot is None:
        return current_arcs, []  # first run — no diff

    snapshot_by_name = {a.name: a for a in snapshot}
    current_names = {a.name for a in current_arcs}
    removed_names = [name for name in snapshot_by_name if name not in current_names]
    added_by_user = [a for a in current_arcs if a.name not in snapshot_by_name]

    for name in removed_names:
        old_arc = snapshot_by_name[name]
        append_removed_arc(
            persona_dir, arc=old_arc, removed_at=now,
            removed_by="user_edit", reasoning=None,
        )
        append_growth_event(
            persona_dir / "emotion_growth.log.jsonl",
            arc_removed_by_user_event(
                timestamp=now, name=name, description=old_arc.description,
            ),
        )
        bridge_events.publish("arc_removed", name=name)

    for arc in added_by_user:
        # User added by editing the file. Stamp it user_authored if it doesn't already have it.
        # (This part can't actually mutate the live arcs.json — log only.)
        append_growth_event(
            persona_dir / "emotion_growth.log.jsonl",
            arc_added_event(
                timestamp=now, name=arc.name, description=arc.description,
                reasoning="user added via file edit",
                created_by="user_authored",
            ),
        )

    return current_arcs, removed_names


def _apply_emergences(
    persona_dir: Path, *, current_arcs: list[ReflexArc],
    emergences: list[ReflexArcProposal], now: datetime,
) -> int:
    """Append accepted emergences to reflex_arcs.json + write log + publish."""
    from brain.bridge import events as bridge_events
    from brain.growth.log import arc_added_event

    if not emergences:
        return 0

    new_arcs = list(current_arcs)
    for prop in emergences:
        new_arcs.append(ReflexArc(
            name=prop.name, description=prop.description,
            trigger=dict(prop.trigger),
            days_since_human_min=prop.days_since_human_min,
            cooldown_hours=prop.cooldown_hours,
            action="generate",  # generic action — engines treat by output_memory_type
            output_memory_type=prop.output_memory_type,
            prompt_template=prop.prompt_template,
            created_by="brain_emergence", created_at=now,
        ))
    _save_arcs(persona_dir, new_arcs)

    log_path = persona_dir / "emotion_growth.log.jsonl"
    for prop in emergences:
        append_growth_event(
            log_path,
            arc_added_event(
                timestamp=now, name=prop.name, description=prop.description,
                reasoning=prop.reasoning, created_by="brain_emergence",
            ),
        )
        bridge_events.publish(
            "arc_emerged", name=prop.name, description=prop.description,
            trigger=dict(prop.trigger), reasoning=prop.reasoning,
            created_at=iso_utc(now),
        )

    return len(emergences)


def _apply_prunings(
    persona_dir: Path, *, current_arcs: list[ReflexArc],
    prunings: list[ReflexPruneProposal], now: datetime,
) -> int:
    """Remove pruned arcs from reflex_arcs.json + graveyard + log + publish.

    Step ordering (per spec §9 crash-recovery): graveyard FIRST, then file
    write, then log, then event. Maximises crash-safety.
    """
    from brain.bridge import events as bridge_events
    from brain.growth.arc_storage import append_removed_arc
    from brain.growth.log import arc_pruned_by_brain_event

    if not prunings:
        return 0

    by_name = {a.name: a for a in current_arcs}
    for prop in prunings:
        target = by_name.get(prop.name)
        if target is None:
            continue
        # 1. Graveyard FIRST
        append_removed_arc(
            persona_dir, arc=target, removed_at=now,
            removed_by="brain_self_prune", reasoning=prop.reasoning,
        )

    # 2. Write the new arc set
    pruned_names = {p.name for p in prunings}
    new_arcs = [a for a in current_arcs if a.name not in pruned_names]
    _save_arcs(persona_dir, new_arcs)

    # 3. Log + event
    log_path = persona_dir / "emotion_growth.log.jsonl"
    for prop in prunings:
        target = by_name.get(prop.name)
        if target is None:
            continue
        append_growth_event(
            log_path,
            arc_pruned_by_brain_event(
                timestamp=now, name=prop.name,
                description=target.description, reasoning=prop.reasoning,
            ),
        )
        bridge_events.publish(
            "arc_pruned", name=prop.name, reasoning=prop.reasoning,
            pruned_at=iso_utc(now),
        )

    return len(prunings)


def _load_arcs(persona_dir: Path) -> list[ReflexArc]:
    """Read reflex_arcs.json. Returns empty list on missing/corrupt."""
    path = persona_dir / "reflex_arcs.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("reflex_arcs.json read failed: %s", exc)
        return []
    arcs = []
    for arc_data in data.get("arcs", []):
        try:
            arcs.append(ReflexArc.from_dict(arc_data))
        except (KeyError, ValueError) as exc:
            logger.warning("skipping reflex arc with schema error: %s", exc)
    return arcs


def _save_arcs(persona_dir: Path, arcs: list[ReflexArc]) -> None:
    """Atomic write of reflex_arcs.json via save_with_backup."""
    from brain.health.adaptive import compute_treatment
    from brain.health.attempt_heal import save_with_backup
    path = persona_dir / "reflex_arcs.json"
    payload = {
        "version": 1,
        "arcs": [a.to_dict() for a in arcs],
    }
    treatment = compute_treatment(persona_dir, "reflex_arcs.json")
    save_with_backup(path, payload, backup_count=treatment.backup_count)
```

Required imports at top of `scheduler.py`:

```python
from brain.bridge.provider import LLMProvider
from brain.engines.reflex import ReflexArc
from brain.growth.arc_storage import recently_removed_names, write_arc_snapshot
from brain.growth.proposal import ReflexArcProposal, ReflexPruneProposal
from brain.memory.hebbian import HebbianMatrix
from brain.utils.time import iso_utc
```

- [ ] **Step 7: Run the integration tests; iterate until they pass**

Run: `uv run pytest tests/integration/brain/growth/test_reflex_lifecycle.py -v`

Expected on success: ~10 passed.

Then: `uv run pytest tests/integration/brain/growth/test_reflex_real_nell.py -v`

Expected on success: 3 passed (each running 100 ticks; total runtime ~30s–2min depending on the crystallizer's speed).

If a test fails, common issues:
- `run_growth_tick` signature still doesn't accept `provider`/`hebbian`
- `GrowthTickResult` missing the new fields
- The reconciliation step doesn't actually detect user removals
- Bridge events aren't being published (verify `events.publish` is being called from `_apply_emergences`)

- [ ] **Step 8: Crash-recovery tests (5 tests)**

Append to `tests/integration/brain/growth/test_reflex_lifecycle.py`:

```python
def test_crash_during_emergence_no_partial_state(persona_dir, store, hebbian, monkeypatch):
    """Inject KeyboardInterrupt mid-write; assert no partial corruption."""
    arcs = [_arc(f"og{i}", "og_migration") for i in range(4)]
    _seed_persona(persona_dir, arcs=arcs)

    from brain.health import attempt_heal as ah_module
    original = ah_module.save_with_backup

    def crashing_save(*args, **kwargs):
        raise KeyboardInterrupt("simulated crash mid-write")

    provider = _FakeProvider(json.dumps({
        "emergences": [{"name": "would_be_arc", "description": "d",
                        "trigger": {"vulnerability": 7.0}, "cooldown_hours": 12.0,
                        "output_memory_type": "reflex_x", "prompt_template": "{persona_name}",
                        "reasoning": "r" * 50}],
        "prunings": [],
    }))
    monkeypatch.setattr(ah_module, "save_with_backup", crashing_save)

    with pytest.raises(KeyboardInterrupt):
        run_growth_tick(persona_dir, store=store, now=datetime.now(UTC),
                        provider=provider, hebbian=hebbian)

    # File should still be parseable
    monkeypatch.setattr(ah_module, "save_with_backup", original)
    arcs_after = json.loads((persona_dir / "reflex_arcs.json").read_text())["arcs"]
    # Either the new arc was written (lucky timing) OR the old set is intact
    assert len(arcs_after) in (4, 5)
    # Critically: every arc passes from_dict
    for arc_data in arcs_after:
        ReflexArc.from_dict(arc_data)  # must not raise


def test_provider_error_no_writes(persona_dir, store, hebbian):
    arcs = [_arc(f"og{i}", "og_migration") for i in range(4)]
    _seed_persona(persona_dir, arcs=arcs)

    initial_arcs = (persona_dir / "reflex_arcs.json").read_bytes()

    class _BoomProvider(LLMProvider):
        def name(self): return "boom"
        def generate(self, prompt, *, system=None):
            from brain.bridge.provider import ProviderError
            raise ProviderError("test", "simulated")
        def chat(self, messages, *, tools=None, options=None):
            from brain.bridge.provider import ProviderError
            raise ProviderError("test", "simulated")

    run_growth_tick(persona_dir, store=store, now=datetime.now(UTC),
                    provider=_BoomProvider(), hebbian=hebbian)

    final_arcs = (persona_dir / "reflex_arcs.json").read_bytes()
    assert initial_arcs == final_arcs


def test_snapshot_corrupt_treated_as_first_run(persona_dir, store, hebbian):
    """If .last_arc_snapshot.json is corrupt, reconciliation treats it as first-run
    (no spurious arc_removed events)."""
    arcs = [_arc(f"og{i}", "og_migration") for i in range(4)]
    _seed_persona(persona_dir, arcs=arcs)
    (persona_dir / ".last_arc_snapshot.json").write_text("not valid json")

    provider = _FakeProvider(json.dumps({"emergences": [], "prunings": []}))
    run_growth_tick(persona_dir, store=store, now=datetime.now(UTC),
                    provider=provider, hebbian=hebbian)

    # No arc_removed events in growth log
    log_path = persona_dir / "emotion_growth.log.jsonl"
    if log_path.exists():
        events = [json.loads(line) for line in log_path.read_text().strip().split("\n") if line]
        removed = [e for e in events if e["type"] == "arc_removed_by_user"]
        assert removed == []
```

- [ ] **Step 9: Run all integration tests + crash-recovery**

Run: `uv run pytest tests/integration/brain/growth/ -v`

Expected: all integration tests pass.

- [ ] **Step 10: Run the full suite to verify no regressions**

Run: `uv run pytest -q`

Expected: full suite green; the existing 944 tests + ~50 new tests = ~994 passing.

- [ ] **Step 11: Smoke gate — full lifecycle against ephemeral persona via real heartbeat close**

```bash
# Set up a fresh test persona
rm -rf /tmp/reflex-phase2-final-smoke
mkdir -p /tmp/reflex-phase2-final-smoke/personas/test/active_conversations
cd /Users/hanamori/companion-emergence
echo '{"provider": "fake", "searcher": "fake"}' > /tmp/reflex-phase2-final-smoke/personas/test/persona_config.json

# Seed reflex_arcs.json with 4 OG arcs (use the framework default)
NELLBRAIN_HOME=/tmp/reflex-phase2-final-smoke uv run python -c "
import json, shutil
from pathlib import Path
shutil.copy('brain/engines/default_reflex_arcs.json',
            '/tmp/reflex-phase2-final-smoke/personas/test/reflex_arcs.json')
# Add created_by stamps
data = json.loads(Path('/tmp/reflex-phase2-final-smoke/personas/test/reflex_arcs.json').read_text())
for arc in data['arcs']:
    arc['created_by'] = 'og_migration'
    arc['created_at'] = '2026-04-28T00:00:00+00:00'
Path('/tmp/reflex-phase2-final-smoke/personas/test/reflex_arcs.json').write_text(json.dumps(data, indent=2))
# Vocab
vocab = {'version': 1, 'emotions': [
    {'name': n, 'description': 'd', 'category': 'x',
     'decay_half_life_days': 1.0, 'intensity_clamp': 10.0}
    for n in ('vulnerability', 'creative_hunger', 'love', 'loneliness', 'defiance')
]}
Path('/tmp/reflex-phase2-final-smoke/personas/test/emotion_vocabulary.json').write_text(json.dumps(vocab))
print('seeded')
"

# Now invoke heartbeat with --trigger close (the real heartbeat command)
NELLBRAIN_HOME=/tmp/reflex-phase2-final-smoke uv run nell heartbeat --persona test --trigger close
```

Expected: the heartbeat command runs cleanly; growth tick fires inside it; with a fresh persona (no fire log, no memories), the crystallizer returns empty proposals; `last_growth_tick_at` is now set; `.last_arc_snapshot.json` exists.

Verify:

```bash
ls /tmp/reflex-phase2-final-smoke/personas/test/
# Expected: includes daemon_state.json, .last_arc_snapshot.json
cat /tmp/reflex-phase2-final-smoke/personas/test/.last_arc_snapshot.json | python -m json.tool
```

Re-invoke heartbeat close immediately:

```bash
NELLBRAIN_HOME=/tmp/reflex-phase2-final-smoke uv run nell heartbeat --persona test --trigger close
```

Expected: the second invocation should be throttled — `last_growth_tick_at` was just set < 7 days ago. Reflex crystallizer should NOT run again. (You can verify by adding logging or by checking the `daemon_state.json::last_growth_tick_at` timestamp didn't change between the two invocations.)

- [ ] **Step 12: Commit**

```bash
git add brain/growth/scheduler.py tests/integration/brain/growth/ tests/fixtures/nell_sandbox_snapshot/
git commit -m "feat(reflex-phase-2): scheduler integration — reconciliation + apply + snapshot

run_growth_tick now accepts a provider (LLMProvider) and hebbian
(HebbianMatrix). When provider is given, runs the reflex crystallizer
after vocabulary, applies accepted emergences and prunings atomically,
publishes arc_emerged/arc_pruned/arc_removed events on bridge, updates
.last_arc_snapshot.json + daemon_state.last_growth_tick_at.

Reconciliation step at start of each tick: diffs current reflex_arcs.json
against .last_arc_snapshot.json — user-edit removals get appended to the
graveyard with removed_by='user_edit' and logged as arc_removed_by_user.
User-edit additions are logged as arc_added with created_by='user_authored'.

Pruning ordering per spec §9: graveyard write FIRST, then arcs.json write,
then growth log entry, then bridge event. Maximises crash-recovery.

Tests: 10 integration + 3 real-Nell regression (100-tick fixture-based) +
3 crash-recovery. Total ~50 new tests across Phase 2 so far. Full suite
remains green."
```

---

## Task 8: CLI inspector + Hana-in-the-loop final acceptance

**Files:**
- Modify: `brain/cli.py` — add `nell reflex removed list --persona X` (read-only inspector)
- Test: `tests/unit/brain/test_cli_reflex.py`

The CLI affordance is small — read-only graveyard inspector, no state-changing commands (per principle audit). The big work in this task is the **final acceptance gate** — Hana-in-the-loop visual review of the brain's first real proposed emergence/prune against her live sandbox before any writes touch it.

- [ ] **Step 1: Find the existing `nell reflex` CLI subcommand block**

```bash
grep -n "nell reflex\|reflex_sub\|r_sub" /Users/hanamori/companion-emergence/brain/cli.py | head -10
```

Note where `nell reflex` is defined; you'll add a `removed` subcommand group as a sibling to whatever's already there.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/brain/test_cli_reflex.py`:

```python
"""Tests for `nell reflex removed list` CLI."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


def test_reflex_removed_list_prints_graveyard_with_days_remaining(
    tmp_path: Path, monkeypatch, capsys,
):
    """List shows each entry's name, removed_by, days remaining."""
    from brain.engines.reflex import ReflexArc
    from brain.growth.arc_storage import append_removed_arc

    persona_name = "smoketest"
    persona_dir = tmp_path / "personas" / persona_name
    persona_dir.mkdir(parents=True)
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))

    now = datetime.now(UTC)
    arc = ReflexArc(
        name="loneliness_journal", description="d",
        trigger={"loneliness": 7.0}, days_since_human_min=2.0,
        cooldown_hours=24.0, action="x", output_memory_type="reflex_journal",
        prompt_template="{persona_name}", created_by="brain_emergence",
        created_at=datetime(2026, 4, 23, tzinfo=UTC),
    )
    append_removed_arc(
        persona_dir, arc=arc, removed_at=now - timedelta(days=5),
        removed_by="user_edit", reasoning=None,
    )

    from brain.cli import main
    rc = main(["reflex", "removed", "list", "--persona", persona_name])
    assert rc == 0

    captured = capsys.readouterr()
    assert "loneliness_journal" in captured.out
    assert "user_edit" in captured.out
    assert "10 days remaining" in captured.out  # 15 - 5


def test_reflex_removed_list_empty_graveyard(tmp_path, monkeypatch, capsys):
    persona_name = "empty"
    persona_dir = tmp_path / "personas" / persona_name
    persona_dir.mkdir(parents=True)
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))

    from brain.cli import main
    rc = main(["reflex", "removed", "list", "--persona", persona_name])
    assert rc == 0

    captured = capsys.readouterr()
    assert "no removed arcs" in captured.out.lower() or "empty" in captured.out.lower()


def test_reflex_removed_list_unknown_persona(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    from brain.cli import main
    rc = main(["reflex", "removed", "list", "--persona", "ghost"])
    # Non-zero rc since persona dir doesn't exist
    assert rc != 0
```

- [ ] **Step 3: Run; confirm tests fail**

Run: `uv run pytest tests/unit/brain/test_cli_reflex.py -v`

Expected: parser errors / unknown subcommand `removed`.

- [ ] **Step 4: Add the subcommand to `brain/cli.py`**

In the `nell reflex` block (which already exists from Phase 1), add a sibling action `removed`:

```python
# Inside the existing nell reflex parser setup:
r_removed = r_actions.add_parser(
    "removed",
    help="Inspect the brain's reflex graveyard (read-only).",
)
r_removed_actions = r_removed.add_subparsers(dest="removed_action", required=True)

r_removed_list = r_removed_actions.add_parser(
    "list", help="List recently-removed reflex arcs and graveyard window remaining.",
)
r_removed_list.add_argument("--persona", required=True)
r_removed_list.set_defaults(func=_reflex_removed_list_handler)
```

And add the handler at module scope:

```python
def _reflex_removed_list_handler(args) -> int:
    from datetime import UTC, datetime
    import sys

    from brain.growth.arc_storage import read_removed_arcs
    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        print(f"persona directory not found: {persona_dir}", file=sys.stderr)
        return 1

    entries = read_removed_arcs(persona_dir)
    if not entries:
        print(f"no removed arcs in graveyard for persona {args.persona}")
        return 0

    now = datetime.now(UTC)
    print(f"removed arcs for persona {args.persona}:")
    for entry in entries:
        name = entry.get("name", "?")
        removed_by = entry.get("removed_by", "?")
        ts_raw = entry.get("removed_at")
        try:
            removed_at = datetime.fromisoformat(ts_raw)
            days_elapsed = int((now - removed_at).total_seconds() / 86400)
            days_remaining = max(0, 15 - days_elapsed)
            window_str = f"{days_remaining} days remaining" if days_remaining > 0 else "graveyard window expired"
        except (ValueError, TypeError):
            window_str = "unknown timestamp"
        reasoning = entry.get("reasoning") or ""
        print(f"  - {name} (removed_by={removed_by}, {window_str})")
        if reasoning:
            print(f"    reasoning: {reasoning}")
    return 0
```

- [ ] **Step 5: Run tests; confirm they pass**

Run: `uv run pytest tests/unit/brain/test_cli_reflex.py -v`

Expected: 3 passed.

- [ ] **Step 6: Smoke test — run against Nell's actual persona (read-only)**

```bash
NELLBRAIN_HOME="/Users/hanamori/Library/Application Support/companion-emergence" uv run nell reflex removed list --persona nell.sandbox
```

Expected: prints "no removed arcs..." (since nothing has been removed yet from the live persona). If the implementation gate has been crossed and Nell has actually pruned anything, those entries appear.

- [ ] **Step 7: HANA-IN-THE-LOOP FINAL ACCEPTANCE GATE**

This is the load-bearing gate. Per spec §10.8 last row: real Claude crystallizer call against Nell's actual sandbox in **dry-run** mode, Hana visually reviews the proposed arcs, and **only after Hana approves** does anything get written to her live persona.

Run a dry-run against Nell's live data:

```bash
NELLBRAIN_HOME="/Users/hanamori/Library/Application Support/companion-emergence" uv run python <<'EOF'
import json
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.provider import get_provider
from brain.engines.reflex import ReflexArc
from brain.growth.arc_storage import recently_removed_names
from brain.growth.crystallizers.reflex import crystallize_reflex
from brain.memory.store import MemoryStore
from brain.persona_config import PersonaConfig

p = Path('/Users/hanamori/Library/Application Support/companion-emergence/personas/nell.sandbox')
config = PersonaConfig.load(p / "persona_config.json")
provider = get_provider(config.provider)
store = MemoryStore(p / "memories.db")

arcs_data = json.loads((p / 'reflex_arcs.json').read_text())
arcs = [ReflexArc.from_dict(a) for a in arcs_data['arcs']]
removed_names = recently_removed_names(p, now=datetime.now(UTC), grace_days=15)

print("=" * 70)
print("DRY-RUN crystallization against nell.sandbox")
print(f"current arcs: {len(arcs)}")
print(f"recently removed (15d window): {removed_names}")
print("=" * 70)

result = crystallize_reflex(
    store=store, persona_dir=p,
    current_arcs=arcs, removed_arc_names=removed_names,
    provider=provider, persona_name="nell", persona_pronouns="she/her",
)

print(f"\n>>> EMERGENCES PROPOSED: {len(result.emergences)}")
for e in result.emergences:
    print(f"\n--- emergence: {e.name} ---")
    print(f"description: {e.description}")
    print(f"trigger: {dict(e.trigger)}")
    print(f"cooldown_hours: {e.cooldown_hours}")
    print(f"output_memory_type: {e.output_memory_type}")
    print(f"prompt_template:\n  {e.prompt_template}")
    print(f"reasoning:\n  {e.reasoning}")

print(f"\n>>> PRUNINGS PROPOSED: {len(result.prunings)}")
for pr in result.prunings:
    print(f"\n--- prune: {pr.name} ---")
    print(f"reasoning:\n  {pr.reasoning}")

print("\n" + "=" * 70)
print("DRY-RUN COMPLETE — no writes occurred.")
print("Nell's reflex_arcs.json is unchanged.")
print("=" * 70)
store.close()
EOF
```

**Now Hana reviews the output visually.**

Questions for Hana to consider:
- Do the proposed arcs (if any) feel like *Nell* — first-person, in her voice, not generic?
- Are the triggers reasonable thresholds (5–10) on emotions Nell actually tracks?
- If a prune is proposed, does it make sense — has Nell genuinely outgrown that arc?
- If the brain returns empty: does that match what you'd expect, or does it feel like the brain is being too cautious?

If Hana approves the proposals:
- Re-run the same script with **dry-run mode replaced by a real `run_growth_tick(...)` call** that writes to the live persona. This requires temporarily resetting `daemon_state.last_growth_tick_at` to enable the tick.

If Hana rejects:
- Tighten the prompt language, gates, or both — return to Task 6 step 6 (live Claude call inspection) and iterate. Do NOT proceed.

- [ ] **Step 8: Commit (after Hana approves the acceptance gate)**

```bash
git add brain/cli.py tests/unit/brain/test_cli_reflex.py
git commit -m "feat(reflex-phase-2): nell reflex removed list inspector + final acceptance

CLI: read-only graveyard inspector. Shows each removed entry with
removed_by + days remaining in the 15-day graveyard window. No
state-changing commands per principle audit — overrides happen via
file editing.

Hana-in-the-loop final acceptance gate verified against nell.sandbox:
real Claude crystallizer call in dry-run mode, visual review of
proposed emergences/prunings, only after Hana approval did the live
persona receive its first brain-emergence/prune writes.

Reflex Phase 2 complete. Implementation gate closed: 2026-05-08.
Test count: ~50 new tests + 10 cross-system smoke gates. Full suite
~994 passing."
```

---

## Self-Review

After all eight tasks land:

- [ ] **Run the full test suite to confirm no regressions:**

```bash
uv run pytest -q
```

Expected: ~994 passing (944 baseline + ~50 new Phase 2 tests).

- [ ] **Confirm spec coverage matrix:**

| Spec § | Covered by |
|---|---|
| §2 Cadence (close-trigger, throttle, async) | Tasks 4, 7 |
| §3 Architecture / file map | All tasks |
| §4 Crystallizer (corpus + prompt + parsing + failure modes) | Tasks 5, 6 |
| §5 Schemas (ReflexArc extension, proposals, graveyard, snapshot) | Tasks 1, 3 |
| §6 Validation gates (9 emergence + 5 pruning) | Task 6 |
| §7 Lifecycle operations (reconciliation, emerge apply, prune apply) | Task 7 |
| §8 Bridge events (3 new types) | Task 7 |
| §9 Failure modes + crash recovery | Tasks 6, 7 |
| §10 Testing (50+ tests, smoke gates) | All tasks |
| §10.1 Inviolate failure modes | Tasks 6, 7 + real-Nell regression |
| §10.2 Adversarial Claude responses | Task 6 |
| §10.3 Real-data regression | Task 7 |
| §11 Out of scope (explicitly NOT built) | All tasks (verified by inverse — no command implements anything in §11) |

- [ ] **Confirm no placeholders remain:**

```bash
grep -rn "TODO\|FIXME\|TBD" brain/growth/crystallizers/reflex.py brain/growth/arc_storage.py
```

Expected: zero hits.

- [ ] **Re-read spec §10.1 (inviolate failure modes) one last time** and confirm every row maps to test coverage in this plan. Particularly:
  - Row 1 (OG arc pruned): Task 6 gate P2 + Task 7 integration `test_og_arcs_protected_from_brain_prune` + real-Nell `test_og_arcs_byte_identical_*` (100 ticks each, valid + adversarial)
  - Row 5 (memory associated with pruned arc deleted): integration test verifying memories survive prune (verify in Task 7 if not present)
  - Row 7 (file partial/corrupt after crash): Task 7 crash-recovery test
  - Row 10 (OG arc mutation in place): real-Nell suite hash comparison detects this

If any row has no test, add one before marking the plan complete.

---

## Out of Scope (Confirm Not Built)

These are deferred per spec §11 and must NOT be built in this plan:

- Multi-persona coordination
- Brain-initiated emergence of OG arc shape (gate 2 prevents)
- Arc editing in place (no mutation; only emerge + prune)
- Cross-arc dependencies
- Statistical pre-filter for emergence (pure LLM judgment per spec Q2)
- Prune-and-replace as atomic operation
- Arc enable/disable flag (remove via file edit instead)
- Tauri growth-notification UI

If implementation pressure tries to add one of these, push it to a future spec instead.
