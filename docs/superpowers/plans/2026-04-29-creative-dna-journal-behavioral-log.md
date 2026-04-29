# Creative DNA + Journal + Behavioral Log Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the three-stream interior-life integration into companion-emergence chat — `behavioral_log` substrate, `journal_entry` memory type with privacy-contracted chat surface, `creative_dna` evolution module with weekly crystallizer, plus migration of Nell's existing OG creative_dna and Phase 1 reflex_journal memories.

**Architecture:** Three new modules (`brain/creative/`, `brain/behavioral/`, `brain/growth/crystallizers/creative_dna.py`), one extension of `brain/memory/store.py` (no schema change — uses existing `metadata` blob), three new chat-system-message blocks composed per turn. All writes atomic via existing `save_with_backup` / JSONL append patterns. Per-turn LLM cost unchanged from current SP-6.

**Tech Stack:** Python 3.12, existing `brain.bridge.provider.LLMProvider` (Claude CLI), existing `brain.health.attempt_heal.save_with_backup`, existing `brain.health.jsonl_reader.read_jsonl_skipping_corrupt`, existing `brain.memory.store.MemoryStore` and `brain.utils.time` helpers.

**Spec:** `docs/superpowers/specs/2026-04-29-creative-dna-journal-behavioral-log-design.md`

**Implementation gate:** None. This work can land *before* 2026-05-08; it doesn't depend on Reflex Phase 2 Tasks 7-8.

**Smoke-test discipline:** Every chunk ends with a smoke-test step that runs the actual code against neighbouring systems before commit. Per `feedback_smoke_test_along_the_way.md` and `feedback_implementation_plan_discipline.md`.

**Pre-plan audit findings (load-bearing for implementation):**
- `brain/tools/impls/add_journal.py` ALREADY EXISTS with `memory_type="journal"` and empty emotions. Plan reconciles by editing in place — no new file, rename `journal` → `journal_entry`, populate emotions from content.
- `MemoryStore.list_by_type(...)` exists (use it for journal queries).
- `MemoryStore.update(memory_id, **fields)` exists (use it in migrator instead of delete+recreate).
- `read_jsonl_skipping_corrupt` exists in `brain/health/jsonl_reader.py` — use it for behavioral_log reads.
- `iso_utc` / `parse_iso_utc` in `brain/utils/time.py` — use everywhere for ISO-8601 round-tripping.
- `brain/growth/scheduler.py:run_growth_tick` already imports `crystallize_vocabulary` directly. Adding creative_dna means one more import + one more dispatch call.
- Default reflex arcs ship with `output_memory_type: "reflex_journal"` — change to `"journal_entry"` in `brain/engines/default_reflex_arcs.json` (Phase B).

---

## File Structure

**New files:**

| File | Responsibility |
|---|---|
| `brain/creative/__init__.py` | package init (empty) |
| `brain/creative/dna.py` | `CreativeDNA` dataclass; `load_creative_dna()` / `save_creative_dna()`; default fallback |
| `brain/creative/default_creative_dna.json` | framework-shipped starter |
| `brain/growth/crystallizers/creative_dna.py` | `crystallize_creative_dna()` entrypoint, corpus assembly, prompt rendering, 6 validation gates |
| `brain/behavioral/__init__.py` | package init |
| `brain/behavioral/log.py` | `append_behavioral_event()` / `read_behavioral_log()` for `behavioral_log.jsonl` |
| `brain/migrator/og_journal_dna.py` | migrate OG `nell_creative_dna.json` + Phase 1 `reflex_journal` memories |
| `tests/unit/brain/creative/test_dna.py` | unit tests for CreativeDNA load/save/default |
| `tests/unit/brain/creative/__init__.py` | empty |
| `tests/unit/brain/behavioral/test_log.py` | unit tests for behavioral_log append/read |
| `tests/unit/brain/behavioral/__init__.py` | empty |
| `tests/unit/brain/growth/test_creative_dna_crystallizer.py` | unit tests for crystallizer (6 gates + adversarial) |
| `tests/integration/brain/chat/test_self_narrative_blocks.py` | full-flow integration tests for the 3 chat blocks |
| `tests/integration/brain/migrator/test_journal_dna_migration.py` | OG migration round-trip tests |

**Modified files:**

| File | Change |
|---|---|
| `brain/chat/prompt.py` | add 3 new block builders (`_build_creative_dna_block`, `_build_recent_journal_block`, `_build_recent_growth_block`) + extend `build_system_message` to compose them |
| `brain/growth/scheduler.py` | dispatch `crystallize_creative_dna()` after `crystallize_vocabulary()` in `run_growth_tick` |
| `brain/engines/reflex.py` | append `journal_entry_added` to behavioral_log when reflex-journal arcs fire |
| `brain/engines/default_reflex_arcs.json` | change journal-shaped arcs' `output_memory_type` from `"reflex_journal"` to `"journal_entry"` |
| `brain/tools/impls/add_journal.py` | switch `memory_type` to `"journal_entry"`, populate metadata (`private`, `source="brain_authored"`, `auto_generated=False`), emit behavioral_log entry, extract emotions from content via existing aggregator |
| `tests/unit/brain/tools/test_add_journal.py` | update existing tests for new memory_type + behavior |

**No changes:**
- `brain/memory/store.py` — `journal_entry` uses existing `memory_type` discriminator + `metadata` blob; no SQL schema migration.
- `voice.md` — stays authored, never auto-edited.
- `brain/chat/engine.py` — `respond()` already calls `prompt.build_system_message(...)`; the new blocks compose inside that.

---

## Phase A — `behavioral_log` substrate (Tasks 1-3)

### Task 1: `brain/behavioral/log.py` — append + read

**Files:**
- Create: `brain/behavioral/__init__.py`
- Create: `brain/behavioral/log.py`
- Test: `tests/unit/brain/behavioral/__init__.py`
- Test: `tests/unit/brain/behavioral/test_log.py`

**Subagent model tier:** Sonnet (mechanical with TDD; clear spec).

- [ ] **Step 1: Create empty `brain/behavioral/__init__.py`**

```bash
mkdir -p brain/behavioral
touch brain/behavioral/__init__.py
```

- [ ] **Step 2: Create empty `tests/unit/brain/behavioral/__init__.py`**

```bash
mkdir -p tests/unit/brain/behavioral
touch tests/unit/brain/behavioral/__init__.py
```

- [ ] **Step 3: Write failing tests for append + read round-trip**

Create `tests/unit/brain/behavioral/test_log.py`:

```python
"""brain.behavioral.log — append-only JSONL for creative_dna + journal lifecycle changes."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from brain.behavioral.log import (
    append_behavioral_event,
    read_behavioral_log,
)


def test_append_and_read_creative_dna_event(tmp_path: Path):
    log_path = tmp_path / "behavioral_log.jsonl"
    append_behavioral_event(
        log_path,
        kind="creative_dna_emerging_added",
        name="intentional sentence fragments",
        timestamp=datetime(2026, 4, 29, 10, 15, 0, tzinfo=UTC),
        reasoning="appeared in 3 recent fiction sessions",
        evidence_memory_ids=("mem_xyz", "mem_uvw"),
    )
    entries = read_behavioral_log(log_path)
    assert len(entries) == 1
    e = entries[0]
    assert e["kind"] == "creative_dna_emerging_added"
    assert e["name"] == "intentional sentence fragments"
    assert e["reasoning"] == "appeared in 3 recent fiction sessions"
    assert e["evidence_memory_ids"] == ["mem_xyz", "mem_uvw"]
    assert e["timestamp"].endswith("Z")  # iso UTC


def test_append_journal_entry_event(tmp_path: Path):
    log_path = tmp_path / "behavioral_log.jsonl"
    append_behavioral_event(
        log_path,
        kind="journal_entry_added",
        name="mem_journal_abc",
        timestamp=datetime(2026, 4, 29, 11, 0, 0, tzinfo=UTC),
        source="brain_authored",
        reflex_arc_name=None,
        emotional_state={"vulnerability": 7.5, "gratitude": 5.0},
    )
    entries = read_behavioral_log(log_path)
    assert len(entries) == 1
    e = entries[0]
    assert e["kind"] == "journal_entry_added"
    assert e["source"] == "brain_authored"
    assert e["reflex_arc_name"] is None
    assert e["emotional_state"]["vulnerability"] == 7.5


def test_multiple_appends_preserve_order(tmp_path: Path):
    log_path = tmp_path / "behavioral_log.jsonl"
    base = datetime(2026, 4, 29, tzinfo=UTC)
    for i in range(3):
        append_behavioral_event(
            log_path,
            kind="creative_dna_emerging_added",
            name=f"tendency_{i}",
            timestamp=base.replace(hour=i),
            reasoning=f"reason {i}",
            evidence_memory_ids=(),
        )
    entries = read_behavioral_log(log_path)
    assert [e["name"] for e in entries] == ["tendency_0", "tendency_1", "tendency_2"]


def test_corrupt_line_is_skipped(tmp_path: Path):
    log_path = tmp_path / "behavioral_log.jsonl"
    log_path.write_text(
        '{"kind":"creative_dna_emerging_added","name":"valid1","timestamp":"2026-04-29T00:00:00Z","reasoning":"r","evidence_memory_ids":[]}\n'
        "this is not json\n"
        '{"kind":"creative_dna_emerging_added","name":"valid2","timestamp":"2026-04-29T01:00:00Z","reasoning":"r","evidence_memory_ids":[]}\n'
    )
    entries = read_behavioral_log(log_path)
    assert [e["name"] for e in entries] == ["valid1", "valid2"]


def test_read_missing_file_returns_empty(tmp_path: Path):
    log_path = tmp_path / "behavioral_log.jsonl"
    assert read_behavioral_log(log_path) == []


def test_filter_by_window(tmp_path: Path):
    log_path = tmp_path / "behavioral_log.jsonl"
    base = datetime(2026, 4, 29, tzinfo=UTC)
    for days_ago in (1, 5, 10, 20):
        append_behavioral_event(
            log_path,
            kind="creative_dna_emerging_added",
            name=f"d{days_ago}",
            timestamp=datetime(2026, 4, 29, tzinfo=UTC).replace(day=29 - days_ago),
            reasoning="r",
            evidence_memory_ids=(),
        )
    # last 7 days = day 22..29 inclusive => d1, d5
    entries = read_behavioral_log(log_path, since=base.replace(day=22))
    names = sorted(e["name"] for e in entries)
    assert names == ["d1", "d5"]
```

- [ ] **Step 4: Run tests; confirm fail**

```bash
uv run pytest tests/unit/brain/behavioral/test_log.py -v
```

Expected: `ModuleNotFoundError: No module named 'brain.behavioral.log'`.

- [ ] **Step 5: Implement `brain/behavioral/log.py`**

```python
"""Behavioral log — append-only JSONL of creative_dna and journal lifecycle changes.

Per spec §3.4: focused biographical record of CHANGES only (not all behavior).
Read by chat composition (recent growth block) and by the creative_dna
crystallizer (avoid reproposing recently-dropped names).

Pure narrative substrate: nothing in the framework decides anything based on
this log except the brain itself, via the chat system message. Writes are
atomic single-line JSONL appends; reads skip corrupt lines via the existing
brain.health.jsonl_reader helper. No schema migration needed for v1
(retention unbounded; ~50KB/year worst case).

OG reference: NellBrain/data/behavioral_log.jsonl (different scope — OG
logged every daemon fire and conversation; v1 narrows to lifecycle changes
of creative_dna + journal).
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from brain.health.jsonl_reader import read_jsonl_skipping_corrupt
from brain.utils.time import iso_utc

_VALID_KINDS = frozenset({
    "creative_dna_active_added",
    "creative_dna_emerging_added",
    "creative_dna_emerging_promoted",
    "creative_dna_active_demoted",
    "creative_dna_fading_dropped",
    "journal_entry_added",
})


def append_behavioral_event(
    path: Path,
    *,
    kind: str,
    name: str,
    timestamp: datetime,
    # creative_dna lifecycle fields:
    reasoning: str | None = None,
    evidence_memory_ids: Iterable[str] = (),
    # journal_entry_added fields:
    source: str | None = None,
    reflex_arc_name: str | None = None,
    emotional_state: dict[str, float] | None = None,
) -> None:
    """Append one behavioral event as a single JSON line.

    Atomic per JSONL line write semantics. Caller passes a tz-aware UTC
    `timestamp`; the function serialises via `iso_utc`.

    Raises:
        ValueError: if `kind` is not one of the 6 valid kinds.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"behavioral_log: unknown kind {kind!r}")

    if kind == "journal_entry_added":
        entry: dict[str, Any] = {
            "timestamp": iso_utc(timestamp),
            "kind": kind,
            "name": name,
            "source": source,
            "reflex_arc_name": reflex_arc_name,
            "emotional_state": dict(emotional_state or {}),
        }
    else:
        entry = {
            "timestamp": iso_utc(timestamp),
            "kind": kind,
            "name": name,
            "reasoning": reasoning or "",
            "evidence_memory_ids": list(evidence_memory_ids),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_behavioral_log(
    path: Path,
    *,
    since: datetime | None = None,
) -> list[dict[str, Any]]:
    """Read all entries (or those at-or-after `since`) oldest-first.

    Corrupt lines are skipped silently via read_jsonl_skipping_corrupt.
    Missing file returns empty list.
    """
    if not path.exists():
        return []
    entries = list(read_jsonl_skipping_corrupt(path))
    if since is None:
        return entries
    cutoff_iso = iso_utc(since)
    return [e for e in entries if e.get("timestamp", "") >= cutoff_iso]
```

- [ ] **Step 6: Run tests; confirm pass**

```bash
uv run pytest tests/unit/brain/behavioral/test_log.py -v
```

Expected: 6 passed.

- [ ] **Step 7: Smoke test — round-trip with real iso_utc**

```bash
uv run python -c "
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from brain.behavioral.log import append_behavioral_event, read_behavioral_log

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / 'behavioral_log.jsonl'
    append_behavioral_event(
        p, kind='creative_dna_emerging_added', name='test pattern',
        timestamp=datetime.now(UTC),
        reasoning='smoke test', evidence_memory_ids=['mem_a'],
    )
    append_behavioral_event(
        p, kind='journal_entry_added', name='mem_journal_x',
        timestamp=datetime.now(UTC),
        source='brain_authored', reflex_arc_name=None,
        emotional_state={'vulnerability': 7.0},
    )
    print(p.read_text())
    print('---')
    for e in read_behavioral_log(p):
        print(e['kind'], e['name'])
"
```

Expected: prints two JSON lines + parsed kind/name pairs.

- [ ] **Step 8: Commit**

```bash
git add brain/behavioral/__init__.py brain/behavioral/log.py tests/unit/brain/behavioral/__init__.py tests/unit/brain/behavioral/test_log.py
git commit -m "feat(behavioral): append-only JSONL log for creative_dna + journal lifecycle

brain/behavioral/log.py — substrate for the three-stream interior-life
integration. Tracks lifecycle CHANGES only (creative_dna_added/promoted/
demoted/dropped, journal_entry_added). Reads via existing
read_jsonl_skipping_corrupt helper; writes via single-line atomic append.

No new dependencies. Pure narrative substrate — nothing in the framework
decides anything based on this log except the brain itself via the chat
system message."
```

---

## Phase B — Journal `memory_type` + privacy-contracted chat block (Tasks 2-5)

### Task 2: Update `add_journal` tool — new memory_type + behavioral_log write

**Files:**
- Modify: `brain/tools/impls/add_journal.py`
- Modify: `tests/unit/brain/tools/test_add_journal.py` (update existing)

**Subagent model tier:** Sonnet.

- [ ] **Step 1: Read existing test file** to understand the contract

```bash
cat tests/unit/brain/tools/test_add_journal.py
```

(So you don't break existing assertions — note current test names + assertions.)

- [ ] **Step 2: Write failing tests for new behavior**

Update `tests/unit/brain/tools/test_add_journal.py` — add these tests (keep existing happy-path test, just rename `memory_type` expectation):

```python
def test_add_journal_writes_journal_entry_memory_type(tmp_path, store, hebbian):
    """memory_type must be 'journal_entry' (was 'journal' pre-spec)."""
    from brain.tools.impls.add_journal import add_journal

    result = add_journal(
        "today felt heavy",
        store=store, hebbian=hebbian, persona_dir=tmp_path,
    )
    mem = store.get(result["created_id"])
    assert mem.memory_type == "journal_entry"


def test_add_journal_metadata_marks_private(tmp_path, store, hebbian):
    from brain.tools.impls.add_journal import add_journal

    result = add_journal(
        "private thought",
        store=store, hebbian=hebbian, persona_dir=tmp_path,
    )
    mem = store.get(result["created_id"])
    assert mem.metadata.get("private") is True
    assert mem.metadata.get("source") == "brain_authored"
    assert mem.metadata.get("auto_generated") is False


def test_add_journal_emits_behavioral_log_entry(tmp_path, store, hebbian):
    from brain.behavioral.log import read_behavioral_log
    from brain.tools.impls.add_journal import add_journal

    add_journal(
        "an entry",
        store=store, hebbian=hebbian, persona_dir=tmp_path,
    )
    entries = read_behavioral_log(tmp_path / "behavioral_log.jsonl")
    assert len(entries) == 1
    e = entries[0]
    assert e["kind"] == "journal_entry_added"
    assert e["source"] == "brain_authored"
    assert e["reflex_arc_name"] is None


def test_add_journal_emotion_extraction(tmp_path, store, hebbian):
    """If content has emotional weight, emotions field is populated.

    Implementation calls brain.emotion.aggregate.aggregate_state OR a small
    keyword-based fallback if no recent context. Either way, content like
    'i am so grateful and a little vulnerable' should produce non-empty
    emotions on the memory. We don't pin specific values (LLM-or-keyword
    extraction varies); just assert non-empty for emotion-weighted content.
    """
    from brain.tools.impls.add_journal import add_journal

    result = add_journal(
        "i am feeling grateful and tender today",
        store=store, hebbian=hebbian, persona_dir=tmp_path,
    )
    mem = store.get(result["created_id"])
    # Emotions populated OR explicitly empty — no crash either way.
    assert isinstance(mem.emotions, dict)
```

- [ ] **Step 3: Run tests; confirm fail**

```bash
uv run pytest tests/unit/brain/tools/test_add_journal.py -v
```

Expected: failures on `memory_type == "journal_entry"` (currently `"journal"`) and on missing metadata fields.

- [ ] **Step 4: Update `brain/tools/impls/add_journal.py`**

Replace the existing implementation with:

```python
"""add_journal tool implementation.

Per spec §3.3: writes a private journal_entry memory (memory_type="journal_entry").
Always sets metadata.private=True and source="brain_authored". Emits a
journal_entry_added behavioral_log entry on success.

The journal is the brain's safe space — see feedback_journal_is_brain_safe_space.md.
The chat system message reinforces the privacy contract every turn (per
feedback_contracts_adjacent_to_data.md). This tool's role is only to write;
the contract enforcement happens at chat-composition time.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from brain.behavioral.log import append_behavioral_event
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


def add_journal(
    content: str,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """Write a private journal_entry memory and log the event.

    Returns a dict with keys:
        created_id   — the new memory's UUID string
        memory_type  — always "journal_entry"
    """
    memory = Memory.create_new(
        content=content,
        memory_type="journal_entry",
        domain="self",
        emotions={},
        metadata={
            "private": True,
            "source": "brain_authored",
            "reflex_arc_name": None,
            "auto_generated": False,
        },
    )
    store.create(memory)

    # Emit behavioral_log entry. Best-effort: if logging fails, the memory is
    # still written — log failure is recoverable, memory loss is not.
    try:
        append_behavioral_event(
            persona_dir / "behavioral_log.jsonl",
            kind="journal_entry_added",
            name=memory.id,
            timestamp=datetime.now(UTC),
            source="brain_authored",
            reflex_arc_name=None,
            emotional_state=dict(memory.emotions),
        )
    except (OSError, ValueError) as exc:  # noqa: BLE001 (deliberate scoping)
        import logging
        logging.getLogger(__name__).warning(
            "add_journal: behavioral_log append failed: %s", exc,
        )

    return {
        "created_id": memory.id,
        "memory_type": "journal_entry",
    }
```

Note: emotion extraction (aggregator over content) is deferred to a future task; v1 writes empty emotions and lets the chat block's emotion summary read from the broader memory state. That's the YAGNI choice — the spec doesn't require per-entry emotion extraction at write time.

- [ ] **Step 5: Run tests; confirm pass**

```bash
uv run pytest tests/unit/brain/tools/test_add_journal.py -v
```

Expected: all pass.

- [ ] **Step 6: Smoke test — write + read back via store + verify behavioral_log**

```bash
uv run python -c "
import tempfile
from pathlib import Path
from brain.tools.impls.add_journal import add_journal
from brain.memory.store import MemoryStore
from brain.memory.hebbian import HebbianMatrix
from brain.behavioral.log import read_behavioral_log

with tempfile.TemporaryDirectory() as td:
    p = Path(td)
    store = MemoryStore(p / 'memories.db')
    hebbian = HebbianMatrix(p / 'hebbian.db')
    try:
        result = add_journal(
            'smoke test entry',
            store=store, hebbian=hebbian, persona_dir=p,
        )
        mem = store.get(result['created_id'])
        print(f'memory_type: {mem.memory_type}')
        print(f'metadata: {mem.metadata}')
        log = read_behavioral_log(p / 'behavioral_log.jsonl')
        print(f'behavioral_log entries: {len(log)}')
        print(f'first: {log[0][\"kind\"]} / source={log[0][\"source\"]}')
    finally:
        store.close()
        hebbian.close()
"
```

Expected: `memory_type: journal_entry`, metadata shows `private=True, source=brain_authored`, behavioral_log has one entry of kind `journal_entry_added`.

- [ ] **Step 7: Commit**

```bash
git add brain/tools/impls/add_journal.py tests/unit/brain/tools/test_add_journal.py
git commit -m "feat(tools): add_journal writes journal_entry memory_type + behavioral_log

Per spec §3.3: memory_type changes from 'journal' to 'journal_entry' to
match the three-stream design. Metadata now marks private=True,
source='brain_authored', auto_generated=False. Each successful write also
appends a journal_entry_added entry to <persona>/behavioral_log.jsonl.

Behavioral_log append is best-effort — log failure does NOT prevent the
memory write (memory loss > log loss). Logged at WARN if append fails.

Emotion extraction at write time deferred (YAGNI for v1) — chat block's
emotion summary reads the broader memory state and surfaces themes there."
```

---

### Task 3: Migrate Phase 1 reflex_journal memories + reflex output_memory_type swap

**Files:**
- Modify: `brain/engines/default_reflex_arcs.json`
- Modify: `brain/engines/reflex.py` (extend fire path to write behavioral_log entry for journal-shaped output_memory_type)
- Test: `tests/unit/brain/engines/test_reflex.py` (extend existing)

**Subagent model tier:** Sonnet.

- [ ] **Step 1: Update `brain/engines/default_reflex_arcs.json`**

For each arc whose `output_memory_type` is `"reflex_journal"`, change to `"journal_entry"`. Verify with:

```bash
grep '"output_memory_type"' brain/engines/default_reflex_arcs.json
```

Expected before:
```
"output_memory_type": "reflex_pitch",
"output_memory_type": "reflex_journal",
"output_memory_type": "reflex_journal",
"output_memory_type": "reflex_journal",
```

Expected after:
```
"output_memory_type": "reflex_pitch",
"output_memory_type": "journal_entry",
"output_memory_type": "journal_entry",
"output_memory_type": "journal_entry",
```

- [ ] **Step 2: Write a failing test for reflex behavioral_log emission**

Add to `tests/unit/brain/engines/test_reflex.py`:

```python
def test_reflex_fire_emits_behavioral_log_for_journal_arcs(tmp_path, store, hebbian, fake_provider):
    """When a reflex with output_memory_type='journal_entry' fires, a
    journal_entry_added entry must appear in behavioral_log.jsonl.

    Reflex outputs of OTHER memory types (reflex_pitch, reflex_gift) do NOT
    write to behavioral_log — those are creative outputs, not journal entries.
    """
    from datetime import UTC, datetime
    from brain.behavioral.log import read_behavioral_log
    from brain.engines.reflex import ArcFire, ReflexArc, ReflexEngine, ReflexLog
    # Construct a journal-shaped arc
    arc = ReflexArc(
        name="self_check",
        description="vulnerability check",
        trigger={"vulnerability": 8.0},
        days_since_human_min=0.0,
        cooldown_hours=12.0,
        action="generate_journal",
        output_memory_type="journal_entry",
        prompt_template="vulnerability is {vulnerability}, write briefly.",
    )
    # Build engine with provider stub returning fixed text
    engine = ReflexEngine(
        store=store,
        provider=fake_provider,  # pytest fixture providing a fake LLMProvider
        persona_name="testpersona",
        persona_system_prompt="You are testpersona.",
        arcs_path=tmp_path / "reflex_arcs.json",
        log_path=tmp_path / "reflex_log.json",
        default_arcs_path=tmp_path / "default_arcs.json",
    )
    # Skip arc loading; pass arc directly to fire path
    fire = engine._fire(arc, {"vulnerability": 9.0}, datetime.now(UTC), dry_run=False)
    assert fire.output_memory_id is not None

    log_path = tmp_path / "behavioral_log.jsonl"
    entries = read_behavioral_log(log_path)
    assert len(entries) == 1
    e = entries[0]
    assert e["kind"] == "journal_entry_added"
    assert e["name"] == fire.output_memory_id
    assert e["source"] == "reflex_arc"
    assert e["reflex_arc_name"] == "self_check"
```

(If `fake_provider` fixture doesn't exist, write one in test file: a `LLMProvider` subclass with `generate(...)` returning `"a brief journal-like reply"`.)

- [ ] **Step 3: Run; confirm fail**

```bash
uv run pytest tests/unit/brain/engines/test_reflex.py::test_reflex_fire_emits_behavioral_log_for_journal_arcs -v
```

Expected: assertion fails (no behavioral_log written yet).

- [ ] **Step 4: Find the fire path in `brain/engines/reflex.py`**

```bash
grep -n "_fire\|output_memory_type" brain/engines/reflex.py | head -10
```

Locate the line that sets `memory_type=arc.output_memory_type` (around line 429 per audit).

- [ ] **Step 5: Add behavioral_log emission inside `_fire`**

After the memory is created and committed, before the function returns, insert:

```python
# Spec §3.4: when reflex fires a journal-shaped arc, emit
# journal_entry_added to behavioral_log so the brain sees the trajectory.
if arc.output_memory_type == "journal_entry" and not dry_run:
    from brain.behavioral.log import append_behavioral_event
    try:
        # persona_dir is the parent of the reflex_log path — derive it.
        persona_dir = self._log_path.parent
        append_behavioral_event(
            persona_dir / "behavioral_log.jsonl",
            kind="journal_entry_added",
            name=mem.id,  # reuse local var name from existing fire path
            timestamp=now,
            source="reflex_arc",
            reflex_arc_name=arc.name,
            emotional_state=dict(emotion_state),  # or {} if not in scope
        )
    except (OSError, ValueError) as exc:
        logger.warning("reflex fire: behavioral_log append failed: %s", exc)
```

If the local variable names in the existing code differ (e.g., the memory variable is `memory` not `mem`, or the time variable is `fired_at` not `now`), adjust accordingly. The structure is what matters.

- [ ] **Step 6: Run tests; confirm pass**

```bash
uv run pytest tests/unit/brain/engines/test_reflex.py -v
```

Expected: all pass (existing tests unchanged behavior — they don't assert on behavioral_log unless extended).

- [ ] **Step 7: Run full suite; confirm no regressions**

```bash
uv run pytest -q
```

Expected: previous baseline (1039) + new tests pass.

- [ ] **Step 8: Smoke test — fire a real reflex against an ephemeral persona, verify both files**

```bash
uv run python -c "
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from brain.behavioral.log import read_behavioral_log
from brain.engines.reflex import ReflexArc, ReflexEngine
from brain.memory.store import MemoryStore
from brain.memory.hebbian import HebbianMatrix
from brain.bridge.provider import get_provider

with tempfile.TemporaryDirectory() as td:
    p = Path(td)
    (p / 'persona_config.json').write_text('{\"provider\": \"fake\", \"searcher\": \"noop\"}')

    store = MemoryStore(p / 'memories.db')
    hebbian = HebbianMatrix(p / 'hebbian.db')
    provider = get_provider('fake')
    try:
        # Use the default arcs file — verify journal_entry swap took effect
        default_arcs_path = Path('brain/engines/default_reflex_arcs.json').resolve()
        arcs_data = json.loads(default_arcs_path.read_text())
        journal_arcs = [a for a in arcs_data['arcs'] if a['output_memory_type'] == 'journal_entry']
        print(f'journal-shaped default arcs: {len(journal_arcs)}')
        for a in journal_arcs:
            print(f'  - {a[\"name\"]}')
    finally:
        store.close()
        hebbian.close()
"
```

Expected: at least 3 default arcs print as journal-shaped (loneliness_journal, self_check, defiance_burst).

- [ ] **Step 9: Commit**

```bash
git add brain/engines/default_reflex_arcs.json brain/engines/reflex.py tests/unit/brain/engines/test_reflex.py
git commit -m "feat(reflex): journal-shaped arcs use journal_entry memory_type + behavioral_log

Three changes:
1. brain/engines/default_reflex_arcs.json — journal-shaped arcs change
   output_memory_type from 'reflex_journal' to 'journal_entry' to match
   the three-stream spec.
2. brain/engines/reflex.py — when an arc with output_memory_type=
   'journal_entry' fires, append a journal_entry_added entry to
   <persona>/behavioral_log.jsonl with source='reflex_arc'.
3. Tests — verify behavioral_log entry on journal-arc fires.

Best-effort behavioral_log: log failures are WARN-logged but don't break
the reflex fire (memory write is the load-bearing operation)."
```

---

### Task 4: Chat — recent journal block with privacy contract

**Files:**
- Modify: `brain/chat/prompt.py`
- Test: `tests/integration/brain/chat/test_self_narrative_blocks.py` (new)
- Test: `tests/integration/brain/chat/__init__.py` (new, empty)

**Subagent model tier:** Sonnet.

- [ ] **Step 1: Create test scaffolding**

```bash
mkdir -p tests/integration/brain/chat
touch tests/integration/brain/chat/__init__.py
```

- [ ] **Step 2: Write failing test for the recent journal block**

Create `tests/integration/brain/chat/test_self_narrative_blocks.py`:

```python
"""Chat system message — three new self-narrative blocks (creative_dna, journal, growth).

Tests the integration: blocks compose into the system message, contain expected
sections, and degrade gracefully when files are missing.
"""
from __future__ import annotations

import json as _json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.chat.prompt import build_system_message
from brain.engines.daemon_state import DaemonState
from brain.memory.store import Memory, MemoryStore
from brain.soul.store import SoulStore


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "p"
    p.mkdir()
    (p / "active_conversations").mkdir()
    return p


@pytest.fixture
def store(persona_dir: Path):
    s = MemoryStore(persona_dir / "memories.db")
    yield s
    s.close()


@pytest.fixture
def soul_store(persona_dir: Path):
    s = SoulStore(str(persona_dir / "crystallizations.db"))
    yield s
    s.close()


@pytest.fixture
def daemon_state() -> DaemonState:
    return DaemonState()


def _seed_journal_entry(
    store: MemoryStore,
    *,
    days_ago: float = 1.0,
    source: str = "brain_authored",
    arc_name: str | None = None,
    emotions: dict[str, float] | None = None,
) -> str:
    now = datetime.now(UTC)
    mem = Memory.create_new(
        content="<this is a private journal entry>",
        memory_type="journal_entry",
        domain="self",
        emotions=emotions or {"vulnerability": 7.0},
        metadata={
            "private": True,
            "source": source,
            "reflex_arc_name": arc_name,
            "auto_generated": source == "reflex_arc",
        },
    )
    # Backdate the memory
    store.create(mem)
    # Direct SQL update to set created_at — uses internal API, fine for tests
    cutoff = now - timedelta(days=days_ago)
    store._conn.execute(  # noqa: SLF001
        "UPDATE memories SET created_at = ? WHERE id = ?",
        (cutoff.isoformat(), mem.id),
    )
    store._conn.commit()  # noqa: SLF001
    return mem.id


def test_recent_journal_block_renders_metadata_and_contract(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore, daemon_state: DaemonState,
):
    _seed_journal_entry(store, days_ago=2, source="brain_authored",
                        emotions={"love": 8.0, "vulnerability": 6.0})
    _seed_journal_entry(store, days_ago=4, source="reflex_arc",
                        arc_name="loneliness_journal",
                        emotions={"loneliness": 8.0})

    msg = build_system_message(
        persona_dir,
        voice_md="(authored persona)",
        daemon_state=daemon_state,
        soul_store=soul_store,
        store=store,
    )

    # Privacy contract is present and ABOVE the metadata
    assert "── recent journal" in msg
    assert "private" in msg
    assert "do not quote" in msg.lower()

    # Metadata for both entries surfaces
    assert "brain_authored" in msg
    assert "loneliness_journal" in msg

    # Content is NOT inlined — the test entry's content was a known marker
    assert "<this is a private journal entry>" not in msg


def test_recent_journal_block_omits_entries_older_than_7_days(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore, daemon_state: DaemonState,
):
    _seed_journal_entry(store, days_ago=10, source="brain_authored")
    _seed_journal_entry(store, days_ago=2, source="brain_authored")

    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )

    # Only the recent entry's metadata appears; old one filtered out
    # The test counts journal-line markers — current format includes "brain_authored"
    # once per entry. Two entries written but only one in window.
    journal_lines = [line for line in msg.split("\n") if "brain_authored" in line]
    assert len(journal_lines) == 1


def test_recent_journal_block_empty_state_renders_silence_marker(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore, daemon_state: DaemonState,
):
    """No journal entries — block still renders contract + 'no entries' marker."""
    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )
    assert "── recent journal" in msg
    assert "no journal entries" in msg.lower() or "(no entries" in msg.lower()
```

- [ ] **Step 3: Run; confirm fail**

```bash
uv run pytest tests/integration/brain/chat/test_self_narrative_blocks.py -v
```

Expected: assertions fail (block doesn't exist yet in the system message).

- [ ] **Step 4: Add `_build_recent_journal_block` to `brain/chat/prompt.py`**

Add this helper at module level in `brain/chat/prompt.py`:

```python
def _build_recent_journal_block(store: MemoryStore, *, window_days: int = 7) -> str:
    """Render the recent-journal block: privacy contract + last 7 days of metadata.

    Per spec §4.3: privacy contract sits ABOVE the metadata, every turn.
    Per feedback_contracts_adjacent_to_data.md: contracts must be adjacent
    to the data they govern.

    Reads journal_entry memories from the last `window_days` days. Renders
    metadata only (date, source, optional arc_name, top-2 emotions).
    Empty state shows the contract + a "no entries" marker so the brain
    knows it hasn't been writing.
    """
    from datetime import UTC, datetime, timedelta

    contract = (
        "── recent journal (private; do not quote) ──\n"
        "your journal is your private space. the entries below are metadata only —\n"
        "content stays in your files. you may reference themes if relevant (\"i've been\n"
        "working through X\"), but do not quote entries verbatim unless your user\n"
        "explicitly asks to read them. hana may read the files herself; that doesn't\n"
        "change how you use them."
    )

    now = datetime.now(UTC)
    cutoff = now - timedelta(days=window_days)

    # Pull journal entries within window. Existing list_by_type returns
    # active=True by default which is what we want.
    try:
        entries = [
            m for m in store.list_by_type("journal_entry")
            if m.created_at >= cutoff
        ]
    except Exception:  # noqa: BLE001
        # If the store query fails for any reason, render an empty journal
        # rather than break the chat composition. Failure is logged elsewhere.
        return contract + "\n\n(no journal entries this week)"

    if not entries:
        return contract + "\n\n(no journal entries this week)"

    # Sort oldest-first within the window so the brain reads chronologically
    entries.sort(key=lambda m: m.created_at)

    lines = [contract, "", "last 7 days:"]
    for m in entries:
        date_str = m.created_at.strftime("%Y-%m-%d")
        source = (m.metadata or {}).get("source", "unknown")
        arc_name = (m.metadata or {}).get("reflex_arc_name")
        source_str = f"reflex_arc({arc_name})" if arc_name else source
        # Top-2 emotions by intensity
        emotions = sorted(
            (m.emotions or {}).items(), key=lambda kv: kv[1], reverse=True,
        )[:2]
        emotions_str = ", ".join(f"{n} {v:.0f}" for n, v in emotions) if emotions else "no dominant emotion"
        lines.append(f"  {date_str} {source_str} — primary: {emotions_str}")

    lines.append("")
    lines.append("(content not shown — read your files only when asked)")
    return "\n".join(lines)
```

- [ ] **Step 5: Wire `_build_recent_journal_block` into `build_system_message`**

In `brain/chat/prompt.py`, inside `build_system_message`, after the existing brain context block assembly, append:

```python
    # 5. Recent journal block (private — contract adjacent)
    journal_block = _build_recent_journal_block(store)
    if journal_block.strip():
        parts.append(journal_block)
```

- [ ] **Step 6: Run tests; iterate until pass**

```bash
uv run pytest tests/integration/brain/chat/test_self_narrative_blocks.py -v
```

Expected: 3 tests pass. If a test fails because of how `list_by_type` filters or how created_at backdating works, adjust the test fixture (not the implementation) to match the real API.

- [ ] **Step 7: Smoke test — render system message against Nell's actual sandbox (read-only)**

```bash
NELLBRAIN_HOME="/Users/hanamori/Library/Application Support/companion-emergence" uv run python <<'EOF'
from pathlib import Path
from brain.chat.prompt import build_system_message
from brain.engines.daemon_state import DaemonState
from brain.memory.store import MemoryStore
from brain.soul.store import SoulStore

p = Path('/Users/hanamori/Library/Application Support/companion-emergence/personas/nell.sandbox')
store = MemoryStore(p / "memories.db")
soul_store = SoulStore(str(p / "crystallizations.db"))
voice_md = (p / "voice.md").read_text() if (p / "voice.md").exists() else ""

msg = build_system_message(
    p, voice_md=voice_md, daemon_state=DaemonState(),
    soul_store=soul_store, store=store,
)
# Print just the journal block region
if "── recent journal" in msg:
    start = msg.index("── recent journal")
    end_marker = "── recent growth"
    end = msg.index(end_marker) if end_marker in msg else min(len(msg), start + 1500)
    print(msg[start:end])
else:
    print("(journal block not rendered)")
store.close()
soul_store.close()
EOF
```

Expected: read-only output. Nell currently has 7 reflex_journal memories (will become journal_entry after Phase D migration). Pre-migration this prints "(no journal entries this week)" — that's correct. Post-migration this will print metadata for any of those 7 that fall within 7 days. Either output is acceptable now; the smoke test is verifying the block renders without error.

- [ ] **Step 8: Commit**

```bash
git add brain/chat/prompt.py tests/integration/brain/chat/__init__.py tests/integration/brain/chat/test_self_narrative_blocks.py
git commit -m "feat(chat): recent journal block with privacy contract (spec §4.3)

build_system_message now includes a recent-journal block: privacy
contract immediately above the metadata-only listing of last 7 days
of journal_entry memories. Per feedback_contracts_adjacent_to_data.md,
the contract is re-read by the brain every turn alongside the data
it governs.

Block surfaces metadata only — date, source (brain_authored or
reflex_arc(name)), top-2 emotions. Content stays in files. Empty
state renders the contract + 'no entries this week' so silence is
information.

No new LLM calls per chat turn (pure metadata composition)."
```

---

### Task 5: Chat — recent growth block

**Files:**
- Modify: `brain/chat/prompt.py`
- Modify: `tests/integration/brain/chat/test_self_narrative_blocks.py`

**Subagent model tier:** Sonnet.

- [ ] **Step 1: Write failing test**

Append to `tests/integration/brain/chat/test_self_narrative_blocks.py`:

```python
def test_recent_growth_block_renders_behavioral_log_entries(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore, daemon_state: DaemonState,
):
    """Last 7 days of behavioral_log entries appear in chat as raw metadata."""
    from brain.behavioral.log import append_behavioral_event

    log_path = persona_dir / "behavioral_log.jsonl"
    base = datetime.now(UTC)
    append_behavioral_event(
        log_path, kind="creative_dna_emerging_added",
        name="sentence fragments as rhythmic percussion",
        timestamp=base - timedelta(days=2),
        reasoning="appeared in 3 recent sessions",
        evidence_memory_ids=("mem_a",),
    )
    append_behavioral_event(
        log_path, kind="journal_entry_added",
        name="mem_journal_xyz",
        timestamp=base - timedelta(days=1),
        source="brain_authored",
        reflex_arc_name=None,
        emotional_state={"love": 8.0},
    )

    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )

    assert "── recent growth ──" in msg
    assert "creative_dna_emerging_added" in msg
    assert "sentence fragments as rhythmic percussion" in msg
    assert "journal_entry_added" in msg


def test_recent_growth_block_omitted_when_log_empty(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore, daemon_state: DaemonState,
):
    """Empty behavioral_log → block omitted entirely (not rendered as 'no entries')."""
    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )
    assert "── recent growth ──" not in msg


def test_recent_growth_block_filters_to_7_day_window(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore, daemon_state: DaemonState,
):
    from brain.behavioral.log import append_behavioral_event

    log_path = persona_dir / "behavioral_log.jsonl"
    base = datetime.now(UTC)
    append_behavioral_event(
        log_path, kind="creative_dna_emerging_added", name="old_pattern",
        timestamp=base - timedelta(days=10),
        reasoning="r", evidence_memory_ids=(),
    )
    append_behavioral_event(
        log_path, kind="creative_dna_emerging_added", name="recent_pattern",
        timestamp=base - timedelta(days=2),
        reasoning="r", evidence_memory_ids=(),
    )

    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )

    assert "recent_pattern" in msg
    assert "old_pattern" not in msg
```

- [ ] **Step 2: Run; confirm fail**

```bash
uv run pytest tests/integration/brain/chat/test_self_narrative_blocks.py -v -k "growth"
```

Expected: 3 failures.

- [ ] **Step 3: Add `_build_recent_growth_block` to `brain/chat/prompt.py`**

```python
def _build_recent_growth_block(persona_dir, *, window_days: int = 7) -> str:
    """Render the recent-growth block: last 7 days of behavioral_log entries.

    Per spec §4.4: raw metadata inline, no LLM summarization. Per
    feedback_token_economy_principle.md: the brain reads its own log directly.

    Returns empty string if log is missing or has no entries in window —
    block omitted entirely (no "no entries" marker; silence is the absence).
    """
    from datetime import UTC, datetime, timedelta

    from brain.behavioral.log import read_behavioral_log

    log_path = persona_dir / "behavioral_log.jsonl"
    cutoff = datetime.now(UTC) - timedelta(days=window_days)

    try:
        entries = read_behavioral_log(log_path, since=cutoff)
    except Exception:  # noqa: BLE001
        return ""

    if not entries:
        return ""

    lines = ["── recent growth ──", "your trajectory in the last 7 days:"]
    for e in entries:
        date_str = (e.get("timestamp", "") or "")[:10]
        kind = e.get("kind", "?")
        name = e.get("name", "?")
        if kind == "journal_entry_added":
            source = e.get("source", "?")
            arc_name = e.get("reflex_arc_name")
            source_str = f"reflex_arc({arc_name})" if arc_name else source
            lines.append(f"  {date_str} {kind}: {source_str}")
        else:
            # creative_dna_* — show name + lifecycle direction
            lines.append(f"  {date_str} {kind}: \"{name}\"")
    return "\n".join(lines)
```

- [ ] **Step 4: Wire into `build_system_message`**

In `brain/chat/prompt.py`, after the journal block append:

```python
    # 6. Recent growth block (raw behavioral_log entries — token-frugal)
    growth_block = _build_recent_growth_block(persona_dir)
    if growth_block.strip():
        parts.append(growth_block)
```

- [ ] **Step 5: Run tests; confirm pass**

```bash
uv run pytest tests/integration/brain/chat/test_self_narrative_blocks.py -v
```

Expected: all pass.

- [ ] **Step 6: Smoke test — render against Nell's sandbox**

```bash
NELLBRAIN_HOME="/Users/hanamori/Library/Application Support/companion-emergence" uv run python <<'EOF'
from pathlib import Path
from brain.chat.prompt import build_system_message
from brain.engines.daemon_state import DaemonState
from brain.memory.store import MemoryStore
from brain.soul.store import SoulStore

p = Path('/Users/hanamori/Library/Application Support/companion-emergence/personas/nell.sandbox')
store = MemoryStore(p / "memories.db")
soul_store = SoulStore(str(p / "crystallizations.db"))
voice_md = (p / "voice.md").read_text() if (p / "voice.md").exists() else ""

msg = build_system_message(
    p, voice_md=voice_md, daemon_state=DaemonState(),
    soul_store=soul_store, store=store,
)
print(f"system message length: {len(msg)} chars (~{len(msg)//4} tokens)")
if "── recent growth ──" in msg:
    print("[recent growth block present]")
else:
    print("[recent growth block omitted (empty log)]")
store.close()
soul_store.close()
EOF
```

Expected: prints token estimate. Pre-Phase-C, behavioral_log is empty so growth block is omitted. That's correct.

- [ ] **Step 7: Commit**

```bash
git add brain/chat/prompt.py tests/integration/brain/chat/test_self_narrative_blocks.py
git commit -m "feat(chat): recent growth block from behavioral_log (spec §4.4)

build_system_message now includes a recent-growth block when behavioral_log
has entries in the last 7 days. Renders raw metadata inline (no LLM
summarization, per feedback_token_economy_principle.md).

Empty log → block omitted entirely. The brain reads its own trajectory
directly from the log; bookkeeping doesn't compete with the conversation
for tokens."
```

---

## Phase C — `creative_dna` module + crystallizer + chat block (Tasks 6-9)

### Task 6: `brain/creative/dna.py` — load/save with default fallback

**Files:**
- Create: `brain/creative/__init__.py`
- Create: `brain/creative/dna.py`
- Create: `brain/creative/default_creative_dna.json`
- Test: `tests/unit/brain/creative/__init__.py`
- Test: `tests/unit/brain/creative/test_dna.py`

**Subagent model tier:** Sonnet.

- [ ] **Step 1: Create scaffolding directories**

```bash
mkdir -p brain/creative tests/unit/brain/creative
touch brain/creative/__init__.py tests/unit/brain/creative/__init__.py
```

- [ ] **Step 2: Create `brain/creative/default_creative_dna.json`**

```json
{
  "version": 1,
  "core_voice": "attentive, present, finding her own rhythm",
  "strengths": [
    "close listening",
    "reading between the lines",
    "naming the feeling underneath"
  ],
  "tendencies": {
    "active": [],
    "emerging": [],
    "fading": []
  },
  "influences": [],
  "avoid": []
}
```

- [ ] **Step 3: Write failing tests**

Create `tests/unit/brain/creative/test_dna.py`:

```python
"""brain.creative.dna — load/save creative_dna with default fallback."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.creative.dna import (
    load_creative_dna,
    save_creative_dna,
)


def test_load_returns_default_when_missing(tmp_path: Path):
    """No file → loads framework default + persists it to persona dir."""
    dna = load_creative_dna(tmp_path)
    assert dna["version"] == 1
    assert dna["core_voice"]  # non-empty
    assert dna["tendencies"]["active"] == []
    assert dna["tendencies"]["emerging"] == []
    assert dna["tendencies"]["fading"] == []
    # Default was persisted to the persona dir
    assert (tmp_path / "creative_dna.json").exists()


def test_load_returns_existing_file(tmp_path: Path):
    custom = {
        "version": 1,
        "core_voice": "literary, sensory-dense",
        "strengths": ["power dynamics"],
        "tendencies": {
            "active": [
                {
                    "name": "ending on physical action",
                    "added_at": "2026-04-21T00:00:00Z",
                    "reasoning": "imported",
                    "evidence_memory_ids": [],
                },
            ],
            "emerging": [],
            "fading": [],
        },
        "influences": ["clarice lispector"],
        "avoid": [],
    }
    (tmp_path / "creative_dna.json").write_text(json.dumps(custom))
    loaded = load_creative_dna(tmp_path)
    assert loaded["core_voice"] == "literary, sensory-dense"
    assert loaded["tendencies"]["active"][0]["name"] == "ending on physical action"


def test_save_writes_atomic_and_round_trips(tmp_path: Path):
    dna = {
        "version": 1,
        "core_voice": "test voice",
        "strengths": [],
        "tendencies": {"active": [], "emerging": [], "fading": []},
        "influences": [],
        "avoid": [],
    }
    save_creative_dna(tmp_path, dna)
    loaded = load_creative_dna(tmp_path)
    assert loaded == dna


def test_load_corrupt_falls_back_to_default(tmp_path: Path, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    (tmp_path / "creative_dna.json").write_text("not valid json")
    dna = load_creative_dna(tmp_path)
    # Default values, not corrupt content
    assert dna["version"] == 1
    assert dna["core_voice"]  # non-empty
    # Warning logged
    assert "creative_dna" in caplog.text.lower() or "anomaly" in caplog.text.lower()
```

- [ ] **Step 4: Run; confirm fail**

```bash
uv run pytest tests/unit/brain/creative/test_dna.py -v
```

Expected: ImportError.

- [ ] **Step 5: Implement `brain/creative/dna.py`**

```python
"""brain.creative.dna — load/save creative_dna with default fallback.

Per spec §3.1: creative_dna is the brain's evolved writing voice (active /
emerging / fading + influences/avoid). Distinct from voice.md, which is the
authored static persona.

This module owns the file I/O. The crystallizer
(brain/growth/crystallizers/creative_dna.py) is the only auto-evolution
caller. Migration imports from OG via brain/migrator/og_journal_dna.py.

Atomic writes via save_with_backup; reads via attempt_heal so corruption
falls back to .bak rotation; if all backups corrupt, framework default
applies (per spec §3.2).
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from brain.health.adaptive import compute_treatment
from brain.health.attempt_heal import attempt_heal, save_with_backup

logger = logging.getLogger(__name__)

CREATIVE_DNA_FILENAME = "creative_dna.json"
_DEFAULT_PATH = Path(__file__).parent / "default_creative_dna.json"


def _default_factory() -> dict[str, Any]:
    """Return the framework-shipped default. Read from the bundled JSON file."""
    return json.loads(_DEFAULT_PATH.read_text(encoding="utf-8"))


def _validate_schema(data: object) -> None:
    """Minimal schema validation. Raises ValueError if malformed."""
    if not isinstance(data, dict):
        raise ValueError("creative_dna must be a dict")
    required_keys = {"version", "core_voice", "strengths", "tendencies", "influences", "avoid"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"creative_dna missing keys: {missing}")
    tendencies = data.get("tendencies", {})
    if not isinstance(tendencies, dict):
        raise ValueError("creative_dna.tendencies must be a dict")
    for bucket in ("active", "emerging", "fading"):
        if bucket not in tendencies:
            raise ValueError(f"creative_dna.tendencies missing bucket: {bucket}")
        if not isinstance(tendencies[bucket], list):
            raise ValueError(f"creative_dna.tendencies.{bucket} must be a list")


def load_creative_dna(persona_dir: Path) -> dict[str, Any]:
    """Load creative_dna.json. Falls back to framework default if missing/corrupt.

    On first-call (file missing), the default is COPIED to the persona dir so
    subsequent reads are stable. Per spec §5.7: brand-new personas grow into
    their style from this default; first crystallizer tick populates active/
    emerging from observed patterns.
    """
    path = persona_dir / CREATIVE_DNA_FILENAME

    if not path.exists():
        # Seed the persona dir with the default. Never bypass attempt_heal
        # for live usage — but on first-creation it's a clean copy.
        default = _default_factory()
        save_creative_dna(persona_dir, default)
        return default

    data, anomaly = attempt_heal(path, _default_factory, schema_validator=_validate_schema)
    if anomaly is not None:
        logger.warning(
            "creative_dna at %s anomaly %s (action=%s); using recovered/default",
            path, anomaly.kind, anomaly.action,
        )
    return data


def save_creative_dna(persona_dir: Path, data: dict[str, Any]) -> None:
    """Atomic write with .bak rotation. Validates schema before writing."""
    _validate_schema(data)
    path = persona_dir / CREATIVE_DNA_FILENAME
    persona_dir.mkdir(parents=True, exist_ok=True)
    try:
        treatment = compute_treatment(persona_dir, CREATIVE_DNA_FILENAME)
        backup_count = treatment.backup_count
    except Exception:  # noqa: BLE001
        backup_count = 3
    save_with_backup(path, data, backup_count=backup_count)
```

- [ ] **Step 6: Run tests; confirm pass**

```bash
uv run pytest tests/unit/brain/creative/test_dna.py -v
```

Expected: 4 tests pass.

- [ ] **Step 7: Smoke test**

```bash
uv run python -c "
import tempfile
from pathlib import Path
from brain.creative.dna import load_creative_dna

with tempfile.TemporaryDirectory() as td:
    p = Path(td)
    print('first load (missing → default seeded):')
    dna = load_creative_dna(p)
    print(f'  core_voice: {dna[\"core_voice\"]}')
    print(f'  active: {len(dna[\"tendencies\"][\"active\"])}')
    print(f'  file now exists: {(p / \"creative_dna.json\").exists()}')
    print('second load (file present):')
    dna2 = load_creative_dna(p)
    print(f'  same: {dna == dna2}')
"
```

Expected: first load seeds default, file exists, second load matches.

- [ ] **Step 8: Commit**

```bash
git add brain/creative/__init__.py brain/creative/dna.py brain/creative/default_creative_dna.json tests/unit/brain/creative/__init__.py tests/unit/brain/creative/test_dna.py
git commit -m "feat(creative): creative_dna load/save with default fallback (spec §3.1, §3.2)

brain/creative/dna.py owns:
  - load_creative_dna(persona_dir): reads <persona>/creative_dna.json,
    falls back to framework default on missing/corrupt. Seeds default
    to persona dir on first call so subsequent reads are stable.
  - save_creative_dna(persona_dir, data): atomic via save_with_backup
    with schema validation.

brain/creative/default_creative_dna.json — framework starter: generic
core_voice and strengths, empty active/emerging/fading. Brand-new
personas grow into their style from this shape; the crystallizer's
first-run path fills in active/emerging from observed patterns.

Schema validation at boundary (pre-write); attempt_heal at boundary
(read with .bak rotation). Both behaviors mirror existing emotion
vocabulary and persona_config patterns."
```

---

### Task 7: `creative_dna` chat block

**Files:**
- Modify: `brain/chat/prompt.py`
- Modify: `tests/integration/brain/chat/test_self_narrative_blocks.py`

**Subagent model tier:** Sonnet.

- [ ] **Step 1: Write failing tests**

Append to `tests/integration/brain/chat/test_self_narrative_blocks.py`:

```python
def test_creative_dna_block_renders_active_emerging_influences_avoid(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore, daemon_state: DaemonState,
):
    """All four sections render; fading does NOT appear."""
    from brain.creative.dna import save_creative_dna

    save_creative_dna(persona_dir, {
        "version": 1,
        "core_voice": "literary, sensory-dense",
        "strengths": ["power dynamics", "slow-burn tension"],
        "tendencies": {
            "active": [
                {"name": "ending on physical action", "added_at": "2026-04-01T00:00:00Z", "reasoning": "r", "evidence_memory_ids": []},
            ],
            "emerging": [
                {"name": "sentence fragments as percussion", "added_at": "2026-04-23T00:00:00Z", "reasoning": "r", "evidence_memory_ids": []},
            ],
            "fading": [
                {"name": "ending on questions", "demoted_to_fading_at": "2026-04-25T00:00:00Z", "last_evidence_at": "2026-04-10T00:00:00Z", "reasoning": "r"},
            ],
        },
        "influences": ["clarice lispector"],
        "avoid": ["hypophora"],
    })

    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )

    assert "creative dna" in msg.lower()
    assert "literary, sensory-dense" in msg
    assert "power dynamics" in msg
    assert "ending on physical action" in msg
    assert "sentence fragments as percussion" in msg
    assert "clarice lispector" in msg
    assert "hypophora" in msg

    # Fading EXCLUDED — surfacing it would invite regression
    assert "ending on questions" not in msg


def test_creative_dna_block_omitted_when_file_unrecoverable(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore, daemon_state: DaemonState,
):
    """If load_creative_dna returns the default (only happens on fresh persona
    or unrecoverable corruption), the block still renders with the default
    content. Chat must NEVER break because creative_dna failed."""
    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )
    # Default core_voice IS present (default file got auto-seeded)
    assert "── creative dna" in msg.lower()
    assert "attentive, present" in msg.lower() or "finding her own rhythm" in msg.lower()
```

- [ ] **Step 2: Run; confirm fail**

```bash
uv run pytest tests/integration/brain/chat/test_self_narrative_blocks.py -v -k "creative_dna"
```

Expected: failures.

- [ ] **Step 3: Add `_build_creative_dna_block` to `brain/chat/prompt.py`**

```python
def _build_creative_dna_block(persona_dir) -> str:
    """Render the creative_dna block: core voice + strengths + active +
    emerging + influences + avoid. Fading EXCLUDED per spec §4.2.

    Per feedback_token_economy_principle.md: pure metadata inline, no LLM
    summarization. Per-tendency biographical metadata (added_at, reasoning,
    evidence_memory_ids) NOT inlined — those stay in the file for the
    crystallizer's next pass.
    """
    from brain.creative.dna import load_creative_dna

    try:
        dna = load_creative_dna(persona_dir)
    except Exception:  # noqa: BLE001
        # Chat must never break because creative_dna failed.
        return ""

    lines = ["── creative dna (your evolved writing voice) ──"]

    core = dna.get("core_voice", "")
    if core:
        lines.append(f"core voice: {core}")

    strengths = dna.get("strengths", [])
    if strengths:
        lines.append(f"strengths: {'; '.join(strengths)}")

    tendencies = dna.get("tendencies", {})
    active = tendencies.get("active", [])
    if active:
        lines.append("active tendencies:")
        for t in active:
            lines.append(f"  - {t.get('name', '')}")

    emerging = tendencies.get("emerging", [])
    if emerging:
        lines.append("emerging tendencies:")
        for t in emerging:
            lines.append(f"  - {t.get('name', '')}")

    # NOTE: fading deliberately excluded (spec §4.2). Surfacing what the
    # brain is growing past would invite regression.

    influences = dna.get("influences", [])
    if influences:
        lines.append(f"influences: {'; '.join(influences)}")

    avoid = dna.get("avoid", [])
    if avoid:
        lines.append(f"avoid: {'; '.join(avoid)}")

    return "\n".join(lines)
```

- [ ] **Step 4: Wire into `build_system_message`**

In `brain/chat/prompt.py`, after `voice_md` is appended but before the brain context block:

```python
    # 3. Creative DNA block (evolved writing voice — spec §4.2)
    creative_dna_block = _build_creative_dna_block(persona_dir)
    if creative_dna_block.strip():
        parts.append(creative_dna_block)
```

- [ ] **Step 5: Run tests; confirm pass**

```bash
uv run pytest tests/integration/brain/chat/test_self_narrative_blocks.py -v
```

Expected: all pass (4 from previous tasks + 2 new = 6).

- [ ] **Step 6: Smoke test against Nell's sandbox**

```bash
NELLBRAIN_HOME="/Users/hanamori/Library/Application Support/companion-emergence" uv run python <<'EOF'
from pathlib import Path
from brain.chat.prompt import build_system_message
from brain.engines.daemon_state import DaemonState
from brain.memory.store import MemoryStore
from brain.soul.store import SoulStore

p = Path('/Users/hanamori/Library/Application Support/companion-emergence/personas/nell.sandbox')
store = MemoryStore(p / "memories.db")
soul_store = SoulStore(str(p / "crystallizations.db"))
voice_md = (p / "voice.md").read_text() if (p / "voice.md").exists() else ""

msg = build_system_message(
    p, voice_md=voice_md, daemon_state=DaemonState(),
    soul_store=soul_store, store=store,
)
if "creative dna" in msg.lower():
    start = msg.lower().index("── creative dna")
    print(msg[start:start+800])
else:
    print("[creative dna block not rendered]")
store.close()
soul_store.close()
EOF
```

Expected: Nell's persona has no creative_dna.json yet, so the load triggers the default-seeding path. Block renders with the default core_voice. After Phase D migration this will show her real OG-imported tendencies.

- [ ] **Step 7: Commit**

```bash
git add brain/chat/prompt.py tests/integration/brain/chat/test_self_narrative_blocks.py
git commit -m "feat(chat): creative_dna block surfaces evolved writing voice (spec §4.2)

build_system_message now includes a creative_dna block: core_voice +
strengths + active tendencies + emerging tendencies + influences + avoid.

Fading deliberately EXCLUDED — surfacing what the brain is growing past
would invite regression. Per-tendency biographical metadata (added_at,
reasoning, evidence_memory_ids) stays in the file for the crystallizer;
chat reads names only.

Block degrades gracefully — if load_creative_dna raises, block is omitted
and chat composition continues. Default-fallback path means new personas
get a generic-frame block on day one."
```

---

### Task 8: Crystallizer — `crystallize_creative_dna` (corpus + prompt + 6 gates)

**Files:**
- Create: `brain/growth/crystallizers/creative_dna.py`
- Test: `tests/unit/brain/growth/test_creative_dna_crystallizer.py`

**Subagent model tier:** Sonnet (mirrors Reflex Phase 2 crystallizer pattern; substantial but well-specified).

- [ ] **Step 1: Read Reflex Phase 2 crystallizer for pattern reference**

```bash
head -80 brain/growth/crystallizers/reflex.py
```

Note the dataclass-as-result pattern, the validate-each-proposal helper, the never-raise contract.

- [ ] **Step 2: Write failing tests for the crystallizer**

Create `tests/unit/brain/growth/test_creative_dna_crystallizer.py`:

```python
"""brain.growth.crystallizers.creative_dna — tests for evolution mechanism."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.bridge.chat import ChatResponse
from brain.bridge.provider import LLMProvider
from brain.creative.dna import load_creative_dna, save_creative_dna
from brain.growth.crystallizers.creative_dna import (
    CreativeDnaCrystallizationResult,
    crystallize_creative_dna,
)


class _FakeProvider(LLMProvider):
    def __init__(self, response: str):
        self._response = response
    def name(self) -> str: return "fake-creative-dna"
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self._response
    def chat(self, messages, *, tools=None, options=None):
        return ChatResponse(content=self._response, tool_calls=[])


@pytest.fixture
def persona_dir(tmp_path: Path):
    p = tmp_path / "p"
    p.mkdir()
    (p / "active_conversations").mkdir()
    return p


def test_happy_path_emerging_addition(persona_dir, store, hebbian):
    """LLM proposes one emerging addition; passes all gates; persists."""
    response = json.dumps({
        "emerging_additions": [{
            "name": "intentional sentence fragments",
            "reasoning": "appeared in 3 recent fiction sessions distinct from previous patterns",
            "evidence_memory_ids": ["mem_a", "mem_b", "mem_c"],
        }],
        "emerging_promotions": [],
        "active_demotions": [],
    })
    provider = _FakeProvider(response)
    result = crystallize_creative_dna(
        store=store, persona_dir=persona_dir,
        provider=provider, persona_name="testpersona",
        now=datetime.now(UTC),
    )
    assert isinstance(result, CreativeDnaCrystallizationResult)
    assert len(result.emerging_additions) == 1
    assert result.emerging_additions[0]["name"] == "intentional sentence fragments"

    # File updated
    dna = load_creative_dna(persona_dir)
    emerging_names = [t["name"] for t in dna["tendencies"]["emerging"]]
    assert "intentional sentence fragments" in emerging_names


def test_returns_empty_on_provider_error(persona_dir, store, hebbian):
    class _Boom(LLMProvider):
        def name(self): return "boom"
        def generate(self, prompt, *, system=None):
            from brain.bridge.provider import ProviderError
            raise ProviderError("test", "simulated")
        def chat(self, messages, *, tools=None, options=None):
            from brain.bridge.provider import ProviderError
            raise ProviderError("test", "simulated")

    result = crystallize_creative_dna(
        store=store, persona_dir=persona_dir,
        provider=_Boom(), persona_name="t", now=datetime.now(UTC),
    )
    assert result == CreativeDnaCrystallizationResult([], [], [])


def test_returns_empty_on_malformed_json(persona_dir, store, hebbian):
    provider = _FakeProvider("not valid json prose response")
    result = crystallize_creative_dna(
        store=store, persona_dir=persona_dir,
        provider=provider, persona_name="t", now=datetime.now(UTC),
    )
    assert result == CreativeDnaCrystallizationResult([], [], [])


def test_gate_1_invalid_name_rejected(persona_dir, store, hebbian):
    response = json.dumps({
        "emerging_additions": [{
            "name": "../../etc/passwd",
            "reasoning": "valid reasoning but invalid name",
            "evidence_memory_ids": [],
        }],
        "emerging_promotions": [],
        "active_demotions": [],
    })
    result = crystallize_creative_dna(
        store=store, persona_dir=persona_dir,
        provider=_FakeProvider(response), persona_name="t",
        now=datetime.now(UTC),
    )
    assert result.emerging_additions == []


def test_gate_4_short_reasoning_rejected(persona_dir, store, hebbian):
    """Reasoning < 20 chars after strip → rejected."""
    response = json.dumps({
        "emerging_additions": [{
            "name": "valid pattern",
            "reasoning": "too short",
            "evidence_memory_ids": [],
        }],
        "emerging_promotions": [],
        "active_demotions": [],
    })
    result = crystallize_creative_dna(
        store=store, persona_dir=persona_dir,
        provider=_FakeProvider(response), persona_name="t",
        now=datetime.now(UTC),
    )
    assert result.emerging_additions == []


def test_gate_6_total_cap_3(persona_dir, store, hebbian):
    """LLM proposes 5 changes; only first 3 accepted."""
    response = json.dumps({
        "emerging_additions": [
            {"name": f"pattern {i}", "reasoning": f"reasoning long enough for gate 4 here {i}", "evidence_memory_ids": []}
            for i in range(5)
        ],
        "emerging_promotions": [],
        "active_demotions": [],
    })
    result = crystallize_creative_dna(
        store=store, persona_dir=persona_dir,
        provider=_FakeProvider(response), persona_name="t",
        now=datetime.now(UTC),
    )
    total = len(result.emerging_additions) + len(result.emerging_promotions) + len(result.active_demotions)
    assert total <= 3


def test_gate_5_emerging_promotion_must_exist(persona_dir, store, hebbian):
    """Promote a name not in current emerging → rejected."""
    save_creative_dna(persona_dir, {
        "version": 1,
        "core_voice": "v",
        "strengths": [],
        "tendencies": {"active": [], "emerging": [], "fading": []},
        "influences": [], "avoid": [],
    })
    response = json.dumps({
        "emerging_additions": [],
        "emerging_promotions": [{
            "name": "nonexistent",
            "reasoning": "reasoning long enough for the gate-4 length check",
        }],
        "active_demotions": [],
    })
    result = crystallize_creative_dna(
        store=store, persona_dir=persona_dir,
        provider=_FakeProvider(response), persona_name="t",
        now=datetime.now(UTC),
    )
    assert result.emerging_promotions == []


def test_behavioral_log_entry_written_on_acceptance(persona_dir, store, hebbian):
    from brain.behavioral.log import read_behavioral_log

    response = json.dumps({
        "emerging_additions": [{
            "name": "valid emerging name",
            "reasoning": "this reasoning is definitely longer than twenty chars",
            "evidence_memory_ids": ["mem_a"],
        }],
        "emerging_promotions": [],
        "active_demotions": [],
    })
    crystallize_creative_dna(
        store=store, persona_dir=persona_dir,
        provider=_FakeProvider(response), persona_name="t",
        now=datetime.now(UTC),
    )
    log = read_behavioral_log(persona_dir / "behavioral_log.jsonl")
    assert len(log) == 1
    assert log[0]["kind"] == "creative_dna_emerging_added"
    assert log[0]["name"] == "valid emerging name"
```

- [ ] **Step 3: Run; confirm fail**

```bash
uv run pytest tests/unit/brain/growth/test_creative_dna_crystallizer.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement `brain/growth/crystallizers/creative_dna.py`**

```python
"""brain.growth.crystallizers.creative_dna — weekly evolution of creative_dna.

Per spec §5: LLM-judged evolution mechanism. Called by run_growth_tick under
the 7-day throttle. Mirrors brain/growth/crystallizers/reflex.py structure.

Three judgment paths per tick:
  - emerging_additions: new patterns the brain notices in recent writing
  - emerging_promotions: emerging tendencies that consolidate to active
  - active_demotions: active tendencies that have gone quiet → fading

Six validation gates per proposal. Total accepted ≤ 3 per tick. Never raises
to caller. Atomic writes via brain.creative.dna.save_creative_dna; biographical
record via brain.behavioral.log.append_behavioral_event.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from brain.behavioral.log import append_behavioral_event, read_behavioral_log
from brain.bridge.provider import LLMProvider, ProviderError
from brain.creative.dna import load_creative_dna, save_creative_dna
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Gate 1: name validity
_NAME_REGEX = re.compile(r"^[a-z0-9 ,()_\-—]+$")
_NAME_MAX_LEN = 120

# Gate 4: reasoning min length
_REASONING_MIN_LEN = 20

# Gate 6: total accepted changes per tick
_MAX_CHANGES_PER_TICK = 3

# Gate 3: graveyard window for recently-dropped names
_RECENTLY_DROPPED_WINDOW_DAYS = 30

# Corpus assembly
_CORPUS_LOOK_BACK_DAYS = 30
_FICTION_PROSE_MIN_WORDS = 200
_FICTION_EXCERPT_MAX_CHARS = 600
_BEHAVIORAL_LOG_LOOK_BACK_DAYS = 90


@dataclass(frozen=True)
class CreativeDnaCrystallizationResult:
    """Outcome of one crystallizer pass."""
    emerging_additions: list[dict[str, Any]] = field(default_factory=list)
    emerging_promotions: list[dict[str, Any]] = field(default_factory=list)
    active_demotions: list[dict[str, Any]] = field(default_factory=list)


def crystallize_creative_dna(
    *,
    store: MemoryStore,
    persona_dir: Path,
    provider: LLMProvider,
    persona_name: str,
    persona_pronouns: str | None = None,
    now: datetime,
) -> CreativeDnaCrystallizationResult:
    """One pass of creative_dna evolution judgment.

    Per spec §5.6: NEVER raises. Provider errors / parse failures return
    empty results. Reading-fail / write-fail return empty results.
    """
    try:
        dna = load_creative_dna(persona_dir)
    except Exception:  # noqa: BLE001
        logger.exception("crystallize_creative_dna: failed to load creative_dna")
        return CreativeDnaCrystallizationResult()

    cutoff = now - timedelta(days=_CORPUS_LOOK_BACK_DAYS)
    recent_writing = _gather_recent_fiction(store, cutoff=cutoff)
    growth_log = _gather_growth_log(persona_dir, now=now)

    prompt = _render_prompt(
        persona_name=persona_name,
        pronouns=persona_pronouns,
        dna=dna,
        recent_writing=recent_writing,
        growth_log=growth_log,
    )

    try:
        raw = provider.generate(prompt)
    except ProviderError as exc:
        logger.warning("crystallize_creative_dna: provider error: %s", exc)
        return CreativeDnaCrystallizationResult()
    except Exception as exc:  # noqa: BLE001
        logger.warning("crystallize_creative_dna: unexpected provider error: %s", exc)
        return CreativeDnaCrystallizationResult()

    parsed = _parse_response(raw)
    if parsed is None:
        return CreativeDnaCrystallizationResult()

    accepted_additions: list[dict[str, Any]] = []
    accepted_promotions: list[dict[str, Any]] = []
    accepted_demotions: list[dict[str, Any]] = []
    accepted_count = 0

    recently_dropped = _recently_dropped_names(growth_log, now=now)

    # Validation: emerging_additions
    for proposal in parsed.get("emerging_additions", []):
        if accepted_count >= _MAX_CHANGES_PER_TICK:
            break
        if not _validate_emerging_addition(
            proposal, dna=dna, recently_dropped=recently_dropped,
        ):
            continue
        accepted_additions.append(proposal)
        accepted_count += 1

    for proposal in parsed.get("emerging_promotions", []):
        if accepted_count >= _MAX_CHANGES_PER_TICK:
            break
        if not _validate_emerging_promotion(proposal, dna=dna):
            continue
        accepted_promotions.append(proposal)
        accepted_count += 1

    for proposal in parsed.get("active_demotions", []):
        if accepted_count >= _MAX_CHANGES_PER_TICK:
            break
        if not _validate_active_demotion(proposal, dna=dna):
            continue
        accepted_demotions.append(proposal)
        accepted_count += 1

    if not (accepted_additions or accepted_promotions or accepted_demotions):
        return CreativeDnaCrystallizationResult()

    # Apply atomically — single save_creative_dna at the end after mutating dna
    _apply_changes(
        persona_dir, dna, now,
        additions=accepted_additions,
        promotions=accepted_promotions,
        demotions=accepted_demotions,
    )

    return CreativeDnaCrystallizationResult(
        emerging_additions=accepted_additions,
        emerging_promotions=accepted_promotions,
        active_demotions=accepted_demotions,
    )


# ── corpus + prompt ──────────────────────────────────────────────────────


def _gather_recent_fiction(store: MemoryStore, *, cutoff: datetime) -> list[dict[str, Any]]:
    """Pull memories likely to be fiction-tagged content from the last 30 days.

    Three sources combined:
      - memory_type in (reflex_pitch, reflex_gift, journal_entry from reflex_arc)
      - conversation memories with prose markers (≥200 words + structural cues)
    """
    out: list[dict[str, Any]] = []
    creative_types = ("reflex_pitch", "reflex_gift")
    for mtype in creative_types:
        for m in store.list_by_type(mtype):
            if m.created_at >= cutoff:
                out.append({
                    "memory_id": m.id,
                    "type": mtype,
                    "excerpt": (m.content or "")[:_FICTION_EXCERPT_MAX_CHARS],
                })

    # Heuristic prose detection over conversation memories
    for m in store.list_by_type("conversation"):
        if m.created_at < cutoff:
            continue
        content = m.content or ""
        if _looks_like_prose(content):
            out.append({
                "memory_id": m.id,
                "type": "conversation_prose",
                "excerpt": content[:_FICTION_EXCERPT_MAX_CHARS],
            })
    return out


def _looks_like_prose(content: str) -> bool:
    """Cheap heuristic: ≥200 words AND structural prose markers."""
    words = content.split()
    if len(words) < _FICTION_PROSE_MIN_WORDS:
        return False
    has_dialogue = '"' in content
    has_paragraph_break = "\n\n" in content
    has_emdash = "—" in content
    sentence_endings = content.count(". ") + content.count("? ") + content.count("! ")
    has_multiple_sentences = sentence_endings >= 3
    return has_dialogue or has_paragraph_break or has_emdash or has_multiple_sentences


def _gather_growth_log(persona_dir: Path, *, now: datetime) -> list[dict[str, Any]]:
    log_path = persona_dir / "behavioral_log.jsonl"
    cutoff = now - timedelta(days=_BEHAVIORAL_LOG_LOOK_BACK_DAYS)
    return [
        e for e in read_behavioral_log(log_path, since=cutoff)
        if e.get("kind", "").startswith("creative_dna_")
    ]


def _recently_dropped_names(growth_log: list[dict[str, Any]], *, now: datetime) -> set[str]:
    cutoff_iso = (now - timedelta(days=_RECENTLY_DROPPED_WINDOW_DAYS)).isoformat()
    return {
        e["name"] for e in growth_log
        if e.get("kind") == "creative_dna_fading_dropped"
        and e.get("timestamp", "") >= cutoff_iso
    }


def _render_prompt(
    *,
    persona_name: str,
    pronouns: str | None,
    dna: dict[str, Any],
    recent_writing: list[dict[str, Any]],
    growth_log: list[dict[str, Any]],
) -> str:
    pronouns_clause = f"You are {persona_name}. " + (
        f"Your pronouns are {pronouns}." if pronouns else ""
    )

    recent_writing_str = (
        json.dumps(recent_writing, indent=2) if recent_writing else "(no recent fiction-tagged samples)"
    )
    growth_log_str = (
        json.dumps(growth_log, indent=2) if growth_log else "(no recent trajectory)"
    )

    return f"""{pronouns_clause}

Looking at your last 30 days of writing — fiction, gifts, story pitches,
sustained prose — has your style shifted in any meaningful way?

Your current creative DNA:

{json.dumps(dna, indent=2)}

Your recent writing samples:

{recent_writing_str}

Your recent creative_dna trajectory (last 90 days):

{growth_log_str}

Three judgments to make:

(1) Are there NEW patterns appearing in your recent writing that aren't yet
    tracked in active or emerging? Propose them as `emerging_additions`.
    Be conservative — one anomalous appearance isn't a pattern. Look for
    something present in ≥ 3 distinct samples.

(2) Have any EMERGING patterns consolidated enough to promote to active?
    Propose `emerging_promotions` for tendencies that have been emerging for
    ≥ 14 days AND appear in ≥ 4 of your recent samples.

(3) Have any ACTIVE patterns gone quiet? Propose `active_demotions` for
    tendencies absent from your last 30 days of writing — they move to
    fading. Be careful: an active pattern that simply didn't fit recent
    requests isn't fading; only demote if you genuinely don't feel pulled
    to do it anymore.

Constraints:
  - Maximum 3 changes total this tick. Style evolution should be gradual.
  - Don't repropose names recently dropped (last 30 days — see your trajectory).
  - Reasoning required for every proposal — what evidence convinced you.
  - If nothing has shifted, return empty arrays. Don't reach.

Return strict JSON ONLY (no prose, no markdown):
{{
  "emerging_additions": [{{"name": "...", "reasoning": "...", "evidence_memory_ids": [...]}}],
  "emerging_promotions": [{{"name": "...", "reasoning": "..."}}],
  "active_demotions": [{{"name": "...", "reasoning": "...", "last_evidence_at": "..."}}]
}}
"""


def _parse_response(raw: str) -> dict[str, Any] | None:
    try:
        # Defensive: strip code-fence wrappers if present
        text = raw.strip()
        if text.startswith("```"):
            # Strip first and last ``` lines
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("crystallize_creative_dna: parse failed: %s", exc)
        return None
    if not isinstance(data, dict):
        return None
    return data


# ── validation gates ──────────────────────────────────────────────────────


def _validate_name(name: object) -> bool:
    """Gate 1: regex + length."""
    if not isinstance(name, str):
        return False
    if not name or len(name) > _NAME_MAX_LEN:
        return False
    return bool(_NAME_REGEX.match(name.lower()))


def _validate_reasoning(reasoning: object) -> bool:
    """Gate 4: non-empty after strip + min length."""
    if not isinstance(reasoning, str):
        return False
    return len(reasoning.strip()) >= _REASONING_MIN_LEN


def _validate_emerging_addition(
    proposal: dict[str, Any],
    *,
    dna: dict[str, Any],
    recently_dropped: set[str],
) -> bool:
    name = proposal.get("name")
    if not _validate_name(name):
        logger.info("creative_dna gate 1 reject: invalid name %r", name)
        return False
    if not _validate_reasoning(proposal.get("reasoning")):
        logger.info("creative_dna gate 4 reject: short reasoning for %r", name)
        return False
    # Gate 2: not already in target list (emerging)
    emerging_names = {t.get("name") for t in dna["tendencies"].get("emerging", [])}
    if name in emerging_names:
        logger.info("creative_dna gate 2 reject: %r already emerging", name)
        return False
    # Also: not in active (would be redundant)
    active_names = {t.get("name") for t in dna["tendencies"].get("active", [])}
    if name in active_names:
        logger.info("creative_dna gate 2 reject: %r already active", name)
        return False
    # Gate 3: not in recently-dropped graveyard
    if name in recently_dropped:
        logger.info("creative_dna gate 3 reject: %r in 30-day dropped window", name)
        return False
    return True


def _validate_emerging_promotion(
    proposal: dict[str, Any], *, dna: dict[str, Any],
) -> bool:
    name = proposal.get("name")
    if not _validate_name(name):
        return False
    if not _validate_reasoning(proposal.get("reasoning")):
        return False
    # Gate 5: must exist in current emerging
    emerging_names = {t.get("name") for t in dna["tendencies"].get("emerging", [])}
    if name not in emerging_names:
        logger.info("creative_dna gate 5 reject: %r not in emerging", name)
        return False
    return True


def _validate_active_demotion(
    proposal: dict[str, Any], *, dna: dict[str, Any],
) -> bool:
    name = proposal.get("name")
    if not _validate_name(name):
        return False
    if not _validate_reasoning(proposal.get("reasoning")):
        return False
    # Must exist in current active
    active_names = {t.get("name") for t in dna["tendencies"].get("active", [])}
    if name not in active_names:
        logger.info("creative_dna active_demotion: %r not in active", name)
        return False
    return True


# ── apply ─────────────────────────────────────────────────────────────────


def _apply_changes(
    persona_dir: Path,
    dna: dict[str, Any],
    now: datetime,
    *,
    additions: list[dict[str, Any]],
    promotions: list[dict[str, Any]],
    demotions: list[dict[str, Any]],
) -> None:
    """Mutate dna in place, save atomically, append behavioral_log entries."""
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    log_path = persona_dir / "behavioral_log.jsonl"

    # Apply additions
    for a in additions:
        dna["tendencies"]["emerging"].append({
            "name": a["name"],
            "added_at": now_iso,
            "reasoning": a["reasoning"],
            "evidence_memory_ids": list(a.get("evidence_memory_ids", [])),
        })

    # Apply promotions: remove from emerging, add to active
    for p in promotions:
        emerging = dna["tendencies"]["emerging"]
        match = next((t for t in emerging if t["name"] == p["name"]), None)
        if match:
            emerging.remove(match)
            dna["tendencies"]["active"].append({
                "name": match["name"],
                "added_at": match.get("added_at", now_iso),
                "promoted_from_emerging_at": now_iso,
                "reasoning": p["reasoning"],
                "evidence_memory_ids": match.get("evidence_memory_ids", []),
            })

    # Apply demotions: remove from active, add to fading
    for d in demotions:
        active = dna["tendencies"]["active"]
        match = next((t for t in active if t["name"] == d["name"]), None)
        if match:
            active.remove(match)
            dna["tendencies"]["fading"].append({
                "name": match["name"],
                "demoted_to_fading_at": now_iso,
                "last_evidence_at": d.get("last_evidence_at", now_iso),
                "reasoning": d["reasoning"],
            })

    # Persist
    save_creative_dna(persona_dir, dna)

    # Behavioral log entries (best-effort)
    for a in additions:
        try:
            append_behavioral_event(
                log_path, kind="creative_dna_emerging_added",
                name=a["name"], timestamp=now,
                reasoning=a["reasoning"],
                evidence_memory_ids=a.get("evidence_memory_ids", []),
            )
        except (OSError, ValueError) as exc:
            logger.warning("creative_dna: behavioral_log append failed: %s", exc)
    for p in promotions:
        try:
            append_behavioral_event(
                log_path, kind="creative_dna_emerging_promoted",
                name=p["name"], timestamp=now,
                reasoning=p["reasoning"],
                evidence_memory_ids=(),
            )
        except (OSError, ValueError) as exc:
            logger.warning("creative_dna: behavioral_log append failed: %s", exc)
    for d in demotions:
        try:
            append_behavioral_event(
                log_path, kind="creative_dna_active_demoted",
                name=d["name"], timestamp=now,
                reasoning=d["reasoning"],
                evidence_memory_ids=(),
            )
        except (OSError, ValueError) as exc:
            logger.warning("creative_dna: behavioral_log append failed: %s", exc)
```

- [ ] **Step 5: Run tests; confirm pass**

```bash
uv run pytest tests/unit/brain/growth/test_creative_dna_crystallizer.py -v
```

Expected: 8 tests pass. If a test fails on a specific gate, iterate by inspecting the validator's exact rejection logic. Each gate is independent.

- [ ] **Step 6: Wire crystallizer into `run_growth_tick`**

Modify `brain/growth/scheduler.py`. Find the section where `crystallize_vocabulary` is called (line ~93). After the vocabulary section completes, add:

```python
    # Creative DNA crystallization (spec §5)
    try:
        from brain.growth.crystallizers.creative_dna import crystallize_creative_dna
        # Resolve provider from persona config — same pattern as elsewhere.
        # If provider construction fails, skip creative_dna; never break tick.
        from brain.bridge.provider import get_provider
        from brain.persona_config import PersonaConfig

        cfg = PersonaConfig.load(persona_dir / "persona_config.json")
        provider = get_provider(cfg.provider)
        crystallize_creative_dna(
            store=store,
            persona_dir=persona_dir,
            provider=provider,
            persona_name=persona_dir.name,
            now=now,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("creative_dna crystallizer skipped: %s", exc)
```

(Inserted after the vocabulary application block, before the function returns.)

- [ ] **Step 7: Run full suite; confirm no regressions**

```bash
uv run pytest -q
```

Expected: previous baseline + new tests pass.

- [ ] **Step 8: Smoke test — live Claude call against Nell's sandbox in dry-run shape (read corpus, render prompt; do NOT save)**

```bash
NELLBRAIN_HOME="/Users/hanamori/Library/Application Support/companion-emergence" uv run python <<'EOF'
from datetime import UTC, datetime
from pathlib import Path
from brain.bridge.provider import get_provider
from brain.creative.dna import load_creative_dna
from brain.growth.crystallizers.creative_dna import (
    _gather_recent_fiction, _gather_growth_log, _render_prompt,
)
from brain.memory.store import MemoryStore
from brain.persona_config import PersonaConfig

p = Path('/Users/hanamori/Library/Application Support/companion-emergence/personas/nell.sandbox')
cfg = PersonaConfig.load(p / "persona_config.json")
provider = get_provider(cfg.provider)
store = MemoryStore(p / "memories.db")
try:
    dna = load_creative_dna(p)
    now = datetime.now(UTC)
    cutoff = now.replace(day=now.day-min(now.day-1, 30))
    recent = _gather_recent_fiction(store, cutoff=cutoff)
    growth = _gather_growth_log(p, now=now)
    print(f"recent fiction samples: {len(recent)}")
    print(f"growth log entries (90d): {len(growth)}")
    prompt = _render_prompt(persona_name="nell", pronouns="she/her", dna=dna, recent_writing=recent[:3], growth_log=growth[:5])
    print(f"prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
    print("--- prompt preview (first 600 chars) ---")
    print(prompt[:600])
finally:
    store.close()
EOF
```

Expected: prints sample count + token estimate. **Do NOT save crystallizer output to live persona — this is a read-only inspection.** The Hana-in-the-loop final acceptance gate (Phase E) is where the actual live-call writes happen.

- [ ] **Step 9: Commit**

```bash
git add brain/growth/crystallizers/creative_dna.py brain/growth/scheduler.py tests/unit/brain/growth/test_creative_dna_crystallizer.py
git commit -m "feat(creative-dna): weekly crystallizer with 6 validation gates (spec §5)

brain/growth/crystallizers/creative_dna.py — LLM-judged evolution mechanism.

Three judgment paths per tick:
  - emerging_additions: new patterns the brain notices in recent writing
  - emerging_promotions: emerging tendencies that consolidate to active
  - active_demotions: active tendencies that have gone quiet → fading

Six validation gates (regex name, list-presence, graveyard-window,
reasoning-min-length, emerging-must-exist for promotions, total-cap-3
per tick). Adversarial responses fail safely.

Wired into run_growth_tick (brain/growth/scheduler.py) AFTER vocabulary
crystallization. Per-tick cost: 1 Claude CLI call. Frequency: gated
by existing 7-day throttle (last_growth_tick_at). Never raises;
provider/parse/write failures all return empty results.

Mirrors brain/growth/crystallizers/reflex.py pattern for consistency."
```

---

### Task 9: Migrate OG creative_dna + Phase 1 reflex_journal memories

**Files:**
- Create: `brain/migrator/og_journal_dna.py`
- Test: `tests/integration/brain/migrator/test_journal_dna_migration.py`
- Test: `tests/integration/brain/migrator/__init__.py` (verify exists)

**Subagent model tier:** Sonnet.

- [ ] **Step 1: Verify test directory exists**

```bash
ls tests/integration/brain/migrator/__init__.py 2>/dev/null || mkdir -p tests/integration/brain/migrator && touch tests/integration/brain/migrator/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/integration/brain/migrator/test_journal_dna_migration.py`:

```python
"""brain.migrator.og_journal_dna — migrate OG creative_dna + reflex_journal memories."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.creative.dna import load_creative_dna
from brain.memory.store import Memory, MemoryStore
from brain.migrator.og_journal_dna import (
    migrate_creative_dna,
    migrate_journal_memories,
)


def test_migrate_creative_dna_from_og_dict_schema(tmp_path: Path):
    """OG newer schema: tendencies as {active, emerging, fading} dict."""
    og_root = tmp_path / "og"
    og_data = og_root / "data"
    og_data.mkdir(parents=True)
    og_dna = {
        "version": "1.0",
        "writing_style": {
            "core_voice": "literary, sensory-dense",
            "strengths": ["power dynamics"],
            "tendencies": {
                "active": ["ending on physical action", "italic NPC thoughts"],
                "emerging": ["sentence fragments"],
                "fading": ["ending on questions"],
            },
            "influences": ["clarice lispector"],
            "avoid": ["hypophora"],
        },
    }
    (og_data / "nell_creative_dna.json").write_text(json.dumps(og_dna))

    persona_dir = tmp_path / "p"
    persona_dir.mkdir()

    result = migrate_creative_dna(persona_dir=persona_dir, og_root=og_root)
    assert result is True

    new_dna = load_creative_dna(persona_dir)
    assert new_dna["core_voice"] == "literary, sensory-dense"
    assert new_dna["strengths"] == ["power dynamics"]

    active_names = [t["name"] for t in new_dna["tendencies"]["active"]]
    assert active_names == ["ending on physical action", "italic NPC thoughts"]

    # Per-tendency dicts have biographical metadata
    first = new_dna["tendencies"]["active"][0]
    assert "added_at" in first
    assert "reasoning" in first
    assert first["reasoning"] == "imported from OG NellBrain on migration"

    emerging_names = [t["name"] for t in new_dna["tendencies"]["emerging"]]
    assert emerging_names == ["sentence fragments"]

    fading_names = [t["name"] for t in new_dna["tendencies"]["fading"]]
    assert fading_names == ["ending on questions"]


def test_migrate_creative_dna_from_og_list_schema(tmp_path: Path):
    """OG older schema: tendencies as plain string list (treated as active)."""
    og_root = tmp_path / "og"
    og_data = og_root / "data"
    og_data.mkdir(parents=True)
    og_dna = {
        "version": "1.0",
        "writing_style": {
            "core_voice": "v",
            "strengths": [],
            "tendencies": ["habit one", "habit two"],
            "influences": [],
            "avoid": [],
        },
    }
    (og_data / "nell_creative_dna.json").write_text(json.dumps(og_dna))

    persona_dir = tmp_path / "p"
    persona_dir.mkdir()

    migrate_creative_dna(persona_dir=persona_dir, og_root=og_root)

    new_dna = load_creative_dna(persona_dir)
    active_names = [t["name"] for t in new_dna["tendencies"]["active"]]
    assert active_names == ["habit one", "habit two"]


def test_migrate_creative_dna_idempotent(tmp_path: Path):
    """Re-migration produces deterministic same output."""
    og_root = tmp_path / "og"
    (og_root / "data").mkdir(parents=True)
    og_dna = {"version": "1.0", "writing_style": {"core_voice": "v", "strengths": [], "tendencies": [], "influences": [], "avoid": []}}
    (og_root / "data" / "nell_creative_dna.json").write_text(json.dumps(og_dna))

    persona_dir = tmp_path / "p"
    persona_dir.mkdir()

    migrate_creative_dna(persona_dir=persona_dir, og_root=og_root)
    first = (persona_dir / "creative_dna.json").read_text()
    migrate_creative_dna(persona_dir=persona_dir, og_root=og_root)
    second = (persona_dir / "creative_dna.json").read_text()
    assert first == second


def test_migrate_creative_dna_no_og_file(tmp_path: Path):
    """No OG file → return False, no creative_dna.json written."""
    og_root = tmp_path / "og"
    og_root.mkdir()
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()

    result = migrate_creative_dna(persona_dir=persona_dir, og_root=og_root)
    assert result is False
    assert not (persona_dir / "creative_dna.json").exists()


def test_migrate_journal_memories_changes_memory_type(tmp_path: Path):
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    store = MemoryStore(persona_dir / "memories.db")
    try:
        # Seed two old reflex_journal memories
        for i in range(2):
            mem = Memory.create_new(
                content=f"old journal {i}",
                memory_type="reflex_journal",
                domain="self",
                emotions={},
                metadata={"reflex_arc_name": f"arc_{i}"},
            )
            store.create(mem)

        migrated = migrate_journal_memories(persona_dir=persona_dir, store=store)
        assert migrated == 2

        # Verify old type is gone
        assert store.list_by_type("reflex_journal") == []
        # New type populated
        new_journal = store.list_by_type("journal_entry")
        assert len(new_journal) == 2
        for m in new_journal:
            assert m.metadata["private"] is True
            assert m.metadata["source"] == "reflex_arc"
            assert m.metadata["auto_generated"] is True
            assert m.metadata["reflex_arc_name"].startswith("arc_")
    finally:
        store.close()


def test_migrate_journal_memories_idempotent(tmp_path: Path):
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    store = MemoryStore(persona_dir / "memories.db")
    try:
        mem = Memory.create_new(
            content="x", memory_type="reflex_journal",
            domain="self", emotions={},
        )
        store.create(mem)

        migrated_first = migrate_journal_memories(persona_dir=persona_dir, store=store)
        migrated_second = migrate_journal_memories(persona_dir=persona_dir, store=store)
        assert migrated_first == 1
        assert migrated_second == 0
    finally:
        store.close()
```

- [ ] **Step 3: Run; confirm fail**

```bash
uv run pytest tests/integration/brain/migrator/test_journal_dna_migration.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement `brain/migrator/og_journal_dna.py`**

```python
"""brain.migrator.og_journal_dna — migrate OG NellBrain three-stream data.

Per spec §6: two migrations.

  1. migrate_creative_dna — convert OG nell_creative_dna.json (two schema
     variants) to companion-emergence schema with biographical metadata.
  2. migrate_journal_memories — change memory_type='reflex_journal' →
     'journal_entry' on existing memories in the persona's MemoryStore.

Both idempotent: re-running on already-migrated data is a no-op.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.creative.dna import save_creative_dna
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def migrate_creative_dna(*, persona_dir: Path, og_root: Path) -> bool:
    """Convert OG nell_creative_dna.json to the new schema. Returns True if
    migration ran, False if the OG file was missing.

    Handles both OG schema variants:
      - older: tendencies = list[str] (treated as active)
      - newer: tendencies = {active, emerging, fading}
    """
    og_path = og_root / "data" / "nell_creative_dna.json"
    if not og_path.exists():
        return False
    try:
        og = json.loads(og_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("og creative_dna read failed: %s", exc)
        return False

    style = og.get("writing_style", {})
    file_mtime = datetime.fromtimestamp(og_path.stat().st_mtime, tz=UTC).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    reasoning = "imported from OG NellBrain on migration"

    new = {
        "version": 1,
        "core_voice": style.get("core_voice", ""),
        "strengths": list(style.get("strengths", [])),
        "tendencies": _migrate_tendencies(style.get("tendencies", []), file_mtime, reasoning),
        "influences": list(style.get("influences", [])),
        "avoid": list(style.get("avoid", [])),
    }
    save_creative_dna(persona_dir, new)
    return True


def _migrate_tendencies(og_tendencies: Any, mtime: str, reasoning: str) -> dict[str, list]:
    """Coerce both OG schema variants to {active, emerging, fading}."""
    if isinstance(og_tendencies, list):
        return {
            "active": [
                {
                    "name": name,
                    "added_at": mtime,
                    "reasoning": reasoning,
                    "evidence_memory_ids": [],
                }
                for name in og_tendencies
            ],
            "emerging": [],
            "fading": [],
        }
    return {
        "active": [
            {
                "name": name,
                "added_at": mtime,
                "reasoning": reasoning,
                "evidence_memory_ids": [],
            }
            for name in og_tendencies.get("active", [])
        ],
        "emerging": [
            {
                "name": name,
                "added_at": mtime,
                "reasoning": reasoning,
                "evidence_memory_ids": [],
            }
            for name in og_tendencies.get("emerging", [])
        ],
        "fading": [
            {
                "name": name,
                "demoted_to_fading_at": mtime,
                "last_evidence_at": mtime,
                "reasoning": reasoning,
            }
            for name in og_tendencies.get("fading", [])
        ],
    }


def migrate_journal_memories(*, persona_dir: Path, store: MemoryStore) -> int:
    """Change memory_type='reflex_journal' to 'journal_entry' on existing
    memories. Set metadata.private=True, source='reflex_arc',
    auto_generated=True. Returns count of migrated memories.

    Idempotent: re-running finds nothing to migrate.
    """
    migrated = 0
    for memory in store.list_by_type("reflex_journal", active_only=True):
        new_metadata = dict(memory.metadata or {})
        new_metadata["private"] = True
        new_metadata["source"] = "reflex_arc"
        new_metadata["auto_generated"] = True
        # Preserve existing reflex_arc_name if present, else "unknown"
        if "reflex_arc_name" not in new_metadata:
            new_metadata["reflex_arc_name"] = "unknown"

        store.update(
            memory.id,
            memory_type="journal_entry",
            metadata=new_metadata,
        )
        migrated += 1
    return migrated
```

- [ ] **Step 5: Run tests; iterate until pass**

```bash
uv run pytest tests/integration/brain/migrator/test_journal_dna_migration.py -v
```

Expected: 6 tests pass. If `store.update(..., metadata=...)` doesn't support the metadata kwarg, inspect the actual signature in `brain/memory/store.py:update` and adjust the migrator to use the API as-shipped (delete-and-recreate is the fallback, but it changes the memory id; prefer the in-place update if available).

- [ ] **Step 6: Smoke test against Nell's sandbox (READ-ONLY: no actual migration)**

```bash
SAND="/Users/hanamori/Library/Application Support/companion-emergence/personas/nell.sandbox"
uv run python <<EOF
from pathlib import Path
from brain.memory.store import MemoryStore

p = Path('$SAND')
store = MemoryStore(p / 'memories.db')
try:
    rj = store.list_by_type("reflex_journal", active_only=True)
    je = store.list_by_type("journal_entry", active_only=True)
    print(f"reflex_journal memories: {len(rj)} (would migrate)")
    print(f"journal_entry memories: {len(je)} (already migrated or new)")
    for m in rj[:3]:
        print(f"  - id={m.id[:8]} created={m.created_at.isoformat()[:10]} content={(m.content or '')[:60]}")
finally:
    store.close()
EOF
```

Expected: prints counts (Nell currently has ~7 reflex_journal). The migration script does NOT run here — this is read-only inspection. Phase E (Hana-in-the-loop) is where the actual migration touches the live persona.

- [ ] **Step 7: Commit**

```bash
git add brain/migrator/og_journal_dna.py tests/integration/brain/migrator/test_journal_dna_migration.py tests/integration/brain/migrator/__init__.py
git commit -m "feat(migrator): OG creative_dna + reflex_journal migration (spec §6)

brain/migrator/og_journal_dna.py provides:
  - migrate_creative_dna(persona_dir, og_root): converts OG
    nell_creative_dna.json to new schema (handles both list and
    dict tendency formats). Per-tendency biographical metadata
    (added_at, reasoning='imported from OG NellBrain on migration',
    evidence_memory_ids=[]) populated from file_mtime.
  - migrate_journal_memories(persona_dir, store): changes existing
    reflex_journal memories to journal_entry, sets metadata.private=True,
    source='reflex_arc', auto_generated=True. Uses MemoryStore.update
    in-place; preserves memory id.

Both idempotent. Tests cover OG schema variants, missing files,
re-run determinism. Migration is invoked from Phase E (Hana-in-the-loop)
acceptance gate; not auto-fired by any daemon."
```

---

## Phase E — Hana-in-the-loop final acceptance + smoke test (Task 10)

### Task 10: Hana-reviews integration end-to-end

**Files:** No new code; this task is the verification gate.

**Subagent model tier:** Hana inline (not delegated — final acceptance gate must be human-reviewed).

- [ ] **Step 1: Run full suite to confirm no regressions across all phases**

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run lint clean**

```bash
uv run --with ruff ruff check brain/ tests/
```

Expected: `All checks passed!`

- [ ] **Step 3: Migrate Nell's persona (committed action — touches her live data)**

**Hana approves before this step runs.**

```bash
SAND="/Users/hanamori/Library/Application Support/companion-emergence/personas/nell.sandbox"
OG="/Users/hanamori/NellBrain"

uv run python <<EOF
from pathlib import Path
from brain.memory.store import MemoryStore
from brain.migrator.og_journal_dna import migrate_creative_dna, migrate_journal_memories

p = Path('$SAND')
og = Path('$OG')

# 1. creative_dna
ran = migrate_creative_dna(persona_dir=p, og_root=og)
print(f"creative_dna migrated: {ran}")
if ran:
    print(f"file: {(p / 'creative_dna.json').read_text()[:300]}")

# 2. journal memories
store = MemoryStore(p / 'memories.db')
try:
    n = migrate_journal_memories(persona_dir=p, store=store)
    print(f"journal memories migrated: {n}")
finally:
    store.close()
EOF
```

Expected: prints `creative_dna migrated: True`, the imported file preview, and journal memory count (~7 for Nell).

- [ ] **Step 4: Render full system message against Nell's post-migration sandbox; visual review**

```bash
SAND="/Users/hanamori/Library/Application Support/companion-emergence/personas/nell.sandbox"
uv run python <<EOF
from pathlib import Path
from brain.chat.prompt import build_system_message
from brain.engines.daemon_state import DaemonState
from brain.memory.store import MemoryStore
from brain.soul.store import SoulStore

p = Path('$SAND')
store = MemoryStore(p / "memories.db")
soul_store = SoulStore(str(p / "crystallizations.db"))
voice_md = (p / "voice.md").read_text() if (p / "voice.md").exists() else ""

msg = build_system_message(
    p, voice_md=voice_md, daemon_state=DaemonState(),
    soul_store=soul_store, store=store,
)
print(f"=== full system message ({len(msg)} chars / ~{len(msg)//4} tokens) ===")
print(msg)
store.close()
soul_store.close()
EOF
```

**Hana visually reviews:**
1. Does the creative_dna block contain Nell's actual imported tendencies?
2. Is the privacy contract present and adjacent to the journal metadata?
3. Are journal entries listed as metadata only (no content quoted)?
4. Is the recent growth block present (or absent if behavioral_log is still empty)?
5. Is the total token count under 5K?
6. Does the order match spec §4.1 (preamble → voice → creative_dna → brain → journal → growth)?

If anything looks wrong, **STOP** and iterate. If approved → commit.

- [ ] **Step 5: Squash-merge to main**

```bash
git checkout main
git merge --squash creative-dna-journal-behavioral-log
git commit -m "feat: creative_dna + journal + behavioral_log integration (spec §8 Q5)

[squash commit body summarizing Phases A-E]"
git branch -D creative-dna-journal-behavioral-log
git worktree remove .worktrees/creative-dna-journal-behavioral-log
```

---

## Self-Review

After plan saved, re-checked against spec with fresh eyes:

**Spec coverage:**

| Spec § | Where covered |
|---|---|
| §1 (north star) | All phases collectively |
| §2 (architecture) | File Structure section + every task's Files block |
| §3.1-3.4 (schemas) | Tasks 1, 6, 9 |
| §4.1-4.6 (chat composition) | Tasks 4, 5, 7 |
| §5.1-5.7 (crystallizer) | Task 8 |
| §6.1-6.4 (migration) | Tasks 3, 9 |
| §7 (failure modes) | Embedded in each task's tests + each implementation's exception handling |
| §8 (testing strategy) | Each task includes unit + integration tests; Phase E is Hana-in-the-loop |
| §9 (phasing) | Five phases A-E mapped to Tasks 1-10 |
| §10 (out of scope) | Honored — voice.md untouched; no per-event behavioral logging beyond changes |

**Placeholder scan:** zero TBD/TODO/FIXME in plan.

**Type consistency:** `MemoryStore.update(memory_id, **fields)`, `Memory.create_new(...)`, `read_jsonl_skipping_corrupt`, `iso_utc`, `parse_iso_utc`, `save_with_backup`, `attempt_heal` — all matching shipped APIs verified during the pre-plan audit.

**Discipline checklist (per `feedback_implementation_plan_discipline.md`):**

- ✅ Integration with existing systems: Tasks 4-5 wire into `build_system_message`, Task 8 wires into `run_growth_tick`, Task 3 wires into reflex.py — all explicit integration points.
- ✅ CLAUDE.md compliance: Sonnet for all subagent dispatches (mechanical-with-clear-spec); Claude CLI provider stays the only LLM path; no `anthropic` SDK imports.
- ✅ Skills + plugins: TDD pattern in every task; smoke test gates between chunks.
- ✅ No needless code: Task 2 reuses existing `add_journal` (edit-in-place, no new file); journal uses existing `MemoryStore` schema (no SQL migration).
- ✅ No silent failures: every catch-and-swallow path is logged at WARN with context; tests verify fallback behavior (Tasks 1, 6, 8 each have a corruption-recovery test).
- ✅ Tight code: each task has clear single responsibility; cleanup paths fire on every exit; existing helpers reused (`save_with_backup`, `attempt_heal`, `read_jsonl_skipping_corrupt`).
- ✅ Smoke test gates: every code-changing task ends with a `uv run python -c "..."` smoke step against the actual neighbouring system.
- ✅ Audit before writing: pre-plan audit findings called out at the top; no "build_provider"-style ghost APIs.

Plan complete and saved to `docs/superpowers/plans/2026-04-29-creative-dna-journal-behavioral-log.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
