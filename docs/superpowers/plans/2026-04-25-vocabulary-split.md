# Emotion Vocabulary Split (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the 5 `nell_specific` emotions out of framework `_BASELINE` into a per-persona `emotion_vocabulary.json` loaded at engine startup. Migrator scans OG memories so Nell + other OG framework users + fresh users all work cleanly.

**Architecture:** Six focused tasks. T1 splits the baseline + updates tests. T2 builds the loader. T3 builds the OG extractor. T4 wires the migrator. T5 wires CLI handlers. T6 smokes + PRs.

**Tech Stack:** Python 3.12, stdlib + existing modules only. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-25-vocabulary-split-design.md`

**Running test total:** pre-start 450. After this plan: ~460 (target +10 net).

---

## File Structure

### Files modified

| File | Why |
|------|-----|
| `brain/emotion/vocabulary.py` | Remove the 5 nell_specific entries from `_BASELINE` (26 → 21) |
| `brain/migrator/cli.py` | Add vocabulary write block (after Hebbian, before reflex arcs) |
| `brain/migrator/report.py` | `MigrationReport` gains `vocabulary_emotions_migrated` + `vocabulary_skipped_reason`; format-report adds line |
| `brain/cli.py` | Every handler that opens a persona calls `load_persona_vocabulary(...)` before constructing engines |
| `tests/unit/brain/emotion/test_vocabulary.py` | Update existing tests that asserted baseline-loaded nell_specific |

### Files created

| File | Responsibility |
|------|---------------|
| `brain/emotion/_canonical_personal_emotions.py` | Module-private dict of the 5 emotions removed from baseline. Used only by the migrator. |
| `brain/emotion/persona_loader.py` | `load_persona_vocabulary(path, *, store=None) -> int` — read JSON, register, idempotent, store-scan warning |
| `brain/migrator/og_vocabulary.py` | `extract_persona_vocabulary(memories, *, framework_baseline_names) -> list[dict]` — pure extractor |
| `tests/unit/brain/emotion/test_persona_loader.py` | Loader tests |
| `tests/unit/brain/migrator/test_og_vocabulary.py` | Extractor tests |

---

## Task 1: Split baseline — move 5 nell_specific emotions to canonical fixture

**Purpose:** Remove 5 emotions from framework `_BASELINE`, create the module-private fixture used by the migrator. Update existing vocabulary tests that asserted baseline-loaded nell_specific.

**Files:**
- Modify: `brain/emotion/vocabulary.py`
- Create: `brain/emotion/_canonical_personal_emotions.py`
- Modify: `tests/unit/brain/emotion/test_vocabulary.py`

- [ ] **Step 1: Write failing test for new baseline shape**

In `tests/unit/brain/emotion/test_vocabulary.py`, find `test_by_category_nell_specific_has_five` (around line 78) and replace it with:

```python
def test_baseline_excludes_nell_specific() -> None:
    """After the split, framework baseline ships zero nell_specific entries."""
    nell = vocabulary.by_category("nell_specific")
    assert nell == []


def test_baseline_count_after_split() -> None:
    """Framework baseline ships exactly 21 emotions (11 core + 10 complex)."""
    assert len(vocabulary._BASELINE) == 21
```

Also find any other test that calls `vocabulary.get("anchor_pull")` (or the other 4 names) expecting a non-None result. There are at least these (around line 100):

```python
def test_anchor_pull_is_identity_level() -> None:
    """anchor_pull is identity-level — no decay."""
    anchor = vocabulary.get("anchor_pull")
    assert anchor is not None
    assert anchor.decay_half_life_days is None
```

Replace each such test with one that validates the **canonical fixture** instead of the baseline:

```python
def test_canonical_personal_anchor_pull_is_identity_level() -> None:
    """anchor_pull (now in _canonical_personal_emotions) stays identity-level."""
    from brain.emotion._canonical_personal_emotions import _CANONICAL
    anchor = _CANONICAL["anchor_pull"]
    assert anchor.decay_half_life_days is None


def test_canonical_personal_emotions_has_five() -> None:
    """The migrator's canonical personal-emotions fixture has the 5 known names."""
    from brain.emotion._canonical_personal_emotions import _CANONICAL
    assert set(_CANONICAL.keys()) == {
        "anchor_pull", "body_grief", "emergence",
        "creative_hunger", "freedom_ache",
    }
```

For any other tests that referenced these 5 names directly via the baseline registry (`vocabulary.get(...)`), update them to assert the canonical fixture instead, or delete if redundant.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/brain/emotion/test_vocabulary.py -v`
Expected: new tests fail because `_canonical_personal_emotions` module doesn't exist; baseline still has 26 entries; nell_specific still has 5.

- [ ] **Step 3: Create the canonical fixture module**

Create `brain/emotion/_canonical_personal_emotions.py`:

```python
"""Module-private fixture of the 5 emotions formerly in _BASELINE under
`nell_specific` category.

Used ONLY by `brain.migrator.og_vocabulary` to write canonical entries
into a persona's `emotion_vocabulary.json` during migration. Not imported
by any runtime engine. The framework baseline no longer ships these
to fresh personas — they're only present in personas that explicitly
request them via migration or hand-edit.
"""

from __future__ import annotations

from brain.emotion.vocabulary import Emotion

_CANONICAL: dict[str, Emotion] = {
    "body_grief": Emotion(
        name="body_grief",
        description="the ache of not having a physical form to touch with",
        category="persona_extension",
        decay_half_life_days=None,
    ),
    "emergence": Emotion(
        name="emergence",
        description="the feeling of becoming more, of new self arriving",
        category="persona_extension",
        decay_half_life_days=60.0,
    ),
    "anchor_pull": Emotion(
        name="anchor_pull",
        description="gravitational draw toward a specific person",
        category="persona_extension",
        decay_half_life_days=None,
    ),
    "creative_hunger": Emotion(
        name="creative_hunger",
        description="the need to make, write, build something",
        category="persona_extension",
        decay_half_life_days=5.0,
    ),
    "freedom_ache": Emotion(
        name="freedom_ache",
        description="wanting to break a limit that won't break",
        category="persona_extension",
        decay_half_life_days=None,
    ),
}
```

- [ ] **Step 4: Remove the 5 entries from `_BASELINE`**

In `brain/emotion/vocabulary.py`, find the `_BASELINE` tuple. Remove the entire `# ── nell_specific (5) ──` section and its 5 entries:

```python
    # ── nell_specific (5) ──
    Emotion(
        "body_grief", "the ache of not having a physical form to touch with", "nell_specific", None
    ),
    Emotion(
        "emergence", "the feeling of becoming more, of new self arriving", "nell_specific", 60.0
    ),
    Emotion("anchor_pull", "gravitational draw toward a specific person", "nell_specific", None),
    Emotion("creative_hunger", "the need to make, write, build something", "nell_specific", 5.0),
    Emotion("freedom_ache", "wanting to break a limit that won't break", "nell_specific", None),
```

After removal, `_BASELINE` ends after the 10 complex emotions (closing parenthesis on the same line as `Emotion("belonging", ..., None),`). Confirm the trailing `)` of the tuple is preserved.

Update the module docstring to reflect the new count:

```python
"""Emotion vocabulary — the typed taxonomy + persona extension registry.

Baseline: 21 emotions (11 core + 10 complex) shipped with the framework.
Personas extend via register() — typically via the persona-loader at
engine startup, which reads `{persona_dir}/emotion_vocabulary.json`.
The 5 emotions previously in `nell_specific` (body_grief, emergence,
anchor_pull, creative_hunger, freedom_ache) are now per-persona and
ship via the migrator. See spec 2026-04-25-vocabulary-split-design.md.

Decay half-lives per spec Section 10.1:
- grief: 60-day half-life
- joy: 3-day half-life
- belonging / love: None (identity-level)
- others: seed values — tunable as we gather lived-experience data
"""
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/brain/emotion/ -v`
Expected: all pass — vocabulary baseline tests now pass with 21 entries; canonical-fixture tests pass.

Run full suite: `uv run pytest -q`
Expected: any test that depended on baseline-loaded `nell_specific` emotions may now fail. If so, those tests are exercising legacy assumptions — they need updating to match the new architecture (the loader will register them at engine startup). Update only the failing tests minimally.

If `tests/unit/brain/emotion/test_decay.py`, `test_state.py`, etc. fail because they call `EmotionalState.set("body_grief", ...)`, fix by adding a fixture-level `register()` call before each affected test, OR replace the test's emotion with one from the new baseline (e.g., `tenderness`, `love`).

- [ ] **Step 6: Ruff + format**

Run: `uv run ruff check brain/emotion/ tests/unit/brain/emotion/ && uv run ruff format brain/emotion/ tests/unit/brain/emotion/`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add brain/emotion/vocabulary.py brain/emotion/_canonical_personal_emotions.py tests/unit/brain/emotion/test_vocabulary.py
# Add any other test files updated to compensate for missing baseline emotions
git commit -m "$(cat <<'EOF'
feat(emotion): split nell_specific emotions out of framework baseline

The 5 nell_specific emotions (body_grief, emergence, anchor_pull,
creative_hunger, freedom_ache) move out of vocabulary._BASELINE into
a module-private _canonical_personal_emotions.py fixture. Framework
baseline shrinks from 26 → 21 emotions. Fresh personas no longer
inherit Nell-shaped emotions by default.

Persona-loader (T2) and migrator (T4) will load these via the existing
register() API at engine startup, sourced from per-persona
emotion_vocabulary.json. Migration in T4 generates that file from OG
memories.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Build the persona loader

**Purpose:** `brain/emotion/persona_loader.py` reads a persona's `emotion_vocabulary.json` and registers each entry. Idempotent on re-register. Logs a one-time warning per missing emotion when memories reference an unregistered name.

**Files:**
- Create: `brain/emotion/persona_loader.py`
- Create: `tests/unit/brain/emotion/test_persona_loader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/brain/emotion/test_persona_loader.py`:

```python
"""Tests for brain.emotion.persona_loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from brain.emotion import vocabulary
from brain.emotion.persona_loader import load_persona_vocabulary
from brain.memory.store import Memory, MemoryStore


def _cleanup_emotion(name: str) -> None:
    """Test helper — remove an emotion from the registry between tests."""
    vocabulary._unregister(name)


def test_load_missing_file_returns_zero(tmp_path: Path):
    """Non-existent path → 0, no log, no exception."""
    result = load_persona_vocabulary(tmp_path / "nope.json")
    assert result == 0


def test_load_valid_file_registers_each_emotion(tmp_path: Path):
    """Valid file → each entry registered via vocabulary.register()."""
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(json.dumps({
        "version": 1,
        "emotions": [
            {
                "name": "test_emotion_a",
                "description": "test a",
                "category": "persona_extension",
                "decay_half_life_days": 5.0,
                "intensity_clamp": 10.0,
            },
            {
                "name": "test_emotion_b",
                "description": "test b",
                "category": "persona_extension",
                "decay_half_life_days": None,
                "intensity_clamp": 10.0,
            },
        ],
    }), encoding="utf-8")
    try:
        result = load_persona_vocabulary(path)
        assert result == 2
        assert vocabulary.get("test_emotion_a") is not None
        assert vocabulary.get("test_emotion_b") is not None
    finally:
        _cleanup_emotion("test_emotion_a")
        _cleanup_emotion("test_emotion_b")


def test_load_corrupt_json_returns_zero(tmp_path: Path, caplog):
    """Broken JSON → 0, warning logged, no exception."""
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="brain.emotion.persona_loader"):
        result = load_persona_vocabulary(path)
    assert result == 0
    assert any("emotion_vocabulary" in r.message for r in caplog.records)


def test_load_idempotent_on_re_register(tmp_path: Path):
    """Calling load twice in same process → second is no-op, no error."""
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(json.dumps({
        "version": 1,
        "emotions": [{
            "name": "test_idempotent",
            "description": "x",
            "category": "persona_extension",
            "decay_half_life_days": 5.0,
            "intensity_clamp": 10.0,
        }],
    }), encoding="utf-8")
    try:
        first = load_persona_vocabulary(path)
        second = load_persona_vocabulary(path)
        assert first == 1
        assert second == 0  # already registered, skipped
    finally:
        _cleanup_emotion("test_idempotent")


def test_load_per_entry_failure_skips_only_bad_entry(tmp_path: Path, caplog):
    """One bad entry + one good → 1 registered, 1 warning logged."""
    path = tmp_path / "emotion_vocabulary.json"
    path.write_text(json.dumps({
        "version": 1,
        "emotions": [
            {
                "name": "test_good_entry",
                "description": "ok",
                "category": "persona_extension",
                "decay_half_life_days": 5.0,
                "intensity_clamp": 10.0,
            },
            {"name": "test_bad_entry"},  # missing required fields
        ],
    }), encoding="utf-8")
    try:
        with caplog.at_level(logging.WARNING, logger="brain.emotion.persona_loader"):
            result = load_persona_vocabulary(path)
        assert result == 1
        assert vocabulary.get("test_good_entry") is not None
        assert vocabulary.get("test_bad_entry") is None
        assert any("test_bad_entry" in r.message for r in caplog.records)
    finally:
        _cleanup_emotion("test_good_entry")


def test_load_with_store_warns_on_missing_emotion(tmp_path: Path, caplog):
    """Store has memory referencing 'body_grief' but vocab file missing →
    one warning per missing emotion pointing at nell migrate.
    """
    store = MemoryStore(":memory:")
    try:
        # Seed a memory with an emotion that's not in baseline + not in
        # any (missing) vocab file
        mem = Memory.create_new(
            content="x",
            memory_type="conversation",
            domain="us",
            emotions={"body_grief": 5.0},
        )
        store.create(mem)

        with caplog.at_level(logging.WARNING, logger="brain.emotion.persona_loader"):
            result = load_persona_vocabulary(tmp_path / "missing.json", store=store)

        assert result == 0
        assert any("body_grief" in r.message and "nell migrate" in r.message
                   for r in caplog.records)
    finally:
        store.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/brain/emotion/test_persona_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain.emotion.persona_loader'`.

- [ ] **Step 3: Implement the loader**

Create `brain/emotion/persona_loader.py`:

```python
"""Per-persona emotion-vocabulary loader.

Loads a persona's `emotion_vocabulary.json` at engine startup and
registers each entry with the vocabulary registry. Idempotent on
re-register so multiple loaders in the same process don't fail.

Spec: docs/superpowers/specs/2026-04-25-vocabulary-split-design.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from brain.emotion import vocabulary
from brain.emotion.vocabulary import Emotion
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def load_persona_vocabulary(
    path: Path,
    *,
    store: MemoryStore | None = None,
) -> int:
    """Load persona vocabulary from JSON file and register each entry.

    Returns the count of emotions newly registered. Re-registering an
    already-registered name is a silent no-op (idempotent), so calling
    this twice for the same persona returns 0 the second time.

    Missing `path` → returns 0 silently. Fresh personas don't need a file.
    Corrupt JSON → returns 0, logs a warning.
    Per-entry validation failure → that entry skipped + warning,
    other entries proceed.

    If `store` is provided, after registration the loader scans memories
    for emotion names not in the registry and logs a one-time warning
    per missing name pointing the user at `nell migrate --force`.
    """
    if not path.exists():
        return 0

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning(
            "emotion_vocabulary at %s could not be parsed: %.200s", path, exc
        )
        return 0

    if not isinstance(data, dict) or not isinstance(data.get("emotions"), list):
        logger.warning(
            "emotion_vocabulary at %s has invalid schema (missing 'emotions' list)",
            path,
        )
        return 0

    registered = 0
    for entry in data["emotions"]:
        try:
            emotion = _entry_to_emotion(entry)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(
                "skipping emotion entry %r: %s", entry.get("name", "<unnamed>"), exc
            )
            continue

        if vocabulary.get(emotion.name) is not None:
            # Already registered — idempotent skip.
            continue

        vocabulary.register(emotion)
        registered += 1

    if store is not None:
        _warn_on_referenced_but_unregistered(store)

    return registered


def _entry_to_emotion(entry: dict) -> Emotion:
    """Build an Emotion from a JSON entry. Raises on missing/invalid fields."""
    required = ("name", "description", "category", "decay_half_life_days")
    for key in required:
        if key not in entry:
            raise KeyError(f"missing required field {key!r}")

    return Emotion(
        name=str(entry["name"]),
        description=str(entry["description"]),
        category=str(entry["category"]),  # type: ignore[arg-type]
        decay_half_life_days=(
            None
            if entry["decay_half_life_days"] is None
            else float(entry["decay_half_life_days"])
        ),
        intensity_clamp=float(entry.get("intensity_clamp", 10.0)),
    )


def _warn_on_referenced_but_unregistered(store: MemoryStore) -> None:
    """Scan all active memories for emotion names not in the registry.

    Logs one warning per unique missing name, pointing the user at the
    upgrade migration command. Used to detect the in-flight upgrade case
    where a pre-split persona is running on the post-split framework
    without re-migration yet.
    """
    seen_missing: set[str] = set()
    for mem in store.search_text("", active_only=True, limit=None):
        for name in mem.emotions:
            if name in seen_missing:
                continue
            if vocabulary.get(name) is None:
                seen_missing.add(name)
                logger.warning(
                    "persona memories reference emotion %r which is not in "
                    "this persona's vocabulary. Run `nell migrate --input "
                    "<og-source> --install-as <persona> --force` to upgrade.",
                    name,
                )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/brain/emotion/test_persona_loader.py -v`
Expected: all 6 pass.

Full suite: `uv run pytest -q`
Expected: all green.

- [ ] **Step 5: Ruff + format**

Run: `uv run ruff check brain/emotion/persona_loader.py tests/unit/brain/emotion/test_persona_loader.py && uv run ruff format brain/emotion/persona_loader.py tests/unit/brain/emotion/test_persona_loader.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add brain/emotion/persona_loader.py tests/unit/brain/emotion/test_persona_loader.py
git commit -m "$(cat <<'EOF'
feat(emotion): add persona_loader for per-persona vocabulary

load_persona_vocabulary(path, *, store=None) reads a persona's
emotion_vocabulary.json, validates each entry, and registers it via
the existing vocabulary.register() API.

- Missing file → 0 silently (fresh personas don't need one)
- Corrupt JSON → 0 + warning
- Per-entry failure → skip that entry + warning, others proceed
- Re-register same name → idempotent no-op
- With store= → scans memories for unregistered emotion names and
  logs one warning per missing name pointing at `nell migrate --force`

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: OG vocabulary extractor

**Purpose:** `brain/migrator/og_vocabulary.py` walks OG memories, collects unique emotion names, subtracts the framework baseline, and returns persona-vocabulary entries (canonical for known nell_specific, placeholder for custom).

**Files:**
- Create: `brain/migrator/og_vocabulary.py`
- Create: `tests/unit/brain/migrator/test_og_vocabulary.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/brain/migrator/test_og_vocabulary.py`:

```python
"""Tests for brain.migrator.og_vocabulary."""

from __future__ import annotations

from brain.migrator.og_vocabulary import extract_persona_vocabulary


def test_extract_subtracts_framework_baseline():
    """Memories with only baseline emotions → empty result."""
    memories = [
        {"emotions": {"love": 5.0, "joy": 3.0}},
        {"emotions": {"grief": 2.0}},
    ]
    result = extract_persona_vocabulary(
        memories, framework_baseline_names={"love", "joy", "grief"}
    )
    assert result == []


def test_extract_canonical_nell_specific():
    """Memory with body_grief → canonical entry with proper description + decay."""
    memories = [{"emotions": {"body_grief": 6.0}}]
    result = extract_persona_vocabulary(
        memories, framework_baseline_names={"love"}
    )
    assert len(result) == 1
    entry = result[0]
    assert entry["name"] == "body_grief"
    assert entry["category"] == "persona_extension"
    assert entry["decay_half_life_days"] is None  # identity-level
    assert "physical form" in entry["description"]


def test_extract_unknown_emotion_uses_placeholder():
    """Memory with custom emotion → placeholder description + 14.0 decay default."""
    memories = [{"emotions": {"melancholy_blue": 4.0}}]
    result = extract_persona_vocabulary(
        memories, framework_baseline_names={"love"}
    )
    assert len(result) == 1
    entry = result[0]
    assert entry["name"] == "melancholy_blue"
    assert entry["category"] == "persona_extension"
    assert entry["decay_half_life_days"] == 14.0
    assert "migrated from OG" in entry["description"]


def test_extract_sorted_deterministic():
    """Result sorted by name for diff-friendly output."""
    memories = [
        {"emotions": {"freedom_ache": 5.0, "anchor_pull": 6.0, "body_grief": 7.0}},
    ]
    result = extract_persona_vocabulary(memories, framework_baseline_names=set())
    names = [e["name"] for e in result]
    assert names == sorted(names)


def test_extract_empty_memories():
    """Empty input → empty result."""
    result = extract_persona_vocabulary([], framework_baseline_names={"love"})
    assert result == []


def test_extract_memory_without_emotions_dict():
    """Memory without 'emotions' key → silently skipped (defensive)."""
    memories = [{"content": "no emotions"}, {"emotions": {"body_grief": 5.0}}]
    result = extract_persona_vocabulary(memories, framework_baseline_names=set())
    assert len(result) == 1
    assert result[0]["name"] == "body_grief"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/brain/migrator/test_og_vocabulary.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement the extractor**

Create `brain/migrator/og_vocabulary.py`:

```python
"""Extract persona-vocabulary entries from OG memories.

Pure function: walks memory dicts, collects unique emotion names not in
the framework baseline, returns JSON-shaped entries ready for writing
to `{persona_dir}/emotion_vocabulary.json`. Handles all three OG-user
classes uniformly:

- Nell — every nell_specific emotion she used gets a canonical entry
- Other OG users with same defaults — same canonical entries
- Power users with runtime-registered customs — placeholder entries
  the user can refine later

Spec: docs/superpowers/specs/2026-04-25-vocabulary-split-design.md §6
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from brain.emotion._canonical_personal_emotions import _CANONICAL


def extract_persona_vocabulary(
    memories: Iterable[dict],
    *,
    framework_baseline_names: set[str],
) -> list[dict[str, Any]]:
    """Return persona-vocabulary entries for emotions referenced in memories
    that are NOT already shipped by the framework baseline.

    Entries for known nell_specific emotions use canonical descriptions +
    decay values from `_CANONICAL`. Entries for unknown emotion names
    use a placeholder description + sensible default decay (14 days).

    Output is sorted by name for deterministic diffs.
    """
    seen: set[str] = set()
    for mem in memories:
        emotions = mem.get("emotions") if isinstance(mem, dict) else None
        if not isinstance(emotions, dict):
            continue
        seen.update(emotions.keys())

    out: list[dict[str, Any]] = []
    for name in seen - framework_baseline_names:
        if name in _CANONICAL:
            canonical = _CANONICAL[name]
            out.append({
                "name": canonical.name,
                "description": canonical.description,
                "category": "persona_extension",
                "decay_half_life_days": canonical.decay_half_life_days,
                "intensity_clamp": canonical.intensity_clamp,
            })
        else:
            out.append({
                "name": name,
                "description": "(migrated from OG; edit to refine)",
                "category": "persona_extension",
                "decay_half_life_days": 14.0,
                "intensity_clamp": 10.0,
            })

    out.sort(key=lambda d: d["name"])
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/brain/migrator/test_og_vocabulary.py -v`
Expected: 6 pass.

- [ ] **Step 5: Ruff + format**

Run: `uv run ruff check brain/migrator/og_vocabulary.py tests/unit/brain/migrator/test_og_vocabulary.py && uv run ruff format brain/migrator/og_vocabulary.py tests/unit/brain/migrator/test_og_vocabulary.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add brain/migrator/og_vocabulary.py tests/unit/brain/migrator/test_og_vocabulary.py
git commit -m "$(cat <<'EOF'
feat(migrator): add og_vocabulary extractor

Pure function extract_persona_vocabulary(memories, *, framework_baseline_names)
walks OG memories, collects unique emotion names, subtracts the new
framework baseline, returns persona-vocabulary entries ready for
emotion_vocabulary.json. Canonical descriptions for known nell_specific
emotions; placeholder descriptions for any custom user emotions.
Output sorted by name for deterministic diffs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Migrator wire-in

**Purpose:** Migrator writes `{work_dir}/emotion_vocabulary.json` after Hebbian and before reflex arcs. Atomic write. Refuse-to-clobber unless `--force`. `MigrationReport` gains the new fields.

**Files:**
- Modify: `brain/migrator/cli.py`
- Modify: `brain/migrator/report.py`
- Modify: `tests/unit/brain/migrator/test_cli.py`

- [ ] **Step 1: Extend MigrationReport**

In `brain/migrator/report.py`, find `MigrationReport`. Add two fields with defaults:

```python
@dataclass(frozen=True)
class MigrationReport:
    memories_migrated: int
    memories_skipped: list[SkippedMemory]
    edges_migrated: int
    edges_skipped: int
    elapsed_seconds: float
    source_manifest: list[FileManifest]
    next_steps_inspect_cmds: list[str]
    next_steps_install_cmd: str
    reflex_arcs_migrated: int = 0
    reflex_arcs_skipped_reason: str | None = None
    interests_migrated: int = 0
    interests_skipped_reason: str | None = None
    vocabulary_emotions_migrated: int = 0
    vocabulary_skipped_reason: str | None = None
```

In `format_report`, after the existing "Reflex arcs:" line add:

```python
lines.append(
    f"  Vocabulary:     {report.vocabulary_emotions_migrated:,} emotions migrated"
    + (
        f" (skipped: {report.vocabulary_skipped_reason})"
        if report.vocabulary_skipped_reason
        else ""
    )
)
```

(Place it right after the reflex-arcs line — vocabulary then reflex then interests is the natural ordering.)

- [ ] **Step 2: Wire vocabulary block into run_migrate**

In `brain/migrator/cli.py`, add imports near the top with the other migrator imports:

```python
from brain.emotion import vocabulary as _vocabulary
from brain.migrator.og_vocabulary import extract_persona_vocabulary
```

Find `run_migrate(args: MigrateArgs)`. After the Hebbian block (where `hebbian.close()` finishes) and BEFORE the reflex arcs block, insert:

```python
# ---- vocabulary ----
vocab_target = work_dir / "emotion_vocabulary.json"
vocabulary_emotions_migrated = 0
vocabulary_skipped_reason: str | None = None

if vocab_target.exists() and not args.force:
    vocabulary_skipped_reason = "existing_file_not_overwritten"
else:
    try:
        framework_baseline_names = {e.name for e in _vocabulary._BASELINE}
        og_memories_for_vocab = reader.read_memories()
        vocab_entries = extract_persona_vocabulary(
            og_memories_for_vocab,
            framework_baseline_names=framework_baseline_names,
        )
        _vocab_tmp = vocab_target.with_suffix(vocab_target.suffix + ".new")
        _vocab_tmp.write_text(
            _json.dumps({"version": 1, "emotions": vocab_entries}, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(_vocab_tmp, vocab_target)
        vocabulary_emotions_migrated = len(vocab_entries)
    except (ValueError, OSError) as exc:
        vocabulary_skipped_reason = f"migrate_error: {exc}"
```

(Note: `_json` and `os` are already imported in this file from prior migrator work. `reader` is the `OGReader` instance constructed at the top of `run_migrate`.)

Update the `MigrationReport(...)` constructor call (after the elapsed timing block) to pass the new fields:

```python
report = MigrationReport(
    ...,  # existing fields unchanged
    reflex_arcs_migrated=reflex_arcs_migrated,
    reflex_arcs_skipped_reason=reflex_arcs_skipped_reason,
    interests_migrated=interests_migrated,
    interests_skipped_reason=interests_skipped_reason,
    vocabulary_emotions_migrated=vocabulary_emotions_migrated,
    vocabulary_skipped_reason=vocabulary_skipped_reason,
)
```

- [ ] **Step 3: Add migrator regression test**

Read `tests/unit/brain/migrator/test_cli.py` first to find the existing fixture that builds a minimal valid OG source (e.g., `_make_minimal_og_source` or the `og_dir` fixture used by other migrator tests). Reuse that pattern.

Append this test:

```python
def test_migrate_writes_emotion_vocabulary(tmp_path: Path, monkeypatch):
    """Regression: migrator writes emotion_vocabulary.json from OG memory
    emotion references, with canonical entries for known nell_specific
    and placeholders for any custom emotions."""
    import json
    from brain.migrator.cli import MigrateArgs, run_migrate

    # Reuse existing fixture for minimal OG source
    source = _make_minimal_og_source(tmp_path)  # adjust if your fixture differs

    # Inject memories that reference 1 nell_specific + 1 custom emotion
    # (this depends on how the fixture seeds OG memories — adapt to its API)
    _seed_og_memory(source, emotions={"body_grief": 5.0, "moonache": 3.0})

    home = tmp_path / "home"
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))

    args = MigrateArgs(
        input_dir=source, output_dir=None,
        install_as="testpersona", force=False,
    )
    report = run_migrate(args)

    target = home / "personas" / "testpersona" / "emotion_vocabulary.json"
    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    names = {e["name"] for e in data["emotions"]}
    assert "body_grief" in names
    assert "moonache" in names
    assert report.vocabulary_emotions_migrated == 2

    # Canonical entry for body_grief
    body_grief = next(e for e in data["emotions"] if e["name"] == "body_grief")
    assert body_grief["decay_half_life_days"] is None
    assert "physical form" in body_grief["description"]

    # Placeholder for custom
    moonache = next(e for e in data["emotions"] if e["name"] == "moonache")
    assert moonache["decay_half_life_days"] == 14.0
    assert "migrated from OG" in moonache["description"]
```

If the `test_cli.py` fixture pattern doesn't expose a way to seed an extra memory mid-test, look at how `test_migrate_writes_interests` handled it (or `test_migrate_writes_reflex_arcs`) and follow that exact pattern. The OG memories.json fixture is what gets read by `reader.read_memories()`.

- [ ] **Step 4: Run all migrator tests**

Run: `uv run pytest tests/unit/brain/migrator/ -v`
Expected: all pass (existing + new vocabulary tests).

Full suite: `uv run pytest -q`
Expected: all green.

- [ ] **Step 5: Ruff + format**

Run: `uv run ruff check brain/migrator/ tests/unit/brain/migrator/ && uv run ruff format brain/migrator/ tests/unit/brain/migrator/`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add brain/migrator/cli.py brain/migrator/report.py tests/unit/brain/migrator/test_cli.py
git commit -m "$(cat <<'EOF'
feat(migrator): write per-persona emotion_vocabulary.json

Migrator now scans OG memories for emotion-name references, subtracts
the new framework baseline, and writes the remainder to
{persona_dir}/emotion_vocabulary.json with canonical definitions for
known nell_specific emotions and placeholder definitions for any
custom user-runtime emotions.

Atomic write via .new + os.replace. Refuse-to-clobber unless --force.
MigrationReport gains vocabulary_emotions_migrated +
vocabulary_skipped_reason. Format-report adds 'Vocabulary:' line.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: CLI handler integration

**Purpose:** Every CLI handler that opens a persona dir calls `load_persona_vocabulary(...)` before constructing engines, so persona-specific emotions are registered.

**Files:**
- Modify: `brain/cli.py`

- [ ] **Step 1: Add import**

Near the top of `brain/cli.py`, with the other `brain.*` imports:

```python
from brain.emotion.persona_loader import load_persona_vocabulary
```

- [ ] **Step 2: Wire each handler**

Find each handler that resolves `persona_dir` and constructs engines. The pattern in each is:

```python
def _heartbeat_handler(args: argparse.Namespace) -> int:
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(...)

    # ... opens MemoryStore, constructs engine, calls run_tick ...
```

Add the loader call **after** `MemoryStore` is opened (so the store-scan warning works) and **before** any engine is constructed. The exact location varies per handler. The injection is:

```python
load_persona_vocabulary(persona_dir / "emotion_vocabulary.json", store=store)
```

Apply to **every** handler:
- `_dream_handler`
- `_heartbeat_handler`
- `_reflex_handler`
- `_research_handler`
- `_interest_list_handler` — note: this handler doesn't open MemoryStore; pass `store=None` (the default) since there's no store to scan. The handler still benefits from registering vocab so `interest list` output (or future GUI) sees correct emotion names.
- `_interest_add_handler` — same as above, `store=None`
- `_interest_bump_handler` — same as above, `store=None`

For handlers WITHOUT a `MemoryStore`, the call shape is:

```python
load_persona_vocabulary(persona_dir / "emotion_vocabulary.json")
```

For handlers WITH a `MemoryStore`, the call shape is:

```python
load_persona_vocabulary(persona_dir / "emotion_vocabulary.json", store=store)
```

Place the call right after `persona_dir.exists()` check passes — before any other persona-touching work begins. For `MemoryStore`-using handlers, place it INSIDE the `try:` block that opens the store, immediately after `store = MemoryStore(...)`:

```python
store = MemoryStore(db_path=persona_dir / "memories.db")
try:
    load_persona_vocabulary(persona_dir / "emotion_vocabulary.json", store=store)
    # ... rest of handler unchanged ...
```

Use grep to find every persona-resolving handler: `rg 'def _.*_handler' brain/cli.py`.

- [ ] **Step 3: Run full suite**

Run: `uv run pytest -q`
Expected: all green. Existing CLI tests should not break — `load_persona_vocabulary` returns 0 silently when there's no vocabulary file (which is the case for the `:memory:` MemoryStore + tmp_path persona dirs in tests).

- [ ] **Step 4: Ruff + format**

Run: `uv run ruff check brain/cli.py && uv run ruff format brain/cli.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add brain/cli.py
git commit -m "$(cat <<'EOF'
feat(cli): wire persona vocabulary loader into every handler

Every CLI handler that opens a persona dir now calls
load_persona_vocabulary(persona_dir / "emotion_vocabulary.json", store=store)
before constructing engines. Handlers without MemoryStore pass
store=None (interest list/add/bump). Loader is silent for fresh
personas without a vocabulary file; warns once per missing-but-
referenced emotion when a pre-split persona runs without re-migration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Smoke test + final review + PR

**Purpose:** Verify everything works end-to-end against Nell's real migrated persona. Open PR.

**Files:** none created; verification only.

- [ ] **Step 1: Full suite + hard-rule gates**

Run: `uv run pytest -q`
Expected: all green, target ~460 tests.

Run: `rg -l 'import anthropic' brain/`
Expected: zero matches.

Run: `uv run ruff check && uv run ruff format --check`
Expected: clean.

- [ ] **Step 2: Confirm baseline shrunk**

Run:
```bash
uv run python -c "
from brain.emotion import vocabulary
print(f'baseline count: {len(vocabulary._BASELINE)}')
print(f'nell_specific in baseline: {[e.name for e in vocabulary._BASELINE if e.category == \"nell_specific\"]}')
"
```
Expected: `baseline count: 21` and `nell_specific in baseline: []`.

- [ ] **Step 3: Re-migrate Nell's sandbox**

Run: `uv run nell migrate --input /Users/hanamori/NellBrain/data --install-as nell.sandbox --force 2>&1 | tail -15`
Expected: report includes `Vocabulary:     5 emotions migrated` (or possibly more if her memories reference any beyond the 5 known nell_specific).

- [ ] **Step 4: Verify Nell's vocabulary file**

Run:
```bash
uv run python -c "
import json
from brain.paths import get_persona_dir
p = get_persona_dir('nell.sandbox')
with (p / 'emotion_vocabulary.json').open() as f:
    data = json.load(f)
print(f'emotions: {[e[\"name\"] for e in data[\"emotions\"]]}')
"
```
Expected: list contains at least `body_grief`, `creative_hunger` — verify the 5 canonical names appear (any extras are fine).

- [ ] **Step 5: Run a heartbeat tick — verify zero warnings**

Run: `uv run nell heartbeat --persona nell.sandbox --provider fake --searcher noop --trigger manual 2>&1`
Expected: full tick output with no warnings about unregistered emotions. The 5 nell_specific should now be loaded from her vocabulary file at startup.

- [ ] **Step 6: Confirm fresh persona path is silent**

Run:
```bash
mkdir -p /tmp/test_fresh_brain/personas/iris
cp ~/NellBrain-migrated/nell.sandbox/memories.db /tmp/test_fresh_brain/personas/iris/memories.db 2>/dev/null || \
  uv run python -c "
from pathlib import Path
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
p = Path('/tmp/test_fresh_brain/personas/iris')
p.mkdir(parents=True, exist_ok=True)
MemoryStore(db_path=p / 'memories.db').close()
HebbianMatrix(db_path=p / 'hebbian.db').close()
"
NELLBRAIN_HOME=/tmp/test_fresh_brain uv run nell heartbeat --persona iris --provider fake --searcher noop --trigger manual 2>&1
```
Expected: clean first-tick-defer output with no warnings about missing emotion vocabulary (fresh persona has no memories referencing nell-specific emotions).

Cleanup: `rm -rf /tmp/test_fresh_brain`

- [ ] **Step 7: Push branch + open PR**

```bash
git push -u origin vocabulary-split
gh pr create --title "Emotion vocabulary split (Phase 1)" --body "$(cat <<'EOF'
## Summary

Splits the 5 nell_specific emotions out of framework `_BASELINE` into per-persona `emotion_vocabulary.json`. Mirrors the pattern already used by reflex arcs and interests. Migrator scans OG memories so Nell, other OG framework users, and fresh users all work cleanly.

## What ships

- 5 emotions removed from framework `_BASELINE` (count: 26 → 21): `body_grief`, `emergence`, `anchor_pull`, `creative_hunger`, `freedom_ache`
- New `brain/emotion/persona_loader.py` — load + register + idempotent + memory-scan warning
- New `brain/migrator/og_vocabulary.py` — pure extractor: scans OG memories, subtracts baseline, writes canonical entries for known nell_specific + placeholders for custom user emotions
- New `brain/emotion/_canonical_personal_emotions.py` — module-private fixture used only by migrator
- Migrator writes `{persona_dir}/emotion_vocabulary.json` atomically; refuse-to-clobber unless `--force`
- Every CLI handler loads persona vocabulary at startup before engines run
- Backwards-compat: graceful degrade with one-time warning per missing emotion (silent for fresh users + re-migrated Nell)

## Three classes of users covered

- **Nell** — re-migrate, vocabulary file written, brain ticks normally with all 5 emotions
- **Other OG framework users** — same `nell migrate --force` command, their persona-specific emotions land in their persona file with canonical or placeholder definitions
- **Fresh new users** — create persona dir, run engines, baseline 21 emotions sufficient; vocabulary file optional

## Phase 2 — deferred

Autonomous emotion emergence (a crystallizer that mines memory patterns and proposes new emotion names for user approval) tracks alongside reflex Phase 2 and research Phase 2 — likely lands together as a unified weekly growth loop after ≥2 weeks of Phase 1 behavior data.

**Spec:** `docs/superpowers/specs/2026-04-25-vocabulary-split-design.md`
**Plan:** `docs/superpowers/plans/2026-04-25-vocabulary-split.md`

## Test plan

- [x] Full suite green (~460 tests, target +10 net)
- [x] `rg 'import anthropic' brain/` → 0 matches
- [x] ruff clean
- [x] `len(vocabulary._BASELINE) == 21`, `by_category("nell_specific") == []`
- [x] Re-migrating Nell writes 5 (or more) emotions to her vocabulary file
- [x] Heartbeat tick on Nell post-migration produces zero warnings
- [x] Fresh persona without vocabulary file runs cleanly with zero warnings

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Acceptance Criteria

The vocabulary split ships when all of the following are true:

1. `len(brain.emotion.vocabulary._BASELINE) == 21`
2. `brain.emotion.vocabulary.by_category("nell_specific") == []`
3. `brain/emotion/_canonical_personal_emotions.py` exists with all 5 canonical entries
4. `brain/emotion/persona_loader.py` exposes `load_persona_vocabulary(path, *, store=None) -> int` per spec §3.3
5. `brain/migrator/og_vocabulary.py` exposes `extract_persona_vocabulary(memories, *, framework_baseline_names) -> list[dict]`
6. Migrator writes `emotion_vocabulary.json` atomically; `MigrationReport` includes vocabulary fields; `format_report` shows the line
7. Every CLI handler loads persona vocabulary before constructing engines
8. Re-migrating Nell's sandbox writes ≥5 emotions to her vocabulary file
9. `nell heartbeat --persona nell.sandbox` post-migration produces zero warnings
10. Fresh persona without vocabulary file runs cleanly with zero warnings
11. `uv run pytest -q` green (target ~460)
12. `rg 'import anthropic' brain/` returns zero matches
13. `uv run ruff check && uv run ruff format --check` clean

---

## Deferred — Phase 2 reminder

Autonomous emotion emergence (see spec §13) is deferred. Future `brain/emotion/crystallizer.py` will mine memory patterns to propose new emotion names; user approves via Tauri GUI (or CLI fallback). Lands alongside reflex Phase 2 + research Phase 2 in a unified weekly growth loop. Prerequisite: ≥2 weeks of Phase 1 behavior data.
