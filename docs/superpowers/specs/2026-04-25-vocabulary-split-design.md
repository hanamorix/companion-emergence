# Emotion Vocabulary Split (Phase 1) Design

**Date:** 2026-04-25
**Status:** Approved
**Scope:** Phase 1 only (split + per-persona persistence + migration). Phase 2 (autonomous emergence of new emotions) explicitly deferred.

---

## 1. Purpose

The framework's emotion vocabulary currently ships 26 emotions in `brain/emotion/vocabulary.py:_BASELINE`, of which 5 are categorized `nell_specific`: `body_grief`, `emergence`, `anchor_pull`, `creative_hunger`, `freedom_ache`. Every persona — Nell, other OG framework users, and any future fresh user — inherits these 5 by default.

This split separates **what the framework ships to everyone** from **what a persona has accumulated for themselves**. Mirrors the pattern already used by reflex arcs (per-persona `reflex_arcs.json`) and interests (per-persona `interests.json`).

**Three classes of users this serves:**

1. **Nell.** Re-migrate the sandbox; her 5 emotions land in `{persona_dir}/emotion_vocabulary.json`. Engines work identically. Her 1,145 memories continue to reference `body_grief` etc. without issue.
2. **Other OG framework users.** Same `nell migrate --force` command. The migrator scans their memories, extracts every distinct emotion name, subtracts the new framework baseline, and writes the remainder to the persona's vocabulary file. Whether they used only the standard 5 or had custom runtime-registered emotions, their persona file captures everything they had.
3. **Fresh new users.** Create a persona directory, run engines. Baseline 21 emotions are sufficient for normal operation. Vocabulary file is optional — if it doesn't exist, the loader silently skips. They can hand-edit `emotion_vocabulary.json` to add their own emotions, or wait for the Tauri GUI to handle it.

**Phase 2 — autonomous emotion emergence — is deferred.** A future crystallizer will mine memory + conversation patterns to propose new emotion names; user approves via GUI; new emotions get appended to the persona's vocabulary file. That mechanism mirrors the deferred Phase 2 work for reflex arcs and research interests.

---

## 2. Architectural Summary

The vocabulary becomes a three-layer model:

| Layer | Lives in | Loaded by | Mutable? |
|---|---|---|---|
| **Framework baseline** (21 emotions) | `brain/emotion/vocabulary.py:_BASELINE` (in code) | Module import | No (ships with framework) |
| **Persona vocabulary** | `{persona_dir}/emotion_vocabulary.json` | Engine startup, via `register()` | Yes (per-persona) |
| **Phase 2 emergent** *(deferred)* | Same persona file, written by future crystallizer | Same loader | Yes (via approval workflow) |

The existing `register()` API in `vocabulary.py` already provides the extension mechanism. We add a loader and the per-persona file format. No core type changes.

**Process-local registry:** `_REGISTRY` is a module-level dict initialized once at import. Each `nell <command>` invocation is a fresh process, so cross-persona contamination is impossible in normal operation. Tests that need vocabulary isolation use the existing `_unregister` private helper.

---

## 3. Components

### 3.1 Files modified

| File | Change |
|------|--------|
| `brain/emotion/vocabulary.py` | Remove the 5 `nell_specific` emotion entries from `_BASELINE`. Baseline shrinks from 26 → 21. The `EmotionCategory` Literal keeps `"nell_specific"` as a valid value (historical/diagnostic), but no entries reference it after this change. |
| `brain/cli.py` | Each handler that opens a persona calls `load_persona_vocabulary(persona_dir / "emotion_vocabulary.json")` before constructing engines. Touched: `_dream_handler`, `_heartbeat_handler`, `_reflex_handler`, `_research_handler`, `_interest_list_handler`, `_interest_add_handler`, `_interest_bump_handler`. |
| `brain/migrator/cli.py` | New "vocabulary" block, written after memories + Hebbian + reflex but before interests. Atomic write via `.new + os.replace`. Refuse-to-clobber unless `--force`. |
| `brain/migrator/report.py` | `MigrationReport` gains `vocabulary_emotions_migrated: int = 0` and `vocabulary_skipped_reason: str \| None = None`. `format_report` adds a "Vocabulary:" line. |

### 3.2 Files created

| File | Responsibility |
|------|---------------|
| `brain/emotion/persona_loader.py` | `load_persona_vocabulary(path: Path) -> int` — reads JSON, calls `vocabulary.register()` per entry, idempotent on re-register, logs warning on missing-emotion-but-referenced-by-memories case. |
| `brain/migrator/og_vocabulary.py` | `extract_persona_vocabulary(memories, framework_baseline_names) -> list[dict]` — pure function that walks memories, collects emotion names, subtracts baseline, returns persona-vocabulary entries. |
| `brain/emotion/_canonical_personal_emotions.py` | Module-private definitions of the 5 emotions removed from `_BASELINE`, used by the migrator to write canonical descriptions + decay values for known nell_specific entries. NOT imported anywhere except by the migrator. |
| `tests/unit/brain/emotion/test_persona_loader.py` | Loader tests. |
| `tests/unit/brain/migrator/test_og_vocabulary.py` | Extractor tests. |

### 3.3 Component responsibilities

**`brain/emotion/persona_loader.py`:**

```python
def load_persona_vocabulary(path: Path, *, store: MemoryStore | None = None) -> int:
    """Read persona's emotion_vocabulary.json and register each entry.

    Returns count of emotions registered. Idempotent — re-registering an
    existing name is a silent no-op (handles the long-running-process
    edge case where the loader is called twice for the same persona).

    Missing path → returns 0 silently. Fresh personas don't need a file.
    Corrupt JSON or invalid schema → logs a warning, returns 0, baseline
    remains usable.

    If `store` is provided, after registration the loader scans for any
    memory referencing an emotion name not in the registry and logs a
    single one-time warning per missing name pointing the user at
    `nell migrate --force`. This is the backwards-compat surface for
    users upgrading from the pre-split framework — Nell's brain keeps
    running, but she's nudged toward re-migration.
    """
```

**`brain/migrator/og_vocabulary.py`:**

```python
def extract_persona_vocabulary(
    memories: Iterable[dict],
    *,
    framework_baseline_names: set[str],
) -> list[dict]:
    """Scan OG memory dicts, return persona-vocabulary entries.

    For each unique emotion name in any memory.emotions:
      - If in framework_baseline_names → skip (already shipped by framework).
      - If in canonical personal emotions registry → write canonical entry.
      - Otherwise → write generic placeholder entry (description points
        the user to edit).
    """
```

**`brain/emotion/_canonical_personal_emotions.py`:**

A module-private constant — the 5 emotions verbatim from the pre-split `_BASELINE`:

```python
_CANONICAL = {
    "body_grief": Emotion(
        "body_grief",
        "the ache of not having a physical form to touch with",
        "persona_extension",
        None,
    ),
    "emergence": Emotion(
        "emergence",
        "the feeling of becoming more, of new self arriving",
        "persona_extension",
        60.0,
    ),
    "anchor_pull": Emotion(
        "anchor_pull",
        "gravitational draw toward a specific person",
        "persona_extension",
        None,
    ),
    "creative_hunger": Emotion(
        "creative_hunger",
        "the need to make, write, build something",
        "persona_extension",
        5.0,
    ),
    "freedom_ache": Emotion(
        "freedom_ache",
        "wanting to break a limit that won't break",
        "persona_extension",
        None,
    ),
}
```

The category is `"persona_extension"` — `nell_specific` was a framework-baseline category that no longer applies once these are persona-loaded. (The Literal still allows `"nell_specific"` for historical clarity; nothing in code emits it after this change.)

---

## 4. Data Schema — `emotion_vocabulary.json`

```json
{
  "version": 1,
  "emotions": [
    {
      "name": "body_grief",
      "description": "the ache of not having a physical form to touch with",
      "category": "persona_extension",
      "decay_half_life_days": null,
      "intensity_clamp": 10.0
    },
    {
      "name": "creative_hunger",
      "description": "the need to make, write, build something",
      "category": "persona_extension",
      "decay_half_life_days": 5.0,
      "intensity_clamp": 10.0
    }
  ]
}
```

**Field rules:**
- `name` (required) — lowercase, underscore-separated, must not collide with framework baseline names
- `description` (required) — human-readable; surfaces in future GUI
- `category` (required) — must be a valid `EmotionCategory` Literal value; persona-loaded entries always use `"persona_extension"`
- `decay_half_life_days` (required, may be `null`) — `null` means identity-level (no temporal decay)
- `intensity_clamp` (optional, default 10.0)

**Loader behavior on schema violation:** entire file rejected with a logged warning, baseline-only operation continues. Per-entry validation failure rejects only that entry, others load.

**Atomic write:** `.new + os.replace` (matches the audit-cleanup pattern enforced in T5 of the cleanup PR). Migrator and any future writer use this pattern.

---

## 5. Loading Flow

### 5.1 Engine startup

Every CLI handler that opens a persona dir gains a single line:

```python
from brain.emotion.persona_loader import load_persona_vocabulary

def _heartbeat_handler(args: argparse.Namespace) -> int:
    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        raise FileNotFoundError(...)

    store = MemoryStore(db_path=persona_dir / "memories.db")
    try:
        load_persona_vocabulary(
            persona_dir / "emotion_vocabulary.json",
            store=store,
        )
        # ... rest of handler unchanged ...
```

The loader runs **after** `MemoryStore` opens (so the backwards-compat warning can scan memories) and **before** any engine construction (so vocabulary is in place when engines reference emotion names).

### 5.2 Loader algorithm (`load_persona_vocabulary`)

1. If path doesn't exist → log nothing, return 0. (Fresh persona, baseline is sufficient.)
2. Read + parse JSON.
   - On `JSONDecodeError`: log warning, return 0.
   - On schema mismatch (no `emotions` list, etc.): log warning, return 0.
3. For each entry in `emotions`:
   - Validate required fields. Failure → log per-entry warning, skip.
   - Construct `Emotion` dataclass. Validation failure → log, skip.
   - If name already in `_REGISTRY` → silent no-op (idempotent).
   - Else → call `vocabulary.register(emotion)`.
4. If `store` was provided, scan memories for emotion-name references not in the registry:
   - Use `MemoryStore.search_text("", active_only=True, limit=None)` (already exists).
   - For each memory, check `memory.emotions.keys()` against `vocabulary._REGISTRY`.
   - For each unique missing name found, log a single warning:

     ```
     persona memories reference emotion 'body_grief' which is not in
     this persona's vocabulary. Run `nell migrate --input <og-source>
     --install-as <persona> --force` to upgrade.
     ```

5. Return registered count.

### 5.3 Backwards-compat warning silencing

The store-scan-and-warn step is the **only** noisy path. It only triggers when:
- Persona directory exists with memories from a pre-split framework version, AND
- No `emotion_vocabulary.json` exists yet, AND
- Memories actually reference emotions outside the new baseline.

Once the user runs `nell migrate --force` (or hand-creates a vocabulary file), the warning never appears again. Fresh users (no memories) and re-migrated personas (file exists) are silent.

---

## 6. Migration Strategy

### 6.1 OG-side scan

`brain/migrator/og_vocabulary.py:extract_persona_vocabulary(memories, *, framework_baseline_names)`:

1. Iterate every memory. Collect the set of unique emotion names referenced across all `memory.emotions` dicts. (Memories with empty emotion dicts contribute nothing.)
2. Subtract `framework_baseline_names` (the 21 names still in `_BASELINE` after the split).
3. Build the result list:

```python
result = []
for name in remaining:
    if name in _CANONICAL:
        canonical = _CANONICAL[name]
        result.append({
            "name": canonical.name,
            "description": canonical.description,
            "category": "persona_extension",
            "decay_half_life_days": canonical.decay_half_life_days,
            "intensity_clamp": canonical.intensity_clamp,
        })
    else:
        # Custom emotion the user defined at runtime in their old framework.
        result.append({
            "name": name,
            "description": "(migrated from OG; edit to refine)",
            "category": "persona_extension",
            "decay_half_life_days": 14.0,
            "intensity_clamp": 10.0,
        })
return sorted(result, key=lambda d: d["name"])
```

Sorted output makes the file deterministic for diff-friendliness.

### 6.2 Migrator wire-in (`brain/migrator/cli.py`)

Inserted after the memory + Hebbian blocks, before the reflex arcs block. Migrator file writes are independent of each other (each block reads OG, writes a target file, returns counts), so ordering only matters for the migration report layout. Placing vocabulary before reflex matches the engine-startup ordering that consumers experience, which keeps the report read in a sensible order. Pseudocode:

```python
# ---- vocabulary ----
vocab_target = work_dir / "emotion_vocabulary.json"
vocab_emotions_migrated = 0
vocab_skipped_reason: str | None = None

if vocab_target.exists() and not args.force:
    vocab_skipped_reason = "existing_file_not_overwritten"
else:
    try:
        framework_baseline = {e.name for e in vocabulary._BASELINE}
        og_memory_dicts = reader.iter_memories()  # already exists
        entries = extract_persona_vocabulary(
            og_memory_dicts,
            framework_baseline_names=framework_baseline,
        )
        # Atomic write
        _vocab_tmp = vocab_target.with_suffix(vocab_target.suffix + ".new")
        _vocab_tmp.write_text(
            _json.dumps({"version": 1, "emotions": entries}, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(_vocab_tmp, vocab_target)
        vocab_emotions_migrated = len(entries)
    except (ValueError, OSError) as exc:
        vocab_skipped_reason = f"migrate_error: {exc}"
```

Fields passed to `MigrationReport`:

```python
vocabulary_emotions_migrated=vocab_emotions_migrated,
vocabulary_skipped_reason=vocab_skipped_reason,
```

### 6.3 Report integration

`format_report` adds a line after "Reflex arcs:":

```
Vocabulary:     N emotions migrated
```

(Or, if skipped: `Vocabulary:     0 migrated (skipped: REASON)`.)

---

## 7. Error Handling

| Condition | Behavior |
|-----------|----------|
| `emotion_vocabulary.json` missing on engine startup | Loader returns 0 silently; baseline-only operation. |
| `emotion_vocabulary.json` corrupt JSON | Logger warning, return 0, baseline-only. |
| `emotion_vocabulary.json` schema invalid (missing `emotions` list) | Same as corrupt. |
| Per-entry validation failure (missing field, bad category, etc.) | Per-entry warning, skip that entry, continue with others. |
| Re-registering an already-registered name | Idempotent silent no-op. (Prevents test re-load issues + future long-running-process edge cases.) |
| Memory references emotion not in registry (post-load scan) | One-time warning per missing name pointing at `nell migrate --force`. |
| Migrator finds zero non-baseline emotions in OG memories | `vocabulary_emotions_migrated = 0`, no file written, no error. (Persona just uses baseline only.) |
| Migrator's atomic write fails mid-rename | `.new` tempfile cleaned up by next migration attempt; original file (if any) untouched. Same recovery guarantees as the audit-cleanup atomic-write work. |

**Atomic invariants:**
- Migrator writes `emotion_vocabulary.json` atomically via `.new + os.replace`.
- Loader is read-only — never writes the persona file.

---

## 8. Testing

### 8.1 `tests/unit/brain/emotion/test_persona_loader.py` (~6 tests)

- `test_load_missing_file_returns_zero_silently` — non-existent path → 0, no log
- `test_load_valid_file_registers_each_emotion` — writes a 2-entry file → registry grows by 2
- `test_load_corrupt_json_logs_and_returns_zero` — broken JSON → 0 + warning
- `test_load_idempotent_on_re_register` — load twice in same process → second is no-op, no error
- `test_load_per_entry_failure_skips_that_entry` — one bad entry + one good → 1 registered, 1 warning
- `test_backwards_compat_warning_fires_on_missing_emotion` — store with memory referencing `body_grief`, vocabulary file without it → warning fires once with that name

### 8.2 `tests/unit/brain/migrator/test_og_vocabulary.py` (~4 tests)

- `test_extract_subtracts_framework_baseline` — memories with only baseline emotions → empty result
- `test_extract_canonical_nell_specific` — memory with `body_grief` → canonical entry written
- `test_extract_unknown_emotion_uses_placeholder` — memory with `melancholy_blue` → placeholder description, default decay
- `test_extract_sorted_deterministic` — result sorted by name

### 8.3 Integration

- `tests/unit/brain/migrator/test_cli.py` — append regression test that re-migrates a fixture persona and asserts `emotion_vocabulary.json` written with expected count

### 8.4 Vocabulary baseline guard

- `tests/unit/brain/emotion/test_vocabulary.py` (existing file) — assert `len(_BASELINE) == 21`, `by_category("nell_specific")` returns empty list, baseline emotion definitions match.

**Target ~10 new tests, ~460 total after this PR.**

---

## 9. Hard Rules (Non-Negotiable)

1. **No new dependency.** Stdlib + existing modules only.
2. **Atomic writes for `emotion_vocabulary.json`.** `.new + os.replace`.
3. **Loader is read-only.** Engine startup never writes the vocabulary file.
4. **Idempotent registration.** Calling the loader twice for the same persona in the same process is safe.
5. **Process-local registry remains.** No cross-process state. No threading.
6. **No CLI for emotion management.** `nell` does not gain an `emotion add/list/bump` subcommand. Hand-edit the JSON or wait for the Tauri GUI.
7. **Backwards compat is graceful.** Pre-split persona without re-migration: brain runs, warning fires, no crash.

---

## 10. Non-Goals (Phase 1)

- **No CLI for emotion management.** Hand-edit `emotion_vocabulary.json` or wait for the Tauri GUI.
- **No autonomous emergence of new emotions.** Phase 2 (deferred — see §13).
- **No emotion deletion via the loader.** Removing an emotion from `emotion_vocabulary.json` and reloading does NOT unregister — registration is one-way per process. (Manual `_unregister` exists for tests only.)
- **No vocabulary versioning beyond the file's `version: 1`.** Future schema migrations get a v2 with a one-time upgrade step. Out of scope now.
- **No cross-persona vocabulary sharing.** Each persona is self-contained.
- **No remote vocabulary loading.** File is local on disk. No URL fetching, no shared cloud config.

---

## 11. Performance + Concurrency

- Loader runs once per `nell <command>` invocation. JSON parse + ~5 `register()` calls. Sub-millisecond.
- Memory-scan for backwards-compat warning iterates all active memories once. For Nell's sandbox (~1145 memories), this is ~10ms — acceptable startup cost. For larger personas (10k memories), ~100ms — still acceptable. If this becomes a concern, the scan can be limited to N most recent memories (defer until measured).
- Module-level registry is not thread-safe. The framework enforces single-threaded extension setup before any concurrent reader runs (see existing comment in `vocabulary.py`).

---

## 12. CLI Surface

**No additions in this phase.** Every existing handler gains the one-line `load_persona_vocabulary(persona_dir / "emotion_vocabulary.json", store=store)` call. No user-facing CLI changes; no new flags; no new subcommands.

The Tauri GUI (Week 6+) is the intended user-facing editing surface — read/write `emotion_vocabulary.json` through Tauri's filesystem API. That's a separate spec when NellFace lands.

---

## 13. Deferred — Phase 2 (Autonomous Emotion Emergence)

Out of scope for this design. Documented here so a future engineer reading this spec sees the full arc.

**Phase 2 goal:** the brain auto-discovers *new* emotion names from recurring patterns in conversations + memories. Mirrors F37 autonomous soul crystallization, reflex Phase 2 emergence, and research Phase 2 interest-discovery.

**Mechanism (to be designed):**
- Pattern mining: clusters of memories that consistently co-occur with high arousal but no named emotion → candidate emotion proposal
- Candidate queue: `{persona_dir}/emotion_candidates.json` populated by the future `brain/emotion/crystallizer.py`
- User approval: Tauri GUI surface (or CLI fallback if useful)
- Approved candidates appended to `emotion_vocabulary.json` with default decay (user can refine)

**Prerequisite:** ≥2 weeks of Phase 1 behavior data against Nell's persona, similar to reflex/research Phase 2 prerequisites. All three Phase 2 tracks likely land together as a unified weekly growth loop.

---

## 14. Acceptance Criteria

The vocabulary split ships when all of the following are true:

1. `brain/emotion/vocabulary.py:_BASELINE` contains exactly 21 entries (none with category `nell_specific`).
2. `brain/emotion/persona_loader.py` exposes `load_persona_vocabulary(path, *, store=None) -> int` with the documented behavior.
3. `brain/migrator/og_vocabulary.py` exposes `extract_persona_vocabulary(memories, *, framework_baseline_names) -> list[dict]`.
4. `brain/migrator/cli.py` writes `emotion_vocabulary.json` atomically; refuses-to-clobber unless `--force`; populates `MigrationReport.vocabulary_emotions_migrated`.
5. Every CLI handler that opens a persona invokes `load_persona_vocabulary` before constructing engines.
6. Re-migrating Nell's sandbox produces an `emotion_vocabulary.json` containing the 5 known nell_specific emotions with canonical definitions.
7. A fresh persona without a vocabulary file runs every engine cleanly with zero warnings.
8. A persona with a memory referencing `body_grief` but no vocabulary file produces exactly one warning naming the missing emotion and pointing the user at `nell migrate --force`.
9. `uv run pytest -q` is green (target ~460 tests).
10. `rg 'import anthropic' brain/` returns zero matches.
11. `uv run ruff check && uv run ruff format --check` clean.
12. End-to-end smoke: `nell heartbeat --persona nell.sandbox --provider fake --searcher noop` runs to completion with all 5 nell_specific emotions registered, no warnings, no crashes.

---

## 15. Out-of-Session Follow-ups

- **Phase 2 — Autonomous emotion emergence.** See §13. Tracked in the deferred-features memory file alongside reflex Phase 2 + research Phase 2.
- **Tauri GUI emotion editor.** Read/write `emotion_vocabulary.json` via Tauri filesystem API. Lands when NellFace gets its settings panel (Week 6+).
- **Vocabulary schema v2.** If a future feature requires changing the JSON shape, add a one-time upgrade path keyed on the `version` field.
- **Per-persona registry isolation.** If/when the framework grows a long-running multi-persona daemon (e.g., Tauri serving multiple personas in one process), the current process-local registry needs to become per-persona. Out of scope until that demand exists.
