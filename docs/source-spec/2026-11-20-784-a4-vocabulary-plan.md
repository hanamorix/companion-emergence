# Emotion rework A4: vocabulary (#784) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the rest of A4. New emotion words pass one gate and register the moment they're written. The registry is thread-safe. Every process picks up another process's new words. The old heal scan runs once more, then never again.

**Architecture:**
- `brain/emotion/vocabulary.py` gains the gate (`validate_new_name`) and a registry lock.
- `brain/emotion/persona_loader.py` gains:
  - `register_new_word`, the one locked read-modify-write of `emotion_vocabulary.json` for new words;
  - `refresh_if_changed`, which Engine 7's `EmotionEngine.read()` already calls by name;
  - a one-shot marker that retires the heal scan.
- `add_memory` and the growth crystallizer call `register_new_word`. The crystallizer reads a bounded window of light `Memory` rows, so its faded-value ranking is kept.

**Tech Stack:** Python 3 (`uv`, pytest, ruff), SQLite (`MemoryStore`), the emotion engine.

**Spec:** `docs/source-spec/2026-10-28-emotion-rework-hana-half-design.md` §4 "A4 — vocabulary" (lands with #659). This plan replaces that design's plan Tasks 4-7, which were written against `44af2447`. It was re-verified on 2026-11-20 against ThinkerOfThoughts' Engine 7 head (#635, `e0840646`, which contains main @ `68247cd9`).

**Build after #635 merges.** Engine 7 rewrites the same functions (`persona_loader._heal_referenced_but_unregistered`, the crystallizer, `vocab_repair._excerpts_for_name`, `add_memory`) and makes `Memory.emotions` private. Start by merging main once #635 is in, then re-run the line checks in each task's Files list.

## Decisions (made while refreshing the plan, 2026-11-20; the owner or ThinkerOfThoughts can overturn any)

1. **Gate regex:** `^[a-z][a-z_]{0,99}$`, applied after `canonical_name`. The old Task 4's tests assume this, and it refuses a leading underscore; the design bullet's `^[a-z_]{1,100}$` is replaced.
2. **`add_memory` registers words only for a write that passes the write gate.** A refused write mints no vocabulary. The old plan registered before the gate.
3. **`refresh_if_changed` is read-only:** it calls `load_persona_vocabulary(path, read_only=True)` (`Recovery.RAISE`, no heal, quarantine or write), so `read()` keeps its "never writes" contract (#628 review). It records the file's stat key even when the read fails, except on an I/O fault, following `engine_files.VocabularyHalfLives._refresh`, so a corrupt file isn't re-parsed on every `read()`.
4. **One lock for the vocabulary file:** `file_lock(vocab_path)`, whose sidecar is `<path>.lock`. `register_new_word`, the crystallizer (through it) and `vocab_repair`'s rewrite all take it.
5. **Backups:** `register_new_word` writes with `compute_treatment(vocab_path.parent, vocab_path.name).backup_count`, as the scheduler's writer does today, so the file keeps its treatment's backup count.
6. **Default half-life for a new word:** `dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS` (14.0). The old plan's interim stub isn't needed.
7. **A blend's half-life:** the mean of its two parents' half-lives. A parent's half-life comes from `eng.half_life(parent)` when the crystallizer has an engine, and from `dev_constants.EMOTION_HALF_LIFE_DAYS.get(parent, dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS)` when it doesn't. So love is 730, not "none".
8. **The crystallizer reads the newest 2,000 active memories** (`_SCAN_LIMIT = 2000`) as light `Memory` objects (no embedding column), through a new `MemoryStore.list_recent_light(limit)`. It used to read every active memory. Older clusters are no longer scanned; that change is deliberate.
9. **The MCP process may now write the vocabulary file** (through `add_memory`). `brain/mcp_server/__init__.py`'s `_load_vocabulary` docstring, which says nothing is written on a turn's path, is corrected to name that one exception.

## Global Constraints

- TDD: one failing test at a time, seen failing, then the code. In a worktree, run red tests with `-o tdd_guard_project_root=<the main checkout>`.
- Full gate before each commit: `uv run pytest` with 0 `FAILED` (grep `^FAILED`), `uv run ruff check .`, `uv run python scripts/lint_windows_pitfalls.py`. No `app/` change.
- Read feelings through Engine 7's helpers (`brain.emotion.engine_utils.memory_feelings` / `feelings_of`, `MemoryStore.stored_emotion_names`), never `memory.emotions`.
- Text I/O with `encoding="utf-8"`. JSON through `brain.health.json_io` (`read_json`, `write_json`). Paths compared or used as keys via `.as_posix()`.
- Persisted state is additive. The marker `cadence/vocab_heal_final.json` is new, written through `brain.paths.cadence_state_path`.
- Registry hygiene: `tests/conftest.py`'s autouse `_vocabulary_registry_restore` restores `vocabulary._REGISTRY` after every test, so tests don't need to unregister.
- Commit messages: imperative, present tense, ending `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Each PR description has `Refs #240`, and the last one has `Closes #784`.

## Review Focus

1. **A corrupt vocabulary file seen by `read()`:** logged once, never quarantined or rewritten, and not re-parsed until it changes. Pinned in Task 2.
2. **Two processes adding different words at once:** both entries survive in the file. Pinned in Task 2 (`..._keeps_entry_written_by_another_process`).
3. **A fresh persona with no vocabulary file:** `register_new_word` creates it rather than raising `JsonFileNotFound`. Pinned in Task 2.
4. **`add_memory` with `emotion_half_lives` that isn't a dict** (the dispatcher doesn't type-check it): ignored, with the default half-life. Pinned in Task 3.
5. **A crystallized blend in a non-live persona:** still gets a parent-mean half-life from the dev table, and the faded-value ranking tests still pass. Pinned in Task 4.

---

### Task 1: One gate for new words; a thread-safe registry

**Files:**
- Modify: `brain/emotion/vocabulary.py`: `import threading` (~19-21); `_REGISTRY_LOCK = threading.Lock()` beside `_REGISTRY` (~126); correct the "NOT thread-safe / startup only" comment (~122-125); `validate_new_name` beside `canonical_name` (~132); the bodies of `register` (~175-185) and `_unregister` (~188-193) under the lock.
- Test: `tests/unit/brain/emotion/test_vocabulary.py`

**Interfaces:**
- Produces: `validate_new_name(raw: str) -> str | None`. It returns `canonical_name(raw)` when that matches `^[a-z][a-z_]{0,99}$`, and otherwise logs a WARNING and returns `None`. It doesn't check registration. `register(emotion)` keeps its contract (`ValueError` on a duplicate canonical name), now under the lock.

- [ ] **Step 1: Write the failing test** `test_validate_new_name`, parametrised:
  - `"Comfort-Seeking"` → `"comfort_seeking"`
  - `"grief_love_blend"` → `"grief_love_blend"`
  - `"love_grief_blend"` → `"grief_love_blend"`
  - `"x" * 100` → itself
  - `"x" * 101` → `None`
  - each of `"Love!!"`, `""`, `"   "`, `"comfort seeking"`, `"_joy"`, `"joy2"` and `"日本"` → `None`
- [ ] **Step 2: Run it**: FAIL (`validate_new_name` not defined). Implement it. Run it again: PASS.
- [ ] **Step 3: Write** `test_register_is_safe_across_threads`: 8 threads each register 25 distinct `t{i}_{j}` names, and `len(list_all())` grows by exactly 200. It may pass before the lock exists; add the lock anyway, and say in the commit body that it was green on its first run.
- [ ] **Step 4: Run** `uv run pytest tests/unit/brain/emotion/test_vocabulary.py -q`: PASS. Full gate. Commit: `add one gate for new emotion words and lock the registry (#784)`.

### Task 2: `register_new_word`, `refresh_if_changed`, and the heal run once

**Files:**
- Modify: `brain/emotion/persona_loader.py`:
  - new functions after `load_persona_vocabulary` (~161);
  - the heal call in `load_persona_vocabulary_with_anomaly` (~130-131).
- Modify: `brain/mcp_server/__init__.py` `_load_vocabulary` docstring (~67-69), per Decision 9.
- Test: `tests/unit/brain/emotion/test_persona_loader.py`, `tests/unit/brain/emotion/test_persona_loader_vocab_recovery.py`

**Interfaces:**
- Consumes: `vocabulary.validate_new_name` (Task 1).
- Uses as they are:
  - `brain.utils.file_lock.file_lock(path)`;
  - `brain.paths.cadence_state_path(persona_dir, filename)`;
  - `brain.health.json_io.read_json` / `write_json` / `JsonFileError`;
  - `brain.health.adaptive.compute_treatment`;
  - `brain.utils.file_ops.stat_path(path) -> PathStat | None` (`.key`);
  - this module's `_default_vocab_factory`, `_vocab_schema_validator` and `_entry_to_emotion`.
- Produces:
  - `DEFAULT_NEW_WORD_HALF_LIFE_DAYS: float = dev_constants.EMOTION_DEFAULT_HALF_LIFE_DAYS`.
  - `register_new_word(vocab_path: Path, raw_name: str, *, half_life_days: float | None = None, description: str = PLACEHOLDER_DESCRIPTION) -> str | None`:
    1. A name the gate refuses → `None`.
    2. A name already registered → its canonical name, and nothing is touched.
    3. Otherwise, under `file_lock(vocab_path)`:
       - re-read the file with `read_json(vocab_path, default=_default_vocab_factory, recover=Recovery.RAISE, expect=dict, validate=_vocab_schema_validator)`. On `JsonFileError`, log and return `None` without overwriting;
       - append the entry if it isn't there: `category="persona_extension"`, `intensity_clamp=10.0`, `decay_half_life_days=half_life_days or DEFAULT_NEW_WORD_HALF_LIFE_DAYS`;
       - `write_json` with the treatment's backup count (Decision 5);
       - register the word in-process.
    4. Return the canonical name.
  - `refresh_if_changed(vocab_path: Path) -> int`:
    - **Unchanged:** if `stat_path(vocab_path)` gives the same key as last time for that path (a module dict under a module lock), return `0`.
    - **Changed:** call `load_persona_vocabulary(vocab_path, read_only=True)`, return its count, and record the key.
    - **Read fails:** record the key all the same (except on an I/O fault) and re-raise; `read()` logs it once.
    - **Missing file:** record `None` and return `0`.
  - In `load_persona_vocabulary_with_anomaly`, `_heal_referenced_but_unregistered` runs only while `cadence_state_path(path.parent, "vocab_heal_final.json")` is absent. After it runs, `write_json` writes `{"healed_at": <iso utc>, "count": n}` there. Reconstruction of a missing or corrupt file still runs whenever `store=` is passed.

- [ ] **Step 1: One test at a time, each red then green,** in `test_persona_loader.py`:
  - `test_register_new_word_appends_and_registers`: the word is in the file, `vocabulary.get(word)` is not `None`, and its half-life is `14.0`.
  - `test_register_new_word_uses_given_half_life`: `2.5` → `2.5`.
  - `test_register_new_word_refuses_invalid_name`: `"Love!!"` → `None`, with the file unchanged.
  - `test_register_new_word_creates_the_file_for_a_fresh_persona` (Review Focus 3): no file → the file is created holding the word.
  - `test_register_new_word_keeps_entry_written_by_another_process` (Review Focus 2):
    1. register `"alpha_word"`;
    2. `vocabulary._unregister("alpha_word")`;
    3. register `"beta_word"`;
    4. the file holds both entries.
  - `test_register_new_word_refuses_on_corrupt_file`: `"{not json"` → `None`, with the bytes unchanged.
  - `test_refresh_if_changed_picks_up_other_process_write`:
    1. load the file;
    2. append an entry with `write_json`, then `_unregister` it;
    3. `refresh_if_changed` → `1`, and the word is registered;
    4. a second call → `0`.
  - `test_refresh_if_changed_never_writes_a_corrupt_file` (Review Focus 1):
    - write `"{not json"`;
    - `refresh_if_changed` raises `JsonFileError`, the bytes are unchanged and no quarantine copy exists;
    - a second call with the file unchanged returns `0` without re-reading (patch `load_persona_vocabulary` to count calls).
- [ ] **Step 2: Same pattern,** in `test_persona_loader_vocab_recovery.py`:
  - `test_heal_runs_once_then_marker_stops_it`: memory A carries `"zeta_feel"`, and the first load with `store=` registers it and writes the marker; memory B carries `"eta_feel"`, and the next load doesn't register it.
  - `test_missing_file_still_reconstructs_after_heal_marker`: the marker exists and the file is deleted; a load with `store=` rebuilds the file and registers the memory's word.
- [ ] **Step 3: Correct the MCP docstring** (Decision 9). Run `uv run pytest tests/unit/brain/emotion tests/unit/brain/mcp_server tests/unit/harness/test_sandbox_vocab_registry_leak.py tests/bridge/test_engine_start_order.py tests/bridge/test_supervisor_rebuild.py -q`: PASS. Also run `tests/unit/brain/emotion/test_engine_read.py`: its `refresh_if_changed` contract tests (~195-240) must still pass. Full gate. Commit: `register new emotion words when written, pick up other processes' words read-only, and run the heal scan once (#784)`.

### Task 3: `add_memory` gates and registers new words, with an optional half-life

**Files:**
- Modify: `brain/tools/impls/add_memory.py`:
  - signature (~16-27);
  - the new step between `tags = …` (~47) and `_write_gate_check` (~49);
  - registration after the gate passes (Decision 2).
- Modify: `brain/tools/schemas.py`, `SCHEMAS["add_memory"]` (~300-339): `emotion_half_lives` after `importance`, not in `required`.
- Test: `tests/unit/brain/tools/test_impls.py`, `tests/unit/brain/tools/test_schemas.py`

**Interfaces:**
- Consumes: `register_new_word` and `DEFAULT_NEW_WORD_HALF_LIFE_DAYS` (Task 2); `vocabulary.validate_new_name` (Task 1); `vocabulary.pool_by_canonical_name`.
- Produces: `add_memory(content, memory_type, domain, emotions, tags=None, importance=None, emotion_half_lives: dict[str, float] | None = None, *, store, hebbian, persona_dir) -> dict`. In order:
  1. Canonicalise and fold spellings with `pool_by_canonical_name`. Drop names `validate_new_name` refuses, and list them under `"refused_emotions"` (sorted raw names).
  2. Run `_write_gate_check` on the kept emotions.
  3. Only if the write passes, call `register_new_word(persona_dir / "emotion_vocabulary.json", name, half_life_days=<valid or None>)` for each unregistered kept name.
  4. Then build and create the memory, as today.
- A half-life is valid when it is a real finite number with `0 < v <= 3650`. An `emotion_half_lives` that isn't a dict is ignored.
- The schema property is `{"type": "object", "additionalProperties": {"type": "number"}}`, with a description marked `# wording: final pass`.

- [ ] **Step 1: One test at a time, each red then green,** using a `tmp_path` store and HebbianMatrix with `importance=8`:
  - `test_add_memory_registers_new_word_with_given_half_life`: `emotions={"love": 9, "quiet_ache": 8}` and `emotion_half_lives={"quiet_ache": 5}` → `vocabulary.get("quiet_ache").decay_half_life_days == 5.0`.
  - `test_add_memory_refuses_junk_names_keeps_valid_ones`: `{"love": 9, "Love!!": 9, "": 5, "comfort seeking": 6}` → `created is True`, `memory_feelings(store.get(id, bump=False), None, None, None).feelings == {"love": 9.0}`, and `refused_emotions == ["", "Love!!", "comfort seeking"]`.
  - `test_add_memory_bad_half_life_falls_back_to_default`: each of `-1`, `0`, `float("nan")`, `"soon"` and `1e9` gives the new word `14.0`, and so does `emotion_half_lives="soon"` (Review Focus 4). No exception.
  - `test_add_memory_refused_write_registers_no_word` (Decision 2): `importance=None` and `emotions={"quiet_ache": 3}` (below the gate) → `created is False`, and `vocabulary.get("quiet_ache") is None`.
  - `test_add_memory_schema_has_optional_half_lives` (in `test_schemas.py`): the property exists and isn't in `required`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/brain/tools tests/unit/brain/body/test_events.py -q`: PASS. Confirm `test_dispatch.py` passes unchanged. Full gate. Commit: `gate and register new emotion words in add_memory, with an optional half-life (#784)`.

### Task 4: The crystallizer registers on write and reads a bounded window; vocab repair stops bumping

**Files:**
- Modify: `brain/memory/store.py`: `list_recent_light` beside `list_active` (~3916), modelled on `list_engine_window` (~3297) and `_ENGINE_WINDOW_COLUMNS` (~3280).
- Modify: `brain/growth/crystallizers/vocabulary.py`:
  - `crystallize_vocabulary` (~36-81): read `store.list_recent_light(_SCAN_LIMIT)`;
  - the proposal half-life per Decision 7;
  - `_cluster_active_memories` keeps `feelings_of(memory, eng)`.
- Modify: `brain/growth/scheduler.py`:
  - the proposal loop (~118-152): `register_new_word` instead of `_is_valid_name` + `_append_to_vocabulary`, and under `dry_run` count a proposal that `validate_new_name` accepts;
  - delete `_INVALID_NAME_CHARS`, `_is_valid_name` and `_append_to_vocabulary`;
  - update the module docstring (~1-16), which still describes vocab repair as startup-only.
- Modify: `brain/health/vocab_repair.py`:
  - `_excerpts_for_name` (~286): `search_text(name, limit=10, bump=False)`;
  - `_atomic_write_vocab`'s rewrite under `file_lock(vocab_path)` (Decision 4);
  - the `should_run_vocab_repair` docstring (~84-93): startup plus every 6 hours.
- Test: `tests/unit/brain/memory/test_store.py`, `tests/unit/brain/growth/test_vocabulary_crystallizer.py`, `tests/unit/brain/growth/test_scheduler.py`, `tests/unit/brain/health/test_vocab_repair.py`

**Interfaces:**
- Consumes: `register_new_word` (Task 2); `vocabulary.validate_new_name` (Task 1).
- Produces:
  - `MemoryStore.list_recent_light(limit: int) -> list[Memory]`: active memories, newest `created_at` first, at most `limit`, with no embedding.
  - `_SCAN_LIMIT = 2000` in the crystallizer.
  - In `run_growth_tick`, a `None` from `register_new_word` increments `proposals_rejected`.

- [ ] **Step 1: One test at a time, each red then green:**
  - `test_list_recent_light_newest_first_with_limit`: 3 memories, `limit=2` → the two newest, with feelings readable through `memory_feelings`.
  - `test_crystallized_blend_half_life_is_parent_mean`: 3 memories `{joy: 8, grief: 7}` with no engine → half-life `(3.0 + 60.0) / 2 == 31.5`.
  - `test_crystallized_blend_with_love_uses_its_dev_half_life` (Review Focus 5): `{love: 8, joy: 7}` → `(730.0 + 3.0) / 2 == 366.5`.
  - `test_growth_tick_registers_word_immediately`: after `run_growth_tick`, `vocabulary.get("grief_joy_blend")` isn't `None` without a reload.
  - `test_vocab_repair_search_does_not_bump`: after `_excerpts_for_name("joy", store)`, the matched memory's `recall_count` is unchanged.
- [ ] **Step 2:** The existing faded-value tests in `test_scheduler.py` (`test_old_memories_do_not_form_a_blend_when_live`, `test_top_two_are_chosen_by_faded_value`) and `test_run_growth_tick_rejects_proposal_with_invalid_chars` must pass unchanged.
- [ ] **Step 3: Run** `uv run pytest tests/unit/brain/growth tests/unit/brain/health tests/unit/brain/memory/test_store.py -q`: PASS. Full gate. Commit: `register crystallized words at once, give blends their parents' half-life, read a bounded window, and stop vocab repair bumping (#784)`.

## After the last task

Mark #784 done in the hana-half design's Status section (#659 or its successor). Then check `grep -rn "validate_new_name\|register_new_word\|refresh_if_changed" brain`: `add_memory` and the scheduler are the only writers that call the gate, and `read()` and the loader are the only callers of `refresh_if_changed`.
