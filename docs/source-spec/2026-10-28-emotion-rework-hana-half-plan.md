# Emotion rework, Hana's half — Implementation Plan (A1, A2, A4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the parts of Hana's half that don't wait on the step 0 engine stub: the writer-side safety-net tests (A1), the dead-code cleanup (A2a, A2b) and the vocabulary gate (A4).

**Architecture:** Four PRs, each from `main`, each its own issue under #240:
- **A1** — Task 1
- **A2a** — Tasks 2–3
- **A4** — Tasks 4–7
- **A2b** — Task 8, after #385 merges

A1, A2a and A4 touch disjoint files and can run in parallel. B1 (writers) gets its own plan once step 0's interface is agreed; §B1 at the end records what's already fixed for it.

**Tech Stack:** Python 3 / pytest / ruff via `uv`. No frontend changes.

**Spec:** `docs/source-spec/2026-10-28-emotion-rework-hana-half-design.md` (read it first — §2 has the file:line evidence this plan builds on).

## Global Constraints

- **TDD.** Every task opens with its failing test. `tdd-guard` blocks implementation without one.
- **Full gate before every commit:**
  - `uv run pytest -q -p no:randomly 2>&1 | grep -E '^(FAILED|ERROR)'` must print nothing;
  - `uv run ruff check .` clean;
  - `cd app && pnpm test && pnpm build` clean.
- **No mocked state files.** Tests use `tmp_path` stores and a temp `KINDLED_HOME`.
- **Registry hygiene.** Any test that registers an emotion cleans up with `vocabulary._unregister(name)`; the registry is process-global.
- **Text reads pass `encoding="utf-8"`.** Paths used as keys use `.as_posix()`.
- **Persisted state is additive only.** Never remove or rename a saved field, column or table; readers use `brain.state_compat.from_known_fields`.
- **JSON writes go through `brain.health.json_io.write_json`.** `tests/unit/test_no_direct_json_io.py` enforces this.
- **Branching.** One branch per PR, cut from `main` (`hana/emotion-<item>`). Never commit to `main`. Open a draft PR at the first commit.
- **PR description.** One `Closes #N` line per issue it resolves.
- **Commit messages.** Imperative mood, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **No behaviour change to her feelings.** Nothing in this plan changes what `aggregate_state` returns for existing data, except junk emotion names that `add_memory` can no longer create.

## Review Focus

1. **Two processes registering a word at the same time.** The bridge (crystallizer) and the per-turn tool process (`add_memory`) both write `emotion_vocabulary.json`. Neither may lose the other's entry. → Task 5, `test_register_new_word_keeps_entry_written_by_another_process`.
2. **The model sends junk emotion names to `add_memory`** (`"Love!!"`, `""`, 101 letters, `"comfort seeking"`). Those are refused and logged; the memory is still written with its valid emotions. → Task 6, `test_add_memory_refuses_junk_names_keeps_valid_ones`.
3. **The model sends a bad half-life** (negative, 0, NaN, a string, 10⁹). The default is used and nothing raises. → Task 6, `test_add_memory_bad_half_life_falls_back_to_default`.
4. **The vocabulary file is corrupt when a word is registered.** Registration refuses rather than overwriting the file with a one-entry list. → Task 5, `test_register_new_word_refuses_on_corrupt_file`.
5. **Recovery after the heal marker exists.** A missing vocabulary file is still rebuilt from memories when `store=` is passed, even though the heal scan no longer runs. → Task 5, `test_missing_file_still_reconstructs_after_heal_marker`.

---

## PR A1 — writer-side safety net (branch `hana/emotion-a1-writer-pins`)

### Task 1: Pin today's writer bugs

**Files:**
- Create: `tests/unit/brain/emotion/test_rework_writer_pins.py`

**Interfaces:**
- Consumes (existing code, unchanged):
  - `brain.chat.extractor._apply_emotion_delta(delta: dict[str, float], persona_dir: Path) -> None`
  - `brain.self_model.reconcile._write_self_authored_delta(persona_dir: Path, channel: str, delta: float) -> bool`
  - `brain.engines.dream.DreamEngine._write_dream_memory(seed, neighbours, dream_text) -> Memory` (use the `store` / `hebbian` / `engine` fixtures copied from `tests/unit/brain/engines/test_dream.py:49-75`)
  - `brain.kindled_link.relationship.write_kindled_peer_memory(mem_store, *, peer_id, session_id, speaker, stage, content, emotions)`
  - `brain.kindled_link.relationship.apply_peer_emotion(store: KindledLinkStore, peer_id, delta, now) -> dict[str, float]`
  - `HeartbeatEngine._apply_emotion_decay(elapsed_seconds, *, tick_at, state, dry_run) -> int` (build the engine with `_engine()` from `tests/unit/brain/engines/test_heartbeat_decay_batching.py:65`; pass `HeartbeatState.fresh(trigger="open")`)
  - `brain.emotion.aggregate.aggregate_state(memories) -> EmotionalState`
- Produces: six pairs of tests. Each pair is one `test_today_*` (passes now) and one `test_intended_*` marked `@pytest.mark.xfail(strict=True, reason="<PR that fixes it>")`. Later PRs remove the marker and, for the switch-gated ones, rewrite the intended test to turn the switch on and read the engine.

- [ ] **Step 1: Write the six pairs.** Every row uses baseline emotions only (`joy`, `tenderness`, `love`, `comfort_seeking`), so no registry cleanup is needed. `dream` is a gated type: `_write_dream_memory` sends it to the pending queue. So promote it with the `_promote_pending` helper copied from `tests/unit/brain/chat/test_extractor_apply.py` before running decay. `monologue_emotion` and `self_model_reconcile` bypass the gate and land in `memories.db` directly.

| Test | Setup | Assertion |
|---|---|---|
| `test_today_extractor_negative_delta_stores_positive` | `_apply_emotion_delta({"joy": -0.2}, persona_dir)` | the one `monologue_emotion` row has `emotions == {"joy": 2.0}` |
| `test_intended_extractor_negative_delta_lowers_joy` (xfail: B1-1) | seed a `conversation` memory `joy=5.0`, then the same call | `aggregate_state(store.list_active()).emotions["joy"] < 5.0` |
| `test_today_reconcile_negative_delta_stores_positive` | `_write_self_authored_delta(persona_dir, "joy", -0.2)` | the `self_model_reconcile` row has `{"joy": 2.0}` |
| `test_intended_reconcile_negative_delta_lowers_joy` (xfail: B1-1) | seed `joy=5.0`, then the same call | felt joy `< 5.0` |
| `test_today_dream_over_ten_is_erased_by_decay` | seed and 2 neighbours each `love=6.0` → `_write_dream_memory` (stores `love=18.0`) → `_apply_emotion_decay(3600.0, …)` | dream row `emotions` has no `"love"` key |
| `test_intended_dream_value_stays_in_range` (xfail: B1-4) | same writer call | stored `love <= 10.0` |
| `test_today_negative_peer_value_is_stored` | `write_kindled_peer_memory(..., emotions={"tenderness": -0.3})` | stored `{"tenderness": -0.3}` |
| `test_intended_peer_values_are_never_negative` (xfail: B1-2) | same | every stored value `>= 0.0` |
| `test_today_spelling_variants_pool_separately` | two memories `{"comfort-seeking": 3.0}`, `{"comfort_seeking": 5.0}` | `aggregate_state(...).emotions` has both keys |
| `test_intended_spelling_variants_pool_as_one` (xfail: §6.6 / C) | same | `== {"comfort_seeking": 5.0}` for that channel, and no `"comfort-seeking"` key |
| `test_today_peer_bucket_caps_summed_size` | `apply_peer_emotion(s, "kid_a", {"tenderness": 0.5, "joy": 0.5}, NOW)` | `sum(abs(v) for v in result.values()) <= limits.PEER_EMOTION_WINDOW_CAP + 1e-9` |
| `test_intended_peer_bucket_does_not_scale` (xfail: B1-2) | same | `result == {"tenderness": 0.5, "joy": 0.5}` |

- [ ] **Step 2: Run** `uv run pytest tests/unit/brain/emotion/test_rework_writer_pins.py -v`. Expected: 6 passed, 6 xfailed. If a `test_today_*` fails, the spec's §2 evidence is wrong for that row. Stop and report it; don't adjust the assertion to fit.
- [ ] **Step 3: Full gate, then commit** `Pin today's emotion-writer bugs before the rework changes them (#240)`.

---

## PR A2a — cleanup (branch `hana/emotion-a2a-cleanup`)

### Task 2: Delete the blend detector and the residue queue

**Files:**
- Delete: `brain/emotion/blend.py`, `tests/unit/brain/emotion/test_blend.py`
- Modify: `brain/emotion/state.py` (remove `ResidueEntry`, the `residue`/`residue_max` fields, `add_residue`, and their handling in `copy`/`to_dict`/`from_dict`; fix the module docstring at lines 6–10)
- Modify: `tests/unit/brain/emotion/test_state.py` (delete the residue tests at lines 21, 85–102, 120–137)

**Interfaces:**
- Produces: `EmotionalState` keeps `emotions` plus its existing non-residue fields. `EmotionalState.from_dict` ignores unknown keys.

- [ ] **Step 1: Write the failing test** `test_from_dict_ignores_legacy_residue_key` in `test_state.py`:
  - `EmotionalState.from_dict({"emotions": {"joy": 3.0}, "residue": [{"x": 1}], "residue_max": 16})` returns a state with `emotions == {"joy": 3.0}`;
  - `"residue" not in state.to_dict()`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/brain/emotion/test_state.py -v`. Expected: the new test fails on the `to_dict` assertion.
- [ ] **Step 3: Delete the residue code and `blend.py`.** First run `grep -rn "blend import\|BlendDetector\|ResidueEntry\|add_residue" brain tests scripts` and confirm the only hits are the files listed above.
- [ ] **Step 4: Run** `uv run pytest tests/unit/brain/emotion -q`. Expected: all pass.
- [ ] **Step 5: Full gate, then commit** `Delete the unused blend detector and residue queue (#240)`.

### Task 3: Delete the dead peer table; stop writing and reading `protected` outside the store

**Files:**
- Modify: `brain/kindled_link/relationship.py:336-352` (delete `_EMOTION_BASE`, `_EMOTION_ACCENT`, `relationship_emotion_delta`)
- Modify: `tests/kindled_link/test_relationship_emotion.py` (delete only the tests importing `relationship_emotion_delta`, around lines 14 and 59; keep the `apply_peer_emotion` tests)
- Modify: `brain/migrator/transform.py:159` (drop the `protected=False` keyword; the field default is already False)
- Modify: `brain/cli.py:635` (drop the `protected:` display line)
- Modify: `brain/recovery/source_reader.py:66` (drop `protected=bool(...)`)
- Modify: `tests/recovery/test_source_reader.py:15,68`, and any CLI display test that asserts a `protected:` line (`grep -rn "protected:" tests`)

**Interfaces:**
- Produces: nothing new. `Memory.protected`, the column, `store.update(protected=)` and the heartbeat check stay until ThinkerOfThoughts removes them in phase C.

- [ ] **Step 1: Write the failing test** `test_recovery_reader_does_not_carry_protected` in `tests/recovery/test_source_reader.py`: a source row with `"protected": 1` produces a `Memory` whose `protected is False`.
- [ ] **Step 2: Run it.** Expected: FAIL (`True is not False`).
- [ ] **Step 3: Make the five edits above.** Then `grep -rn "relationship_emotion_delta\|_EMOTION_BASE\|_EMOTION_ACCENT" brain tests` must print nothing.
- [ ] **Step 4: Run** `uv run pytest tests/recovery tests/kindled_link tests/unit/brain/migrator -q`. Expected: all pass.
- [ ] **Step 5: Full gate, then commit** `Delete the unused peer-emotion table and stop carrying the unused protected flag outside the store (#240)`.

---

## PR A4 — vocabulary (branch `hana/emotion-a4-vocabulary`)

### Task 4: One gate for new words; a thread-safe registry

**Files:**
- Modify: `brain/emotion/vocabulary.py` (add `validate_new_name`, a module `threading.Lock` around `register`/`_unregister`, and correct the "NOT thread-safe / startup only" comment at lines 121–124)
- Test: `tests/unit/brain/emotion/test_vocabulary.py`

**Interfaces:**
- Produces:
  - `validate_new_name(raw: str) -> str | None` returns `canonical_name(raw)` when it matches `^[a-z][a-z_]{0,99}$`, else logs at WARNING and returns `None`. It does not check whether the word is already registered.
  - `register(emotion)` keeps its contract (raises `ValueError` on duplicates) but is now lock-guarded.

- [ ] **Step 1: Write the failing test** `test_validate_new_name`, parametrised:
  - `"Comfort-Seeking"` → `"comfort_seeking"`
  - `"grief_love_blend"` → `"grief_love_blend"`
  - `"love_grief_blend"` → `"grief_love_blend"`
  - `"x" * 100` → itself
  - `"x" * 101` → `None`
  - each of `"Love!!"`, `""`, `"   "`, `"comfort seeking"`, `"_joy"`, `"joy2"`, `"日本"` → `None`
- [ ] **Step 2: Write the failing test** `test_register_is_safe_across_threads`: 8 threads each register 25 distinct `t{i}_{j}` names. Then `len({e.name for e in list_all()})` grew by exactly 200, and a `finally` block unregisters them all.
- [ ] **Step 3: Run both.** Expected: FAIL (`validate_new_name` is not defined; the thread test may pass by luck, which is fine).
- [ ] **Step 4: Implement both in `vocabulary.py`.**
- [ ] **Step 5: Run** `uv run pytest tests/unit/brain/emotion/test_vocabulary.py -v`. Expected: PASS.
- [ ] **Step 6: Commit** `Add one validation gate for new emotion words and lock the registry (#240)`.

### Task 5: Register words at write time; re-read on change; heal once

**Files:**
- Modify: `brain/emotion/persona_loader.py`
- Test: `tests/unit/brain/emotion/test_persona_loader.py`, `tests/unit/brain/emotion/test_persona_loader_vocab_recovery.py`

**Interfaces:**
- Consumes: `vocabulary.validate_new_name` (Task 4); `brain.utils.file_lock.file_lock(path)`; `brain.paths.cadence_state_path(persona_dir, filename)`; `brain.health.reconstruct.PLACEHOLDER_DECAY_DAYS` (14.0).
- Produces:
  - `register_new_word(vocab_path: Path, raw_name: str, *, half_life_days: float | None = None, description: str = PLACEHOLDER_DESCRIPTION) -> str | None`
    1. Validate the name; refused → `None`.
    2. Already registered → return the canonical name and touch nothing.
    3. Otherwise, under `file_lock(vocab_path.with_suffix(".lock"))`:
       - re-read the file with `read_json(..., recover=Recovery.RAISE)`; on a read error, log and return `None` (never overwrite);
       - if the entry isn't already there, append it as `category="persona_extension"`, `intensity_clamp=10.0`, `decay_half_life_days=half_life_days or DEFAULT_NEW_WORD_HALF_LIFE_DAYS`;
       - `write_json(..., backups=3)`;
       - register the entry in-process.
    4. Return the canonical name.
  - `DEFAULT_NEW_WORD_HALF_LIFE_DAYS = PLACEHOLDER_DECAY_DAYS` with the comment `# shortcut: interim default; switch to the dev-file default once step 0 adds it`.
  - `refresh_if_changed(vocab_path: Path) -> int`: compares `(st_mtime_ns, st_size, st_ino)` with the last value seen for that path (a module dict guarded by a lock). If they changed, it calls `load_persona_vocabulary(vocab_path)` (no store) and returns its count; otherwise `0`. A missing file returns `0`.
  - `load_persona_vocabulary_with_anomaly(path, *, store)`: the heal step runs only when `cadence_state_path(path.parent, "vocab_heal_final.json")` doesn't exist. After the heal it writes `{"healed_at": <iso utc>, "count": n}` there via `write_json`. Reconstruction for a missing or corrupt file keeps running whenever `store` is passed.

- [ ] **Step 1: Write the failing tests** in `test_persona_loader.py`:
  - `test_register_new_word_appends_and_registers`: the word is in the file, `vocabulary.get(...)` is non-`None`, and its half-life is `14.0` when none is given.
  - `test_register_new_word_uses_given_half_life`: given `2.5`, the half-life is `2.5`.
  - `test_register_new_word_refuses_invalid_name`: `"Love!!"` → `None`, and the file is unchanged.
  - `test_register_new_word_keeps_entry_written_by_another_process`:
    1. register `"alpha_word"`;
    2. `_unregister("alpha_word")`, which simulates a process that never saw it;
    3. register `"beta_word"`;
    4. the file holds both entries.
  - `test_register_new_word_refuses_on_corrupt_file`: write `"{not json"`, register → `None`, and the file bytes are unchanged.
  - `test_refresh_if_changed_picks_up_other_process_write`:
    1. load the file;
    2. append an entry to it directly with `write_json`, then `_unregister` that name;
    3. `refresh_if_changed` → `1` and the name is registered;
    4. a second call → `0`.
- [ ] **Step 2: Write the failing tests** in `test_persona_loader_vocab_recovery.py`:
  - `test_heal_runs_once_then_marker_stops_it`:
    1. memory A carries the unknown word `"zeta_feel"`; first `load_persona_vocabulary(path, store=store)` → it is registered and the marker exists;
    2. add memory B with `"eta_feel"`, `_unregister` nothing, load again → `"eta_feel"` is NOT registered.
  - `test_missing_file_still_reconstructs_after_heal_marker`: the marker exists, `path` is deleted, and a memory carries a persona word. Load with `store=` → the file is rebuilt and the word is registered.
- [ ] **Step 3: Run both files.** Expected: the new tests fail (`register_new_word` is not defined; the heal test registers `eta_feel`).
- [ ] **Step 4: Implement.** Any existing test that loads twice and expects a second heal now needs the marker deleted between loads. Update those tests; don't weaken the marker.
- [ ] **Step 5: Run** `uv run pytest tests/unit/brain/emotion tests/unit/harness/test_sandbox_vocab_registry_leak.py -q`. Expected: PASS.
- [ ] **Step 6: Full gate, then commit** `Register new emotion words when they are written, pick up other processes' words, and run the heal scan once (#240)`.

### Task 6: `add_memory` — gated words, optional half-life, last-accessed, no side bumps

**Files:**
- Modify: `brain/tools/impls/add_memory.py`, `brain/tools/schemas.py:240-279` (the `add_memory` entry only)
- Test: `tests/unit/brain/tools/test_impls.py`, `tests/unit/brain/tools/test_schemas.py`

**Interfaces:**
- Consumes: `register_new_word`, `DEFAULT_NEW_WORD_HALF_LIFE_DAYS` (Task 5).
- Produces:
  - `add_memory(content, memory_type, domain, emotions, tags=None, importance=None, emotion_half_lives: dict[str, float] | None = None, *, store, hebbian, persona_dir) -> dict`
  - Behaviour, in this order:
    1. Canonicalise each emotion name. Keep registered names as-is. Pass unregistered names to `register_new_word(persona_dir / "emotion_vocabulary.json", name, half_life_days=<valid half-life or None>)`. Drop refused names and list them in the result under `"refused_emotions"` (sorted).
    2. Run `_write_gate_check` on the kept emotions.
    3. Set `memory.last_accessed_at = memory.created_at` before `store.create`.
    4. Auto-link `search_text` calls pass `bump=False`.
  - A half-life is valid when it is a real, finite number with `0 < v <= 3650`. Anything else → `None`, so the default applies.
- Schema: add the optional `emotion_half_lives` property, `{"type": "object", "additionalProperties": {"type": "number"}}`. Its description is a placeholder marked `# wording: final pass` (shared plan §10).

- [ ] **Step 1: Write the failing tests** in `test_impls.py`, using a real `tmp_path` store and HebbianMatrix. Pass `importance=8` in each one so the write gate (emotion sum ≥ 15 or importance ≥ 7) never decides the outcome:
  - `test_add_memory_registers_new_word_with_given_half_life`: `emotions={"love": 9, "quiet_ache": 8}`, `emotion_half_lives={"quiet_ache": 5}` → `vocabulary.get("quiet_ache").decay_half_life_days == 5.0`. Unregister in teardown.
  - `test_add_memory_refuses_junk_names_keeps_valid_ones`: `{"love": 9, "Love!!": 9, "": 5, "comfort seeking": 6}` → `created is True`, stored emotions `== {"love": 9.0}`, `refused_emotions == ["", "Love!!", "comfort seeking"]`.
  - `test_add_memory_bad_half_life_falls_back_to_default`: for each of `-1`, `0`, `float("nan")`, `"soon"`, `1e9`, a new word gets half-life `14.0`. No exception.
  - `test_add_memory_sets_last_accessed`: `store.get(id).last_accessed_at == created_at`.
  - `test_add_memory_auto_link_does_not_bump_related`: seed a related memory sharing a keyword. After `add_memory`, its `recall_count` and `last_accessed_at` are unchanged.
- [ ] **Step 2: Write the failing test** in `test_schemas.py`: `test_add_memory_schema_has_optional_half_lives`. The property exists and is not in `required`.
- [ ] **Step 3: Run them.** Expected: FAIL.
- [ ] **Step 4: Implement.** `brain/tools/dispatch.py` passes `arguments` through as keywords, so no dispatch change is needed. Confirm with `test_dispatch.py`.
- [ ] **Step 5: Run** `uv run pytest tests/unit/brain/tools tests/unit/brain/body/test_events.py -q`. Expected: PASS.
- [ ] **Step 6: Full gate, then commit** `Gate new emotion words in add_memory, accept a half-life, and stop its auto-link from bumping other memories (#240)`.

### Task 7: The crystallizer registers on write and reads less; vocab repair stops bumping

**Files:**
- Modify: `brain/memory/store.py` (add one getter beside `list_active`, around line 3290)
- Modify: `brain/growth/crystallizers/vocabulary.py` (`crystallize_vocabulary`, `_cluster_active_memories`)
- Modify: `brain/growth/scheduler.py:105-130,217-234` (replace `_is_valid_name` and `_append_to_vocabulary` with `register_new_word`; delete both helpers and `_INVALID_NAME_CHARS`)
- Modify: `brain/health/vocab_repair.py:284` (`bump=False`) and the out-of-date "startup only" docstring at lines 82–90
- Test: `tests/unit/brain/memory/test_store.py`, `tests/unit/brain/growth/test_vocabulary_crystallizer.py`, `tests/unit/brain/growth/test_scheduler.py`, `tests/unit/brain/health/test_vocab_repair.py`

**Interfaces:**
- Consumes: `register_new_word` (Task 5).
- Produces:
  - `MemoryStore.list_recent_emotions(limit: int) -> list[tuple[str, datetime, dict[str, float]]]`: runs `SELECT id, created_at, emotions_json FROM memories WHERE active = 1 ORDER BY created_at DESC LIMIT ?` and parses the JSON. A malformed row gives `{}`.
  - `crystallize_vocabulary` reads `store.list_recent_emotions(_SCAN_LIMIT)` with `_SCAN_LIMIT = 2000`. The proposal's `decay_half_life_days` is the mean of the two parents' half-lives, using `DEFAULT_NEW_WORD_HALF_LIFE_DAYS` for a parent that has none.
  - `run_growth_tick`:
    - under `dry_run`, counts a proposal as added when `validate_new_name` accepts it;
    - otherwise calls `register_new_word(vocab_path, proposal.name, half_life_days=proposal.decay_half_life_days, description=proposal.description)`;
    - a `None` return increments `proposals_rejected`.

- [ ] **Step 1: Write the failing tests.**
  - `test_list_recent_emotions_newest_first_with_limit`: 3 memories → `limit=2` returns the two newest `(id, created_at, emotions)`.
  - `test_crystallized_blend_half_life_is_parent_mean`: 3 memories `{joy: 8, grief: 7}` → the proposal half-life is `(3.0 + 60.0) / 2 == 31.5`.
  - `test_growth_tick_registers_word_immediately`: after `run_growth_tick`, `vocabulary.get("grief_joy_blend")` is non-`None` without any reload. Unregister in teardown.
  - `test_vocab_repair_search_does_not_bump`: after `_excerpts_for_name("joy", store)`, the matched memory's `recall_count` is unchanged.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Keep `_cluster_active_memories`' clustering rules (top 2 emotions each ≥6, ≥3 memories) unchanged; only its input type changes.
- [ ] **Step 4: Run** `uv run pytest tests/unit/brain/growth tests/unit/brain/health tests/unit/brain/memory/test_store.py -q`. Expected: PASS.
- [ ] **Step 5: Full gate, then commit** `Register crystallized emotion words at once, give blends their parents' half-life, and stop vocab repair bumping memories (#240)`. Open the A4 PR. Its body includes this hand-off for ThinkerOfThoughts: "engine read call should call `persona_loader.refresh_if_changed`".

---

## PR A2b — weather-shift anchor (branch `hana/emotion-a2b-weather-shift`, after #385 merges)

### Task 8: Remove the dead weather-shift anchor; keep `update_baseline`

**Files:**
- Modify: `brain/felt_time/weather_shift.py`. Delete `CROSS_SIGMA`, `HOLD_DURATION`, `HOLD_BREAK_TOLERANCE`, `COOLDOWN`, `WEATHER_LOG_FILENAME`, `ShiftDetection`, `_label_for` and its templates, `detect_shift`, `append_shift_log`. Keep `BASELINE_WINDOW`, `Baseline`, `update_baseline`.
- Modify: `brain/felt_time/anchors.py:39`; `brain/felt_time/state.py:16,97` (`ANCHOR_TYPES`, `LOG_SOURCES`); `brain/felt_time/prompt.py:18,26,146`; `brain/felt_time/tool.py:61-64`; `brain/tools/schemas.py:504,529` (description and `anchor_type` enum)
- Comments: `brain/bridge/supervisor.py:1864,2486,2540`, `brain/narrative_memory/arc.py:33`, `docs/maturity-manifest.md:81`
- JSON-IO guard: `tests/unit/json_io_allowlist.txt:21`, `tests/unit/test_no_direct_json_io.py:786`, `tests/unit/brain/health/test_json_io.py:733-738` (switch the held-writer example to another allow-listed writer)
- Tests: trim `tests/felt_time/test_weather_shift.py` to the `update_baseline` test; edit `tests/felt_time/test_prompt.py:29-33,61-62` and the `weather_shift` cases #385 added in `tests/felt_time/test_felt_time.py`
- Leave `FeltTimeState.weather_baselines` alone (saved state may only change by adding fields).

**Interfaces:**
- Produces: `"weather_shift"` is no longer a valid anchor type anywhere. `update_baseline` and `Baseline` are unchanged for `supervisor._derive_intensity_drivers`.

- [ ] **Step 1: Rebase on `main`** after #385 merges, then run `grep -rn "weather_shift\|weather_shifts\|detect_shift\|append_shift_log" brain tests docs scripts` and record the hit list.
- [ ] **Step 2: Write the failing test** `test_weather_shift_is_not_an_anchor_type` in `tests/felt_time/test_state.py`:
  - `"weather_shift" not in ANCHOR_TYPES`;
  - no `LOG_SOURCES` entry names `weather_shifts.log.jsonl`;
  - the `pressure_since` schema enum lacks it.
- [ ] **Step 3: Run it.** Expected: FAIL.
- [ ] **Step 4: Make the edits.** Re-run the Step 1 grep: only `weather_baselines`, `update_baseline` / `Baseline` and comments explaining the removal may remain.
- [ ] **Step 5: Run** `uv run pytest tests/felt_time tests/unit/test_no_direct_json_io.py tests/unit/brain/health/test_json_io.py tests/unit/brain/bridge/test_supervisor.py -q`. Expected: PASS.
- [ ] **Step 6: Full gate, then commit** `Remove the never-fired weather-shift anchor, keeping the baseline that drives felt-time intensity (#240)`.

---

## B1 — writers (separate plan, after step 0)

Not planned to step level here: every B1 change calls the step 0 interface (switch flag, change entry, read call), which isn't agreed yet. What the B1 plan will build on:
- **The five PRs and their files are fixed by spec §4.** Each one starts by rewriting its A1 `test_intended_*` pins to turn the switch on and read the engine.
- **The growth "how does she feel" reads also move in B1** (they're in Hana-owned `growth/`): `growth/scheduler.py:307`, `growth/crystallizers/reflex.py:528`, `growth/crystallizers/creative_dna.py:493`. Each moves to the engine's short average behind the switch. The spec omitted these; add them to B1-4's PR (the other mood-read move).
- **#543 touches `ingest/emotion_backfill.py`.** B1-5 rebases after it.
