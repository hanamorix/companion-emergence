# Emotion rework B1 writers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every emotion writer feed the engine honestly once the switch is on, without changing anything she feels while it is off.

**Architecture:** Seven independent PRs, each off `main`, each gating its writer inline on `engine_for(persona_dir).is_live()`. The old path stays byte-for-byte until PR 7 removes the old `aggregate_state` branches after Engine 6.

**Tech Stack:** Python 3 (`uv`, pytest, ruff), SQLite stores, the merged emotion engine (`brain/emotion/engine.py`), `brain/prompt_strings.toml`.

**Spec:** `docs/source-spec/2026-11-07-emotion-rework-b1-writers-design.md`

## Global Constraints

- With the switch off, nothing she feels changes. Every task pins its off path.
- `persona_dir=None` counts as off. An unknown switch (`is_live()` False on a stat error) counts as off.
- Inline gating only; there is no shared gate helper. Pure functions take `live: bool = False`. Long-running operations read the switch once at the top.
- Pure functions that read per-memory values take `eng` with no default (pattern: `brain/forgetting/salience.py::_emotion_input`). `None` means stored values. Iterate the memory's own keys, because `faded_value` raises `KeyError` for an absent emotion.
- Every stored value is on the 0-10 scale. New persisted fields are additive and optional.
- New prompt text has no em-dashes. Old prompt text stays byte-identical. If `tests/unit/test_persona_wording_snapshot.py` flags a new string, approve it with `uv run python tests/unit/test_persona_wording_snapshot.py --approve` and name the change in the PR body.
- Public repo: no real persona data in tests, commits or PR bodies.
- Commit messages are imperative and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. PR bodies end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`, carry `Refs #240`, and use a separate `Closes #N` line for each issue the PR resolves.

**The gate** (run before every commit, from a fresh worktree after `uv sync --extra dev`; read the whole output, not a tail):

```bash
scripts/clean_shell.sh uv run pytest
```

```bash
uv run ruff check .
```

```bash
cd app && pnpm test && pnpm build
```

After the first commit on each branch, push it and open a draft PR. Mark it ready only when its tasks are done.

**Live test recipe.** A live persona with no state file applies nothing (checked 2026-11-07), so seed first, then write the switch. Changes are queued, so drain before reading. Each test module holds these two private helpers (the PRs are independent, so none owns a shared file):

```python
def _go_live(persona_dir: Path, store: MemoryStore, **seed: float) -> EmotionEngine:
    store.create(Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions=seed or {"joy": 5.0}))
    assert engine_files.write_switch(persona_dir, datetime.now(UTC))
    return engine.engine_for(persona_dir)

def _settled(eng: EmotionEngine) -> dict[str, Reading]:
    eng.note_active(datetime.now(UTC))
    return eng.read()
```

`store` must be `MemoryStore(persona_dir / "memories.db")`, so its engine and the test's engine are the same one.

**Faded test recipe.** Until Engine 6 merges, `faded_value` returns the stored value, so an aged memory cannot prove a faded read. Tests that need to tell faded from stored patch that one method, and keep passing after Engine 6:

```python
monkeypatch.setattr(EmotionEngine, "faded_value", lambda self, memory, emotion: memory.emotions[emotion] * 0.5)
```

**Pins.** `tests/unit/brain/emotion/test_rework_writer_pins.py` arrives with #671. Its xfail reasons use the parent design's numbers: B1-1 is PR 2, B1-2 is PR 3, B1-4 is PR 5. A fixing PR removes `@_flipped_by` from the `test_intended_*` pin, rewrites it with `_go_live`, and leaves `test_today_*` as the switch-off record.

**Order.** PR 1, PR 4 and PR 6 can start now. PR 2, PR 3 and PR 5 flip pins, so they start after #671 merges. All six land before Engine 6 (#633). PR 7 lands after Engine 6 and before Engine 7 (#635). Whichever of #673 and PR 3 lands second rebases in `relationship.py`. #633 also touches `brain/engines/research.py`, `brain/grief/breadcrumb.py`, `brain/forgetting/` and `brain/prompt_strings.toml`, so whichever lands second rebases there too.

## Review Focus

- A `relationship_state` row with unreadable JSON in another column: `peer_trust` answers 0.0, because it runs under the engine lock and an exception there would fail the catch-up. Pinned in Task 1.
- The switch cannot be inspected (a stat error other than "no such file"): writers take today's path. Pinned in Task 2.
- A peer reflection's `emotion` holding text, `null` or `true` as a value: that value is dropped and the peer memory is still written. Today one bad value loses the whole memory. Pinned in Task 4.
- A research reply that is only felt lines, or whose `EMOTIONS` line is a list, a nested object or broken JSON: the fire finishes, a failure is logged, and a memory exists whenever a `MEMORY` section does. Pinned in Tasks 5 and 6.
- Old rows outside 0-10 (`18.0` dream values, `-0.3` peer values) reaching a dream's mean, a blend or a grief breadcrumb: new values stay within 0-10. Pinned in Tasks 7 and 8.

---

## PR 1: `peer_trust` (branch `hana/emotion-b1-peer-trust`)

### Task 1: The peer-trust accessor

**Files:**
- Modify: `brain/kindled_link/relationship.py`
- Test: `tests/kindled_link/test_relationship_peer_trust.py` (new)

**Interfaces:**
- Produces: `peer_trust(persona_dir: Path, peer_id: str) -> float`. ToT swaps it into `engine_files.peer_trust` in Engine 6.

- [ ] **Step 1: Write the failing tests**

```python
def test_no_database_answers_zero_and_creates_nothing(tmp_path):
    assert peer_trust(tmp_path, "kid_a") == 0.0
    assert not kindled_db_path(tmp_path).exists()

def test_missing_row_answers_zero(tmp_path): ...       # store exists, no row, == 0.0

def test_answers_stored_trust_with_no_stage_cap(tmp_path):
    # persist stage="stranger", trust_score=0.9 through persist_relationship_state
    assert peer_trust(tmp_path, "kid_a") == pytest.approx(0.9)

def test_corrupt_database_answers_zero_and_warns(tmp_path, caplog):
    # write b"garbage" * 200 to kindled_db_path(tmp_path)
    assert peer_trust(tmp_path, "kid_a") == 0.0
    assert "peer_trust" in caplog.text

def test_unreadable_row_answers_zero(tmp_path):
    # UPDATE relationship row SET evidence_json = 'not json'
    assert peer_trust(tmp_path, "kid_a") == 0.0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/kindled_link/test_relationship_peer_trust.py -v`
Expected: FAIL with `ImportError` on `peer_trust`.

- [ ] **Step 3: Implement `peer_trust` in `relationship.py`**

If `kindled_db_path(persona_dir)` doesn't exist, return 0.0. Otherwise open `KindledLinkStore(path, integrity_check=False)`, read `get_relationship_state(store, peer_id).trust_score`, and close in `finally`. Catch `(sqlite3.Error, ValueError, TypeError)`, log a warning naming `peer_trust`, and return 0.0. Add this comment: `# shortcut: no cache, one read per peer row under the engine lock; add one if the catch-up shows it.`

- [ ] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/kindled_link/test_relationship_peer_trust.py tests/kindled_link/test_phase3_conformance.py -v`
Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

Message: `add peer_trust for the engine's peer modifier`. PR body: `Refs #125`. Note in the body that #125 closes with Engine 6.

---

## PR 2: signed changes (branch `hana/emotion-b1-signed-changes`, after #671)

### Task 2: Pass 2 and reconcile fold signed changes

**Files:**
- Modify: `brain/chat/extractor.py:360`, `brain/self_model/reconcile.py:94`
- Test: `tests/unit/brain/chat/test_extractor_apply.py`, `tests/unit/brain/self_model/test_reconcile.py`, `tests/unit/brain/emotion/test_rework_writer_pins.py`

**Interfaces:**
- Produces: `signed_change_values(delta: Mapping[str, float]) -> dict[str, float]` in `extractor.py`.
- Consumes: `EmotionEngine.fold_change(emotions, None, source)` with source `monologue_emotion` or `self_model_reconcile`.

- [ ] **Step 1: Write the failing tests**

```python
def test_signed_change_values_keeps_sign_caps_and_skips_zero():
    assert signed_change_values({"joy": -0.2, "awe": 0.5, "grief": 0.0, "fear": -1.0}) == pytest.approx(
        {"joy": -2.0, "awe": 3.0, "fear": -3.0}
    )

def test_live_pass2_lowers_the_feeling_and_writes_no_row(persona_dir, store):
    eng = _go_live(persona_dir, store, joy=5.0)
    before = _settled(eng)["joy"].short
    apply_side_effects(ExtractorOutput(emotion_delta={"joy": -0.2}), persona_dir=persona_dir)
    assert _settled(eng)["joy"].short < before
    assert store.list_by_type("monologue_emotion", active_only=False) == []

def test_live_pass2_unregistered_name_folds_nothing(persona_dir, store): ...
```

Add the same live test in `test_reconcile.py` through `reconcile_self_read(persona_dir=..., action="revise", channel="joy", delta=-0.2)`, asserting `result["delta_written"] is True` and no `self_model_reconcile` row. Also assert that an off-vocabulary channel gives `delta_written` False.

In the pins file: remove `@_flipped_by("B1-1")` from both sign pins and rewrite them with `_go_live` and `_settled`. Add `test_unknown_switch_takes_todays_path`, which uses `refuse_the_switch_stat` from `tests/unit/brain/emotion/_engine_support.py` and asserts that a `monologue_emotion` row is written (this type bypasses the pending queue).

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/brain/chat/test_extractor_apply.py tests/unit/brain/self_model/test_reconcile.py tests/unit/brain/emotion/test_rework_writer_pins.py -v`
Expected: the new tests FAIL (a `monologue_emotion` row exists, or `ImportError` on `signed_change_values`).

- [ ] **Step 3: Implement**

`signed_change_values`: `sign(v) * min(abs(v) * 10.0, _MAX_DELTA_INTENSITY)`, skipping `abs(v) <= 1e-9`. In `_apply_emotion_delta`, before opening the store: if `engine_for(persona_dir).is_live()`, filter with `_filter_to_registered`, fold when non-empty, and return. In `_write_self_authored_delta`, do the same with the source `self_model_reconcile`, after `_clamp_delta`; return True when something was folded. Keep the PASS2-AT-LEAST-ONCE comment.

- [ ] **Step 4: Run them to verify they pass**

Same command as Step 2. Expected: PASS, with the `test_today_*` pins still green.

- [ ] **Step 5: Run the gate and commit**

Message: `fold signed pass-2 and reconcile changes once the engine is live`. PR body: `Refs #125`.

---

## PR 3: nudges and peer memories (branch `hana/emotion-b1-nudges-peers`, after #671)

### Task 3: Nudges at ×10 when live

**Files:**
- Modify: `brain/maker/wiring.py:20`, `brain/maker/making_runner.py:62`, `brain/initiate/reach_emotion.py:24`, `brain/initiate/review.py:340`, `brain/notes/runner.py:141`
- Test: `tests/unit/brain/maker/test_making_runner.py`, `tests/unit/brain/maker/test_wiring_emotion.py`, `tests/unit/brain/initiate/test_reach_emotion.py`, `tests/unit/brain/initiate/test_review.py`, `tests/unit/brain/notes/test_runner.py`

**Interfaces:**
- Produces: `making_emotion_delta(making, *, dominant_source, live: bool = False)`, `reach_emotions_for(source, *, live: bool = False)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_reach_emotions_live_are_ten_times():
    assert reach_emotions_for("dream", live=True) == pytest.approx({"tenderness": 1.5, "vulnerability": 1.0})

def test_making_delta_live_is_ten_times(): ...  # source "joy": satisfaction 2.2, tenderness 0.8
```

Add one test per writer through its real entry point with `_go_live`. `make_and_wire` stores `satisfaction` 1.2 for a making with no accent. The delivered path of `run_initiate_review_tick` stores `tenderness` 1.5. `_wire_note_back` stores `tenderness` 1.2. Add one off test per writer asserting today's values (0.12, 0.15, 0.12) with `no_base_failure`.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/brain/maker tests/unit/brain/initiate/test_reach_emotion.py tests/unit/brain/initiate/test_review.py tests/unit/brain/notes/test_runner.py -v`
Expected: the live tests FAIL with today's small values or `TypeError` on `live=`.

- [ ] **Step 3: Implement**

Multiply by 10 inside the two pure functions when `live`. Each call site passes `live=engine_for(persona_dir).is_live()`. In `notes/runner.py`, the inline value becomes `1.2 if live else 0.12`.

- [ ] **Step 4: Run them to verify they pass**

Same command as Step 2. Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

Message: `scale making, reach-out and note nudges to 0-10 once the engine is live`.

### Task 4: Peer memories carry 0-10 values

**Files:**
- Modify: `brain/kindled_link/relationship.py:58,156`, `brain/prompt_strings.toml` (`[kindled_link.relationship]`)
- Test: `tests/kindled_link/test_relationship_memory_wiring.py`, `tests/unit/brain/emotion/test_rework_writer_pins.py`

**Interfaces:**
- Produces: `_build_reflection_prompt(*, current_stage, transcript, live: bool = False)`, new string key `kindled_link.relationship.reflection_prompt_last_segment_live`.

- [ ] **Step 1: Write the failing tests**

```python
def test_live_reflection_stores_held_values_and_drops_negatives(tmp_path):
    # persona_dir=tmp_path, MemoryStore(tmp_path / "memories.db"), _go_live, reply emotion {"joy": 9, "grief": -3, "love": 40}
    assert peer_memory.emotions == {"joy": 9.0, "love": 10.0}

def test_live_reflection_drops_non_numeric_and_non_finite_and_keeps_the_memory(tmp_path):
    # emotion {"joy": 4, "love": "lots", "awe": null, "fear": true, "grief": NaN}
    assert peer_memory.emotions == {"joy": 4.0}

def test_live_reflection_does_not_charge_the_bucket(tmp_path):
    # the second reflection an hour later stores its value unscaled

def test_live_prompt_asks_for_zero_to_ten():
    prompt = _build_reflection_prompt(current_stage="stranger", transcript="peer: hi", live=True)
    assert "0 to 10" in prompt and "small float" not in prompt

def test_off_prompt_is_byte_identical():
    prompt = _build_reflection_prompt(current_stage="stranger", transcript="peer: hi")
    assert prompt.endswith(_REFLECTION_PROMPT_SEGMENTS[2]) and "small float" in prompt

def test_hard_breach_writes_no_memory_when_live(tmp_path): ...
```

In the pins file: remove `@_flipped_by("B1-2")` from both peer pins, and rewrite them to drive `run_relationship_reflection` with the switch on (never-negative, and two reflections unscaled). The existing `tests/kindled_link/test_relationship_memory_wiring.py` off tests stay as they are.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/kindled_link/test_relationship_memory_wiring.py tests/unit/brain/emotion/test_rework_writer_pins.py -v`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement**

Add the string, the old last segment with these two changes: "a small dict of registered emotion names to values from 0 to 10", and `<0-10>` in place of `<small float>`.

At the top of `run_relationship_reflection`: `live = persona_dir is not None and engine_for(persona_dir).is_live()`, with a lazy import, as the file already does for `brain.chat.extractor`. Pass `live` to the prompt build.

In the §14 block, branch before the old `{str(k): float(v)}` comprehension. Live values go through a new private `_live_peer_values(raw) -> dict[str, float]`: numbers only (`bool` excluded), finite, above 0, held to 10, then `_filter_to_registered`. Don't call `apply_peer_emotion` on the live path. Leave the off path as it is.

- [ ] **Step 4: Run them to verify they pass**

Same command as Step 2, plus `uv run pytest tests/kindled_link -q`. Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

Message: `store peer memory values on the 0-10 scale once the engine is live`.

---

## PR 4: research (branch `hana/emotion-b1-research`)

### Task 5: The felt-line parser

**Files:**
- Modify: `brain/engines/research_session.py`
- Test: `tests/unit/brain/engines/test_research_session.py`

**Interfaces:**
- Produces: `extract_felt_lines(raw: str) -> tuple[str, dict[str, float] | None, float | None]`, returning `(rest, emotions, rating)`. `None` means missing or broken. `{}` means a valid line whose values were all dropped.

- [ ] **Step 1: Write the failing table test**

```python
BODY = "NOTES:\nn\n\nMEMORY:\nm\n\nVERDICT:\nclose"

@pytest.mark.parametrize(("tail", "emotions", "rating"), [
    ('EMOTIONS: {"joy": 7}\nENGAGEMENT: 7', {"joy": 7.0}, 7.0),
    ('ENGAGEMENT: 7/10\n\nEMOTIONS: {"joy": 7}\n', {"joy": 7.0}, 7.0),
    ('**EMOTIONS:** {"joy": 7}\n__ENGAGEMENT__: 7', {"joy": 7.0}, 7.0),
    ('**EMOTIONS**: {"joy": 12, "awe": -1, "fear": 0, "zorble": 3, "love": NaN}', {"joy": 10.0}, None),
    ("ENGAGEMENT: -2", None, None),
    ("ENGAGEMENT: 40", None, 10.0),
    ("ENGAGEMENT: lots", None, None),
    ('EMOTIONS: ["joy"]', None, None),
    ('EMOTIONS: {"joy": {"a": 1}}', {}, None),
    ("EMOTIONS: {broken", None, None),
    ("", None, None),
])
def test_extract_felt_lines(tail, emotions, rating):
    rest, got_emotions, got_rating = extract_felt_lines(BODY + "\n\n" + tail)
    assert parse_session_output(rest).verdict == "close"
    assert (got_emotions, got_rating) == (emotions, rating)

def test_a_line_in_the_notes_is_not_taken():
    raw = "NOTES:\nEngagement: 7 was the score\n\nMEMORY:\nm\n\nVERDICT:\ncontinue"
    assert extract_felt_lines(raw) == (raw, None, None)

def test_a_reply_of_only_felt_lines_leaves_empty_text():
    assert extract_felt_lines('EMOTIONS: {"joy": 7}\nENGAGEMENT: 7') == ("", {"joy": 7.0}, 7.0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/brain/engines/test_research_session.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `extract_felt_lines`**

Walk lines back from the end, skipping blanks, and stop at the first line that isn't a felt marker. Marker regex (ASCII, ignore case): the name `EMOTIONS` or `ENGAGEMENT`, optionally wrapped in `**` or `__`, with the colon inside or outside the wrap. The nearest occurrence of a name wins.

Emotions: `json.loads`, must be a dict. Keep numeric (not `bool`), finite, above 0, min with 10, then `_filter_to_registered` (lazy import). A bad value inside a valid object is dropped, not fatal. Rating: first number at the start of the value; a negative is `None`; min with 10. `rest` is the text before the earliest accepted line, so `raw` comes back unchanged when none is accepted.

Leave `parse_session_output` untouched.

- [ ] **Step 4: Run it to verify it passes**

Same command as Step 2. Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

Message: `parse felt lines from the end of a research reply`.

### Task 6: A live fire carries feelings and a rating

**Files:**
- Modify: `brain/engines/research.py:199,314,580,584,616`, `brain/prompt_strings.toml` (`[engines.research]`)
- Test: `tests/unit/brain/engines/test_research.py`

**Interfaces:**
- Consumes: `extract_felt_lines` from Task 5.
- Produces: `build_research_memory(..., emotions: dict[str, float] | None = None, engagement_rating: float | None = None)`; `_render_system_prompt(self, *, live: bool = False)`; `_render_prompt(..., live: bool = False)`; string key `engines.research.render_system_prompt_segments_live`.

- [ ] **Step 1: Write the failing tests**

```python
def test_live_fire_stores_feelings_and_rating(tmp_path):
    # MemoryStore(tmp_path / "memories.db"), _go_live(tmp_path, store), ScriptedProvider with the select reply, then
    # 'NOTES:\nn\n\nMEMORY:\nm\n\nVERDICT:\nclose\n\nEMOTIONS: {"awe": 8}\nENGAGEMENT: 6/10'
    memory = PendingQueue(tmp_path).read_recent("research", limit=1)[0]  # research is a gated type
    assert memory.emotions == {"awe": 8.0}
    assert memory.metadata["engagement_rating"] == 6.0
    assert result.fired is not None
    # and the interest went dormant: the verdict still reads "close" with the felt lines after it

def test_live_fire_with_only_the_three_markers_keeps_the_memory_and_logs(tmp_path):
    # memory written, emotions == {}, "engagement_rating" not in metadata, parse_failures row has site "research_felt"

def test_live_system_prompt_asks_for_the_felt_lines_and_off_does_not(tmp_path):
    assert 'EMOTIONS: {"<emotion name>": <0-10>}' in engine._render_system_prompt(live=True)
    assert "ENGAGEMENT" not in engine._render_system_prompt()

def test_off_fire_does_not_parse_felt_lines(tmp_path):
    # the same reply with the switch off leaves the felt lines to parse_session_output as today, emotions == {}, no rating key

def test_build_research_memory_leaves_the_rating_key_out_when_missing(): ...
```

Mark the off tests `no_base_failure` where they pass on base.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/brain/engines/test_research.py -v`
Expected: the new live tests FAIL.

- [ ] **Step 3: Implement**

Live segments: copy `render_system_prompt_segments`, with "Write plain prose under exactly three markers, then two last lines, nothing else:" in segment 1 and this appended to segment 2:

```
Then, each on its own last line:
EMOTIONS: {"<emotion name>": <0-10>}
ENGAGEMENT: <0-10>
```

Move the switch read above `if emo_state is None:`, so `live = live_engine.is_live()` holds for the whole fire and the existing reader branch reuses it. When live, `_render_prompt` ends with "Write NOTES / MEMORY / VERDICT, then EMOTIONS and ENGAGEMENT, now." instead of today's line, so the two instructions agree.

Before `parse_session_output(raw)` when live: `raw_rest, felt, rating = extract_felt_lines(raw)`, then parse `raw_rest`. If `felt is None or rating is None`, call `record_parse_failure(persona_dir, site="research_felt", raw=raw, error=...)`. Pass `emotions=felt or {}` and `engagement_rating=rating` to `build_research_memory`, which writes the metadata key only when not `None`. Off path unchanged. `research_recovery.py` unchanged.

- [ ] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/unit/brain/engines/test_research.py tests/unit/brain/engines/test_research_recovery.py tests/unit/brain/engines/test_research_session.py -v`
Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

Message: `give a research fire its own feelings and rating once the engine is live`.

---

## PR 5: dreams, grief, growth (branch `hana/emotion-b1-dreams-grief-growth`, after #671)

### Task 7: Dream row, mood and seeds

**Files:**
- Modify: `brain/engines/dream.py:102,172,241`, `brain/engines/dream_seeds.py:27,94,108,128`
- Test: `tests/unit/brain/engines/test_dream.py`, `tests/unit/brain/engines/test_dream_seeds.py`, `tests/unit/brain/emotion/test_rework_writer_pins.py`

**Interfaces:**
- Produces: keyword-only `eng: EmotionEngine | None` with no default on `emotional_congruence`, `grief_pull`, `composite_score`, `select_seed`, `DreamEngine._select_seed` and `DreamEngine._write_dream_memory`.

- [ ] **Step 1: Write the failing tests**

```python
def test_live_cycle_stores_the_mean_of_faded_values(persona_dir, store, hebbian, monkeypatch):
    # _go_live with love=6.0, a persona_dir engine, the faded patch; seed love=8, neighbours love=4 and awe=6
    result = dream_engine.run_cycle(seed_id=seed.id)
    assert result.memory.emotions == pytest.approx({"love": 3.0, "awe": 3.0})

def test_live_dream_holds_old_rows_within_range(...):
    # neighbours carrying love=18.0 and tenderness=-0.3 stored directly
    assert all(0.0 <= v <= 10.0 for v in result.memory.emotions.values())

def test_live_mood_comes_from_the_engine_read(...):
    # a grief_event row cannot set its own mood; a live short average on "joy" picks the joy-congruent seed
    # while an old high-valued row of another feeling does not

def test_congruence_and_grief_pull_read_faded_values(monkeypatch):
    assert grief_pull(breadcrumb, eng=half) == pytest.approx(stored / 20.0)
```

In `test_dream_seeds.py` and `test_dream.py`, every existing call gains `eng=None` (19 calls to the seed functions, 3 to `_select_seed`). In the pins file, remove `@_flipped_by("B1-4")`. `_write_dream_over_ten` gains `eng=None` for `test_today_*`, and the intended pin rewrites with `_go_live`.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/brain/engines/test_dream.py tests/unit/brain/engines/test_dream_seeds.py tests/unit/brain/emotion/test_rework_writer_pins.py -v`
Expected: FAIL with `TypeError` on `eng` until the signatures change, then on the values.

- [ ] **Step 3: Implement**

`dream_seeds.py`: add `_values(memory, eng) -> dict[str, float]` (stored when `eng is None`, else faded by the memory's own keys). `emotional_congruence` pools `_values`. `grief_pull` reads `memory_grief` through it when present. Import `EmotionEngine` under `TYPE_CHECKING` and update the module docstring's "no engine dependencies" line.

`run_cycle`: `eng = engine_for(self.persona_dir) if self.persona_dir else None`, set to `None` when not live, and passed down once. `_select_seed`: mood is `EmotionalState(emotions={name: r.short for name, r in eng.read().items()})` when `eng`, else `aggregate_state(convo)`. `_write_dream_memory`: when `eng`, take the mean per emotion of faded values across the seed and the neighbours carrying it, hold it to 0-10, and drop what lands at 0; else today's sum. The snapshot at `dream.py:342` copies the row and needs no change.

- [ ] **Step 4: Run them to verify they pass**

Same command as Step 2. Expected: PASS, with `test_today_dream_over_ten_is_erased_by_one_heartbeat` still green.

- [ ] **Step 5: Run the gate and commit**

Message: `use faded means and the engine's mood for live dreams`.

### Task 8: Grief reads faded values

**Files:**
- Modify: `brain/grief/__init__.py:53`
- Test: `tests/grief/test_breadcrumb.py`, `tests/forgetting/test_orchestrator.py`

**Interfaces:**
- Consumes: `faded_value` for a row that is already deleted. This is the new question for ToT on #633.

- [ ] **Step 1: Write the failing tests**

```python
def test_live_drop_uses_faded_values(persona_dir, store, monkeypatch):
    # _go_live, faded patch, a memory joy=8, handle_drop(memory=m, persona_dir=persona_dir, store=store)
    # with the stored 8 it writes; assert the breadcrumb's intensity and residue match value 4

def test_live_drop_reached_through_the_real_forgetting_pass(...):
    # run_pass with a memory past the lost threshold: the row is gone, a breadcrumb exists

def test_faded_value_works_after_hard_delete(persona_dir, store):
    # the real unpatched faded_value on a memory object after store.hard_delete; handle_drop writes its breadcrumb

def test_old_row_over_ten_stays_in_range(...):
    # memory love=18.0 live: breadcrumb values within 0-10
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/grief/test_breadcrumb.py tests/forgetting/test_orchestrator.py -v`
Expected: the patched test FAILS, because stored values are used. Mark `test_faded_value_works_after_hard_delete` as `no_base_failure` if it passes on base.

- [ ] **Step 3: Implement**

Inside the existing `try` of `handle_drop`: `eng = engine_for(persona_dir)`, and `values = {k: eng.faded_value(memory, k) for k in memory.emotions} if eng.is_live() else dict(memory.emotions)`. Use `values` for both the maximum and `_dominant_non_grief_emotion`. Hold the residue value to 10 before passing it on, if the over-ten test shows it is needed.

- [ ] **Step 4: Run them to verify they pass**

Same command as Step 2. Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

Message: `use faded values for grief at drop time once the engine is live`.

### Task 9: Growth context and vocabulary blends

**Files:**
- Modify: `brain/growth/scheduler.py:106,262,285`, `brain/growth/crystallizers/reflex.py:483,504`, `brain/growth/crystallizers/creative_dna.py:185,468`, `brain/growth/crystallizers/vocabulary.py:31,76`
- Test: `tests/unit/brain/growth/test_scheduler.py`, `test_reflex_crystallizer.py`, `test_creative_dna_crystallizer.py`, `test_vocabulary_crystallizer.py`

**Interfaces:**
- Produces: `_aggregate_recent_emotion_vector(store, *, now, persona_dir, look_back_days=...)` in all three files; `crystallize_vocabulary(store, *, current_vocabulary_names, eng)`; `_cluster_active_memories(memories, *, eng)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_live_initiate_candidate_carries_short_averages(persona_dir, store):
    # _go_live(love=6.0) plus evidence for a blend; run_growth_tick; the emitted candidate's vector equals {name: r.short} above 0

def test_old_memories_do_not_form_a_blend_when_live(persona_dir, store, monkeypatch):
    # three memories love=7, grief=7, faded patch makes them 3.5: below the 6.0 floor, no proposal

def test_top_two_are_chosen_by_faded_value(...):
    # a patch that changes the order changes which pair clusters

def test_off_tick_is_unchanged(...): ...  # no_base_failure
```

Add the short-average test for `crystallize_reflex` and `crystallize_creative_dna` through their real entry points with the stub provider those test files already use. Existing `crystallize_vocabulary(...)` calls in `test_vocabulary_crystallizer.py` gain `eng=None` (6 calls).

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/brain/growth -v`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement**

Each `_aggregate_recent_emotion_vector`, first thing inside its `try`: when live, return `{name: r.short for name, r in eng.read().items() if r.short > 0}`. Call sites pass `persona_dir=persona_dir`.

`run_growth_tick`: `eng` is the engine when live, else `None`, passed to `crystallize_vocabulary`. `_cluster_active_memories` takes values through the same own-keys rule, ranks the top two by them, applies `_MIN_EMOTION_INTENSITY` to them, and scores from them. Reconcile reaches this through `run_growth_tick`, so it needs no change.

- [ ] **Step 4: Run them to verify they pass**

Same command as Step 2, plus `uv run pytest tests/unit/brain/self_model/test_reconcile.py -q`. Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

Message: `read growth context and blends from the engine once it is live`.

---

## PR 6: monologue note and remaining writers (branch `hana/emotion-b1-monologue-note`)

### Task 10: The trace becomes a note

**Files:**
- Modify: `brain/monologue/trace.py:34`, `brain/chat/monologue_capture.py:87,99`
- Test: `tests/unit/brain/tools/test_dispatch_record_monologue.py`, `tests/unit/brain/monologue/test_trace.py`, `test_ambient.py`, `test_recall.py`, `tests/unit/brain/engines/test_interest_sweep.py`, plus the callers below

**Interfaces:**
- Produces: `write_trace_memory(store, monologue, *, feed_digest: str, persona_dir: Path | None) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
def test_live_capture_stores_digest_two_feelings_and_keeps_raw_for_pass2(persona_dir, store):
    # _go_live(joy=5.0); a second fold makes the strongest long differ from the strongest short
    result = _dispatch_record_monologue(monologue="raw", feed_digest="she was thinking", store=store, persona_dir=persona_dir)
    trace = PendingQueue(persona_dir).read_recent(MONOLOGUE_TRACE_TYPE, limit=1)[0]
    assert trace.content == "she was thinking"
    assert set(trace.emotions) == {strongest_short, strongest_long}
    assert trace.importance == pytest.approx(_trace_importance(trace.emotions))
    assert result["monologue_text"] == "raw"

def test_live_one_key_when_both_are_the_same_feeling(...): ...   # the larger value
def test_live_empty_read_stores_no_feelings(...):                # read() == {} gives emotions == {}, importance 0.2
def test_live_dedupe_compares_digests(...):                      # same digest, different thought: {"ok": True, "deduped": True}
def test_live_same_thought_with_new_digest_is_kept(...): ...     # two traces
def test_off_capture_is_verbatim_with_aggregate(...): ...        # no_base_failure
```

Add one live reader test per reader: `build_interior_continuity_block` shows the digest and not the raw thought, `recall_monologue` matches digest words only, and `interest_sweep._lived_sample` shows the digest for a promoted trace.

Every existing `write_trace_memory(store, x)` test call becomes `write_trace_memory(store, x, feed_digest="", persona_dir=None)`: `test_ambient.py` (6), `test_recall.py` (4), `test_trace.py` (2), `test_monologue_trace_lifecycle.py` (2), `test_prompt.py` (1), `test_cascade_compaction.py` (1).

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/brain/tools/test_dispatch_record_monologue.py tests/unit/brain/monologue tests/unit/brain/engines/test_interest_sweep.py tests/test_gate_coverage.py -v`
Expected: the live tests FAIL because the raw thought is stored.

- [ ] **Step 3: Implement**

`write_trace_memory`: when `persona_dir` is set and live, content is `feed_digest` and emotions come from a new `_strongest_feelings(eng) -> dict[str, float]`. It takes the name with the highest `.short` and the name with the highest `.long`, keeps each only when its value is above 0, and when the names match keeps the larger value. Importance stays `_trace_importance(emotions)`. Off path as today.

`capture_monologue`: `live = engine_for(persona_dir).is_live()` above the guard, and the guard compares `recent[0].content` with `feed_digest if live else monologue`. Call `write_trace_memory` with the new keywords. Update the guard's comment and the "persist the verbatim trace" comment. The digest log, `surface` and the returned `monologue` stay as they are.

- [ ] **Step 4: Run them to verify they pass**

Same command as Step 2, plus `uv run pytest tests/unit/brain/chat/test_prompt.py tests/unit/brain/chat/test_cascade_compaction.py tests/unit/brain/forgetting/test_monologue_trace_lifecycle.py -q`. Expected: PASS.

- [ ] **Step 5: Update `CLAUDE.local.md`**

Rewrite the pass-2 gotcha's "raw monologue is persisted synchronously at record time": once the engine is live, only the digest persists, so a restart before the queue drains loses that thought's extraction. This file isn't tracked, so don't stage it.

- [ ] **Step 6: Run the gate and commit**

Message: `store the feed digest as the monologue note once the engine is live`.

### Task 11: Remaining writers

**Files:**
- Modify: `brain/tools/impls/add_memory.py:77`, `brain/ingest/emotion_backfill.py:149`
- Test: `tests/unit/brain/tools/test_impls.py`, `tests/unit/brain/ingest/test_emotion_backfill.py`, `tests/unit/brain/emotion/test_rework_writer_scale.py` (new)

- [ ] **Step 1: Write the tests**

```python
@pytest.mark.parametrize(("raw", "kept"), [(float("nan"), {}), (float("inf"), {}), (12.0, {"joy": 10.0}), (3.0, {"joy": 3.0})])
def test_normalize_drops_non_finite_and_holds_to_ten(raw, kept):
    assert _normalize({"joy": raw}, frozenset({"joy"})) == kept

@pytest.mark.no_base_failure(reason="pins the threshold while the literal moves to the shared constant")
def test_climax_journal_fires_at_the_threshold(...):
    # climax=CLIMAX_THRESHOLD writes the journal entry; just below it does not
```

`test_rework_writer_scale.py`: one test per writer, through `commit_item` (ingest), `commit_write` (file commit) and `write_breadcrumb` (grief), each given its largest legal input and asserting `all(0.0 <= v <= 10.0 for v in row.emotions.values())`. Each carries `no_base_failure` if it passes on base. The breadcrumb's residue is the source value × 0.5, so an old value over 20 can exceed 10. If the test shows that, hold the residue to 10 in `build_breadcrumb`'s `emotions` line here, and update spec §9.

- [ ] **Step 2: Run them to verify the NaN case fails**

Run: `uv run pytest tests/unit/brain/ingest/test_emotion_backfill.py tests/unit/brain/tools/test_impls.py tests/unit/brain/emotion/test_rework_writer_scale.py -v`
Expected: the NaN and inf cases FAIL. The rest pass and carry the marker.

- [ ] **Step 3: Implement**

`_normalize`: `if not math.isfinite(v) or v <= 0.0: continue`. `add_memory.py`: `from brain.body.events import CLIMAX_THRESHOLD, record_climax_event` above the `if`, comparing with `>= CLIMAX_THRESHOLD`. No gate.

- [ ] **Step 4: Run them to verify they pass**

Same command as Step 2. Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

Message: `drop non-finite backfill values and share the climax threshold`.

---

## PR 7: cleanup (branch `hana/emotion-b1-cleanup`, after Engine 6, before Engine 7)

### Task 12: Remove the old branches

**Files:**
- Modify: `brain/engines/dream.py`, `brain/monologue/trace.py`, `brain/chat/monologue_capture.py`, `brain/growth/scheduler.py`, `brain/growth/crystallizers/creative_dna.py`, `brain/growth/crystallizers/reflex.py`, `brain/tools/schemas.py:623,633`
- Test: the same test files, plus `tests/unit/brain/emotion/test_rework_writer_pins.py`

**Interfaces:**
- Produces: `write_trace_memory(store, *, feed_digest, persona_dir)` once the raw `monologue` argument has no reader.

- [ ] **Step 1: Re-check the ground**

Run: `git grep -n "aggregate_state" -- brain/engines/dream.py brain/monologue/trace.py brain/growth`
Expected: only the branches this task removes. Re-read Engine 6 as merged before deleting anything. In particular, confirm `faded_value` works on a deleted row (Task 8's test covers it) and how `DreamEngine(persona_dir=None)` should now choose a mood: an empty `EmotionalState()`, unless Engine 6 settled it differently.

- [ ] **Step 2: Write the failing schema test**

```python
def test_digest_and_surface_descriptions_are_first_person():
    # the record_monologue schema's feed_digest and surface descriptions contain "third-person" nowhere and ask for first person
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/unit/brain/tools -k first_person -v`
Expected: FAIL.

- [ ] **Step 4: Delete and reword**

Delete each `aggregate_state` branch and its import. Rewrite the off-path tests that pin the removed code, and delete the `test_today_*` pins whose writers are gone. Change the two schema descriptions to first person, which changes the Feed's voice. Approve the wording snapshot if it flags, and name the change in the PR body.

- [ ] **Step 5: Run the gate and commit**

Message: `remove the old emotion branches from the writers`. PR body: `Refs #240`, and `Closes #N` for any issue this finishes.

---

## After each PR

- [ ] Verify each `Closes #N` closed after merge.
- [ ] File any out-of-scope finding as a new issue, not a PR note.
- [ ] Run `graphify update .` after the merge.
- [ ] After PR 7: move the resolved rows in `project_companion_emergence_deferred.md` and refresh `docs/maturity-manifest.md`.
