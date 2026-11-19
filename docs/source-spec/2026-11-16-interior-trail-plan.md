# Interior trail of notes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Once the emotion engine is live, the interior-continuity block shows a trail of whole recent notes under a neutral header, and `recall_monologue`'s description says only what is true.

**Architecture:** `build_interior_continuity_block` reads the switch once through `store.emotion_engine()`. The switch-off path stays exactly as it is. The live path uses a new header key and a fit that cuts only between notes. The tool description is one static string, true in both states.

**Tech Stack:** Python 3 (`uv`, pytest, ruff), `brain/prompt_strings.toml`, the emotion engine switch (`brain/emotion/engine.py`, `brain/emotion/engine_files.py`).

**Spec:** `docs/source-spec/2026-11-16-interior-trail-design.md`

## Global Constraints

- Switch off, or unknown: the block is byte-for-byte today's.
- The count stays 5 (`_AMBIENT_LIMIT`) and the cap 1,200 (`_CHAR_CAP`). The footer is unchanged, appended after the fit.
- Live header text, exactly: `── interior continuity (your recent thoughts, as you noted them) ──`, under the key `monologue.ambient.header_live`.
- New description, exactly: `Search your own recent interior for an earlier thought. Each match comes back as you kept it, with when you had it: the short note you made at the time, or the whole thought if it is from before you kept notes.`
- New prose has no em-dashes. The header's `──` rule glyphs match today's header.
- Tests use a temp persona dir. No mocked state files.
- Full gate before every commit: `uv run pytest` (check with `grep '^FAILED'`), `uv run ruff check .`. No frontend files change, so `pnpm` isn't needed.

## Review Focus

1. **Fewer than five traces while live:** one or two notes render whole under the live header, with no blank bullets. Task 1, `test_live_with_two_notes`.
2. **An in-memory store (no persona dir):** `store.emotion_engine()` returns None, so the block takes the off path without raising. Task 1, `test_in_memory_store_takes_today_path`.
3. **A raw thought queued just before the switch-over, in second place:** it is dropped whole, not cut, and the newest note still shows. Task 1, `test_live_drops_long_older_thought_whole`.
4. **A switch read that raises:** this counts as off and gives today's string. Task 1, `test_unknown_switch_takes_today_path`.
5. **Newlines inside a note:** they are collapsed to spaces as today, so one note is one line and the fit counts it once. Task 1, `test_live_collapses_whitespace`.

---

### Task 1: The live trail in the interior block

**Files:**
- Modify: `brain/monologue/ambient.py` (`build_interior_continuity_block`, plus a new header registration)
- Modify: `brain/prompt_strings.toml` (`[monologue.ambient]`: add `header_live`)
- Test: `tests/unit/brain/monologue/test_ambient.py`
- Test: `tests/unit/brain/chat/test_prompt_caching_split.py` (one through-path test)

**Interfaces:**
- Consumes: `MemoryStore.emotion_engine() -> EmotionEngine | None`, `EmotionEngine.is_live() -> bool`, `engine_files.write_switch(persona_dir, at) -> bool`, `write_trace_memory(store, monologue, *, feed_digest, persona_dir, live=None)`.
- Produces: the same `build_interior_continuity_block(store, *, limit=5, char_cap=1200, user_name="the user") -> str`. Its signature doesn't change.

Test helpers, in `test_ambient.py`:
- `_go_live(persona_dir, store)`, as in `tests/unit/brain/maker/test_making_runner.py:89`: seed one counted memory, then `engine_files.write_switch(persona_dir, datetime.now(UTC))`.
- `_queue_digest(store, persona_dir, digest)`: `write_trace_memory(store, "raw " + digest, feed_digest=digest, persona_dir=persona_dir, live=True)`.

Use `MemoryStore(tmp_path / "memories.db")` with `persona_dir = tmp_path`. Digests of realistic length are about 220 characters, for example `"she " + "w" * 216`.

- [ ] **Step 1: Write the failing tests**

```python
LIVE_HEADER = "── interior continuity (your recent thoughts, as you noted them) ──"

def test_live_shows_whole_notes_newest_first(tmp_path):
    # live; queue 5 digests of 220 chars, d0 oldest .. d4 newest
    # body = block minus its last (footer) line
    # body.splitlines()[0] == LIVE_HEADER
    # every other body line is "· " + a whole digest, in order d4, d3, d2, ...
    # len(body) <= 1200; no body line is a truncated digest

def test_live_drops_the_note_that_would_pass_the_cap(tmp_path):
    # live; 5 digests of 300 chars -> header + 3 lines fit (50 + 3*303 < 1200), the 4th doesn't
    # exactly 3 note lines, each whole

def test_live_newest_longer_than_cap_is_cut_and_alone(tmp_path):
    # live; newest digest 1500 chars (written straight to the queue, bypassing the 400 schema limit), older one short
    # body has exactly 2 lines: LIVE_HEADER and a cut "· ..." line; len(body) == 1200

def test_live_drops_long_older_thought_whole(tmp_path):
    # live; older trace queued with live=False (a raw 1400-char thought), then a 220-char digest
    # body lines == [LIVE_HEADER, "· " + digest]

def test_live_with_two_notes(tmp_path):
    # live; 2 digests -> exactly 2 note lines, both whole

def test_live_collapses_whitespace(tmp_path):
    # live; digest "she paused.\n\nthen  smiled" -> line "· she paused. then smiled"

def test_switch_off_is_todays_block(tmp_path):
    # no switch; the same 5 queued traces (live=False)
    # == today's rendering: "\n".join([OLD_HEADER] + ["· " + t for t newest first])[:1200] + "\n" + footer
    # OLD_HEADER == "── interior continuity (your own recent thought) ──"

def test_unknown_switch_takes_today_path(tmp_path, monkeypatch):
    # write the switch, then refuse_the_switch_stat(monkeypatch) (tests/unit/brain/emotion/_engine_support.py)
    # the block starts with OLD_HEADER and uses today's slice

def test_in_memory_store_takes_today_path():
    # MemoryStore(":memory:"): emotion_engine() is None -> no raise; returns "" (no queue) or today's shape
```

In `test_prompt_caching_split.py`, the through-path test:

```python
def test_volatile_context_carries_the_live_trail(persona_dir, store, soul_store):
    # _go_live(persona_dir, store); queue 3 digests of 220 chars
    # volatile = build_volatile_context(persona_dir, voice_md=..., daemon_state=DaemonState(), soul_store=soul_store, store=store, user_input="hi")
    # LIVE_HEADER in volatile; each digest appears whole; OLD_HEADER not in volatile
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/brain/monologue/test_ambient.py tests/unit/brain/chat/test_prompt_caching_split.py -k "live or todays or unknown or in_memory or trail" -v`

Expected: every `live` test and the trail test FAIL, because the header and fit don't exist yet. `test_switch_off_is_todays_block`, `test_unknown_switch_takes_today_path` and `test_in_memory_store_takes_today_path` PASS, because they pin today's behaviour.

If the in-memory test errors because the queue needs a persona dir, keep the assertion as "returns a string, doesn't raise".

- [ ] **Step 3: Implement**

1. Add to `[monologue.ambient]` in `prompt_strings.toml`, with a one-line comment in the file's style:
   `header_live = '''── interior continuity (your recent thoughts, as you noted them) ──'''`
2. In `ambient.py`, register it beside the existing header: `_HEADER_LIVE = prompt_strings.register("monologue.ambient.header_live")`.
3. In `build_interior_continuity_block`, after the traces are sorted, read the switch once:
   `live = eng is not None and eng.is_live()`, where `eng = store.emotion_engine()`. Wrap the read in `try/except Exception`; any exception means `live = False`.
4. Off: run today's lines, unchanged.
5. Live, after collapsing each note's whitespace:

```python
lines = [f"· {t}" for t in texts]                 # newest first
body = (_HEADER_LIVE + "\n" + lines[0])[:char_cap]
for line in lines[1:]:
    candidate = body + "\n" + line
    if len(candidate) > char_cap:
        break
    body = candidate
```

   Return `body + "\n" + _FOOTER_TEMPLATE.format(user_name=user_name)`, as today.

Update the module docstring: once live, the block is a trail of whole notes.

- [ ] **Step 4: Run them to verify they pass**

Run the Step 2 command. Expected: all PASS. Then run `uv run pytest tests/unit/brain/monologue tests/unit/brain/chat -q`. Expected: no new failures.

- [ ] **Step 5: Approve the wording snapshot**

Run: `uv run pytest tests/unit/test_persona_wording_snapshot.py -q`. If it flags `monologue.ambient.header_live`, run `uv run python tests/unit/test_persona_wording_snapshot.py --approve`, re-run, and expect PASS. Note the approved string for the PR body.

- [ ] **Step 6: Full gate, then commit**

```bash
uv run pytest -q 2>&1 | grep '^FAILED' ; uv run ruff check .
git add brain/monologue/ambient.py brain/prompt_strings.toml tests/unit/brain/monologue/test_ambient.py tests/unit/brain/chat/test_prompt_caching_split.py <snapshot file if changed>
git commit -m "show a trail of whole notes in the interior block once the engine is live (#688)"
```

The grep should print nothing. If the known `test_streaming_proxy_done_only.py` order flake shows up, re-run that file alone and say so in the PR.

---

### Task 2: A true `recall_monologue` description

**Files:**
- Modify: `brain/tools/schemas.py` (`recall_monologue.description`)
- Test: `tests/unit/brain/tools/test_dispatch.py`, or a new `tests/unit/brain/tools/test_recall_monologue_description.py` if `test_dispatch.py` has no schema-text tests

**Interfaces:**
- Consumes: `brain.tools.schemas.SCHEMAS: dict[str, dict]`.
- Produces: nothing new. The schema shape doesn't change.

- [ ] **Step 1: Write the failing test**

```python
NEW = ("Search your own recent interior for an earlier thought. Each match comes back as you kept it, "
       "with when you had it: the short note you made at the time, or the whole thought if it is from "
       "before you kept notes.")

def test_recall_monologue_description_is_true_in_both_states():
    desc = SCHEMAS["recall_monologue"]["description"]
    assert desc == NEW
    for word in ("verbatim", "blurred", "vivid", "—"):
        assert word not in desc
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest <test file> -k description_is_true -v`. Expected: FAIL.

- [ ] **Step 3: Replace the description string** with `NEW`, split across lines in the file's existing style.

- [ ] **Step 4: Run it, then the tools tests**

Run: `uv run pytest tests/unit/brain/tools -q`. Expected: PASS. If a test pinned the old text, update it to the new text and name it in the PR body.

- [ ] **Step 5: Wording snapshot, full gate, commit**

Approve the snapshot if it flags (Task 1 step 5 recipe). Then:

```bash
uv run pytest -q 2>&1 | grep '^FAILED' ; uv run ruff check .
git commit -m "say only what is true in recall_monologue's description (#688)"
```

---

## After the tasks

- [ ] Push `hana/688-interior-trail`. In the draft PR body, put `Closes #688` on its own line, plus `Refs #240`, both approved strings, and the one-time cached-prefix move.
- [ ] Move the design's two deferred rows into `project_companion_emergence_deferred.md`.
- [ ] Run `graphify update .` after merge.
- [ ] Once Engine 6 lands, switch `_go_live` in these tests to `make_live`, following the B1 plan's "After Engine 6" recipe, wherever that rebase touches them.
