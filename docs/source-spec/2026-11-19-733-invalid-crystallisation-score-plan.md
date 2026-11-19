# One bad crystallisation item costs only that item (#733): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a crystallisation item that fails its own validation is dropped and recorded, and the rest of pass 2's reply applies.

**Architecture:** #640's pre-validation filter in `brain/chat/extractor.py` grows from "remove unscored items" to "remove unscored items and items `CrystallisationCandidate` rejects", returning both counts. `extract_from_thinking` builds the parse-failure note from the two counts. The whole-reply `ExtractorOutput.model_validate` is unchanged.

**Tech Stack:** Python, pydantic v2, pytest.

**Spec:** `docs/source-spec/2026-11-19-733-invalid-crystallisation-score-design.md`

## Global Constraints

- One parse-failure row per reply (#640).
- The row's `error` field never holds reply text or a pydantic message (#427): counts and field names only.
- Today's note text `N crystallisation item(s) without importance` is kept word for word.
- No clamping of an out-of-range score.
- Only `crystallisation` items change. The other reply fields keep whole-reply behaviour.
- Tests use a temp persona dir, never mocked state files. Run with `scripts/clean_shell.sh uv run pytest -o "tdd_guard_project_root=<main checkout>"`, where `<main checkout>` is the path of the main clone (tdd-guard reads its results there).

## Review Focus

1. A theme over 200 or evidence over 500 characters is truncated by the model's `mode="before"` validators, so it is valid and must be kept, not dropped.
2. A numeric string score (`"9"`) passes pydantic's lax int and must be kept.
3. An item that isn't a dict (a string or a number inside the list) is invalid and dropped. It must not crash the filter.
4. A `crystallisation` value that isn't a list goes on to the whole-reply validation untouched, as today.
5. A kept item must pass the whole-reply validation too: the filter and `ExtractorOutput` use the same model, so no item is dropped that the whole-reply check would accept.

Tests for 1 to 4 are in Task 1 Step 1. Item 5 is held by construction (the same model) and by the acceptance test passing.

---

### Task 1: Drop invalid crystallisation items per item

**Files:**
- Modify: `brain/chat/extractor.py` (`_drop_unscored_crystallisation`, the note-building lines in `extract_from_thinking`)
- Test: `tests/unit/brain/chat/test_extractor_llm.py` (append after the #640 block)

**Interfaces:**
- Produces: `_drop_bad_crystallisation(data: dict) -> tuple[int, int]` returns `(unscored, invalid)` and edits `data["crystallisation"]` in place. It replaces `_drop_unscored_crystallisation`, whose one caller is `extract_from_thinking`.
- Consumes: `CrystallisationCandidate`, `pydantic.ValidationError`, the test file's existing `_FakeProvider` and `_parse_rows`.

- [ ] **Step 1: Write the failing tests**

```python
# ── #733: an item with an invalid score costs only that item ──────────────
def test_invalid_score_is_dropped_and_the_rest_applies_through_pass2(tmp_path, monkeypatch):
    # as #640's through-pass2 test, with {"theme": "Bad score", "evidence": "...", "importance": "high"}
    # next to {"theme": "Scored pull", ..., "importance": 9} and one reflex_audit entry
    assert "Bad score" not in queued
    assert len(texts) == 1 and texts[0].startswith("Scored pull")
    assert (persona_dir / "reflex_audit.jsonl").exists()
    (row,) = _parse_rows(persona_dir)
    assert row["error"] == "1 invalid crystallisation item(s) dropped"
    assert "Bad score" not in row["error"] and "high" not in row["error"]

@pytest.mark.parametrize("bad", ["", 0, 11, 9.5, "high"])
def test_each_invalid_score_drops_only_its_item(tmp_path, bad):
    # [{"theme": "bad", "evidence": "e", "importance": bad}, {"theme": "good", "evidence": "e", "importance": 9}]
    assert [c.theme for c in out.crystallisation] == ["good"]
    (row,) = _parse_rows(tmp_path)
    assert row["error"] == "1 invalid crystallisation item(s) dropped"

def test_a_whitespace_theme_drops_only_its_item(tmp_path): ...       # theme "   " dropped, good kept, same row text
def test_a_non_dict_item_is_dropped(tmp_path): ...                  # ["just a string", good]: good kept, one row
def test_unscored_and_invalid_are_counted_in_one_row(tmp_path):
    # one item with no importance, one with importance 0, one good
    assert row["error"] == "1 crystallisation item(s) without importance, 1 invalid crystallisation item(s) dropped"
def test_invalid_item_plus_schema_error_writes_one_schema_row(tmp_path):
    # memory_writes [{"episode": "x", "salience": 5}] (invalid) + an item with importance 11
    assert out == ExtractorOutput()
    assert row["error"].startswith("schema: 1 invalid crystallisation item(s); ")

@pytest.mark.no_base_failure(reason="pins that a long theme is truncated, not dropped (#733); main truncates it too")
def test_a_long_theme_is_kept_truncated(tmp_path): ...              # theme "x" * 300, importance 9: kept, theme length 200, no row
@pytest.mark.no_base_failure(reason="pins that a numeric string score is kept (#733); main keeps it too")
def test_a_numeric_string_score_is_kept(tmp_path): ...              # importance "9": kept with importance 9, no row
@pytest.mark.no_base_failure(reason="pins that a non-list crystallisation still fails the whole reply (#733); main does too")
def test_a_non_list_crystallisation_still_fails_the_reply(tmp_path): ...  # "crystallisation": "oops": out == ExtractorOutput(), row starts "schema: "
```

- [ ] **Step 2: Run them and see the new ones fail**

Run: `scripts/clean_shell.sh uv run pytest tests/unit/brain/chat/test_extractor_llm.py -q -o "tdd_guard_project_root=<main checkout>"`
Expected: the first six fail (the reply is discarded and the row starts `schema:`). The three marked `no_base_failure` pass. #640's tests pass.

- [ ] **Step 3: Implement `_drop_bad_crystallisation(data: dict) -> tuple[int, int]`**

A dict item with `importance` missing or None is unscored. Any other item is checked with `CrystallisationCandidate.model_validate(item)`, and an item that isn't a dict or raises `ValidationError` is invalid. Keep the original items (not the validated copies). A non-list value returns `(0, 0)` untouched. In `extract_from_thinking`, the note is the non-empty parts of `[f"{u} crystallisation item(s) without importance", f"{i} invalid crystallisation item(s)"]` joined by `", "`. Write it as `f"{note} dropped"` after a successful validation and as the `f"schema: {note}; "` prefix on failure, as today. Update the docstring and the `importance` field's comment to say an invalid score is dropped too.

- [ ] **Step 4: Run the extractor tests and see them pass**

Same command as Step 2, plus `tests/unit/brain/chat/test_extractor_apply.py tests/unit/brain/chat/test_extractor_schema.py`. Expected: all pass.

- [ ] **Step 5: Run the gate**

`scripts/clean_shell.sh uv run pytest -q` with no `^FAILED` line, `uv run ruff check .`, `cd app && pnpm test && pnpm build`. After committing, `uv run python scripts/new_tests_fail_on_base.py --base origin/main` reports every new test biting or exempt.

- [ ] **Step 6: Commit**

```bash
git add brain/chat/extractor.py tests/unit/brain/chat/test_extractor_llm.py
git commit -m "drop an invalid crystallisation item instead of the whole reply (#733)"
```
