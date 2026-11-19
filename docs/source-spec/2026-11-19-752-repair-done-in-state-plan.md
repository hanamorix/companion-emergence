# The self-model repair records "done" in the state save (#752): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a repair whose state save succeeded is never run again, even when its marker write fails.

**Architecture:** `self_model_state.json` gains an additive top-level key, `windowed_peak_cleared`, that `save()` keeps once set. The repair sets it in the same save that clears the gap, and `should_run_self_model_repair` reads it before the marker.

**Tech Stack:** Python, pytest.

**Spec:** `docs/source-spec/2026-11-19-752-repair-done-in-state-design.md`

## Global Constraints

- `SelfModelState` and the six writers outside `state.py` and the repair stay unchanged. In particular, `reconcile.py` and `supervisor.py` aren't touched.
- The key is written only as `true`, never `false`, and is never removed by `save()` once on disk.
- Persisted state stays readable by an older brain: the change is additive only.
- #737's behaviour and its three tests are unchanged.
- Tests use real files in `tmp_path`, never mocked state files. Run with `scripts/clean_shell.sh uv run pytest -o "tdd_guard_project_root=<main checkout>"`, where `<main checkout>` is the path of the main clone.

## Review Focus

1. A later writer saves a state built from scratch (`SelfModelState(current_gap=g, gap_history=h)`), as the supervisor's spare rule and reconcile do. Expected: the flag survives.
2. The state file on disk is corrupt or not an object when a writer saves. Expected: the save proceeds as today, without carrying a flag it can't read.
3. The repair finds a recovered file (a malformed history entry) and no gap. Expected: no save, so the malformed entry stays on disk for the supervisor's next save, as today.
4. The marker write fails after the clear. Expected: the next start doesn't run the repair.
5. A save that raises inside the repair. Expected: #737's counting, and no flag on disk.

Tests for 1 to 5 are in Task 1 Step 1 (2 in `test_state.py`, the rest in `test_self_model_repair.py`).

---

### Task 1: The flag in the state, read by the repair

**Files:**
- Modify: `brain/self_model/state.py` (`save`, a new `windowed_peak_cleared`)
- Modify: `brain/health/self_model_repair.py` (`should_run_self_model_repair`, `run_self_model_repair`)
- Test: `tests/unit/brain/self_model/test_state.py`, `tests/unit/brain/health/test_self_model_repair.py`

**Interfaces:**
- Produces: `save(persona_dir: Path, state: SelfModelState, *, windowed_peak_cleared: bool = False) -> None` and `windowed_peak_cleared(persona_dir: Path) -> bool` in `brain.self_model.state`.

- [ ] **Step 1: Write the tests**

`test_state.py`:

```python
def test_the_cleared_flag_survives_a_save_built_from_scratch(tmp_path):
    save(tmp_path, SelfModelState(), windowed_peak_cleared=True)
    save(tmp_path, SelfModelState(current_gap=_gap(), gap_history=[]))  # how the supervisor and reconcile write
    save(tmp_path, push_gap(load_or_recover(tmp_path)[0], _gap()))
    assert windowed_peak_cleared(tmp_path) is True

def test_the_cleared_flag_is_never_written_false(tmp_path):
    save(tmp_path, SelfModelState())
    assert "windowed_peak_cleared" not in json.loads(_state_path(tmp_path).read_text(encoding="utf-8"))
    assert windowed_peak_cleared(tmp_path) is False

@pytest.mark.parametrize("content", [None, "{not json", "[1, 2]"])  # None = no file
def test_the_cleared_flag_reads_false_for_a_missing_or_bad_file(tmp_path, content): ...

@pytest.mark.no_base_failure(reason="pins that a save over a corrupt state file still writes the new state (#752); main does too")
def test_a_save_over_a_corrupt_file_writes_the_new_state(tmp_path):
    # corrupt file on disk, then save(SelfModelState(current_gap=g)): load_or_recover returns g, no flag
```

`test_self_model_repair.py`:

```python
def test_a_marker_that_cannot_be_written_does_not_rerun_the_repair(tmp_path):
    # bogus gap saved; make the marker path a directory so write_json fails (on_error="log")
    run_self_model_repair(tmp_path)
    assert sm_state.load_or_recover(tmp_path)[0].current_gap is None
    assert should_run_self_model_repair(tmp_path) is False
    # an honest gap saved afterwards (plain save, as the reflection tick writes) is kept: the repair isn't due
    sm_state.save(tmp_path, sm_state.SelfModelState(current_gap=_honest_gap(), gap_history=[]))
    assert should_run_self_model_repair(tmp_path) is False

def test_the_repair_records_the_flag_when_there_is_no_gap(tmp_path):
    # no state file; marker path a directory
    run_self_model_repair(tmp_path)
    assert sm_state.windowed_peak_cleared(tmp_path) is True
    assert should_run_self_model_repair(tmp_path) is False

def test_a_failed_save_leaves_no_flag(tmp_path, monkeypatch):
    # as #737's failing_save: the flag is absent and the repair is still due

@pytest.mark.no_base_failure(reason="pins that the repair does not overwrite a recovered state file with no gap (#752); main does not either")
def test_a_recovered_file_with_no_gap_is_not_overwritten(tmp_path):
    # a file whose gap_history holds one malformed entry and no current_gap: bytes unchanged after the repair
```

Update `test_repair_noop_when_no_state`: the repair now creates the state file, so it also asserts `sm_state.windowed_peak_cleared(tmp_path) is True`.

- [ ] **Step 2: Run them and see the new ones fail**

Run: `scripts/clean_shell.sh uv run pytest tests/unit/brain/self_model/test_state.py tests/unit/brain/health/test_self_model_repair.py -q -o "tdd_guard_project_root=<main checkout>"`
Expected: the flag tests fail. On main `save` takes no `windowed_peak_cleared` and `state.windowed_peak_cleared` doesn't exist, so most fail with `TypeError` or `AttributeError`. The marker-failure test fails on `should_run_self_model_repair(...) is False`. The two `no_base_failure` pins and #737's tests pass.

- [ ] **Step 3: Implement**

`state.py`:
- `windowed_peak_cleared` reads with the same `read_json_result(path, default=lambda: None, expect=dict)` call `load_or_recover` uses. It returns `res.outcome == "ok" and res.data.get("windowed_peak_cleared") is True`.
- `save` adds `"windowed_peak_cleared": True` to the payload when its argument is true or `windowed_peak_cleared(persona_dir)` is.

`self_model_repair.py`:
- `should_run_self_model_repair` returns False first when `sm_state.windowed_peak_cleared(persona_dir)`.
- In `run_self_model_repair`, keep `recovered` from `load_or_recover`. Then:
  - if `state.current_gap is not None`: save the cleared state with `windowed_peak_cleared=True`;
  - elif `not recovered`: save `state` with `windowed_peak_cleared=True`;
  - otherwise: no save.
- Update both functions' docstrings and the module docstring's "Marker-gated" line to say the state flag is the first record.

- [ ] **Step 4: Run them and see them pass**

Same command as Step 2, plus `tests/unit/brain/bridge/test_self_model_state_lock.py tests/unit/brain/health/test_migrated_group_b.py tests/unit/test_persisted_state_tolerance.py`. Expected: all pass.

- [ ] **Step 5: Run the gate**

- `scripts/clean_shell.sh uv run pytest -q` with no `^FAILED` line.
- `uv run ruff check .` clean.
- `cd app && pnpm test && pnpm build` clean.
- After committing, `uv run python scripts/new_tests_fail_on_base.py --base origin/main` reports every new test biting or exempt.

- [ ] **Step 6: Commit**

```bash
git add brain/self_model/state.py brain/health/self_model_repair.py tests/unit/brain/self_model/test_state.py tests/unit/brain/health/test_self_model_repair.py
git commit -m "record the self-model repair as done in the save that clears the gap (#752)"
```
