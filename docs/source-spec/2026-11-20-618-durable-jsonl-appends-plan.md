# Durable JSONL appends and the one-time NUL repair (#618) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `append_jsonl` fsyncs by default with reasoned opt-outs, and a one-time startup repair removes the NUL runs an unclean stop left in a persona's JSONL files.

**Architecture:** One default changes in `brain/health/json_io.py`, guarded by an AST test over `brain/`. A new provider-free startup migration, `brain/health/jsonl_nul_repair.py`, streams each damaged file through a NUL-run filter into an atomic replace, keeps a `.pre-618` copy, and records completion in a state file. It is called from `run_folded`'s startup block like `soul_candidate_repair`.

**Tech Stack:** Python 3.12+, pytest, `uv run`.

**Spec:** `docs/source-spec/2026-11-20-618-durable-jsonl-appends-design.md`

## Global Constraints

- TDD: each test is written and seen failing before its code (tdd-guard enforces one failing test at a time; run red tests in this worktree with `-o tdd_guard_project_root=<the main checkout>`).
- Full gate before each commit: `uv run pytest` with 0 `FAILED`, `uv run ruff check .`, `uv run python scripts/lint_windows_pitfalls.py`. No `app/` change, so no frontend gate.
- Read and write text with `encoding="utf-8"`; any relative path used as a key or compared is `.as_posix()` (CLAUDE.local.md Windows gotchas).
- No whole-file load of a JSONL file: scan and rewrite in chunks.
- No hand-rolled atomic write (`tests/unit/test_no_hand_rolled_atomic_write.py`): publish through `brain/utils/atomic_write.py`.
- Persisted state is additive: the new state file is read with defaults and tolerates unknown keys.
- Opt-out comment marker, exactly: `# fsync off:` followed by the reason.
- Names: state file `jsonl_nul_repair_state.json`; backup suffix `.pre-618` (then `.pre-618.1`, `.pre-618.2`, …).
- Commit messages: imperative, present tense, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. An append to a file while the repair rewrites it: the append waits for the repair's `file_lock(path)`, and no record is lost. Pinned in Task 2.
2. A NUL run split across two read chunks: repaired exactly as if it were in one chunk. Pinned in Task 2 (`chunk_size=4`).
3. A file that is only NULs: becomes an empty file, with its backup kept. Pinned in Task 2.
4. A `.pre-618` backup already present (a run that repaired a file but failed elsewhere before writing the state): the new backup is `.pre-618.1`, and the old one is unchanged. Pinned in Task 2.
5. A NUL run directly before a newline after a torn record (`torn\0\0\n`): becomes `torn\n`, not a blank line. Pinned in Task 2.

---

### Task 1: `append_jsonl` fsyncs by default, with reasoned opt-outs

**Files:**
- Modify: `brain/health/json_io.py` (`append_jsonl` signature and docstring)
- Modify: `brain/emotion/engine_files.py` (`_append_line`: pass `fsync=False` with a `# fsync off:` comment)
- Modify: `brain/memory/recall_diagnostics.py:189-191` (comment becomes `# fsync off: ...`)
- Modify: `tests/unit/brain/health/test_json_io.py` (`test_ca5_fsync_is_opt_in_for_appends` is replaced)
- Modify: `tests/unit/brain/emotion/test_engine_queue.py` (`test_the_append_asks_append_jsonl_for_a_held_lock_and_logging_with_no_fsync`: expected kwargs gain `"fsync": False`)
- Create: `tests/unit/brain/health/test_append_jsonl_durable_618.py`
- Create: `tests/unit/test_append_jsonl_fsync_opt_outs.py`

**Interfaces:**
- Produces: `append_jsonl(path, records, *, lock="block", fsync: bool = True, on_error="raise", nonfinite="raise") -> int | None` (only the default changes).

- [ ] **Step 1: Write the failing crash test** in `test_append_jsonl_durable_618.py`:
  - Helper `_crash_after(monkeypatch, path) -> Callable[[], None]`: monkeypatch `os.fsync` to record `os.fstat(fd).st_size` as the file's synced size (and call the real `fsync`); the returned `crash()` keeps the file's length and overwrites every byte after the largest recorded synced size with `b"\0"`.
  - `test_a_default_append_survives_an_unclean_stop`: `append_jsonl(p, {"n": 1})`, `crash()`, then `list(iter_jsonl(p)) == [{"n": 1}]`.
- [ ] **Step 2: Run it and see it fail**: `uv run pytest tests/unit/brain/health/test_append_jsonl_durable_618.py -o tdd_guard_project_root=... -q`. Expected: FAIL (the record reads as NUL; `iter_jsonl` yields nothing).
- [ ] **Step 3: Change the default** to `fsync: bool = True` in `append_jsonl`, and add to its docstring: an unclean machine stop can leave a just-appended record as NUL bytes (#618), so appends are fsynced unless the caller passes `fsync=False` with a `# fsync off:` reason.
- [ ] **Step 4: Run it and see it pass.** Same command. Expected: PASS.
- [ ] **Step 5: Replace `test_ca5_fsync_is_opt_in_for_appends`** with `test_ca5_appends_fsync_by_default`: with `_count_fsyncs`, `append_jsonl(a)` records a call; `append_jsonl(b, fsync=False)` adds none.
- [ ] **Step 6: Write the failing guard** `tests/unit/test_append_jsonl_fsync_opt_outs.py::test_every_fsync_opt_out_states_its_reason`: parse every `brain/**/*.py` (utf-8) with `ast`; for each call whose function name is `append_jsonl` and that passes `fsync=` anything other than the constant `True`, the call's first line or one of the 3 lines above it contains `# fsync off:`. The failure message lists `path:line` (posix). Run it: FAIL on `brain/memory/recall_diagnostics.py:191` (its comment says "No fsync").
- [ ] **Step 7: Reword the `recall_diagnostics.py` comment** to start `# fsync off: this runs on the recall path (I6) and the data is diagnostic.`, keeping the lock sentence. Run the guard: PASS.
- [ ] **Step 8: Emotion engine opt-out.** In `test_engine_queue.py`, the expected dict becomes `{"lock": "held", "on_error": "log", "fsync": False, "path": ...}`; run it: FAIL. In `engine_files._append_line`, pass `fsync=False` with the comment `# fsync off: the engine's design (section 2.9) chose no fsync per append; asked of ToT on #618's PR.` Run that test and the guard: PASS.
- [ ] **Step 9: Full gate** (Global Constraints). Expected: 0 FAILED. If another test pins the old default, change it to the new default in this task and name it in the commit body.
- [ ] **Step 10: Commit**: `make JSONL appends durable by default (#618)`.

### Task 2: The one-time NUL repair

**Files:**
- Modify: `brain/utils/atomic_write.py` (add `save_stream_atomic`)
- Modify: `docs/shared-utilities.md` (one row under `brain/utils/atomic_write.py`)
- Create: `brain/health/jsonl_nul_repair.py`
- Create: `tests/unit/brain/health/test_jsonl_nul_repair.py`

**Interfaces:**
- Produces:
  - `save_stream_atomic(path: Path, write: Callable[[BinaryIO], None], *, fsync: bool = True) -> None`: like `save_bytes_atomic`, but `write(fh)` fills the temp. Built on `_save_payload` with a `write_temp` that opens the temp `"wb"`, calls `write`, flushes and (when `fsync`) fsyncs.
  - `STATE_FILENAME = "jsonl_nul_repair_state.json"`, `BACKUP_SUFFIX = ".pre-618"`.
  - `@dataclass(frozen=True) class FileRepair: runs: int; bytes_removed: int; backup: str` (`backup` is the posix path relative to the persona folder).
  - `@dataclass(frozen=True) class RepairReport: scanned: int; repaired: dict[str, FileRepair]; failed: list[str]; done: bool` (keys and `failed` are posix relative paths).
  - `should_run_jsonl_nul_repair(persona_dir: Path) -> bool`: false only when the state file reads as an object with `"done": true`; a missing or unreadable file is true.
  - `run_jsonl_nul_repair(persona_dir: Path, *, chunk_size: int = 1 << 20) -> RepairReport`.

**The filter (an algorithm the tests alone don't fix).** Read the source in `chunk_size` pieces and write everything that isn't a raw `b"\0"`. A run of NULs is dropped. At its end, emit one `b"\n"` only when the byte before the run is not `b"\n"`, the run isn't at the start of the file, and the byte after it isn't `b"\n"`. A run that ends the file gets the newline only when the byte before it isn't `b"\n"`. The state (last written byte, inside a run, run length) carries across chunks. The filter returns `(runs, bytes_removed)`.

**Which files.** `os.walk(persona_dir)`, names ending `.jsonl` or matching `r".+\.jsonl\.\d+"`; names containing `BACKUP_SUFFIX` and anything ending `.gz` are skipped. A file is damaged when a chunked scan finds `b"\0"`. For each damaged file, under `file_lock(path)`: stream a copy to the first free name of `<name>.pre-618`, `<name>.pre-618.1`, … with `save_stream_atomic`, then rewrite the file with `save_stream_atomic` through the filter, reading the original. An `OSError` on a file adds it to `failed` and moves on. `done` is true when `failed` is empty, and only then is the state file written with `write_json`: `{"done": true, "completed_at": <iso utc>, "repaired": {path: {"runs", "bytes_removed", "backup"}}}`. Log one info line per repaired file and one summary line.

- [ ] **Step 1: Write the failing test** `test_a_nul_run_at_a_line_start_is_removed_and_the_swallowed_record_comes_back`: file `b'{"a":1}\n' + b"\0"*20 + b'{"b":2}\n'` → after `run_jsonl_nul_repair`, `list(iter_jsonl(p)) == [{"a": 1}, {"b": 2}]`, `report.repaired["x.jsonl"].runs == 1`, `.bytes_removed == 20`, and `x.jsonl.pre-618` holds the original bytes. Run: FAIL (module missing).
- [ ] **Step 2: Implement** `save_stream_atomic`, its `docs/shared-utilities.md` row, and `jsonl_nul_repair.py` per the Interfaces and the filter rule. Run: PASS.
- [ ] **Step 3: One test at a time, each red then green** (all in `test_jsonl_nul_repair.py`):
  - `test_a_run_after_a_torn_record_becomes_a_newline`: `b'{"a":1}\n{"to' + b"\0"*8 + b'{"b":2}\n'` → bytes `b'{"a":1}\n{"to\n{"b":2}\n'`.
  - `test_a_run_before_a_newline_after_a_torn_record_adds_no_blank_line` (Review Focus 5): `b'{"to' + b"\0"*4 + b'\n{"b":2}\n'` → `b'{"to\n{"b":2}\n'`.
  - `test_a_run_at_the_end_of_the_file_is_dropped`: `b'{"a":1}\n' + b"\0"*30` → `b'{"a":1}\n'`.
  - `test_a_file_of_only_nuls_becomes_empty_with_its_backup_kept` (Review Focus 3).
  - `test_a_run_split_across_chunks_is_repaired_the_same` (Review Focus 2): the first test's input with `chunk_size=4` gives the same bytes as with the default.
  - `test_a_second_run_changes_nothing`: run twice; second report has `repaired == {}`, file bytes unchanged, no `.pre-618.1`.
  - `test_an_existing_backup_is_kept_and_the_new_one_is_numbered` (Review Focus 4): a pre-existing `x.jsonl.pre-618` with other bytes stays byte-identical; the new copy is `x.jsonl.pre-618.1`.
  - `test_rotated_files_are_repaired_and_gz_files_are_not`: `x.jsonl.1` repaired; `x.jsonl.1.gz` holding NULs is byte-identical afterwards.
  - `test_the_reader_warns_no_more_once_the_file_is_repaired`: with `caplog` at WARNING, reading the damaged file logs a warning before the repair and none after.
  - `test_an_append_during_the_repair_waits_and_is_kept` (Review Focus 1): patch the filter (via a seam or by wrapping `save_stream_atomic`) to call `append_jsonl(p, {"c": 3}, lock="try")` while the repair holds the lock: it returns `None`. After the repair, a blocking `append_jsonl(p, {"c": 3})` lands on its own line after the repaired records.
  - `test_an_unreadable_file_leaves_the_state_unwritten_and_the_next_run_repairs_it`: patch `open` for one path to raise `PermissionError` on the first run; `report.done is False`, `report.failed == ["x.jsonl"]`, no state file, `should_run_jsonl_nul_repair` true; the second run repairs it and writes the state; `should_run_jsonl_nul_repair` false.
- [ ] **Step 4: Full gate.** Expected: 0 FAILED.
- [ ] **Step 5: Commit**: `repair the NUL runs an unclean stop left in JSONL files, once (#618)`.

### Task 3: Run the repair at supervisor start

**Files:**
- Modify: `brain/bridge/supervisor.py` (imports beside `_soul_candidate_repair_*`; a startup block)
- Modify: `tests/unit/brain/bridge/test_central_cadence.py` (`_NON_GATED_ONE_SHOTS` gains `"_jsonl_nul_repair_run"`)
- Test: `tests/unit/brain/health/test_jsonl_nul_repair.py`

**Interfaces:**
- Consumes: `should_run_jsonl_nul_repair`, `run_jsonl_nul_repair` from Task 2, imported as `_jsonl_nul_repair_should_run` and `_jsonl_nul_repair_run`.

- [ ] **Step 1: Write the failing test** `test_supervisor_startup_repairs_a_damaged_file`: build the persona the way `test_soul_candidate_repair.py::test_repair_runs_from_supervisor_startup_once` does (`active_conversations/`, `persona_config.json` with `{"provider": "fake", "searcher": "noop"}`), add `heartbeats.log.jsonl` with the Task 2 Step 1 bytes, call the real `run_folded(stop, persona_dir=..., provider=FakeProvider(), event_bus=EventBus(), tick_interval_s=0.1, heartbeat_interval_s=None)` with `stop` already set, and assert the file now reads both records and `jsonl_nul_repair_state.json` holds `"done": true`. No patching of the repair. Run: FAIL.
- [ ] **Step 2: Add the startup block** to `run_folded` as the first one-shot, immediately before the attunement backfill block (`# One-shot startup: run the attunement backfill ...`), since later one-shots read JSONL files: `if _jsonl_nul_repair_should_run(persona_dir): _jsonl_nul_repair_run(persona_dir)` in its own `try/except Exception` that logs `"jsonl NUL repair failed during startup: %s"` as a warning. A comment names #618 and says it is provider-free and runs before the readers. Add `"_jsonl_nul_repair_run"` to `_NON_GATED_ONE_SHOTS`. Run the test and `test_central_cadence.py`: PASS.
- [ ] **Step 3: Full gate.** Expected: 0 FAILED.
- [ ] **Step 4: Commit**: `run the JSONL NUL repair at supervisor start (#618)`.

### Task 4: Record the defers

**Files:**
- Modify: the maintainer's `project_companion_emergence_deferred.md` memory note (outside the repo; not committed)

- [ ] **Step 1:** Add two active defers with this branch's name: `F_FULLFSYNC` on macOS (plain `fsync` leaves the drive cache unflushed; repo-wide decision), and #618 item 3 (the cause of the original stops, on the filer's machine). Each says "raise in the next brainstorm". The spec's §Deferred already lists both.
