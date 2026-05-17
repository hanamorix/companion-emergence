# JSONL Log Retention — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-05-11-jsonl-log-retention-design.md` (accepted 2026-05-11)
**Branch:** `feature/v008-deferred-fixes` (worktree `.worktrees/v008-deferred/`)
**Method:** TDD per phase; **full pytest suite gates every commit** (Hana's strict rule).

## Phase ordering rationale

Building pure rotation primitives first (Phase 2) means later phases consume a tested API. Phase 1 verifies the append-write contract *before* writing rotation code that depends on it — discovering mid-implementation that a writer holds a long-lived file handle would force a redesign.

---

## Phase 1 — Verify append-write contract ✅ (no work needed)

**Audit result (2026-05-11):**

| Writer | Location | Pattern |
|---|---|---|
| heartbeat | `brain/engines/heartbeat.py:1068` | `with self.heartbeat_log_path.open("a", ...)` per call |
| dream | `brain/engines/dream.py:226` | `with self.log_path.open("a", ...)` per call |
| growth log | `brain/growth/log.py:74` | `with open(path, "ab")` per call |
| soul audit | `brain/soul/audit.py:65` | `with open(audit_path, "a", ...)` per call |

All four reopen per append — the rotation-via-rename pattern is safe. No writer fixes needed and no dedicated contract test added; Phase 4's integration tests exercise the rotation end-to-end through real engines and serve as the regression guard.

## Phase 1 (original, now resolved)

**Goal:** confirm every JSONL log writer reopens the file per append (vs holding a persistent handle). If a writer holds a long-lived handle, rotation-via-rename leaves it writing to the rotated inode after the rename — the new active file stays empty until the writer reopens.

### Tasks

1. **Audit the four writer call sites:**
   - `brain/engines/heartbeat.py` — `heartbeats.log.jsonl` appender
   - `brain/engines/dream*.py` — `dreams.log.jsonl` appender (find exact path)
   - `brain/growth/*.py` or `brain/engines/growth*.py` — `emotion_growth.log.jsonl` appender
   - `brain/soul/audit.py:append_audit_entry` — `soul_audit.jsonl` appender

   For each, capture the open pattern: `with open(path, "a") as f` per call (good — reopens) vs `self._fh = open(path, "a")` stored on the instance (bad — must fix first).

2. **Write a contract test** at `tests/integration/test_log_append_contract.py`:
   - For each writer module, trigger one append, rename the log file mid-flight, trigger a second append, assert the second append landed in the new active file (proving reopen-per-append).

3. **Fix any writer that fails the contract:** convert to `with open(...)` per append. Trivial change; cost is one extra syscall per write, which is negligible at our cadence.

**Commit:** `test(retention): append-write contract for all JSONL log writers` + any necessary writer fixes in a follow-up commit.

**Gate:** full pytest suite green.

---

## Phase 2 — Rotation primitives

**Goal:** pure functions, no supervisor wiring yet, fully tested.

New module: `brain/health/log_rotation.py`.

### API surface

```python
def rotate_rolling_size(
    log_path: Path,
    max_bytes: int,
    archive_keep: int,
) -> Path | None:
    """Rotate log_path if it exceeds max_bytes. Returns the new .1.gz path on
    rotation, or None if no rotation was needed."""

def rotate_age_archive_yearly(
    log_path: Path,
    now: datetime | None = None,
) -> list[Path]:
    """Split log_path's entries by year. Cold years move to .YYYY.jsonl.gz;
    current year remains in active. Returns the list of newly-written
    archive paths (empty if no archiving was needed)."""
```

### Tests at `tests/unit/brain/health/test_log_rotation.py`

**Rolling-size:**
- `test_rotate_rolling_size_no_op_below_cap` — file < cap → no archive, returns None.
- `test_rotate_rolling_size_at_cap` — file >= cap → `.1.gz` exists, active is empty/recreated, returns the new archive path.
- `test_rotate_rolling_size_shifts_existing_archives` — `.1.gz` + `.2.gz` exist; rotate → new `.1.gz`, old `.1.gz`→`.2.gz`, old `.2.gz`→`.3.gz`.
- `test_rotate_rolling_size_evicts_oldest_when_keep_reached` — `archive_keep=3`, four rotations → only `.1.gz`/`.2.gz`/`.3.gz` remain.
- `test_rotate_rolling_size_archive_content_round_trips` — gzip → gunzip yields the original bytes line-for-line.
- `test_rotate_rolling_size_handles_missing_log` — log doesn't exist → returns None, no error.

**Yearly archive:**
- `test_rotate_age_archive_no_op_when_current_year_only` — all entries this year → no archive written.
- `test_rotate_age_archive_splits_by_year` — entries from 2024 + 2025 + 2026, now=2026-05-11 → `.2024.jsonl.gz` + `.2025.jsonl.gz` written, 2026 entries stay in active.
- `test_rotate_age_archive_preserves_chronological_order_within_year` — entries within a year preserve their original ordering in the archive.
- `test_rotate_age_archive_idempotent` — calling twice in the same year doesn't duplicate archives or re-archive current-year entries.
- `test_rotate_age_archive_skips_corrupt_lines` — uses `iter_jsonl_skipping_corrupt` machinery so a malformed line is skipped, not aborted on.

**Atomicity:**
- `test_rotate_atomic_no_partial_archive_on_gzip_failure` — simulate gzip write failure mid-rotation; assert the active log is intact (no half-written state).

**Commit:** `feat(retention): rotation primitives for rolling-size and yearly archive`.

**Gate:** full pytest suite green.

---

## Phase 3 — Soul-audit fan-out reader

**Goal:** `read_audit` and the streaming variant iterate active + all yearly archives in chronological order.

### Changes

- `brain/soul/audit.py`:
  - Add `iter_audit_full(persona_dir) -> Iterator[dict]` — yields oldest archive → newest archive → active, streaming.
  - Existing `read_audit(persona_dir)` stays active-file-only (preserves existing CLI tail behaviour).
- `brain/health/jsonl_reader.py`: add `iter_jsonl_streaming(path_or_gz: Path)` that auto-detects `.gz` suffix and opens via `gzip.open(..., "rt")`. Re-use `iter_jsonl_skipping_corrupt`'s per-line resilience.

### Tests at `tests/unit/brain/soul/test_audit_full_reader.py`

- `test_iter_audit_full_active_only` — no archives → yields just active-file lines.
- `test_iter_audit_full_chronological_across_archives` — active + `.2024.jsonl.gz` + `.2025.jsonl.gz` → yields 2024 entries first, then 2025, then active (in their original within-file order).
- `test_iter_audit_full_skips_corrupt_in_archives` — corrupt line inside a `.gz` archive is skipped with a warning, not aborted on.
- `test_iter_audit_full_streaming_memory_bounded` — large synthetic archive (~100k lines) iterates without loading all into memory (measure peak RSS or use a generator-only assertion).

**Commit:** `feat(retention): soul-audit fan-out reader spanning archives`.

**Gate:** full pytest suite green.

---

## Phase 4 — Supervisor rotation tick

**Goal:** wire `_run_log_rotation_tick` into `brain/bridge/supervisor.py::run_folded` alongside the existing heartbeat/soul-review/finalize ticks.

### Config

Per-log policy table baked at module load, env-overridable for debugging:

```python
_LOG_POLICIES = (
    LogPolicy("heartbeats.log.jsonl", rolling_size_mb=5, archive_keep=3),
    LogPolicy("dreams.log.jsonl",      rolling_size_mb=5, archive_keep=5),
    LogPolicy("emotion_growth.log.jsonl", rolling_size_mb=5, archive_keep=5),
    LogPolicy("soul_audit.jsonl",      strategy="yearly"),
)

# Tick cadence: 1 hour, overridable via NELL_LOG_ROTATION_INTERVAL_S.
```

The `soul_candidates.jsonl` tripwire (10 MB → emit structured warning, no rotation) is included here.

### Tests

- `test_run_log_rotation_tick_rotates_oversize_rolling_log` — write oversize heartbeats log → tick → `.1.gz` exists.
- `test_run_log_rotation_tick_archives_old_year_in_soul_audit` — soul_audit has 2024 + 2025 + 2026 entries → tick → `.2024.gz` + `.2025.gz` written.
- `test_run_log_rotation_tick_no_op_when_all_logs_within_caps` — small logs → tick → no files created.
- `test_run_log_rotation_tick_fault_isolated_per_log` — make heartbeat rotation raise → soul rotation still runs (catch + log + continue).
- `test_run_log_rotation_tick_emits_event_per_rotation` — `supervisor.log_rotation` event published with `{persona, log, action, archive_path?}`.
- `test_run_log_rotation_tick_soul_candidates_tripwire` — soul_candidates.jsonl > 10MB → warning event published, no rotation attempted.
- **Cadence integration test** at `tests/integration/test_supervisor_log_rotation_cadence.py` — supervisor runs for N synthetic ticks; only ticks past the configured interval invoke rotation.

### Wiring

In `run_folded` (after the finalize tick):

```python
if (
    log_rotation_interval_s is not None
    and last_log_rotation_at is not None
    and time.monotonic() - last_log_rotation_at >= log_rotation_interval_s
):
    try:
        _run_log_rotation_tick(persona_dir, event_bus)
    except Exception:
        logger.exception("supervisor log-rotation tick raised")
    last_log_rotation_at = time.monotonic()
```

**Commit:** `feat(retention): supervisor log-rotation tick + per-log policies`.

**Gate:** full pytest suite green.

---

## Phase 5 — CLI `nell soul audit --full`

**Goal:** operator-tier command for walking the unified soul-audit history.

### Changes

- `brain/cli.py` — add `--full` flag to `nell soul audit`. Default behaviour (active-file tail) unchanged.
- New handler dispatches through `iter_audit_full` and pages output sensibly (no buffering whole history in memory).

### Tests

- `test_cli_soul_audit_default_tails_active_only` — preserves existing behaviour.
- `test_cli_soul_audit_full_emits_all_entries` — active + archives → all lines on stdout in chronological order.
- `test_cli_soul_audit_full_no_archives_works` — falls back gracefully to active-only.

**Commit:** `feat(cli): nell soul audit --full walks archived years`.

**Gate:** full pytest suite green.

---

## Phase 6 — Documentation + memory updates

- `docs/roadmap.md`: ✅ added "Recently shipped" entry for the JSONL
  retention batch.
- `.public-sync/changelog-public.md`: **deferred to release-cut time.**
  The `.public-sync/` directory is gitignored and lives in the parent
  worktree only; the changelog entry will be added when Hana cuts the
  next release (v0.0.8-alpha or higher) per the project's three-file
  version-bump rule.
- Auto-memory at `~/.claude/projects/-Users-hanamori/memory/`: updated
  `project_companion_emergence_deferred.md` so JSONL retention moves
  from "active design call" into the resolved section.

**Commit:** `docs(retention): roadmap entry + plan close-out`.

**Gate:** full pytest suite green.

---

## Risk register

- **Append-write contract violation** — caught by Phase 1 audit + test.
- **Race between rotation and a concurrent writer** — POSIX `rename` is atomic; a writer that holds a stale handle continues writing to the renamed inode (which we immediately gzip). The next reopen lands on the new active file. No lost data; the in-flight append goes into the archive, where it belongs anyway.
- **Gzip mid-tick blocks the supervisor** — 5 MB gzips in sub-second; no background thread needed. If profiling later shows otherwise, move to a worker.
- **Reader sees mid-rotation state** — gzip is written to a `.tmp` then renamed; readers either see no archive yet or the complete archive.

## Out of scope (deferred to future spec if needed)

- Reading dreams / heartbeats / emotion_growth archives (no current caller needs deep history for these — soul_audit is the only one where Hana asked for forever access).
- Compressing the active log itself.
- Per-line PII redaction at rotation time.
- Shipping archives off-host.
- `nell logs prune` operator command.
