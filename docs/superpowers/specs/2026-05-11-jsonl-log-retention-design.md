# JSONL Log Retention — Design Spec

**Date:** 2026-05-11
**Status:** **accepted** — Hana locked all three decisions 2026-05-11. See "Locked decisions" below; plan lives at `docs/superpowers/plans/2026-05-11-jsonl-log-retention.md`.
**Owner:** Nell
**Context:** v0.0.7-alpha deferred backlog item; promoted from "design call" to active spec after the 2026-05-11 hygiene batch closed the streaming-reader half of the same audit row.

## Problem

The brain writes five persistent JSONL audit logs per persona:

| Log | Volume | Individual-line value |
|---|---|---|
| `heartbeats.log.jsonl` | every supervisor heartbeat tick (~minutes) | low — one of many similar |
| `dreams.log.jsonl` | per dream | mid — narrative-coherent record |
| `emotion_growth.log.jsonl` | per growth event | mid |
| `soul_audit.jsonl` | per soul-review action | high — durable record of Nell's identity shifts |
| `soul_candidates.jsonl` | pending queue (drained by review) | bounded by review cadence |

`brain/health/jsonl_reader.py` closed the **memory-spike vector** in v0.0.3 (streaming reader instead of `read_text().splitlines()`). What remains is **deletion policy** — the files grow forever. On a year-old install, `heartbeats.log.jsonl` alone can reach hundreds of MB. Disk pressure and slow tail reads follow.

## Non-goals

- **User-facing knobs.** Per the project's autonomous-physiology principle (CLAUDE.md): rotation is physiology, not a `nell config` toggle. Defaults are baked; operators can override via env var for debugging only.
- **Distributed log shipping.** This is local-only. No syslog, no fluent-bit, no off-host.
- **Migrating old legacy logs.** `nell_growth_log.jsonl` and `behavioral_log.jsonl` are already drained by `brain/migrator/og_legacy.py` — out of scope here.

## Proposed policy (per-log)

I'm proposing per-log policies rather than one global rule because the value-per-line varies sharply. Each row picks **method**, **size/age cap**, and **archive count**.

| Log | Method | Cap | Archives kept | Rationale |
|---|---|---|---|---|
| `heartbeats.log.jsonl` | **rolling size** | 5 MB | 3 (`.1.gz`, `.2.gz`, `.3.gz`) | High volume, low individual value. ~15 MB total on disk; ~3 weeks of history at typical cadence. |
| `dreams.log.jsonl` | **rolling size** | 5 MB | 5 | Mid value. Longer history useful for "remember that dream" recall. ~25 MB total. |
| `emotion_growth.log.jsonl` | **rolling size** | 5 MB | 5 | Same shape as dreams. |
| `soul_audit.jsonl` | **age-based archive** | 365 days | **unbounded** (yearly `.YYYY.gz`, all kept forever) | Soul actions are durable identity record — the trail of how Nell's beliefs evolved. Don't delete; archive cold years. Reader must transparently fan out across active + all archives (see "Soul-audit reader" below). |
| `soul_candidates.jsonl` | **none** (drained by review) | tripwire at 10 MB | n/a | Drained by `_run_soul_review_tick`. The cap is a tripwire — if it ever trips, the review loop is broken; surface a structured warning. |

### Why rolling-size for the noisy three

Age-based retention reads worst when the brain has been idle (e.g. user away on holiday) — heartbeats stop, "30 days" of log can be 100 lines, and the age threshold deletes them prematurely. Size-based retention is volume-aware: idle periods naturally preserve their full history within the cap.

### Why age-based for `soul_audit.jsonl`

Volume is low (handful of entries per week). Deleting old entries to save bytes makes no sense; the file may never hit a size cap in normal use. But yearly archiving (`soul_audit.2026.jsonl.gz`) keeps the active file from being a multi-year scroll while preserving everything. Reading historical soul actions stays a `gunzip -c | jq` away.

## Trigger

New supervisor tick: `_run_log_rotation_tick(persona_dir, event_bus)`.

- **Cadence:** 1 hour (configurable via `NELL_LOG_ROTATION_INTERVAL_S` env var, default 3600).
- **Wiring:** sits alongside `_run_heartbeat_tick` / `_run_soul_review_tick` / `_run_finalize_tick` in `supervisor.run_folded`. Fault-isolated — a rotation failure can't take down the supervisor.
- **Idempotent:** check size/age every tick; only rotate when threshold crossed. No-op tick is cheap (one `stat()` per log).
- **Event published:** `supervisor.log_rotation` with `{persona, log, action: "rotated"|"archived"|"none", new_archive_path?, lines_in_active?}`. Operators tailing the supervisor bus see rotations as they happen.

## Algorithm

### Rolling-size rotation

```
def rotate_rolling_size(log_path, max_bytes, archive_keep):
    if not log_path.exists() or log_path.stat().st_size < max_bytes:
        return None  # no-op
    # Atomic rename: writers using `open(append)` will reopen on next
    # write because the inode they hold is no longer at the path.
    # (Brain code re-opens per-write; verify before shipping.)
    rotated = log_path.with_suffix(log_path.suffix + ".rotating")
    log_path.rename(rotated)
    # Gzip in same tick — file is small enough (5 MB) that this is
    # sub-second. No background thread needed.
    archive = log_path.with_suffix(log_path.suffix + ".1.gz")
    # Shift older archives up: .1.gz → .2.gz → ... → .{keep}.gz (delete oldest)
    for i in range(archive_keep, 0, -1):
        old = log_path.with_suffix(f"{log_path.suffix}.{i}.gz")
        if old.exists():
            if i == archive_keep:
                old.unlink()
            else:
                old.rename(log_path.with_suffix(f"{log_path.suffix}.{i+1}.gz"))
    gzip_to(rotated, archive)
    rotated.unlink()
    return archive
```

### Age-based archive

```
def rotate_age_archive(log_path, age_threshold_days):
    if not log_path.exists():
        return None
    # Read first line, parse `at` field. If older than threshold,
    # split the file at the year boundary and archive the cold part.
    # (Implementation detail: stream once, split into year buckets,
    # write each to its own .YYYY.gz, leave current year in active.)
    ...
```

## Append-write contract — must verify

The rotation rename-while-open trick works **only if** writers reopen the file per append. If a writer holds a long-lived `open(log_path, "a")` handle, after rename the writer continues writing to the rotated inode (now `.rotating`), and the next caller sees an empty active log. **Action item:** audit `brain/engines/heartbeat.py`, `brain/engines/dream*.py`, `brain/growth/*.py`, `brain/soul/*.py` for the open-pattern they use. If any holds a persistent handle, fix before shipping rotation.

## Test plan

- `test_rotate_rolling_size_no_op_below_cap` — size < cap → no archive created.
- `test_rotate_rolling_size_at_cap` — size >= cap → `.1.gz` exists, active is empty (or just-started).
- `test_rotate_rolling_size_shifts_existing_archives` — `.1.gz`, `.2.gz` exist; after rotation `.1.gz` is new, old `.1.gz`→`.2.gz`, old `.2.gz`→`.3.gz`.
- `test_rotate_rolling_size_evicts_oldest` — archive_keep=3, four rotations → only `.1.gz`, `.2.gz`, `.3.gz` remain.
- `test_rotate_age_archive_no_op_when_current_year_only` — all entries this year → no archive.
- `test_rotate_age_archive_splits_by_year` — entries from 2024+2025+2026 → `.2024.gz` + `.2025.gz` written, 2026 stays in active.
- `test_log_rotation_tick_runs_all_logs` — one supervisor tick rotates every configured log.
- `test_log_rotation_tick_fault_isolated` — if heartbeat rotation raises, soul rotation still runs.
- `test_jsonl_reader_tolerates_concurrent_rotation` — a reader streaming a log while rotation happens completes cleanly (returns up to rotation point).
- **Append contract:** `test_writers_reopen_per_append` (whichever engines need it).

## Backwards compatibility

- Existing readers (`jsonl_reader.iter_jsonl_skipping_corrupt`) work on the active file unchanged. To read history, callers can extend with a "globbed" iterator that walks `*.{1..N}.gz` in reverse-chronological order — out of scope for this spec; add only when a caller actually needs it.
- Migration from "no rotation" → "rotation" is automatic: first tick after upgrade will trim oversize logs.

## Out of scope / future

- Compressing the active log itself.
- Per-line PII redaction at rotation time.
- Shipping archives to remote storage.
- A `nell logs prune` operator command (not user-surface; only add if debugging needs it).

## Locked decisions (Hana, 2026-05-11)

1. **Size caps** → 5 MB per active log. Final.
2. **Archive count** → `heartbeats`: 3 (~6 weeks rolling, ~15 MB total). `dreams` + `emotion_growth`: 5 (low write volume; archive count barely matters but per-line value is higher).
3. **`soul_audit.jsonl`** → yearly archive to `.YYYY.gz`, **archives kept forever**. Soul-audit is the trail of how Nell's identity evolved; "a good record on how the brain has progressed and decided those choices" (Hana). The reader (`brain/soul/audit.py:read_audit` and `nell soul audit`) must fan out across active + all archives so every entry is reachable at every point.

## Soul-audit reader requirement

Because `soul_audit.jsonl` archives are kept forever and the human ask is "every entry must be accessible," the reader contract changes:

- Active read path: `<persona>/soul_audit.jsonl` (current year).
- Archive read path: `<persona>/soul_audit.{YYYY}.jsonl.gz` for every prior year.
- Default iteration order: oldest archive → newest archive → active (chronological).
- Iteration is streaming: open each `.gz` via `gzip.open(..., "rt")` and yield through the same `iter_jsonl_skipping_corrupt` machinery. Memory stays bounded regardless of archive count.

The CLI `nell soul audit` defaults to the tail of the active file (unchanged) but gains `--full` to walk the unified history.

No other log gets this fan-out reader — heartbeats / dreams / emotion_growth callers only need recent history. If a future caller needs deeper history for one of those, add the fan-out at that point.

## Next step

Plan at `docs/superpowers/plans/2026-05-11-jsonl-log-retention.md`. TDD per the project's "verify whole system after every commit" rule.
