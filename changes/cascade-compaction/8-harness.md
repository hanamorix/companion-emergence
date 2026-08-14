# 8-harness.md — Cascade Compaction (stage 8)

**Posture:** the gc config declares regression **advisory-only** (no replay/held workload on this VM — no live
bridge). "Done" therefore rests on **conformance** — the C1–C22 gating criteria (1.5), verified by execution
on scratch pytest fixtures that drive the real governed paths. Frozen criteria hash (gate-4):
`6a12be35e71558237412e8faad5956960c0117bc45ebe3fed42fdee50166a914` — **re-verified unchanged at stage 8**
(see foot). Build base `cd29bc61`; 7 commits.

**Harness validation (H6 — "an unreviewed check is not a check"):** the C1–C22 fixtures were authored during
the build (stage 5) and are cold-reviewed at **stage 6** (the reviewer was explicitly charged to confirm each
fixture drives the REAL governed path — not a unit proxy — and FAILS against the pre-change/known-bad variant).
The fail-demo tests below (C4/C10/C12/C16/C18/C21 carry explicit `*_fail_demo` / `*_pre_fix` / `*_faildemo`
siblings) are the executed shown-able-to-fail evidence.

## CI (independently re-run by the orchestrator-runner, not accepted from the build agent)
- `uv run ruff check .` → **All checks passed!**
- `uv run pytest -m "not live and not requires_claude_cli and not integration"` (py3.12):
  - **Pre-C-1-fix run:** `1 failed, 4250 passed, 19 skipped, 1 xfailed in 230s`.
  - **Post-C-1-fix run (a485e72d):** `[foot — same shape; +1 new C-1 regression test]`.
  The sole failure `tests/unit/brain/initiate/test_review.py::test_review_tick_gate_blocks_send_records_hold`
  is **PRE-EXISTING**: reproduced identically at base `cd29bc61` in the main worktree (1 failed / 25 passed), a
  time/date-dependent flake in `brain/initiate/` **untouched by this branch** (`git diff --stat
  cd29bc61..HEAD` shows no `initiate/` files).
- Cascade-specific suite (the C1–C22 fixtures): **43 passed in 2.75s.**

## Per-criterion verification table (H7)

| Crit | Gating | Test(s) — real path exercised | Verified by execution? | Result |
|---|---|---|---|---|
| C1 | gating | `test_c1_three_sections` — real `cascade_conversation` → parse row | yes | PASS |
| C2 | gating | `test_c2_age_gated_wiring` — real pass, age-partition + classify recorded | yes | PASS |
| C3 | gating | `test_c3_tier1_hard_cap`, `test_c3_terminal_tier3_hard_cap_multi_cycle` (tier3 cap over many cycles) | yes | PASS |
| C4 | gating | `test_c4_fold_validation_unit`, `test_c4_cascade_double_reject_preserves_marker` (real cascade, both inputs survive) | yes | PASS |
| C5 | gating | `test_c5_temporal_markers_and_labels` — coarse span + owner labels | yes | PASS |
| C6 | gating | `test_c6_head_prefix_bytestable_and_reparse` — render×2 byte-eq + real `apply_budget` re-parse | yes (execution, H3) | PASS |
| C7 | gating | `test_c7_cursor_guard_and_idempotence` — un-extracted no-op, byte-verify, double-run no-op | yes | PASS |
| C8 | gating | `test_c8_idle_rollover_sync_and_selection` — real `/sessions/active` TestClient + multi-stale selection | yes | PASS |
| C9 | gating | `test_c9_weekly_rollover_daily_tick` — real `_run_compaction_tick`, quiet-gap defer | yes | PASS |
| C10 | gating | `test_c10_finalize_no_delete_interleave` + `..._pre_change_finalize_deleted_buffer_fail_demo` | yes (interleave + fail-demo, H4) | PASS |
| C11 | gating | `test_c11_archive_segments_reader_crash` — segments, in-order reader, crash-mid-roll | yes | PASS |
| C12 | gating | `test_c12_legacy_migrates_to_tier3_and_stays_tier3`, `..._idempotent_rerun_is_noop`, `..._delayed_backlog_flatten_self_heals_to_tier3`, `..._faildemo_covers_until_would_mislabel_tier1` | yes (+ fail-demo) | PASS (4) |
| C13 | gating | `test_c13_interior_not_starved` — monologue_trace read after cascade | yes | PASS |
| C14 | gating | `test_c14_graduation_and_terminal_persistence`, `test_c14_multi_input_and_long_inactivity` — marker-preserving provider, asserts on marker (not covers_until_ts) | yes | PASS |
| C15 | gating | `test_c15_active_set_bounded_after_rollover` — old buffer reaped, active set bounded | yes | PASS |
| C16 | gating | `test_c16_post_rollover_continuation_redirect`, `test_c16_multi_generation_and_cyclic_redirect` — real TestClient, sid1→sid2→sid3, cyclic aborts | yes | PASS |
| C17 | gating | `test_c17_cascade_write_atomic` — one `rewrite_session_atomic`, failure-before-replace leaves pre-pass row | yes | PASS |
| C18 | gating | `test_c18_carried_raw_tail_extraction_state` + `..._without_carried_cursor_would_reextract_fail_demo` | yes (+ fail-demo) | PASS |
| C19 | gating | `test_c19_inflight_lock_keyed_by_resolved_sid` — two clients, one lock key | yes | PASS |
| C20 | gating | `test_c20_close_cleanup_uses_resolved_sid` — close on old sid reaps successor, /state correct | yes | PASS |
| C21 | gating | `test_c21_no_raw_sid_downstream_of_resolution` (structural) + `test_c21_flags_all_five_sites_on_pre_fix_base_commit` (shown-able-to-fail on base) | yes (structural + fail-demo) | PASS |
| C22 | gating | `test_c22_apply_budget_sectioned_row` — real backstop on a sectioned row | yes | PASS |
| **C-1** | gating (stage-6 regression) | `test_c1_mid_rollover_window_redirects_not_resurrect` — the mid-rollover window (pointer written + old still cached + old buffer present) resolves to the successor, not a resurrectable old buffer | yes — **shown-able-to-fail proven**: stashing the session.py fix → the test FAILS (resolves to old_sid); with the fix → PASS | PASS |
| **RP-1** | gating (stage-6-r3 regression) | `test_owner_idle_gate_defers_busy_session` — the supervisor compaction tick SKIPS a busy session (no cascade, no rollover) and fires when idle; `test_weekly_rollover_belt_defers_when_busy` — `maybe_weekly_rollover` defers when `is_session_busy` true. Best-effort UX/efficiency belt (don't churn an active session) — NOT the race-safety mechanism (see RP-2) | yes — busy→no-roll assertion is the discriminator (pre-gate would roll) | PASS |
| **RP-2** | gating (stage-6-r4 regression) | `test_persist_after_rollover_redirects_to_successor` (a late persist for a rolled-over sid threads into the successor, no resurrection) + `test_plain_ingest_after_rollover_resurrects_fail_demo` (the pre-fix plain-ingest path resurrects the deleted buffer) + `test_persist_during_rollover_destructive_window_not_orphaned` (two-thread: a live persist registering in the exact r3 post-check/pre-delete window is not orphaned). Closes the resolve-persist race by construction: `perform_rollover` holds `session.registry_lock()` across seed-read → `rolled_to` write, and `session.persist_turns_following_successor` takes the same lock and redirects the append | yes — fail-demo reproduces the resurrection; redirect test + threaded window test prove closure | PASS |
| A1 | advisory | regression metrics — no comparable workload (H8); surfaced, not gating | n/a | advisory |
| A2 | advisory | CI green (ruff + pytest subset) — see above; the one failure is pre-existing/isolated | n/a | advisory (met) |

**Every gating criterion C1–C22 is `verified = yes` by execution on its real governed path.** No deferral, no
proxy, no silent drop. Additional build-added migration tests (`test_cm1..cm12`, `test_time_stepping_...`)
exercise the migration backlog mechanics beyond C12 and also pass.

## Frozen-criteria re-check (FRZ, stage 8)
1.5-criteria.md sha256 at stage 8 == `6a12be35…` (matches the gate-4 freeze) — no post-freeze edit.

## Verdict
Conformance: **PASS** (C1–C22 all verified by execution). Regression: advisory-only (no workload) — not
gating, none observed. Routing pending the **stage-6 code red-team verdict** (a stage-6 Major would route back
to stage 5); this harness records conformance, the stage-6 record records code-vs-plan correctness.
