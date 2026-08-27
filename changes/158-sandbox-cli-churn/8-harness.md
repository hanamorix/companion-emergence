# 8 — Harness / conformance verdict (#158)

No prior-version regression workload exists for this project (Layer-2 config: regression is
advisory-only, and this is a detection-scope harness change with no chat/tool-log metric). So
stage 8 is **conformance-only**: the 1.5 criteria, verified by the unit tests, plus CI.

## Conformance to 1.5 criteria (all gating criteria met)

| Criterion | Verified by | Result |
|---|---|---|
| G1 mtime-only bump is silent | `test_cli_housekeeping_mtime_bump_is_silent` (oracle-can-fail preface + real run: zero guard warning, no raise) | PASS |
| G1b content change still warns | `test_cli_housekeeping_content_change_still_warns` (direct-fingerprint content-axis oracle + real-run `pytest.warns` DOWNGRADED) | PASS |
| G2 non-housekeeping file still surfaces | `test_non_housekeeping_claude_file_still_surfaces` (settings.json warns) | PASS |
| G3 non-`~/.claude` escape still hard-raises | `test_housekeeping_churn_alongside_real_escape_still_hard_raises` + existing `test_downgrade_is_scoped...` | PASS |
| G4 targeted / fail-closed (3rd json still seen) | `test_non_housekeeping_claude_file_still_surfaces` (some-other.json warns) | PASS |
| G4b top-level-only (subdir same-basename stays mtime-guarded) | `test_same_basename_in_subdir_stays_mtime_guarded` (direct-fingerprint scope pin + real-run warn) | PASS |
| G5 constant + helper pinned | `test_cli_housekeeping_constant_and_helper_pinned` | PASS |
| G6 F4 session-log sets unchanged | `test_af2_exclusion_set_unchanged` (existing, green) + the not-in-exclude asserts in the pin test | PASS |
| G7 raise-vs-warn semantics unchanged | diff review (post-run block byte-untouched — confirmed by the stage-6 reviewer) + the three existing option-(c) tests green | PASS |
| A1 (advisory) comment states the real safety argument | sandbox.py housekeeping-constant comment | MET |

## Red-team gates
- Stage 3 (plan): MINOR → adopted content-aware amendment (reviewer's own recommended fix); all hard constraints honored. Record: `3-redteam-plan.reviewer-verbatim.md`.
- Stage 6 (code): **CLEAN** (one nitpick, no action). Reviewer ran a revert probe confirming G1 + G4b truly gate the mechanism; raise/warn block + F4 exclude sets byte-untouched; diff harness-only; don't-full-prune safety argument verified against `provider.py`. Record: `6-redteam-code.reviewer-verbatim.md`.

## CI (verified pre-push, this working tree)
- `uv run ruff check .` — clean.
- `uv run pytest -m "not live and not requires_claude_cli and not integration" -q` — 4357 passed,
  2 failed, 19 skipped, 6 deselected, 1 xfailed. The 2 failures are exactly the known
  out-of-scope ones: `test_history_returns_buffered_turns_in_order` (#155 flaky) and
  `test_review_tick_gate_blocks_send_records_hold` (#136 / PR #142). No new failures; all 31
  `test_sandbox_isolation.py` tests pass (6 new).

## Verdict: CONFORMANT — cleared to commit + push.
