# 6 — Red-team of the code (verbatim record)

## Round 1 — BLOCKER (G8 CI red)

### Reviewer
- Agent type `general-purpose`, model `sonnet`, cold (agentId ad1a8a3c5cd00de75), spawned
  run-in-background. **The reviewer completed to MAIN, not to this runner**; its verdict was
  relayed by the coordinator. Its full transcript lives at the agent's output. The blocker it
  found was independently reproduced by this runner's own full CI run (b47hauh9o), so the review
  demonstrably ran.

### Reviewed diff (mechanical)
`git diff 3e98ee57 -- brain/` + the full new file `tests/test_gate_coverage.py`, captured to
`changes/gate-leak-coverage/_stage6_diff.txt`.

### Context given
`{1.5-criteria (frozen), 2-plan}` + the diff + worktree source (pending.py, the 4 routed loci,
consolidation.py, extractor.py, monologue_capture.py, trace.py, vocabulary.py, propose_write.py).

### Verbatim verdict (relayed)
Everything checks out EXCEPT G8:
- G1-G7 routing correct; mechanism fidelity good; position/concurrency N/A; A1-A3 accurate.
- **G8 CI RED — BLOCKER.** 7 PRE-EXISTING tests assert the OLD synchronous-write contract; the
  4 routed types now enqueue (async) instead of writing memories.db directly:
  - `tests/unit/brain/files/test_commit.py` (3: test_commit_create_writes_file_and_memory,
    test_decline_writes_nothing_records_memory, test_committed_write_surfaces_in_feed)
  - `tests/unit/brain/maker/test_making_runner.py` (2)
  - `tests/unit/brain/maker/test_wiring_memory.py` (1)
  - `tests/unit/brain/self_model/test_reconcile.py` (1)
  FIX: migrate to the new gated queue-contract (enqueue → drain → promote), do NOT weaken/delete.
- Blast radius flagged: `brain.bridge.feed.build_file_write_entries` reads memories.db directly →
  committed file-writes vanish from the Feed until the gate promotes them (user-facing latency,
  parallels A3); gating file_write also permanently dedups identical audit records (A2). Owner
  keeps all 4 GATE rulings; record consequences.
- Optional minors: G6 enumerator's bare-substring match is comment-blind (inert now);
  test_g6_guard re-implements the predicate inline rather than exercising the real enumerator.

### Disposition (see decisions.md Gate 7 round 1)
Blocker → return to build. All 7 tests migrated to the gated queue-contract (assert-not-in-db →
run_consolidation promote-all → assert-in-db), matching the temp-gate reference
`tests/unit/brain/chat/test_extractor_apply.py::_promote_pending`. Blast radius recorded as
accepted consequences in decisions.md. Optional minors applied (G6 fail-test now drives the real
`_enumerate_create_sites`; the feed-latency consequence documented in the migrated feed test).
A fresh stage-6 re-review (round 2) follows on the fixed code.

---

## Round 2 — CLEAN (fixed code)

### Reviewer
- Agent type `general-purpose`, model `sonnet`, cold (agentId a2909437fb4976453), synchronous
  (returned to this runner — full verbatim below). It ran the oracles empirically (checked out the
  pre-fix loci and ran the migrated tests against them; re-implemented the enumerator standalone;
  ran ruff + the full CI selection).

### Reviewed diff (mechanical)
`git diff 3e98ee57 -- brain/ tests/` → `changes/gate-leak-coverage/_stage6_diff_r2.txt`
(sha256 a50bf068…): 4 routing edits + 7 migrated tests + new tests/test_gate_coverage.py.

### Verbatim verdict
**Worst severity: CLEAN.** The round-1 blocker (CI red from 7 stale-contract tests) is fully and
correctly resolved.
- **7 migrated tests empirically discriminate the fix:** the reviewer checked out the pre-fix
  (direct store.create) loci and re-ran the migrated tests — 6 of 7 FAIL against reverted source
  (proving they test the new enqueue→promote contract), plus test_g1–g4 and
  test_g6_no_ungated_direct_create_writer FAIL against reverted source. EXCEPTION:
  `test_committed_write_surfaces_in_feed` passes with-and-without the fix — but it is a
  blast-radius/Feed-visibility documentation test (its own comments cite it as such, NOT a gating
  oracle) and is redundant with `test_commit_create_writes_file_and_memory` in the same file, which
  DOES discriminate on the identical call path. Not a defect.
- **G6 guard exactly correct:** independent standalone re-enumeration of `brain/` = 16 unique
  `(relpath, stripped-text)` tuples (recovery/engine.py's two identical `store.create(mem)` lines
  collapse in the set); the test's `_PINNED` = 2 + 14 = 16, exact match. `test_g6_guard_can_fail`
  genuinely drives the real `_enumerate_create_sites` against a synthetic novel-receiver file.
- **No new cross-test pollution:** `registered_channel` fixture registers + tears down
  `gatecoverage_channel`; no other suite references it; full `tests/unit` (3125 passed) and the CI
  selection (4244 passed, 0 failed, exit 0) both green. The flaky `bridge/test_endpoints.py`
  image_shas websocket tests are untouched by the diff (not in the file list) — pre-existing
  order-dependent flakiness, unrelated.
- **G5 unchanged**, re-ran and passes (drives capture_monologue + public apply_side_effects; has
  the shown-able-to-fail control). Position/concurrency lenses N/A confirmed.
- **One MINOR (documentation-only):** 2-plan.md's `.create(` counts (21→17) count recovery's two
  identical lines separately, while the set-based guard collapses them (20→16 unique). No
  functional impact — the executable guard is exactly correct. [FIXED in place: 2-plan.md prose
  now states both the physical-line and unique-tuple counts. The frozen 1.5-criteria.md contains
  no such count, so the freeze is untouched.]

(Full round-2 transcript retained; agentId a2909437fb4976453.)
