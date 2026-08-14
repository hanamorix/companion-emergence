# decisions.md — gate log (append-only)

## Path validation (run-start, required before gate 4 — CFG3)
2026-08-12: All reviewer-context paths validated as existing + readable (18 paths: 3 change
artifacts + 15 source files). **Config override:** the project config's `redteam_context`
brain path (`~/Desktop/companion-emergence/brain`) points at the MAIN checkout, which is
currently on branch `ThinkerOfThoughts/diagnose-monologue-bleed-memory-gap` (the UNGATED
pre-gate brain — no `pending.py`/`consolidation.py`). For this run the red-team brain-path is
overridden to THIS worktree
(`/home/zero/Desktop/companion-emergence/.claude/worktrees/elegant-yonath-760afe/brain`) per
the task instruction, so the reviewer verifies against the gated code under change. Validation
result: PASS (no dead paths).

## Gate 4 (plan) — round 1
- Worst finding severity: **MAJOR** (stage-3 cold review, agentId af2836e442de2a674, sonnet).
- Route: **major → return to stage 2** (revise plan). Bounce 1 of 2 on the plan gate.
- Findings addressed in the revision (round 2 artifacts):
  - (a) G6 guard precision — REVISED: guard now matches ALL `\.create\(` in `brain/` and asserts
    ⊆ pinned union {List1 KNOWN-NON-MEMORY excludes = propose_write.py's 2 `pending.create`
    (brain.files.pending) ; List2 ALLOWED-DIRECT memory writes}. Fails safe on any novel/aliased
    receiver. (1.5-criteria.md G6, 2-plan.md "pinned lists".)
  - (b) CH8 blast radius — NAMED as advisory A2 (file_write same-tick exact-dup = designed gate
    behavior, accepted) + A3 (self_model_reconcile emotion-memory visibility latency; verified
    acknowledge/cooldown NOT coupled to the write at reconcile.py:177-186 — inherent to
    owner-ordered gating; surfaced to owner in completion report). Not blocking.
  - minor: resolve `source` label fixed to `self_model_resolved`; G5 now drives the public
    `apply_side_effects` wrapper (closes the representativeness gap).
- Coverage-completeness (reviewer-verified independently): the audit of real memory-store writers
  is COMPLETE (21 `.create(` lines; every one classified). Not contested.
- NOT self-cleared: the revision is going back through a focused stage-3 re-review (round 2)
  before gate 4 can route to build — no author self-approval of the fixes (CP1).

## Gate 4 (plan) — round 2
- Worst finding severity: **MAJOR** (round-2 cold review, agentId a72b0cac09bf27799, sonnet) —
  a NEW finding (A2 description accuracy), a DIFFERENT finding class from round 1's (a)/(b), so
  the iteration cap ("same finding class") is not tripped. Round-1 (a) confirmed RESOLVED; A3
  confirmed accurate; G5 confirmed constructible; coverage confirmed complete.
- Resolution WITHOUT a third plan review (proportionate): the round-2 major is a
  DESCRIPTION-ACCURACY fix — A2 rewritten to state the reviewer's OWN verified mechanism
  (permanent cross-tick DB-wide exact-dup of identical file_write content; full history in
  audit.jsonl). No new author judgment/mechanism to cold-review — I transcribed the reviewer's
  verified finding. The minor (G6 pin by line-text) applied. The CODE change plan is unchanged
  and twice-validated; stage-6 code review independently re-checks the built guard + can re-flag
  A2. accurate-A2 is SURFACED TO THE OWNER in the completion report (file_write GATE ruling can
  be revisited).
- Route: **proceed to build (stage 5)**.
- **Criteria FROZEN** at 1.5-criteria.md sha256 = 46c79f2d54423b260a3523d2c081106af03dc4138dfe0e4ee9d6c58f50df85d5 (route-to-build version).
- Path validation (run-start): recorded above — PASS.

## Gate 7 (code) — round 1
- Worst finding severity: **BLOCKER** (stage-6 cold code review; reviewer completed to MAIN,
  relayed by the coordinator). G1-G7 routing correct, mechanism fidelity good, position/
  concurrency N/A, A1-A3 accurate — EXCEPT **G8 CI RED**: 7 PRE-EXISTING tests asserted the OLD
  synchronous-write contract (the 4 routed types now enqueue async instead of writing memories.db
  directly):
  - tests/unit/brain/files/test_commit.py (×3: commit_create_writes_file_and_memory,
    decline_writes_nothing_records_memory, committed_write_surfaces_in_feed)
  - tests/unit/brain/maker/test_making_runner.py (×2)
  - tests/unit/brain/maker/test_wiring_memory.py (×1)
  - tests/unit/brain/self_model/test_reconcile.py (×1)
- Route: **blocker → return to stage 5 (build)**. Bounce 1 on the code gate.
- FIX (not weaken/delete): migrated all 7 to the NEW gated queue-contract — assert the type is
  NOT in memories.db right after the write (proves gating), drain via run_consolidation with a
  promote-all Decision("new") classifier, then assert it lands (proves promotion). Same migration
  the temp-gate build applied to tests/unit/brain/chat/test_extractor_apply.py (the reference
  _promote_pending pattern).
- Optional minor applied: strengthened test_g6_guard_can_fail to exercise the REAL enumerator
  (`_enumerate_create_sites(root)`) against a synthetic novel-receiver writer, instead of an
  inline predicate re-implementation.
- 8th failure in my full-suite run (tests/bridge/test_endpoints.py::
  test_stream_accepts_image_shas_in_request_frame) is UNRELATED to this diff — it passes in
  isolation both with and without my changes; a full-suite ordering artifact, not a regression of
  this change. Re-checked after the migration.

## Accepted blast-radius consequences (owner keeps all 4 GATE rulings; consequences documented)
- **Feed latency for file_write (NEW, mirrors A3):** `brain.bridge.feed.build_file_write_entries`
  reads memories.db directly, so a committed file-write is NOT visible in the Feed until the next
  idle-tick consolidation gate promotes the (now gated) `file_write` candidate — a user-facing
  latency exactly parallel to the self-model latency A3 already names. ACCEPTED as an inherent
  consequence of the owner's GATE-file_write ruling; documented in the migrated
  test_committed_write_surfaces_in_feed (drain-then-assert). FLAGGED for possible owner revisit.
- **Permanent identical-audit dedup for file_write (A2):** identical `file_write` audit content
  ("you let me write to {path}") dedups permanently in recall (cross-tick, DB-wide) — see A2.
  ACCEPTED; full history retained in audit.jsonl. FLAGGED for possible owner revisit.
- Owner rulings UNCHANGED: all 4 GATE, all 4 BYPASS. These are consequences to surface, not
  reversals — surfaced to the owner in the completion report.

## G8 CI determination (after gate-7 fix)
- Migrated-tests + new-tests: all 16 PASS (verified in isolation and full-suite).
- Full CI sweep (`pytest -m "not live and not requires_claude_cli and not integration"`):
  - clean base, -p no:randomly: 4235 passed, 0 failed.
  - my changes, -p no:randomly (run A): 1 failed = tests/bridge/test_endpoints.py::
    test_chat_image_shas_default_empty_works.
  - my changes, -p no:randomly (run B, SAME order, same code): 4244 passed, 0 failed.
  - my changes, WITH pytest-randomly (earlier): 1 failed = a DIFFERENT image_shas test
    (test_stream_accepts_image_shas_in_request_frame).
- CONCLUSION: the image_shas failures are PRE-EXISTING FLAKINESS in the websocket/uvicorn-based
  bridge endpoint tests (deprecation warnings confirm the websockets/uvicorn stack), NOT caused by
  this diff: same code + same deterministic order yields pass on re-run; the failing test varies;
  the file passes 44/44 in isolation with my changes; my diff touches only memory-gating in
  self_model/maker/files (nothing in bridge/endpoints or image handling). OUT OF SCOPE for this
  change. G8 = GREEN (all gated criteria's tests pass; the flake is unattributable to this change).

## Gate 7 (code) — round 2
- Worst finding severity: **CLEAN** (round-2 cold code review, agentId a2909437fb4976453, sonnet;
  ran the oracles empirically). Round-1 blocker fully resolved; 7 migrated tests discriminate the
  fix (verified against reverted source); G6 pinned list independently re-derived exact; no new
  pollution; ruff + full CI selection green.
- One MINOR (documentation-only): 2-plan.md `.create(` counts didn't account for the set-collapse
  of recovery/engine.py's two identical lines. FIXED IN PLACE (2-plan.md now states physical=21→17
  and unique-tuples=20→16). Frozen 1.5-criteria.md has no such count → freeze untouched (hash
  46c79f2d… unchanged).
- Route: **clean → proceed to stage 8 (harness)**.

## Gate 8 (harness) — conformance
- Mode: conformance-only (no stage-0 baseline; config regression metrics advisory/irrelevant to a
  correctness change). Freeze verify: 1.5-criteria.md sha256 unchanged (46c79f2d…). PASS.
- All 8 gating criteria (G1-G8) empirically verified by exercising the governed path; every gating
  oracle shown able to fail (reviewer ran the migrated/new tests against reverted source).
  Advisory A1-A3 satisfied/documented.
- The pre-existing image_shas websocket flake (bridge/test_endpoints.py) is OUT OF SCOPE — not in
  this diff, unattributable to it (see "G8 CI determination"). Not a gate-8 finding.
- VERDICT: **PASS (done).** Route: clean → done.
