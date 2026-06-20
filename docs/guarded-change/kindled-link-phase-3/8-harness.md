# Stage-8 Harness + Final Review — kindled-link Phase 3

## Final whole-branch review (most-capable model, SDD final gate)

Verdict: **READY TO MERGE.** All six load-bearing safety invariants verified PASS end-to-end
with cited test evidence; the three stage-6 fixes verified correct + non-vacuous; the
import-allowlist oracle has no false-negative path. Findings: 3 Minor + 1 Nitpick (none break
a Phase-3 safety property). Full report in agent transcript.

Resolutions:
- Minor #1 (engine `_store._conn` reach-in for cooldown) + stage-6 logged cooldown finding →
  **RESOLVED**: added `KindledLinkStore.latest_cooldown_until` (`MAX(cooldown_until)` across
  ended sessions, multi-session-correct); `can_start_session` calls it. 2 new store tests
  prove the later-ended-but-earlier-cooldown case the old `ORDER BY ended_at LIMIT 1` got wrong.
- Minor #3 (spec §5.6 "recovery flag" unimplemented) → **RELOGGED** to §10 Phase 6 (owns the
  banner); `recover()` returns the action list this phase. No silent drop.
- Minor #2 (`_check_day` UTC vs local-wall-clock counter reset) → **LOGGED** to §10 wiring phase
  (no live caller this phase).
- Nitpick (`gate.revision_constraints` unused) → accepted forward seam for Phase 4.

## Stage-8 harness verdict

**Conformance (spec §7.1) — GREEN.** kindled_link suite **90 passed** (the six criteria:
tool-path unreachability [T6 behavioural + T9 AST allowlist], inert send [T7], caps
send-path-wired incl. provider-cap-in-generation [T5/T6/T7], recovery re-gate + defer-not-drop
[T8], prompt exclusions/fenced-untrusted [T2], background discipline [T6]). ruff clean.

**Regression (spec §7.2) — FLAT (advisory).** Phase 3 touches no chat-path code; the AST
isolation test `test_chat_path_does_not_import_kindled_link` proves `brain/chat/` + provider
import no `kindled_link`. DORMANT → no live provider/cost surface; the config's chat_usage
cost/cache/turns metrics are structurally unaffected (no replay workload exists; advisory by
config). No regression measurable or expected.

**Full-suite note.** Branch build ran the full suite (3506 deterministic passed, ruff clean).
One integration test — `tests/integration/brain/attunement/test_adversarial_corpus.py::
test_adversarial_corpus_does_not_crystallise_any_pattern` — failed once then PASSED on isolated
re-run (507s). It is `@requires_claude_cli`, runs live Haiku 12×, imports only attunement, and
has zero kindled_link mechanism: a confirmed live-model nondeterminism flake, NOT a Phase-3
regression.

## Loop complete

stage 1 spec → 1.5 criteria → 2 plan → 3 red-team ×2 (Major→revise→revise) → 4 gate → 5 build
(9 TDD tasks) → 6 cold code red-team (Minor, fixed) → final whole-branch review (READY) →
8 harness (conformance GREEN, regression FLAT). DORMANT/EXPERIMENTAL on land.
