# 2 — Plan (persisted-cadence, defer #21)

Full task-by-task plan: `docs/superpowers/plans/2026-06-23-persisted-cadence.md`
(4 tasks, TDD, one commit each).

## How (summary)

1. **Task 1** — `brain/bridge/persisted_cadence.py`: `CadenceState(next_at)` +
   `load_cadence`/`save_cadence` (atomic temp+rename) + `is_due` + `advance`.
   Unit-tested (C1).
2. **Tasks 2–4** — wire `finalize`, `maintenance`, `voice_reflection` in
   `run_folded`: replace each `time.monotonic()` init with `load_cadence`,
   each guard with `is_due(now)`, each footer with `advance`+`save`. Tick bodies
   untouched (C5). Each gets a through-path test (C2).

## Measurement (how each criterion is verified)

- C1/C2/C4 — pytest invocations named in `1.5-criteria.md`.
- C3 — grep for the removed monotonic locals.
- C5 — `git diff brain/bridge/supervisor.py` reviewed: hunks confined to
  init/guard/footer; no tick-body edit.
- C6 — logically entailed by C1+C2 (no extra run).

## Instrumentation

**None required.** The change is observable through the existing test harness
(through-path tests assert the state file is written + the tick fires). No new
runtime log needed — the cadence state files (`*_cadence.json`) are themselves
the persisted signal, asserted directly in C2. The Layer-2 cost/cache logs are
orthogonal (stage 0) so no chat-path instrumentation applies.

## Thresholds (finding severity → routing)

Standard METHODOLOGY severity model. Project-specific notes:
- A finding that a **tick body is altered** (C5 violated) = **blocker** (changes
  decay/voice/finalise behaviour — the exact regression this change must NOT make).
- A finding that **advance/save can be skipped on a tick exception** (tight
  retry-storm reintroduced) = **major** (correctness of the always-advance
  contract).
- A finding that the **maintenance block's end-of-block advance can be bypassed**
  because an inner tick raises = needs checking: each inner tick is individually
  `try/except`-wrapped, so the block body cannot raise; if the reviewer finds a
  path where it can, that's **major**.
- Missing/weak through-path coverage for a cadence = **minor** (fix in place).

## Gating vs advisory

- **Gating:** C1–C6 (all automated, all gating). Full-suite green (C4) is the
  regression gate.
- **Advisory:** none. The standing cost/cache/tool metrics do not apply to this
  change (orthogonal; stage 0).
