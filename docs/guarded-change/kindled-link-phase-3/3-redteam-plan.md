# Stage-3 Plan Red-Team — kindled-link Phase 3

Cold independent reviewer (no shared context), source-cited against the real code.
Artefacts reviewed: `specs/2026-06-20-kindled-link-phase-3-design.md`,
`plans/2026-06-20-kindled-link-phase-3.md`, parent design, `guarded-change.companion.md`.

## Worst severity: Major → revise plan (no spec restart — architecture verified sound)

The tool-less `provider.complete` contract, fail-closed gate seam, and DORMANT claim are
all factually verified against the real code (provider.py:212-220 `complete` delegates to
`generate(system=None)`; ClaudeCliProvider does not override; tools only wired in
`chat`/`chat_stream` via `--mcp-config`). The five forbidden tokens are the genuinely
correct injection-hole names. No supervisor reference to `kindled_link` exists. Revise the
plan, do not restart the spec.

## Findings ranked

| # | Severity | Finding | Evidence | Route |
|---|---|---|---|---|
| 1 | Major | T9 conformance grep FAILS by construction — planned `session_engine.py` module docstring contains all five forbidden literal substrings | plan T5 docstring vs T9 grep | revise plan: reword docstring + AST-based T9 |
| 2 | Major | Cap/pacing predicates unit-tested but NOT consulted by `process_outbound`/`recover` (draft_space dead-reader; runaway-pacing hole under a real Phase-4 gate) | plan T7/T8 vs spec crit 3 | revise plan: wire predicates + refusal test |
| 3 | Major | `_INBOUND_FLOOD_CAP` declared, criterion 3 lists inbound flood, nothing implements/tests it (no inbound path in Phase 3) | plan:_INBOUND_FLOOD_CAP, spec §7.1 crit 3 | revise: defer inbound flood criterion + drop constant |
| 4 | Minor | §5.5 autonomous-start suppression on `should_yield()` unimplemented (`can_start_session` never consults throttle) | plan can_start_session, cli_throttle.py:42 | fix in place + test |
| 5 | Minor | Spec §5 step 6 transport (`build_envelope`+`relay_client.push`) implemented by no task; self-review wrongly claims "no gaps" | protocol.py:124, relay_client.py:40 | fix in place: explicit defer + correct self-review |
| 6 | Minor | No test proves a populated `relationship_hint` reaches the gate (inv. 127) | plan T7 tests all body-only | fix in place: spy-gate captures hint |
| 7 | Minor | Conformance grep substring false-positive risk (generic) | plan T9 grep | subsumed by #1 AST fix |
| 8 | Nitpick | `recover` producer unwired (only test seeds drafts) | plan T7/T8 | log in manifest |
| 9 | Nitpick | Provider spy only implements `complete`; accidental `chat` route would AttributeError not assert | plan T6 spy | add chat/chat_stream no-touch assertion |
| 10 | Nitpick | `background_slot()` called without injected `now=` (real monotonic in clock-injected engine) | plan T6 | log |

## Disposition (author)

All three Majors + Minors 4/5/6 + Nitpick 9 accepted and addressed in plan revision v2
(see `decisions.md`). Nitpicks 8 and 10 logged (accepted DORMANT seams), not changed.
